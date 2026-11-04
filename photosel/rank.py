"""Ranking stage: filters, bursts, events, composite score, finalists.

Cheap (seconds), so it is rebuilt from scratch on every run from the cached
ingest and GPU tables; tune the `rank:` section of config.yaml and re-run.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .config import Config
from .db import connect
from .embed import load_embeddings

DEFAULTS = dict(
    min_megapixels=3.0,        # below this, too small to print well (8x10 at 200 dpi = 3.2 MP)
    junk_threshold=0.5,        # exclude if summed junk_* zero-shot probability exceeds this
    blank_brightness=(12, 245),
    burst_seconds=60,          # consecutive shots this close in time ...
    burst_similarity=0.90,     # ... and this similar (CLIP cosine) form one burst
    event_gap_hours=3.0,
    event_gap_km=30.0,
    weights=dict(aesthetic=1.0, iaa=0.5, iqa=0.7, sharpness=0.3, people=0.3,
                 favorited=2.0, edited=0.7),
    per_event_factor=0.6,      # finalists per event = factor * sqrt(photos in event) ...
    max_per_event=12,          # ... capped here
    min_composite=0.0,         # event picks must at least reach this (z-units)
    global_top_frac=0.01,      # plus the best 1% overall regardless of event
)


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load_frame(con) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT p.*, s.aesthetic, s.iqa, s.iaa, s.zeroshot FROM photos p LEFT JOIN scores s USING(sha1)", con)
    zs = df["zeroshot"].map(lambda v: json.loads(v) if v else {})
    labels = sorted({k for d in zs for k in d})
    for k in labels:
        df["zs_" + k] = zs.map(lambda d, k=k: d.get(k, np.nan))
    junk = [c for c in df.columns if c.startswith("zs_junk_")]
    df["junk_prob"] = df[junk].sum(axis=1, min_count=1) if junk else np.nan
    df["junk_top"] = df[junk].idxmax(axis=1, skipna=True).str[8:] if junk and df[junk].notna().any().any() else None
    return df.sort_values("taken_ts").reset_index(drop=True)


def exclusions(df: pd.DataFrame, p: dict) -> pd.Series:
    reason = pd.Series([None] * len(df), index=df.index, dtype=object)

    def mark(mask, why):
        mask = mask.fillna(False) & reason.isna()
        reason[mask] = why if isinstance(why, str) else why[mask]

    lo, hi = p["blank_brightness"]
    mark(df["format"].eq("GIF"), "gif")
    mark(df["megapixels"] < p["min_megapixels"], "low-res")
    mark(df["format"].eq("PNG") & df["camera_model"].isna(), "screenshot")
    mark(df["junk_prob"] > p["junk_threshold"], "junk:" + df["junk_top"].fillna("?"))
    mark((df["brightness"] < lo) | (df["brightness"] > hi), "blank")
    reason[df["favorited"].eq(1)] = None          # your favorites are never excluded
    return reason


def composite(df: pd.DataFrame, keep: pd.Series, w: dict) -> pd.Series:
    sharp = np.log1p(df[["sharpness", "sharp_center"]].max(axis=1))
    parts = {
        "aesthetic": df["aesthetic"], "iaa": df["iaa"], "iqa": df["iqa"], "sharpness": sharp,
    }
    score = pd.Series(0.0, index=df.index)
    for name, col in parts.items():
        ref = col[keep]
        med, iqr = ref.median(), ref.quantile(0.75) - ref.quantile(0.25)
        z = ((col - med) / (iqr / 1.349 if iqr > 0 else 1.0)).clip(-4, 4).fillna(0.0)
        score += w.get(name, 0) * z
    score += w.get("people", 0) * df["people"].notna()
    score += w.get("favorited", 0) * df["favorited"].eq(1)
    score += w.get("edited", 0) * df["edited_path"].notna()
    return score


def bursts(df: pd.DataFrame, keep: pd.Series, emb: dict, p: dict) -> pd.Series:
    """Burst id per row (kept rows only, in time order)."""
    burst = pd.Series(-1, index=df.index)
    bid, prev = -1, None
    for i in df.index[keep]:
        e = emb.get(df.at[i, "sha1"])
        if prev is not None and e is not None and prev[1] is not None:
            dt = df.at[i, "taken_ts"] - prev[0]
            same = dt <= p["burst_seconds"] and float(e @ prev[1]) >= p["burst_similarity"]
        else:
            same = False
        if not same:
            bid += 1
        burst[i] = bid
        prev = (df.at[i, "taken_ts"], e)
    return burst


def events(df: pd.DataFrame, p: dict) -> pd.Series:
    ts = df["taken_ts"].to_numpy()
    gap_t = np.diff(ts, prepend=ts[0]) > p["event_gap_hours"] * 3600
    lat = pd.to_numeric(df["lat"], errors="coerce").ffill().to_numpy(dtype=float)
    lon = pd.to_numeric(df["lon"], errors="coerce").ffill().to_numpy(dtype=float)
    dist = np.zeros(len(df))
    dist[1:] = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    gap_d = np.nan_to_num(dist) > p["event_gap_km"]
    return pd.Series(np.cumsum(gap_t | gap_d), index=df.index)


def run_rank(cfg: Config, params: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    p["weights"] = {**DEFAULTS["weights"], **(params or {}).get("weights", {})}
    con = connect(cfg.db_path)
    df = load_frame(con)
    sha1s, E = load_embeddings(con)
    emb = dict(zip(sha1s, E))

    df["excluded"] = exclusions(df, p)
    keep = df["excluded"].isna()
    df["composite"] = composite(df, keep, p["weights"])
    df["burst_id"] = bursts(df, keep, emb, p)
    g = df[keep].groupby("burst_id")["composite"]
    df["burst_size"] = df["burst_id"].map(g.size()).where(keep)
    best_idx = g.idxmax()
    df["burst_best"] = 0
    df.loc[best_idx.values, "burst_best"] = 1
    df["event_id"] = events(df, p)

    # Finalists: top-k per event among burst winners, plus the global top, plus favorites.
    cand = df[keep & df["burst_best"].eq(1)]
    fin = set()
    for _, ev in cand.groupby("event_id"):
        k = min(p["max_per_event"], max(1, round(p["per_event_factor"] * math.sqrt(len(ev)))))
        top = ev.nlargest(k, "composite")
        fin.update(top.index[top["composite"] >= p["min_composite"]])
    fin.update(cand.nlargest(max(1, int(p["global_top_frac"] * len(cand))), "composite").index)
    fin.update(df.index[df["favorited"].eq(1) & df["burst_best"].eq(1)])
    df["finalist"] = 0
    df.loc[list(fin), "finalist"] = 1

    cols = ["sha1", "excluded", "burst_id", "burst_size", "burst_best", "event_id", "composite", "finalist"]
    con.execute("DELETE FROM rank")
    con.executemany(f"INSERT INTO rank ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
                     for r in df[cols].itertuples(index=False)])
    con.commit()
    con.close()
    return dict(
        photos=len(df), excluded=int((~keep).sum()),
        excluded_by_reason=df["excluded"].value_counts().to_dict(),
        bursts=int(df.loc[keep, "burst_id"].nunique()), events=int(df["event_id"].nunique()),
        finalists=len(fin),
    )
