"""Purpose-specific scoring ("profiles"), e.g. profiles/office.yaml.

Method (see docs/SCORING.md):
1. Each metric is put on a common scale: its percentile in the candidate pool
   (scale: percentile) or its raw value (scale: raw, for absolute quantities
   such as megapixels or the fraction of the frame filled by a person).
2. A desirability curve maps it to d in [eps, 1]; "up" = more is better,
   "down" = less is better; lo/hi bound the ramp, so a must-not-be-bad metric
   saturates at 1 once adequate and only penalizes deficiency.
3. Desirabilities combine by a weighted power mean with exponent p
   (1 = arithmetic mean, 0 = geometric mean, -1 = harmonic, -> -inf = minimum),
   so one badly deficient metric drags the score down without a hard filter.
4. Uncertainty in weights and p: Monte Carlo draws (weights ~ Dirichlet around
   the nominal weights, p ~ uniform range); each photo's probability of being in
   the top K is the robust ranking.
5. Final picks are made greedily for diversity (maximal marginal relevance on
   CLIP similarity).

Two stages: stage 1 scores every candidate with cheap metrics (stage-2 metrics
fall back to cheap proxies); the shortlist then gets the expensive metrics
(focus map from the originals, small-view aesthetic) and is re-scored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import REPO_ROOT, Config
from .db import connect
from .embed import load_embeddings
from .zeroshot import DEFAULT_LABELS, concept_probs, label_probs

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_scores (
    profile   TEXT,
    sha1      TEXT,
    stage     INTEGER,      -- 1 = scored on cheap metrics only, 2 = shortlist with all metrics
    score     REAL,         -- nominal power-mean desirability (0..1)
    p_topk    REAL,         -- Monte Carlo probability of being in the top K
    weakest   TEXT,         -- metric with the lowest desirability
    d         TEXT,         -- JSON {metric: desirability}
    selected  INTEGER,      -- order of pick in the diverse final set, NULL if not picked
    PRIMARY KEY (profile, sha1)
);
"""


# ---------------------------------------------------------------- the maths

def percentile(x: pd.Series) -> pd.Series:
    """Empirical CDF in (0, 1]; NaN stays NaN."""
    return x.rank(pct=True, method="average")


def desirability(x: np.ndarray, kind: str, lo: float, hi: float, eps: float = 0.05,
                 shape: float = 1.0) -> np.ndarray:
    """Derringer-Suich style ramp onto [eps, 1]."""
    x = np.asarray(x, dtype=float)
    if kind == "up":
        t = (x - lo) / (hi - lo)
    elif kind == "down":
        t = (hi - x) / (hi - lo)
    else:
        raise ValueError(kind)
    t = np.clip(t, 0.0, 1.0) ** shape
    return eps + (1.0 - eps) * t


def power_mean(D: np.ndarray, w: np.ndarray, p: float) -> np.ndarray:
    """Weighted power mean over the last axis of D (values in (0, 1]); w sums to 1."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    if abs(p) < 1e-9:
        return np.exp(np.log(D) @ w)
    return (D ** p @ w) ** (1.0 / p)


def monte_carlo_topk(D: np.ndarray, w0: np.ndarray, k: int, draws: int = 1000, concentration: float = 20.0,
                     p_range=(-2.0, 0.5), seed: int = 0) -> np.ndarray:
    """Fraction of (weights, p) draws in which each row is in the top k."""
    rng = np.random.default_rng(seed)
    n = len(D)
    k = min(k, n)
    hits = np.zeros(n)
    alpha = concentration * np.asarray(w0) / np.sum(w0) * len(w0)
    logD = np.log(D)
    for _ in range(draws):
        w = rng.dirichlet(alpha)
        p = rng.uniform(*p_range)
        s = np.exp(logD @ w) if abs(p) < 1e-9 else (np.exp(p * logD) @ w) ** (1.0 / p)
        hits[np.argpartition(-s, k - 1)[:k]] += 1
    return hits / draws


def diverse_pick(score: np.ndarray, E: np.ndarray, n: int, lam: float = 1.0, sim_floor: float = 0.75) -> list[int]:
    """Greedy maximal-marginal-relevance selection: high score, unlike what is already picked."""
    n = min(n, len(score))
    s = (score - score.min()) / (np.ptp(score) or 1.0)
    picked: list[int] = []
    max_sim = np.full(len(score), -1.0)
    avail = np.ones(len(score), dtype=bool)
    for _ in range(n):
        pen = lam * np.clip((max_sim - sim_floor) / (1 - sim_floor), 0, None)
        util = np.where(avail, s - pen, -np.inf)
        i = int(np.argmax(util))
        picked.append(i)
        avail[i] = False
        max_sim = np.maximum(max_sim, E @ E[i])
    return picked


# ---------------------------------------------------------------- data

def load_profile(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = REPO_ROOT / "profiles" / f"{name_or_path}.yaml"
    prof = yaml.safe_load(p.read_text())
    prof.setdefault("name", p.stem)
    return prof


def candidates(con) -> pd.DataFrame:
    """One row per distinct moment: burst winners from the rank stage, minus non-photos."""
    df = pd.read_sql_query("""
        SELECT p.*, s.aesthetic, s.iqa, s.iaa,
               d.n_persons, d.max_area, d.total_area,
               t.coarse_contrast, t.coarse_edges, t.coarse_ratio, t.tonal_range,
               t.focus_max, t.focus_frac, t.small_aesthetic,
               r.burst_best, r.event_id
        FROM photos p
        JOIN rank r USING(sha1)
        LEFT JOIN scores s USING(sha1)
        LEFT JOIN persons d USING(sha1)
        LEFT JOIN structure t USING(sha1)
        WHERE r.burst_best = 1
          AND COALESCE(p.format, '') <> 'GIF'
          AND NOT (p.format = 'PNG' AND p.camera_model IS NULL)
    """, con)
    return df.reset_index(drop=True)


def metric_values(df: pd.DataFrame, emb: dict, prof: dict, cache: Path) -> pd.DataFrame:
    """Raw value of every metric named in the profile (NaN where not computed)."""
    E = np.stack([emb[s] for s in df["sha1"]])
    concepts = concept_probs(E, prof.get("concepts", {}), cache)
    labels = label_probs(E, DEFAULT_LABELS, cache)
    junk = sum(v for k, v in labels.items() if k.startswith("junk_"))
    pct = percentile
    m = pd.DataFrame(index=df.index)
    m["aesthetic"] = df["aesthetic"]
    m["iaa"] = df["iaa"]
    m["iqa"] = df["iqa"]
    for name, v in concepts.items():
        m["concept_" + name] = v
    good = [c for c in prof.get("subject_concepts", []) if "concept_" + c in m]
    m["subject"] = m[["concept_" + c for c in good]].max(axis=1) if good else np.nan
    for c, f in (prof.get("subject_partial") or {}).items():
        if "concept_" + c in m:
            m["subject"] = np.fmax(m["subject"], f * m["concept_" + c])
    # People prominence: largest person box as a fraction of the frame; crowds of small figures are fine.
    crowd = prof.get("crowd", {"min_persons": 8, "max_area": 0.06})
    is_crowd = (df["n_persons"] >= crowd["min_persons"]) & (df["max_area"] <= crowd["max_area"])
    m["people"] = df["max_area"].where(~is_crowd, 0.0)
    m["art_junk"] = np.maximum(labels["artwork"], junk)
    m["megapixels"] = df["megapixels"]
    # Reads from a distance: large-scale contrast and edges (stage 1), plus small-view aesthetic (stage 2).
    m["distance_coarse"] = (pct(df["coarse_contrast"]) + pct(df["coarse_edges"]) + pct(df["tonal_range"])
                            + pct(df["coarse_ratio"])) / 4
    m["distance"] = np.where(df["small_aesthetic"].notna(),
                             0.5 * m["distance_coarse"] + 0.5 * pct(df["small_aesthetic"]), m["distance_coarse"])
    # Focus somewhere: tile-sharpness map of the original (stage 2), else whole-image proxy.
    m["focus"] = df["focus_frac"]
    m["focus_proxy"] = pct(np.log1p(df[["sharpness", "sharp_center"]].max(axis=1)))
    return m


def desirabilities(m: pd.DataFrame, prof: dict) -> pd.DataFrame:
    out = {}
    for name, spec in prof["metrics"].items():
        col = spec.get("source", name)
        x = m[col]
        if spec.get("scale", "percentile") == "percentile":
            x = percentile(x)
        d = desirability(x, spec["kind"], spec.get("lo", 0.0), spec.get("hi", 1.0),
                         spec.get("eps", prof.get("eps", 0.05)), spec.get("shape", 1.0))
        fb = spec.get("fallback")
        if fb:
            fspec = {**spec, **fb}
            fx = m[fspec["source"]]
            if fspec.get("scale", "percentile") == "percentile":
                fx = percentile(fx)
            fd = desirability(fx, fspec["kind"], fspec.get("lo", 0.0), fspec.get("hi", 1.0),
                              fspec.get("eps", prof.get("eps", 0.05)), fspec.get("shape", 1.0))
            d = np.where(np.isnan(np.asarray(x, dtype=float)), fd, d)
        d = np.where(np.isnan(d), 1.0, d)      # metric unavailable: neutral
        out[name] = d
    return pd.DataFrame(out, index=m.index)


# ---------------------------------------------------------------- run

def _score(D: pd.DataFrame, prof: dict, k: int, seed: int = 0):
    w = np.array([prof["metrics"][c].get("weight", 1.0) for c in D.columns])
    A = D.to_numpy()
    nominal = power_mean(A, w, prof.get("p", 0.0))
    mc = prof.get("monte_carlo", {})
    ptop = monte_carlo_topk(A, w, k, draws=mc.get("draws", 1000), concentration=mc.get("concentration", 20.0),
                            p_range=tuple(mc.get("p_range", (-2.0, 0.5))), seed=seed)
    return nominal, ptop


def run_profile(cfg: Config, name: str, stage2: bool = True) -> dict:
    prof = load_profile(name)
    con = connect(cfg.db_path)
    con.executescript(SCHEMA)
    sha1s, E = load_embeddings(con)
    emb = dict(zip(sha1s, E))
    df = candidates(con)
    df = df[df["sha1"].isin(emb)].reset_index(drop=True)
    cache = cfg.work_dir / "models" / "text_emb.npz"
    con.close()

    # Stage 1: everything, cheap metrics.
    m = metric_values(df, emb, prof, cache)
    D = desirabilities(m, prof)
    n_short = prof.get("shortlist", 2000)
    nominal, ptop = _score(D, prof, n_short)
    order = np.lexsort((-nominal, -ptop))
    short = order[:n_short]

    # Stage 2: expensive metrics on the shortlist, then rescore the shortlist.
    stage = np.ones(len(df), dtype=int)
    if stage2:
        from .embed import run_small_aesthetic
        from .structure import run_focus
        ids = df.loc[short, "sha1"].tolist()
        run_focus(cfg, ids)
        run_small_aesthetic(cfg, ids)
        con = connect(cfg.db_path)
        df = candidates(con)
        con.close()
        df = df[df["sha1"].isin(emb)].reset_index(drop=True)
        m = metric_values(df, emb, prof, cache)
        short_ids = set(ids)
        is_short = df["sha1"].isin(short_ids).to_numpy()
        D = desirabilities(m, prof)          # percentiles still over the whole pool for stage-1 metrics
        stage = np.where(is_short, 2, 1)
        # stage-2 metrics only exist for the shortlist; score the shortlist on its own
        Ds = D[is_short]
        nom_s, p_s = _score(Ds, prof, prof.get("target", 300), seed=1)
        nominal = np.full(len(df), np.nan)
        ptop = np.full(len(df), np.nan)
        nominal[is_short], ptop[is_short] = nom_s, p_s
        nominal1, _ = _score(D, prof, n_short)
        nominal[~is_short] = nominal1[~is_short]
        ptop[~is_short] = 0.0

    # Diverse final pick among the best (by MC probability, ties by nominal score).
    rank_score = np.nan_to_num(ptop) + 1e-3 * np.nan_to_num(nominal)
    pool = np.argsort(-rank_score)[: prof.get("target", 300) * 3]
    Ep = np.stack([emb[s] for s in df.loc[pool, "sha1"]])
    div = prof.get("diversity", {})
    picks = diverse_pick(rank_score[pool], Ep, prof.get("target", 300), lam=div.get("lambda", 1.0),
                         sim_floor=div.get("sim_floor", 0.75))
    selected = np.full(len(df), None, dtype=object)
    for order_i, j in enumerate(picks):
        selected[pool[j]] = order_i + 1

    weakest = D.idxmin(axis=1)
    rows = [(prof["name"], df.at[i, "sha1"], int(stage[i]), float(nominal[i]), float(ptop[i]), weakest[i],
             json.dumps({c: round(float(D.at[i, c]), 3) for c in D.columns}), selected[i])
            for i in df.index]
    con = connect(cfg.db_path)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM profile_scores WHERE profile=?", (prof["name"],))
    con.executemany("INSERT INTO profile_scores VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    sel = [i for i in df.index if selected[i] is not None]
    return dict(profile=prof["name"], candidates=len(df), shortlist=int((stage == 2).sum()), selected=len(sel),
                weakest_among_selected=pd.Series(weakest[sel]).value_counts().to_dict())
