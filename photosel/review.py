"""Local rating page: you rate photos for a profile, the pipeline learns your taste.

Serves on 127.0.0.1 only, because the photos are private. Ratings go to the
`ratings` and `pairs` tables and are also appended to work/ratings_log.jsonl as
a backup that survives a rebuilt database.

Each batch mixes photos from several sources, recorded with every rating, so the
ratings both check the current scoring and train the next one:
    top        selected by the profile
    edge       on the profile's shortlist but not selected
    stage1     scored low on the cheap metrics (never reached the shortlist)
    excluded   removed by the general filters (junk, low-res, ...)
    random     any candidate, uniformly (an unbiased check)
    uncertain  once there are ~30 ratings: photos a model of your ratings can't call
    gem        once there are ~30 ratings: photos that model likes but the profile missed
"""
from __future__ import annotations

import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import Config
from .db import connect
from .taste import MIN_RATINGS as MIN_FOR_MODEL

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings (
    profile TEXT,
    sha1    TEXT,
    rating  REAL,          -- 1 yes, 0.5 maybe, 0 no
    reason  TEXT,          -- optional reason for a "no"
    source  TEXT,          -- which sampling stratum showed it
    ts      REAL,
    PRIMARY KEY (profile, sha1)
);
CREATE TABLE IF NOT EXISTS pairs (
    profile TEXT,
    a       TEXT,
    b       TEXT,
    winner  TEXT,          -- 'a', 'b' or '=' (can't choose)
    ts      REAL
);
"""

RATING = {"yes": 1.0, "maybe": 0.5, "no": 0.0}
REASONS = {"people": "people", "boring": "boring / cliché", "soft": "soft / blurry",
           "cluttered": "cluttered", "distance": "weak from a distance", "junk": "junk / not my photo",
           "personal": "personal: save for another set"}   # a "no" here, kept for a personal collection

MIX_COLD = {"top": 4, "edge": 5, "stage1": 5, "random": 4, "excluded": 2}
MIX_MODEL = {"top": 3, "edge": 3, "uncertain": 5, "gem": 2, "stage1": 3, "random": 2, "excluded": 2}
MIX_FINAL = {"top": 10, "gem": 10}     # final pass, after your maybes: only the strongest unrated candidates
SHA1 = re.compile(r"^[0-9a-f]{40}$")


class Sampler:
    """Chooses which photos to show next. Never repeats a rated or already-issued photo."""

    def __init__(self, con, profile: str, emb: dict | None = None, seed: int | None = None,
                 recheck: list[str] | None = None, final: bool = False):
        con.executescript(SCHEMA)
        self.profile = profile
        self.emb = emb or {}
        self.rng = np.random.default_rng(seed)
        self.df = pd.read_sql_query(
            "SELECT f.sha1, f.stage, f.score, f.p_topk, f.weakest, f.selected, s.aesthetic "
            "FROM profile_scores f LEFT JOIN scores s USING(sha1) WHERE f.profile=?",
            con, params=(profile,)).set_index("sha1")
        self.excl = pd.read_sql_query(
            "SELECT r.sha1, r.excluded, s.aesthetic FROM rank r JOIN photos p USING(sha1) "
            "LEFT JOIN scores s USING(sha1) WHERE r.excluded IS NOT NULL AND p.has_thumb=1",
            con).set_index("sha1")
        if self.df.empty:
            raise ValueError(f"no scores for profile {profile!r}: run `python -m photosel profile {profile}` first")
        self.rated = {s for (s,) in con.execute("SELECT sha1 FROM ratings WHERE profile=?", (profile,))}
        self.issued: set[str] = set()
        # earlier ratings to show again first (e.g. "soft" judged on too small an image)
        self.final = final
        self.recheck = [s for (s,) in con.execute(
            f"SELECT sha1 FROM ratings WHERE profile=? AND reason IN ({','.join('?' * len(recheck or []))}) "
            "ORDER BY ts", (profile, *(recheck or [])))] if recheck else []
        if final:
            # final pass: every photo you said maybe (or yes) to, in random order -> print / backup / drop
            pos = [s for (s,) in con.execute("SELECT sha1 FROM ratings WHERE profile=? AND rating >= 0.5 "
                                             "ORDER BY ts", (profile,))]
            self.recheck += [pos[i] for i in self.rng.permutation(len(pos))]
        self.model_p: pd.Series | None = None
        self.lock = threading.Lock()

    # ---- strata -------------------------------------------------------------------------------
    def _pools(self) -> dict[str, tuple[pd.Index, np.ndarray | None]]:
        df = self.df
        sel = df[df["selected"].notna()]
        short = df[(df["stage"] == 2) & df["selected"].isna()].sort_values("score", ascending=False)
        s1 = df[df["stage"] == 1]
        s1 = s1[s1["aesthetic"] >= s1["aesthetic"].median()]          # the pretty ones the profile dropped
        ex = self.excl[self.excl["aesthetic"] >= self.excl["aesthetic"].median()]
        pools = {
            "top": (sel.index, None),
            "edge": (short.index, 1.0 / (1.0 + np.arange(len(short)) / 300.0)),   # favour near the cutoff
            "stage1": (s1.index, None),
            "random": (df.index, None),
            "excluded": (ex.index, None),
        }
        if self.model_p is not None:
            p = self.model_p
            unc = p.reindex(df.index).dropna()
            unc = unc.iloc[np.argsort(np.abs(unc.to_numpy() - 0.5))[:400]]
            gem = p.reindex(df.index[df["selected"].isna()]).dropna().sort_values(ascending=False)[:400]
            pools["uncertain"] = (unc.index, None)
            pools["gem"] = (gem.index, None)
        return pools

    def _draw(self, idx: pd.Index, w: np.ndarray | None, k: int) -> list[str]:
        ok = ~idx.isin(list(self.rated | self.issued))
        idx = idx[ok]
        if len(idx) == 0 or k <= 0:
            return []
        if w is not None:
            w = w[ok] / w[ok].sum()
        pick = self.rng.choice(len(idx), size=min(k, len(idx)), replace=False, p=w)
        return [idx[i] for i in pick]

    def fit(self, con) -> int:
        """Refit the active-learning model (the profile's taste probe); returns the number of ratings."""
        from .taste import fit_taste, predict

        model = fit_taste(con, self.profile, self.emb)
        ids = [s for s in self.df.index if s in self.emb]
        self.model_p = pd.Series(predict(model, self.emb, ids), index=ids) if model is not None else None
        return con.execute("SELECT count(*) FROM ratings WHERE profile=?", (self.profile,)).fetchone()[0]

    def batch(self, n: int = 20) -> list[dict]:
        with self.lock:
            mix = MIX_FINAL if self.final else MIX_MODEL if self.model_p is not None else MIX_COLD
            scale = n / sum(mix.values())
            pools = self._pools()
            out = []
            while self.recheck and len(out) < n:
                s = self.recheck.pop(0)
                self.issued.add(s)
                out.append(self.info(s, "final" if self.final else "recheck"))
            if out:
                return out
            for src, k in mix.items():
                if src not in pools:            # "gem"/"uncertain" need the taste model
                    continue
                for s in self._draw(*pools[src], k=max(1, round(k * scale))):
                    self.issued.add(s)
                    out.append(self.info(s, src))
            self.rng.shuffle(out)
            return out[:n]

    def pair(self) -> dict | None:
        """Two good photos of similar standing but different content, for an A-vs-B choice."""
        with self.lock:
            sel = self.df[self.df["selected"].notna()].sort_values("selected")
            ids = [s for s in sel.index if s in self.emb]
            if len(ids) < 2:
                return None
            for _ in range(50):
                i = int(self.rng.integers(len(ids)))
                j = int(np.clip(i + self.rng.integers(-30, 31), 0, len(ids) - 1))
                if i != j and float(self.emb[ids[i]] @ self.emb[ids[j]]) < 0.8:
                    return {"a": self.info(ids[i], "pair"), "b": self.info(ids[j], "pair")}
            return None

    def info(self, sha1: str, source: str) -> dict:
        if sha1 in self.df.index:
            r = self.df.loc[sha1]
            if pd.notna(r["selected"]):
                verdict = f"selected #{int(r['selected'])}"
            elif r["stage"] == 2:
                verdict = f"shortlisted, not selected (weakest: {r['weakest']})"
            else:
                verdict = f"dropped at stage 1 (weakest: {r['weakest']})"
        elif sha1 in self.excl.index:
            verdict = f"filtered out: {self.excl.at[sha1, 'excluded']}"
        else:
            verdict = "not scored"
        return {"sha1": sha1, "source": source, "verdict": verdict}

    def mark_rated(self, sha1: str, rated: bool):
        with self.lock:
            (self.rated.add if rated else self.rated.discard)(sha1)


# ---- persistence ------------------------------------------------------------------------------
def save_rating(con, log: Path, profile: str, sha1: str, rating: str | None, reason: str | None, source: str):
    if rating is None:
        con.execute("DELETE FROM ratings WHERE profile=? AND sha1=?", (profile, sha1))
    else:
        con.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?,?)",
                    (profile, sha1, RATING[rating], reason, source, time.time()))
    con.commit()
    with log.open("a") as f:
        f.write(json.dumps({"kind": "rating", "profile": profile, "sha1": sha1, "rating": rating,
                            "reason": reason, "source": source, "ts": time.time()}) + "\n")


def save_pair(con, log: Path, profile: str, a: str, b: str, winner: str | None):
    if winner is None:
        con.execute("DELETE FROM pairs WHERE rowid = (SELECT max(rowid) FROM pairs WHERE profile=? AND a=? AND b=?)",
                    (profile, a, b))
    else:
        con.execute("INSERT INTO pairs VALUES (?,?,?,?,?)", (profile, a, b, winner, time.time()))
    con.commit()
    with log.open("a") as f:
        f.write(json.dumps({"kind": "pair", "profile": profile, "a": a, "b": b, "winner": winner,
                            "ts": time.time()}) + "\n")


def stats(con, profile: str) -> dict:
    rows = con.execute("SELECT source, rating, count(*) FROM ratings WHERE profile=? GROUP BY 1, 2",
                       (profile,)).fetchall()
    by = {}
    for src, rating, n in rows:
        key = {1.0: "yes", 0.5: "maybe", 0.0: "no"}[rating]
        by.setdefault(src, {"yes": 0, "maybe": 0, "no": 0})[key] = n
    total = {k: sum(v[k] for v in by.values()) for k in ("yes", "maybe", "no")}
    n_pairs = con.execute("SELECT count(*) FROM pairs WHERE profile=?", (profile,)).fetchone()[0]
    return {"by_source": by, "total": total, "pairs": n_pairs}


VIEW_STEP = 800        # screen-sized views are cached in steps of this many pixels (long edge)
VIEW_MAX = 4800


def view_edge(requested: int) -> int:
    """Round a requested long edge up to a cache step; 0 means full resolution."""
    if requested <= 0:
        return 0
    return int(min(VIEW_MAX, max(VIEW_STEP, -(-requested // VIEW_STEP) * VIEW_STEP)))


def _load_original(con, sha1: str, edge: int):
    row = con.execute("SELECT path, edited_path FROM photos WHERE sha1=?", (sha1,)).fetchone()
    if not row:
        return None
    img = Image.open(io.BytesIO(Path(row[1] or row[0]).read_bytes()))
    if img.format == "JPEG" and edge:
        img.draft("RGB", (edge, edge))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if edge:
        img.thumbnail((edge, edge), Image.LANCZOS)
    return img


def view_image(cfg: Config, con, sha1: str, edge: int = 1600) -> Path | None:
    """A JPEG of the original (the edited version if there is one) with this long edge, cached in work/views."""
    out = cfg.work_dir / "views" / str(edge) / sha1[:2] / f"{sha1}.jpg"
    if out.exists():
        return out
    try:
        img = _load_original(con, sha1, edge)
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{threading.get_ident()}.tmp")
    img.save(tmp, "JPEG", quality=90)
    tmp.replace(out)
    return out


def full_image(con, sha1: str) -> bytes | None:
    """The original at full resolution as JPEG bytes (not cached: only fetched on demand)."""
    try:
        img = _load_original(con, sha1, 0)
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


# ---- server -----------------------------------------------------------------------------------
def make_handler(cfg: Config, sampler: Sampler):
    from .profile import load_profile

    question = load_profile(sampler.profile).get("question", f"Rate photos for '{sampler.profile}'")
    page = (Path(__file__).with_name("review.html").read_text()
            .replace("__PROFILE__", sampler.profile).replace("__QUESTION__", question)
            .replace("__REASONS__", json.dumps(REASONS)))
    log = cfg.work_dir / "ratings_log.jsonl"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _file(self, path: Path | None):
            if path is None or not path.exists():
                return self._send(404, b"not found", "text/plain")
            self._send(200, path.read_bytes(), "image/jpeg", cache=True)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            parts = u.path.strip("/").split("/")
            if u.path == "/":
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if len(parts) == 2 and parts[0] in ("thumb", "view", "full") and SHA1.match(parts[1]):
                if parts[0] == "thumb":
                    return self._file(cfg.thumb_path(parts[1]))
                con = connect(cfg.db_path)
                try:
                    if parts[0] == "full":
                        b = full_image(con, parts[1])
                        return (self._send(200, b, "image/jpeg", cache=True) if b
                                else self._send(404, b"not found", "text/plain"))
                    edge = view_edge(int(q.get("edge", ["1600"])[0]))
                    return self._file(view_image(cfg, con, parts[1], edge))
                finally:
                    con.close()
            con = connect(cfg.db_path)
            try:
                if u.path == "/api/next":
                    if q.get("refit"):
                        sampler.fit(con)
                    return self._json({"items": sampler.batch(int(q.get("n", ["20"])[0])),
                                       "model": sampler.model_p is not None, "final": sampler.final})
                if u.path == "/api/pair":
                    return self._json(sampler.pair())
                if u.path == "/api/stats":
                    return self._json(stats(con, sampler.profile))
            finally:
                con.close()
            self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            con = connect(cfg.db_path)
            try:
                if self.path == "/api/rate":
                    s = body.get("sha1", "")
                    rating, reason = body.get("rating"), body.get("reason")
                    if not SHA1.match(s) or (rating is not None and rating not in RATING) or \
                            (reason is not None and reason not in REASONS):
                        return self._json({"error": "bad request"}, 400)
                    save_rating(con, log, sampler.profile, s, rating, reason, body.get("source", "?"))
                    sampler.mark_rated(s, rating is not None)
                    return self._json(stats(con, sampler.profile))
                if self.path == "/api/pair":
                    a, b, w = body.get("a", ""), body.get("b", ""), body.get("winner")
                    if not (SHA1.match(a) and SHA1.match(b)) or w not in ("a", "b", "=", None):
                        return self._json({"error": "bad request"}, 400)
                    save_pair(con, log, sampler.profile, a, b, w)
                    return self._json(stats(con, sampler.profile))
            finally:
                con.close()
            self._send(404, b"not found", "text/plain")

    return H


def serve(cfg: Config, profile: str, port: int = 8765, recheck: list[str] | None = None,
          final: bool = False):
    from .embed import load_embeddings

    con = connect(cfg.db_path)
    sha1s, E = load_embeddings(con)
    sampler = Sampler(con, profile, emb=dict(zip(sha1s, E)), recheck=recheck, final=final)
    n = sampler.fit(con)
    con.close()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(cfg, sampler))
    print(f"rating page for '{profile}' ({n} ratings so far): http://localhost:{port}  (Ctrl-C to stop)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
