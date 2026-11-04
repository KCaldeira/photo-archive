"""Image-structure metrics.

Stage 1 (all photos, from 512 px thumbnails, CPU):
    coarse_contrast  std of luma at 32 px: is there large-scale tonal structure?
    coarse_edges     mean gradient at 64 px: big shapes that read from across a room
    coarse_ratio     coarse_edges / fine gradient at 512 px: simple strong forms vs fine clutter
    tonal_range      luma p95 - p5 at 64 px

Stage 2 (shortlist only, from the full-resolution originals):
    focus_max        95th percentile of tile sharpness at print scale (3000 px long edge)
    focus_frac       fraction of tiles that are sharp (so shallow depth of field is fine)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import io
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import Config
from .db import connect

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

PRINT_EDGE = 3000      # long edge (px) for focus analysis: ~16x20 in at 150 dpi
TILE = 125             # tile size at PRINT_EDGE
SHARP_TILE = 150.0     # Laplacian variance above which a tile counts as sharp


def _init():
    try:
        os.nice(19)
    except OSError:
        pass
    cv2.setNumThreads(1)


def _grad(g: np.ndarray) -> float:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def _resize_long(g: np.ndarray, edge: int) -> np.ndarray:
    h, w = g.shape[:2]
    s = edge / max(h, w)
    return cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def coarse_metrics(thumb_path: str) -> dict:
    g = cv2.imread(thumb_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return {}
    g = g.astype(np.float32) / 255.0
    g32, g64 = _resize_long(g, 32), _resize_long(g, 64)
    fine = _grad(g)
    coarse = _grad(g64)
    p5, p95 = np.percentile(g64, [5, 95])
    return dict(coarse_contrast=float(g32.std()), coarse_edges=coarse,
                coarse_ratio=coarse / fine if fine > 1e-6 else 0.0, tonal_range=float(p95 - p5))


def _coarse_job(args):
    sha1, path = args
    return sha1, coarse_metrics(path)


def focus_metrics(path: str) -> dict:
    img = Image.open(io.BytesIO(Path(path).read_bytes()))
    if img.format == "JPEG":
        img.draft("L", (PRINT_EDGE, PRINT_EDGE))
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if max(g.shape) > PRINT_EDGE:
        g = _resize_long(g, PRINT_EDGE)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    h, w = lap.shape
    vals = [lap[y:y + TILE, x:x + TILE].var()
            for y in range(0, h - TILE + 1, TILE) for x in range(0, w - TILE + 1, TILE)]
    if not vals:
        return {}
    vals = np.array(vals)
    return dict(focus_max=float(np.percentile(vals, 95)), focus_frac=float((vals > SHARP_TILE).mean()))


def _focus_job(args):
    sha1, path = args
    try:
        return sha1, focus_metrics(path)
    except Exception:  # noqa: BLE001
        return sha1, {}


def _upsert(con, sha1, d: dict):
    if not d:
        return
    cols = list(d)
    con.execute(f"INSERT INTO structure (sha1, {','.join(cols)}) VALUES (?{',?' * len(cols)}) "
                f"ON CONFLICT(sha1) DO UPDATE SET {','.join(f'{c}=excluded.{c}' for c in cols)}",
                [sha1] + [d[c] for c in cols])


def _run(cfg: Config, jobs, fn, workers: int, label: str) -> int:
    con = connect(cfg.db_path)
    print(f"{label}: {len(jobs)} photos")
    with mp.get_context("fork").Pool(workers, initializer=_init) as pool:
        for k, (sha1, d) in enumerate(tqdm(pool.imap_unordered(fn, jobs, chunksize=8), total=len(jobs))):
            _upsert(con, sha1, d)
            if k % 500 == 0:
                con.commit()
    con.commit()
    con.close()
    return len(jobs)


def run_coarse(cfg: Config, workers: int = 4) -> int:
    con = connect(cfg.db_path)
    jobs = [(s, str(cfg.thumb_path(s))) for (s,) in con.execute(
        "SELECT p.sha1 FROM photos p LEFT JOIN structure t USING(sha1) "
        "WHERE p.has_thumb=1 AND t.coarse_contrast IS NULL")]
    con.close()
    return _run(cfg, jobs, _coarse_job, workers, "coarse structure")


def run_focus(cfg: Config, sha1s: list[str], workers: int = 4) -> int:
    con = connect(cfg.db_path)
    done = {s for (s,) in con.execute("SELECT sha1 FROM structure WHERE focus_max IS NOT NULL")}
    paths = dict(con.execute("SELECT sha1, path FROM photos"))
    con.close()
    jobs = [(s, paths[s]) for s in sha1s if s not in done and s in paths]
    return _run(cfg, jobs, _focus_job, workers, "focus map (originals)")
