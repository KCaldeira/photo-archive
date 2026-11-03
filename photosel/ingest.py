"""Stage 0: index the Takeout tree into SQLite, make thumbnails, compute cheap stats.

Each image is read exactly once: the bytes are hashed, decoded at reduced size
(JPEG draft mode), and the reduced image gives the thumbnail plus sharpness,
exposure and colour statistics. Re-running only processes new or changed files.
"""
from __future__ import annotations

import os

# One maths thread per worker process (set before numpy/cv2 import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import io
import json
import multiprocessing as mp
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import Config
from .db import connect
from .takeout import SidecarIndex, album_for_folder, classify, parse_media_name, read_sidecar

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = 300_000_000

EXIF_IFD, GPS_IFD = 0x8769, 0x8825
TAG = dict(make=0x010F, model=0x0110, orientation=0x0112, datetime=0x0132,
           dt_original=0x9003, exposure=0x829A, fnumber=0x829D, iso=0x8827,
           focal=0x920A, lens=0xA434)


# ---------------------------------------------------------------- worker side

def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _gps(gps: dict):
    def dms(vals, ref):
        try:
            d, m, s = (float(x) for x in vals)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        v = d + m / 60 + s / 3600
        return -v if ref in ("S", "W") else v

    if not gps or 2 not in gps or 4 not in gps:
        return None, None
    lat, lon = dms(gps[2], gps.get(1)), dms(gps[4], gps.get(3))
    if lat is None or lon is None or (lat == 0 and lon == 0):
        return None, None
    return lat, lon


def _exif(img: Image.Image) -> dict:
    out: dict = {}
    try:
        ex = img.getexif()
    except Exception:
        return out
    sub = ex.get_ifd(EXIF_IFD) if ex else {}
    def s(v):
        return str(v).strip("\x00 ").strip() or None if v is not None else None
    out["camera_make"] = s(ex.get(TAG["make"]))
    out["camera_model"] = s(ex.get(TAG["model"]))
    out["orientation"] = ex.get(TAG["orientation"], 1)
    dt = sub.get(TAG["dt_original"]) or ex.get(TAG["datetime"])
    if dt:
        try:
            out["taken_local"] = datetime.strptime(str(dt).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            pass
    out["lens"] = s(sub.get(TAG["lens"]))
    iso = sub.get(TAG["iso"])
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    out["iso"] = int(iso) if _num(iso) else None
    out["exposure_s"] = _num(sub.get(TAG["exposure"]))
    out["f_number"] = _num(sub.get(TAG["fnumber"]))
    out["focal_mm"] = _num(sub.get(TAG["focal"]))
    try:
        out["lat"], out["lon"] = _gps(ex.get_ifd(GPS_IFD))
    except Exception:
        pass
    return out


def _stats(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    center = lap[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return dict(
        sharpness=float(lap.var()),
        sharp_center=float(center.var()) if center.size else None,
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        clip_dark=float((gray <= 5).mean()),
        clip_light=float((gray >= 250).mean()),
        colorfulness=float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
    )


def _worker_init():
    try:
        os.nice(19)
    except OSError:
        pass
    cv2.setNumThreads(1)


def process_image(job: dict) -> dict:
    """Hash, read metadata, make thumbnail and stats for one image file."""
    path = Path(job["path"])
    res = {"path": job["path"]}
    try:
        data = path.read_bytes()
        sha1 = hashlib.sha1(data).hexdigest()
        res["sha1"] = sha1
        if job.get("hash_only"):
            return res
        img = Image.open(io.BytesIO(data))
        res["format"] = img.format
        w, h = img.size
        ex = _exif(img)
        if ex.pop("orientation", 1) in (5, 6, 7, 8):
            w, h = h, w
        res.update(width=w, height=h, megapixels=round(w * h / 1e6, 2), **ex)

        a = job["analysis_size"]
        if img.format == "JPEG":
            img.draft("RGB", (a, a))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((a, a), Image.BILINEAR)
        res.update(_stats(np.asarray(img)))
        res["dhash"] = str(imagehash.dhash(img))

        thumb = Path(job["thumbs_dir"]) / sha1[:2] / f"{sha1}.jpg"
        if not thumb.exists():
            thumb.parent.mkdir(parents=True, exist_ok=True)
            t = img.copy()
            t.thumbnail((job["thumb_size"],) * 2, Image.LANCZOS)
            tmp = thumb.with_suffix(".tmp")
            t.save(tmp, "JPEG", quality=85)
            tmp.replace(thumb)
        res["has_thumb"] = 1
    except Exception as e:  # noqa: BLE001 - record and carry on
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    if job.get("sidecar"):
        res["sidecar_data"] = read_sidecar(Path(job["sidecar"]))
    return res


# ---------------------------------------------------------------- main side

PHOTO_COLS = [
    "path", "format", "width", "height", "megapixels", "taken_at", "taken_local", "taken_ts", "date_source",
    "camera_make", "camera_model", "lens", "iso", "exposure_s", "f_number", "focal_mm", "lat", "lon",
    "people", "favorited", "description", "sharpness", "sharp_center", "brightness", "contrast",
    "clip_dark", "clip_light", "colorfulness", "dhash", "has_thumb",
]


def scan(root: Path) -> dict[Path, list[Path]]:
    """All files under root, grouped by folder."""
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for dirpath, _dirs, files in os.walk(root):
        d = Path(dirpath)
        for f in files:
            by_dir[d].append(d / f)
    return by_dir


def _photo_row(r: dict, mtime: float) -> dict:
    sc = r.get("sidecar_data") or {}
    row = {k: r.get(k) for k in PHOTO_COLS}
    for k in ("lat", "lon", "favorited", "description"):
        if sc.get(k) is not None:
            row[k] = sc[k]
    row["people"] = json.dumps(sc["people"]) if sc.get("people") else None
    if sc.get("taken_ts"):
        ts, src = sc["taken_ts"], "sidecar"
    elif r.get("taken_local"):
        ts, src = int(datetime.fromisoformat(r["taken_local"]).replace(tzinfo=timezone.utc).timestamp()), "exif"
    else:
        ts, src = int(mtime), "mtime"
    row["taken_ts"], row["date_source"] = ts, src
    row["taken_at"] = datetime.fromtimestamp(ts, timezone.utc).isoformat()
    return row


def _upsert_photo(con: sqlite3.Connection, sha1: str, row: dict):
    cols = ["sha1"] + list(row)
    con.execute(
        f"INSERT INTO photos ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
        "ON CONFLICT(sha1) DO UPDATE SET "
        "people=COALESCE(photos.people, excluded.people), "
        "favorited=MAX(COALESCE(photos.favorited,0), COALESCE(excluded.favorited,0)), "
        "description=COALESCE(photos.description, excluded.description), "
        "lat=COALESCE(photos.lat, excluded.lat), lon=COALESCE(photos.lon, excluded.lon)",
        [sha1] + list(row.values()),
    )


def _finalize(con: sqlite3.Connection):
    """Albums per photo, and links from originals to their '-edited' versions."""
    albums: dict[str, set] = defaultdict(set)
    for sha1, album in con.execute("SELECT sha1, album FROM files WHERE album IS NOT NULL AND sha1 IS NOT NULL"):
        albums[sha1].add(album)
    con.execute("UPDATE photos SET albums=NULL")
    con.executemany("UPDATE photos SET albums=? WHERE sha1=?",
                    [(json.dumps(sorted(a)), s) for s, a in albums.items()])

    files_by_dir: dict[str, dict[str, str]] = defaultdict(dict)
    for path, sha1 in con.execute("SELECT path, sha1 FROM files WHERE kind='image' AND is_edited=0"):
        p = Path(path)
        files_by_dir[str(p.parent)][p.name.lower()] = sha1
    links = []
    for path, in con.execute("SELECT path FROM files WHERE kind='image' AND is_edited=1"):
        p = Path(path)
        mn = parse_media_name(p.name)
        stem, _, ext = mn.original.rpartition(".")
        orig_name = f"{stem}({mn.n}).{ext}" if mn.n else mn.original
        sha1 = files_by_dir[str(p.parent)].get(orig_name.lower())
        if sha1:
            links.append((path, sha1))
    con.executemany("UPDATE photos SET edited_path=? WHERE sha1=?", links)
    con.commit()
    return len(links)


def run_ingest(cfg: Config, limit: int | None = None) -> dict:
    con = connect(cfg.db_path)
    root = cfg.takeout_dir
    if not root.exists():
        raise SystemExit(f"Takeout folder not found: {root}")

    known = {p: (s, m) for p, s, m in con.execute("SELECT path, size, mtime FROM files WHERE error IS NULL")}
    jobs, plain_rows = [], []
    n_sidecar = n_images = 0
    print(f"Scanning {root} ...")
    for d, paths in scan(root).items():
        index = SidecarIndex([p for p in paths if classify(p) == "json"])
        album = album_for_folder(d.name)
        for p in paths:
            kind = classify(p)
            if kind == "json":
                continue
            st = p.stat()
            if known.get(str(p)) == (st.st_size, st.st_mtime):
                continue
            mn = parse_media_name(p.name)
            sidecar = index.match(p.name) if kind in ("image", "raw", "video") else None
            base = dict(path=str(p), size=st.st_size, mtime=st.st_mtime, kind=kind, folder=d.name,
                        album=album, is_edited=int(mn.is_edited), sidecar=str(sidecar) if sidecar else None)
            if kind == "image":
                n_images += 1
                n_sidecar += sidecar is not None
                jobs.append(dict(base, analysis_size=cfg.analysis_size, thumb_size=cfg.thumb_size,
                                 hash_only=mn.is_edited))
            else:
                plain_rows.append(base)

    file_cols = ["path", "size", "mtime", "kind", "sha1", "folder", "album", "is_edited", "sidecar", "error"]
    con.executemany(
        f"INSERT OR REPLACE INTO files ({','.join(file_cols)}) VALUES ({','.join('?' * len(file_cols))})",
        [[r.get(c) for c in file_cols] for r in plain_rows],
    )
    con.commit()
    if limit:
        jobs = jobs[:limit]
    print(f"New/changed: {len(jobs)} images ({n_sidecar}/{n_images} with sidecar), "
          f"{len(plain_rows)} other files")

    for j in jobs:
        j["thumbs_dir"] = str(cfg.thumbs_dir)
    errors = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(cfg.workers, initializer=_worker_init) as pool:
        it = pool.imap_unordered(_process_job, jobs, chunksize=4)
        for i, r in enumerate(tqdm(it, total=len(jobs), unit="img", smoothing=0.05)):
            job = r.pop("_job")
            errors += "error" in r
            file_row = dict(job, sha1=r.get("sha1"), error=r.get("error"))
            con.execute(
                f"INSERT OR REPLACE INTO files ({','.join(file_cols)}) VALUES ({','.join('?' * len(file_cols))})",
                [file_row.get(c) for c in file_cols],
            )
            if r.get("sha1") and "error" not in r and not job["is_edited"]:
                _upsert_photo(con, r["sha1"], _photo_row(r, job["mtime"]))
            if i % 500 == 0:
                con.commit()
    con.commit()
    n_links = _finalize(con)
    n_photos = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    con.close()
    return dict(processed=len(jobs), errors=errors, photos=n_photos, edited_links=n_links)


def _process_job(job: dict) -> dict:
    res = process_image(job)
    res["_job"] = {k: v for k, v in job.items()
                   if k not in ("analysis_size", "thumb_size", "hash_only", "thumbs_dir")}
    return res
