"""SQLite schema and helpers.

Two levels:
- `files`: every media file found in the Takeout tree (one photo can appear
  in several folders, e.g. "Photos from 2023" and an album).
- `photos`: one row per distinct image, keyed by SHA-1 of the file bytes.
  Later stages (filters, scores, grades, decisions) add columns/tables keyed
  on sha1.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    size        INTEGER,
    mtime       REAL,
    kind        TEXT,            -- image | video | other
    sha1        TEXT,            -- NULL for videos (not hashed)
    folder      TEXT,            -- parent folder name
    album       TEXT,            -- folder name if it is a named album, else NULL
    is_edited   INTEGER DEFAULT 0,
    sidecar     TEXT,            -- matched JSON sidecar path, if any
    error       TEXT
);
CREATE INDEX IF NOT EXISTS files_sha1 ON files(sha1);

CREATE TABLE IF NOT EXISTS photos (
    sha1         TEXT PRIMARY KEY,
    path         TEXT,           -- a representative original (non-edited) path
    edited_path  TEXT,           -- Google Photos "-edited" version, if any
    format       TEXT,
    width        INTEGER,        -- after EXIF orientation
    height       INTEGER,
    megapixels   REAL,
    taken_at     TEXT,           -- ISO 8601 UTC
    taken_local  TEXT,           -- EXIF DateTimeOriginal (camera local time)
    taken_ts     INTEGER,        -- unix seconds
    date_source  TEXT,           -- sidecar | exif | mtime
    camera_make  TEXT,
    camera_model TEXT,
    lens         TEXT,
    iso          INTEGER,
    exposure_s   REAL,
    f_number     REAL,
    focal_mm     REAL,
    lat          REAL,
    lon          REAL,
    people       TEXT,           -- JSON list of names tagged in Google Photos
    favorited    INTEGER,        -- from sidecar, if Takeout provides it
    albums       TEXT,           -- JSON list of named albums containing it
    description  TEXT,
    -- cheap image statistics, computed on a ~1024 px version
    sharpness    REAL,           -- variance of Laplacian
    sharp_center REAL,           -- same, central 50% region
    brightness   REAL,           -- mean luma 0..255
    contrast     REAL,           -- std of luma
    clip_dark    REAL,           -- fraction of pixels <= 5
    clip_light   REAL,           -- fraction of pixels >= 250
    colorfulness REAL,           -- Hasler-Suesstrunk metric
    dhash        TEXT,           -- 64-bit difference hash, hex
    has_thumb    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS photos_taken ON photos(taken_ts);

-- GPU stage (embed.py): cached, expensive
CREATE TABLE IF NOT EXISTS clip (
    sha1 TEXT PRIMARY KEY,
    emb  BLOB                    -- float16 CLIP ViT-L/14 image embedding, L2-normalized
);
CREATE TABLE IF NOT EXISTS scores (
    sha1      TEXT PRIMARY KEY,
    aesthetic REAL,              -- LAION aesthetic predictor (~1..10)
    iqa       REAL,              -- TOPIQ-NR technical quality (~0..1)
    iaa       REAL,              -- TOPIQ-IAA aesthetic (~1..10)
    zeroshot  TEXT               -- JSON {label: prob}
);

-- Person detection (detect.py): how much of the frame people occupy
CREATE TABLE IF NOT EXISTS persons (
    sha1       TEXT PRIMARY KEY,
    n_persons  INTEGER,          -- confident person boxes (score >= 0.6)
    max_area   REAL,             -- largest person box, fraction of frame
    total_area REAL,             -- sum of person boxes, fraction of frame (capped at 1)
    boxes      TEXT              -- JSON [[x0,y0,x1,y1], ...] normalized, up to 20
);

-- Image structure (structure.py): coarse-scale metrics from thumbnails, focus map from originals
CREATE TABLE IF NOT EXISTS structure (
    sha1          TEXT PRIMARY KEY,
    coarse_contrast REAL,        -- std of luma at 32 px (large-scale tonal structure)
    coarse_edges  REAL,          -- mean gradient magnitude at 64 px
    coarse_ratio  REAL,          -- coarse_edges / fine edge energy at 512 px (simplicity vs clutter)
    tonal_range   REAL,          -- luma p95 - p5 at 64 px
    focus_max     REAL,          -- 95th percentile of tile sharpness on the original (stage 2)
    focus_frac    REAL,          -- fraction of tiles that are sharp (stage 2)
    small_aesthetic REAL         -- LAION aesthetic of the photo shrunk to ~48 px (stage 2)
);

-- Ranking stage (rank.py): cheap, rebuilt on every run from the tables above
CREATE TABLE IF NOT EXISTS rank (
    sha1       TEXT PRIMARY KEY,
    excluded   TEXT,             -- reason, or NULL if kept
    burst_id   INTEGER,
    burst_size INTEGER,
    burst_best INTEGER,          -- 1 if best of its burst
    event_id   INTEGER,
    composite  REAL,             -- weighted score (z-units)
    finalist   INTEGER DEFAULT 0
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=120)  # several stages may write concurrently
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con
