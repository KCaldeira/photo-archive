# photo-archive

Tools to pick the best photos (for printing or other special treatment) out of
a Google Photos library of ~200k photos / 600 GB, working from Google Takeout
exports. The full strategy is in `docs/PLAN.md`.

## Machine and paths

Runs on the desktop (`SmallPinkDot`: 16 cores, GTX 1650 4 GB, CUDA 12.3 driver).

| What | Path | Notes |
| --- | --- | --- |
| Takeout zips | `/mnt/f/photo takeout/takeout-*.zip` | F: has the free space; C: is nearly full |
| Extracted Takeout | `/mnt/f/photo takeout/extracted/` | never modified by the pipeline |
| Selected originals | `/mnt/f/photo takeout/selects/<set>/` | output of export |
| Work dir (DB, thumbs, sheets) | `~/photo-archive/work/` | gitignored; on ext4 for speed and SQLite locking |

Paths live in `config.yaml`.

## Rules

- Photos are private: never publish thumbnails, contact sheets or the review
  page as an Artifact or to any external service. Viewing contact sheets
  locally (Claude reading them in-session) is fine.
- Never delete or modify anything under the Takeout tree; exclusions are flags
  in the DB.
- Heavy stages run under `nice -n 19` with at most 14 workers; GPU stages use
  one process.
- Pipeline is incremental and keyed on file SHA-1, so new Takeout years can be
  added later.

## Running

```
.venv/bin/python -m photosel ingest            # index + thumbnails + cheap stats
.venv/bin/python -m photosel stats             # summary of what is in the DB
.venv/bin/python -m photosel sheet --sample 40 # numbered contact sheets for review
.venv/bin/python -m photosel review office    # local rating page, http://localhost:8765
.venv/bin/python -m photosel review office --final   # final pass over your maybes: maybe print / backup / drop
.venv/bin/python -m photosel profile office   # rescore (refits the taste model on the ratings)
.venv/bin/python -m pytest -q
```
