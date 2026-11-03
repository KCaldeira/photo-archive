"""Summary statistics of the database."""
from __future__ import annotations

from .config import Config
from .db import connect


def print_stats(cfg: Config):
    con = connect(cfg.db_path)
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    print("Files by kind:")
    for kind, n, err in q("SELECT kind, COUNT(*), SUM(error IS NOT NULL) FROM files GROUP BY kind ORDER BY 2 DESC"):
        print(f"  {kind:8s} {n:8d}   errors: {err}")
    img = q("SELECT COUNT(*), SUM(sidecar IS NOT NULL) FROM files WHERE kind='image'")[0]
    if img[0]:
        print(f"Image files with a matched sidecar: {img[1]}/{img[0]} ({100 * img[1] / img[0]:.1f}%)")
    print(f"Distinct photos: {q('SELECT COUNT(*) FROM photos')[0][0]}")
    print("Photos by year (date source):")
    for y, n, sc, ex, mt in q("""SELECT substr(taken_at,1,4), COUNT(*), SUM(date_source='sidecar'),
                                 SUM(date_source='exif'), SUM(date_source='mtime')
                                 FROM photos GROUP BY 1 ORDER BY 1"""):
        print(f"  {y}  {n:7d}   sidecar {sc}  exif {ex}  mtime {mt}")
    for label, sql in [
        ("favorited", "favorited=1"), ("in a named album", "albums IS NOT NULL"),
        ("has people tags", "people IS NOT NULL"), ("has GPS", "lat IS NOT NULL"),
        ("has -edited version", "edited_path IS NOT NULL"), ("< 6 MP", "megapixels < 6"),
        ("no camera EXIF", "camera_model IS NULL"),
    ]:
        print(f"  {label:22s} {q(f'SELECT COUNT(*) FROM photos WHERE {sql}')[0][0]}")
