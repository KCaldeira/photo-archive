"""Numbered contact sheets of thumbnails, for review by eye or by Claude.

Each sheet is a grid of thumbnails with a large index number on each; a CSV
next to the sheets maps sheet/number -> sha1 and original path.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import Config

CELL = 300   # px per thumbnail cell
COLS, ROWS = 5, 4
LABEL_H = 34


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sheets(cfg: Config, rows: list[dict], name: str, captions: list[str] | None = None) -> list[Path]:
    """rows: dicts with at least sha1 and path. Returns the sheet image paths."""
    out_dir = cfg.sheets_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    big, small = _font(26), _font(15)
    sheets = []
    with open(out_dir / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sheet", "number", "sha1", "path", "caption"])
        for s0 in range(0, len(rows), per):
            chunk = rows[s0:s0 + per]
            sheet_no = s0 // per + 1
            img = Image.new("RGB", (COLS * CELL, ROWS * (CELL + LABEL_H)), (24, 24, 24))
            draw = ImageDraw.Draw(img)
            for k, r in enumerate(chunk):
                num = s0 + k + 1
                x, y = (k % COLS) * CELL, (k // COLS) * (CELL + LABEL_H)
                tp = cfg.thumb_path(r["sha1"])
                if tp.exists():
                    t = Image.open(tp)
                    t.thumbnail((CELL - 8, CELL - 8))
                    img.paste(t, (x + (CELL - t.width) // 2, y + (CELL - t.height) // 2))
                cap = captions[s0 + k] if captions else ""
                draw.text((x + 6, y + CELL + 2), str(num), font=big, fill=(255, 220, 80))
                if cap:
                    draw.text((x + 70, y + CELL + 9), cap[:30], font=small, fill=(200, 200, 200))
                w.writerow([sheet_no, num, r["sha1"], r["path"], cap])
            p = out_dir / f"sheet_{sheet_no:03d}.jpg"
            img.save(p, "JPEG", quality=88)
            sheets.append(p)
    return sheets
