"""GPU stage: person detection on thumbnails (how prominent are people in the frame?).

Uses torchvision's COCO-trained Faster R-CNN (MobileNet-v3 FPN backbone), fp32.
Writes the `persons` table: number of confident person boxes and the area of the
largest / all person boxes as a fraction of the frame. Cached; re-runs only
process photos not yet in the table.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .db import connect
from .embed import Thumbs, _loader_init

PERSON = 1          # COCO label id
MIN_SCORE = 0.6


def _collate(batch):
    imgs, idx = zip(*batch)
    return list(imgs), list(idx)


def run_detect(cfg: Config, batch_size: int = 8, limit: int | None = None) -> int:
    from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                              fasterrcnn_mobilenet_v3_large_fpn)

    con = connect(cfg.db_path)
    todo = [r[0] for r in con.execute(
        "SELECT p.sha1 FROM photos p LEFT JOIN persons d USING(sha1) WHERE p.has_thumb=1 AND d.sha1 IS NULL")]
    if limit:
        todo = todo[:limit]
    print(f"detect: {len(todo)} photos")
    if not todo:
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
                                              box_score_thresh=0.3).to(device).eval()
    dl = DataLoader(Thumbs(cfg, todo, TF.to_tensor), batch_size=batch_size, num_workers=3,
                    worker_init_fn=_loader_init, collate_fn=_collate)
    rows = []
    for imgs, idx in tqdm(dl, unit="batch"):
        with torch.no_grad():
            out = model([im.to(device) for im in imgs])
        for im, i, o in zip(imgs, idx, out):
            h, w = im.shape[1:]
            keep = (o["labels"] == PERSON) & (o["scores"] >= MIN_SCORE)
            b = o["boxes"][keep].cpu()
            areas = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (w * h)).tolist() if len(b) else []
            boxes = [[round(v / s, 3) for v, s in zip(bb, (w, h, w, h))] for bb in b.tolist()]
            rows.append((todo[i], len(areas), max(areas, default=0.0), min(1.0, sum(areas)),
                         json.dumps(boxes[:20])))
        if len(rows) >= 400:
            _write(con, rows)
            rows = []
    _write(con, rows)
    con.close()
    return len(todo)


def _write(con, rows):
    con.executemany("INSERT OR REPLACE INTO persons (sha1, n_persons, max_area, total_area, boxes) "
                    "VALUES (?,?,?,?,?)", rows)
    con.commit()
