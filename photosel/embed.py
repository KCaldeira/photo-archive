"""GPU stage: CLIP embeddings, aesthetic and quality scores, zero-shot labels.

Works from the 512 px thumbnails (on the fast ext4 disk), so it never touches
the originals. Results are cached in SQLite keyed on sha1; re-running only
processes photos without scores.

Scores written to the `scores` table:
- aesthetic   LAION improved aesthetic predictor (MLP on CLIP ViT-L/14), ~1..10
- iqa         pyiqa TOPIQ-NR technical quality (KonIQ-trained), ~0..1
- iaa         pyiqa TOPIQ-IAA aesthetic (AVA-trained), ~1..10
- zeroshot    JSON {label: probability} over the CLIP prompts in LABELS
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config
from .db import connect

AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
                 "sac+logos+ava1-l14-linearMSE.pth")

# Zero-shot labels. Keys starting with "junk_" are candidates for exclusion.
LABELS = {
    "junk_screenshot": "a screenshot of a phone or computer screen",
    "junk_document": "a photo of a document, receipt, form or printed page",
    "junk_slide": "a photo of a whiteboard, presentation slide or poster",
    "junk_screen": "a photo of a computer monitor or TV screen",
    "junk_meme": "a meme, cartoon or graphic with text",
    "junk_accident": "an accidental photo of the floor, a pocket or a ceiling",
    "junk_object": "a photo of a product, label or object for reference",
    "people": "a photo of a person or people",
    "group": "a group photo of several people",
    "landscape": "a landscape or nature scene",
    "animal": "a photo of an animal or bird",
    "food": "a photo of food or a meal",
    "city": "a photo of a city, street or building",
    "flower": "a close-up photo of a flower or plant",
    "event": "a photo of a party, wedding or celebration",
}


class AestheticMLP(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _aesthetic_model(models_dir: Path, device) -> AestheticMLP:
    path = models_dir / "sac+logos+ava1-l14-linearMSE.pth"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AESTHETIC_URL, path)
    m = AestheticMLP()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).eval()


class Thumbs(Dataset):
    def __init__(self, cfg: Config, sha1s: list[str], transform):
        self.cfg, self.sha1s, self.transform = cfg, sha1s, transform

    def __len__(self):
        return len(self.sha1s)

    def __getitem__(self, i):
        img = Image.open(self.cfg.thumb_path(self.sha1s[i])).convert("RGB")
        return self.transform(img), i


def _loader_init(_):
    try:
        os.nice(19)
    except OSError:
        pass


def run_clip(cfg: Config, batch_size: int = 32, limit: int | None = None) -> int:
    """CLIP embeddings + LAION aesthetic + zero-shot labels."""
    import open_clip

    con = connect(cfg.db_path)
    todo = [r[0] for r in con.execute(
        "SELECT p.sha1 FROM photos p LEFT JOIN clip c USING(sha1) WHERE p.has_thumb=1 AND c.sha1 IS NULL")]
    if limit:
        todo = todo[:limit]
    print(f"CLIP: {len(todo)} photos to embed")
    if not todo:
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    # fp32 on purpose: on the GTX 1650 (no tensor cores) fp16 ran ~4x slower (2 vs 10 img/s).
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    aest = _aesthetic_model(cfg.work_dir / "models", device)

    with torch.no_grad():
        t = model.encode_text(tokenizer(list(LABELS.values())).to(device))
        t = (t / t.norm(dim=-1, keepdim=True)).float()

    dl = DataLoader(Thumbs(cfg, todo, preprocess), batch_size=batch_size, num_workers=4,
                    worker_init_fn=_loader_init)
    keys = list(LABELS)
    for imgs, idx in tqdm(dl, unit="batch"):
        with torch.no_grad():
            imgs = imgs.to(device)
            e = model.encode_image(imgs).float()
            e = e / e.norm(dim=-1, keepdim=True)
            a = aest(e).squeeze(1).cpu().numpy()
            p = (100.0 * e @ t.T).softmax(dim=-1).cpu().numpy()
        e16 = e.cpu().numpy().astype(np.float16)
        rows_c, rows_s = [], []
        for j, i in enumerate(idx.tolist()):
            sha1 = todo[i]
            rows_c.append((sha1, e16[j].tobytes()))
            rows_s.append((sha1, float(a[j]), json.dumps({k: round(float(v), 4) for k, v in zip(keys, p[j])})))
        con.executemany("INSERT OR REPLACE INTO clip (sha1, emb) VALUES (?, ?)", rows_c)
        con.executemany(
            "INSERT INTO scores (sha1, aesthetic, zeroshot) VALUES (?, ?, ?) "
            "ON CONFLICT(sha1) DO UPDATE SET aesthetic=excluded.aesthetic, zeroshot=excluded.zeroshot", rows_s)
        con.commit()
    con.close()
    return len(todo)


def run_iqa(cfg: Config, metrics=("topiq_nr", "topiq_iaa"), limit: int | None = None) -> int:
    """No-reference quality (technical) and aesthetic models from pyiqa."""
    import pyiqa
    import torchvision.transforms.functional as TF

    con = connect(cfg.db_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    col = {"topiq_nr": "iqa", "topiq_iaa": "iaa"}
    total = 0
    for metric in metrics:
        c = col[metric]
        todo = [r[0] for r in con.execute(
            f"SELECT p.sha1 FROM photos p LEFT JOIN scores s USING(sha1) WHERE p.has_thumb=1 AND s.{c} IS NULL")]
        if limit:
            todo = todo[:limit]
        print(f"{metric}: {len(todo)} photos")
        if not todo:
            continue
        model = pyiqa.create_metric(metric, device=device)
        # Thumbnails differ in aspect ratio, so run one image at a time.
        dl = DataLoader(Thumbs(cfg, todo, TF.to_tensor), batch_size=None, num_workers=4,
                        worker_init_fn=_loader_init)
        rows = []
        for img, i in tqdm(dl, total=len(todo), unit="img"):
            with torch.no_grad():
                v = float(model(img.unsqueeze(0).to(device)).item())
            rows.append((todo[i], v))
            if len(rows) >= 500:
                _write_scores(con, c, rows)
                rows = []
        _write_scores(con, c, rows)
        total += len(todo)
        del model
        torch.cuda.empty_cache()
    con.close()
    return total


def _write_scores(con, col, rows):
    con.executemany(
        f"INSERT INTO scores (sha1, {col}) VALUES (?, ?) ON CONFLICT(sha1) DO UPDATE SET {col}=excluded.{col}",
        rows)
    con.commit()


def load_embeddings(con) -> tuple[list[str], np.ndarray]:
    rows = con.execute("SELECT sha1, emb FROM clip").fetchall()
    sha1s = [r[0] for r in rows]
    emb = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1)
    return sha1s, emb.astype(np.float32)
