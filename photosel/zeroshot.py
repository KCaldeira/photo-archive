"""Zero-shot CLIP labels computed from the stored image embeddings.

Text prompts are encoded once and cached in work/models/text_emb.npz, so
labels can be added or reworded in config.yaml and the ranking re-run
without touching the GPU stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Keys starting with "junk_" are candidates for exclusion.
DEFAULT_LABELS = {
    "junk_screenshot": "a screenshot of a phone or computer screen",
    "junk_document": "a photo of a document, receipt, form or printed page",
    "junk_slide": "a photo of a whiteboard, presentation slide or poster",
    "junk_screen": "a photo of a computer monitor or TV screen",
    "junk_meme": "a meme, cartoon or graphic with text",
    "junk_accident": "an accidental photo of the floor, a pocket or a ceiling",
    "junk_object": "a photo of a product, label or object for reference",
    "junk_caption": "a museum wall label or caption card with printed text",
    "artwork": "a photo of a painting or artwork hanging in a museum",
    "people": "a photo of a person or people",
    "group": "a group photo of several people",
    "pet": "a photo of a dog or cat",
    "animal": "a photo of a wild animal or bird",
    "landscape": "a landscape or nature scene",
    "food": "a photo of food or a meal",
    "city": "a photo of a city, street or building",
    "flower": "a close-up photo of a flower or plant",
    "event": "a photo of a party, wedding or celebration",
}


def text_embeddings(prompts: list[str], cache: Path) -> np.ndarray:
    have = {}
    if cache.exists():
        z = np.load(cache)
        have = dict(zip(z["prompts"].tolist(), z["emb"]))
    missing = [p for p in prompts if p not in have]
    if missing:
        import open_clip
        import torch

        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        tok = open_clip.get_tokenizer("ViT-L-14")
        with torch.no_grad():
            t = model.eval().encode_text(tok(missing)).float()
        t = (t / t.norm(dim=-1, keepdim=True)).numpy()
        have.update(zip(missing, t))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, prompts=np.array(list(have)), emb=np.stack(list(have.values())))
    return np.stack([have[p] for p in prompts])


def label_probs(emb: np.ndarray, labels: dict[str, str], cache: Path) -> dict[str, np.ndarray]:
    """Softmax over the labels for each image embedding (rows L2-normalized)."""
    T = text_embeddings(list(labels.values()), cache)
    logits = 100.0 * emb @ T.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return {k: p[:, i] for i, k in enumerate(labels)}
