"""Learned taste: a linear probe on CLIP embeddings, trained on your ratings for one profile.

With ~200 ratings this predicts held-out ratings better (cross-validated AUC ~0.81 for the
office profile, 2026-11-06) than the hand-built desirability score (~0.72), so profiles use
it as a metric (`source: taste`), with the hand-built metrics kept as guards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_RATINGS = 30
MIN_PER_CLASS = 5


def training_set(con, profile: str, emb: dict, positive: float = 0.5, ignore_reasons=("soft",)):
    """(sha1s, X, y) from the ratings table; a rating >= `positive` counts as a yes.

    A "no" with reason 'personal' (loved, but not for this purpose) is a no like any other.
    Reasons in `ignore_reasons` are left out: "soft" is a flaw of that one frame, not of
    what it shows, so it says nothing about taste (the focus metric covers it).
    """
    r = pd.read_sql_query("SELECT sha1, rating, reason FROM ratings WHERE profile=?", con, params=(profile,))
    r = r[~r["reason"].isin(set(ignore_reasons or [])) & r["sha1"].isin(emb)]
    X = np.stack([emb[s] for s in r["sha1"]]) if len(r) else np.zeros((0, 1))
    return r["sha1"].tolist(), X, (r["rating"] >= positive).to_numpy()


def make_model(C: float = 0.01):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def fit_taste(con, profile: str, emb: dict, positive: float = 0.5, C: float = 0.01, ignore_reasons=("soft",)):
    """Fitted model, or None when there are too few ratings (or only one kind of answer)."""
    try:
        _, X, y = training_set(con, profile, emb, positive, ignore_reasons)
    except Exception:  # noqa: BLE001  (no ratings table yet)
        return None
    if len(y) < MIN_RATINGS or y.sum() < MIN_PER_CLASS or (~y).sum() < MIN_PER_CLASS:
        return None
    return make_model(C).fit(X, y)


def predict(model, emb: dict, sha1s) -> np.ndarray:
    """Probability of a yes for each sha1 (NaN where there is no embedding)."""
    out = np.full(len(sha1s), np.nan)
    have = [i for i, s in enumerate(sha1s) if s in emb]
    if model is not None and have:
        out[have] = model.predict_proba(np.stack([emb[sha1s[i]] for i in have]))[:, 1]
    return out
