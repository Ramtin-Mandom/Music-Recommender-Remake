"""Ranking and beyond-accuracy metrics (all pure functions)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precision_at_k(relevant: Sequence[bool], k: int) -> float:
    rel = np.asarray(relevant[:k], dtype=float)
    return float(rel.sum() / k) if k else 0.0


def hit_at_k(relevant: Sequence[bool], k: int) -> float:
    return float(np.any(np.asarray(relevant[:k], dtype=bool)))


def ndcg_at_k(relevant: Sequence[bool], k: int, n_relevant: int | None = None) -> float:
    """Binary-relevance nDCG.  ``n_relevant`` (size of the ground-truth set)
    bounds the ideal DCG; defaults to the number of hits in the list."""
    rel = np.asarray(relevant[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    n_ideal = int(rel.sum()) if n_relevant is None else min(int(n_relevant), k)
    idcg = float(discounts[:n_ideal].sum())
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(vectors: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between the items of one list."""
    n = len(vectors)
    if n < 2:
        return 0.0
    diff = vectors[:, None, :] - vectors[None, :, :]
    d = np.sqrt((diff**2).sum(-1))
    return float(d[np.triu_indices(n, 1)].mean())


def unique_ratio(values: Sequence) -> float:
    return len(set(values)) / len(values) if len(values) else 0.0


def bucket_agreement(seed_label, rec_labels: Sequence) -> float:
    """Share of recommendations whose bucket label equals the seed's."""
    rec_labels = np.asarray(rec_labels, dtype=object)
    return float((rec_labels == seed_label).mean()) if rec_labels.size else 0.0
