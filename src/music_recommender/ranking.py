"""Candidate filtering, scoring helpers and diversity-aware re-ranking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

LN2 = np.log(2.0)


@dataclass(frozen=True)
class Filters:
    """Hard constraints applied to candidates before ranking."""

    year_min: int | None = None
    year_max: int | None = None
    min_popularity: int | None = None
    exclude_explicit: bool = False
    max_duration_s: float | None = None
    exclude_artists: tuple[str, ...] = ()

    def is_active(self) -> bool:
        return any(
            [
                self.year_min is not None,
                self.year_max is not None,
                self.min_popularity is not None,
                self.exclude_explicit,
                self.max_duration_s is not None,
                bool(self.exclude_artists),
            ]
        )

    def mask(self, tracks: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        """Boolean mask over ``rows`` of the candidates that pass every filter."""
        sub = tracks.iloc[rows]
        ok = np.ones(len(rows), dtype=bool)
        if self.year_min is not None:
            ok &= sub["year"].to_numpy() >= self.year_min
        if self.year_max is not None:
            ok &= sub["year"].to_numpy() <= self.year_max
        if self.min_popularity is not None:
            ok &= sub["popularity"].to_numpy() >= self.min_popularity
        if self.exclude_explicit:
            ok &= sub["explicit"].to_numpy() == 0
        if self.max_duration_s is not None:
            ok &= sub["duration_ms"].to_numpy() <= self.max_duration_s * 1000
        if self.exclude_artists:
            banned = {a.casefold() for a in self.exclude_artists}
            ok &= ~sub["primary_artist"].str.casefold().isin(banned).to_numpy()
        return ok


def kernel_similarity(distances: np.ndarray, sigma: float) -> np.ndarray:
    """Map distances to (0, 1] with a Gaussian kernel calibrated so that a
    distance of ``sigma`` (the typical distance to a track's 50th nearest
    neighbour) scores exactly 0.5."""
    d = np.asarray(distances, dtype=float)
    return np.exp(-LN2 * (d / sigma) ** 2)


def mmr_select(
    candidates: np.ndarray,
    scores: np.ndarray,
    k: int,
    *,
    diversity: float,
    pair_similarity: Callable[[int, np.ndarray], np.ndarray],
    artists: np.ndarray,
    max_per_artist: int,
) -> list[int]:
    """Greedy Maximal-Marginal-Relevance selection with a per-artist cap.

    At each step the candidate maximising
    ``(1 - diversity) * score - diversity * max_sim_to_already_selected``
    is taken, unless its artist already reached ``max_per_artist``.

    Returns positions into ``candidates``.
    """
    n = len(candidates)
    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    redundancy = np.zeros(n)
    artist_counts: dict[str, int] = {}

    while len(selected) < k and available.any():
        mmr = (1.0 - diversity) * scores - diversity * redundancy
        mmr[~available] = -np.inf
        best = int(np.argmax(mmr))
        available[best] = False
        artist = artists[best]
        if artist_counts.get(artist, 0) >= max_per_artist:
            continue
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        selected.append(best)
        if diversity > 0:
            redundancy = np.maximum(redundancy, pair_similarity(best, np.arange(n)))
    return selected
