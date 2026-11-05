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
    genre: str | None = None
    """Keep only tracks whose artist has a genre tag containing this text
    (case-insensitive, e.g. ``"soul"`` matches "neo soul" and "classic soul")."""

    def is_active(self) -> bool:
        return any(
            [
                self.year_min is not None,
                self.year_max is not None,
                self.min_popularity is not None,
                self.exclude_explicit,
                self.max_duration_s is not None,
                bool(self.exclude_artists),
                bool(self.genre),
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
            # unknown durations pass
            ok &= sub["duration_ms"].fillna(0).to_numpy() <= self.max_duration_s * 1000
        if self.exclude_artists:
            banned = {a.casefold() for a in self.exclude_artists}
            ok &= ~sub["primary_artist"].str.casefold().isin(banned).to_numpy()
        if self.genre:
            needle = self.genre.casefold().strip()
            ok &= sub["genres_display"].str.casefold().str.contains(needle, regex=False).to_numpy()
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
    artist_budget: dict | None = None,
) -> list[int]:
    """Greedy Maximal-Marginal-Relevance selection with a per-artist cap.

    At each step the candidate maximising
    ``(1 - diversity) * score - diversity * max_sim_to_already_selected``
    is taken, unless its artist already reached ``max_per_artist`` (or its
    own limit in ``artist_budget``, used to cap the seeds' artists).

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
        limit = artist_budget.get(artist, max_per_artist) if artist_budget else max_per_artist
        if artist_counts.get(artist, 0) >= limit:
            continue
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        selected.append(best)
        if diversity > 0:
            redundancy = np.maximum(redundancy, pair_similarity(best, np.arange(n)))
    return selected


def transition_cost(tempo: np.ndarray, energy: np.ndarray, key_angle: np.ndarray, i: int, j: np.ndarray) -> np.ndarray:
    """How jarring it is to go from track ``i`` to tracks ``j``.

    Tempo is compared on a log scale with half/double time treated as a match
    (a 70 BPM ballad mixes into a 140 BPM track), energy linearly, and key by
    distance on the circle of fifths.
    """
    ratio = np.abs(np.log2(tempo[j] / tempo[i]))
    tempo_gap = np.minimum(ratio, np.abs(ratio - 1.0))
    energy_gap = np.abs(energy[j] - energy[i])
    key_gap = np.abs(np.angle(np.exp(1j * (key_angle[j] - key_angle[i])))) / np.pi
    return tempo_gap / 0.08 + energy_gap / 0.15 + key_gap / 0.5


def flow_order(tempo: np.ndarray, energy: np.ndarray, key_angle: np.ndarray) -> list[int]:
    """Order tracks into smooth transitions (greedy walk + 2-opt clean-up).

    Starts from the first track (the most relevant one) and keeps it first.
    Returns positions into the inputs.
    """
    n = len(tempo)
    if n <= 2:
        return list(range(n))
    tempo = np.maximum(np.asarray(tempo, float), 1.0)
    cost = np.vstack([transition_cost(tempo, energy, key_angle, i, np.arange(n)) for i in range(n)])
    order, left = [0], set(range(1, n))
    while left:
        rest = np.fromiter(left, dtype=int)
        nxt = int(rest[np.argmin(cost[order[-1], rest])])
        order.append(nxt)
        left.remove(nxt)
    improved = True
    while improved:  # 2-opt on the open path; the first track stays put
        improved = False
        for a in range(1, n - 1):
            for b in range(a + 1, n):
                before = cost[order[a - 1], order[a]] + (cost[order[b], order[b + 1]] if b + 1 < n else 0.0)
                after = cost[order[a - 1], order[b]] + (cost[order[a], order[b + 1]] if b + 1 < n else 0.0)
                if after < before - 1e-9:
                    order[a : b + 1] = order[a : b + 1][::-1]
                    improved = True
    return order
