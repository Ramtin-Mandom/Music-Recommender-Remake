"""The original CMPT 310 Group 2 model, expressed as an engine preset.

The original ``KNNModel.py`` used:

* 8 audio features (no loudness, key, era, genre or artist information),
* ``StandardScaler`` followed by hand-picked weights
  (tempo x2.5, energy x1.5, danceability x1.3),
* Euclidean k-NN on the plain average of the liked tracks,
* a cap of 2 tracks per artist and no other re-ranking.

Reproducing it as an :class:`EngineConfig` lets the evaluation harness compare
old vs new on exactly the same cleaned catalogue and protocol.  Note that the
original code also fitted ``NearestNeighbors(n_neighbors=5)`` and asked it for
the top 10-15 results, so it could never return more than 4-5 tracks; that
bug is *not* reproduced here (the preset gets a normal candidate pool) so the
comparison is about modelling choices rather than the bug.
"""

from __future__ import annotations

from .config import EngineConfig

LEGACY_WEIGHTS: dict[str, float] = {
    "danceability": 1.3,
    "energy": 1.5,
    "valence": 1.0,
    "acousticness": 1.0,
    "instrumentalness": 1.0,
    "liveness": 1.0,
    "speechiness": 1.0,
    "tempo": 2.5,
}

LEGACY_CONFIG = EngineConfig(
    weights=LEGACY_WEIGHTS,
    scaling="standard",
    metric="euclidean",
    strategy="centroid",
    popularity_weight=0.0,
    diversity=0.0,
    max_per_artist=2,
    dedupe_versions=False,
    dislike_penalty=0.0,
    version_penalty=0.0,
    max_seed_artist=2,
)
