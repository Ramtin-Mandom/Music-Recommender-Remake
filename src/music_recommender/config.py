"""Engine configuration.

Every tunable knob of the recommender lives in :class:`EngineConfig` so that an
experiment is fully described by one (serialisable) object.  The defaults are
validated on the v2 dataset with the collaborator protocol, weight sweeps and a
per-feature sensitivity analysis (``music-rec analyze``; see
``docs/EVALUATION.md``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tracks.csv"

#: Continuous Spotify audio descriptors in the dataset (all in [0, 1] except
#: tempo in BPM and loudness in dB).
AUDIO_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
    "loudness",
)

#: Every feature *group* the pipeline can produce.  A group may expand to
#: several columns: ``key`` becomes a sin/cos pair on the circle of fifths,
#: ``genre`` a dense embedding of the artist's genre tags, and
#: ``artist_sound`` the average sound of the rest of the artist's catalogue.
#: Groups the dataset cannot support (``genre`` without genres, ``duration``
#: when most durations are missing) are skipped automatically.
FEATURE_GROUPS: tuple[str, ...] = (
    *AUDIO_FEATURES,
    "key",
    "mode",
    "meter",
    "era",
    "duration",
    "explicit",
    "genre",
    "artist_sound",
)

#: A weight multiplies a feature *after* scaling, so it scales that feature's
#: share of the distance.  Groups missing from the mapping get weight 0 (i.e.
#: are ignored).  ``era`` sits at the knee of the accuracy / audio-coherence
#: trade-off; see docs/EVALUATION.md.
DEFAULT_WEIGHTS: dict[str, float] = {
    "danceability": 1.2,
    "energy": 1.4,
    "valence": 1.0,
    "acousticness": 1.3,
    "instrumentalness": 0.9,
    "liveness": 0.4,
    "speechiness": 1.0,
    "tempo": 0.8,
    "loudness": 0.9,
    "key": 0.15,
    "mode": 0.15,
    "meter": 0.15,
    "era": 1.5,
    "duration": 0.35,
    "explicit": 0.5,
    "genre": 3.0,
    "artist_sound": 1.5,
}

Strategy = Literal["centroid", "nearest_seed", "hybrid"]
Scaling = Literal["standard", "robust", "quantile"]
Order = Literal["score", "flow"]


@dataclass(frozen=True)
class EngineConfig:
    """All hyper-parameters of the recommendation engine."""

    # ---- feature space -------------------------------------------------
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    scaling: Scaling = "quantile"
    """``quantile`` maps every feature to a standard normal (robust to the very
    skewed speechiness / liveness / instrumentalness distributions)."""

    genre_dims: int = 16
    """Size of the genre embedding (TF-IDF over tags -> truncated SVD)."""
    artist_shrinkage: float = 3.0
    """Pseudo-count pulling ``artist_sound`` towards the track itself for
    artists with few songs (an artist with n other songs gets weight
    n / (n + shrinkage) on their average)."""

    # ---- retrieval -----------------------------------------------------
    metric: Literal["euclidean", "cosine", "manhattan"] = "euclidean"
    candidate_pool: int = 300
    """How many raw neighbours to fetch before filtering and re-ranking."""

    # ---- multi-seed profiles ---------------------------------------------
    strategy: Strategy = "hybrid"
    hybrid_alpha: float = 0.5
    """Weight of the centroid score in the ``hybrid`` strategy (the rest goes
    to the closest-seed score)."""
    dislike_penalty: float = 0.6
    """How strongly similarity to a disliked track pushes a candidate down."""

    # ---- re-ranking ------------------------------------------------------
    popularity_weight: float = 0.08
    """Blend factor of the (normalised) popularity prior into the final score."""
    artist_popularity_share: float = 0.3
    """Share of the popularity prior taken from the *artist's* popularity
    (the rest is the track's own; a quarter of tracks have popularity 0)."""
    version_penalty: float = 0.15
    """Score subtracted from alternate versions (live, demo, karaoke, sped up
    ...) unless a seed is the same kind of version."""
    diversity: float = 0.15
    """MMR trade-off: 0 = pure relevance, 1 = pure novelty w.r.t. the list."""
    max_per_artist: int = 2
    max_seed_artist: int = 1
    """How many songs by the seeds' own artists a list may contain."""
    order: Order = "score"
    """``score`` lists by relevance; ``flow`` re-orders the chosen tracks into
    smooth transitions (tempo, energy, key) like a DJ set."""
    dedupe_versions: bool = True
    """Collapse remasters / live versions of the same song by the same artist."""

    # ---- data cleaning ---------------------------------------------------
    min_tempo: float = 30.0
    """Rows with a tempo below this are treated as broken (tempo 0 = unknown)."""
    require_artist: bool = True
    """Drop tracks whose artist name cannot be recovered."""

    random_state: int = 42

    # ------------------------------------------------------------------
    def with_overrides(self, **overrides: Any) -> EngineConfig:
        """Return a copy with some fields replaced (``weights`` is merged)."""
        if "weights" in overrides and overrides["weights"] is not None:
            merged = dict(self.weights)
            merged.update(overrides["weights"])
            overrides["weights"] = merged
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EngineConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        unknown = set(self.weights) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups in weights: {sorted(unknown)}")
        if not any(w > 0 for w in self.weights.values()):
            raise ValueError("At least one feature weight must be positive.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Feature weights must be non-negative.")
        if not 0.0 <= self.diversity <= 1.0:
            raise ValueError("diversity must be in [0, 1].")
        if not 0.0 <= self.popularity_weight <= 1.0:
            raise ValueError("popularity_weight must be in [0, 1].")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be in [0, 1].")
        if self.candidate_pool < 10:
            raise ValueError("candidate_pool must be at least 10.")
        if self.max_per_artist < 1:
            raise ValueError("max_per_artist must be at least 1.")
        if self.order not in ("score", "flow"):
            raise ValueError("order must be 'score' or 'flow'.")
        if self.max_seed_artist < 0:
            raise ValueError("max_seed_artist must be non-negative.")
        if not 0.0 <= self.artist_popularity_share <= 1.0:
            raise ValueError("artist_popularity_share must be in [0, 1].")
        if self.version_penalty < 0:
            raise ValueError("version_penalty must be non-negative.")
        if not 2 <= self.genre_dims <= 128:
            raise ValueError("genre_dims must be between 2 and 128.")
        if self.artist_shrinkage < 0:
            raise ValueError("artist_shrinkage must be non-negative.")
