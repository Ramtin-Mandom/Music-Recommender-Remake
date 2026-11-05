"""Feature engineering: raw track attributes -> weighted embedding space.

The original project standard-scaled eight audio features and multiplied three
of them by hand-picked weights.  This pipeline extends that idea:

* **More signal** – loudness, musical key/mode, time signature, release era,
  duration and the explicit flag are available as feature *groups*.
* **Genre** – the artist's Spotify genre tags are embedded with TF-IDF (over
  whole tags *and* their words, so "indie pop" and "la indie" share "indie")
  followed by a truncated SVD.  Artists without tags borrow the average genre
  vector of the artists whose overall sound is closest to theirs.
* **Artist sound** – the average sound of the *rest* of the artist's
  catalogue (leave-one-out, shrunk towards the track itself for artists with
  few songs).  Two tracks by artists with similar catalogues end up close even
  when the tracks themselves differ, which acts as a genre signal that works
  without tags.
* **Better scaling** – skewed features (speechiness, liveness,
  instrumentalness) are mapped through a quantile transform to a standard
  normal, so that a handful of extreme tracks no longer dominate distances.
* **Music-aware key encoding** – keys are placed on the *circle of fifths*
  (minor keys at their relative major), encoded as a sin/cos pair, so that
  harmonically related keys (C <-> G) are closer than unrelated ones (C <-> F#).
* **Group weights** – every group is multiplied by a weight after scaling; a
  weight of 0 removes the group from the space entirely.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, normalize

from .config import AUDIO_FEATURES, FEATURE_GROUPS

#: Continuous groups scaled by the configurable scaler, with the raw column
#: each one is derived from.
_CONTINUOUS: dict[str, str] = {**{f: f for f in AUDIO_FEATURES}, "era": "year", "duration": "duration_ms"}

#: Human readable labels used in explanations and the UI.
GROUP_LABELS: dict[str, str] = {
    "danceability": "Danceability",
    "energy": "Energy",
    "valence": "Mood (valence)",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "speechiness": "Speechiness",
    "tempo": "Tempo",
    "loudness": "Loudness",
    "key": "Key (circle of fifths)",
    "mode": "Major / minor",
    "meter": "Time signature",
    "era": "Release era",
    "duration": "Duration",
    "explicit": "Explicit lyrics",
    "genre": "Genre",
    "artist_sound": "Artist sound",
}

TEMPO_RANGE = (40.0, 220.0)
DURATION_RANGE_MS = (30_000.0, 1_200_000.0)
GENRE_NEIGHBOURS = 10


def key_to_fifths_angle(key: np.ndarray, mode: np.ndarray) -> np.ndarray:
    """Angle (radians) of a key on the circle of fifths.

    Minor keys are mapped to their relative major (A minor -> C major), so the
    two share a position.
    """
    key = np.asarray(key, dtype=int) % 12
    mode = np.asarray(mode, dtype=int)
    major_equiv = np.where(mode == 1, key, (key + 3) % 12)
    position = (major_equiv * 7) % 12
    return 2.0 * np.pi * position / 12.0


def _make_scaler(kind: str, n_rows: int, random_state: int):
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind == "quantile":
        return QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, max(10, n_rows)),
            subsample=200_000,
            random_state=random_state,
        )
    raise ValueError(f"Unknown scaling '{kind}' (expected standard, robust or quantile).")


def _genre_tokens(tags: Sequence[str]) -> list[str]:
    """Whole tags plus their individual words (prefixed so they cannot clash)."""
    out = list(tags)
    for tag in tags:
        out.extend("w:" + w for w in tag.split())
    return out


class GenreEncoder:
    """Artist genre tags -> dense, L2-normalised vectors.

    Fitted on one row per *artist* (so prolific artists do not dominate the
    vocabulary).  Artists without usable tags get the mean vector of the
    ``GENRE_NEIGHBOURS`` tagged artists whose average sound is closest; that
    mean is not re-normalised, so a vague guess stays short and pulls less.
    """

    def __init__(self, dims: int = 16, random_state: int = 42):
        self.dims = dims
        self.random_state = random_state

    def fit(self, genres: Sequence[Sequence[str]], artist_audio: np.ndarray) -> GenreEncoder:
        tagged = np.array([len(g) > 0 for g in genres])
        if tagged.sum() < 3:
            raise ValueError("Not enough artists with genre tags.")
        docs = [g for g, t in zip(genres, tagged, strict=True) if t]
        self._vectorizer = TfidfVectorizer(analyzer=_genre_tokens, min_df=2 if len(docs) > 50 else 1, sublinear_tf=True)
        x = self._vectorizer.fit_transform(docs)
        if x.shape[1] < 3:
            raise ValueError("Genre vocabulary too small.")
        self._svd = TruncatedSVD(n_components=min(self.dims, x.shape[1] - 1), random_state=self.random_state)
        ref = normalize(self._svd.fit_transform(x))
        ok = np.linalg.norm(ref, axis=1) > 0
        self._ref_vectors = ref[ok]
        self._ref_index = NearestNeighbors(n_neighbors=min(GENRE_NEIGHBOURS, int(ok.sum()))).fit(
            artist_audio[tagged][ok]
        )
        self.n_components = self._svd.n_components
        return self

    def transform(self, genres: Sequence[Sequence[str]], artist_audio: np.ndarray) -> np.ndarray:
        out = np.zeros((len(genres), self.n_components))
        tagged = np.array([len(g) > 0 for g in genres], dtype=bool)
        if tagged.any():
            vec = normalize(
                self._svd.transform(self._vectorizer.transform([g for g, t in zip(genres, tagged, strict=True) if t]))
            )
            out[tagged] = vec
        missing = np.linalg.norm(out, axis=1) == 0  # no tags, or only unseen tags
        if missing.any():
            _, idx = self._ref_index.kneighbors(artist_audio[missing])
            out[missing] = self._ref_vectors[idx].mean(axis=1)
        return out


@dataclass
class FeaturePipeline:
    """Fit-once transformer from a tracks table to the weighted embedding."""

    weights: Mapping[str, float]
    scaling: str = "quantile"
    random_state: int = 42
    genre_dims: int = 16
    artist_shrinkage: float = 3.0
    dropped: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
        self.groups: list[str] = [g for g in FEATURE_GROUPS if self.weights.get(g, 0.0) > 0]
        if not self.groups:
            raise ValueError("No feature group has a positive weight.")
        self._scaler = None
        self._genre: GenreEncoder | None = None
        self._medians: dict[str, float] = {}
        self._layout()

    # ------------------------------------------------------------------
    def _layout(self) -> None:
        """Column layout of the output matrix for the current ``groups``."""
        self.columns: list[str] = []
        self.column_groups: list[str] = []
        weights: list[float] = []
        audio_w = self._audio_weights()
        for g in self.groups:
            if g == "key":
                cols, w = ["key_sin", "key_cos"], [self.weights[g]] * 2
            elif g == "genre":
                n = self._genre.n_components if self._genre is not None else self.genre_dims
                cols, w = [f"genre_{i}" for i in range(n)], [self.weights[g]] * n
            elif g == "artist_sound":
                cols = [f"artist_{f}" for f in AUDIO_FEATURES]
                w = list(self.weights[g] * audio_w)
            else:
                cols, w = [g], [self.weights[g]]
            self.columns.extend(cols)
            self.column_groups.extend([g] * len(cols))
            weights.extend(w)
        self.column_weights = np.array(weights, dtype=float)
        # A column with weight 0 (artist_sound of a switched-off audio feature)
        # would break the unweighting in explanations.
        self.column_weights[self.column_weights == 0] = 1e-9

    def _audio_weights(self) -> np.ndarray:
        """Per-feature weights inside ``artist_sound``: the audio weights,
        rescaled to unit root-mean-square so the group weight keeps its meaning."""
        w = np.array([self.weights.get(f, 0.0) for f in AUDIO_FEATURES], dtype=float)
        if not w.any():
            w = np.ones_like(w)
        return w / np.sqrt((w**2).mean())

    def _scale_columns(self) -> list[str]:
        cols = list(AUDIO_FEATURES) + ["era"]
        if "duration" in self.groups:
            cols.append("duration")
        return cols

    @staticmethod
    def _raw_continuous(df: pd.DataFrame, groups: list[str]) -> np.ndarray:
        cols = []
        for g in groups:
            values = df[_CONTINUOUS[g]].to_numpy(dtype=float)
            if g == "tempo":
                values = np.clip(values, *TEMPO_RANGE)
            elif g == "duration":
                values = np.log(np.clip(np.nan_to_num(values, nan=210_000.0), *DURATION_RANGE_MS))
            cols.append(values)
        return np.column_stack(cols)

    @staticmethod
    def _supported(group: str, df: pd.DataFrame) -> bool:
        if group == "duration":
            return "duration_ms" in df and df["duration_ms"].notna().mean() >= 0.5
        if group == "genre":
            return "genres" in df and df["genres"].map(len).gt(0).sum() >= 3
        if group == "artist_sound":
            return "artist_key" in df
        if group == "meter":
            return "time_signature" in df
        return True

    def _artist_stats(self, df: pd.DataFrame, audio: np.ndarray):
        """Factorised artist ids, per-artist audio sums/counts and genre lists."""
        keys = df["artist_key"].to_numpy() if "artist_key" in df else np.arange(len(df)).astype(str)
        codes, uniques = pd.factorize(keys)
        sums = pd.DataFrame(audio).groupby(codes).sum().to_numpy()
        counts = np.bincount(codes, minlength=len(uniques)).astype(float)
        if "genres" in df:
            first = pd.Series(np.arange(len(df))).groupby(codes).first().to_numpy()
            genres = df["genres"].to_numpy()[first]
        else:
            genres = np.empty(len(uniques), dtype=object)
            genres[:] = [()] * len(uniques)
        return codes, sums, counts, genres

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        self.dropped = [g for g in self.groups if not self._supported(g, df)]
        self.groups = [g for g in self.groups if g not in self.dropped]
        raw = self._raw_continuous(df, self._scale_columns())
        self._scaler = _make_scaler(self.scaling, len(df), self.random_state).fit(raw)
        if "genre" in self.groups:
            audio = self._scaler.transform(raw)[:, : len(AUDIO_FEATURES)]
            _, sums, counts, genres = self._artist_stats(df, audio)
            try:
                self._genre = GenreEncoder(self.genre_dims, self.random_state).fit(list(genres), sums / counts[:, None])
            except ValueError:
                self.groups.remove("genre")
                self.dropped.append("genre")
        if not self.groups:
            raise ValueError("No usable feature group for this dataset.")
        self._layout()
        self._medians = {
            col: float(df[col].median())
            for col in {*AUDIO_FEATURES, "year", "duration_ms", "key", "mode", "explicit", "time_signature"}
            if col in df.columns and df[col].notna().any()
        }
        return self

    @property
    def is_fitted(self) -> bool:
        return bool(self._medians)

    def transform(self, df: pd.DataFrame, *, weighted: bool = True) -> np.ndarray:
        """Return the (n_tracks, n_columns) embedding of ``df`` (float32)."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit().")
        n = len(df)
        scale_cols = self._scale_columns()
        scaled = self._scaler.transform(self._raw_continuous(df, scale_cols))
        pos = {g: i for i, g in enumerate(scale_cols)}
        audio = scaled[:, : len(AUDIO_FEATURES)]
        needs_artists = "genre" in self.groups or "artist_sound" in self.groups
        if needs_artists:
            codes, sums, counts, genres = self._artist_stats(df, audio)

        out = np.empty((n, len(self.columns)), dtype=np.float32)
        col = 0
        for g in self.groups:
            if g in pos:
                out[:, col] = scaled[:, pos[g]]
                col += 1
            elif g == "key":
                angle = key_to_fifths_angle(df["key"].to_numpy(), df["mode"].to_numpy())
                out[:, col] = np.sin(angle)
                out[:, col + 1] = np.cos(angle)
                col += 2
            elif g == "meter":
                out[:, col] = (df["time_signature"].fillna(4).to_numpy() != 4).astype(float)
                col += 1
            elif g == "genre":
                vecs = self._genre.transform(list(genres), sums / counts[:, None])
                out[:, col : col + vecs.shape[1]] = vecs[codes]
                col += vecs.shape[1]
            elif g == "artist_sound":
                others = (counts[codes] - 1)[:, None]
                loo = (sums[codes] - audio) / np.maximum(others, 1)
                lam = others / (others + self.artist_shrinkage) if self.artist_shrinkage > 0 else (others > 0) * 1.0
                out[:, col : col + len(AUDIO_FEATURES)] = lam * loo + (1 - lam) * audio
                col += len(AUDIO_FEATURES)
            else:  # binary groups: mode, explicit -> raw 0/1
                out[:, col] = df[g].to_numpy(dtype=float)
                col += 1
        if weighted:
            out *= self.column_weights.astype(np.float32)
        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    def target_vector(self, targets: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Embed a *partial* description of a track (used by the mood explorer).

        ``targets`` maps raw column names (``energy``, ``tempo``, ``year`` ...)
        to desired values.  Returns ``(vector, mask)`` where ``mask`` is True for
        the embedding columns that the targets actually constrain; unspecified
        columns are filled with dataset medians and should be ignored.
        """
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline.target_vector called before fit().")
        row = dict(self._medians)
        row.setdefault("duration_ms", 210_000.0)
        row.setdefault("time_signature", 4.0)
        row.update({k: float(v) for k, v in targets.items()})
        vec = self.transform(pd.DataFrame([row]))[0]

        constrained_groups = set()
        for g in self.groups:
            source = _CONTINUOUS.get(g, "time_signature" if g == "meter" else g)
            if g == "key" and {"key", "mode"} & set(targets):
                constrained_groups.add(g)
            elif g not in ("genre", "artist_sound") and source in targets:
                constrained_groups.add(g)
        mask = np.array([g in constrained_groups for g in self.column_groups])
        return vec, mask

    def group_slices(self) -> dict[str, list[int]]:
        """Map each group to the embedding columns it occupies."""
        out: dict[str, list[int]] = {}
        for i, g in enumerate(self.column_groups):
            out.setdefault(g, []).append(i)
        return out
