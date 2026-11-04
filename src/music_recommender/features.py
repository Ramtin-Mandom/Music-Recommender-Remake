"""Feature engineering: raw track attributes -> weighted embedding space.

The original project standard-scaled eight audio features and multiplied three
of them by hand-picked weights.  This pipeline extends that idea:

* **More signal** – loudness, musical key/mode, release era, duration and the
  explicit flag are available as optional feature *groups*.
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

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

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
    "era": "Release era",
    "duration": "Duration",
    "explicit": "Explicit lyrics",
}

TEMPO_RANGE = (40.0, 220.0)
DURATION_RANGE_MS = (30_000.0, 1_200_000.0)


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


@dataclass
class FeaturePipeline:
    """Fit-once transformer from a tracks table to the weighted embedding."""

    weights: Mapping[str, float]
    scaling: str = "quantile"
    random_state: int = 42

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
        self.groups: list[str] = [g for g in FEATURE_GROUPS if self.weights.get(g, 0.0) > 0]
        if not self.groups:
            raise ValueError("No feature group has a positive weight.")
        self._continuous = [g for g in self.groups if g in _CONTINUOUS]
        self._scaler = None
        self._medians: dict[str, float] = {}
        # Column layout of the output matrix.
        self.columns: list[str] = []
        self.column_groups: list[str] = []
        for g in self.groups:
            cols = ["key_sin", "key_cos"] if g == "key" else [g]
            self.columns.extend(cols)
            self.column_groups.extend([g] * len(cols))
        self.column_weights = np.array([self.weights[g] for g in self.column_groups], dtype=float)

    # ------------------------------------------------------------------
    @staticmethod
    def _raw_continuous(df: pd.DataFrame, groups: list[str]) -> np.ndarray:
        cols = []
        for g in groups:
            values = df[_CONTINUOUS[g]].to_numpy(dtype=float)
            if g == "tempo":
                values = np.clip(values, *TEMPO_RANGE)
            elif g == "duration":
                values = np.log(np.clip(values, *DURATION_RANGE_MS))
            cols.append(values)
        return np.column_stack(cols) if cols else np.empty((len(df), 0))

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        if self._continuous:
            raw = self._raw_continuous(df, self._continuous)
            self._scaler = _make_scaler(self.scaling, len(df), self.random_state).fit(raw)
        self._medians = {
            col: float(df[col].median())
            for col in {*AUDIO_FEATURES, "year", "duration_ms", "key", "mode", "explicit"}
            if col in df.columns
        }
        return self

    @property
    def is_fitted(self) -> bool:
        return bool(self._medians)

    def transform(self, df: pd.DataFrame, *, weighted: bool = True) -> np.ndarray:
        """Return the (n_tracks, n_columns) embedding of ``df``."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit().")
        n = len(df)
        scaled_cont = (
            self._scaler.transform(self._raw_continuous(df, self._continuous)) if self._continuous else np.empty((n, 0))
        )
        cont_pos = {g: i for i, g in enumerate(self._continuous)}

        out = np.empty((n, len(self.columns)), dtype=float)
        col = 0
        for g in self.groups:
            if g in cont_pos:
                out[:, col] = scaled_cont[:, cont_pos[g]]
                col += 1
            elif g == "key":
                angle = key_to_fifths_angle(df["key"].to_numpy(), df["mode"].to_numpy())
                out[:, col] = np.sin(angle)
                out[:, col + 1] = np.cos(angle)
                col += 2
            else:  # binary groups: mode, explicit -> raw 0/1
                out[:, col] = df[g].to_numpy(dtype=float)
                col += 1
        if weighted:
            out *= self.column_weights
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
        row.update({k: float(v) for k, v in targets.items()})
        vec = self.transform(pd.DataFrame([row]))[0]

        constrained_groups = set()
        for g in self.groups:
            source = _CONTINUOUS.get(g, g)
            if g == "key" and {"key", "mode"} & set(targets):
                constrained_groups.add(g)
            elif source in targets:
                constrained_groups.add(g)
        mask = np.array([g in constrained_groups for g in self.column_groups])
        return vec, mask

    def group_slices(self) -> dict[str, list[int]]:
        """Map each group to the embedding columns it occupies."""
        out: dict[str, list[int]] = {}
        for i, g in enumerate(self.column_groups):
            out.setdefault(g, []).append(i)
        return out
