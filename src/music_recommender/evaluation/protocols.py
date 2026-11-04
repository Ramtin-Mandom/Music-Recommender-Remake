"""Offline evaluation protocols.

There is no listening-history data in the dataset, so we need a proxy for
"would this listener like this track".  Three complementary views are used:

1. **Same-artist retrieval** (accuracy proxy).  Pick an artist with several
   distinct songs, give the engine 1 song (*radio* task) or 3 songs (*playlist*
   task), and measure how many of the top-10 are *other songs by that artist*.
   Artists are not a feature of the model, so this tests whether the feature
   space captures an artist's style rather than echoing an input.  Other
   versions of the seed songs are excluded so a remaster cannot count as a hit.
   Re-ranking that deliberately spreads artists (cap, MMR) is disabled for this
   protocol – it measures the ranking signal itself.

2. **Beyond-accuracy** (with each engine's own production re-ranking): artist
   diversity, intra-list audio diversity, catalogue coverage, mean popularity
   and how far recommendations drift in release year from the seeds.

3. **Bucket agreement** – the *original project's* metric (share of
   neighbours in the same tempo / energy / loudness bucket as the seed), kept
   for continuity.  Beware: it is partly circular because tempo and energy are
   model inputs.

Artists are split 50/50 into a *validation* half (used by the tuner) and a
*test* half (used for reported numbers) so tuning cannot overfit the report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import AUDIO_FEATURES
from ..recommender import Recommender
from .metrics import (
    bucket_agreement,
    hit_at_k,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    unique_ratio,
)


@dataclass(frozen=True)
class Query:
    artist: str
    seeds: tuple[int, ...]
    targets: frozenset[str]  # version keys of the artist's *other* songs
    exclude: tuple[int, ...]  # rows that are versions of the seed songs


@dataclass
class EvalResult:
    name: str
    metrics: dict[str, float]
    seconds: float
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_row(self) -> dict[str, float | str]:
        return {"engine": self.name, **self.metrics, "seconds": round(self.seconds, 1)}


def _tempo_bucket(t):
    return np.digitize(t, [90, 130])


def _energy_bucket(e):
    return np.digitize(e, [0.33, 0.66])


def _loudness_bucket(lv):
    return np.digitize(lv, [-20, -10])


class Evaluator:
    """Builds a fixed, seeded query set once and scores any number of engines."""

    def __init__(
        self,
        tracks: pd.DataFrame,
        *,
        split: str = "test",
        n_queries: int = 300,
        k: int = 10,
        min_songs_per_artist: int = 6,
        seed: int = 7,
    ):
        if split not in ("validation", "test"):
            raise ValueError("split must be 'validation' or 'test'")
        self.tracks = tracks.reset_index(drop=True)
        self.k = k
        rng = np.random.default_rng(seed)

        artist_key = self.tracks["primary_artist"].str.casefold()
        songs_per_artist = self.tracks.groupby(artist_key)["version_key"].nunique()
        eligible = np.array(sorted(songs_per_artist[songs_per_artist >= min_songs_per_artist].index))
        if len(eligible) < 4:
            raise ValueError("Not enough artists with multiple songs to evaluate.")
        rng.shuffle(eligible)
        half = len(eligible) // 2
        pool = eligible[:half] if split == "validation" else eligible[half:]
        chosen = rng.choice(pool, size=min(n_queries, len(pool)), replace=False)

        rows_by_artist = self.tracks.groupby(artist_key).indices
        versions = self.tracks["version_key"].to_numpy(dtype=object)
        self.radio: list[Query] = []
        self.playlist: list[Query] = []
        for artist in chosen:
            rows = np.asarray(rows_by_artist[artist])
            # one representative row (most popular) per distinct song
            per_song = (
                self.tracks.iloc[rows]
                .sort_values("popularity", ascending=False)
                .drop_duplicates("version_key")
                .index.to_numpy()
            )
            for n_seeds, bucket in ((1, self.radio), (3, self.playlist)):
                seeds = rng.choice(per_song, size=n_seeds, replace=False)
                seed_versions = set(versions[seeds])
                targets = frozenset(set(versions[rows]) - seed_versions)
                exclude = tuple(int(r) for r in np.flatnonzero(np.isin(versions, list(seed_versions))))
                bucket.append(Query(artist, tuple(int(s) for s in seeds), targets, exclude))

        # Fixed reference space for diversity: standard-scaled raw audio
        # features, identical for every engine so numbers are comparable.
        self._ref = StandardScaler().fit_transform(self.tracks[list(AUDIO_FEATURES)].to_numpy(float))
        self._versions = versions
        self._artists = artist_key.to_numpy(dtype=object)
        self._years = self.tracks["year"].to_numpy()
        self._pop = self.tracks["popularity"].to_numpy()
        self._buckets = {
            "tempo": _tempo_bucket(self.tracks["tempo"].to_numpy()),
            "energy": _energy_bucket(self.tracks["energy"].to_numpy()),
            "loudness": _loudness_bucket(self.tracks["loudness"].to_numpy()),
        }

    # ------------------------------------------------------------------
    def _accuracy(self, engine: Recommender, queries: list[Query]) -> dict[str, float]:
        p, h, n = [], [], []
        k = self.k
        for q in queries:
            recs = engine.recommend_rows(q.seeds, k=k, exclude_rows=q.exclude, max_per_artist=k, diversity=0.0)
            rows = recs["row"].to_numpy(dtype=int)
            seen: set[str] = set()
            rel = []
            for r in rows:  # a song only counts once even if several versions are listed
                v = self._versions[r]
                rel.append(self._artists[r] == q.artist and v in q.targets and v not in seen)
                seen.add(v)
            p.append(precision_at_k(rel, k))
            h.append(hit_at_k(rel, k))
            n.append(ndcg_at_k(rel, k, len(q.targets)))
        return {"precision@k": float(np.mean(p)), "hit@k": float(np.mean(h)), "ndcg@k": float(np.mean(n))}

    def _beyond(self, engine: Recommender, queries: list[Query]) -> dict[str, float]:
        k = self.k
        art_div, ild, pop, year_gap, coverage = [], [], [], [], set()
        agree = {name: [] for name in self._buckets}
        for q in queries:
            rows = engine.recommend_rows(q.seeds, k=k, exclude_rows=q.exclude)["row"].to_numpy(dtype=int)
            if rows.size == 0:
                continue
            art_div.append(unique_ratio(self._artists[rows]))
            ild.append(intra_list_diversity(self._ref[rows]))
            pop.append(self._pop[rows].mean())
            year_gap.append(np.abs(self._years[rows] - self._years[list(q.seeds)].mean()).mean())
            coverage.update(rows.tolist())
            for name, labels in self._buckets.items():
                agree[name].append(bucket_agreement(labels[q.seeds[0]], labels[rows]))
        out = {
            "artist_diversity": float(np.mean(art_div)),
            "intra_list_diversity": float(np.mean(ild)),
            "mean_popularity": float(np.mean(pop)),
            "year_gap": float(np.mean(year_gap)),
            "coverage": len(coverage) / (len(queries) * k),
        }
        for name, vals in agree.items():
            out[f"{name}_bucket_agreement"] = float(np.mean(vals))
        return out

    def evaluate(self, engine: Recommender, name: str = "engine", *, beyond: bool = True) -> EvalResult:
        start = time.perf_counter()
        radio = self._accuracy(engine, self.radio)
        playlist = self._accuracy(engine, self.playlist)
        metrics = {f"radio_{k}": v for k, v in radio.items()}
        metrics.update({f"playlist_{k}": v for k, v in playlist.items()})
        metrics["objective"] = (radio["precision@k"] + playlist["precision@k"]) / 2
        if beyond:
            metrics.update(self._beyond(engine, self.radio + self.playlist))
        return EvalResult(
            name,
            {k: round(v, 4) for k, v in metrics.items()},
            time.perf_counter() - start,
            {"radio": radio, "playlist": playlist},
        )
