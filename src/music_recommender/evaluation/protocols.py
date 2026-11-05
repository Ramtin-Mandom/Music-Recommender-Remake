"""Offline evaluation protocols.

There is no listening-history data in the dataset, so we need a proxy for
"would this listener like this track".  Three complementary views are used:

1. **Collaborator retrieval** (accuracy proxy).  Artists who feature on each
   other's songs ("Low (feat. T-Pain)") almost always make music for the same
   audience.  Give the engine 1 song (*radio* task) or 3 songs (*playlist*
   task) by an artist, remove that artist's own songs from the candidates, and
   measure how many of the top-10 are by one of the artist's collaborators.
   Nothing about collaborations reaches the model: featured credits are parsed
   from titles, and titles are not features.  That matters because ``genre``
   and ``artist_sound`` are shared by all of an artist's tracks, so the v2
   same-artist protocol would now be circular.
   Re-ranking that deliberately spreads artists (cap, MMR) is disabled for this
   protocol – it measures the ranking signal itself.

2. **Beyond-accuracy** (with each engine's own production re-ranking): artist
   diversity, intra-list audio diversity, genre coherence, catalogue coverage,
   mean popularity, how far recommendations drift in release year from the
   seeds, how often alternate versions slip in and how much of a list is by
   the seeds' own artists.

3. **Bucket agreement** – the *original project's* metric (share of
   neighbours in the same tempo / energy / loudness bucket as the seed), kept
   for continuity.  Beware: it is partly circular because tempo and energy are
   model inputs.

Query artists are split 50/50 into a *validation* half (used by the tuner) and
a *test* half (used for reported numbers) so tuning cannot overfit the report.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import AUDIO_FEATURES
from ..data import normalize_text
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
    artist: str  # artist_key of the seeds' artist
    seeds: tuple[int, ...]
    targets: frozenset[str]  # artist keys of the artist's collaborators
    n_relevant: int  # catalogue tracks by those collaborators
    exclude: tuple[int, ...]  # the seed artist's own rows


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


def collaboration_graph(tracks: pd.DataFrame) -> dict[str, set[str]]:
    """Undirected artist graph from featured credits (``artist_list[1:]``).

    Featured names are matched to catalogue artists by normalised name.
    """
    keys = tracks["artist_key"].to_numpy(dtype=object)
    names = tracks["primary_artist"].map(normalize_text).to_numpy(dtype=object)
    votes: dict[str, Counter] = defaultdict(Counter)
    for name, key in zip(names, keys, strict=True):
        votes[name][key] += 1
    key_by_name = {name: c.most_common(1)[0][0] for name, c in votes.items()}

    graph: dict[str, set[str]] = defaultdict(set)
    for key, artists in zip(keys, tracks["artist_list"], strict=True):
        for featured in artists[1:]:
            other = key_by_name.get(normalize_text(featured))
            if other is not None and other != key:
                graph[key].add(other)
                graph[other].add(key)
    return dict(graph)


class Evaluator:
    """Builds a fixed, seeded query set once and scores any number of engines."""

    def __init__(
        self,
        tracks: pd.DataFrame,
        *,
        split: str = "test",
        n_queries: int = 300,
        k: int = 10,
        min_songs_per_artist: int = 3,
        min_relevant: int = 5,
        seed: int = 7,
    ):
        if split not in ("validation", "test"):
            raise ValueError("split must be 'validation' or 'test'")
        self.tracks = tracks.reset_index(drop=True)
        self.k = k
        rng = np.random.default_rng(seed)

        artist_key = self.tracks["artist_key"].to_numpy(dtype=object)
        rows_by_artist = self.tracks.groupby("artist_key").indices
        songs_per_artist = self.tracks.groupby("artist_key")["version_key"].nunique()
        graph = collaboration_graph(self.tracks)

        def n_tracks(keys) -> int:
            return sum(len(rows_by_artist.get(k_, ())) for k_ in keys)

        eligible = np.array(
            sorted(
                a
                for a, collabs in graph.items()
                if songs_per_artist.get(a, 0) >= min_songs_per_artist and n_tracks(collabs) >= min_relevant
            ),
            dtype=object,
        )
        if len(eligible) < 4:
            raise ValueError("Not enough artists with featured collaborators to evaluate.")
        rng.shuffle(eligible)
        half = len(eligible) // 2
        pool = eligible[:half] if split == "validation" else eligible[half:]
        chosen = rng.choice(pool, size=min(n_queries, len(pool)), replace=False)

        self.radio: list[Query] = []
        self.playlist: list[Query] = []
        for artist in chosen:
            rows = np.asarray(rows_by_artist[artist])
            per_song = (
                self.tracks.iloc[rows]
                .sort_values("popularity", ascending=False)
                .drop_duplicates("version_key")
                .index.to_numpy()
            )
            targets = frozenset(graph[artist])
            for n_seeds, bucket in ((1, self.radio), (3, self.playlist)):
                seeds = rng.choice(per_song, size=min(n_seeds, len(per_song)), replace=False)
                bucket.append(
                    Query(artist, tuple(int(s) for s in seeds), targets, n_tracks(targets), tuple(int(r) for r in rows))
                )

        # Fixed reference space for diversity: standard-scaled raw audio
        # features, identical for every engine so numbers are comparable.
        self._ref = StandardScaler().fit_transform(self.tracks[list(AUDIO_FEATURES)].to_numpy(float))
        self._artists = artist_key
        self._years = self.tracks["year"].to_numpy()
        self._pop = self.tracks["popularity"].to_numpy()
        self._alt = self.tracks["alt_version"].to_numpy(dtype=object)
        self._genres = [frozenset(g) for g in self.tracks["genres"]]
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
            recs = engine.recommend_rows(
                q.seeds, k=k, exclude_rows=q.exclude, max_per_artist=k, diversity=0.0, order="score"
            )
            rel = [self._artists[r] in q.targets for r in recs["row"].to_numpy(dtype=int)]
            p.append(precision_at_k(rel, k))
            h.append(hit_at_k(rel, k))
            n.append(ndcg_at_k(rel, k, q.n_relevant))
        return {"precision@k": float(np.mean(p)), "hit@k": float(np.mean(h)), "ndcg@k": float(np.mean(n))}

    def _beyond(self, engine: Recommender, queries: list[Query]) -> dict[str, float]:
        k = self.k
        art_div, ild, pop, year_gap, coverage, coherence, alt, own = [], [], [], [], set(), [], [], []
        agree = {name: [] for name in self._buckets}
        for q in queries:
            rows = engine.recommend_rows(q.seeds, k=k)["row"].to_numpy(dtype=int)
            if rows.size == 0:
                continue
            art_div.append(unique_ratio(self._artists[rows]))
            ild.append(intra_list_diversity(self._ref[rows]))
            pop.append(self._pop[rows].mean())
            year_gap.append(np.abs(self._years[rows] - self._years[list(q.seeds)].mean()).mean())
            coverage.update(rows.tolist())
            alt.append(np.mean(self._alt[rows] != ""))
            own.append(np.mean(self._artists[rows] == q.artist))
            seed_tags = frozenset().union(*(self._genres[s] for s in q.seeds))
            tagged = [r for r in rows if self._genres[r]]
            if seed_tags and tagged:
                coherence.append(np.mean([bool(self._genres[r] & seed_tags) for r in tagged]))
            for name, labels in self._buckets.items():
                agree[name].append(bucket_agreement(labels[q.seeds[0]], labels[rows]))
        out = {
            "genre_coherence": float(np.mean(coherence)) if coherence else float("nan"),
            "artist_diversity": float(np.mean(art_div)),
            "intra_list_diversity": float(np.mean(ild)),
            "mean_popularity": float(np.mean(pop)),
            "year_gap": float(np.mean(year_gap)),
            "alt_version_rate": float(np.mean(alt)),
            "seed_artist_share": float(np.mean(own)),
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
