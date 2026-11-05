"""The recommendation engine.

Pipeline for every request::

    seeds (liked tracks, weights)          disliked tracks
          |                                      |
          v                                      |
    1. embed seeds in the weighted feature space |
    2. candidate generation: k-NN of the         |
       weighted centroid  +  k-NN of each seed   |
    3. hard filters (year, popularity, genre ...) |
    4. relevance = f(centroid sim, best-seed sim) - penalty * dislike sim
    5. blend in a popularity prior (track + artist), push down alternate
       versions (live, demo, karaoke ...) unless a seed is one
    6. collapse versions of the same song
    7. MMR re-ranking for diversity + per-artist cap (stricter for the
       seeds' own artists)
    8. optionally re-order the list into smooth transitions ("flow")
          |
          v
    ranked DataFrame with similarity scores and a short "why" for each track
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from . import __version__
from .catalog import TrackCatalog, TrackRef
from .config import DEFAULT_DATA_PATH, EngineConfig
from .data import load_tracks
from .features import GROUP_LABELS, FeaturePipeline, key_to_fifths_angle
from .importer import PlaylistImport, import_playlist
from .ranking import Filters, flow_order, kernel_similarity, mmr_select

BINARY_GROUPS = frozenset({"explicit", "mode", "meter"})
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

RESULT_COLUMNS = [
    "rank",
    "id",
    "name",
    "artists",
    "year",
    "genres",
    "popularity",
    "similarity",
    "score",
    "why",
    "row",
]

TrackLike = TrackRef | str | tuple | dict | int


class Recommender:
    """Content-based k-NN music recommender.

    Typical use::

        rec = Recommender.from_csv()                 # load + fit (~5 s)
        rec.similar_to("Bohemian Rhapsody", k=10)    # "song radio"
        rec.recommend(["Africa by Toto", "Take On Me"], disliked=["bad guy"])
        rec.from_targets({"energy": 0.9, "valence": 0.8, "tempo": 128})
    """

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.config.validate()
        self.tracks: pd.DataFrame | None = None
        self.catalog: TrackCatalog | None = None
        self.pipeline: FeaturePipeline | None = None
        self.embedding: np.ndarray | None = None
        self.index: NearestNeighbors | None = None
        self.sigma: float = 1.0

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_DATA_PATH, config: EngineConfig | None = None) -> Recommender:
        config = config or EngineConfig()
        tracks = load_tracks(path, min_tempo=config.min_tempo, require_artist=config.require_artist)
        return cls(config).fit(tracks)

    def fit(self, tracks: pd.DataFrame) -> Recommender:
        """Fit on a *cleaned* tracks table (see :func:`music_recommender.data.clean_tracks`)."""
        if len(tracks) < 2:
            raise ValueError("Need at least two tracks to fit the recommender.")
        cfg = self.config
        self.tracks = tracks.reset_index(drop=True)
        self.catalog = TrackCatalog(self.tracks)
        self.pipeline = FeaturePipeline(
            cfg.weights, cfg.scaling, cfg.random_state, genre_dims=cfg.genre_dims, artist_shrinkage=cfg.artist_shrinkage
        )
        self.embedding = self.pipeline.fit_transform(self.tracks)
        # Exact brute-force search.  Euclidean (the default) uses a hand-rolled
        # matrix product: sklearn re-validates the whole catalogue on every
        # query, which costs more than the search itself at ~500k tracks.
        if cfg.metric == "euclidean":
            self.index = None
            self._sq_norms = np.einsum("ij,ij->i", self.embedding, self.embedding)
        else:
            self.index = NearestNeighbors(metric=cfg.metric, algorithm="brute").fit(self.embedding)
        track_pop = self.tracks["popularity"].to_numpy(dtype=float) / 100.0
        artist_pop = self.tracks["artist_popularity"].to_numpy(dtype=float) / 100.0
        self._track_popularity = track_pop
        self._artist_popularity = np.where(np.isnan(artist_pop), track_pop, artist_pop)
        self._artists = self.tracks["artist_key"].to_numpy(dtype=object)
        self._versions = self.tracks["version_key"].to_numpy(dtype=object)
        self._alt = self.tracks["alt_version"].to_numpy(dtype=object)
        self._tempo = self.tracks["tempo"].to_numpy(dtype=float)
        self._energy = self.tracks["energy"].to_numpy(dtype=float)
        self._key_angle = key_to_fifths_angle(self.tracks["key"].to_numpy(), self.tracks["mode"].to_numpy())
        self.sigma = self._calibrate_sigma()
        self._typical_gap = self._typical_group_gaps()
        return self

    def _typical_group_gaps(self, pairs: int = 4000) -> dict[str, float]:
        """Median unweighted per-group distance between random track pairs –
        the yardstick for "unusually close" in explanations."""
        rng = np.random.default_rng(self.config.random_state)
        n = len(self.tracks)
        a, b = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
        diff = (self.embedding[a] - self.embedding[b]) / self.pipeline.column_weights
        out = {}
        for group, cols in self.pipeline.group_slices().items():
            gap = float(np.median(np.sqrt((diff[:, cols] ** 2).sum(axis=1))))
            out[group] = gap if gap > 1e-9 else 1.0
        return out

    def _calibrate_sigma(self, sample: int = 400, neighbour: int = 50) -> float:
        n = len(self.tracks)
        neighbour = min(neighbour, n - 1)
        rng = np.random.default_rng(self.config.random_state)
        rows = rng.choice(n, size=min(sample, n), replace=False)
        dist, _ = self._knn(self.embedding[rows], neighbour + 1)
        sigma = float(np.median(dist[:, -1]))
        return sigma if sigma > 0 else 1.0

    def with_config(self, **overrides: Any) -> Recommender:
        """A new engine sharing this one's data but with a modified config.

        Only re-fits when the feature space changes (weights / scaling /
        metric); re-ranking knobs are applied instantly.
        """
        self._require_fitted()
        new_cfg = self.config.with_overrides(**overrides)
        space_keys = ("weights", "scaling", "metric", "random_state", "genre_dims", "artist_shrinkage")
        if all(getattr(new_cfg, k) == getattr(self.config, k) for k in space_keys):
            clone = object.__new__(Recommender)
            clone.__dict__.update(self.__dict__)
            clone.config = new_cfg
            new_cfg.validate()
            return clone
        return Recommender(new_cfg).fit(self.tracks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        self._require_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": __version__, "engine": self}, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> Recommender:
        blob = joblib.load(path)
        if blob.get("version") != __version__:
            raise ValueError(
                f"Model file was built with version {blob.get('version')}, "
                f"this is {__version__}; rebuild it with `music-rec build`."
            )
        return blob["engine"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, query: str, artist: str | None = None, limit: int = 10) -> pd.DataFrame:
        self._require_fitted()
        return self.catalog.search(query, artist, limit)

    def resolve(self, track: TrackLike) -> int:
        """Catalogue row for a track reference (an ``int`` is taken as a row)."""
        self._require_fitted()
        if isinstance(track, (int, np.integer)):
            if not 0 <= int(track) < len(self.tracks):
                raise IndexError(f"Row {track} out of range.")
            return int(track)
        return self.catalog.resolve(track)

    def import_playlist(self, text: str) -> PlaylistImport:
        """Match a pasted playlist (Spotify links / URIs / "Title - Artist"
        lines / CSV export) to catalogue rows.  ``result.rows`` are the seeds::

            imp = rec.import_playlist(open("playlist.txt").read())
            print(imp.summary())
            rec.recommend_rows(imp.rows, k=20)
        """
        self._require_fitted()
        return import_playlist(text, self.catalog)

    def similar_to(self, track: TrackLike, k: int = 10, filters: Filters | None = None, **overrides) -> pd.DataFrame:
        """Tracks that sound like a single seed track ("song radio")."""
        return self.recommend([track], k=k, filters=filters, **overrides)

    def recommend(
        self,
        liked: Sequence[TrackLike],
        disliked: Sequence[TrackLike] = (),
        k: int = 15,
        filters: Filters | None = None,
        **overrides,
    ) -> pd.DataFrame:
        """Build a playlist from one or more liked (and optional disliked) tracks.

        ``overrides`` temporarily replace config fields for this call only,
        e.g. ``diversity=0.5`` or ``strategy="centroid"``.
        """
        self._require_fitted()
        if not liked:
            raise ValueError("Provide at least one liked track.")
        seed_rows, seed_w = self._resolve_weighted(liked)
        dislike_rows, _ = self._resolve_weighted(disliked)
        return self.recommend_rows(
            seed_rows, k=k, seed_weights=seed_w, disliked_rows=dislike_rows, filters=filters, **overrides
        )

    def recommend_rows(
        self,
        seed_rows: Sequence[int],
        k: int = 15,
        *,
        seed_weights: Sequence[float] | None = None,
        disliked_rows: Sequence[int] = (),
        filters: Filters | None = None,
        exclude_rows: Iterable[int] = (),
        **overrides,
    ) -> pd.DataFrame:
        """Same as :meth:`recommend` but with already-resolved catalogue rows."""
        self._require_fitted()
        cfg = self.config.with_overrides(**overrides) if overrides else self.config
        seed_rows = np.asarray(list(seed_rows), dtype=int)
        weights = np.ones(len(seed_rows)) if seed_weights is None else np.asarray(seed_weights, float)
        seeds = self.embedding[seed_rows]
        centroid = np.average(seeds, axis=0, weights=weights)
        exclude = set(int(r) for r in seed_rows) | set(int(r) for r in disliked_rows) | set(exclude_rows)
        exclude_versions = set(self._versions[seed_rows]) if cfg.dedupe_versions else set()

        queries = np.vstack([centroid] if cfg.strategy == "centroid" or len(seeds) == 1 else [centroid, *seeds])
        disliked_rows = np.asarray(list(disliked_rows), dtype=int)
        pool = max(cfg.candidate_pool, 2 * k)
        while True:
            cand, exhausted = self._candidates(queries, k, pool, filters, exclude, exclude_versions)
            if cand.size == 0:
                return self._empty()
            result = self._rank_candidates(cand, k, cfg, centroid, seeds, weights, disliked_rows, seed_rows)
            # The artist cap / version dedupe can starve a small pool: widen it.
            if len(result) >= k or exhausted:
                return result
            pool *= 4

    def _prior(self, cand: np.ndarray, cfg: EngineConfig) -> np.ndarray:
        share = cfg.artist_popularity_share
        return (1 - share) * self._track_popularity[cand] + share * self._artist_popularity[cand]

    def _shape_scores(self, cand, relevance, cfg, seed_rows=()) -> np.ndarray:
        """Final score: relevance blended with the popularity prior, minus the
        alternate-version penalty."""
        score = (1 - cfg.popularity_weight) * relevance + cfg.popularity_weight * self._prior(cand, cfg)
        if cfg.version_penalty > 0:
            allowed = [self._alt[r] for r in seed_rows]
            alt = self._alt[cand]
            score = score - cfg.version_penalty * ((alt != "") & ~np.isin(alt, allowed))
        return score

    def _select(self, cand, score, relevance, k, cfg, seed_rows=()):
        """Version dedupe, MMR with artist caps, optional flow ordering."""
        if cfg.dedupe_versions:
            keep = self._best_per_version(cand, score)
            cand, score, relevance = cand[keep], score[keep], relevance[keep]

        def pair_sim(i: int, all_idx: np.ndarray) -> np.ndarray:
            d = self._dist(self.embedding[cand[i]][None, :], self.embedding[cand[all_idx]], cfg)[0]
            return kernel_similarity(d, self.sigma)

        budget = {self._artists[r]: min(cfg.max_seed_artist, cfg.max_per_artist) for r in seed_rows}
        picked = np.asarray(
            mmr_select(
                cand,
                score,
                k,
                diversity=cfg.diversity,
                pair_similarity=pair_sim,
                artists=self._artists[cand],
                max_per_artist=cfg.max_per_artist,
                artist_budget=budget,
            ),
            dtype=int,
        )
        if cfg.order == "flow" and len(picked) > 2:
            rows = cand[picked]
            picked = picked[flow_order(self._tempo[rows], self._energy[rows], self._key_angle[rows])]
        return cand[picked], relevance[picked], score[picked]

    def _rank_candidates(self, cand, k, cfg, centroid, seeds, weights, disliked_rows, seed_rows) -> pd.DataFrame:
        cand_emb = self.embedding[cand]
        sim_centroid = kernel_similarity(self._dist(centroid[None, :], cand_emb, cfg)[0], self.sigma)
        seed_sims = kernel_similarity(self._dist(seeds, cand_emb, cfg), self.sigma)
        sim_nearest = (seed_sims * (weights / weights.max())[:, None]).max(axis=0)
        relevance = {
            "centroid": sim_centroid,
            "nearest_seed": sim_nearest,
            "hybrid": cfg.hybrid_alpha * sim_centroid + (1 - cfg.hybrid_alpha) * sim_nearest,
        }[cfg.strategy]
        if len(disliked_rows):
            dis = self.embedding[disliked_rows]
            dis_sim = kernel_similarity(self._dist(dis, cand_emb, cfg), self.sigma).max(axis=0)
            relevance = relevance - cfg.dislike_penalty * dis_sim

        score = self._shape_scores(cand, relevance, cfg, seed_rows)
        rows, rel, sc = self._select(cand, score, relevance, k, cfg, seed_rows)
        anchor = np.vstack([centroid, seeds])
        return self._format(rows, rel, sc, anchor)

    def from_targets(
        self,
        targets: Mapping[str, float],
        k: int = 15,
        filters: Filters | None = None,
        **overrides,
    ) -> pd.DataFrame:
        """Recommend tracks matching a *description* rather than a seed track.

        ``targets`` maps raw attributes to desired values, e.g.
        ``{"energy": 0.9, "danceability": 0.8, "tempo": 125, "year": 1985}``.
        Only the specified dimensions are used in the distance.
        """
        self._require_fitted()
        cfg = self.config.with_overrides(**overrides) if overrides else self.config
        vec, mask = self.pipeline.target_vector(targets)
        if not mask.any():
            raise ValueError("None of the targets correspond to an active feature group.")
        diff = self.embedding[:, mask] - vec[mask]
        dist = np.sqrt((diff**2).sum(axis=1)) * np.sqrt(mask.size / mask.sum())
        rows = np.arange(len(self.tracks))
        if filters is not None and filters.is_active():
            ok = filters.mask(self.tracks, rows)
            rows, dist = rows[ok], dist[ok]
        top = np.argsort(dist)[: max(cfg.candidate_pool, 20 * k)]
        cand, d = rows[top], dist[top]
        relevance = kernel_similarity(d, self.sigma)
        score = self._shape_scores(cand, relevance, cfg)
        rows, rel, sc = self._select(cand, score, relevance, k, cfg)
        anchor = np.where(mask, vec, np.nan)[None, :]
        return self._format(rows, rel, sc, anchor)

    # ------------------------------------------------------------------
    # Explanations / inspection
    # ------------------------------------------------------------------
    def describe(self, rows: Sequence[int]) -> pd.DataFrame:
        """Human-readable raw attributes of some tracks (for tables and radar charts)."""
        self._require_fitted()
        sub = self.tracks.iloc[list(rows)]
        out = sub[["id", "name", "artists_display", "year", "popularity"]].copy()
        out["genres"] = sub["genres_display"]
        for col in ["danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness"]:
            out[col] = sub[col].round(3)
        out["tempo"] = sub["tempo"].round(1)
        out["loudness"] = sub["loudness"].round(1)
        out["key"] = [
            f"{KEY_NAMES[k]} {'major' if m else 'minor'}" for k, m in zip(sub["key"], sub["mode"], strict=True)
        ]
        out["time_signature"] = sub["time_signature"].map(lambda t: f"{t}/4" if t else "?")
        out["duration"] = (sub["duration_ms"] / 60000).map(
            lambda m: "?" if pd.isna(m) else f"{int(m)}:{int((m % 1) * 60):02d}"
        )
        out["explicit"] = sub["explicit"].astype(bool)
        return out.rename(columns={"artists_display": "artists"}).reset_index(drop=True)

    def explain(self, seed_rows: Sequence[int], row: int) -> pd.DataFrame:
        """Per-feature-group breakdown of how close ``row`` is to the seeds' centroid.

        ``share`` is each group's fraction of the total squared distance (the
        groups with the largest share are what makes the track *different*);
        ``closeness`` is a 0-1 per-group similarity in unweighted units.
        """
        self._require_fitted()
        centroid = self.embedding[list(seed_rows)].mean(axis=0)
        diff = (self.embedding[row] - centroid).astype(float)
        unweighted = diff / self.pipeline.column_weights
        total = float((diff**2).sum()) or 1.0
        records = []
        for group, cols in self.pipeline.group_slices().items():
            d2 = float((diff[cols] ** 2).sum())
            u = float(np.sqrt((unweighted[cols] ** 2).sum()))
            records.append(
                {
                    "feature": GROUP_LABELS[group],
                    "group": group,
                    "weight": self.config.weights[group],
                    "closeness": round(float(np.exp(-(u**2) / 2)), 3),
                    "share": round(d2 / total, 3),
                }
            )
        return pd.DataFrame(records).sort_values("share", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if self.embedding is None:
            raise RuntimeError("Recommender is not fitted; call fit() or from_csv() first.")

    def _resolve_weighted(self, items: Sequence[TrackLike]) -> tuple[list[int], list[float]]:
        rows, weights = [], []
        for item in items:
            if isinstance(item, (int, np.integer)):
                rows.append(self.resolve(item))
                weights.append(1.0)
                continue
            ref = TrackRef.parse(item)
            rows.append(self.resolve(ref))
            weights.append(max(float(ref.weight), 1e-6))
        return rows, weights

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray, cfg: EngineConfig) -> np.ndarray:
        return pairwise_distances(a, b, metric=cfg.metric)

    def _knn(self, queries: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        """(distances, indices) of the ``n_neighbors`` nearest catalogue rows, sorted."""
        if self.index is not None:
            return self.index.kneighbors(queries, n_neighbors=n_neighbors)
        q = np.asarray(queries, dtype=np.float32)
        n = len(self.embedding)
        dists, idxs = [], []
        for start in range(0, len(q), 64):  # bound the (queries x catalogue) block
            block = q[start : start + 64]
            d2 = self._sq_norms[None, :] - 2.0 * (block @ self.embedding.T)
            d2 += np.einsum("ij,ij->i", block, block)[:, None]
            if n_neighbors < n:
                part = np.argpartition(d2, n_neighbors - 1, axis=1)[:, :n_neighbors]
            else:
                part = np.tile(np.arange(n), (len(block), 1))
            pd2 = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(pd2, axis=1, kind="stable")
            idxs.append(np.take_along_axis(part, order, axis=1))
            dists.append(np.sqrt(np.maximum(np.take_along_axis(pd2, order, axis=1), 0.0)))
        return np.vstack(dists), np.vstack(idxs)

    def _candidates(self, queries, k, pool, filters, exclude, exclude_versions) -> tuple[np.ndarray, bool]:
        """k-NN candidates for every query vector, filtered, growing the pool
        until enough candidates survive the filters.

        Returns ``(candidates, exhausted)``; ``exhausted`` is True when the
        whole catalogue was searched."""
        n = len(self.tracks)
        need = max(3 * k, 30)
        pool = min(n, pool)
        while True:
            per_query = pool if len(queries) == 1 else max(pool // 2, 2 * k)
            _, idx = self._knn(queries, min(n, per_query))
            cand = np.unique(idx.ravel())
            if exclude:
                cand = cand[~np.isin(cand, list(exclude))]
            if exclude_versions:
                cand = cand[~np.isin(self._versions[cand], list(exclude_versions))]
            if filters is not None and filters.is_active():
                cand = cand[filters.mask(self.tracks, cand)]
            if len(cand) >= need or per_query >= n:
                return cand, per_query >= n
            pool = min(n, pool * 4)

    def _best_per_version(self, cand: np.ndarray, score: np.ndarray) -> np.ndarray:
        """Indices into ``cand`` keeping only the best-scoring copy of each song."""
        order = np.argsort(-score, kind="stable")
        _, first = np.unique(self._versions[cand][order], return_index=True)
        return np.sort(order[first])

    def _reason(self, row: int, anchor: np.ndarray) -> str:
        """Short text naming the feature groups where ``row`` is *unusually*
        close to the nearest anchor (centroid or a seed): the gap is compared
        with the typical gap between two random tracks, and groups that carry
        little weight in the space are ignored."""
        emb = self.embedding[row]
        valid = ~np.isnan(anchor)
        d = np.where(valid, emb - np.nan_to_num(anchor), 0.0)
        nearest = anchor[np.argmin((d**2).sum(axis=1))] if anchor.shape[0] > 1 else anchor[0]
        unweighted = (emb - nearest) / self.pipeline.column_weights
        scores = []
        for group, cols in self.pipeline.group_slices().items():
            if np.isnan(nearest[cols]).any() or self.config.weights[group] < 0.5:
                continue
            # Sharing the common value of a flag (both *not* explicit) says
            # nothing; only a shared rare value is worth mentioning.
            if group in BINARY_GROUPS and not (emb[cols[0]] > 0 and nearest[cols[0]] > 0):
                continue
            gap = float(np.sqrt((unweighted[cols] ** 2).sum()))
            # Floor the ratio so near-exact ties (e.g. two tracks with zero
            # instrumentalness) do not always win; ties go to heavier groups.
            ratio = max(gap / self._typical_gap[group], 0.1)
            scores.append((ratio, -self.config.weights[group], group))
        scores.sort()
        return ", ".join(GROUP_LABELS[g].split(" (")[0].lower() for _, _, g in scores[:3])

    def _format(self, rows, relevance, score, anchor) -> pd.DataFrame:
        if len(rows) == 0:
            return self._empty()
        sub = self.tracks.iloc[rows]
        out = pd.DataFrame(
            {
                "rank": np.arange(1, len(rows) + 1),
                "id": sub["id"].to_numpy(),
                "name": sub["name"].to_numpy(),
                "artists": sub["artists_display"].to_numpy(),
                "year": sub["year"].to_numpy(),
                "genres": sub["genres"].map(lambda g: ", ".join(g[:3])).to_numpy(),
                "popularity": sub["popularity"].to_numpy(),
                "similarity": np.clip(relevance, 0, 1).round(3),
                "score": np.round(score, 4),
                "why": [self._reason(int(r), anchor) for r in rows],
                "row": rows,
            }
        )
        return out[RESULT_COLUMNS]

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=RESULT_COLUMNS)
