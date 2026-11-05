import numpy as np
import pytest

from music_recommender import LEGACY_CONFIG, EngineConfig, Recommender
from music_recommender.evaluation import (
    Evaluator,
    collaboration_graph,
    hit_at_k,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    random_search,
    run_benchmark,
)


def test_ranking_metrics():
    rel = [True, False, True, False]
    assert precision_at_k(rel, 4) == 0.5
    assert hit_at_k(rel, 1) == 1.0 and hit_at_k([False, True], 1) == 0.0
    assert ndcg_at_k([True, True], 2) == pytest.approx(1.0)
    assert ndcg_at_k([False, True], 2) < ndcg_at_k([True, False], 2)
    assert ndcg_at_k([False, False], 2) == 0.0


def test_intra_list_diversity():
    assert intra_list_diversity(np.zeros((3, 2))) == 0.0
    assert intra_list_diversity(np.array([[0, 0], [3, 4]])) == pytest.approx(5.0)


def test_collaboration_graph(tracks):
    graph = collaboration_graph(tracks)
    assert graph
    for a, others in graph.items():
        assert a not in others
        for b in others:
            assert a in graph[b]  # undirected


def test_splits_are_disjoint(tracks):
    val = Evaluator(tracks, split="validation", n_queries=50)
    test = Evaluator(tracks, split="test", n_queries=50)
    assert {q.artist for q in val.radio}.isdisjoint({q.artist for q in test.radio})
    artist = tracks["artist_key"].to_numpy()
    for q in test.playlist:
        assert len(q.seeds) == 3
        assert set(q.seeds) <= set(q.exclude)
        assert {artist[r] for r in q.exclude} == {q.artist}  # the seed artist's own rows
        assert q.artist not in q.targets and q.n_relevant > 0


def test_evaluate_reports_metrics(tracks):
    ev = Evaluator(tracks, n_queries=20)
    res = ev.evaluate(Recommender(EngineConfig()).fit(tracks), "remake")
    legacy = ev.evaluate(Recommender(LEGACY_CONFIG).fit(tracks), "legacy")
    for key in [
        "radio_precision@k",
        "playlist_ndcg@k",
        "objective",
        "artist_diversity",
        "coverage",
        "genre_coherence",
        "alt_version_rate",
        "seed_artist_share",
        "tempo_bucket_agreement",
    ]:
        assert key in res.metrics
    # collaborators share a genre family; genre + artist sound should find them
    assert res.metrics["objective"] > legacy.metrics["objective"]
    assert res.metrics["genre_coherence"] > 0.8
    assert 0 <= res.metrics["coverage"] <= 1


def test_benchmark_and_tuning_smoke(tracks):
    table = run_benchmark(tracks, {"legacy": LEGACY_CONFIG, "remake": EngineConfig()}, n_queries=10, log=lambda _: None)
    assert list(table["engine"]) == ["legacy", "remake"]
    result = random_search(tracks, trials=2, n_queries=10, log=lambda _: None)
    assert len(result.history) >= 1
    result.best_config.validate()
