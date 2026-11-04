import numpy as np
import pytest

from music_recommender import LEGACY_CONFIG, EngineConfig, Recommender
from music_recommender.evaluation import (
    Evaluator,
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


def test_splits_are_disjoint(tracks):
    val = Evaluator(tracks, split="validation", n_queries=50, min_songs_per_artist=5)
    test = Evaluator(tracks, split="test", n_queries=50, min_songs_per_artist=5)
    assert {q.artist for q in val.radio}.isdisjoint({q.artist for q in test.radio})
    for q in test.playlist:
        assert len(q.seeds) == 3
        assert set(q.seeds) <= set(q.exclude)


def test_evaluate_reports_metrics(tracks):
    ev = Evaluator(tracks, n_queries=20, min_songs_per_artist=5)
    res = ev.evaluate(Recommender(EngineConfig()).fit(tracks), "remake")
    for key in [
        "radio_precision@k",
        "playlist_ndcg@k",
        "objective",
        "artist_diversity",
        "coverage",
        "tempo_bucket_agreement",
    ]:
        assert key in res.metrics
    # synthetic artists are well separated, so the engine should find them
    assert res.metrics["objective"] > 0.3
    assert 0 <= res.metrics["coverage"] <= 1


def test_benchmark_and_tuning_smoke(tracks):
    table = run_benchmark(tracks, {"legacy": LEGACY_CONFIG, "remake": EngineConfig()}, n_queries=10, log=lambda _: None)
    assert list(table["engine"]) == ["legacy", "remake"]
    result = random_search(tracks, trials=2, n_queries=10, log=lambda _: None)
    assert len(result.history) >= 1
    result.best_config.validate()
