import numpy as np
import pytest

from music_recommender import LEGACY_CONFIG, EngineConfig, Filters, Recommender
from music_recommender.recommender import RESULT_COLUMNS


def test_similar_to_basic_contract(engine):
    recs = engine.similar_to("Song 10", k=8)
    assert list(recs.columns) == RESULT_COLUMNS
    assert len(recs) == 8
    assert list(recs["rank"]) == list(range(1, 9))
    seed = engine.resolve("Song 10")
    assert seed not in set(recs["row"])
    assert recs["similarity"].between(0, 1).all()
    assert recs["score"].is_monotonic_decreasing or engine.config.diversity > 0


def test_returns_k_results_unlike_legacy_bug(engine):
    # The original fitted NearestNeighbors(n_neighbors=5) and could never return 10+.
    assert len(engine.similar_to("Song 3", k=25)) == 25


def test_versions_of_seed_are_excluded(engine):
    recs = engine.similar_to("Bohemian Rhapsody - Remastered 2011", k=20, max_per_artist=5, max_seed_artist=5)
    queen = recs[recs["artists"] == "Queen"]
    assert queen.empty  # the live version is the same song


def test_versions_are_deduplicated(engine, tracks):
    v = tracks["version_key"].to_numpy()
    for seed in ["Song 5", "Song 77"]:
        rows = engine.similar_to(seed, k=30)["row"].to_numpy()
        assert len(set(v[rows])) == len(rows)


def test_max_per_artist(engine):
    recs = engine.similar_to("Song 10", k=20, max_per_artist=1)
    assert recs["artists"].is_unique


def test_filters(engine):
    f = Filters(year_min=1990, min_popularity=50, exclude_explicit=True)
    recs = engine.recommend(["Song 1", "Song 2"], k=10, filters=f)
    assert not recs.empty
    assert (recs["year"] >= 1990).all() and (recs["popularity"] >= 50).all()
    rows = recs["row"].to_numpy()
    assert (engine.tracks.iloc[rows]["explicit"] == 0).all()


def test_impossible_filter_returns_empty(engine):
    assert engine.similar_to("Song 1", k=5, filters=Filters(year_min=3000)).empty


def test_dislike_pushes_tracks_down(engine):
    base = engine.recommend(["Song 10"], k=10, diversity=0.0)
    disliked = int(base.loc[0, "row"])
    after = engine.recommend_rows([engine.resolve("Song 10")], k=10, disliked_rows=[disliked], diversity=0.0)
    assert disliked not in set(after["row"])


@pytest.mark.parametrize("strategy", ["centroid", "nearest_seed", "hybrid"])
def test_strategies(engine, strategy):
    recs = engine.recommend(["Song 1", "Song 50", "Song 99"], k=10, strategy=strategy)
    assert len(recs) == 10


def test_seed_weights_shift_results(engine):
    a, b = engine.resolve("Song 1"), engine.resolve("Song 2")
    heavy_a = engine.recommend_rows([a, b], k=10, seed_weights=[10, 1], strategy="centroid", diversity=0)
    heavy_b = engine.recommend_rows([a, b], k=10, seed_weights=[1, 10], strategy="centroid", diversity=0)
    assert set(heavy_a["row"]) != set(heavy_b["row"])


def test_diversity_increases_spread(engine):
    seed = engine.resolve("Song 10")

    def spread(div):
        rows = engine.recommend_rows([seed], k=10, diversity=div, max_per_artist=10)["row"].to_numpy()
        e = engine.embedding[rows]
        return np.mean([np.linalg.norm(x - y) for i, x in enumerate(e) for y in e[i + 1 :]])

    assert spread(0.8) > spread(0.0)


def test_recovers_artist_structure(engine, tracks):
    """Synthetic artists have a distinct style – neighbours should share it."""
    seed = engine.resolve("Song 10")
    recs = engine.similar_to(seed, k=10, max_per_artist=10, max_seed_artist=10, diversity=0.0, popularity_weight=0.0)
    same = (recs["artists"] == tracks.loc[seed, "artists_display"]).mean()
    assert same >= 0.5


def test_from_targets(engine):
    recs = engine.from_targets({"energy": 0.95, "danceability": 0.9}, k=10)
    assert len(recs) == 10
    rows = recs["row"].to_numpy()
    assert engine.tracks.iloc[rows]["energy"].mean() > engine.tracks["energy"].mean()
    with pytest.raises(ValueError):
        engine.from_targets({"nonexistent": 1.0})


def test_explain_shares_sum_to_one(engine):
    seed = engine.resolve("Song 10")
    rec = int(engine.similar_to(seed, k=1).loc[0, "row"])
    exp = engine.explain([seed], rec)
    assert abs(exp["share"].sum() - 1) < 0.01
    assert exp["closeness"].between(0, 1).all()


def test_describe(engine):
    d = engine.describe([0, 1])
    assert {"key", "duration", "tempo", "genres", "time_signature"} <= set(d.columns)
    assert (d["duration"] == "?").all()  # no durations in the v2 data


def test_seed_artist_cap(engine, tracks):
    seed = engine.resolve("Song 10")
    artist = tracks.loc[seed, "artist_key"]

    def own(**kw):
        rows = engine.recommend_rows([seed], k=15, max_per_artist=10, diversity=0.0, **kw)["row"]
        return int((tracks.loc[rows, "artist_key"] == artist).sum())

    assert own(max_seed_artist=10) > 1  # synthetic artists are tight clusters
    assert own(max_seed_artist=1) == 1
    assert own(max_seed_artist=0) == 0


def test_version_penalty(engine, tracks):
    # Make the closest neighbour of a seed look like a live recording.
    seed = engine.resolve("Song 10")
    first = int(engine.recommend_rows([seed], k=1, version_penalty=0.0, diversity=0.0).loc[0, "row"])
    alt = engine._alt.copy()
    try:
        engine._alt[first] = "live"
        demoted = engine.recommend_rows([seed], k=1, version_penalty=1.0, diversity=0.0)
        assert int(demoted.loc[0, "row"]) != first
        engine._alt[seed] = "live"  # a live seed does not penalise live candidates
        kept = engine.recommend_rows([seed], k=1, version_penalty=1.0, diversity=0.0)
        assert int(kept.loc[0, "row"]) == first
    finally:
        engine._alt[:] = alt


def test_flow_order_keeps_tracks_and_smooths(engine):
    seeds = [engine.resolve("Song 1"), engine.resolve("Song 50")]
    by_score = engine.recommend_rows(seeds, k=12, diversity=0.4)
    flow = engine.recommend_rows(seeds, k=12, diversity=0.4, order="flow")
    assert set(flow["row"]) == set(by_score["row"])
    assert flow.loc[0, "row"] == by_score.loc[0, "row"]  # the best match still opens
    assert list(flow["rank"]) == list(range(1, 13))

    from music_recommender.ranking import transition_cost

    def path_cost(rows):
        rows = list(rows)
        return sum(
            transition_cost(engine._tempo, engine._energy, engine._key_angle, a, np.array([b]))[0]
            for a, b in zip(rows, rows[1:], strict=False)
        )

    assert path_cost(flow["row"]) <= path_cost(by_score["row"]) + 1e-9


def test_genre_filter(engine, tracks):
    recs = engine.similar_to("Song 10", k=5, filters=Filters(genre="jazz"))
    assert not recs.empty
    assert tracks.loc[recs["row"], "genres_display"].str.contains("jazz").all()


def test_genre_drives_neighbours(engine, tracks):
    """Neighbours from other artists should mostly share the seed's genre family."""
    seed = engine.resolve("Song 10")
    tags = set(tracks.loc[seed, "genres"])
    recs = engine.recommend_rows([seed], k=10, max_seed_artist=0, diversity=0.0)
    rec_tags = tracks.loc[recs["row"], "genres"]
    tagged = rec_tags[rec_tags.map(len) > 0]
    assert tagged.map(lambda g: bool(set(g) & tags)).mean() >= 0.7


def test_with_config_reuses_fit_for_ranking_changes(engine):
    clone = engine.with_config(diversity=0.5)
    assert clone.embedding is engine.embedding and clone.config.diversity == 0.5
    refit = engine.with_config(weights={"era": 0.0})
    assert refit.embedding is not engine.embedding
    assert "era" not in refit.pipeline.columns


def test_save_load_roundtrip(engine, tmp_path):
    path = tmp_path / "model.joblib"
    engine.save(path)
    loaded = Recommender.load(path)
    a = engine.similar_to("Song 7", k=5)["id"].tolist()
    b = loaded.similar_to("Song 7", k=5)["id"].tolist()
    assert a == b


def test_legacy_preset_runs(tracks):
    legacy = Recommender(LEGACY_CONFIG).fit(tracks)
    assert legacy.pipeline.groups == [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
    ]
    assert len(legacy.similar_to("Song 3", k=10)) == 10


def test_unfitted_and_bad_input(tracks):
    with pytest.raises(RuntimeError):
        Recommender().similar_to("x")
    eng = Recommender(EngineConfig()).fit(tracks)
    with pytest.raises(ValueError):
        eng.recommend([])
    with pytest.raises(IndexError):
        eng.resolve(10**9)


def test_config_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        EngineConfig(diversity=2).validate()
    with pytest.raises(ValueError):
        EngineConfig(weights={"bogus": 1.0}).validate()
    cfg = EngineConfig().with_overrides(weights={"era": 0.5}, diversity=0.3)
    assert cfg.weights["era"] == 0.5 and cfg.weights["energy"] == EngineConfig().weights["energy"]
    path = tmp_path / "cfg.json"
    cfg.save(path)
    assert EngineConfig.load(path) == cfg
    with pytest.raises(ValueError):
        EngineConfig.from_dict({"nope": 1})
