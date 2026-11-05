import numpy as np
import pytest

from music_recommender.config import DEFAULT_WEIGHTS
from music_recommender.features import FeaturePipeline, key_to_fifths_angle


def test_relative_minor_shares_position_with_major():
    c_major = key_to_fifths_angle(np.array([0]), np.array([1]))
    a_minor = key_to_fifths_angle(np.array([9]), np.array([0]))
    assert np.allclose(c_major, a_minor)


def test_fifth_is_closer_than_tritone():
    def chord(a, b):
        return 2 - 2 * np.cos(a - b)  # squared distance on the unit circle

    c, g, fs = key_to_fifths_angle(np.array([0, 7, 6]), np.array([1, 1, 1]))
    assert chord(c, g) < chord(c, fs)


@pytest.mark.parametrize("scaling", ["standard", "robust", "quantile"])
def test_pipeline_shapes_and_weights(tracks, scaling):
    pipe = FeaturePipeline(DEFAULT_WEIGHTS, scaling=scaling)
    X = pipe.fit_transform(tracks)
    assert X.shape == (len(tracks), len(pipe.columns))
    assert X.dtype == np.float32
    assert {"key_sin", "key_cos", "genre_0", "artist_energy", "meter"} <= set(pipe.columns)
    assert np.isfinite(X).all()
    unweighted = pipe.transform(tracks, weighted=False)
    assert np.allclose(X, unweighted * pipe.column_weights, atol=1e-5)


def test_unsupported_groups_are_dropped(tracks):
    pipe = FeaturePipeline(DEFAULT_WEIGHTS).fit(tracks)
    assert pipe.dropped == ["duration"]  # the v2 data has (almost) no durations
    assert "duration" not in pipe.columns


def test_genre_vectors_per_artist(tracks):
    pipe = FeaturePipeline(DEFAULT_WEIGHTS).fit(tracks)
    X = pipe.transform(tracks, weighted=False)
    cols = pipe.group_slices()["genre"]
    g = X[:, cols]
    for _, rows in tracks.groupby("artist_key").indices.items():
        assert np.allclose(g[rows], g[rows[0]], atol=1e-5)  # one vector per artist
    tagged = tracks["genres"].map(len).to_numpy() > 0
    assert np.allclose(np.linalg.norm(g[tagged], axis=1), 1.0, atol=1e-4)
    # untagged artists borrow a (shorter, non-zero) vector from similar-sounding artists
    norms = np.linalg.norm(g[~tagged], axis=1)
    assert (~tagged).any() and (norms > 0).all() and (norms <= 1 + 1e-4).all()


def test_same_family_genres_are_closer(tracks):
    pipe = FeaturePipeline(DEFAULT_WEIGHTS).fit(tracks)
    g = pipe.transform(tracks, weighted=False)[:, pipe.group_slices()["genre"]]
    tags = tracks["genres_display"]
    rock = g[tags.str.contains("rock").to_numpy()].mean(axis=0)
    rock2 = g[tags.str.contains("classic rock").to_numpy()].mean(axis=0)
    jazz = g[tags.str.contains("jazz").to_numpy()].mean(axis=0)
    assert np.linalg.norm(rock - rock2) < np.linalg.norm(rock - jazz)


def test_artist_sound_is_leave_one_out(tracks):
    pipe = FeaturePipeline({"energy": 1.0, "artist_sound": 1.0}, artist_shrinkage=0.0).fit(tracks)
    X = pipe.transform(tracks, weighted=False)
    energy = X[:, pipe.columns.index("energy")]
    artist_energy = X[:, pipe.columns.index("artist_energy")]
    key = tracks["artist_key"].value_counts().index[0]
    rows = np.flatnonzero(tracks["artist_key"].to_numpy() == key)
    i = rows[0]
    assert artist_energy[i] == pytest.approx(np.delete(energy[rows], 0).mean(), abs=1e-4)


def test_zero_weight_removes_group(tracks):
    pipe = FeaturePipeline({**DEFAULT_WEIGHTS, "era": 0.0, "key": 0.0, "genre": 0.0}).fit(tracks)
    assert "era" not in pipe.columns and "key_sin" not in pipe.columns and "genre_0" not in pipe.columns


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        FeaturePipeline({"not_a_feature": 1.0})
    with pytest.raises(ValueError):
        FeaturePipeline({"energy": 0.0})


def test_target_vector_mask(tracks):
    pipe = FeaturePipeline(DEFAULT_WEIGHTS).fit(tracks)
    vec, mask = pipe.target_vector({"energy": 0.9, "year": 1990, "time_signature": 3})
    constrained = {pipe.column_groups[i] for i in np.flatnonzero(mask)}
    assert constrained == {"energy", "era", "meter"}
    assert vec.shape == (len(pipe.columns),)


def test_transform_before_fit_raises(tracks):
    with pytest.raises(RuntimeError):
        FeaturePipeline(DEFAULT_WEIGHTS).transform(tracks)
