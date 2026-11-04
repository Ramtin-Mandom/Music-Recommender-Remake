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
    assert "key_sin" in pipe.columns and "key_cos" in pipe.columns
    assert np.isfinite(X).all()
    unweighted = pipe.transform(tracks, weighted=False)
    assert np.allclose(X, unweighted * pipe.column_weights)


def test_zero_weight_removes_group(tracks):
    pipe = FeaturePipeline({**DEFAULT_WEIGHTS, "era": 0.0, "key": 0.0}).fit(tracks)
    assert "era" not in pipe.columns and "key_sin" not in pipe.columns


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        FeaturePipeline({"not_a_feature": 1.0})
    with pytest.raises(ValueError):
        FeaturePipeline({"energy": 0.0})


def test_target_vector_mask(tracks):
    pipe = FeaturePipeline(DEFAULT_WEIGHTS).fit(tracks)
    vec, mask = pipe.target_vector({"energy": 0.9, "year": 1990})
    constrained = {pipe.column_groups[i] for i in np.flatnonzero(mask)}
    assert constrained == {"energy", "era"}
    assert vec.shape == (len(pipe.columns),)


def test_transform_before_fit_raises(tracks):
    with pytest.raises(RuntimeError):
        FeaturePipeline(DEFAULT_WEIGHTS).transform(tracks)
