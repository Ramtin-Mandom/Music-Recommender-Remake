"""Smoke tests on the real 170k-track dataset (``pytest -m slow``)."""

import pytest

from music_recommender import Recommender
from music_recommender.config import DEFAULT_DATA_PATH

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DEFAULT_DATA_PATH.exists(), reason="dataset not present"),
]


@pytest.fixture(scope="module")
def real_engine():
    return Recommender.from_csv()


def test_catalogue_size(real_engine):
    assert 150_000 < len(real_engine.tracks) < 171_000


def test_famous_songs_resolve_to_canonical_artist(real_engine):
    t = real_engine.tracks
    assert t.loc[real_engine.resolve("Bohemian Rhapsody"), "primary_artist"] == "Queen"
    assert t.loc[real_engine.resolve("Africa by Toto"), "primary_artist"].casefold() == "toto"
    assert t.loc[real_engine.resolve("Stand by Me"), "name"] == "Stand by Me"


def test_recommendations_on_real_data(real_engine):
    recs = real_engine.recommend(["Bohemian Rhapsody", "Hotel California", "Stairway to Heaven"], k=15)
    assert len(recs) == 15
    assert recs["artists"].value_counts().max() <= real_engine.config.max_per_artist
    # era-aware: recommendations should stay roughly in the seeds' decade(s)
    assert recs["year"].between(1955, 2000).mean() > 0.7
