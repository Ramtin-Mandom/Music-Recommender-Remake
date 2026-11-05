"""Smoke tests on the real ~900k-row v2 dataset (``pytest -m slow``)."""

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


def test_catalogue_size_and_features(real_engine):
    assert 500_000 < len(real_engine.tracks) < 650_000
    assert {"genre", "artist_sound"} <= set(real_engine.pipeline.groups)
    assert real_engine.pipeline.dropped == ["duration"]


def test_famous_songs_resolve_to_canonical_artist(real_engine):
    t = real_engine.tracks
    assert t.loc[real_engine.resolve("Bohemian Rhapsody"), "primary_artist"] == "Queen"
    assert t.loc[real_engine.resolve("Africa by Toto"), "primary_artist"].casefold() == "toto"
    assert t.loc[real_engine.resolve("Stand by Me"), "name"].casefold() == "stand by me"
    assert t.loc[real_engine.resolve("bohemian rapsody"), "primary_artist"] == "Queen"  # typo


def test_recommendations_on_real_data(real_engine):
    recs = real_engine.recommend(["Bohemian Rhapsody", "Hotel California", "Stairway to Heaven"], k=15)
    assert len(recs) == 15
    assert recs["artists"].value_counts().max() <= real_engine.config.max_per_artist
    # era-aware: recommendations should stay roughly in the seeds' decade(s)
    assert recs["year"].between(1955, 2000).mean() > 0.7


def test_genre_aware_radio(real_engine):
    """The O'Jays' "Love Train" should get soul/funk/disco, not folk or country."""
    recs = real_engine.similar_to("Love Train", k=10)
    t = real_engine.tracks.loc[recs["row"]]
    soulful = t["genres_display"].str.contains("soul|funk|disco|motown|r&b", regex=True)
    assert soulful.mean() >= 0.7
    assert (t["artist_key"] == real_engine.tracks.loc[real_engine.resolve("Love Train"), "artist_key"]).sum() <= 1


def test_paste_import_on_real_data(real_engine):
    t = real_engine.tracks
    link = "https://open.spotify.com/track/" + t.loc[real_engine.resolve("Blinding Lights"), "id"] + "?si=abc"
    text = "\n".join([link, "Mr. Brightside - The Killers", "Daft Punk - Get Lucky", "Totally Made Up Song - Nobody"])
    imp = real_engine.import_playlist(text)
    assert [t.loc[r, "primary_artist"] for r in imp.rows] == ["The Weeknd", "The Killers", "Daft Punk"]
    assert [i.status for i in imp.unmatched] == ["no match"]
