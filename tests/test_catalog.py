import pytest

from music_recommender import TrackCatalog, TrackNotFoundError, TrackRef


@pytest.fixture(scope="module")
def catalog(tracks):
    return TrackCatalog(tracks)


def test_exact_title_ignores_version_suffix_and_prefers_popular(catalog, tracks):
    hits = catalog.search("bohemian rhapsody")
    assert hits.loc[0, "name"] == "Bohemian Rhapsody - Remastered 2011"  # popularity 90
    assert (hits["match"] == 1.0).all()


def test_artist_filter(catalog):
    hits = catalog.search("Bohemian Rhapsody", artist="rockabye")
    assert list(hits["artists_display"]) == ["Rockabye Baby!"]


def test_partial_and_fuzzy_matches(catalog):
    assert not catalog.search("rhapsody").empty
    assert catalog.search("bohemian rapsody").loc[0, "name"].startswith("Bohemian")


def test_resolve_variants(catalog, tracks):
    row = catalog.resolve("Bohemian Rhapsody by Rockabye Baby!")
    assert tracks.loc[row, "artists_display"] == "Rockabye Baby!"
    row = catalog.resolve(("Bohemian Rhapsody", "Rockabye"))
    assert tracks.loc[row, "artists_display"] == "Rockabye Baby!"
    # "by" inside a real title must not be split off
    row = catalog.resolve("Stand by Me")
    assert tracks.loc[row, "name"] == "Stand by Me"
    row = catalog.resolve(TrackRef(track_id="sbm0000000000000000001"))
    assert tracks.loc[row, "name"] == "Stand by Me"


def test_trackref_parse_spotify_uri():
    ref = TrackRef.parse("spotify:track:4uLU6hMCjMI75M1A2tKUQC")
    assert ref.track_id == "4uLU6hMCjMI75M1A2tKUQC"


def test_not_found_has_suggestions(catalog):
    with pytest.raises(TrackNotFoundError) as info:
        catalog.resolve("zzzz qqqq nothing")
    assert "No track found" in str(info.value)
    with pytest.raises(TrackNotFoundError):
        catalog.resolve(TrackRef(track_id="does-not-exist"))


def test_empty_query(catalog):
    assert catalog.search("   ").empty
