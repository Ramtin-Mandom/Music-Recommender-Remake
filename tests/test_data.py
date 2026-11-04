import pytest

from music_recommender.data import base_title, clean_tracks, normalize_text, parse_artists


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("['Queen']", ("Queen",)),
        ("['Queen', 'David Bowie']", ("Queen", "David Bowie")),
        ('["Guns N\' Roses"]', ("Guns N' Roses",)),
        ("Queen, David Bowie", ("Queen", "David Bowie")),
        ("", ()),
        (None, ()),
        (["A", "B"], ("A", "B")),
    ],
)
def test_parse_artists(raw, expected):
    assert parse_artists(raw) == expected


def test_normalize_text_strips_accents_and_punctuation():
    assert normalize_text("  Beyoncé — Halo!! ") == "beyonce halo"
    assert normalize_text(None) == ""


@pytest.mark.parametrize(
    "title",
    [
        "Bohemian Rhapsody",
        "Bohemian Rhapsody - Remastered 2011",
        "Bohemian Rhapsody (Live)",
        "Bohemian Rhapsody [Mono] - Live at Wembley '86",
    ],
)
def test_base_title_collapses_versions(title):
    assert base_title(title) == "bohemian rhapsody"


def test_clean_tracks_drops_broken_and_duplicate_rows(raw_tracks, tracks):
    assert "Broken Tempo" not in set(tracks["name"])  # tempo 0 removed
    assert len(tracks) == len(raw_tracks) - 2  # broken tempo + exact duplicate
    song1 = tracks[tracks["name"] == "Song 1"]
    assert len(song1) == 1 and song1["popularity"].iloc[0] > 0  # most popular copy kept


def test_clean_tracks_adds_helper_columns(tracks):
    for col in ["artist_list", "primary_artist", "artists_display", "title_norm", "artist_norm", "version_key"]:
        assert col in tracks.columns
    queen = tracks[tracks["primary_artist"] == "Queen"]
    assert queen["version_key"].nunique() == 1  # remaster + live are the same song
    assert tracks.index.equals(tracks.reset_index(drop=True).index)


def test_clean_tracks_clips_values(raw_tracks):
    raw = raw_tracks.copy()
    raw.loc[0, "loudness"] = 5.0
    raw.loc[0, "energy"] = 1.7
    out = clean_tracks(raw)
    assert out["loudness"].max() <= 0.0
    assert out["energy"].max() <= 1.0
