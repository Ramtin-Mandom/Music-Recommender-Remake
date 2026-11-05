import pandas as pd
import pytest

from music_recommender.data import (
    alt_version,
    base_title,
    clean_tracks,
    detect_schema,
    featured_artists,
    normalize_text,
    parse_artists,
    parse_genres,
    standardize,
)


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("['indie pop', 'la indie']", ("indie pop", "la indie")),
        ("[\"children's music\", 'lullaby']", ("children's music", "lullaby")),
        ("[]", ()),
        (None, ()),
    ],
)
def test_parse_genres(raw, expected):
    assert parse_genres(raw) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Low (feat. T-Pain)", ("T-Pain",)),
        ("Mo Money Mo Problems (feat. Puff Daddy & Mase) - 2014 Remaster", ("Puff Daddy", "Mase")),
        ("Song [with A, B and C]", ("A", "B", "C")),
        ("Featherweight", ()),
    ],
)
def test_featured_artists(title, expected):
    assert featured_artists(title) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Hotel California - Live 1994", "live"),
        ("Getting Better - Take 12", "demo"),
        ("Blinding Lights (Sped Up)", "sped up"),
        ("Last Christmas", "holiday"),
        ("Bohemian Rhapsody - Remastered 2011", ""),
        ("Olive Tree", ""),  # "live" only as a whole word
    ],
)
def test_alt_version(title, expected):
    assert alt_version(title) == expected


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


def test_detect_schema(raw_tracks):
    assert detect_schema(raw_tracks.columns) == "v2"
    assert detect_schema(["id", "artists", "year", "name"]) == "v1"
    with pytest.raises(ValueError):
        detect_schema(["foo"])


def test_clean_tracks_drops_broken_duplicate_and_orphan_rows(raw_tracks, tracks):
    names = set(tracks["name"])
    assert "Broken Tempo" not in names  # tempo 0 removed
    assert "Orphan Song" not in names  # artist never credited anywhere
    assert len(tracks) == len(raw_tracks) - 3  # + the exact duplicate
    song1 = tracks[tracks["name"] == "Song 1"]
    assert len(song1) == 1 and song1["popularity"].iloc[0] > 0  # most popular copy kept


def test_keep_unnamed_artists_when_asked(raw_tracks):
    out = clean_tracks(raw_tracks, require_artist=False)
    orphan = out[out["name"] == "Orphan Song"]
    assert orphan["primary_artist"].tolist() == ["Unknown artist"]


def test_artist_name_recovered_from_artist_key(tracks):
    live = tracks[tracks["name"] == "Bohemian Rhapsody - Live"].iloc[0]
    assert live["primary_artist"] == "Queen"
    remaster = tracks[tracks["name"] == "Bohemian Rhapsody - Remastered 2011"].iloc[0]
    assert live["artist_key"] == remaster["artist_key"]
    assert live["genres"] == ("classic rock", "glam rock", "rock")


def test_clean_tracks_adds_helper_columns(tracks):
    for col in [
        "artist_list",
        "artist_key",
        "primary_artist",
        "artists_display",
        "genres_display",
        "title_norm",
        "artist_norm",
        "version_key",
        "alt_version",
    ]:
        assert col in tracks.columns
    queen = tracks[tracks["primary_artist"] == "Queen"]
    assert queen["version_key"].nunique() == 1  # remaster + live are the same song
    assert set(queen["alt_version"]) == {"", "live"}
    assert tracks.index.equals(tracks.reset_index(drop=True).index)
    feat = tracks[tracks["name"].str.contains("feat.", regex=False)]
    assert not feat.empty and (feat["artist_list"].map(len) == 2).all()


def test_clean_tracks_clips_values(raw_tracks):
    raw = raw_tracks.copy()
    raw.loc[0, "loudness"] = 5.0
    raw.loc[0, "energy"] = 1.7
    out = clean_tracks(raw)
    assert out["loudness"].max() <= 0.0
    assert out["energy"].max() <= 1.0


def test_v1_layout_still_supported():
    raw = pd.DataFrame(
        {
            "id": ["a" * 22, "b" * 22],
            "name": ["One", "Two"],
            "artists": ["['X']", "['Y', 'Z']"],
            "year": [1990, 2001],
            "popularity": [10, 20],
            "duration_ms": [200_000, 180_000],
            "explicit": [0, 1],
            "key": [0, 5],
            "mode": [1, 0],
            **{f: [0.5, 0.6] for f in ["danceability", "energy", "valence", "acousticness"]},
            **{f: [0.1, 0.2] for f in ["instrumentalness", "liveness", "speechiness"]},
            "tempo": [120.0, 90.0],
            "loudness": [-8.0, -6.0],
        }
    )
    out = clean_tracks(standardize(raw))
    assert list(out["artists_display"]) == ["X", "Y, Z"]
    assert (out["genres"].map(len) == 0).all() and out["duration_ms"].notna().all()
