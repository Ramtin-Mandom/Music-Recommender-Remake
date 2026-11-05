from __future__ import annotations

import pandas as pd
import pytest

from music_recommender import EngineConfig, Recommender, clean_tracks, make_synthetic_tracks

QUEEN = {"genres": "['classic rock', 'glam rock', 'rock']", "artist_followers": 45_000_000.0, "artist_popularity": 85.0}


def _with_named_tracks(raw: pd.DataFrame) -> pd.DataFrame:
    """Add a few hand-made rows (v2 layout) used by search / dedupe / version tests."""
    base = raw.iloc[0].to_dict()
    row1 = raw.iloc[1].to_dict()
    extra = [
        {
            **base,
            **QUEEN,
            "track_id": "boh0000000000000000001",
            "name": "Bohemian Rhapsody - Remastered 2011",
            "track_artists": "Queen",
            "popularity": 90.0,
            "album_release_date": "1975-10-31",
        },
        {
            # no credited artist: the name must be recovered from the artist key
            **base,
            **QUEEN,
            "track_id": "boh0000000000000000002",
            "name": "Bohemian Rhapsody - Live",
            "track_artists": None,
            "popularity": 30.0,
            "album_release_date": "1979",
            "energy": min(base["energy"] + 0.01, 1.0),
        },
        {
            **base,
            "track_id": "boh0000000000000000003",
            "name": "Bohemian Rhapsody",
            "track_artists": "Rockabye Baby!",
            "genres": "['lullaby']",
            "artist_followers": 70_000.0,
            "artist_popularity": 40.0,
            "popularity": 20.0,
            "album_release_date": "2009-01-01",
        },
        {
            **base,
            "track_id": "sbm0000000000000000001",
            "name": "Stand by Me",
            "track_artists": "Ben E. King",
            "genres": "['soul', 'rhythm and blues']",
            "artist_followers": 1_200_000.0,
            "artist_popularity": 70.0,
            "popularity": 80.0,
            "album_release_date": "1961-01-01",
        },
        {**row1, "track_id": "dup0000000000000000001", "popularity": 0.0},  # duplicate of row 1, less popular
        {**base, "track_id": "brk0000000000000000001", "name": "Broken Tempo", "tempo": 0.0},
        {
            # an artist that is never credited anywhere: dropped by default
            **base,
            "track_id": "orp0000000000000000001",
            "name": "Orphan Song",
            "track_artists": None,
            "genres": "['mystery']",
            "artist_followers": 123.0,
            "artist_popularity": 1.0,
        },
    ]
    return pd.concat([raw, pd.DataFrame(extra)], ignore_index=True)


@pytest.fixture(scope="session")
def raw_tracks() -> pd.DataFrame:
    return _with_named_tracks(make_synthetic_tracks(n=600, seed=0))


@pytest.fixture(scope="session")
def tracks(raw_tracks) -> pd.DataFrame:
    return clean_tracks(raw_tracks)


@pytest.fixture(scope="session")
def engine(tracks) -> Recommender:
    return Recommender(EngineConfig(candidate_pool=100)).fit(tracks)
