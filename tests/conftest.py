from __future__ import annotations

import pandas as pd
import pytest

from music_recommender import EngineConfig, Recommender, clean_tracks, make_synthetic_tracks


def _with_named_tracks(raw: pd.DataFrame) -> pd.DataFrame:
    """Add a few hand-made rows used by search / dedupe tests."""
    base = raw.iloc[0].to_dict()
    extra = [
        {
            **base,
            "id": "boh0000000000000000001",
            "name": "Bohemian Rhapsody - Remastered 2011",
            "artists": "['Queen']",
            "popularity": 90,
            "year": 1975,
        },
        {
            **base,
            "id": "boh0000000000000000002",
            "name": "Bohemian Rhapsody - Live",
            "artists": "['Queen']",
            "popularity": 30,
            "year": 1979,
            "energy": min(base["energy"] + 0.01, 1.0),
        },
        {
            **base,
            "id": "boh0000000000000000003",
            "name": "Bohemian Rhapsody",
            "artists": "['Rockabye Baby!']",
            "popularity": 20,
            "year": 2009,
        },
        {
            **base,
            "id": "sbm0000000000000000001",
            "name": "Stand by Me",
            "artists": "['Ben E. King']",
            "popularity": 80,
            "year": 1961,
        },
        {
            **base,
            "id": "dup0000000000000000001",
            "name": "Song 1",
            "artists": raw.iloc[1]["artists"],
            "popularity": 0,
            "year": int(raw.iloc[1]["year"]),
        },  # exact duplicate of row 1, less popular
        {**base, "id": "brk0000000000000000001", "name": "Broken Tempo", "artists": "['Nobody']", "tempo": 0.0},
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
