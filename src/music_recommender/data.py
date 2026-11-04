"""Loading and cleaning of the Spotify tracks dataset.

The raw CSV (``data/raw/tracks.csv``, ~170k rows) stores the artist list as a
Python-literal string (``"['Queen', 'David Bowie']"``) and contains many
re-releases of the same song (remasters, live cuts, compilation copies).  This
module turns it into a tidy :class:`pandas.DataFrame` with helper columns used
by the search and re-ranking stages.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AUDIO_FEATURES, DEFAULT_DATA_PATH

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "artists",
    "year",
    "popularity",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    *AUDIO_FEATURES,
)

# Suffixes such as " - Remastered 2011", " - Live at Wembley '86", " (Mono)".
_PAREN_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_text(text: object) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    return _NON_ALNUM_RE.sub(" ", text).strip()


def base_title(title: object) -> str:
    """Normalised title with version information removed.

    ``"Bohemian Rhapsody - Remastered 2011"`` and ``"Bohemian Rhapsody (Live)"``
    both map to ``"bohemian rhapsody"``.
    """
    if not isinstance(title, str):
        return ""
    stripped = _PAREN_RE.sub("", title).split(" - ")[0]
    return normalize_text(stripped) or normalize_text(title)


def parse_artists(raw: object) -> tuple[str, ...]:
    """Parse the dataset's ``"['A', 'B']"`` artist encoding into a tuple."""
    if isinstance(raw, (list, tuple)):
        return tuple(str(a) for a in raw)
    if not isinstance(raw, str) or not raw.strip():
        return ()
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return tuple(str(a).strip() for a in parsed if str(a).strip())
        except (ValueError, SyntaxError):
            raw = raw.strip("[]")
    return tuple(part.strip(" '\"") for part in raw.split(",") if part.strip(" '\""))


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. See data/README.md for how to obtain it.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean_tracks(df: pd.DataFrame, *, min_tempo: float = 30.0) -> pd.DataFrame:
    """Validate, enrich and de-duplicate a raw tracks table.

    Steps
    -----
    1. Drop rows with missing audio features or an empty title / artist.
    2. Drop rows whose tempo is below ``min_tempo`` (tempo 0 means "unknown").
    3. Clip physically-implausible values (loudness > 0 dB, unit features
       outside [0, 1]).
    4. Add helper columns: ``artist_list``, ``primary_artist``,
       ``artists_display``, ``title_norm``, ``artist_norm``, ``version_key``.
    5. Remove exact duplicates (same normalised title *and* artist list),
       keeping the most popular copy.
    """
    df = df.copy()
    df = df.dropna(subset=[*AUDIO_FEATURES, "name", "artists"])
    df = df[df["tempo"] >= min_tempo]

    unit_cols = [c for c in AUDIO_FEATURES if c not in ("tempo", "loudness")]
    df[unit_cols] = df[unit_cols].clip(0.0, 1.0)
    df["loudness"] = df["loudness"].clip(upper=0.0)
    df["year"] = df["year"].astype(int)
    df["popularity"] = df["popularity"].clip(0, 100).astype(int)
    df["explicit"] = df["explicit"].astype(int)
    df["key"] = df["key"].astype(int) % 12
    df["mode"] = df["mode"].astype(int)

    artist_lists = df["artists"].map(parse_artists)
    df["artist_list"] = artist_lists
    df["primary_artist"] = artist_lists.map(lambda a: a[0] if a else "")
    df["artists_display"] = artist_lists.map(", ".join)
    df = df[df["primary_artist"] != ""]

    df["title_norm"] = df["name"].map(normalize_text)
    df["artist_norm"] = df["artists_display"].map(normalize_text)
    df["version_key"] = df["name"].map(base_title) + "|" + df["primary_artist"].map(normalize_text)

    df = df.sort_values("popularity", ascending=False, kind="stable")
    df = df.drop_duplicates(subset=["title_norm", "artist_norm"], keep="first")
    df = df.sort_values(["year", "id"], kind="stable").reset_index(drop=True)
    return df


def load_tracks(path: str | Path = DEFAULT_DATA_PATH, *, min_tempo: float = 30.0) -> pd.DataFrame:
    """Load and clean the dataset in one call."""
    return clean_tracks(load_raw(path), min_tempo=min_tempo)


def make_synthetic_tracks(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Generate a small, *structured* fake catalogue (used by tests and demos).

    Tracks are drawn around a handful of artist "styles" so that tracks by the
    same artist really are closer in feature space — which lets tests check
    that the engine recovers that structure.
    """
    rng = np.random.default_rng(seed)
    n_artists = max(4, n // 15)
    styles = rng.uniform(0.05, 0.95, size=(n_artists, 7))
    tempos = rng.uniform(70, 170, size=n_artists)
    louds = rng.uniform(-20, -4, size=n_artists)
    years = rng.integers(1950, 2021, size=n_artists)

    rows = []
    for i in range(n):
        a = i % n_artists
        s = np.clip(styles[a] + rng.normal(0, 0.05, 7), 0, 1)
        rows.append(
            {
                "id": f"trk{i:05d}",
                "name": f"Song {i}",
                "artists": str([f"Artist {a}"]),
                "year": int(np.clip(years[a] + rng.integers(-3, 4), 1921, 2020)),
                "popularity": int(rng.integers(0, 100)),
                "duration_ms": int(rng.normal(220_000, 40_000)),
                "explicit": int(rng.random() < 0.1),
                "key": int(rng.integers(0, 12)),
                "mode": int(rng.random() < 0.7),
                "danceability": s[0],
                "energy": s[1],
                "valence": s[2],
                "acousticness": s[3],
                "instrumentalness": s[4],
                "liveness": s[5],
                "speechiness": s[6] * 0.3,
                "tempo": float(tempos[a] + rng.normal(0, 4)),
                "loudness": float(min(louds[a] + rng.normal(0, 1), 0)),
            }
        )
    return pd.DataFrame(rows)
