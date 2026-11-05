"""Loading and cleaning of the Spotify tracks dataset.

Two raw layouts are understood and mapped onto one canonical table:

* **v2** (default, ``data/raw/tracks.csv``, ~900k rows): one row per track with
  the *artist's* Spotify genres, follower count and popularity, but the artist
  *name* (``track_artists``) is filled in for only ~11% of rows.
* **v1** (``data/raw/tracks_old.csv``, ~170k rows, the CMPT 310 dataset):
  artist lists as Python literals, a ``year`` column, no genres.

Recovering artists in v2
------------------------
Every track carries its artist's ``genres`` list, ``artist_followers`` and
``artist_popularity``.  Together they identify the artist almost uniquely
(follower counts are exact integers), so they form an ``artist_key``.  Tracks
sharing a key share an artist, and the name is propagated from whichever of
that artist's tracks has ``track_artists`` filled in.  That recovers a name for
~75% of the catalogue; tracks whose artist is never named are dropped by
default (a recommendation without an artist is not useful to a listener).

Featured artists are parsed from titles (``"Low (feat. T-Pain)"``) and added
to ``artist_list``; the evaluation uses them as a collaboration graph.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AUDIO_FEATURES, DEFAULT_DATA_PATH

#: Columns every canonical (standardised) table has.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "artist_list",
    "artist_key",
    "year",
    "popularity",
    "artist_popularity",
    "genres",
    "explicit",
    "key",
    "mode",
    "time_signature",
    "duration_ms",
    *AUDIO_FEATURES,
)

V1_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "artists",
    "year",
    "popularity",
    "explicit",
    "key",
    "mode",
    *AUDIO_FEATURES,
)
V2_COLUMNS: tuple[str, ...] = (
    "track_id",
    "name",
    "track_artists",
    "genres",
    "artist_followers",
    "artist_popularity",
    "album_release_date",
    "popularity",
    "explicit",
    "key",
    "mode",
    "time_signature",
    *AUDIO_FEATURES,
)
#: Optional v2 columns that are read when present.
V2_OPTIONAL: tuple[str, ...] = ("duration_ms", "album_name")

# Suffixes such as " - Remastered 2011", " - Live at Wembley '86", " (Mono)".
_PAREN_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_FEAT_RE = re.compile(r"[\(\[](?:feat\.?|ft\.?|featuring|with)\s+([^\)\]]+)[\)\]]", re.IGNORECASE)
_FEAT_SPLIT_RE = re.compile(r",\s*|\s+&\s+|\s+and\s+|\s+x\s+", re.IGNORECASE)
_TAG_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")

#: Alternate-version markers found in titles, checked in order.  A track
#: carrying one of these is ranked down unless a seed is the same kind of
#: version (see ``EngineConfig.version_penalty``).
ALT_VERSION_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("karaoke", re.compile(r"\bkaraoke\b|\bin the style of\b", re.I)),
    ("sped up", re.compile(r"\bsped[\s-]?up\b|\bnightcore\b", re.I)),
    ("slowed", re.compile(r"\bslowed\b|\breverb\b", re.I)),
    ("8d", re.compile(r"\b8d\b", re.I)),
    ("live", re.compile(r"\blive\b|\ben vivo\b|\bao vivo\b|\bunplugged\b", re.I)),
    ("demo", re.compile(r"\bdemo\b|\bearly version\b|\bouttake\b|\btake \d+\b|\brehearsal\b", re.I)),
    ("instrumental", re.compile(r"\binstrumental\b", re.I)),
    ("acoustic", re.compile(r"\bacoustic( version)?\b", re.I)),
    # Not a version, but just as jarring in a July playlist; same treatment.
    ("holiday", re.compile(r"\bchristmas\b|\bxmas\b|\bsanta\b|\bjingle bells?\b|\bnavidad\b|\bnoel\b", re.I)),
)


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------
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


def alt_version(title: object) -> str:
    """Kind of alternate version a title denotes (``"live"``, ``"demo"`` ...) or ``""``."""
    if not isinstance(title, str):
        return ""
    for label, pattern in ALT_VERSION_PATTERNS:
        if pattern.search(title):
            return label
    return ""


def featured_artists(title: object) -> tuple[str, ...]:
    """Artists credited in the title: ``"Low (feat. T-Pain)"`` -> ``("T-Pain",)``."""
    if not isinstance(title, str):
        return ()
    m = _FEAT_RE.search(title)
    if not m:
        return ()
    return tuple(p.strip() for p in _FEAT_SPLIT_RE.split(m.group(1)) if p.strip())


def parse_artists(raw: object) -> tuple[str, ...]:
    """Parse the v1 ``"['A', 'B']"`` artist encoding into a tuple."""
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


def parse_genres(raw: object) -> tuple[str, ...]:
    """Parse ``"['indie pop', 'la indie']"`` into ``("indie pop", "la indie")``."""
    if isinstance(raw, (list, tuple)):
        return tuple(str(g) for g in raw)
    if not isinstance(raw, str):
        return ()
    return tuple(a or b for a, b in _TAG_RE.findall(raw))


# ----------------------------------------------------------------------
# Loading and schema mapping
# ----------------------------------------------------------------------
def detect_schema(columns) -> str:
    cols = set(columns)
    if {"track_id", "genres", "track_artists"} <= cols:
        return "v2"
    if {"id", "artists", "year"} <= cols:
        return "v1"
    raise ValueError(
        "Unrecognised dataset layout: expected the v2 columns (track_id, genres, track_artists, ...) "
        "or the v1 columns (id, artists, year, ...). See data/README.md."
    )


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read a raw CSV (either layout) and return the *canonical* table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. See data/README.md for how to obtain it.")
    header = pd.read_csv(path, nrows=0).columns
    schema = detect_schema(header)
    required = V2_COLUMNS if schema == "v2" else V1_COLUMNS
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if schema == "v2":
        use = [*V2_COLUMNS, *(c for c in V2_OPTIONAL if c in header)]
        return standardize_v2(pd.read_csv(path, usecols=use, low_memory=False))
    return standardize_v1(pd.read_csv(path))


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Map an in-memory raw table of either layout to the canonical layout."""
    return standardize_v2(df) if detect_schema(df.columns) == "v2" else standardize_v1(df)


def standardize_v1(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "name": df["name"],
            "artist_list": df["artists"].map(parse_artists),
            "year": pd.to_numeric(df["year"], errors="coerce"),
            "popularity": df["popularity"],
            "genres": [()] * len(df),
            "explicit": df["explicit"],
            "key": df["key"],
            "mode": df["mode"],
            "time_signature": df["time_signature"] if "time_signature" in df else 4,
            "duration_ms": df["duration_ms"] if "duration_ms" in df else np.nan,
            **{f: df[f] for f in AUDIO_FEATURES},
        }
    )
    out["artist_key"] = out["artist_list"].map(lambda a: normalize_text(a[0]) if a else "")
    out["artist_popularity"] = np.nan
    return out[list(CANONICAL_COLUMNS)]


def standardize_v2(df: pd.DataFrame) -> pd.DataFrame:
    followers = pd.to_numeric(df["artist_followers"], errors="coerce")
    art_pop = pd.to_numeric(df["artist_popularity"], errors="coerce")
    genres_raw = df["genres"].fillna("").astype(str)
    has_key = followers.notna() & art_pop.notna()
    artist_key = np.where(
        has_key,
        genres_raw + "|" + followers.fillna(-1).astype("int64").astype(str) + "|" + art_pop.fillna(-1).astype(str),
        "",
    )
    out = pd.DataFrame(
        {
            "id": df["track_id"].astype(str),
            "name": df["name"],
            "artist_key": artist_key,
            "year": pd.to_numeric(df["album_release_date"].astype(str).str[:4], errors="coerce"),
            "popularity": df["popularity"],
            "artist_popularity": art_pop,
            "genres": genres_raw.map(parse_genres),
            "explicit": df["explicit"].map({True: 1, False: 0, "True": 1, "False": 0}),
            "key": df["key"],
            "mode": df["mode"],
            "time_signature": df["time_signature"],
            "duration_ms": pd.to_numeric(df["duration_ms"], errors="coerce") if "duration_ms" in df else np.nan,
            **{f: df[f] for f in AUDIO_FEATURES},
        }
    )
    # Name each artist key after its most frequent credited name.
    named = pd.DataFrame({"key": artist_key, "artist": df["track_artists"]})
    named = named[(named["key"] != "") & named["artist"].notna()]
    names = named.groupby("key")["artist"].agg(lambda s: s.value_counts().index[0])
    # Rows whose key could not be built but which do carry a name get a name-based key.
    primary = pd.Series(artist_key, index=df.index).map(names)
    fallback = df["track_artists"].where(primary.isna())
    primary = primary.fillna(fallback)
    no_key = out["artist_key"] == ""
    out.loc[no_key, "artist_key"] = "name:" + fallback[no_key].map(normalize_text).fillna("")
    out.loc[out["artist_key"] == "name:", "artist_key"] = ""
    out["artist_list"] = [
        ((p,) + tuple(f for f in featured_artists(t) if f.casefold() != p.casefold())) if isinstance(p, str) else ()
        for p, t in zip(primary, out["name"], strict=True)
    ]
    return out[list(CANONICAL_COLUMNS)]


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------
def clean_tracks(df: pd.DataFrame, *, min_tempo: float = 30.0, require_artist: bool = True) -> pd.DataFrame:
    """Validate, enrich and de-duplicate a tracks table.

    Accepts a raw table of either layout or an already canonical one.

    Steps
    -----
    1. Drop rows with missing audio features, title or year (< 1900 = unknown).
    2. Drop rows whose tempo is below ``min_tempo`` (tempo 0 means "unknown").
    3. Drop rows without a known artist (unless ``require_artist=False``, in
       which case they are shown as "Unknown artist").
    4. Clip physically-implausible values.
    5. Add helper columns: ``primary_artist``, ``artists_display``,
       ``title_norm``, ``artist_norm``, ``version_key``, ``alt_version``,
       ``genres_display``.
    6. Remove duplicates (same normalised title and artists), keeping the most
       popular copy.
    """
    if "artist_list" not in df.columns:
        df = standardize(df)
    df = df.copy()
    df = df.dropna(subset=[*AUDIO_FEATURES, "name", "year", "key", "mode"])
    df = df[(df["tempo"] >= min_tempo) & (df["year"] >= 1900)]

    has_artist = df["artist_list"].map(len) > 0
    if require_artist:
        df = df[has_artist]
    else:
        df.loc[~has_artist, "artist_list"] = pd.Series(
            [("Unknown artist",)] * int((~has_artist).sum()), index=df.index[~has_artist], dtype=object
        )
        df.loc[df["artist_key"] == "", "artist_key"] = "row:" + df["id"]

    unit_cols = [c for c in AUDIO_FEATURES if c not in ("tempo", "loudness")]
    df[unit_cols] = df[unit_cols].clip(0.0, 1.0)
    df["loudness"] = df["loudness"].clip(upper=0.0)
    df["year"] = df["year"].astype(int)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).clip(0, 100).astype(int)
    df["artist_popularity"] = pd.to_numeric(df["artist_popularity"], errors="coerce")
    df["explicit"] = pd.to_numeric(df["explicit"], errors="coerce").fillna(0).astype(int)
    df["key"] = df["key"].astype(int) % 12
    df["mode"] = df["mode"].astype(int)
    df["time_signature"] = pd.to_numeric(df["time_signature"], errors="coerce").fillna(4).astype(int)
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
    df["genres"] = df["genres"].map(lambda g: tuple(g) if isinstance(g, (list, tuple)) else parse_genres(g))

    df["primary_artist"] = df["artist_list"].map(lambda a: a[0])
    df["artists_display"] = df["artist_list"].map(", ".join)
    df["genres_display"] = df["genres"].map(", ".join)
    df["title_norm"] = df["name"].map(normalize_text)
    df["artist_norm"] = df["artists_display"].map(normalize_text)
    df["version_key"] = df["name"].map(base_title) + "|" + df["primary_artist"].map(normalize_text)
    df["alt_version"] = df["name"].map(alt_version)

    df = df.sort_values("popularity", ascending=False, kind="stable")
    # Remember the Spotify ids of the copies we drop, so a link to any copy
    # (a compilation, a re-release) still finds the song.
    dup = df.duplicated(subset=["title_norm", "artist_norm"], keep="first")
    first_id = df.groupby(["title_norm", "artist_norm"], sort=False)["id"].transform("first")
    aliases = df.loc[dup, "id"].groupby(first_id[dup]).agg(tuple)
    df = df[~dup].copy()
    df["alias_ids"] = df["id"].map(aliases)
    df["alias_ids"] = df["alias_ids"].map(lambda a: a if isinstance(a, tuple) else ())
    df = df.sort_values(["year", "id"], kind="stable").reset_index(drop=True)
    return df


def load_tracks(
    path: str | Path = DEFAULT_DATA_PATH, *, min_tempo: float = 30.0, require_artist: bool = True
) -> pd.DataFrame:
    """Load and clean the dataset in one call."""
    return clean_tracks(load_raw(path), min_tempo=min_tempo, require_artist=require_artist)


# ----------------------------------------------------------------------
# Synthetic data (tests and demos)
# ----------------------------------------------------------------------
#: Genre "families" for the synthetic catalogue; artists in one family share
#: tags and a broadly similar sound.
_SYNTH_FAMILIES = (
    ("rock", "classic rock", "album rock"),
    ("pop", "dance pop", "electropop"),
    ("hip hop", "rap", "trap"),
    ("classical", "baroque", "early music"),
    ("jazz", "bebop", "cool jazz"),
    ("country", "outlaw country", "country rock"),
)


def make_synthetic_tracks(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Generate a small, *structured* fake catalogue in the v2 raw layout.

    Artists belong to genre families; tracks are drawn around their artist's
    "style", which is drawn around the family's style.  About 1 in 12 titles
    features another artist of the same family, giving the evaluation a
    collaboration graph.  Roughly 1 in 10 artists has no genres, and about
    half the rows leave ``track_artists`` empty (as in the real data).
    """
    rng = np.random.default_rng(seed)
    n_artists = max(6, n // 15)
    n_fam = len(_SYNTH_FAMILIES)
    fam_styles = rng.uniform(0.1, 0.9, size=(n_fam, 7))
    fam_tempo = rng.uniform(80, 160, size=n_fam)
    fam_loud = rng.uniform(-18, -5, size=n_fam)
    family = np.arange(n_artists) % n_fam
    styles = np.clip(fam_styles[family] + rng.normal(0, 0.08, (n_artists, 7)), 0.02, 0.98)
    tempos = fam_tempo[family] + rng.normal(0, 8, n_artists)
    louds = fam_loud[family] + rng.normal(0, 1.5, n_artists)
    years = rng.integers(1960, 2024, size=n_artists)
    followers = rng.integers(1_000, 5_000_000, size=n_artists)
    art_pop = rng.integers(10, 90, size=n_artists)
    genres = []
    for a in range(n_artists):
        tags = list(_SYNTH_FAMILIES[family[a]])
        genres.append([] if a % 10 == 9 else tags[: 1 + a % 3])

    rows = []
    for i in range(n):
        a = i % n_artists
        s = np.clip(styles[a] + rng.normal(0, 0.05, 7), 0, 1)
        name = f"Song {i}"
        if i % 12 == 5:
            same_family = [b for b in range(n_artists) if family[b] == family[a] and b != a]
            if same_family:
                name += f" (feat. Artist {int(rng.choice(same_family))})"
        rows.append(
            {
                "track_id": f"trk{i:019d}",
                "name": name,
                "track_artists": f"Artist {a}" if (i // n_artists) % 2 == 0 else None,
                "genres": str(genres[a]),
                "artist_followers": float(followers[a]),
                "artist_popularity": float(art_pop[a]),
                "album_release_date": f"{int(np.clip(years[a] + rng.integers(-3, 4), 1950, 2024))}-01-01",
                "popularity": float(rng.integers(0, 100)),
                "explicit": bool(rng.random() < 0.1),
                "key": float(rng.integers(0, 12)),
                "mode": float(rng.random() < 0.7),
                "time_signature": float(3 if rng.random() < 0.1 else 4),
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
