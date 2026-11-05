"""Track lookup: turning what a user types into rows of the catalogue.

The original project used a substring match and silently took the *first*
hit, which often picked an obscure cover or a karaoke version.  Here every
candidate gets a match-quality score and ties are broken by popularity, and
callers get the full ranked list so a UI can let the user disambiguate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches

import numpy as np
import pandas as pd

from .data import normalize_text

#: How many of the most popular tracks' titles typo matching looks at.
FUZZY_LIMIT = 60_000
_TRACK_LINK_RE = re.compile(
    r"https?://(?:open|play)\.spotify\.com/(?:intl-[a-z]{2}(?:-[a-z]{2})?/)?(?:embed/)?track/([0-9A-Za-z]{22})",
    re.IGNORECASE,
)


class TrackNotFoundError(LookupError):
    """Raised when a query does not match any track."""

    def __init__(self, query: str, artist: str | None = None, suggestions: list[str] | None = None):
        self.query = query
        self.artist = artist
        self.suggestions = suggestions or []
        msg = f"No track found for '{query}'"
        if artist:
            msg += f" by '{artist}'"
        if self.suggestions:
            msg += ". Did you mean: " + "; ".join(self.suggestions) + "?"
        super().__init__(msg)


@dataclass(frozen=True)
class TrackRef:
    """A user-supplied reference to a track: a Spotify id, or a title (+ artist)."""

    title: str | None = None
    artist: str | None = None
    track_id: str | None = None
    weight: float = 1.0

    @classmethod
    def parse(cls, value: str | tuple | TrackRef | dict) -> TrackRef:
        """Accept the many shapes callers like to use.

        * ``TrackRef`` – returned unchanged
        * ``"spotify:track:<id>"`` or a bare 22-char Spotify id
        * ``"Title"`` – may also be ``"Title by Artist"`` / ``"Title - Artist"``;
          :meth:`TrackCatalog.resolve` tries both readings
        * ``("Title", "Artist")`` / ``("Title", None)``
        * ``{"title": ..., "artist": ..., "track_id": ..., "weight": ...}``
        """
        if isinstance(value, TrackRef):
            return value
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, tuple):
            title, artist = (value + (None,))[:2]
            return cls(title=title, artist=artist or None)
        text = str(value).strip()
        if text.startswith("spotify:track:"):
            return cls(track_id=text.split(":")[-1])
        link = _TRACK_LINK_RE.match(text)
        if link:  # https://open.spotify.com/track/<id>?si=...
            return cls(track_id=link.group(1))
        if len(text) == 22 and text.isalnum() and not text.isalpha() and " " not in text:
            return cls(track_id=text)
        return cls(title=text)

    def readings(self) -> list[tuple[str, str | None]]:
        """Possible (title, artist) interpretations, most literal first."""
        assert self.title is not None
        out: list[tuple[str, str | None]] = [(self.title, self.artist)]
        if self.artist is None:
            for sep in (" by ", " - "):
                if sep in self.title:
                    title, artist = self.title.rsplit(sep, 1)
                    if title.strip() and artist.strip():
                        out.append((title.strip(), artist.strip()))
        return out


class TrackCatalog:
    """Search index over the cleaned tracks table."""

    def __init__(self, tracks: pd.DataFrame):
        self.tracks = tracks
        self._title = tracks["title_norm"].to_numpy(dtype=object)
        self._title_s = pd.Series(self._title)
        self._base = tracks["version_key"].str.split("|").str[0].to_numpy(dtype=object)
        self._artist = tracks["artist_norm"].to_numpy(dtype=object)
        self._popularity = tracks["popularity"].to_numpy(dtype=float)
        id_to_row = dict(zip(tracks["id"], range(len(tracks)), strict=True))
        if "alias_ids" in tracks:  # ids of duplicate copies dropped during cleaning
            for row, aliases in enumerate(tracks["alias_ids"]):
                for alias in aliases:
                    id_to_row.setdefault(alias, row)
        self._id_to_row = id_to_row
        self._unique_titles: list[str] | None = None
        self._exact_index: tuple | None = None

    def __len__(self) -> int:
        return len(self.tracks)

    # ------------------------------------------------------------------
    def row_for_id(self, track_id: str) -> int:
        try:
            return int(self._id_to_row[track_id])
        except KeyError:
            raise TrackNotFoundError(track_id) from None

    def search(self, query: str, artist: str | None = None, limit: int = 10) -> pd.DataFrame:
        """Rank catalogue rows by how well they match ``query`` (and ``artist``).

        Returns a frame with a ``row`` column (position in :attr:`tracks`) and a
        ``match`` score in (0, 1].
        """
        q = normalize_text(query)
        if not q:
            return self._empty_result()

        score = np.zeros(len(self._title))
        # Only one full scan; the finer match levels are computed on the hits.
        hits = np.flatnonzero(self._title_s.str.contains(q, regex=False).to_numpy())
        sub = self._title_s.iloc[hits]
        level = np.full(len(hits), 0.7)
        starts = sub.str.startswith(q).to_numpy()
        level[starts] = 0.85
        # Whole-word prefix ("bohemian rhapsody operatic section ...").
        level[starts & sub.str.slice(0, len(q) + 1).isin([q, q + " "]).to_numpy()] = 0.95
        score[hits] = level
        # Exact title, ignoring version suffixes: "Bohemian Rhapsody" and
        # "Bohemian Rhapsody - Remastered 2011" both count as exact matches, and
        # popularity then picks the canonical release.
        score[(self._title == q) | (self._base == q)] = 1.0

        tokens = q.split()
        if len(tokens) > 1:
            # Every word present, in any order: narrow the candidates word by word.
            cand = np.flatnonzero(self._title_s.str.contains(max(tokens, key=len), regex=False).to_numpy())
            for tok in tokens:
                cand = cand[self._title_s.iloc[cand].str.contains(tok, regex=False).to_numpy()]
            cand = cand[score[cand] == 0]
            score[cand] = 0.6

        if score.max() < 0.7:
            # Typo fallback ("bohemian rapsody"); also when only the loose
            # every-word match fired ("rapsody" is inside "Bohemian Trapsody").
            close = get_close_matches(q, self._fuzzy_titles(), n=5, cutoff=0.75)
            for rank, title in enumerate(close):
                hit = self._title == title
                score[hit] = np.maximum(score[hit], 0.65 - 0.01 * rank)

        if artist:
            a = normalize_text(artist)
            artist_ok = pd.Series(self._artist).str.contains(a, regex=False).to_numpy()
            score = np.where(artist_ok, score, 0.0)

        hits = np.flatnonzero(score > 0)
        if hits.size == 0:
            return self._empty_result()
        order = np.lexsort((-self._popularity[hits], -score[hits]))[:limit]
        rows = hits[order]
        out = self.tracks.iloc[rows][["id", "name", "artists_display", "year", "popularity"]].copy()
        out.insert(0, "row", rows)
        out["match"] = score[rows].round(3)
        return out.reset_index(drop=True)

    def resolve(self, ref: TrackRef | str | tuple) -> int:
        """Return the single best catalogue row for a reference."""
        ref = TrackRef.parse(ref)
        if ref.track_id:
            return self.row_for_id(ref.track_id)
        if not ref.title:
            raise ValueError("A track reference needs a title or a track_id.")
        best_row, best_score = None, 0.0
        for title, artist in ref.readings():
            hits = self.search(title, artist, limit=1)
            # Strictly greater: on ties the literal reading wins ("Stand by Me").
            if not hits.empty and float(hits.loc[0, "match"]) > best_score:
                best_row, best_score = int(hits.loc[0, "row"]), float(hits.loc[0, "match"])
        if best_row is None:
            raise TrackNotFoundError(ref.title, ref.artist, self.suggest(ref.title))
        return best_row

    def exact(self, title: str, artist: str | None = None) -> int | None:
        """Most popular row whose title (with or without version suffix) is
        exactly ``title``, optionally by an artist containing ``artist``.
        A dictionary lookup, so much faster than :meth:`search`."""
        if getattr(self, "_exact_index", None) is None:
            n = len(self._title)
            codes, uniques = pd.factorize(np.concatenate([self._title, self._base]))
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
            self._exact_index = ({k: i for i, k in enumerate(uniques)}, order % n, bounds)
        lookup, rows_sorted, bounds = self._exact_index
        i = lookup.get(normalize_text(title))
        if i is None:
            return None
        rows = np.unique(rows_sorted[bounds[i] : bounds[i + 1]])
        if artist:
            a = normalize_text(artist)
            rows = rows[[a in self._artist[r] for r in rows]]
            if rows.size == 0:
                return None
        return int(rows[np.argmax(self._popularity[rows])])

    def suggest(self, query: str, n: int = 3) -> list[str]:
        return get_close_matches(normalize_text(query), self._fuzzy_titles(), n=n, cutoff=0.6)

    def _fuzzy_titles(self) -> list[str]:
        """Distinct titles for typo matching, most popular first and capped
        (difflib is linear in the number of titles)."""
        if self._unique_titles is None:
            order = np.argsort(-self._popularity, kind="stable")[:FUZZY_LIMIT]
            self._unique_titles = list(dict.fromkeys(self._title[order]))
        return self._unique_titles

    def _empty_result(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["row", "id", "name", "artists_display", "year", "popularity", "match"])
