"""Import a playlist pasted from Spotify (or typed by hand) as seed tracks.

No Spotify account or API access is needed.  Every catalogue row carries its
Spotify track id, so a pasted link matches exactly.  Accepted input, mixed
freely, one or more items per line:

* Track links, as produced by selecting songs in the Spotify app and pressing
  Ctrl+C / Cmd+C: ``https://open.spotify.com/track/<id>?si=...`` (also
  ``/intl-xx/track/...`` and ``play.spotify.com``)
* Track URIs: ``spotify:track:<id>``, or a bare 22-character track id
* ``Title - Artist``, ``Title – Artist``, ``Title by Artist`` or
  ``Artist - Title`` lines, optionally numbered (``"1. "``)
* A CSV export with a header, e.g. from Exportify: columns such as
  ``Track URI`` / ``Track ID`` and/or ``Track Name`` + ``Artist Name(s)``

Album, artist and playlist *links* cannot be expanded without the Spotify
API.  They are reported as unsupported, with a hint to copy the songs instead.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import pandas as pd

from .catalog import TrackCatalog, TrackNotFoundError, TrackRef

_ID = r"[0-9A-Za-z]{22}"
_TRACK_RE = re.compile(
    rf"(?:https?://(?:open|play)\.spotify\.com/(?:intl-[a-z]{{2}}(?:-[a-z]{{2}})?/)?(?:embed/)?track/|spotify:track:)({_ID})",
    re.IGNORECASE,
)
_OTHER_LINK_RE = re.compile(
    r"(?:https?://(?:open|play)\.spotify\.com/(?:intl-[a-z]{2}(?:-[a-z]{2})?/)?|spotify:)(album|artist|playlist|show|episode)\b",
    re.IGNORECASE,
)
_BARE_ID_RE = re.compile(rf"^{_ID}$")
_NUMBERING_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
_DASHES = (" – ", " — ", " - ")

#: A typed line is only accepted when the title matches at least this well
#: (0.85 = the title starts with the query; see ``TrackCatalog.search``).
MIN_TEXT_MATCH = 0.85

_ID_COLUMNS = ("track uri", "spotify uri", "uri", "track id", "spotify id", "id", "track_id")
_TITLE_COLUMNS = ("track name", "name", "title", "song", "track")
_ARTIST_COLUMNS = ("artist name(s)", "artist names", "artist name", "artists", "artist")


@dataclass(frozen=True)
class ImportItem:
    """One pasted entry and what became of it."""

    text: str
    status: str  # "matched", "not in catalogue", "no match", "unsupported", "duplicate"
    row: int | None = None
    method: str = ""  # "spotify id" or "title"
    note: str = ""


@dataclass
class PlaylistImport:
    """Result of :func:`import_playlist`."""

    items: list[ImportItem] = field(default_factory=list)

    @property
    def rows(self) -> list[int]:
        """Matched catalogue rows, in paste order, without duplicates."""
        return [i.row for i in self.items if i.status == "matched"]

    @property
    def unmatched(self) -> list[ImportItem]:
        return [i for i in self.items if i.status not in ("matched", "duplicate")]

    @property
    def n_entries(self) -> int:
        return sum(1 for i in self.items if i.status != "duplicate")

    def summary(self) -> str:
        n, m = self.n_entries, len(self.rows)
        text = f"Matched {m} of {n} pasted song(s)"
        by_id = sum(1 for i in self.items if i.status == "matched" and i.method == "spotify id")
        if m:
            text += f" ({by_id} by Spotify id, {m - by_id} by title)"
        missing = sum(1 for i in self.items if i.status == "not in catalogue")
        if missing:
            text += f"; {missing} not in the catalogue"
        other = len(self.unmatched) - missing
        if other:
            text += f"; {other} not recognised"
        return text + "."

    def report(self, tracks: pd.DataFrame | None = None) -> pd.DataFrame:
        """One row per entry; with ``tracks``, matched entries show the catalogue song."""
        records = []
        for item in self.items:
            rec = {"input": item.text, "status": item.status, "method": item.method, "note": item.note}
            if tracks is not None and item.row is not None:
                t = tracks.iloc[item.row]
                rec["matched"] = f"{t['name']} — {t['artists_display']} ({t['year']})"
            records.append(rec)
        return pd.DataFrame(records)


# ----------------------------------------------------------------------
def _entries(text: str) -> list[tuple[str, str | None, TrackRef | None]]:
    """Split pasted text into ``(display text, track id or None, TrackRef or None)``.

    Exactly one of id / ref is set for usable entries; both are None for
    unsupported ones.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    csv_entries = _csv_entries(text)
    if csv_entries is not None:
        return csv_entries

    out: list[tuple[str, str | None, TrackRef | None]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ids = _TRACK_RE.findall(line)
        if ids:
            out.extend((f"spotify:track:{i}", i, None) for i in ids)
            continue
        if _OTHER_LINK_RE.search(line):
            out.append((line, None, None))
            continue
        tokens = [t for t in re.split(r"[\s,;]+", line) if t]
        if tokens and all(_BARE_ID_RE.match(t) and not t.isalpha() for t in tokens):
            out.extend((t, t, None) for t in tokens)
            continue
        out.append((line, None, TrackRef(title=_NUMBERING_RE.sub("", line))))
    return out


def _csv_entries(text: str):
    """Entries from a CSV export with a recognisable header, else None."""
    first = text.split("\n", 1)[0]
    if "," not in first and "\t" not in first:
        return None
    try:
        dialect = csv.Sniffer().sniff(first, delimiters=",\t;")
    except csv.Error:
        return None
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        return None
    lower = {name.strip().lower(): name for name in reader.fieldnames if name}

    def pick(options):
        return next((lower[o] for o in options if o in lower), None)

    id_col, title_col, artist_col = pick(_ID_COLUMNS), pick(_TITLE_COLUMNS), pick(_ARTIST_COLUMNS)
    if id_col is None and title_col is None:
        return None
    out = []
    for rec in reader:
        value = (rec.get(id_col) or "").strip() if id_col else ""
        m = _TRACK_RE.search(value)
        track_id = m.group(1) if m else (value if _BARE_ID_RE.match(value) else None)
        title = (rec.get(title_col) or "").strip() if title_col else ""
        artist = (rec.get(artist_col) or "").strip() if artist_col else ""
        artist = re.split(r"\s*[,;]\s*", artist)[0] if artist else ""  # first credited artist
        label = f"{title} - {artist}" if title and artist else (title or value)
        if not label:
            continue
        ref = TrackRef(title=title, artist=artist or None) if title else None
        out.append((label, track_id, ref))
    return out


def _match_text(catalog: TrackCatalog, ref: TrackRef) -> tuple[int | None, float]:
    """Best catalogue row for a typed line, trying title/artist splits both ways."""
    readings = list(ref.readings()) if ref.artist is None else [(ref.title, ref.artist)]
    if ref.artist is None:
        for sep in _DASHES:
            if sep in ref.title:
                left, right = (p.strip() for p in ref.title.split(sep, 1))
                if left and right:
                    readings += [(left, right), (right, left)]  # "Title - Artist" and "Artist - Title"
    # Fast path: an exact title + artist (dictionary lookup), artist readings first.
    for title, artist in sorted(readings, key=lambda r: r[1] is None):
        row = catalog.exact(title, artist)
        if row is not None:
            return row, 1.0
    best_row, best = None, 0.0
    for title, artist in readings:
        hits = catalog.search(title, artist, limit=1)
        if not hits.empty and float(hits.loc[0, "match"]) > best:
            best_row, best = int(hits.loc[0, "row"]), float(hits.loc[0, "match"])
    return best_row, best


def import_playlist(text: str, catalog: TrackCatalog) -> PlaylistImport:
    """Match pasted playlist text against the catalogue."""
    result = PlaylistImport()
    seen: set[int] = set()
    for label, track_id, ref in _entries(text):
        row, method, note = None, "", ""
        if track_id is not None:
            try:
                row, method = catalog.row_for_id(track_id), "spotify id"
            except TrackNotFoundError:
                if ref is None:
                    result.items.append(ImportItem(label, "not in catalogue", note="track id not in the dataset"))
                    continue
        if row is None and ref is not None:
            row, score = _match_text(catalog, ref)
            if row is None or score < MIN_TEXT_MATCH:
                result.items.append(ImportItem(label, "no match", note="no song with that title/artist"))
                continue
            method = "title"
        if row is None:
            result.items.append(
                ImportItem(
                    label, "unsupported", note="album/artist/playlist links need the Spotify API; copy the songs"
                )
            )
            continue
        if row in seen:
            result.items.append(ImportItem(label, "duplicate", row, method))
            continue
        seen.add(row)
        result.items.append(ImportItem(label, "matched", row, method, note))
    return result
