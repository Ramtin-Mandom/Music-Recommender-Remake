import pytest

from music_recommender import TrackRef
from music_recommender.importer import import_playlist


@pytest.fixture(scope="module")
def ids(tracks):
    return {name: tracks.loc[tracks["name"] == name, "id"].iloc[0] for name in ["Song 3", "Song 7", "Stand by Me"]}


def test_spotify_links_and_uris(engine, tracks, ids):
    text = f"""
https://open.spotify.com/track/{ids["Song 3"]}?si=0123456789abcdef
https://open.spotify.com/intl-de/track/{ids["Song 7"]}
spotify:track:{ids["Stand by Me"]}
"""
    imp = engine.import_playlist(text)
    assert [tracks.loc[r, "name"] for r in imp.rows] == ["Song 3", "Song 7", "Stand by Me"]
    assert all(i.method == "spotify id" for i in imp.items)
    assert imp.summary().startswith("Matched 3 of 3")


def test_several_links_on_one_line_and_bare_ids(engine, ids):
    imp = engine.import_playlist(f"{ids['Song 3']}, {ids['Song 7']}")
    assert len(imp.rows) == 2
    one_line = f"https://open.spotify.com/track/{ids['Song 3']} https://open.spotify.com/track/{ids['Song 7']}"
    assert len(engine.import_playlist(one_line).rows) == 2


def test_typed_lines(engine, tracks):
    text = """1. Stand by Me - Ben E. King
Queen - Bohemian Rhapsody
Bohemian Rhapsody by Rockabye Baby!
Song 12 – Artist 12
"""
    imp = engine.import_playlist(text)
    got = [(tracks.loc[r, "name"], tracks.loc[r, "primary_artist"]) for r in imp.rows]
    assert got == [
        ("Stand by Me", "Ben E. King"),
        ("Bohemian Rhapsody - Remastered 2011", "Queen"),  # "Artist - Title" also works
        ("Bohemian Rhapsody", "Rockabye Baby!"),
        ("Song 12", "Artist 12"),
    ]
    assert all(i.method == "title" for i in imp.items)


def test_problems_are_reported_not_raised(engine, ids):
    text = f"""
https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy
spotify:track:0000000000000000000000
zzzz qqqq - nobody at all
spotify:track:{ids["Song 3"]}
https://open.spotify.com/track/{ids["Song 3"]}
"""
    imp = engine.import_playlist(text)
    statuses = [i.status for i in imp.items]
    assert statuses == ["unsupported", "unsupported", "not in catalogue", "no match", "matched", "duplicate"]
    assert len(imp.rows) == 1 and len(imp.unmatched) == 4 and imp.n_entries == 5
    assert "1 not in the catalogue" in imp.summary()
    report = imp.report(engine.tracks)
    assert list(report["status"]) == statuses and report.loc[4, "matched"].startswith("Song 3")


def test_csv_export(engine, tracks, ids):
    csv_text = f"""Track URI,Track Name,Artist Name(s),Album Name
spotify:track:{ids["Song 7"]},Song 7,Whoever,Album
spotify:track:0000000000000000000000,Stand by Me,"Ben E. King, Someone",Album
,Song 12,Artist 12,Album
"""
    imp = engine.import_playlist(csv_text)
    names = [tracks.loc[r, "name"] for r in imp.rows]
    # an unknown id falls back to the title + first artist
    assert names == ["Song 7", "Stand by Me", "Song 12"]
    assert [i.method for i in imp.items] == ["spotify id", "title", "title"]


def test_ids_of_dropped_duplicates_still_resolve(engine, tracks):
    # conftest adds "dup…01", a less popular copy of Song 1 that cleaning removes
    assert "dup0000000000000000001" not in set(tracks["id"])
    row = engine.resolve(TrackRef(track_id="dup0000000000000000001"))
    assert tracks.loc[row, "name"] == "Song 1"
    assert engine.import_playlist("spotify:track:dup0000000000000000001").rows == [row]


def test_link_works_as_a_normal_seed(engine, ids):
    recs = engine.similar_to(f"https://open.spotify.com/track/{ids['Song 3']}?si=x", k=3)
    assert len(recs) == 3


def test_imported_rows_drive_recommendations(engine, ids):
    imp = engine.import_playlist("\n".join(f"spotify:track:{i}" for i in ids.values()))
    recs = engine.recommend_rows(imp.rows, k=10)
    assert len(recs) == 10 and not set(recs["row"]) & set(imp.rows)


def test_empty_input(engine):
    assert import_playlist("  \n# just a comment\n", engine.catalog).rows == []
