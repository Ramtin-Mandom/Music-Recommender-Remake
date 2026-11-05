"""Headless tests of the PyQt6 desktop app (Qt "offscreen" platform, synthetic data)."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from music_recommender import (  # noqa: E402
    LEGACY_CONFIG,
    Recommender,
    gui,  # noqa: E402
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def popups(monkeypatch):
    """Record message boxes instead of blocking on them."""
    shown: list[tuple[str, str]] = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(gui.QMessageBox, kind, lambda *a, kind=kind: shown.append((kind, a[1])))
    return shown


@pytest.fixture()
def win(qapp, engine, popups):
    w = gui.MainWindow(engine=engine)
    yield w
    w.close()


def wait_until(qapp, cond, timeout=20.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError("condition not met")
        qapp.processEvents()
        time.sleep(0.01)


def search(tab, title: str) -> int:
    tab.search.title.setText(title)
    tab.search.refresh()
    row = tab.search.selected_row()
    assert row is not None
    return row


def test_window_ready(win):
    assert win.tabs.isEnabled()
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
        "Song radio",
        "Playlist builder",
        "Mood explorer",
        "Explain",
    ]
    assert "Ready" in win.statusBar().currentMessage()


def test_search_lists_ranked_matches(win):
    win.radio.search.title.setText("bohemian")
    win.radio.search.refresh()
    first = win.radio.search.matches.item(0).text()
    assert first.startswith("Bohemian Rhapsody - Remastered 2011")  # most popular exact match
    win.radio.search.title.setText("Bohemian Rhapsody by Rockabye Baby!")  # "Title by Artist" in one box
    win.radio.search.refresh()
    assert "Rockabye Baby!" in win.radio.search.matches.item(0).text()
    win.radio.search.title.setText("zzzz qqqq")
    win.radio.search.refresh()
    assert win.radio.search.matches.count() == 0
    assert "No match" in win.radio.search.status.text()


def test_song_radio(win):
    seed = search(win.radio, "Song 10")
    win.radio._go()
    model = win.radio.results.model
    assert model.rowCount() == win.settings.k.value() == 15
    assert seed not in set(model.frame()["row"])
    assert "Similar to" in win.radio.results.caption.text()
    assert win.radio.results.export_btn.isEnabled()


def test_settings_change_refreshes_results(win):
    search(win.radio, "Song 10")
    win.radio._go()
    win.tabs.setCurrentWidget(win.radio)
    win.settings.k.setValue(5)
    win.refresh_visible()  # what the debounced `changed` signal triggers
    assert win.radio.results.model.rowCount() == 5
    win.settings.year_min.setValue(1990)
    win.refresh_visible()
    assert (win.radio.results.model.frame()["year"] >= 1990).all()
    win.settings.reset()
    assert win.settings.k.value() == 15 and win.settings.filters().year_min is None


def test_playlist_builder(win, popups):
    tab = win.playlist
    tab.run()
    assert popups and popups[-1][0] == "information"  # no liked songs yet

    search(tab, "Song 1")
    tab.weight.setValue(2.0)
    tab.add(liked=True)
    search(tab, "Song 2")
    tab.add(liked=True)
    tab.add(liked=True)  # duplicates ignored
    disliked = search(tab, "Song 3")
    tab.add(liked=False)
    assert tab.liked.count() == 2 and tab.disliked.count() == 1
    assert "[2]" in tab.liked.item(0).text()

    tab.run()
    frame = tab.results.model.frame()
    assert len(frame) == 15 and disliked not in set(frame["row"])
    assert "2 liked and 1 disliked" in tab.results.caption.text()

    tab.liked.item(0).setSelected(True)
    tab._remove()
    assert tab.liked.count() == 1
    tab._clear()
    assert tab.liked.count() == 0 and tab.disliked.count() == 0


def test_mood_explorer(win, popups):
    tab = win.mood
    frame_targets = tab.targets()
    assert set(frame_targets) == {"energy", "danceability", "valence"}
    tab.run()
    assert tab.results.model.rowCount() == 15
    for cb, *_ in tab.controls.values():
        cb.setChecked(False)
    tab.run()
    assert popups[-1][0] == "information"


def test_explain_from_double_click(win):
    seed = search(win.radio, "Song 10")
    win.radio._go()
    rec_row = int(win.radio.results.model.frame().loc[0, "row"])
    win.radio.results.activated.emit(rec_row)
    assert win.tabs.currentWidget() is win.explain_tab
    assert win.explain_tab.features.rowCount() == len(win.engine.pipeline.groups)
    attrs = win.explain_tab.attrs_model.frame()
    assert len(attrs) == 2 and attrs.iloc[0]["name"] == win.engine.tracks.loc[seed, "name"]


def test_compare_with_legacy(win, qapp, engine):
    search(win.radio, "Song 10")
    win.radio._go()
    win.settings.compare.setChecked(True)  # fits the legacy model in a background thread
    wait_until(qapp, lambda: win.legacy is not None)
    wait_until(qapp, lambda: win.radio.results.legacy_model.rowCount() > 0)
    assert win.radio.results.legacy_box.isVisibleTo(win.radio.results)
    assert win.legacy.config == LEGACY_CONFIG
    win.settings.compare.setChecked(False)
    win.refresh_visible()
    assert not win.radio.results.legacy_box.isVisibleTo(win.radio.results)


def test_background_loading(qapp, engine, popups):
    w = gui.MainWindow(loader=lambda: engine)
    assert not w.tabs.isEnabled()
    wait_until(qapp, lambda: w.engine is not None)
    assert w.tabs.isEnabled()
    w.close()


def test_load_failure_is_reported(qapp, popups):
    def boom() -> Recommender:
        raise FileNotFoundError("tracks.csv missing")

    w = gui.MainWindow(loader=boom)
    wait_until(qapp, lambda: any(kind == "critical" for kind, _ in popups))
    assert w.engine is None
    w.close()


def test_dataframe_model_formats(engine):
    recs = engine.similar_to("Song 10", k=3)
    model = gui.DataFrameModel(recs, columns=gui.RESULT_COLS)
    assert model.rowCount() == 3 and model.columnCount() == len(gui.RESULT_COLS)
    sim_col = gui.RESULT_COLS.index("similarity")
    assert model.data(model.index(0, sim_col)).endswith("%")
    assert model.data(model.index(0, 0), gui.ROW_ROLE) == int(recs.loc[0, "row"])


def test_paste_playlist_import(win, popups, tracks):
    tab = win.playlist
    tab._clear()
    song3 = tracks.loc[tracks["name"] == "Song 3", "id"].iloc[0]
    lines = [
        f"https://open.spotify.com/track/{song3}?si=x",
        "Stand by Me - Ben E. King",
        "spotify:track:0000000000000000000000",
    ]
    text = "\n".join(lines)
    tab.import_text(text)
    assert tab.liked.count() == 2
    assert popups[-1] == ("information", "Playlist imported")
    assert tab.last_import.summary().startswith("Matched 2 of 3")
    assert [i.status for i in tab.last_import.unmatched] == ["not in catalogue"]
    assert "Matched 2 of 3" in win.statusBar().currentMessage()
    tab.import_text("nothing that exists here - nobody")
    assert popups[-1] == ("warning", "Nothing imported") and tab.liked.count() == 2
    tab.run()
    assert tab.results.model.rowCount() == 15
    dlg = gui.PasteDialog(win)  # builds without errors
    assert dlg.text.toPlainText() == ""


def test_light_green_cells_use_black_text(engine):
    recs = engine.similar_to("Song 10", k=3)
    model = gui.DataFrameModel(recs, columns=gui.RESULT_COLS)
    idx = model.index(0, gui.RESULT_COLS.index("similarity"))
    assert model.data(idx, gui.Qt.ItemDataRole.ForegroundRole) == gui.QColor("black")
    assert "QHeaderView::section { background: #d5f5e3; color: black;" in gui.STYLE
