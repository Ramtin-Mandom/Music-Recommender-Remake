"""PyQt6 desktop application for testing the recommendation engine.

Run with ``music-rec-gui`` or ``python -m music_recommender.gui``
(requires the ``desktop`` extra: ``pip install -e ".[desktop]"``).

The window has four tabs that share one results view per tab:

* **Song radio** – search a track (ranked matches as you type) and get similar tracks
* **Playlist builder** – liked songs (with weights) and disliked songs -> playlist
* **Mood explorer** – target values for chosen attributes -> matching tracks
* **Explain** – double-click any recommendation to see why it was picked

The left-hand panel holds engine settings and filters; changing them refreshes
the visible results.  "Compare with legacy model" shows the original CMPT 310
model's results next to the remake's.  The engine (and the legacy model) load
in background threads so the window never freezes.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .catalog import TrackNotFoundError, TrackRef
from .config import PROJECT_ROOT, EngineConfig
from .legacy import LEGACY_CONFIG
from .ranking import Filters
from .recommender import Recommender

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
RESULT_COLS = ["rank", "name", "artists", "year", "genres", "popularity", "similarity", "why"]
YEAR_LO, YEAR_HI = 1900, 2025
ROW_ROLE = Qt.ItemDataRole.UserRole
WEIGHT_ROLE = Qt.ItemDataRole.UserRole.value + 1

STYLE = """
QMainWindow, QWidget { font-family: 'Segoe UI', 'Helvetica Neue', Arial; font-size: 10pt; }
QGroupBox { font-weight: 600; border: 1px solid #c8e6d4; border-radius: 6px;
            margin-top: 10px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #0b5d3e; }
QPushButton { background: #2ecc71; color: white; border: none; border-radius: 5px;
              padding: 6px 12px; font-weight: 600; }
QPushButton:hover { background: #27ae60; }
QPushButton:pressed { background: #1e8449; }
QPushButton:disabled { background: #a9dfbf; }
QPushButton[flat="true"] { background: transparent; color: #1e8449; }
QTabBar::tab { padding: 7px 14px; }
QTabBar::tab:selected { background: #2ecc71; color: white; border-radius: 4px; }
QHeaderView::section { background: #d5f5e3; color: black; padding: 4px; border: none; font-weight: 600; }
QLabel#Title { font-size: 16pt; font-weight: 700; color: #064420; }
QLabel#Subtitle { color: #2c6e49; }
QLabel#Hint { color: #5d6d7e; }
QProgressBar { border: 1px solid #c8e6d4; border-radius: 3px; text-align: center; height: 14px;
               color: black; background: #f4fbf7; }
QProgressBar::chunk { background: #58d68d; }
"""


def ranged(widget, lo, hi, value):
    """Set range *then* value (keyword-property order is not guaranteed to do this)."""
    widget.setRange(lo, hi)
    widget.setValue(value)
    return widget


# ----------------------------------------------------------------------
# Background workers
# ----------------------------------------------------------------------
class _Worker(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], object]):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # surfaced in a message box
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_in_thread(parent: QObject, fn: Callable[[], object], on_done, on_failed) -> QThread:
    """Run ``fn`` on a QThread and deliver its result on the GUI thread."""
    thread = QThread(parent)
    worker = _Worker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.done.connect(on_done)
    worker.failed.connect(on_failed)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread._worker = worker  # keep a reference alive
    thread.start()
    return thread


def load_engine(model_path: Path = MODEL_PATH) -> Recommender:
    if model_path.exists():
        try:
            return Recommender.load(model_path)
        except Exception:  # stale model file -> refit from CSV
            pass
    return Recommender.from_csv()


# ----------------------------------------------------------------------
# Table model for DataFrames
# ----------------------------------------------------------------------
class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, columns: list[str] | None = None):
        super().__init__()
        self._columns = columns
        self._df = pd.DataFrame()
        self.set_frame(df if df is not None else pd.DataFrame())

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._full = df.reset_index(drop=True)
        cols = [c for c in (self._columns or list(df.columns)) if c in df.columns]
        self._df = self._full[cols]
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        return self._full

    def rowCount(self, parent=None) -> int:  # noqa: N802 (Qt API)
        return 0 if parent is not None and parent.isValid() else len(self._df)

    def columnCount(self, parent=None) -> int:  # noqa: N802
        return 0 if parent is not None and parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        col = self._df.columns[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if col == "similarity":
                return f"{float(value):.0%}"
            if isinstance(value, float):
                return f"{value:.3f}".rstrip("0").rstrip(".")
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and col in ("rank", "year", "popularity", "similarity"):
            return Qt.AlignmentFlag.AlignCenter.value
        if role == Qt.ItemDataRole.BackgroundRole and col == "similarity":
            v = max(0.0, min(1.0, float(value)))
            return QColor(int(255 - 120 * v), 255, int(255 - 120 * v))
        if role == Qt.ItemDataRole.ForegroundRole and col == "similarity":
            # Always black on the light-green cell, whatever the system theme's text colour.
            return QColor("black")
        if role == ROW_ROLE and "row" in self._full.columns:
            return int(self._full.iloc[index.row()]["row"])
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            name = str(self._df.columns[section])
            return "closest on" if name == "why" else name
        return None


def make_table(columns: list[str] | None = None) -> tuple[QTableView, DataFrameModel]:
    model = DataFrameModel(columns=columns)
    view = QTableView()
    view.setModel(model)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.setAlternatingRowColors(True)
    view.verticalHeader().setVisible(False)
    view.horizontalHeader().setStretchLastSection(True)
    view.setWordWrap(False)
    return view, model


# ----------------------------------------------------------------------
# Reusable widgets
# ----------------------------------------------------------------------
class TrackSearch(QWidget):
    """Title + artist inputs with a live, ranked list of catalogue matches."""

    picked = pyqtSignal(int)  # catalogue row double-clicked / Enter

    def __init__(self, title_hint: str = "Song title", parent=None):
        super().__init__(parent)
        self.engine: Recommender | None = None
        self.title = QLineEdit(placeholderText=f"{title_hint}  (e.g. Bohemian Rhapsody)")
        self.artist = QLineEdit(placeholderText="Artist (optional)")
        self.matches = QListWidget()
        self.matches.setMinimumHeight(90)
        self.matches.setMaximumHeight(130)
        self.status = QLabel("Type a title to search the catalogue.", objectName="Hint")

        row = QHBoxLayout()
        row.addWidget(self.title, 3)
        row.addWidget(self.artist, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(self.matches)
        lay.addWidget(self.status)

        self._timer = QTimer(self, singleShot=True, interval=250)
        self._timer.timeout.connect(self.refresh)
        self.title.textChanged.connect(self._timer.start)
        self.artist.textChanged.connect(self._timer.start)
        self.title.returnPressed.connect(self._enter)
        self.artist.returnPressed.connect(self._enter)
        self.matches.itemDoubleClicked.connect(lambda item: self.picked.emit(item.data(ROW_ROLE)))

    def set_engine(self, engine: Recommender) -> None:
        self.engine = engine
        self.refresh()

    def refresh(self) -> None:
        self.matches.clear()
        q = self.title.text().strip()
        if not q or self.engine is None:
            self.status.setText("Type a title to search the catalogue." if self.engine else "Engine loading...")
            return
        artist = self.artist.text().strip() or None
        hits = self.engine.search(q, artist, limit=25)
        if hits.empty and artist is None:  # maybe "Title by Artist" / "Title - Artist"
            for title, art in TrackRef(title=q).readings()[1:]:
                hits = self.engine.search(title, art, limit=25)
                if not hits.empty:
                    break
        if hits.empty:
            sugg = self.engine.catalog.suggest(q)
            self.status.setText("No match." + (f" Did you mean: {', '.join(sugg)}?" if sugg else ""))
            return
        for _, h in hits.iterrows():
            item = QListWidgetItem(f"{h['name']}  —  {h['artists_display']}  ({h['year']}, pop {h['popularity']})")
            item.setData(ROW_ROLE, int(h["row"]))
            self.matches.addItem(item)
        self.matches.setCurrentRow(0)
        self.status.setText(f"{len(hits)} match(es), best first. Double-click or press Enter to use one.")

    def selected_row(self) -> int | None:
        item = self.matches.currentItem()
        return None if item is None else int(item.data(ROW_ROLE))

    def _enter(self) -> None:
        self._timer.stop()
        self.refresh()
        row = self.selected_row()
        if row is not None:
            self.picked.emit(row)


class ResultsPanel(QWidget):
    """Recommendation table(s): remake, plus the legacy model when comparing."""

    activated = pyqtSignal(int)  # catalogue row of a double-clicked recommendation

    def __init__(self, parent=None):
        super().__init__(parent)
        self.caption = QLabel("No recommendations yet.", objectName="Hint")
        self.view, self.model = make_table(RESULT_COLS)
        self.legacy_view, self.legacy_model = make_table(RESULT_COLS)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setEnabled(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Remake engine</b>"))
        ll.addWidget(self.view)
        self.legacy_box = QWidget()
        rl = QVBoxLayout(self.legacy_box)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("<b>Legacy CMPT 310 model</b>"))
        rl.addWidget(self.legacy_view)
        self.legacy_box.setVisible(False)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.legacy_box)

        bottom = QHBoxLayout()
        bottom.addWidget(self.caption, 1)
        bottom.addWidget(self.export_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split, 1)
        lay.addLayout(bottom)

        for v in (self.view, self.legacy_view):
            v.doubleClicked.connect(lambda idx, v=v: self.activated.emit(v.model().data(idx, ROW_ROLE)))
        self.export_btn.clicked.connect(self._export)

    def show_results(self, recs: pd.DataFrame, legacy: pd.DataFrame | None, caption: str) -> None:
        self.model.set_frame(recs)
        self.legacy_box.setVisible(legacy is not None)
        if legacy is not None:
            self.legacy_model.set_frame(legacy)
        for v in (self.view, self.legacy_view):
            v.resizeColumnsToContents()
            v.setColumnWidth(1, min(v.columnWidth(1), 280))
            v.setColumnWidth(2, min(v.columnWidth(2), 200))
        self.caption.setText(caption if not recs.empty else "No tracks satisfy the current filters.")
        self.export_btn.setEnabled(not recs.empty)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export recommendations", "recommendations.csv", "CSV (*.csv)")
        if path:
            self.model.frame().drop(columns=["row"], errors="ignore").to_csv(path, index=False)


class SettingsPanel(QWidget):
    """Engine knobs and filters. Emits ``changed`` (debounced) on any edit."""

    changed = pyqtSignal()
    compare_toggled = pyqtSignal(bool)

    def __init__(self, config: EngineConfig, parent=None):
        super().__init__(parent)
        self.k = ranged(QSpinBox(), 1, 50, 15)
        self.diversity = self._slider(0, 100, int(config.diversity * 100))
        self.popularity = self._slider(0, 50, int(config.popularity_weight * 100))
        self.strategy = QComboBox()
        self.strategy.addItems(["hybrid", "centroid", "nearest_seed"])
        self.strategy.setCurrentText(config.strategy)
        self.max_artist = ranged(QSpinBox(), 1, 10, config.max_per_artist)
        self.max_seed = ranged(QSpinBox(), 0, 10, config.max_seed_artist)
        self.max_seed.setToolTip("How many songs by the seed songs' own artists a list may contain.")
        self.flow = QCheckBox("Order as a smooth flow")
        self.flow.setToolTip("Re-order the chosen tracks so tempo, energy and key change gradually.")
        self.div_label = QLabel()
        self.pop_label = QLabel()

        eng = QGroupBox("Engine")
        f = QFormLayout(eng)
        f.addRow("Results (k)", self.k)
        f.addRow(self.div_label, self.diversity)
        f.addRow(self.pop_label, self.popularity)
        f.addRow("Multi-seed strategy", self.strategy)
        f.addRow("Max per artist", self.max_artist)
        f.addRow("Songs by seed artist", self.max_seed)
        f.addRow(self.flow)

        self.year_min = ranged(QSpinBox(), YEAR_LO, YEAR_HI, YEAR_LO)
        self.year_max = ranged(QSpinBox(), YEAR_LO, YEAR_HI, YEAR_HI)
        self.min_pop = ranged(QSpinBox(), 0, 100, 0)
        self.genre = QLineEdit(placeholderText="e.g. soul, indie, metal")
        self.genre.setClearButtonEnabled(True)
        self.no_explicit = QCheckBox("Exclude explicit tracks")
        flt = QGroupBox("Filters")
        g = QFormLayout(flt)
        years = QHBoxLayout()
        years.addWidget(self.year_min)
        years.addWidget(QLabel("to"))
        years.addWidget(self.year_max)
        g.addRow("Years", years)
        g.addRow("Min popularity", self.min_pop)
        g.addRow("Genre contains", self.genre)
        g.addRow(self.no_explicit)

        self.compare = QCheckBox("Compare with legacy model")
        self.compare.setToolTip("Show the original CMPT 310 model's results side by side.")
        self.reset_btn = QPushButton("Reset settings")
        self.reset_btn.setProperty("flat", True)

        lay = QVBoxLayout(self)
        lay.addWidget(eng)
        lay.addWidget(flt)
        cmp_box = QGroupBox("Compare")
        QVBoxLayout(cmp_box).addWidget(self.compare)
        lay.addWidget(cmp_box)
        lay.addWidget(self.reset_btn)
        lay.addStretch(1)

        self._defaults = config
        self._timer = QTimer(self, singleShot=True, interval=200)
        self._timer.timeout.connect(self.changed.emit)
        for w in (self.k, self.max_artist, self.max_seed, self.year_min, self.year_max, self.min_pop):
            w.valueChanged.connect(self._timer.start)
        self.genre.textChanged.connect(self._timer.start)
        self.flow.toggled.connect(self._timer.start)
        for s in (self.diversity, self.popularity):
            s.valueChanged.connect(self._update_labels)
            s.valueChanged.connect(self._timer.start)
        self.strategy.currentTextChanged.connect(self._timer.start)
        self.no_explicit.toggled.connect(self._timer.start)
        self.compare.toggled.connect(self.compare_toggled.emit)
        self.reset_btn.clicked.connect(self.reset)
        self._update_labels()

    @staticmethod
    def _slider(lo: int, hi: int, value: int) -> QSlider:
        s = ranged(QSlider(Qt.Orientation.Horizontal), lo, hi, value)
        s.setSingleStep(1)
        s.setPageStep(5)
        return s

    def _update_labels(self) -> None:
        self.div_label.setText(f"Diversity {self.diversity.value() / 100:.2f}")
        self.pop_label.setText(f"Popularity prior {self.popularity.value() / 100:.2f}")

    def reset(self) -> None:
        c = self._defaults
        self.k.setValue(15)
        self.diversity.setValue(int(c.diversity * 100))
        self.popularity.setValue(int(c.popularity_weight * 100))
        self.strategy.setCurrentText(c.strategy)
        self.max_artist.setValue(c.max_per_artist)
        self.max_seed.setValue(c.max_seed_artist)
        self.flow.setChecked(c.order == "flow")
        self.year_min.setValue(YEAR_LO)
        self.year_max.setValue(YEAR_HI)
        self.min_pop.setValue(0)
        self.genre.clear()
        self.no_explicit.setChecked(False)

    def overrides(self) -> dict:
        return {
            "diversity": self.diversity.value() / 100,
            "popularity_weight": self.popularity.value() / 100,
            "strategy": self.strategy.currentText(),
            "max_per_artist": self.max_artist.value(),
            "max_seed_artist": self.max_seed.value(),
            "order": "flow" if self.flow.isChecked() else "score",
        }

    def filters(self) -> Filters:
        lo, hi = sorted((self.year_min.value(), self.year_max.value()))
        return Filters(
            year_min=lo if lo > YEAR_LO else None,
            year_max=hi if hi < YEAR_HI else None,
            min_popularity=self.min_pop.value() or None,
            exclude_explicit=self.no_explicit.isChecked(),
            genre=self.genre.text().strip() or None,
        )


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
class RecommendTab(QWidget):
    """Base class: owns a ResultsPanel and knows how to (re)compute results."""

    explain_requested = pyqtSignal(list, int)  # seed rows, recommended row

    def __init__(self, window: MainWindow):
        super().__init__()
        self.win = window
        self.results = ResultsPanel()
        self.results.activated.connect(lambda row: self.explain_requested.emit(self.seed_rows(), row))
        self.has_results = False

    def seed_rows(self) -> list[int]:
        return []

    def compute(self, engine: Recommender, **kw) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def caption(self) -> str:
        return ""

    def run(self) -> None:
        engine = self.win.engine
        if engine is None:
            return
        kw = {"k": self.win.settings.k.value(), "filters": self.win.settings.filters()}
        start = time.perf_counter()
        try:
            recs = self.compute(engine, **kw, **self.win.settings.overrides())
            legacy = self.compute(self.win.legacy, **kw) if self.win.legacy_active() else None
        except (TrackNotFoundError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot recommend", str(exc))
            return
        ms = (time.perf_counter() - start) * 1000
        self.results.show_results(
            recs,
            legacy,
            f"{self.caption()}  ·  {len(recs)} tracks in {ms:.0f} ms  ·  double-click a row to see why it was picked",
        )
        self.has_results = True


class RadioTab(RecommendTab):
    def __init__(self, window):
        super().__init__(window)
        self.search = TrackSearch()
        self.button = QPushButton("Find similar songs")
        self.seed: int | None = None
        top = QGroupBox("Seed song")
        tl = QVBoxLayout(top)
        tl.addWidget(self.search)
        tl.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignRight)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.results)
        split.setStretchFactor(1, 3)
        QVBoxLayout(self).addWidget(split)
        self.button.clicked.connect(self._go)
        self.search.picked.connect(self._go_row)

    def _go(self) -> None:
        row = self.search.selected_row()
        if row is None:
            QMessageBox.information(self, "Pick a song", "Search for a song and select one of the matches.")
            return
        self._go_row(row)

    def _go_row(self, row: int) -> None:
        self.seed = row
        self.run()

    def seed_rows(self) -> list[int]:
        return [] if self.seed is None else [self.seed]

    def compute(self, engine, **kw):
        return engine.similar_to(self.seed, **kw)

    def caption(self) -> str:
        t = self.win.engine.tracks.iloc[self.seed]
        return f"Similar to “{t['name']}” — {t['artists_display']} ({t['year']})"


class PasteDialog(QDialog):
    """Paste songs copied from Spotify (or typed) to import them as liked seeds."""

    HELP = (
        "<b>From the Spotify app:</b> open a playlist, click a song, press <b>Ctrl+A</b> then "
        "<b>Ctrl+C</b> (Cmd on a Mac), and paste here. You get one link per song.<br>"
        "Also accepted: <code>spotify:track:</code> URIs, lines like <i>Title - Artist</i>, "
        "or a CSV export (e.g. Exportify). Album and playlist <i>links</i> can't be expanded "
        "without the Spotify API: copy the songs instead."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paste a playlist")
        self.resize(640, 420)
        info = QLabel(self.HELP, wordWrap=True)
        self.text = QPlainTextEdit(
            placeholderText="https://open.spotify.com/track/...\nhttps://open.spotify.com/track/...\nTake On Me - a-ha"
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.text, 1)
        lay.addWidget(buttons)


class PlaylistTab(RecommendTab):
    def __init__(self, window):
        super().__init__(window)
        self.search = TrackSearch("Add a song")
        self.weight = ranged(QDoubleSpinBox(singleStep=0.25), 0.25, 3.0, 1.0)
        add_like = QPushButton("Add to liked ♥")
        add_dislike = QPushButton("Add to disliked ✗")
        add_dislike.setStyleSheet("background:#e67e22;")
        self.liked = QListWidget()
        self.disliked = QListWidget()
        for lst in (self.liked, self.disliked):
            lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            lst.setMaximumHeight(110)
        remove = QPushButton("Remove selected")
        remove.setProperty("flat", True)
        clear = QPushButton("Clear all")
        clear.setProperty("flat", True)
        paste = QPushButton("Paste from Spotify…")
        paste.setToolTip("Import songs copied from a Spotify playlist (links), URIs, 'Title - Artist' lines or a CSV.")
        paste.setStyleSheet("background:#1db954;")
        self.button = QPushButton("Generate playlist")

        adder = QHBoxLayout()
        adder.addWidget(QLabel("Weight"))
        adder.addWidget(self.weight)
        adder.addStretch(1)
        adder.addWidget(add_like)
        adder.addWidget(add_dislike)

        lists = QGridLayout()
        lists.addWidget(QLabel("<b>Liked</b> (weight in brackets)"), 0, 0)
        lists.addWidget(QLabel("<b>Disliked</b>"), 0, 1)
        lists.addWidget(self.liked, 1, 0)
        lists.addWidget(self.disliked, 1, 1)
        actions = QHBoxLayout()
        actions.addWidget(remove)
        actions.addWidget(clear)
        actions.addWidget(paste)
        actions.addStretch(1)
        actions.addWidget(self.button)

        top = QGroupBox("Your taste")
        tl = QVBoxLayout(top)
        tl.addWidget(self.search)
        tl.addLayout(adder)
        tl.addLayout(lists)
        tl.addLayout(actions)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.results)
        split.setStretchFactor(1, 2)
        split.setSizes([420, 400])
        QVBoxLayout(self).addWidget(split)

        add_like.clicked.connect(lambda: self.add(liked=True))
        add_dislike.clicked.connect(lambda: self.add(liked=False))
        self.search.picked.connect(lambda _row: self.add(liked=True))
        remove.clicked.connect(self._remove)
        clear.clicked.connect(self._clear)
        paste.clicked.connect(self._paste_dialog)
        self.button.clicked.connect(self.run)

    def add(self, liked: bool, row: int | None = None) -> None:
        row = self.search.selected_row() if row is None else row
        if row is None or self.win.engine is None:
            return
        target = self.liked if liked else self.disliked
        if row in self._rows(self.liked) + self._rows(self.disliked):
            return
        t = self.win.engine.tracks.iloc[row]
        label = f"{t['name']} — {t['artists_display']} ({t['year']})"
        weight = self.weight.value() if liked else 1.0
        item = QListWidgetItem(f"{label}  [{weight:g}]" if liked else label)
        item.setData(ROW_ROLE, row)
        item.setData(WEIGHT_ROLE, weight)
        target.addItem(item)

    def _paste_dialog(self) -> None:
        if self.win.engine is None:
            return
        dlg = PasteDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.import_text(dlg.text.toPlainText())

    def import_text(self, text: str) -> None:
        """Add every matched song from pasted text to the liked list and report the rest."""
        engine = self.win.engine
        if engine is None or not text.strip():
            return
        imp = engine.import_playlist(text)
        before = self.liked.count()
        for row in imp.rows:
            self.add(liked=True, row=row)
        added = self.liked.count() - before
        message = imp.summary() + f"\n{added} added to your liked songs."
        problems = imp.unmatched
        if problems:
            shown = "\n".join(f"• {i.text}  ({i.status})" for i in problems[:12])
            more = f"\n… and {len(problems) - 12} more" if len(problems) > 12 else ""
            message += f"\n\nNot imported:\n{shown}{more}"
        self.last_import = imp
        self.win.statusBar().showMessage(imp.summary(), 8000)
        if imp.rows:
            QMessageBox.information(self, "Playlist imported", message)
        else:
            QMessageBox.warning(self, "Nothing imported", message)

    @staticmethod
    def _rows(lst: QListWidget) -> list[int]:
        return [int(lst.item(i).data(ROW_ROLE)) for i in range(lst.count())]

    def _remove(self) -> None:
        for lst in (self.liked, self.disliked):
            for item in lst.selectedItems():
                lst.takeItem(lst.row(item))

    def _clear(self) -> None:
        self.liked.clear()
        self.disliked.clear()

    def seed_rows(self) -> list[int]:
        return self._rows(self.liked)

    def run(self) -> None:
        if not self.seed_rows():
            QMessageBox.information(self, "No liked songs", "Add at least one liked song first.")
            return
        super().run()

    def compute(self, engine, **kw):
        weights = [float(self.liked.item(i).data(WEIGHT_ROLE)) for i in range(self.liked.count())]
        if engine is self.win.legacy:  # the original had no weights or dislikes
            return engine.recommend_rows(self.seed_rows(), **kw)
        return engine.recommend_rows(
            self.seed_rows(), seed_weights=weights, disliked_rows=self._rows(self.disliked), **kw
        )

    def caption(self) -> str:
        n, d = self.liked.count(), self.disliked.count()
        return f"Playlist from {n} liked" + (f" and {d} disliked" if d else "") + " song(s)"


MOOD_SPEC = [
    # name, min, max, default, scale (slider ints -> value), on by default
    ("energy", 0, 100, 80, 0.01, True),
    ("danceability", 0, 100, 70, 0.01, True),
    ("valence", 0, 100, 70, 0.01, True),
    ("acousticness", 0, 100, 20, 0.01, False),
    ("instrumentalness", 0, 100, 0, 0.01, False),
    ("speechiness", 0, 100, 5, 0.01, False),
    ("liveness", 0, 100, 10, 0.01, False),
    ("tempo", 40, 220, 120, 1.0, False),
    ("loudness", -40, 0, -6, 1.0, False),
    ("year", 1950, 2025, 2015, 1.0, False),
]


class MoodTab(RecommendTab):
    def __init__(self, window):
        super().__init__(window)
        box = QGroupBox("Describe the sound — only ticked attributes are used")
        grid = QGridLayout(box)
        self.controls: dict[str, tuple[QCheckBox, QSlider, QLabel, float]] = {}
        for i, (name, lo, hi, default, scale, on) in enumerate(MOOD_SPEC):
            cb = QCheckBox(name, checked=on)
            sl = ranged(QSlider(Qt.Orientation.Horizontal), lo, hi, default)
            val = QLabel()
            val.setMinimumWidth(48)
            r, c = divmod(i, 2)
            grid.addWidget(cb, r, c * 3)
            grid.addWidget(sl, r, c * 3 + 1)
            grid.addWidget(val, r, c * 3 + 2)
            self.controls[name] = (cb, sl, val, scale)
            sl.valueChanged.connect(self._update_labels)
            cb.toggled.connect(self._update_labels)
        self.button = QPushButton("Find matching songs")
        grid.addWidget(self.button, len(MOOD_SPEC) // 2 + 1, 4, 1, 2)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(box)
        split.addWidget(self.results)
        split.setStretchFactor(1, 3)
        QVBoxLayout(self).addWidget(split)
        self.button.clicked.connect(self.run)
        self._update_labels()

    def _update_labels(self) -> None:
        for cb, sl, val, scale in self.controls.values():
            v = sl.value() * scale
            val.setText(f"{v:.2f}" if scale < 1 else f"{v:.0f}")
            sl.setEnabled(cb.isChecked())

    def targets(self) -> dict[str, float]:
        return {n: sl.value() * s for n, (cb, sl, _, s) in self.controls.items() if cb.isChecked()}

    def run(self) -> None:
        if not self.targets():
            QMessageBox.information(self, "No attributes", "Tick at least one attribute.")
            return
        super().run()

    def compute(self, engine, **kw):
        targets = self.targets()
        if engine is self.win.legacy:
            # legacy space has no loudness / era: drop targets it cannot use
            usable = {k: v for k, v in targets.items() if k in engine.pipeline.groups}
            if not usable:
                return engine._empty()
            targets = usable
        return engine.from_targets(targets, **kw)

    def caption(self) -> str:
        return "Matching " + ", ".join(f"{k}={v:g}" for k, v in self.targets().items())


class ExplainTab(QWidget):
    """Per-feature breakdown of why a track was recommended."""

    def __init__(self, window: MainWindow):
        super().__init__()
        self.win = window
        self.header = QLabel("Double-click a recommendation in any tab to explain it.", objectName="Hint")
        self.header.setWordWrap(True)
        self.features = QTableWidget(0, 4)
        self.features.setHorizontalHeaderLabels(["Feature", "Weight", "Closeness", "Share of difference"])
        self.features.verticalHeader().setVisible(False)
        self.features.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.features.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.features.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.features.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.attrs_view, self.attrs_model = make_table()
        note = QLabel(
            "<b>Closeness</b>: 0–100 % similarity on that feature alone.  "
            "<b>Share of difference</b>: how much of the remaining distance to the seeds the feature "
            "accounts for — the top rows are what makes this track sound different.",
            objectName="Hint",
        )
        note.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.header)
        split = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.features)
        tl.addWidget(note)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(QLabel("<b>Seeds vs. this recommendation</b> (last row)"))
        bl.addWidget(self.attrs_view)
        split.addWidget(top)
        split.addWidget(bottom)
        lay.addWidget(split, 1)

    def explain(self, seeds: list[int], row: int) -> None:
        engine = self.win.engine
        if engine is None:
            return
        t = engine.tracks.iloc[row]
        if not seeds:  # mood search: no seed tracks, just show attributes
            self.header.setText(
                f"<b>{t['name']}</b> — {t['artists_display']} ({t['year']}). "
                "Mood results have no seed tracks, so only attributes are shown."
            )
            self.features.setRowCount(0)
            self.attrs_model.set_frame(engine.describe([row]).drop(columns=["id"]))
            self.attrs_view.resizeColumnsToContents()
            return
        exp = engine.explain(seeds, row)
        self.header.setText(
            f"Why <b>{t['name']}</b> — {t['artists_display']} ({t['year']}) "
            f"was recommended for {len(seeds)} seed song(s)"
        )
        self.features.setRowCount(len(exp))
        for i, r in exp.iterrows():
            self.features.setItem(i, 0, QTableWidgetItem(r["feature"]))
            self.features.setItem(i, 1, QTableWidgetItem(f"{r['weight']:g}"))
            for col, value in ((2, r["closeness"]), (3, r["share"])):
                bar = ranged(QProgressBar(), 0, 100, int(round(value * 100)))
                bar.setFormat(f"{value:.0%}")
                self.features.setCellWidget(i, col, bar)
        self.attrs_model.set_frame(engine.describe([*seeds, row]).drop(columns=["id"]))
        self.attrs_view.resizeColumnsToContents()


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    engine_ready = pyqtSignal()

    def __init__(self, engine: Recommender | None = None, loader: Callable[[], Recommender] = load_engine):
        super().__init__()
        self.setWindowTitle("Music Recommender Remake")
        self.resize(1280, 820)
        self.engine: Recommender | None = None
        self.legacy: Recommender | None = None
        self._threads: list[QThread] = []

        self.settings = SettingsPanel(EngineConfig())
        self.radio = RadioTab(self)
        self.playlist = PlaylistTab(self)
        self.mood = MoodTab(self)
        self.explain_tab = ExplainTab(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.radio, "Song radio")
        self.tabs.addTab(self.playlist, "Playlist builder")
        self.tabs.addTab(self.mood, "Mood explorer")
        self.tabs.addTab(self.explain_tab, "Explain")
        self.rec_tabs: list[RecommendTab] = [self.radio, self.playlist, self.mood]
        for tab in self.rec_tabs:
            tab.explain_requested.connect(self._explain)

        title = QLabel("Music Recommender Remake", objectName="Title")
        subtitle = QLabel(
            "Content-based k-NN recommendations over Spotify audio features, genres and artist sound",
            objectName="Subtitle",
        )
        side = QScrollArea()
        side.setWidgetResizable(True)
        side.setWidget(self.settings)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side.setMinimumWidth(self.settings.sizeHint().width() + 24)
        side.setMaximumWidth(self.settings.sizeHint().width() + 60)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(side)
        body.addWidget(self.tabs)
        body.setStretchFactor(1, 1)
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addWidget(body, 1)
        self.setCentralWidget(central)

        self.progress = QProgressBar(maximum=0)  # busy indicator
        self.progress.setMaximumWidth(160)
        self.statusBar().addPermanentWidget(self.progress)
        self._build_menu()

        self.settings.changed.connect(self.refresh_visible)
        self.settings.compare_toggled.connect(self._compare_toggled)
        self.tabs.setEnabled(False)

        if engine is not None:
            self._engine_loaded(engine)
        else:
            self.statusBar().showMessage("Loading the catalogue (~7 s with a built model, ~35 s from the CSV)…")
            self._threads.append(run_in_thread(self, loader, self._engine_loaded, self._load_failed))

    # ---- menu ------------------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        export = QAction("&Export current results…", self, shortcut=QKeySequence("Ctrl+E"))
        export.triggered.connect(self._export_current)
        quit_ = QAction("&Quit", self, shortcut=QKeySequence("Ctrl+Q"))
        quit_.triggered.connect(self.close)
        file_menu.addAction(export)
        file_menu.addAction(quit_)
        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _export_current(self) -> None:
        tab = self.tabs.currentWidget()
        if isinstance(tab, RecommendTab) and tab.has_results:
            tab.results._export()

    def _about(self) -> None:
        from . import __version__

        n = f"{len(self.engine.tracks):,}" if self.engine else "—"
        QMessageBox.about(
            self,
            "About",
            f"<b>Music Recommender Remake</b> v{__version__}<br>"
            f"Content-based k-NN recommender over {n} Spotify tracks.<br><br>"
            "Remake of the CMPT 310 Group 2 project. See docs/ for how it works.",
        )

    # ---- engine lifecycle -------------------------------------------------
    def _engine_loaded(self, engine: Recommender) -> None:
        self.engine = engine
        for tab in (self.radio, self.playlist):
            tab.search.set_engine(engine)
        self.tabs.setEnabled(True)
        self.progress.setVisible(False)
        dims = len(engine.pipeline.columns)
        self.statusBar().showMessage(f"Ready — {len(engine.tracks):,} tracks, {dims}-d feature space.")
        self.radio.search.title.setFocus()
        self.engine_ready.emit()

    def _load_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.statusBar().showMessage("Failed to load the engine.")
        QMessageBox.critical(self, "Could not load the engine", message)

    def legacy_active(self) -> bool:
        return self.settings.compare.isChecked() and self.legacy is not None

    def _compare_toggled(self, on: bool) -> None:
        if on and self.legacy is None and self.engine is not None:
            self.settings.compare.setEnabled(False)
            self.progress.setVisible(True)
            self.statusBar().showMessage("Fitting the legacy CMPT 310 model…")
            tracks = self.engine.tracks
            self._threads.append(
                run_in_thread(
                    self, lambda: Recommender(LEGACY_CONFIG).fit(tracks), self._legacy_loaded, self._load_failed
                )
            )
            return
        self.refresh_visible()

    def _legacy_loaded(self, legacy: Recommender) -> None:
        self.legacy = legacy
        self.settings.compare.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("Legacy model ready.", 4000)
        self.refresh_visible()

    def refresh_visible(self) -> None:
        tab = self.tabs.currentWidget()
        if isinstance(tab, RecommendTab) and tab.has_results:
            tab.run()

    def _explain(self, seeds: list[int], row: int) -> None:
        self.explain_tab.explain(seeds, row)
        self.tabs.setCurrentWidget(self.explain_tab)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        for t in self._threads:
            t.quit()
            t.wait(3000)
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Music Recommender Remake")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
