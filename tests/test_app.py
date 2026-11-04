"""Smoke test of the Streamlit test bench with Streamlit's AppTest harness."""

from pathlib import Path

import pytest

from music_recommender.config import DEFAULT_DATA_PATH

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DEFAULT_DATA_PATH.exists(), reason="dataset not present"),
]


def test_app_renders_and_recommends():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Song radio",
        "Playlist builder",
        "Mood explorer",
        "Explain",
        "Evaluate",
        "Dataset",
    ]

    at.text_input(key="radio_q").input("Hotel California").run()
    assert not at.exception
    assert at.selectbox(key="radio_sel").value is not None
    assert len(at.dataframe) >= 1  # recommendations table

    at.checkbox(key="compare").check().run()  # side-by-side with the legacy model
    assert not at.exception
