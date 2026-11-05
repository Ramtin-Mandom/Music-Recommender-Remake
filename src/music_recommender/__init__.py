"""Music Recommender Remake – a content-based k-NN music recommendation engine.

Quick start::

    from music_recommender import Recommender
    rec = Recommender.from_csv()
    print(rec.similar_to("Bohemian Rhapsody", k=10))
"""

__version__ = "3.1.0"

from .catalog import TrackCatalog, TrackNotFoundError, TrackRef  # noqa: E402
from .config import EngineConfig  # noqa: E402
from .data import clean_tracks, load_tracks, make_synthetic_tracks  # noqa: E402
from .importer import PlaylistImport  # noqa: E402
from .legacy import LEGACY_CONFIG  # noqa: E402
from .ranking import Filters  # noqa: E402
from .recommender import Recommender  # noqa: E402

__all__ = [
    "EngineConfig",
    "Filters",
    "LEGACY_CONFIG",
    "PlaylistImport",
    "Recommender",
    "TrackCatalog",
    "TrackNotFoundError",
    "TrackRef",
    "__version__",
    "clean_tracks",
    "load_tracks",
    "make_synthetic_tracks",
]
