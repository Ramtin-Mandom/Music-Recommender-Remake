# Music Recommender Remake

A content-based **k-nearest-neighbour music recommendation engine** built on
154k Spotify tracks, with an interactive test bench, a CLI, an offline
evaluation harness and a test suite.

It is a ground-up remake of the CMPT 310 (Fall 2025) Group 2 final project. The
original's core idea (recommend songs whose audio features sit close to the
ones you like) is kept. The feature space, ranking, evaluation and tooling are
rebuilt around it.

| | CMPT 310 model | Remake |
|---|---|---|
| Same-artist retrieval on held-out artists (objective) | 0.0067 | **0.0334 (5.0×)** |
| Radio: at least one same-artist song in the top 10 | 5 % of queries | **19 %** |
| Mean release-year gap to the seed | 17.1 years | **4.1 years** |
| Results returned when asking for 15 | ≤ 5 (bug) | 15 |
| Features | 8 | 14 groups (15 dims) |
| Tests | 0 | 78 |
| Apps | PyQt6 GUI | PyQt6 desktop app + Streamlit web test bench |

How these numbers are measured, and their caveats, is in
[docs/EVALUATION.md](docs/EVALUATION.md).

## Features

**Engine** (`src/music_recommender`)
- Weighted feature space: 9 audio descriptors + release era, duration, explicit
  flag, and **musical key on the circle of fifths**
- Quantile scaling, so skewed features (speechiness, liveness) don't dominate
- **Multi-seed playlists** with per-seed weights and three strategies
  (centroid / nearest-seed / hybrid)
- **Dislikes** as negative feedback
- **Re-ranking**: popularity prior, remaster/live-version dedupe, MMR diversity,
  per-artist cap
- **Filters** (year range, popularity, explicit, duration, artists) that never
  starve the list
- **Mood search**: recommend from target values ("energy 0.9, 128 BPM, 1985")
- **Explanations**: a `why` for every track, plus a per-feature breakdown
- Calibrated 0–1 similarity scores, robust track search (exact / fuzzy /
  "Title by Artist" / Spotify ids)
- Save/load fitted models; every knob lives in one JSON-serialisable `EngineConfig`

**Tooling**
- **Desktop app** (PyQt6, `music-rec-gui`): song radio with live search, playlist
  builder with weights and dislikes, mood explorer, per-recommendation
  explanations, CSV export, and side-by-side comparison with the legacy model
- **Streamlit test bench**: song radio, playlist builder, mood explorer, explain
  view, live benchmark, dataset explorer, and a side-by-side comparison with the
  legacy model
- **`music-rec` CLI**: `search`, `similar`, `playlist`, `mood`, `build`,
  `evaluate`, `analyze`, `tune`
- **Evaluation harness**: same-artist retrieval on a validation/test artist
  split, beyond-accuracy metrics, ablations, weight sweeps, sensitivity
  analysis, random-search tuner
- **78 pytest tests**: a fast synthetic-data suite including headless desktop-GUI tests (~3 s), plus
  opt-in full-dataset and Streamlit smoke tests

## Quick start

```bash
git clone git@github.com:Ramtin-Mandom/Music-Recommender-Remake.git
cd Music-Recommender-Remake
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -e ".[desktop,app,dev]"
```

Launch the desktop app:

```bash
music-rec-gui
```

Or the browser-based test bench:

```bash
streamlit run app/streamlit_app.py
```

Or use the CLI:

```bash
music-rec similar "Hotel California" -k 10
music-rec playlist --like "Take On Me" --like "Africa by Toto" --dislike "Macarena" -k 15
music-rec mood energy=0.9 danceability=0.8 tempo=124 year=2012
```

Or Python:

```python
from music_recommender import Recommender, Filters

rec = Recommender.from_csv()
rec.similar_to("Bohemian Rhapsody", k=10)
rec.recommend(["Africa by Toto", "Billie Jean"], disliked=["Macarena"],
              k=20, filters=Filters(year_min=1975, year_max=1995), diversity=0.3)
```

Tip: `music-rec build` caches the fitted model in `artifacts/`, so the app and
CLI start instantly.

Example: `music-rec playlist --like "Africa by Toto" --like "Take On Me" --like "Billie Jean" -k 6`

```
 rank                             name          artists  year  popularity  similarity                                 why
    1                     Typical Male      Tina Turner  1986          47       0.514  release era, speechiness, loudness
    2    Soy un Corazón Tendido al Sol    Víctor Manuel  1978          43       0.502  energy, acousticness, danceability
    3                   Get Used to It  Roger Voudouris  1979          37       0.504               tempo, mood, loudness
    4 Safe From Harm - 2012 Mix/Master   Massive Attack  1991          55       0.494            tempo, mood, release era
    5                  Reach The Beach         The Fixx  1983          31       0.515 release era, acousticness, loudness
    6               Don't Send Me Away Garfield Fleming  1981          44       0.513      release era, mood, speechiness
```

(The ranking also weighs diversity and popularity, so `similarity` is not
strictly decreasing.)

## How it works

```
seeds ─► weighted feature space ─► k-NN candidates (centroid + each seed)
      ─► filters ─► relevance (hybrid) − dislike penalty + popularity prior
      ─► version dedupe ─► MMR diversity + artist cap ─► playlist with "why"
```

Details and formulas: [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Installation, app guide, CLI reference, Python API, config files |
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | Feature engineering, scoring formulas, re-ranking, explanations |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout, design decisions, how to extend |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Protocols, results, ablations, era trade-off, sensitivity, limitations |
| [docs/MIGRATION.md](docs/MIGRATION.md) | What changed from CMPT 310: file mapping, bugs fixed, upgrades |
| [data/README.md](data/README.md) | Dataset card and cleaning steps |

## Project layout

```
src/music_recommender/   engine package (config, data, features, catalog, ranking, recommender, legacy, cli)
src/music_recommender/evaluation/   metrics, protocols, benchmarks, tuning
src/music_recommender/gui.py   PyQt6 desktop app (`music-rec-gui`)
app/streamlit_app.py     browser-based test bench
tests/                   pytest suite
data/raw/tracks.csv      170k-track Spotify dataset
data/legacy/             original CMPT 310 results
configs/default.json     default engine configuration
reports/                 generated evaluation results
docs/                    documentation
```

## Development

```bash
pytest -m "not slow"        # fast suite (~2 s)
pytest                      # + full-dataset tests
ruff check src tests app && ruff format --check src tests app
music-rec evaluate          # regenerate reports/benchmark.csv
```

## Credits

- Original CMPT 310 Group 2 project: Cyrus Sandhu, Ramtin Rezaei, Jaskamal
  Chouhan, Varsha Adusumalli
- Dataset: "Spotify Dataset 1921–2020, 160k+ Tracks" (Kaggle), built from
  Spotify Web API audio features
- Remake: Ramtin Rezaei. MIT License (see [LICENSE](LICENSE)).
