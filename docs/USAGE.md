# Usage guide

## Installation

Python 3.10+ is required (3.12 recommended).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[desktop,app,dev]"
```

`pip install -e .` alone installs just the engine and CLI. `[desktop]` adds
PyQt6 for the desktop app, `[app]` adds Streamlit and Plotly, and `[dev]` adds pytest and ruff. `requirements.txt` lists
the same packages for a plain `pip install -r requirements.txt`.

Optional, for fast start-up: pre-build the model once (~5 s).

```bash
music-rec build          # writes artifacts/model.joblib
```

## The desktop app

```bash
music-rec-gui                       # or: python -m music_recommender.gui
```

A native window (PyQt6). The engine loads in the background (instantly if
`artifacts/model.joblib` exists), and the status bar says when it's ready.

| Tab | How to use it |
|---|---|
| **Song radio** | Type a title (optionally an artist, or "Title by Artist") and matches appear as you type, best first. Double-click a match, press Enter, or click **Find similar songs**. |
| **Playlist builder** | Search, pick a weight, then **Add to liked** or **Add to disliked** (double-clicking a match adds it to liked). **Generate playlist** builds from all of them. Select entries and use **Remove selected** to edit. |
| **Mood explorer** | Tick attributes, drag sliders, then click **Find matching songs**. |
| **Explain** | Double-click any recommendation in another tab to see per-feature closeness and share of difference, plus a seeds-vs-track attribute table. |

The left panel holds engine settings (k, diversity, popularity prior,
strategy, artist cap) and filters (years, minimum popularity, explicit).
Changing them refreshes the visible results. **Compare with legacy model**
fits the original CMPT 310 model in the background and shows its results
side by side. **Export CSV…** (or *File → Export*, Ctrl+E) saves the current
list.

## The browser test bench

```bash
streamlit run app/streamlit_app.py
```

| Tab | What you can test |
|---|---|
| **Song radio** | Search for a track (ranked matches, pick the right version), get similar tracks, and compare seed and recommendation profiles on a radar chart |
| **Playlist builder** | Add liked songs (each with a weight) and disliked songs, then generate a playlist and download it as CSV |
| **Mood explorer** | Tick attributes such as energy, tempo or year, set target values, and get matching tracks |
| **Explain** | Pick any recommendation and see which features make it differ from the seeds, with a side-by-side attribute table |
| **Evaluate** | Run the offline benchmark (legacy vs remake vs ablations) live, or view the saved report, plus the original CMPT 310 results |
| **Dataset** | Catalogue statistics, feature distributions, active weights |

The sidebar controls the engine (k, diversity, popularity prior, multi-seed
strategy, artist cap) and filters (year range, minimum popularity, explicit).
**Compare** shows the legacy CMPT 310 model's results next to the remake's.

## Command line

All commands accept `--data PATH` (dataset), `--config FILE.json` (engine
config) and `--model FILE.joblib` (pre-built model).

```bash
music-rec search "bohemian rhapsody"
music-rec similar "Hotel California" -k 10
music-rec similar "Africa by Toto" --year-min 1980 --year-max 1989 --diversity 0.4
music-rec playlist --like "Take On Me" --like "Billie Jean" --like "Africa - Toto" --dislike "Macarena" -k 20
music-rec mood energy=0.9 danceability=0.8 tempo=124 year=2012 -k 15 --no-explicit
music-rec evaluate --queries 500        # → reports/benchmark.csv
music-rec analyze                       # → reports/era_sweep.csv, reports/feature_sensitivity.csv
music-rec tune --trials 40              # → configs/tuned.json (+ history)
```

Recommendation commands also accept `--min-popularity`, `--strategy
{centroid,nearest_seed,hybrid}`, `--popularity-weight` and `--max-per-artist`.

Seeds can be written as `"Title"`, `"Title by Artist"`, `"Title - Artist"`, a
Spotify track id, or `spotify:track:<id>`. When a title is ambiguous, the most
popular exact match wins. Use `search` to see the alternatives.

`python -m music_recommender ...` works the same way as `music-rec ...`.

## Python API

```python
from music_recommender import Recommender, Filters, EngineConfig, LEGACY_CONFIG

rec = Recommender.from_csv()                     # or Recommender.load("artifacts/model.joblib")

rec.search("take on me")                         # ranked matches (DataFrame)
rec.similar_to("Bohemian Rhapsody", k=10)

rec.recommend(
    liked=["Africa by Toto", ("Take On Me", "a-ha"), {"title": "Billie Jean", "weight": 2}],
    disliked=["Macarena"],
    k=20,
    filters=Filters(year_min=1975, year_max=1995, exclude_explicit=True),
    diversity=0.3,                               # per-call override of any EngineConfig field
)

rec.from_targets({"energy": 0.9, "valence": 0.8, "tempo": 128}, k=10)

recs = rec.similar_to("Hotel California", k=5)
rec.explain([rec.resolve("Hotel California")], recs.loc[0, "row"])   # per-feature breakdown
rec.describe(recs["row"])                         # human-readable attributes (key names, mm:ss, ...)

# Experiments
quiet = rec.with_config(diversity=0.0, popularity_weight=0.0)   # instant (same feature space)
no_era = rec.with_config(weights={"era": 0.0})                  # refits (~2 s)
legacy = Recommender(LEGACY_CONFIG).fit(rec.tracks)             # the original model

rec.save("artifacts/model.joblib")
```

Every recommendation call returns a DataFrame with the columns
`rank, id, name, artists, year, popularity, similarity, score, why, row`.

| Column | Meaning |
|---|---|
| `similarity` | Calibrated 0–1 relevance before the popularity prior (see ALGORITHM.md §3) |
| `score` | Final ranking score (relevance blended with popularity) |
| `why` | The 3 features where the track is unusually close to the seed(s) |
| `row` | Internal catalogue row (use with `explain` / `describe`) |

### Evaluation API

```python
from music_recommender import load_tracks
from music_recommender.evaluation import Evaluator, run_benchmark, feature_sensitivity

tracks = load_tracks()
ev = Evaluator(tracks, split="test", n_queries=300)
ev.evaluate(Recommender().fit(tracks), "remake").metrics
run_benchmark(tracks)          # legacy vs remake vs ablations → DataFrame
```

## Configuration files

```json
{
  "weights": {"danceability": 1.2, "energy": 1.4, "era": 1.0, "...": "..."},
  "scaling": "quantile",
  "metric": "euclidean",
  "candidate_pool": 300,
  "strategy": "hybrid",
  "hybrid_alpha": 0.5,
  "dislike_penalty": 0.6,
  "popularity_weight": 0.08,
  "diversity": 0.15,
  "max_per_artist": 2,
  "dedupe_versions": true,
  "min_tempo": 30.0,
  "random_state": 42
}
```

`configs/default.json` holds the defaults. Copy it, edit it, and pass it with
`--config`. A config file must list every weight you want active, because
groups that are missing get weight 0.

## Running the tests

```bash
pytest -m "not slow"     # ~2 s, synthetic data only
pytest                   # also runs the full-dataset and Streamlit app tests (~20 s)
ruff check src tests app
```
