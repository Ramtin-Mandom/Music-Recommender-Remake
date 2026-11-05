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

**Dataset.** The default catalogue is the ~900 MB v2 file at
`data/raw/tracks.csv`. It is git-ignored, so copy it there after cloning (see
[data/README.md](../data/README.md)). Without it, use the bundled CMPT 310 file
with `--data data/raw/tracks_old.csv` (or `Recommender.from_csv("data/raw/tracks_old.csv")`).

Recommended: pre-build the model once (~35 s). Loading the 900 MB CSV takes
~25 s, and the built model loads in ~7 s.

```bash
music-rec build          # writes artifacts/model.joblib (~220 MB, git-ignored)
```

## The desktop app

```bash
music-rec-gui                       # or: python -m music_recommender.gui
```

A native window (PyQt6). The engine loads in the background (~7 s if
`artifacts/model.joblib` exists, ~35 s from the CSV), and the status bar says
when it's ready.

| Tab | How to use it |
|---|---|
| **Song radio** | Type a title (optionally an artist, or "Title by Artist") and matches appear as you type, best first. Double-click a match, press Enter, or click **Find similar songs**. |
| **Playlist builder** | Search, pick a weight, then **Add to liked** or **Add to disliked** (double-clicking a match adds it to liked), or **Paste from Spotify…** to import a whole playlist (see below). **Generate playlist** builds from all of them. Select entries and use **Remove selected** to edit. |
| **Mood explorer** | Tick attributes, drag sliders, then click **Find matching songs**. |
| **Explain** | Double-click any recommendation in another tab to see per-feature closeness and share of difference, plus a seeds-vs-track attribute table. |

The left panel holds engine settings (k, diversity, popularity prior,
strategy, artist cap, **songs by seed artist**, **smooth flow ordering**) and
filters (years, minimum popularity, **genre contains**, explicit). Results
show each artist's top genres.
Changing them refreshes the visible results. **Compare with legacy model**
fits the original CMPT 310 model in the background and shows its results
side by side. **Export CSV…** (or *File → Export*, Ctrl+E) saves the current
list.

## Importing a playlist from Spotify

No Spotify account, API key or login is needed. Every song in the catalogue
carries its Spotify track id, so a pasted link matches exactly.

1. In the Spotify app, open a playlist, click any song, then press **Ctrl+A**
   and **Ctrl+C** (Cmd on a Mac). The clipboard now holds one
   `https://open.spotify.com/track/...` link per song.
2. Paste it into one of these:
   * the desktop app: **Playlist builder → Paste from Spotify…**
   * the web app: **Playlist builder → Paste a playlist from Spotify**
   * a text file for the CLI: `music-rec playlist --from-file playlist.txt`
     (`-` reads stdin), or `music-rec import playlist.txt` to only see the
     match report
   * Python: `rec.import_playlist(text)`

Accepted input (mixed freely):

| Input | Example |
|---|---|
| Track links (any `?si=` suffix, `/intl-xx/`) | `https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=…` |
| Track URIs or bare ids | `spotify:track:0VjIjW4GlUZAMYd2vXMi3b` |
| Typed lines, optionally numbered | `Mr. Brightside - The Killers`, `Daft Punk – Get Lucky`, `3. Shape of You by Ed Sheeran` |
| CSV exports with a header (e.g. [Exportify](https://exportify.app)) | columns `Track URI` / `Track ID` and/or `Track Name` + `Artist Name(s)` |

Every entry is reported as *matched* (by Spotify id or by title), *not in
catalogue*, *no match*, *unsupported* or *duplicate*, and nothing is imported
silently. Album, artist and playlist **links** can't be expanded without the
Spotify Web API, whose Development Mode only lets the app owner and up to 5
allowlisted users read their *own* playlists. Copy the songs instead.

**Coverage.** The catalogue holds 558k songs. Links to other copies of a song
(re-releases, compilations) still match, because cleaning remembers the ids of
the duplicates it removes. Of the dataset's tracks with popularity ≥ 70, 90 %
resolve (82 % at ≥ 50). The rest belong to artists the dataset never names.
A link to a song that isn't in the dataset at all can't be matched, but a typed
`Title - Artist` line for it might find another copy.

## The browser test bench

```bash
streamlit run app/streamlit_app.py
```

| Tab | What you can test |
|---|---|
| **Song radio** | Search for a track (ranked matches, pick the right version), get similar tracks, and compare seed and recommendation profiles on a radar chart |
| **Playlist builder** | Add liked songs (each with a weight) and disliked songs, or paste a Spotify playlist, then generate a playlist and download it as CSV |
| **Mood explorer** | Tick attributes such as energy, tempo or year, set target values, and get matching tracks |
| **Explain** | Pick any recommendation and see which features make it differ from the seeds, with a side-by-side attribute table |
| **Evaluate** | Run the offline benchmark (legacy vs remake vs ablations) live, or view the saved report, plus the original CMPT 310 results |
| **Dataset** | Catalogue statistics, feature distributions, active weights |

The sidebar controls the engine (k, diversity, popularity prior, multi-seed
strategy, artist caps, flow ordering) and filters (year range, minimum
popularity, genre, explicit).
**Compare** shows the legacy CMPT 310 model's results next to the remake's.

## Command line

All commands accept `--data PATH` (dataset), `--config FILE.json` (engine
config) and `--model FILE.joblib` (pre-built model).

```bash
music-rec search "bohemian rhapsody"
music-rec similar "Hotel California" -k 10
music-rec similar "Africa by Toto" --year-min 1980 --year-max 1989 --diversity 0.4
music-rec playlist --like "Take On Me" --like "Billie Jean" --like "Africa - Toto" --dislike "bad guy" -k 20
music-rec playlist --from-file my_playlist.txt -k 25    # songs pasted from Spotify
music-rec import my_playlist.txt                        # just show how the paste matched
music-rec mood energy=0.9 danceability=0.8 tempo=124 year=2012 -k 15 --no-explicit
music-rec similar "Love Train" --genre soul --flow       # genre filter + DJ-style ordering
music-rec playlist --like "Blinding Lights" --like "Levitating" --max-seed-artist 0
music-rec evaluate --queries 500        # → reports/benchmark.csv
music-rec analyze                       # → reports/{era,genre,artist_sound}_sweep.csv, feature_sensitivity.csv
music-rec tune --trials 40              # → configs/tuned.json (+ history)
```

Recommendation commands also accept `--min-popularity`, `--strategy
{centroid,nearest_seed,hybrid}`, `--popularity-weight`, `--max-per-artist`,
`--max-seed-artist N` (songs allowed by the seeds' own artists), `--genre TEXT`
and `--flow`.

Seeds can be written as `"Title"`, `"Title by Artist"`, `"Title - Artist"`, a
Spotify track link, a
Spotify track id, or `spotify:track:<id>`. When a title is ambiguous, the most
popular exact match wins. Use `search` to see the alternatives.

`python -m music_recommender ...` works the same way as `music-rec ...`.

## Python API

```python
from music_recommender import Recommender, Filters, EngineConfig, LEGACY_CONFIG

rec = Recommender.from_csv()                     # or Recommender.load("artifacts/model.joblib")

rec.search("take on me")                         # ranked matches (DataFrame)

imp = rec.import_playlist(open("my_playlist.txt", encoding="utf-8").read())
print(imp.summary())                             # "Matched 48 of 50 pasted song(s) ..."
imp.report(rec.tracks)                           # per-entry status
rec.recommend_rows(imp.rows, k=25)               # recommend from the whole playlist
rec.similar_to("Bohemian Rhapsody", k=10)

rec.recommend(
    liked=["Africa by Toto", ("Take On Me", "a-ha"), {"title": "Billie Jean", "weight": 2}],
    disliked=["bad guy"],
    k=20,
    filters=Filters(year_min=1975, year_max=1995, exclude_explicit=True, genre="rock"),
    diversity=0.3,                               # per-call override of any EngineConfig field
    order="flow",                                # smooth tempo / energy / key transitions
)

rec.from_targets({"energy": 0.9, "valence": 0.8, "tempo": 128}, k=10)

recs = rec.similar_to("Hotel California", k=5)
rec.explain([rec.resolve("Hotel California")], recs.loc[0, "row"])   # per-feature breakdown
rec.describe(recs["row"])                         # human-readable attributes (key names, mm:ss, ...)

# Experiments
quiet = rec.with_config(diversity=0.0, popularity_weight=0.0)   # instant (same feature space)
no_genre = rec.with_config(weights={"genre": 0.0})              # refits (~8 s)
legacy = Recommender(LEGACY_CONFIG).fit(rec.tracks)             # the original model

rec.save("artifacts/model.joblib")
```

Every recommendation call returns a DataFrame with the columns
`rank, id, name, artists, year, genres, popularity, similarity, score, why, row`.

| Column | Meaning |
|---|---|
| `similarity` | Calibrated 0–1 relevance before the popularity prior (see ALGORITHM.md §3) |
| `score` | Final ranking score (relevance blended with popularity) |
| `genres` | The artist's first three Spotify genre tags |
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
  "weights": {"danceability": 1.2, "energy": 1.4, "era": 1.5, "genre": 3.0, "artist_sound": 1.5, "...": "..."},
  "scaling": "quantile",
  "genre_dims": 16,
  "artist_shrinkage": 3.0,
  "metric": "euclidean",
  "candidate_pool": 300,
  "strategy": "hybrid",
  "hybrid_alpha": 0.5,
  "dislike_penalty": 0.6,
  "popularity_weight": 0.08,
  "artist_popularity_share": 0.3,
  "version_penalty": 0.15,
  "diversity": 0.15,
  "max_per_artist": 2,
  "max_seed_artist": 1,
  "order": "score",
  "dedupe_versions": true,
  "min_tempo": 30.0,
  "require_artist": true,
  "random_state": 42
}
```

`configs/default.json` holds the defaults. Copy it, edit it, and pass it with
`--config`. A config file must list every weight you want active, because
groups that are missing get weight 0.

## Running the tests

```bash
pytest -m "not slow"     # ~8 s, synthetic data only (incl. headless desktop-app tests)
pytest                   # also runs the full-dataset and Streamlit app tests (~2 min, needs the v2 file)
ruff check src tests app
```
