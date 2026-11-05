# Music Recommender Remake

A content-based **k-nearest-neighbour music recommendation engine** over
558k Spotify tracks. It combines audio features, **artist genres** and each
artist's overall **sound**, and comes with a desktop app, a web test bench, a
CLI, an offline evaluation harness and a test suite.

It is a ground-up remake of the CMPT 310 (Fall 2025) Group 2 final project. The
original's core idea (recommend songs whose features sit close to the ones you
like) is kept. The data, feature space, ranking, evaluation and tooling are
rebuilt around it.

| | CMPT 310 model | Remake |
|---|---|---|
| Collaborator retrieval on held-out artists (objective) | 0.0020 | **0.0336 (17×)** |
| Recommendations sharing a genre with the seed | 7 % | **58 %** |
| Mean release-year gap to the seed | 10.0 years | **3.1 years** |
| Live / demo / karaoke / holiday versions in lists | 4.3 % | **0.7 %** |
| Results returned when asking for 15 | ≤ 5 (bug) | 15 |
| Catalogue | 170k tracks, no genres | 558k tracks with artist genres |
| Features | 8 | 16 groups active (40 dimensions) |
| Tests | 0 | 119 |
| Apps | PyQt6 GUI | PyQt6 desktop app + Streamlit web test bench |

How these numbers are measured, and their caveats, is in
[docs/EVALUATION.md](docs/EVALUATION.md).

## Features

**Engine** (`src/music_recommender`)
- Weighted feature space: 9 audio descriptors, release era, explicit flag,
  time signature, **musical key on the circle of fifths**, plus:
  - **Genre**: the artist's Spotify tags embedded with TF-IDF + SVD. Artists
    without tags borrow genres from artists who sound like them.
  - **Artist sound**: the average sound of the rest of the artist's catalogue.
- **Artist recovery**: 89 % of the raw rows have no artist name. They are
  matched to their artist through genre + follower + popularity keys.
- Quantile scaling, so skewed features (speechiness, liveness) don't dominate
- **Multi-seed playlists** with per-seed weights and three strategies
  (centroid / nearest-seed / hybrid)
- **Dislikes** as negative feedback
- **Spotify playlist import**: paste a playlist copied from the Spotify app
  (track links), URIs, `Title - Artist` lines or a CSV export. No Spotify login
  is needed, and songs match exactly by track id.
- **Re-ranking**:
  - track + artist popularity prior
  - penalty for live / demo / karaoke / sped-up / holiday versions
  - remaster dedupe
  - MMR diversity
  - per-artist cap, and a stricter cap on the seed's own artist
- **Flow ordering**: optionally re-orders a playlist so tempo, energy and key
  change smoothly
- **Filters** (year range, popularity, explicit, genre, artists) that never
  starve the list
- **Mood search**: recommend from target values ("energy 0.9, 128 BPM, 2015")
- **Explanations**: a `why` for every track, plus a per-feature breakdown
- Calibrated 0–1 similarity scores, robust track search (exact / fuzzy /
  "Title by Artist" / Spotify ids), exact k-NN in ~20 ms per request
- Reads both the v2 dataset and the original CMPT 310 file; one
  JSON-serialisable `EngineConfig` holds every setting

**Tooling**
- **Desktop app** (PyQt6, `music-rec-gui`): song radio with live search, playlist
  builder with weights and dislikes, mood explorer, per-recommendation
  explanations, CSV export, and side-by-side comparison with the legacy model
- **Streamlit test bench**: song radio, playlist builder, mood explorer, explain
  view, live benchmark, dataset explorer, and a side-by-side comparison with the
  legacy model
- **`music-rec` CLI**: `search`, `similar`, `playlist` (incl. `--from-file`), `import`, `mood`, `build`,
  `evaluate`, `analyze`, `tune`
- **Evaluation harness**: collaborator retrieval on a validation/test artist
  split, beyond-accuracy metrics, ablations, weight sweeps, sensitivity
  analysis, random-search tuner
- **119 pytest tests**: a fast synthetic-data suite including headless desktop-GUI
  tests (~5 s), plus opt-in full-dataset and Streamlit smoke tests

## Quick start

```bash
git clone git@github.com:Ramtin-Mandom/Music-Recommender-Remake.git
cd Music-Recommender-Remake
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -e ".[desktop,app,dev]"
```

**Data.** The full dataset (~900 MB) is not in git. Download `tracks.csv` from
[Almost a million Spotify tracks on Zenodo](https://doi.org/10.5281/zenodo.11453410)
(CC BY 4.0) and put it at `data/raw/tracks.csv` (see [data/README.md](data/README.md)).
Then build the model once (~45 s):

```bash
music-rec build
```

Without the large file, add `--data data/raw/tracks_old.csv` to any command to
use the original 170k-track CMPT 310 dataset, which is in the repository.

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
music-rec similar "Love Train" --genre soul --flow
music-rec playlist --like "Take On Me" --like "Africa by Toto" --dislike "bad guy" -k 15
music-rec mood energy=0.9 danceability=0.8 tempo=124 year=2015
music-rec playlist --from-file my_playlist.txt -k 25   # Spotify links pasted into a file
```

**Use your Spotify playlist:** in the Spotify app, open a playlist, press
Ctrl+A then Ctrl+C, and paste into **Playlist builder → Paste from Spotify…**
in the desktop app. See [docs/USAGE.md](docs/USAGE.md#importing-a-playlist-from-spotify).

Or Python:

```python
from music_recommender import Recommender, Filters

rec = Recommender.load("artifacts/model.joblib")     # or Recommender.from_csv()
rec.similar_to("Bohemian Rhapsody", k=10)
rec.recommend(["Africa by Toto", "Billie Jean"], disliked=["bad guy"],
              k=20, filters=Filters(year_min=1975, year_max=1995), diversity=0.3, order="flow")
```

Example: `music-rec similar "Love Train" -k 6`

```
Seed: Love Train – The O'Jays (1972) [classic soul, disco, funk, philly soul, quiet storm, soul, southern soul]

 rank                                          name            artists  year                                  genres  popularity  similarity                               why
    1                          Message In Our Music         The O'Jays  1976               classic soul, disco, funk          28       0.845 genre, artist sound, acousticness
    2                        Automatically Sunshine       The Supremes  1972 classic girl group, classic soul, disco          29       0.834     release era, energy, loudness
    3 Harvest for the World, Pt. 1 - Single Version The Isley Brothers  1976              classic soul, funk, motown          41       0.817   genre, energy, instrumentalness
    4                              Come Get To This        Marvin Gaye  1973          classic soul, motown, neo soul          59       0.810          genre, release era, mood
    5                                Sexual Healing        Marvin Gaye  1982          classic soul, motown, neo soul          72       0.721           genre, energy, loudness
    6                      You Haven't Done Nothin'      Stevie Wonder  1974                            motown, soul          49       0.749  genre, release era, danceability
```

For the same seed, the first remake (audio features only, on the CMPT 310 data)
suggested David Bowie, a folk duo and a country singer.

## How it works

```
seeds ─► feature space (audio + genre + artist sound + era …) ─► exact k-NN candidates
      ─► filters ─► relevance (hybrid) − dislike penalty + popularity prior − alt-version penalty
      ─► version dedupe ─► MMR diversity + artist caps ─► optional flow order ─► playlist with "why"
```

Details and formulas: [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Documentation

| Document | Contents |
|---|---|
| [MODEL.md](MODEL.md) | **The model in one place**: every feature, weight, hyperparameter and test result |
| [docs/USAGE.md](docs/USAGE.md) | Installation, app guide, CLI reference, Python API, config files |
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | Feature engineering (genre, artist sound), scoring formulas, re-ranking, flow, explanations |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout, design decisions, how to extend |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Protocols, results, ablations, weight sweeps, sensitivity, limitations |
| [docs/MIGRATION.md](docs/MIGRATION.md) | What changed from CMPT 310: file mapping, bugs fixed, upgrades |
| [data/README.md](data/README.md) | Dataset card, artist recovery and cleaning steps |

## Project layout

```
src/music_recommender/   engine package (config, data, features, catalog, ranking, recommender, legacy, cli)
src/music_recommender/evaluation/   metrics, protocols, benchmarks, tuning
src/music_recommender/gui.py   PyQt6 desktop app (`music-rec-gui`)
app/streamlit_app.py     browser-based test bench
tests/                   pytest suite
data/raw/tracks.csv      ~900k-row v2 Spotify dataset (git-ignored)
data/raw/tracks_old.csv  170k-track CMPT 310 dataset
data/legacy/             original CMPT 310 results
configs/default.json     default engine configuration
reports/                 generated evaluation results
docs/                    documentation
```

## Development

```bash
pytest -m "not slow"        # fast suite (~5 s)
pytest                      # + full-dataset and Streamlit tests (~1 min, needs the v2 file)
ruff check src tests app && ruff format --check src tests app
music-rec evaluate          # regenerate reports/benchmark.csv
```

## Credits

- Original CMPT 310 Group 2 project: Cyrus Sandhu, Ramtin Rezaei, Jaskamal
  Chouhan, Varsha Adusumalli
- Dataset (v2, default): Oleg Fostenko (2024). *Almost a million Spotify
  tracks* (v1) [Data set]. Zenodo.
  [doi:10.5281/zenodo.11453410](https://doi.org/10.5281/zenodo.11453410),
  licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used
  unmodified; cleaning happens at load time.
- Dataset (v1): "Spotify Dataset 1921–2020, 160k+ Tracks" (Kaggle), from the
  original CMPT 310 project
- Remake: Ramtin Rezaei. MIT License (see [LICENSE](LICENSE)).
