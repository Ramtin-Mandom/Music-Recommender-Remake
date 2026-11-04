# From CMPT 310 Group 2 to the Remake

This project is a rewrite of the CMPT 310 (Fall 2025) Group 2 final project, a
k-NN playlist recommender over Spotify audio features. This page maps the
original code onto the new one and lists what changed and why.

## File mapping

| CMPT 310 | Remake | Notes |
|---|---|---|
| `Data/data.csv` | `data/raw/tracks.csv` | Moved unchanged |
| `Data/optimization_result.csv` | `data/legacy/cv_optimization_results.csv` | Kept for reference |
| `Data/Final_Test_Data.csv` | `data/legacy/user_study_scores.csv` | Participant names replaced by ids |
| `Src/KNNModel.py` | `data.py`, `features.py`, `catalog.py`, `recommender.py`, `ranking.py` | Split by responsibility |
| model settings in `KNNModel.py` | `legacy.LEGACY_CONFIG` | Exact preset, used as the benchmark baseline |
| `Src/Evaluation.py` | `evaluation/` | Old metric kept, new protocols added |
| `Src/GUI.py` (PyQt6) | `app/streamlit_app.py` | Web test bench with comparison, explain and eval tabs |
| `Src/testApp.py` | `cli.py` (`music-rec`) | Full CLI with sub-commands |
| — | `tests/` | 67 automated tests (none before) |

## Bugs fixed

| # | Original behaviour | Effect | Fix |
|---|---|---|---|
| 1 | `NearestNeighbors(n_neighbors=5)` but asked for `top_k=10` / `15` | **Never returned more than 4–5 tracks**, whatever the user asked for | Candidate pool decoupled from *k* and grown on demand |
| 2 | `similarity = 1 − euclidean_distance` | Negative or meaningless "similarity" scores | Kernel similarity calibrated on the data, in (0, 1] |
| 3 | Lookup by substring, **first match taken** | A common title resolved to whichever row came first in the file, often an obscure cover or re-release | Match-quality scoring (exact > prefix > contains > fuzzy) with popularity tie-break |
| 4 | Liked-track exclusion keyed on `(title, artist or "")` vs the raw `"['Artist']"` string | Seeds could be recommended back to the user | Exclusion by catalogue row **and** by song version |
| 5 | Remasters / live versions treated as different songs | "Song", "Song – Remastered" and "Song – Live" filled one list | `version_key` dedupe |
| 6 | Artist cap on the raw artists string | Features ("A, B") bypassed the cap for A | Cap on the parsed primary artist |
| 7 | Tempo 0 rows kept | Broken analyses sat in a corner of the space | Rows with tempo < 30 BPM dropped |
| 8 | Evaluation labels built from tempo/energy, which are model inputs | Circular metric (high scores by construction) | Same-artist retrieval protocol on held-out artists |
| 9 | GUI "sometimes needs to be run twice" (per the old README) | Unreliable start-up | The app loads the engine once through a cached resource (or a pre-built model file) |

## Modelling upgrades

| Area | CMPT 310 | Remake |
|---|---|---|
| Features | 8 audio features | + loudness, release era, duration, key (circle of fifths), mode, explicit |
| Scaling | StandardScaler | Quantile → normal (robust to skewed features); standard/robust selectable |
| Weights | tempo 2.5, energy 1.5, danceability 1.3 (hand-picked) | All 14 groups weighted and validated by sweep + sensitivity analysis |
| Multiple seeds | Plain average | Weighted seeds; centroid / nearest-seed / hybrid strategies |
| Negative feedback | — | Disliked tracks push similar candidates down |
| Ranking | Raw distance | Relevance + popularity prior |
| Diversity | Artist cap | Version dedupe + MMR + artist cap |
| Filters | Year, popularity, explicit (CLI only, applied after k-NN, so they could empty the list) | Year range, popularity, explicit, duration, artists; pool grows until *k* survive |
| Mood search | — | Recommend from target attribute values |
| Explanations | — | `why` column + per-feature breakdown |
| Persistence | Refit on every start | `music-rec build` → instant start |

## Results

See [EVALUATION.md](EVALUATION.md) for the full numbers. In short, on held-out
artists the remake ranks other songs by the seed's artist **~5× more often** than
the original model, keeps recommendations in the seed's era (mean gap ~4
years vs ~17), and still returns more varied lists.
