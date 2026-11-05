# From CMPT 310 Group 2 to the Remake

This project is a rewrite of the CMPT 310 (Fall 2025) Group 2 final project, a
k-NN playlist recommender over Spotify audio features. This page maps the
original code onto the new one and lists what changed and why.

## File mapping

| CMPT 310 | Remake | Notes |
|---|---|---|
| `Data/data.csv` | `data/raw/tracks_old.csv` | Moved unchanged; still supported (v1 layout). The default catalogue is now the larger v2 dataset at `data/raw/tracks.csv` |
| `Data/optimization_result.csv` | `data/legacy/cv_optimization_results.csv` | Kept for reference |
| `Data/Final_Test_Data.csv` | `data/legacy/user_study_scores.csv` | Participant names replaced by ids |
| `Src/KNNModel.py` | `data.py`, `features.py`, `catalog.py`, `recommender.py`, `ranking.py` | Split by responsibility |
| model settings in `KNNModel.py` | `legacy.LEGACY_CONFIG` | Exact preset, used as the benchmark baseline |
| `Src/Evaluation.py` | `evaluation/` | Old metric kept, new protocols added |
| `Src/GUI.py` (PyQt6) | `gui.py` (`music-rec-gui`) + `app/streamlit_app.py` | Rebuilt PyQt6 desktop app on the new engine, plus a web test bench with evaluation tabs |
| `Src/testApp.py` | `cli.py` (`music-rec`) | Full CLI with sub-commands |
| — | `tests/` | 119 automated tests (none before) |

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
| 8 | Evaluation labels built from tempo/energy, which are model inputs | Circular metric (high scores by construction) | Collaborator retrieval on held-out artists (featured credits, never seen by the model) |
| 9 | GUI "sometimes needs to be run twice" (per the old README) | Unreliable start-up | The app loads the engine once through a cached resource (or a pre-built model file) |

## Modelling upgrades

| Area | CMPT 310 | Remake |
|---|---|---|
| Data | 170k tracks, no genres | 558k tracks (v2) with artist genre tags; artist names recovered for unnamed rows |
| Features | 8 audio features | + loudness, release era, key (circle of fifths), mode, time signature, explicit, **genre embedding**, **artist sound** |
| Scaling | StandardScaler | Quantile → normal (robust to skewed features); standard/robust selectable |
| Weights | tempo 2.5, energy 1.5, danceability 1.3 (hand-picked) | All groups weighted; genre / artist sound / era chosen by sweeps, everything checked by sensitivity analysis |
| Multiple seeds | Plain average | Weighted seeds; centroid / nearest-seed / hybrid strategies |
| Negative feedback | — | Disliked tracks push similar candidates down |
| Ranking | Raw distance | Relevance + track/artist popularity prior − penalty for live/demo/karaoke/sped-up/holiday versions |
| Diversity | Artist cap | Version dedupe + MMR + artist cap + a stricter cap on the seeds' own artists |
| Order | By distance | By score, or a smooth tempo / energy / key **flow** |
| Filters | Year, popularity, explicit (CLI only, applied after k-NN, so they could empty the list) | Year range, popularity, explicit, duration, artists, genre; pool grows until *k* survive |
| Mood search | — | Recommend from target attribute values |
| Explanations | — | `why` column + per-feature breakdown |
| Persistence | Refit on every start | `music-rec build` → ~7 s start instead of ~35 s |

## Results

See [EVALUATION.md](EVALUATION.md) for the full numbers. In short, on held-out
artists the remake finds the seed artist's collaborators **~17× more often**
than the original model. 58 % of its recommendations share a genre with the
seed (7 % for the original), they stay in the seed's era (mean gap ~3 years vs
~10), and alternate versions are 6× rarer.
