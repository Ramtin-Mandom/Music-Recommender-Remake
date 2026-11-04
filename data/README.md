# Data

## `raw/tracks.csv` — Spotify tracks dataset

The catalogue the engine recommends from. It is the `data.csv` file from the
original CMPT 310 Group 2 project, moved here unchanged: the Kaggle **"Spotify
Dataset 1921–2020, 160k+ Tracks"**, ~29.7 MB.

| | |
|---|---|
| Rows | 170,653 tracks (154,330 after cleaning) |
| Years | 1921 – 2020 |
| Source | Spotify Web API audio features, collected by a third party (published on Kaggle) |

### Columns

| Column | Type | Meaning |
|---|---|---|
| `id` | str | Spotify track id (22 chars) |
| `name` | str | Track title |
| `artists` | str | Python-literal list, e.g. `"['Queen', 'David Bowie']"` |
| `year`, `release_date` | int / str | Release year / date |
| `popularity` | 0–100 | Spotify popularity at collection time |
| `duration_ms` | int | Length in milliseconds |
| `explicit` | 0/1 | Explicit-lyrics flag |
| `key` | 0–11 | Pitch class (0 = C, 1 = C♯, …) |
| `mode` | 0/1 | Minor / major |
| `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `liveness`, `speechiness` | 0–1 | Spotify audio descriptors |
| `tempo` | BPM | Estimated tempo |
| `loudness` | dB | Mean loudness (typically −60 … 0) |

### Cleaning (`music_recommender.data.clean_tracks`)

1. Drop rows with missing features, a missing title, or no artist.
2. Drop 143 rows with `tempo < 30` BPM (tempo 0 marks a failed analysis).
3. Clip impossible values (unit features to [0, 1], loudness ≤ 0 dB).
4. Parse the artist list and add helper columns: `primary_artist`,
   `artists_display`, `title_norm`, `artist_norm`, `version_key`.
5. Remove exact duplicates (same normalised title **and** artists), keeping
   the most popular copy (~16k rows).

`version_key` = base title (version suffixes such as "- Remastered 2011" or
"(Live)" removed) + primary artist. It lets the engine treat a remaster and the
original as the *same song*, so they never fill a playlist together.

### Known limitations

* No genre labels and no listening histories, so the engine is purely
  content-based (see `docs/EVALUATION.md` for how it is evaluated anyway).
* Coverage is uneven: ~2,000 tracks per year from the 1950s on, far fewer
  before. Very recent music (after 2020) is absent.
* A few titles contain mojibake from the original scrape.

## `legacy/` — results carried over from CMPT 310

| File | Content |
|---|---|
| `cv_optimization_results.csv` | The original 5-fold cross-validation grid (k ∈ {5,10,15,20} × {euclidean, cosine}) scored by tempo/energy/loudness bucket agreement. Best: k=5, euclidean. |
| `user_study_scores.csv` | The original informal user study: 10 participants rated single-track and playlist recommendations out of 10 (mean 8.0 and 7.8). **Participant names were replaced by ids (P01–P10).** |
