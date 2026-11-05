# Data

## `raw/tracks.csv`: Spotify tracks dataset (v2, default)

The catalogue the engine recommends from: **"Almost a million Spotify
tracks"** by Oleg Fostenko (Lomonosov Moscow State University), published on
Zenodo on 3 June 2024, sampled from the Spotify API.

* Download: <https://doi.org/10.5281/zenodo.11453410> (file `tracks.csv`, 931 MB)
* License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse is
  allowed with attribution. The file is used unmodified, and all cleaning
  happens in code at load time.
* Cite as: Fostenko, O. (2024). *Almost a million Spotify tracks* (Version v1)
  [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11453410

It is about **900 MB**, so it is **not in git** (it's in `.gitignore`). Put the
file at `data/raw/tracks.csv`, or point the engine somewhere else with
`--data PATH` / `Recommender.from_csv(path)`.

| | |
|---|---|
| Rows | 899,702 tracks (**558,328 after cleaning**) |
| Years | 1900 – 2024 (58% from 2015 on, 2% before 1970) |
| Artists | 43,705 after cleaning; 71% of them have Spotify genre tags (4,890 distinct tags) |
| Source | Spotify Web API track, artist and audio-feature data, plus a small number of chart rows ([Zenodo record](https://doi.org/10.5281/zenodo.11453410)) |

### Columns used

| Column | Type | Meaning |
|---|---|---|
| `track_id` | str | Spotify track id (22 chars) |
| `name` | str | Track title |
| `track_artists` | str | Primary artist name. **Only filled for ~11% of rows** (see below) |
| `genres` | str | The *artist's* Spotify genre tags, e.g. `"['indie pop', 'la indie']"` |
| `artist_followers`, `artist_popularity` | float | The artist's follower count and popularity (0–100) |
| `album_release_date` | str | `YYYY`, `YYYY-MM` or `YYYY-MM-DD`; the year is used |
| `popularity` | 0–100 | Track popularity (21% of tracks are 0) |
| `explicit` | bool | Explicit-lyrics flag |
| `key` | 0–11 | Pitch class (0 = C, 1 = C♯, …) |
| `mode` | 0/1 | Minor / major |
| `time_signature` | int | Beats per bar (85% are 4, 12% are 3) |
| `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `liveness`, `speechiness` | 0–1 | Spotify audio descriptors |
| `tempo` | BPM | Estimated tempo |
| `loudness` | dB | Mean loudness (typically −60 … 0) |
| `duration_ms` | int | Track length. **Missing for 99% of rows**, so the duration feature is switched off automatically. |

The file also has chart columns (`chart`, `region`, `rank`, `streams`,
`trend`, < 1% filled), `available_markets`, `added_at`, `album_name` and album
metadata. The engine doesn't read these (`available_markets` alone is most of
the file's size).

### Recovering artist names

Only 99,943 rows name their artist, but every row carries the artist's
`genres`, `artist_followers` and `artist_popularity`. Follower counts are exact
integers, so the three together identify an artist almost uniquely (only 2.4%
of keys map to more than one credited name). They form the `artist_key`, and
the name is copied from whichever of that artist's tracks is credited. This
recovers a name for 75% of the catalogue. Tracks whose artist is never named
anywhere (229k rows) are dropped, because a recommendation without an artist
isn't useful. Set `require_artist=False` in the config to keep them as
"Unknown artist".

Featured artists are parsed from titles (`"Low (feat. T-Pain)"`, 19.5k tracks
after cleaning) and added to `artist_list`. The evaluation uses them as a
collaboration graph.

### Cleaning (`music_recommender.data.clean_tracks`)

| Step | Rows removed |
|---|---|
| Missing audio features, title, key or year | 965 |
| `tempo < 30` BPM (tempo 0 marks a failed analysis) | 529 |
| Year before 1900 (placeholder dates such as `0000`) | 26 |
| Artist never named | 229,307 |
| Duplicates (same normalised title **and** artists; the most popular copy is kept) | 110,547 |

Cleaning also clips impossible values and adds helper columns:
`primary_artist`, `artists_display`, `genres_display`, `title_norm`,
`artist_norm`, `version_key` and `alt_version`.

* `version_key` = base title (version suffixes such as "- Remastered 2011" or
  "(Live)" removed) + primary artist. It lets the engine treat a remaster and
  the original as the *same song*, so they never fill a playlist together.
* `alt_version` marks live, demo/alternate-take, instrumental, acoustic,
  karaoke, sped-up, slowed, 8D and holiday tracks. These are ranked down unless
  a seed is the same kind.

### Known limitations

* Genres belong to the **artist**, not the track, and 29% of artists have none.
  Those artists borrow genre information from artists with a similar overall
  sound.
* Popularity was captured at a different time from the audio features. A
  fifth of tracks show popularity 0, so the engine blends in artist popularity.
* Strong skew towards recent music (58% from 2015 on).
* There are still no listening histories, so the engine is content-based (see
  `docs/EVALUATION.md` for how it is evaluated anyway).

## `raw/tracks_old.csv`: the CMPT 310 dataset (v1)

The original project's `data.csv` (Kaggle "Spotify Dataset 1921–2020, 160k+
Tracks", 29.7 MB, 170,653 rows, 154,330 after cleaning). It stays in the
repository so the engine still runs without the large file:

```bash
music-rec --data data/raw/tracks_old.csv similar "Bohemian Rhapsody"
```

The loader detects the layout from the columns. The v1 file has no genres,
artist popularity or time signature, so the `genre` group is switched off
automatically (`artist_sound` still works, keyed by artist name).

## `legacy/`: results carried over from CMPT 310

| File | Content |
|---|---|
| `cv_optimization_results.csv` | The original 5-fold cross-validation grid (k ∈ {5,10,15,20} × {euclidean, cosine}) scored by tempo/energy/loudness bucket agreement. Best: k=5, euclidean. |
| `user_study_scores.csv` | The original informal user study: 10 participants rated single-track and playlist recommendations out of 10 (mean 8.0 and 7.8). **Participant names were replaced by ids (P01–P10).** |
