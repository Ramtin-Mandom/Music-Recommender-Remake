# The recommendation algorithm

This document describes exactly what the engine computes. All formulas map
one-to-one onto code in `src/music_recommender/` (file names in brackets).

## 1. Overview

The engine is a **content-based k-nearest-neighbour recommender**. A track is
a point in a weighted feature space; tracks close to what the user likes are
recommended. That is the same core idea as the CMPT 310 model. The remake changes
how the space is built, how several seeds are combined, and what happens
after the neighbours are found.

```
             ┌──────────── fit (once, ~2 s) ────────────┐
tracks.csv → clean → FeaturePipeline → embedding X (n × 15) → NearestNeighbors index
                                                           → σ calibration
             └──────────────────────────────────────────┘

request → resolve seeds → candidate generation (k-NN) → filters → scoring
        → version dedupe → MMR + artist cap → ranked list + "why"
```

## 2. Feature space  [`features.py`]

### 2.1 Feature groups

| Group | Source | Transform | Default weight |
|---|---|---|---|
| danceability, energy, valence, acousticness, instrumentalness, liveness, speechiness | Spotify descriptors (0–1) | scaler | 1.2, 1.4, 1.0, 1.3, 0.9, 0.4, 1.0 |
| tempo | BPM | clip to [40, 220], then scaler | 0.8 |
| loudness | dB | scaler | 0.9 |
| era | `year` | scaler | 1.5 |
| duration | `duration_ms` | log, clip to [30 s, 20 min], then scaler | 0.35 |
| key | `key` + `mode` | circle of fifths → (sin, cos) | 0.15 |
| mode | `mode` | raw 0/1 | 0.15 |
| explicit | `explicit` | raw 0/1 | 0.5 |

The original model used only the first eight rows (no loudness, era,
duration, key, mode or explicit).

### 2.2 Scaling

Continuous groups are scaled together by one of three scalers
(`EngineConfig.scaling`):

* `quantile` (**default**): `QuantileTransformer(output_distribution="normal")`.
  Each feature is mapped through its empirical CDF to a standard normal.
  This matters because speechiness, liveness and instrumentalness are heavily
  skewed (most tracks are near 0). With a standard scaler, a few spoken-word
  or live tracks sit many standard deviations out and distort every distance.
* `standard`: z-scores (what the original used).
* `robust`: median / IQR.

### 2.3 Circle-of-fifths key encoding

Pitch classes are not ordinal: key 11 (B) is harmonically far from key 10
(A♯) but close to key 4 (E). Each key is placed on the circle of fifths, with
minor keys moved to their relative major (A minor and C major share a
position):

```
major_equiv = key                if mode == major
            = (key + 3) mod 12   if mode == minor
position    = (7 · major_equiv) mod 12
θ           = 2π · position / 12      →   (sin θ, cos θ)
```

Squared distance between two keys is `2 − 2 cos Δθ`. Fifths (C↔G) are close
and tritones (C↔F♯) are far. The expected value for a random pair is 2,
the same as for a standardised 1-D feature, so the key group needs no
extra scaling.

### 2.4 Weights

After scaling, column *j* of group *g* is multiplied by `w_g`. With Euclidean
distance, group *g* therefore contributes `w_g² · Δ²` to the squared distance.
A weight of 0 removes the group from the space entirely. The defaults were set
by hand and checked with an era-weight sweep and a one-at-a-time sensitivity
analysis (see [EVALUATION.md](EVALUATION.md)).

## 3. Similarity calibration  [`recommender.py`, `ranking.py`]

Raw distances are hard to read, and the original reported `1 − distance`,
which goes negative for Euclidean distances above 1. The remake uses a
Gaussian kernel calibrated on the data:

```
σ          = median over 400 random tracks of the distance to their 50th nearest neighbour
similarity = exp(−ln 2 · (d / σ)²)          ∈ (0, 1]
```

So a track exactly as far away as a typical 50th neighbour scores 0.5, and
identical tracks score 1.

## 4. Seeds and candidate generation

Seeds are resolved to catalogue rows ([catalog.py](../src/music_recommender/catalog.py)).
Exact titles win, version suffixes are ignored, ties go to the most popular
release, and fuzzy matching handles typos. Each seed `s_i` can carry a weight
`a_i` (default 1). The weighted centroid is `c = Σ a_i s_i / Σ a_i`.

Candidates are the union of the k-NN of the centroid and the k-NN of each seed
(`candidate_pool`, default 300, split across queries). Seeds, disliked tracks,
and **every version of a seed song** are removed, then hard `Filters` are
applied (year range, minimum popularity, explicit, duration, banned artists).
If too few candidates survive, the pool grows ×4 until it is big enough or
the whole catalogue has been searched. The same happens when the per-artist
cap later leaves fewer than *k* results.

## 5. Scoring

For each candidate *x*:

```
sim_centroid(x) = K(d(x, c))
sim_nearest(x)  = max_i  (a_i / max a) · K(d(x, s_i))

relevance(x) =  sim_centroid                              strategy = "centroid"
                sim_nearest                               strategy = "nearest_seed"
                α·sim_centroid + (1−α)·sim_nearest        strategy = "hybrid" (default, α = 0.5)

relevance(x) −= λ_dislike · max_j K(d(x, n_j))            for disliked tracks n_j (λ = 0.6)

score(x) = (1 − β) · relevance(x) + β · popularity(x)/100   (β = 0.08)
```

Why combine strategies? The centroid of a mixed playlist (say, a ballad and a
dance track) can land in an empty region of space that sounds like neither.
`nearest_seed` respects each seed but can ignore all but one of them.
`hybrid` rewards tracks that fit the overall taste *and* closely match at
least one specific seed.

The dislike term is a Rocchio-style negative-feedback penalty. The popularity
prior is a small nudge so that, among near-identical sounding tracks, the one
people actually listen to wins. It also reduces the zero-play tracks
(popularity 0) that pure k-NN surfaces.

## 6. Re-ranking  [`ranking.py`]

1. **Version dedupe**: only the best-scoring track per `version_key` is kept
   (no "Song" + "Song – Remastered" + "Song – Live").
2. **Maximal Marginal Relevance**: picks the list greedily:

   ```
   next = argmax_x  (1 − δ) · score(x) − δ · max_{y ∈ selected} K(d(x, y))
   ```

   `δ = diversity` (default 0.15). δ = 0 returns the plain top-k; higher
   values trade a little relevance for a list that does not repeat one sound.
3. **Artist cap**: at most `max_per_artist` (default 2) tracks per primary
   artist.

## 7. Mood / target search

`Recommender.from_targets({"energy": 0.9, "tempo": 128, "year": 1985})`
embeds a partial description: unspecified features are filled with medians
and **masked out** of the distance. Distance is computed only over the
constrained columns and rescaled by `sqrt(total_cols / constrained_cols)`, so
the σ calibration still roughly applies. The same scoring and re-ranking
then follow.

## 8. Explanations

* **`why` column**: the three feature groups where the recommendation is
  *unusually* close to its nearest anchor (centroid or seed). Per-group gaps
  are divided by the median gap between random track pairs, so a perfect match
  on a feature where everything is similar counts for less. Ratios are
  floored at 0.1, which keeps exact ties (e.g. two tracks with zero
  instrumentalness) from always winning, and groups with less than 35 % of the
  largest weight are skipped.
* **`explain(seeds, row)`**: per-group share of the squared distance to the
  seed centroid (what makes the track *different*), plus a 0–1 closeness per
  group.

## 9. Complexity

| Step | Cost |
|---|---|
| Fit | O(n log n) for the quantile transform and tree index; ~2 s for 154k tracks |
| Candidate generation | tree k-NN, ~1–5 ms per query vector |
| Scoring + MMR | O(P · m) + O(k · P) for pool size P and m seeds |
| End to end | ~20–60 ms per request on a laptop |

## 10. Configuration reference

Every knob above lives in `EngineConfig` (`config.py`). Configs serialise to
JSON (`EngineConfig.save/load`, `music-rec --config my.json …`), and
`with_overrides(...)` / per-call keyword overrides make experiments cheap.
Changing only re-ranking knobs (`diversity`, `strategy`, …) reuses the fitted
index; changing `weights`, `scaling` or `metric` refits.
