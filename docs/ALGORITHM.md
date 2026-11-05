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
             ┌──────────────── fit (once, ~8 s for 558k tracks) ────────────────┐
tracks.csv → load (v1/v2) → recover artists → clean → FeaturePipeline → X (n × 40, float32)
                                                                     → σ calibration
             └──────────────────────────────────────────────────────────────────┘

request → resolve seeds → candidate generation (exact k-NN) → filters → scoring
        → version penalty + dedupe → MMR + artist caps → optional flow order → list + "why"
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
| meter | `time_signature` | 1 if not 4/4, else 0 | 0.15 |
| **genre** | artist's genre tags | TF-IDF over tags + their words → 16-d SVD → unit length (§2.4) | 3.0 |
| **artist_sound** | artist's other tracks | leave-one-out mean of the scaled audio vector, shrunk (§2.5) | 1.5 |

The original model used only the first eight audio rows. Groups the dataset
cannot support are dropped automatically at fit time and listed in
`pipeline.dropped`: `duration` on the v2 data (99% of durations are missing)
and `genre` on the v1 data (no genres).

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

### 2.4 Genre embedding

Spotify genres belong to the **artist** and are very fine-grained (4,890 tags
such as "la indie", "pov: indie", "philly soul"). Exact tag overlap would make
"philly soul" and "classic soul" strangers, so each artist's tags become a
TF-IDF document over **whole tags and their individual words**
(`"philly soul"` → `philly soul`, `w:philly`, `w:soul`). The vectorizer is fitted
on one document per *artist*, so prolific artists don't dominate the
vocabulary. A truncated SVD compresses it to `genre_dims` = 16 dimensions, and
each vector is scaled to unit length.

29% of artists have no tags. They borrow the **mean genre vector of the 10
tagged artists whose average sound is closest** (k-NN over per-artist mean
audio vectors). That mean isn't re-normalised: when the neighbours disagree,
the vector is short and pulls less, which reflects the uncertainty.

### 2.5 Artist sound

For track *t* by artist *A* with *n* tracks in the catalogue, and scaled audio
vectors `a(·)`:

```
loo(t)   = mean of a(u) over A's other tracks u ≠ t        (leave-one-out)
λ        = (n − 1) / (n − 1 + κ)                           κ = artist_shrinkage = 3
sound(t) = λ · loo(t) + (1 − λ) · a(t)
```

Inside the group, each audio column keeps its relative audio weight (rescaled
to unit root-mean-square), and the whole group is then multiplied by
`w_artist_sound`. Two tracks by artists with similar catalogues end up close
even when the tracks themselves differ, which acts as a genre signal that needs
no tags. Leave-one-out stops a track from matching itself through its artist,
and shrinkage stops a one-hit artist's "average" from being pure noise.

### 2.6 Weights

After scaling, column *j* of group *g* is multiplied by `w_g`. With Euclidean
distance, group *g* therefore contributes `w_g² · Δ²` to the squared distance.
A weight of 0 removes the group from the space entirely. The audio weights were
set by hand on the v1 data. `genre`, `artist_sound` and `era` were chosen by
weight sweeps on the v2 validation artists, and everything was checked with a
one-at-a-time sensitivity analysis (see [EVALUATION.md](EVALUATION.md)). A
group's share of a typical distance depends on its width as well as its weight:
the 16 genre columns carry a unit vector, so a random pair differs by about
√2 · w there.

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
(`candidate_pool`, default 300, split across queries). The search is **exact
brute force**: one matrix product `‖x‖² − 2 X q + ‖q‖²` over the float32
embedding plus `argpartition`, about 10 ms per query vector for 558k tracks
(sklearn re-validates the whole matrix on every call, which costs more than
the search itself). Seeds, disliked tracks, and **every version of a seed
song** are removed, then hard `Filters` are applied (year range, minimum
popularity, explicit, duration, banned artists, genre substring).
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

prior(x) = (1 − ρ) · track_popularity(x) + ρ · artist_popularity(x)     (ρ = 0.3, both / 100)
score(x) = (1 − β) · relevance(x) + β · prior(x)                         (β = 0.08)
score(x) −= π   if x is an alternate version the seeds are not           (π = 0.15)
```

Why combine strategies? The centroid of a mixed playlist (say, a ballad and a
dance track) can land in an empty region of space that sounds like neither.
`nearest_seed` respects each seed but can ignore all but one of them.
`hybrid` rewards tracks that fit the overall taste *and* closely match at
least one specific seed.

The dislike term is a Rocchio-style negative-feedback penalty. The popularity
prior is a small nudge so that, among near-identical sounding tracks, the one
people actually listen to wins. A fifth of v2 tracks have popularity 0, so 30%
of the prior comes from the artist's popularity.

**Alternate versions.** `alt_version` (from the title) marks live, demo /
alternate take, instrumental, acoustic, karaoke, sped-up, slowed, 8D and
holiday tracks. Such a candidate loses π unless one of the seeds is the same
kind, so a live seed still gets live recommendations.

## 6. Re-ranking  [`ranking.py`]

1. **Version dedupe**: only the best-scoring track per `version_key` is kept
   (no "Song" + "Song – Remastered" + "Song – Live").
2. **Maximal Marginal Relevance**: picks the list greedily:

   ```
   next = argmax_x  (1 − δ) · score(x) − δ · max_{y ∈ selected} K(d(x, y))
   ```

   `δ = diversity` (default 0.15). δ = 0 returns the plain top-k; higher
   values trade a little relevance for a list that does not repeat one sound.
3. **Artist caps**: at most `max_per_artist` (default 2) tracks per artist
   (by `artist_key`), and at most `max_seed_artist` (default 1) by the seeds'
   own artists. With genre and artist sound in the space, a seed's own artist is
   always very close, so without the second cap radio turns into a discography.
4. **Flow order** (`order="flow"`, off by default): the chosen tracks are
   re-ordered into a smooth sequence. The transition cost between two tracks is

   ```
   tempo:  |log2(t₂/t₁)|, with half/double time counted as a match, / 0.08
   energy: |e₂ − e₁| / 0.15
   key:    circle-of-fifths angle / π / 0.5
   ```

   A greedy walk from the top-ranked track is followed by 2-opt improvement on
   the open path. The best match still opens the list, and the order changes
   but the set of tracks doesn't.

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
  instrumentalness) from always winning. Groups weighted below 0.5 are
  skipped, and a binary flag (explicit) is only mentioned when both tracks
  share its *rare* value.
* **`explain(seeds, row)`**: per-group share of the squared distance to the
  seed centroid (what makes the track *different*), plus a 0–1 closeness per
  group.

## 9. Complexity

| Step | Cost |
|---|---|
| Load + clean CSV | ~25 s for the 900 MB v2 file (once; `music-rec build` caches the result) |
| Fit | quantile transform, genre TF-IDF/SVD, artist statistics: ~8 s for 558k tracks |
| Candidate generation | exact brute force, ~10 ms per query vector |
| Scoring + MMR | O(P · m) + O(k · P) for pool size P and m seeds |
| End to end | ~25–40 ms per request on a laptop (plus ~100 ms if a title has to be searched) |

## 10. Configuration reference

Every knob above lives in `EngineConfig` (`config.py`). Configs serialise to
JSON (`EngineConfig.save/load`, `music-rec --config my.json …`), and
`with_overrides(...)` / per-call keyword overrides make experiments cheap.
Changing only re-ranking knobs (`diversity`, `strategy`, `order`, …) reuses
the fitted space. Changing `weights`, `scaling`, `metric`, `genre_dims` or
`artist_shrinkage` refits.
