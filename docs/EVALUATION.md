# Evaluation

All numbers below can be regenerated:

```bash
music-rec evaluate --queries 500    # reports/benchmark.csv        (test artists)
music-rec analyze                   # reports/era_sweep.csv,
                                    # reports/feature_sensitivity.csv (validation artists)
```

## 1. Why a new evaluation?

The CMPT 310 evaluation bucketed every track by tempo (slow/medium/fast),
energy (low/medium/high) and loudness, then measured how often a track's k
nearest neighbours fell in the same bucket ("agreement@k"). It chose k = 5 and
Euclidean distance (combined agreement 0.83).

That metric is **partly circular**: tempo and energy are inputs to the model,
so neighbours in that space share tempo and energy buckets by construction. A
model that used *only* tempo and energy would score close to 1.0 and still make
poor recommendations. The metric is kept below for continuity, but it cannot
tell good models from bad ones.

## 2. Protocols

The dataset has no listening histories or genres, so we need a proxy for "a
listener who likes X would also like Y". We use **artists**:

> Tracks by the same artist tend to share a style. Artist identity is **not** a
> model feature. If the feature space captures style, a seed track's
> neighbourhood should contain the artist's *other* songs more often than chance.

### 2.1 Same-artist retrieval (accuracy)

* Eligible artists have ≥ 6 distinct songs (versions merged). They are shuffled
  with a fixed seed and **split 50/50 into validation and test**. Tuning and
  analysis use validation artists only; the headline numbers use test artists.
* **Radio task**: 1 random song by the artist is the seed. **Playlist task**:
  3 random songs are the seeds.
* Every version of the seed songs is excluded, so a remaster cannot count as a
  hit. A song counts once, however many of its versions appear.
* Each engine returns its top 10 with the artist cap and MMR **disabled**, so
  we measure the ranking signal rather than deliberate diversification.
* Metrics: **precision@10** (share of the list that is another song by the
  artist), **hit@10** (at least one), **nDCG@10**. *Objective* = mean of radio
  and playlist precision@10.

Absolute values are small by nature: the catalogue has 154k tracks and the
typical artist has only a handful of songs in it. Random ranking would score
around 10⁻⁴. The numbers are best read **relative** to each other.

### 2.2 Beyond accuracy (production settings)

Using each engine's real re-ranking (cap, MMR, dedupe), over the same queries:

| Metric | Meaning |
|---|---|
| artist_diversity | unique artists / list length |
| intra_list_diversity | mean pairwise distance in a **fixed** reference space (z-scored raw audio features, identical for all engines) |
| mean_popularity | mean Spotify popularity of recommendations |
| year_gap | mean \|year − seed year\| |
| coverage | distinct tracks recommended / list slots |
| *_bucket_agreement | the original CMPT 310 metric (share of the list in the seed's tempo / energy / loudness bucket) |

## 3. Results: legacy vs remake (test artists, 500 queries per task)

| Engine | Radio P@10 | Radio hit@10 | Playlist P@10 | Playlist hit@10 | Objective | Year gap | ILD |
|---|---|---|---|---|---|---|---|
| **Legacy (CMPT 310 model)** | 0.0060 | 0.050 | 0.0074 | 0.044 | 0.0067 | 17.1 y | 0.91 |
| **Remake (default)** | **0.0298** | **0.190** | **0.0370** | **0.210** | **0.0334** | **4.1 y** | **1.44** |
| Remake, no era feature | 0.0156 | 0.112 | 0.0198 | 0.132 | 0.0177 | 14.6 y | 1.37 |
| Remake, standard scaling | 0.0280 | 0.154 | 0.0324 | 0.182 | 0.0302 | 3.5 y | 1.12 |
| Remake, centroid only | 0.0298 | 0.190 | 0.0320 | 0.176 | 0.0309 | 4.0 y | 1.32 |
| Remake, no popularity prior | 0.0302 | 0.194 | 0.0394 | 0.214 | 0.0348 | 4.0 y | 1.42 |
| Remake, no MMR diversity | 0.0298 | 0.190 | 0.0370 | 0.210 | 0.0334 | 4.0 y | 1.32 |
| Remake, cosine metric | 0.0278 | 0.170 | 0.0338 | 0.202 | 0.0308 | 5.1 y | 1.56 |

*(Full table with nDCG, coverage, popularity and bucket agreement:
`reports/benchmark.csv`.)*

### Reading the table

* **5× better same-artist retrieval** (objective 0.0067 → 0.0334). In about
  1 in 5 radio queries, the remake surfaces another song by the seed's artist in
  the top 10, against 1 in 20 for the original.
* **Not only era.** With the era feature removed, the remake still scores
  **2.6×** the legacy model (0.0177), from quantile scaling, loudness,
  duration, key/mode and the re-balanced weights.
* **Quantile scaling helps**: +11 % objective over standard scaling, and much
  larger list diversity (ILD 1.44 vs 1.12) because a few extreme speechiness /
  liveness values no longer dominate.
* **The hybrid strategy helps multi-seed playlists**: playlist precision 0.037
  vs 0.032 with the plain centroid (+16 %). Radio is unaffected, since one seed
  has no centroid problem.
* **MMR costs no accuracy here** (it is disabled in the accuracy protocol by
  design) and raises intra-list diversity from 1.32 to 1.44.
* **The popularity prior costs a little accuracy** (0.0348 → 0.0334, within
  noise). It is kept on purpose: without it the engine surfaces more
  zero-popularity uploads, which do poorly in hands-on testing. Set
  `popularity_weight=0` to remove it.
* **Cosine is worse and 3× slower** (brute-force search), so Euclidean stays
  the default.

**Noise level.** With 500 queries × 10 slots per task, the standard error of a
precision around 0.03 is about ±0.0025. Differences smaller than ~0.005 (for
example standard vs quantile scaling on radio, or the popularity prior) should
be treated as ties. The legacy vs remake gap is roughly 10 standard errors.

### The original metric, for continuity

| Engine | Tempo agreement | Energy agreement | Loudness agreement |
|---|---|---|---|
| Legacy | **0.77** | **0.82** | 0.69 |
| Remake | 0.69 | 0.79 | **0.83** |

The legacy model wins on tempo because it gives tempo a weight of 2.5, which is
exactly the circularity described in §1. The remake wins on loudness, which the
legacy model did not use at all. Neither column says much about recommendation
quality.

## 4. The era trade-off

Release year is the most predictive single feature for the artist proxy,
because artists are active in a limited window. Sweeping its weight (validation
artists, 300 queries):

| era weight | Objective | Audio bucket agreement | Year gap | ILD |
|---|---|---|---|---|
| 0.0 | 0.0220 | 0.790 | 14.3 y | 1.41 |
| 0.5 | 0.0288 | 0.789 | 8.1 y | 1.42 |
| 1.0 | 0.0373 | 0.786 | 5.2 y | 1.44 |
| **1.5 (default)** | **0.0415** | **0.781** | **3.9 y** | **1.46** |
| 2.0 | 0.0443 | 0.778 | 3.2 y | 1.49 |
| 2.5 | 0.0465 | 0.777 | 2.8 y | 1.50 |
| 3.5 | 0.0508 | 0.770 | 2.2 y | 1.52 |

The objective keeps rising, but each step buys less, and the engine
increasingly becomes a "songs from the same year" recommender. Audio coherence
(bucket agreement) falls, and the window of eras shrinks below what a listener
would expect from "sounds like this". 1.5 is at the knee: about 80 % of the
gain of 2.5 (relative to no era at all), with a ±4-year window.

### What happened when the tuner was allowed to touch era

An unconstrained random search (60 trials, history in
`reports/tuning_unfrozen_history.csv`) pushed era to 2.4 and shrank almost every
audio weight (tempo and duration went to 0, liveness to 0.01). It reached a
validation objective of 0.053. On inspection:

* With era held equal, its audio weights **scored worse** than the hand-set
  ones (0.0385 vs 0.0415 at era = 1.5) and had much lower audio agreement
  (0.71 vs 0.78).
* So its "gain" was entirely "more era". The audio changes were
  hill-climbing on evaluation noise.

Because of this, `music-rec tune` **freezes era by default**
(`--unfreeze-era` to override) and only accepts candidates that improve by more
than 0.001. This is a good example of why a proxy objective needs guard
metrics.

## 5. Per-feature sensitivity (validation artists)

Each group is dropped (weight 0) or doubled, one at a time, from the defaults
(reference objective 0.0415):

| Group | Weight | Δ objective if dropped | Δ audio agreement if dropped | Δ objective if doubled |
|---|---|---|---|---|
| era | 1.50 | −0.0195 | +0.009 | +0.0070 |
| loudness | 0.90 | −0.0060 | −0.026 | +0.0027 |
| acousticness | 1.30 | −0.0052 | +0.005 | −0.0030 |
| speechiness | 1.00 | −0.0052 | +0.011 | −0.0037 |
| instrumentalness | 0.90 | −0.0040 | +0.012 | −0.0043 |
| energy | 1.40 | −0.0038 | −0.035 | −0.0038 |
| duration | 0.35 | −0.0037 | +0.001 | +0.0018 |
| danceability | 1.20 | −0.0035 | +0.007 | −0.0048 |
| valence | 1.00 | −0.0030 | +0.007 | −0.0060 |
| liveness | 0.40 | −0.0020 | +0.005 | −0.0020 |
| explicit | 0.50 | −0.0008 | −0.000 | +0.0007 |
| mode | 0.15 | −0.0005 | −0.000 | +0.0002 |
| key | 0.15 | −0.0002 | −0.001 | −0.0015 |
| tempo | 0.80 | +0.0003 | −0.080 | −0.0032 |

* **Every audio feature carries signal**: dropping any of them lowers the
  objective, and doubling most of them lowers it too. The defaults sit near a
  local optimum. (Changes under ~0.003 are within noise.)
* **Tempo** does not help the artist proxy (Spotify's tempo estimates often
  suffer half/double-time errors), but removing it makes playlists far less
  rhythmically consistent (tempo agreement −0.08). Listeners notice that, so
  it stays at a moderate 0.8. The original's 2.5 was too high.
* **Key and mode** barely matter for retrieval. They are kept at a small weight
  as a harmonic tie-breaker.

## 6. Limitations

* **Proxy, not people.** Same-artist retrieval rewards style consistency, but
  real taste also crosses artists and genres. Only a user study (like the
  original's 10-person study, mean 8.0/10 for single tracks and 7.8/10 for
  playlists) or interaction data can measure satisfaction directly. A natural
  next step is to repeat that study with legacy vs remake shown blind.
* **Content only.** Spotify audio descriptors are coarse (no timbre,
  instrumentation, lyrics or genre). Two tracks can share every descriptor and
  still sound unrelated, which caps what any weighting can do.
* **Dataset scope.** 154k tracks up to 2020; sparse before 1950.
