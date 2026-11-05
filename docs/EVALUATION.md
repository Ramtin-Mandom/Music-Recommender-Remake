# Evaluation

All numbers below come from the v2 dataset (558k tracks after cleaning) and can
be regenerated:

```bash
music-rec evaluate --queries 500    # reports/benchmark.csv                    (test artists)
music-rec analyze                   # reports/{era,genre,artist_sound}_sweep.csv,
                                    # reports/feature_sensitivity.csv          (validation artists)
```

## 1. Why a new evaluation (twice)?

**The CMPT 310 metric.** The original bucketed every track by tempo, energy
and loudness, then measured how often a track's k nearest neighbours fell in
the same bucket. That is **partly circular**: tempo and energy are model
inputs, so neighbours share those buckets by construction. It is kept below for
continuity only.

**The v2.0 remake's metric.** The first remake used *same-artist retrieval*:
"give one song by an artist, count the artist's other songs in the top 10".
That was fair while artist identity was not a feature. The v2 engine adds two
**artist-level** features: `genre` (the artist's tags) and `artist_sound` (the
average of the artist's other songs). Both are identical across an artist's
tracks, so same-artist retrieval would now reward the engine for recognising
the artist rather than for good recommendations.

## 2. Protocols

The dataset still has no listening histories, so we need a proxy for "a
listener who likes X would also like Y" that the model **cannot see**. We use
**collaborations**:

> Artists who feature on each other's songs ("Low (feat. T-Pain)") almost
> always make music for the same audience. Featured credits are parsed from
> titles, and titles are not a model input.

### 2.1 Collaborator retrieval (accuracy)

* The collaboration graph links an artist to every catalogue artist that
  features on their songs or whose songs feature them (names are matched after
  normalisation). An artist is **eligible** with ≥ 3 distinct songs and ≥ 5
  catalogue tracks by collaborators. That gives 4,797 artists, shuffled with a
  fixed seed and **split 50/50 into validation and test**. Sweeps, sensitivity
  and tuning use validation artists only; the headline numbers use test artists.
* **Radio task**: 1 random song by the artist is the seed. **Playlist task**:
  3 random songs are the seeds.
* **All of the seed artist's own tracks are removed** from the candidates. The
  question is "who else?".
* Each engine returns its top 10 with the artist caps and MMR **disabled**, so
  we measure the ranking signal rather than deliberate diversification.
* Metrics: **precision@10** (share of the list by a collaborator), **hit@10**
  (at least one), **nDCG@10**. *Objective* = mean of radio and playlist
  precision@10.

Absolute values are small by nature. An artist has a handful of collaborators
among 43,700 artists and 558k tracks, so random ranking scores around 10⁻⁴,
and collaborators often work in a *related* rather than identical style. Read
the numbers **relative** to each other.

### 2.2 Beyond accuracy (production settings)

Using each engine's real re-ranking (caps, MMR, dedupe, version penalty), over
the same queries:

| Metric | Meaning |
|---|---|
| genre_coherence | share of tagged recommendations sharing ≥ 1 genre tag with a seed. **Partly circular for engines that use the genre feature**, so it's a guard metric, not a target |
| artist_diversity | unique artists / list length |
| intra_list_diversity | mean pairwise distance in a **fixed** reference space (z-scored raw audio features, identical for all engines) |
| mean_popularity | mean Spotify track popularity of recommendations |
| year_gap | mean \|year − seed year\| |
| alt_version_rate | share of live / demo / karaoke / sped-up / holiday … tracks |
| seed_artist_share | share of the list by the seeds' own artist |
| coverage | distinct tracks recommended / list slots |
| *_bucket_agreement | the original CMPT 310 metric (share of the list in the seed's tempo / energy / loudness bucket) |

## 3. Results: legacy vs remake (test artists, 500 queries per task)

| Engine | Radio P@10 | Radio hit@10 | Playlist P@10 | **Objective** | Genre coherence | Year gap | Alt versions | ILD |
|---|---|---|---|---|---|---|---|---|
| Legacy (CMPT 310 model) | 0.0014 | 0.014 | 0.0026 | 0.0020 | 0.07 | 10.0 y | 4.3 % | 0.73 |
| Remake, audio only (the v2.0 feature set) | 0.0064 | 0.052 | 0.0110 | 0.0087 | 0.12 | 2.1 y | 0.6 % | 1.41 |
| **Remake (default)** | **0.0366** | 0.104 | **0.0306** | **0.0336** | **0.58** | 3.1 y | **0.7 %** | **1.80** |
| Remake, no genre | 0.0158 | 0.062 | 0.0228 | 0.0193 | 0.26 | 2.6 y | 0.6 % | 1.60 |
| Remake, no artist sound | 0.0250 | 0.110 | 0.0270 | 0.0260 | 0.60 | 2.6 y | 0.8 % | 1.70 |
| Remake, no era feature | 0.0356 | 0.108 | 0.0306 | 0.0331 | 0.58 | 7.2 y | 0.8 % | 1.73 |
| Remake, standard scaling | 0.0352 | 0.110 | 0.0310 | 0.0331 | 0.62 | 3.0 y | 1.0 % | 1.48 |
| Remake, centroid only | 0.0366 | 0.104 | 0.0306 | 0.0336 | 0.57 | 2.9 y | 0.7 % | 1.74 |
| Remake, no popularity prior | 0.0352 | 0.106 | 0.0304 | 0.0328 | 0.58 | 3.1 y | 0.7 % | 1.79 |
| Remake, no version penalty | 0.0368 | 0.104 | 0.0306 | 0.0337 | 0.59 | 3.1 y | 3.5 % | 1.81 |
| Remake, no MMR diversity | 0.0366 | 0.104 | 0.0306 | 0.0336 | 0.59 | 3.0 y | 0.7 % | 1.73 |

*(Full table with nDCG, coverage, popularity, seed-artist share and bucket
agreement: `reports/benchmark.csv`.)*

### Reading the table

* **17× the legacy model** (objective 0.0020 → 0.0336). The legacy model
  almost never finds a collaborator (1.4 % of radio queries), and only 7 % of
  its tagged recommendations share a genre with the seed.
* **Genre is the biggest single gain.** Removing it drops the objective by
  43 % and genre coherence from 0.58 to 0.26. Audio descriptors alone can't
  tell soul from folk.
* **Artist sound adds +29 %** on top of genre (0.0260 → 0.0336). It also
  covers the 29 % of artists without tags. Without it, hit@10 is slightly
  higher but precision lower: artist sound concentrates the list on the right
  neighbourhood.
* **Both together are 3.9× the audio-only feature set** of the first remake
  (0.0087). With audio only, the collaborator protocol is much harsher on the
  first remake than same-artist retrieval was. That is expected, since its
  space never encoded anything artist-level.
* **Era no longer matters for accuracy** (0.0331 without it, within noise),
  because genre already places a track in its scene. Era is kept for
  naturalness: without it the mean year gap more than doubles (3.1 → 7.2
  years).
* **The version penalty works.** Alternate versions drop from 3.5 % to 0.7 %
  of list slots at no accuracy cost (there is no protocol reason they should
  be found, so this is purely a quality-of-list effect).
* **Hybrid vs centroid, MMR, scaling and the popularity prior** are all within
  noise on accuracy here. They are kept for their list-quality effects: MMR
  raises artist diversity (0.77 → 0.83) and intra-list diversity, quantile
  scaling raises ILD (1.48 → 1.80), and the popularity prior lifts mean
  popularity from 31 to 37.

**Noise level.** With 500 queries × 10 slots per task, the standard error of a
precision around 0.03 is about ±0.003. Treat differences under ~0.005 as ties.
The legacy vs remake gap is roughly 10 standard errors, and the genre ablation
about 5.

### The original metric, for continuity

| Engine | Tempo agreement | Energy agreement | Loudness agreement |
|---|---|---|---|
| Legacy | **0.77** | **0.81** | 0.78 |
| Remake, audio only | 0.70 | 0.77 | **0.86** |
| Remake (default) | 0.61 | 0.70 | 0.84 |

The legacy model wins on tempo because it gives tempo a weight of 2.5, which is
exactly the circularity described in §1. The genre and artist-sound features
cost the default engine some tempo and energy agreement. That is the price of
recommending "the same kind of music" rather than "the same BPM". It also
motivates the optional **flow ordering** (`order="flow"`), which fixes the
*sequence* of a list for tempo and energy without changing which tracks are
chosen.

## 4. Weight sweeps (validation artists, 300 queries)

Each sweep varies one weight from the defaults. "Audio agreement" is the mean
of the tempo / energy / loudness bucket agreements: a guard that drops when
recommendations stop *sounding* like the seed. With 300 queries the standard
error on the objective is about ±0.004.

**Genre**

| genre weight | Objective | Audio agreement | Year gap | ILD |
|---|---|---|---|---|
| 0.0 | 0.0237 | 0.756 | 2.3 y | 1.57 |
| 1.0 | 0.0345 | 0.751 | 2.4 y | 1.63 |
| 2.5 | 0.0392 | 0.737 | 2.7 y | 1.74 |
| *3.0 (default)* | *≈ 0.039* | | | |
| 3.5 | 0.0387 | 0.733 | 2.8 y | 1.79 |
| 5.0 | 0.0388 | 0.727 | 3.0 y | 1.85 |

The first unit of genre weight is worth +46 %, and the gain plateaus from about
2.5. Beyond that only audio agreement keeps falling.

**Artist sound**

| artist_sound weight | Objective | Audio agreement | Year gap | ILD |
|---|---|---|---|---|
| 0.0 | 0.0272 | 0.753 | 2.4 y | 1.68 |
| 0.5 | 0.0317 | 0.752 | 2.4 y | 1.69 |
| 1.0 | 0.0363 | 0.743 | 2.6 y | 1.72 |
| *1.5 (default)* | *≈ 0.039* | | | |
| 1.75 | 0.0393 | 0.734 | 2.9 y | 1.79 |
| 2.5 | 0.0405 | 0.726 | 3.1 y | 1.84 |

The objective keeps rising, but the heavier this group, the more a track is
judged by its artist's *catalogue* rather than by itself. A radio seeded with an
artist's quiet ballad would drift towards the energy of their typical song,
which audio agreement shows falling. 1.5 keeps most of the gain.

**Era**

| era weight | Objective | Audio agreement | Year gap | ILD |
|---|---|---|---|---|
| 0.0 | 0.0353 | 0.747 | 6.7 y | 1.71 |
| 0.75 | 0.0368 | 0.742 | 4.2 y | 1.74 |
| **1.5 (default)** | **0.0390** | **0.736** | **2.8 y** | **1.77** |
| 2.5 | 0.0388 | 0.732 | 2.0 y | 1.81 |
| 3.5 | 0.0375 | 0.732 | 1.6 y | 1.84 |

On the v1 data, era was the strongest single feature, and an unconstrained
tuner pushed it until the engine became a "same release year" recommender.
With genre in the space, era matters much less for accuracy: the objective
peaks around 1.5–2.5 and then falls. 1.5 is at the peak, and heavier weights
only narrow the time window. `music-rec tune` still freezes era by default
(`--unfreeze-era` to override) and only accepts candidates that improve by more
than 0.001, a guard against hill-climbing on evaluation noise.

**Choosing genre and artist sound together** (validation artists, 500 queries):

| genre | artist_sound | Objective | Genre coherence | Audio agreement |
|---|---|---|---|---|
| 2.5 | 1.0 | 0.0367 | 0.568 | 0.745 |
| **3.0** | **1.5** | **0.0410** | **0.579** | **0.734** |
| 3.5 | 1.5 | 0.0407 | 0.598 | 0.732 |
| 3.5 | 2.0 | 0.0426 | 0.577 | 0.727 |
| 4.0 | 1.75 | 0.0424 | 0.605 | 0.726 |

3.0 / 1.5 gives +12 % over 2.5 / 1.0. The heavier pairs add another ~0.0015,
which is within noise, and cost more audio agreement.

## 5. Per-feature sensitivity (validation artists)

Each group is dropped (weight 0) or doubled, one at a time, from the defaults
(reference objective 0.0390, 300 queries, so changes under ~0.004 are within
noise):

| Group | Weight | Δ objective if dropped | Δ audio agreement if dropped | Δ objective if doubled |
|---|---|---|---|---|
| genre | 3.00 | **−0.0153** | +0.021 | −0.0002 |
| artist_sound | 1.50 | **−0.0118** | +0.017 | +0.0030 |
| tempo | 0.80 | −0.0040 | **−0.054** | +0.0003 |
| era | 1.50 | −0.0037 | +0.011 | 0.0000 |
| acousticness | 1.30 | −0.0028 | −0.006 | −0.0007 |
| valence | 1.00 | −0.0025 | +0.002 | 0.0000 |
| danceability | 1.20 | −0.0018 | −0.000 | −0.0037 |
| loudness | 0.90 | −0.0012 | −0.009 | −0.0007 |
| instrumentalness | 0.90 | −0.0010 | +0.010 | −0.0030 |
| energy | 1.40 | −0.0008 | **−0.030** | −0.0033 |
| liveness | 0.40 | −0.0008 | +0.002 | +0.0010 |
| explicit | 0.50 | −0.0005 | −0.000 | +0.0003 |
| speechiness | 1.00 | −0.0002 | +0.004 | −0.0033 |
| mode | 0.15 | 0.0000 | −0.000 | +0.0003 |
| meter | 0.15 | 0.0000 | −0.000 | −0.0002 |
| key | 0.15 | +0.0005 | −0.001 | −0.0005 |
| duration | 0.35 | (switched off automatically: 99 % of v2 durations are missing) | | |

* **Genre and artist sound carry the accuracy.** Dropping either costs 30–40 %
  of the objective. Doubling them doesn't help, so the defaults sit on the
  plateau.
* **Tempo and energy keep lists sounding coherent.** Their effect on the
  collaborator proxy is small, but dropping them costs 5.4 and 3.0 points of
  audio agreement. That is exactly what a listener hears as "this doesn't fit",
  so they keep their weights.
* **Doubling any single audio feature does not help** (all changes are ≤ 0 or
  within noise), so the hand-set audio weights are still near a local optimum
  in the new space.
* **Key, mode and time signature** barely matter for retrieval. They are kept
  at a small weight as harmonic and rhythmic tie-breakers.

## 6. Limitations

* **Proxy, not people.** Collaborator retrieval rewards "same audience", but
  real taste also crosses scenes. Only a user study (like the original's
  10-person study, mean 8.0/10 for single tracks and 7.8/10 for playlists) or
  interaction data can measure satisfaction directly. A natural next step is to
  repeat that study with legacy vs remake shown blind.
* **Artist-level genres.** Tags describe the artist, not the track, so a
  rock band's one ballad still carries "rock". The track's own audio features
  are what separate it from the band's other songs.
* **Name matching.** Collaborations are only found when a featured name
  matches a catalogue artist's name exactly (after normalisation), so the
  graph is incomplete. That makes the absolute numbers conservative but
  shouldn't bias the comparison between engines.
* **Dataset scope.** Strong skew towards recent music (58 % from 2015 on);
  thin before 1970.
