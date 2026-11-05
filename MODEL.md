# Model

A complete description of the Music Recommender Remake's k-NN model: data,
features, weights, every hyperparameter, the ranking pipeline, and the test
results. This file covers model version **3.1.0** on the v2 dataset.
Deeper derivations are in [docs/ALGORITHM.md](docs/ALGORITHM.md), and the full
evaluation write-up is in [docs/EVALUATION.md](docs/EVALUATION.md).

---

## 1. Summary

| | |
|---|---|
| Model type | Content-based **k-nearest-neighbour** recommender with re-ranking |
| Training data | 558,328 tracks / 43,705 artists after cleaning ([Almost a million Spotify tracks](https://doi.org/10.5281/zenodo.11453410), CC BY 4.0) |
| Input | 1+ liked tracks (optionally weighted; typed, searched, or pasted from a Spotify playlist), optional disliked tracks, or target attribute values |
| Output | Ranked list of *k* tracks with a calibrated 0–1 similarity, a score and a "why" |
| Feature space | 16 active feature groups → **40 dimensions** (float32) |
| Distance | Euclidean, exact brute-force search |
| "Training" | No labels. Fitting = scaling, genre embedding, artist statistics and similarity calibration (~8 s) |
| Latency | ~20 ms per request on a laptop (plus ~100 ms when a title must be searched) |
| Headline result | **17×** the original CMPT 310 model on held-out collaborator retrieval (0.0336 vs 0.0020) |

```
                   ┌──────────────────── fit (once) ────────────────────┐
tracks.csv ─► load + recover artists ─► clean ─► FeaturePipeline ─► X (558k × 40)
                   └──────────────► σ calibration, typical feature gaps ┘

request ─► resolve seeds ─► exact k-NN candidates ─► filters ─► relevance
        ─► popularity prior − version penalty ─► version dedupe
        ─► MMR + artist caps ─► (optional) flow order ─► results + "why"
```

---

## 2. Data preparation

Source columns, artist recovery and cleaning are documented in
[data/README.md](data/README.md). In short:

| Step | Effect |
|---|---|
| Read 23 of the CSV's 33 columns (`available_markets`, chart columns and most album metadata are skipped) | 899,702 rows |
| Build `artist_key` = genres + follower count + artist popularity, and copy the credited name to every track with the same key | Names for 75 % of rows (only 11 % are credited in the raw file) |
| Parse featured artists from titles ("feat.", "with", "ft.") | 19.5k tracks with collaborators (used only by the evaluation) |
| Drop rows with missing features (965), tempo < 30 BPM (529), year < 1900 (26), no recoverable artist (229,307) | |
| Drop duplicates (same normalised title + artists, keeping the most popular copy) | −110,547 |
| Tag alternate versions (`alt_version`) and build `version_key` (title without version suffix + artist) | |
| **Result** | **558,328 tracks** |

---

## 3. Feature space

### 3.1 Feature groups and weights

Every group is scaled, then multiplied by its weight. With Euclidean distance a
group contributes `weight² × (difference)²` to the squared distance.

| # | Group | Source column(s) | Transform | Dims | **Weight** |
|---|---|---|---|---|---|
| 1 | danceability | `danceability` (0–1) | quantile → normal | 1 | **1.2** |
| 2 | energy | `energy` (0–1) | quantile → normal | 1 | **1.4** |
| 3 | valence (mood) | `valence` (0–1) | quantile → normal | 1 | **1.0** |
| 4 | acousticness | `acousticness` (0–1) | quantile → normal | 1 | **1.3** |
| 5 | instrumentalness | `instrumentalness` (0–1) | quantile → normal | 1 | **0.9** |
| 6 | liveness | `liveness` (0–1) | quantile → normal | 1 | **0.4** |
| 7 | speechiness | `speechiness` (0–1) | quantile → normal | 1 | **1.0** |
| 8 | tempo | `tempo` (BPM) | clip to [40, 220], quantile → normal | 1 | **0.8** |
| 9 | loudness | `loudness` (dB) | quantile → normal | 1 | **0.9** |
| 10 | key | `key`, `mode` | circle of fifths, relative minor = major → (sin θ, cos θ) | 2 | **0.15** |
| 11 | mode | `mode` | raw 0/1 | 1 | **0.15** |
| 12 | meter | `time_signature` | 1 if not 4/4, else 0 | 1 | **0.15** |
| 13 | era | `album_release_date` → year | quantile → normal | 1 | **1.5** |
| 14 | explicit | `explicit` | raw 0/1 | 1 | **0.5** |
| 15 | **genre** | artist's `genres` | TF-IDF (tags + words) → SVD (16) → unit length | 16 | **3.0** |
| 16 | **artist_sound** | artist's other tracks | leave-one-out mean of the scaled audio vector, shrunk | 9 | **1.5** |
| – | duration | `duration_ms` | log, clip to [30 s, 20 min], quantile | (1) | 0.35, **auto-disabled**: 99 % missing in v2 |
| | | | | **40** | |

A group's real influence depends on its width as well as its weight. The genre
group is a 16-d unit vector, so a random pair of tracks differs there by about
1.3 × 3.0. The artist-sound group spans 9 columns.

### 3.2 Genre embedding (group 15)

| Hyperparameter | Value |
|---|---|
| Unit of fitting | one document per **artist** (43,705 docs; 31,116 have tags) |
| Tokens | each whole tag + each word of each tag (`"philly soul"` → `philly soul`, `w:philly`, `w:soul`) |
| Vectorizer | `TfidfVectorizer(min_df=2, sublinear_tf=True)`. Vocabulary: 6,138 tokens |
| Compression | `TruncatedSVD(n_components=genre_dims=16, random_state=42)` |
| Normalisation | L2 → unit vectors |
| Untagged artists (12,589) | mean genre vector of the **10** tagged artists with the closest average sound (k-NN over per-artist mean audio vectors); not re-normalised, so uncertain guesses are shorter |

### 3.3 Artist sound (group 16)

For a track by an artist with *n* catalogue tracks:

```
loo      = mean scaled audio vector of the artist's other n − 1 tracks
λ        = (n − 1) / (n − 1 + κ),   κ = artist_shrinkage = 3.0
sound    = λ · loo + (1 − λ) · own audio vector
```

The 9 columns keep the audio weights' proportions, rescaled to unit RMS:
danceability 1.17, energy 1.36, valence 0.97, acousticness 1.27,
instrumentalness 0.88, liveness 0.39, speechiness 0.97, tempo 0.78,
loudness 0.88. The whole group is then multiplied by 1.5.

### 3.4 Scaling

| Hyperparameter | Value |
|---|---|
| `scaling` | `"quantile"`: `QuantileTransformer(output_distribution="normal")` |
| `n_quantiles` | 1,000 |
| `subsample` | 200,000 |
| Alternatives | `"standard"` (z-score, used by the original), `"robust"` (median/IQR) |

---

## 4. Retrieval and similarity

| Hyperparameter | Value | Meaning |
|---|---|---|
| `metric` | `"euclidean"` | Exact search via `‖x‖² − 2Xq + ‖q‖²` + `argpartition` (cosine / manhattan fall back to sklearn) |
| `candidate_pool` | **300** | Neighbours fetched before filtering, split across query vectors |
| Minimum survivors | max(3k, 30) | If filters leave fewer, the pool grows **×4** until enough survive or the catalogue is exhausted |
| Starvation guard | ×4 | If artist caps / dedupe leave fewer than *k*, the whole ranking re-runs with a 4× pool |
| σ (kernel width) | **2.47** | Median distance to the 50th nearest neighbour over 400 random tracks |
| Similarity | `exp(−ln 2 · (d/σ)²)` | 1 = identical, 0.5 = as far as a typical 50th neighbour |

---

## 5. Scoring

### 5.1 Multiple seeds

Each seed `s_i` has a weight `a_i` (default 1), and the centroid is
`c = Σ a_i s_i / Σ a_i`. Candidates come from the k-NN of the centroid **and**
of every seed.

```
sim_centroid = K(d(x, c))
sim_nearest  = max_i (a_i / max a) · K(d(x, s_i))
relevance    = α · sim_centroid + (1 − α) · sim_nearest       (strategy = "hybrid")
relevance   −= λ · max_j K(d(x, disliked_j))
```

### 5.2 Final score

```
prior = (1 − ρ) · track_popularity/100 + ρ · artist_popularity/100
score = (1 − β) · relevance + β · prior
score −= π   if x is an alternate version and no seed is the same kind
```

| Hyperparameter | Value | Meaning |
|---|---|---|
| `strategy` | `"hybrid"` | Also `"centroid"` (plain average) and `"nearest_seed"` |
| `hybrid_alpha` (α) | **0.5** | Weight of the centroid term |
| `dislike_penalty` (λ) | **0.6** | Rocchio-style push away from disliked tracks |
| `popularity_weight` (β) | **0.08** | Popularity prior blend |
| `artist_popularity_share` (ρ) | **0.3** | Share of the prior from the artist (21 % of tracks have popularity 0) |
| `version_penalty` (π) | **0.15** | Penalty for alternate versions |

Alternate-version tags (checked in this order on the title): **karaoke** (also
"in the style of"), **sped up** / nightcore, **slowed** / reverb, **8D**,
**live** / en vivo / ao vivo / unplugged, **demo** / early version / outtake /
"take N" / rehearsal, **instrumental**, **acoustic**, **holiday** (christmas,
xmas, santa, jingle bell(s), navidad, noel).

---

## 6. Re-ranking

| Step | Hyperparameter | Value |
|---|---|---|
| Version dedupe: keep the best copy per `version_key` | `dedupe_versions` | **true** |
| MMR: `next = argmax (1 − δ)·score − δ·max_sim_to_selected` | `diversity` (δ) | **0.15** |
| Per-artist cap (by `artist_key`) | `max_per_artist` | **2** |
| Cap on the seeds' own artists | `max_seed_artist` | **1** |
| Ordering | `order` | **`"score"`**, or `"flow"` |

**Flow ordering** (`order="flow"`) keeps the same tracks and re-orders them.
The transition cost from track *i* to *j* is

```
tempo  : min(|log2(t_j/t_i)|, ||log2(t_j/t_i)| − 1|) / 0.08    (half/double time counts as a match)
energy : |e_j − e_i| / 0.15
key    : circle-of-fifths angle / π / 0.5
```

The order comes from a greedy walk from the top-ranked track followed by 2-opt
on the open path. The first track stays first.

---

## 7. Filters

`Filters(year_min, year_max, min_popularity, exclude_explicit, max_duration_s,
exclude_artists, genre)`. Filters are applied to candidates before ranking.
`genre` is a case-insensitive substring of any of the artist's tags, and unknown
durations pass the duration filter.

## 8. Mood search

`from_targets({"energy": 0.9, "tempo": 128, "year": 2015})` embeds a partial
track. Unspecified columns are masked out, and the distance is rescaled by
`sqrt(total_columns / constrained_columns)` so σ still applies. Genre and
artist sound can't be targeted. Filters (including `genre`) can be used
instead.

## 9. Explanations

* **`why`**: the 3 groups where the track is *unusually* close to its nearest
  anchor (the centroid or a seed). Each gap is divided by the median gap between
  random track pairs (4,000 pairs), with the ratio floored at 0.1. Groups
  weighted below 0.5 are skipped, and binary flags are only mentioned when both
  tracks share the rare value.
* **`explain(seeds, row)`**: each group's share of the squared distance to the
  seed centroid, plus a 0–1 closeness per group.

---

## 10. Complete hyperparameter reference

Everything below is a field of `EngineConfig`
([config.py](src/music_recommender/config.py)), serialised in
[configs/default.json](configs/default.json).

| Field | Default | Affects | Changing it refits? |
|---|---|---|---|
| `weights` | see §3.1 | feature space | yes |
| `scaling` | `"quantile"` | feature space | yes |
| `genre_dims` | 16 | genre embedding size (2–128) | yes |
| `artist_shrinkage` | 3.0 | artist-sound shrinkage κ | yes |
| `metric` | `"euclidean"` | distance | yes |
| `random_state` | 42 | quantile subsample, SVD, calibration sampling | yes |
| `candidate_pool` | 300 | retrieval breadth | no |
| `strategy` | `"hybrid"` | multi-seed scoring | no |
| `hybrid_alpha` | 0.5 | multi-seed scoring | no |
| `dislike_penalty` | 0.6 | negative feedback | no |
| `popularity_weight` | 0.08 | popularity prior | no |
| `artist_popularity_share` | 0.3 | popularity prior | no |
| `version_penalty` | 0.15 | alternate versions | no |
| `diversity` | 0.15 | MMR | no |
| `max_per_artist` | 2 | artist cap | no |
| `max_seed_artist` | 1 | seed-artist cap | no |
| `order` | `"score"` | list order | no |
| `dedupe_versions` | true | remaster/live dedupe | no |
| `min_tempo` | 30.0 | cleaning | reload data |
| `require_artist` | true | cleaning | reload data |

Fixed internal constants: tempo clip 40–220 BPM; duration clip 30 s–20 min;
genre-imputation neighbours 10; TF-IDF `min_df` 2; σ from the 50th neighbour of
400 tracks; typical gaps from 4,000 random pairs; candidate growth ×4; typo
search over the 60,000 most popular titles.

Any field can be overridden per call
(`rec.similar_to("…", diversity=0.4, order="flow")`) or for a whole engine
(`rec.with_config(...)`).

---

## 11. How the weights were chosen

| Weights | Method |
|---|---|
| 9 audio features, key, mode, explicit | Hand-set on the v1 data, checked by sensitivity analysis on v1 and again on v2 (doubling any single audio weight doesn't help) |
| genre 3.0, artist_sound 1.5 | One-at-a-time sweeps on the **validation** artists, then a joint comparison at 500 queries (below) |
| era 1.5 | Sweep: the objective peaks at 1.5–2.5 and falls beyond. Heavier weights only narrow the time window |
| meter 0.15 | Tie-breaker. No measurable effect on accuracy |

Joint comparison (validation artists, 500 queries per task):

| genre | artist_sound | Objective | Genre coherence | Audio agreement |
|---|---|---|---|---|
| 2.5 | 1.0 | 0.0367 | 0.568 | 0.745 |
| **3.0** | **1.5** | **0.0410** | **0.579** | **0.734** |
| 3.5 | 1.5 | 0.0407 | 0.598 | 0.732 |
| 3.5 | 2.0 | 0.0426 | 0.577 | 0.727 |
| 4.0 | 1.75 | 0.0424 | 0.605 | 0.726 |

3.0 / 1.5 gains +12 % over 2.5 / 1.0. The heavier pairs add only ~0.0015
(within noise) and make recommendations sound less like the seed track itself.

---

## 12. Test results

### 12.1 Protocol

* **Collaborator retrieval.** 4,797 artists have ≥ 3 songs and ≥ 5 catalogue
  tracks by artists they collaborated with ("feat." credits parsed from titles,
  which the model never sees). They are split 50/50 into validation (used for
  tuning) and **test** (reported here).
* **Radio** = 1 seed song, **playlist** = 3 seed songs. All of the seed
  artist's own tracks are removed from the candidates.
* Top 10 with artist caps and MMR off. **precision@10** = share by a
  collaborator. **Objective** = mean of radio and playlist precision@10.
* Beyond-accuracy metrics use each engine's full production re-ranking.
* Standard error ≈ ±0.003 on precision at 500 queries.

### 12.2 Main results (test artists, 500 queries per task)

| Engine | Radio P@10 | Radio hit@10 | Radio nDCG@10 | Playlist P@10 | Playlist hit@10 | **Objective** |
|---|---|---|---|---|---|---|
| Legacy CMPT 310 model | 0.0014 | 0.014 | 0.0015 | 0.0026 | 0.024 | 0.0020 |
| Remake, audio features only | 0.0064 | 0.052 | 0.0061 | 0.0110 | 0.074 | 0.0087 |
| **Remake (this model)** | **0.0366** | **0.104** | **0.0383** | **0.0306** | **0.086** | **0.0336** |

### 12.3 List quality (same queries, production settings)

| Metric | Legacy | Audio only | **This model** |
|---|---|---|---|
| Genre coherence (shares a tag with a seed) | 0.07 | 0.12 | **0.58** |
| Artist diversity (unique artists / 10) | 0.98 | 0.98 | 0.83 |
| Intra-list diversity (fixed audio space) | 0.73 | 1.41 | 1.80 |
| Mean popularity | 28.3 | 38.2 | 37.4 |
| Mean year gap to seed | 10.0 y | 2.1 y | 3.1 y |
| Alternate-version rate | 4.3 % | 0.6 % | 0.7 % |
| Seed-artist share | 0.3 % | 0.7 % | 8.3 % |
| Coverage (distinct tracks / slots) | 0.97 | 0.95 | 0.90 |
| Tempo bucket agreement | 0.77 | 0.70 | 0.61 |
| Energy bucket agreement | 0.81 | 0.77 | 0.70 |
| Loudness bucket agreement | 0.78 | 0.86 | 0.84 |

### 12.4 Ablations (test artists, one change each)

| Variant | Objective | Change | Note |
|---|---|---|---|
| **Default** | **0.0336** | | |
| No genre | 0.0193 | −43 % | genre coherence 0.58 → 0.26 |
| No artist sound | 0.0260 | −23 % | |
| Audio only (no genre, no artist sound) | 0.0087 | −74 % | |
| No era | 0.0331 | tie | year gap 3.1 → 7.2 years |
| Standard scaling | 0.0331 | tie | intra-list diversity 1.80 → 1.48 |
| Centroid only | 0.0336 | tie | |
| No popularity prior | 0.0328 | tie | mean popularity 37 → 31 |
| No version penalty | 0.0337 | tie | alternate versions 0.7 % → 3.5 % |
| No MMR | 0.0336 | tie | artist diversity 0.83 → 0.77 |

### 12.5 Per-feature sensitivity (validation artists, reference objective 0.0390)

| Group | Δ objective if dropped | Δ audio agreement if dropped | Δ objective if doubled |
|---|---|---|---|
| genre | −0.0153 | +0.021 | −0.0002 |
| artist_sound | −0.0118 | +0.017 | +0.0030 |
| tempo | −0.0040 | −0.054 | +0.0003 |
| era | −0.0037 | +0.011 | 0.0000 |
| acousticness | −0.0028 | −0.006 | −0.0007 |
| valence | −0.0025 | +0.002 | 0.0000 |
| danceability | −0.0018 | −0.000 | −0.0037 |
| loudness | −0.0012 | −0.009 | −0.0007 |
| instrumentalness | −0.0010 | +0.010 | −0.0030 |
| energy | −0.0008 | −0.030 | −0.0033 |
| liveness | −0.0008 | +0.002 | +0.0010 |
| explicit | −0.0005 | −0.000 | +0.0003 |
| speechiness | −0.0002 | +0.004 | −0.0033 |
| mode / meter / key | ≈ 0 | ≈ 0 | ≈ 0 |

### 12.6 Software tests

119 pytest tests: 113 run on synthetic data in ~5 s (including headless
desktop-app tests), and 6 use the full dataset and the Streamlit app (~1 min).
They cover data loading for both layouts, artist recovery, every feature group,
the scoring and re-ranking rules, the flow order, filters, explanations,
Spotify playlist import, persistence, the CLI, both apps and the evaluation
harness.

---

## 13. Known limitations

* **Content only.** There's no listening history, so "people who like X also
  like Y" patterns are invisible. The evaluation is a proxy, not user
  satisfaction.
* **Artist-level genres.** Tags describe the artist, so a rock band's ballad
  still reads as rock. The track's own audio features have to separate it.
* **Rhythmic consistency.** Genre-aware lists match the seed's tempo band less
  often than the tempo-heavy original (0.61 vs 0.77). `order="flow"` smooths
  the sequence.
* **Coverage gaps.** Tracks whose artist is never credited in the source file
  (229k rows) are excluded, and the catalogue skews towards music from 2015 on.
* **Popularity snapshot.** Track popularity was captured separately from the
  audio features, and a fifth of tracks read 0.
