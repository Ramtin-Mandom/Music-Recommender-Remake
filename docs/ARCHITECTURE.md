# Architecture

## Repository layout

```
Music-Recommender-Remake/
├── src/music_recommender/      # the engine (installable package)
│   ├── config.py               # EngineConfig: every hyper-parameter, JSON (de)serialisation
│   ├── data.py                 # CSV loading, cleaning, artist parsing, version keys, synthetic data
│   ├── features.py             # FeaturePipeline: raw columns → weighted embedding
│   ├── catalog.py              # TrackCatalog / TrackRef: search and seed resolution
│   ├── ranking.py              # Filters, kernel similarity, MMR selection
│   ├── recommender.py          # Recommender: the public facade (fit / recommend / explain / save)
│   ├── legacy.py               # the original CMPT 310 model as an EngineConfig preset
│   ├── evaluation/
│   │   ├── metrics.py          # precision/hit/nDCG@k, diversity, bucket agreement
│   │   ├── protocols.py        # Evaluator: seeded query sets + accuracy / beyond-accuracy protocols
│   │   └── tuning.py           # benchmark presets, random search, sensitivity, weight sweeps
│   ├── cli.py                  # `music-rec` command line
│   ├── gui.py                  # PyQt6 desktop app (`music-rec-gui`)
│   └── __main__.py             # `python -m music_recommender`
├── app/streamlit_app.py        # browser-based test bench
├── tests/                      # pytest suite (synthetic data + opt-in full-data tests)
├── data/
│   ├── raw/tracks.csv          # 170k-track Spotify dataset (from CMPT 310)
│   └── legacy/                 # original CV results + anonymised user study
├── reports/                    # generated evaluation outputs (CSV)
└── docs/                       # this documentation
```

## Layering

```
      gui.py      app/streamlit_app.py      cli.py
          \                 │                  /
           ▼                ▼                 ▼
                   recommender.Recommender  ◄──── evaluation/* (uses only the public API
                    │      │       │                          + recommend_rows)
          ┌─────────┘      │       └─────────┐
          ▼                ▼                 ▼
     catalog.py       features.py        ranking.py
          │                │
          └──────► data.py ◄┘            config.py (imported by everything)
```

* **Nothing below `recommender.py` knows about UIs or evaluation.** The desktop
  app, web app and CLI are thin. Anything they can do is available from Python.
* **`EngineConfig` is the single source of truth** for behaviour. It is a
  frozen dataclass, so it is hashable, comparable and safe to share. The
  benchmark uses this to reuse a fitted index across configs that share a
  feature space.
* **The legacy model is just a config** (`legacy.LEGACY_CONFIG`), so old and new
  run through the same code path on the same cleaned data. That makes the
  comparison in `reports/` apples-to-apples.

## Key design decisions

| Decision | Why |
|---|---|
| Keep k-NN (no deep model) | The data is 15 dense, interpretable features with no interaction data. k-NN is transparent, needs no training labels, and every result can be explained. |
| `src/` layout + `pyproject.toml` | Installable (`pip install -e .`), with a console script, and tests import the installed package rather than whatever happens to be on the path. |
| Positional "row" ids internally, Spotify `id` externally | Rows index straight into the embedding matrix. Spotify ids stay stable across re-cleaning and are what results expose. |
| Grow the candidate pool on demand | Restrictive filters or the artist cap could otherwise silently return fewer than *k* results. (The original always returned at most 4–5 because the index was built with `n_neighbors=5`.) |
| Synthetic data generator in the package | Tests run in ~2 s without the 30 MB CSV, and the generator builds real "artist style" structure so the tests check behaviour, not just shapes. |
| Validation/test artist split in evaluation | Tuning reads only validation artists and reported numbers come from test artists, so the report cannot overfit. |
| joblib model cache (`music-rec build`) | The app and CLI start instantly from `artifacts/model.joblib`. Files carry the package version and stale ones are refused. |

## Extending

* **New feature group**: add it to `FEATURE_GROUPS` / `DEFAULT_WEIGHTS` in
  `config.py`, teach `FeaturePipeline` how to derive its column(s), and add a
  label in `GROUP_LABELS`.
* **New re-ranking rule**: add a field to `EngineConfig`, then apply it in
  `Recommender._rank_candidates` (score shaping) or `ranking.mmr_select`
  (list construction).
* **New dataset**: any CSV with the columns listed in `data/README.md` works:
  `Recommender.from_csv("other.csv")` or `music-rec --data other.csv …`.
* **New evaluation protocol**: add a method on `Evaluator` that calls
  `engine.recommend_rows(...)`, then expose it in `Evaluator.evaluate`.
