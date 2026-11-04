# Reports

Generated evaluation outputs. They are interpreted in
[docs/EVALUATION.md](../docs/EVALUATION.md).

| File | Produced by | Split | Content |
|---|---|---|---|
| `benchmark.csv` | `music-rec evaluate --queries 500` | test artists | Legacy vs remake vs 6 ablations: accuracy, beyond-accuracy and bucket-agreement metrics |
| `era_sweep.csv` | `music-rec analyze` | validation artists | Objective and guard metrics for era weight 0 to 3.5 |
| `feature_sensitivity.csv` | `music-rec analyze` | validation artists | Change in objective / audio agreement when each feature group is dropped or doubled |
| `tuning_unfrozen_history.csv` | `random_search(trials=60, n_queries=200, seed=1, frozen=())` | validation artists | Trial history of the unconstrained weight search that over-fitted era (see EVALUATION.md §4) |

Each command overwrites its files. The query sets are seeded, so re-running on
the same data and config gives the same numbers.
