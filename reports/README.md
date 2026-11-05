# Reports

Generated evaluation outputs on the v2 dataset. They are interpreted in
[docs/EVALUATION.md](../docs/EVALUATION.md).

| File | Produced by | Split | Content |
|---|---|---|---|
| `benchmark.csv` | `music-rec evaluate --queries 500` | test artists | Legacy vs remake vs 9 ablations: collaborator-retrieval accuracy, beyond-accuracy and bucket-agreement metrics |
| `era_sweep.csv`, `genre_sweep.csv`, `artist_sound_sweep.csv` | `music-rec analyze` | validation artists | Objective and guard metrics as one group's weight varies |
| `feature_sensitivity.csv` | `music-rec analyze` | validation artists | Change in objective / audio agreement when each feature group is dropped or doubled |

Each command overwrites its files. The query sets are seeded, so re-running on
the same data and config gives the same numbers.
