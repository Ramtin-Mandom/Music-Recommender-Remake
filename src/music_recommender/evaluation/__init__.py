"""Offline evaluation: metrics, protocols, benchmarks and tuning."""

from .metrics import hit_at_k, intra_list_diversity, ndcg_at_k, precision_at_k
from .protocols import EvalResult, Evaluator, collaboration_graph
from .tuning import TuningResult, benchmark_configs, feature_sensitivity, random_search, run_benchmark, weight_sweep

__all__ = [
    "EvalResult",
    "Evaluator",
    "TuningResult",
    "benchmark_configs",
    "collaboration_graph",
    "feature_sensitivity",
    "hit_at_k",
    "intra_list_diversity",
    "ndcg_at_k",
    "precision_at_k",
    "random_search",
    "run_benchmark",
    "weight_sweep",
]
