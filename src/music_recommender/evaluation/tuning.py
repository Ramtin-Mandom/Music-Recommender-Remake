"""Benchmark presets and random-search tuning of the feature weights."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import EngineConfig
from ..legacy import LEGACY_CONFIG
from ..recommender import Recommender
from .protocols import Evaluator


def benchmark_configs(base: EngineConfig | None = None) -> dict[str, EngineConfig]:
    """The original model, the new default, and single-change ablations."""
    base = base or EngineConfig()
    return {
        "legacy (CMPT 310 model)": LEGACY_CONFIG,
        "remake (default)": base,
        "remake - no era feature": base.with_overrides(weights={"era": 0.0}),
        "remake - standard scaling": base.with_overrides(scaling="standard"),
        "remake - centroid only": base.with_overrides(strategy="centroid"),
        "remake - no popularity prior": base.with_overrides(popularity_weight=0.0),
        "remake - no MMR diversity": base.with_overrides(diversity=0.0),
        "remake - cosine metric": base.with_overrides(metric="cosine"),
    }


def run_benchmark(
    tracks: pd.DataFrame,
    configs: dict[str, EngineConfig] | None = None,
    *,
    split: str = "test",
    n_queries: int = 300,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    configs = configs or benchmark_configs()
    evaluator = Evaluator(tracks, split=split, n_queries=n_queries)
    rows = []
    fitted: dict[tuple, Recommender] = {}
    for name, cfg in configs.items():
        space = (tuple(sorted(cfg.weights.items())), cfg.scaling, cfg.metric)
        engine = fitted.get(space)
        if engine is None:
            engine = Recommender(cfg).fit(tracks)
            fitted[space] = engine
        else:  # same feature space: reuse the fitted index
            engine = engine.with_config(**{k: v for k, v in cfg.to_dict().items() if k != "weights"})
        result = evaluator.evaluate(engine, name)
        log(f"  {name:<32} objective={result.metrics['objective']:.4f}  ({result.seconds:.0f}s)")
        rows.append(result.as_row())
    return pd.DataFrame(rows)


@dataclass
class TuningResult:
    best_config: EngineConfig
    best_score: float
    history: pd.DataFrame


def random_search(
    tracks: pd.DataFrame,
    *,
    base: EngineConfig | None = None,
    trials: int = 30,
    n_queries: int = 150,
    max_weight: float = 2.5,
    frozen: tuple[str, ...] = ("era",),
    min_gain: float = 0.001,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> TuningResult:
    """(1+1)-style evolutionary random search over the feature weights.

    Each trial multiplies the current best weights by log-normal noise (and
    occasionally switches a group off/on), re-fits the feature space, and
    keeps the candidate if the validation objective improves by more than
    ``min_gain`` (a guard against hill-climbing on evaluation noise).  Scored
    on the *validation* artists only.

    ``frozen`` groups keep their weight.  ``era`` is frozen by default: the
    same-artist objective rewards era matching so strongly that an unfrozen
    search turns the engine into a "same release year" recommender (see
    docs/EVALUATION.md, "Era trade-off").
    """
    rng = np.random.default_rng(seed)
    base = base or EngineConfig()
    evaluator = Evaluator(tracks, split="validation", n_queries=n_queries)

    def score(cfg: EngineConfig) -> float:
        return evaluator.evaluate(Recommender(cfg).fit(tracks), beyond=False).metrics["objective"]

    best_cfg, best = base, score(base)
    history = [{"trial": 0, "objective": best, "accepted": True, **best_cfg.weights}]
    log(f"  trial  0  objective={best:.4f}  (start)")
    for t in range(1, trials + 1):
        step = 0.35 if t < trials // 2 else 0.2
        weights = {}
        for g, w in best_cfg.weights.items():
            if g in frozen:
                weights[g] = w
                continue
            w = w if w > 0 else 0.1 * rng.random()  # allow switched-off groups to come back
            w *= float(np.exp(rng.normal(0, step)))
            if rng.random() < 0.05:
                w = 0.0
            weights[g] = round(float(np.clip(w, 0.0, max_weight)), 3)
        cand = best_cfg.with_overrides(weights=weights)
        try:
            cand.validate()
        except ValueError:
            continue
        s = score(cand)
        accepted = s > best + min_gain
        if accepted:
            best_cfg, best = cand, s
        history.append({"trial": t, "objective": s, "accepted": accepted, **weights})
        log(f"  trial {t:2d}  objective={s:.4f}  best={best:.4f}{'  *' if accepted else ''}")
    return TuningResult(best_cfg, best, pd.DataFrame(history))


def feature_sensitivity(
    tracks: pd.DataFrame,
    *,
    base: EngineConfig | None = None,
    n_queries: int = 300,
    split: str = "validation",
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """One-at-a-time ablation: drop each feature group (weight 0) and double it,
    reporting the change in the objective and in audio bucket agreement."""
    base = base or EngineConfig()
    evaluator = Evaluator(tracks, split=split, n_queries=n_queries)

    def run(cfg: EngineConfig) -> tuple[float, float]:
        m = evaluator.evaluate(Recommender(cfg).fit(tracks)).metrics
        agree = np.mean([m[f"{b}_bucket_agreement"] for b in ("tempo", "energy", "loudness")])
        return m["objective"], float(agree)

    ref_obj, ref_agree = run(base)
    rows = []
    for group, w in base.weights.items():
        if w <= 0:
            continue
        drop_obj, drop_agree = run(base.with_overrides(weights={group: 0.0}))
        dbl_obj, _ = run(base.with_overrides(weights={group: min(2 * w, 5.0)}))
        rows.append(
            {
                "group": group,
                "weight": w,
                "objective_if_dropped": round(drop_obj - ref_obj, 4),
                "audio_agreement_if_dropped": round(drop_agree - ref_agree, 4),
                "objective_if_doubled": round(dbl_obj - ref_obj, 4),
            }
        )
        log(f"  {group:<17} drop {drop_obj - ref_obj:+.4f}   double {dbl_obj - ref_obj:+.4f}")
    out = pd.DataFrame(rows).sort_values("objective_if_dropped")
    out.attrs["reference_objective"] = ref_obj
    return out


def weight_sweep(
    tracks: pd.DataFrame,
    group: str,
    values: list[float],
    *,
    base: EngineConfig | None = None,
    n_queries: int = 300,
    split: str = "validation",
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Evaluate the engine for several weights of one feature group."""
    base = base or EngineConfig()
    evaluator = Evaluator(tracks, split=split, n_queries=n_queries)
    rows = []
    for v in values:
        m = evaluator.evaluate(Recommender(base.with_overrides(weights={group: v})).fit(tracks)).metrics
        agree = np.mean([m[f"{b}_bucket_agreement"] for b in ("tempo", "energy", "loudness")])
        rows.append(
            {
                f"{group}_weight": v,
                "objective": m["objective"],
                "radio_precision@k": m["radio_precision@k"],
                "playlist_precision@k": m["playlist_precision@k"],
                "audio_bucket_agreement": round(float(agree), 4),
                "intra_list_diversity": m["intra_list_diversity"],
                "year_gap": m["year_gap"],
            }
        )
        log(f"  {group}={v:<5} objective={m['objective']:.4f}  year_gap={m['year_gap']:.1f}")
    return pd.DataFrame(rows)
