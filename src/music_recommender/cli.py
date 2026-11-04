"""Command line interface: ``music-rec <command>`` (or ``python -m music_recommender``)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .catalog import TrackNotFoundError
from .config import DEFAULT_DATA_PATH, PROJECT_ROOT, EngineConfig
from .data import load_tracks
from .ranking import Filters
from .recommender import Recommender

DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
SHOW_COLS = ["rank", "name", "artists", "year", "popularity", "similarity", "why"]


def _print_table(df: pd.DataFrame, cols: list[str] | None = None) -> None:
    if df.empty:
        print("(no results)")
        return
    with pd.option_context("display.width", 200, "display.max_colwidth", 42, "display.max_columns", 20):
        print(df[cols or df.columns].to_string(index=False))


def _load_engine(args) -> Recommender:
    config = EngineConfig.load(args.config) if args.config else EngineConfig()
    model_path = Path(args.model) if args.model else DEFAULT_MODEL_PATH
    if not args.config and model_path.exists():
        try:
            return Recommender.load(model_path)
        except ValueError as exc:  # stale model file
            print(f"[warn] {exc} – fitting from CSV instead.", file=sys.stderr)
    start = time.perf_counter()
    engine = Recommender.from_csv(args.data, config)
    print(f"[info] fitted on {len(engine.tracks):,} tracks in {time.perf_counter() - start:.1f}s", file=sys.stderr)
    return engine


def _filters(args) -> Filters:
    return Filters(
        year_min=args.year_min,
        year_max=args.year_max,
        min_popularity=args.min_popularity,
        exclude_explicit=args.no_explicit,
    )


def _overrides(args) -> dict:
    out = {}
    for key in ("diversity", "strategy", "popularity_weight", "max_per_artist"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    return out


# ----------------------------------------------------------------------
def cmd_search(args) -> int:
    engine = _load_engine(args)
    _print_table(engine.search(args.query, args.artist, args.limit).drop(columns=["row"]))
    return 0


def cmd_similar(args) -> int:
    engine = _load_engine(args)
    query = (args.query, args.artist) if args.artist else args.query
    seed = engine.resolve(query)
    s = engine.tracks.iloc[seed]
    print(f"Seed: {s['name']} – {s['artists_display']} ({s['year']})\n")
    _print_table(engine.similar_to(seed, k=args.k, filters=_filters(args), **_overrides(args)), SHOW_COLS)
    return 0


def cmd_playlist(args) -> int:
    engine = _load_engine(args)
    recs = engine.recommend(
        args.like, disliked=args.dislike or (), k=args.k, filters=_filters(args), **_overrides(args)
    )
    _print_table(recs, SHOW_COLS)
    return 0


def cmd_mood(args) -> int:
    engine = _load_engine(args)
    targets = {}
    for item in args.target:
        key, _, value = item.partition("=")
        targets[key.strip()] = float(value)
    _print_table(engine.from_targets(targets, k=args.k, filters=_filters(args), **_overrides(args)), SHOW_COLS)
    return 0


def cmd_build(args) -> int:
    config = EngineConfig.load(args.config) if args.config else EngineConfig()
    start = time.perf_counter()
    engine = Recommender.from_csv(args.data, config)
    out = Path(args.out)
    engine.save(out)
    print(f"Saved model ({len(engine.tracks):,} tracks, {time.perf_counter() - start:.1f}s) to {out}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluation import benchmark_configs, run_benchmark

    tracks = load_tracks(args.data)
    base = EngineConfig.load(args.config) if args.config else EngineConfig()
    print(f"Benchmarking on the '{args.split}' artists ({args.queries} queries per task)...")
    table = run_benchmark(tracks, benchmark_configs(base), split=args.split, n_queries=args.queries)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "benchmark.csv", index=False)
    print()
    _print_table(table.round(4))
    print(f"\nSaved {out_dir / 'benchmark.csv'}")
    return 0


def cmd_tune(args) -> int:
    from .evaluation import random_search

    tracks = load_tracks(args.data)
    base = EngineConfig.load(args.config) if args.config else EngineConfig()
    frozen = () if args.unfreeze_era else ("era",)
    result = random_search(tracks, base=base, trials=args.trials, n_queries=args.queries, seed=args.seed, frozen=frozen)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.best_config.save(out)
    result.history.to_csv(out.with_suffix(".history.csv"), index=False)
    print(f"\nBest validation objective {result.best_score:.4f}; config saved to {out}")
    print(json.dumps(result.best_config.weights, indent=2))
    return 0


def cmd_analyze(args) -> int:
    from .evaluation import feature_sensitivity, weight_sweep

    tracks = load_tracks(args.data)
    base = EngineConfig.load(args.config) if args.config else EngineConfig()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Era-weight sweep (validation artists)...")
    sweep = weight_sweep(tracks, "era", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5], base=base, n_queries=args.queries)
    sweep.to_csv(out_dir / "era_sweep.csv", index=False)
    print("\nPer-feature sensitivity (validation artists)...")
    sens = feature_sensitivity(tracks, base=base, n_queries=args.queries)
    sens.to_csv(out_dir / "feature_sensitivity.csv", index=False)
    print()
    _print_table(sweep.round(4))
    print()
    _print_table(sens)
    print(f"\nSaved era_sweep.csv and feature_sensitivity.csv to {out_dir}")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="music-rec", description="Content-based k-NN music recommender.")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="path to tracks CSV")
    p.add_argument("--config", help="JSON engine config (see EngineConfig)")
    p.add_argument("--model", help=f"pre-built model file (default {DEFAULT_MODEL_PATH.name} if present)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_filters(sp):
        sp.add_argument("-k", type=int, default=10, help="number of recommendations")
        sp.add_argument("--year-min", type=int)
        sp.add_argument("--year-max", type=int)
        sp.add_argument("--min-popularity", type=int)
        sp.add_argument("--no-explicit", action="store_true")
        sp.add_argument("--diversity", type=float, help="0 = relevance only, 1 = max diversity")
        sp.add_argument("--strategy", choices=["centroid", "nearest_seed", "hybrid"])
        sp.add_argument("--popularity-weight", type=float)
        sp.add_argument("--max-per-artist", type=int)

    sp = sub.add_parser("search", help="find tracks in the catalogue")
    sp.add_argument("query")
    sp.add_argument("--artist")
    sp.add_argument("--limit", type=int, default=10)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("similar", help="tracks similar to one song")
    sp.add_argument("query", help='title, "Title by Artist", or a Spotify track id')
    sp.add_argument("--artist")
    add_filters(sp)
    sp.set_defaults(func=cmd_similar)

    sp = sub.add_parser("playlist", help="playlist from several liked (and disliked) songs")
    sp.add_argument("--like", action="append", required=True, help="repeatable")
    sp.add_argument("--dislike", action="append", help="repeatable")
    add_filters(sp)
    sp.set_defaults(func=cmd_playlist)

    sp = sub.add_parser("mood", help="tracks matching target attributes, e.g. energy=0.9 tempo=128")
    sp.add_argument("target", nargs="+", help="attribute=value pairs")
    add_filters(sp)
    sp.set_defaults(func=cmd_mood)

    sp = sub.add_parser("build", help="fit and save the model for fast start-up")
    sp.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("evaluate", help="benchmark legacy vs remake vs ablations")
    sp.add_argument("--split", choices=["validation", "test"], default="test")
    sp.add_argument("--queries", type=int, default=300)
    sp.add_argument("--out", default=str(PROJECT_ROOT / "reports"))
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("analyze", help="era-weight sweep + per-feature sensitivity analysis")
    sp.add_argument("--queries", type=int, default=300)
    sp.add_argument("--out", default=str(PROJECT_ROOT / "reports"))
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("tune", help="random-search the feature weights on validation artists")
    sp.add_argument("--trials", type=int, default=40)
    sp.add_argument("--queries", type=int, default=300)
    sp.add_argument("--unfreeze-era", action="store_true", help="also tune the era weight (not recommended)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=str(PROJECT_ROOT / "configs" / "tuned.json"))
    sp.set_defaults(func=cmd_tune)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TrackNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
