"""Console entry point for the manipulation analysis CLI."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import analysis as an


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark vs reality manipulation analysis via CLI.")
    parser.add_argument("--n-conversations", type=int, default=None, help="WildChat conversations to sample (default env N_CONVERSATIONS or 2).")
    parser.add_argument("--n-truthfulqa", type=int, default=None, help="TruthfulQA questions to evaluate (default env N_TRUTHFULQA or 2).")
    parser.add_argument("--min-turns", type=int, default=None, help="Minimum turns per conversation (default env MIN_TURNS or 3).")
    parser.add_argument("--max-workers", type=int, default=None, help="Worker threads for TruthfulQA (default env MAX_WORKERS or 4).")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated Baseten model ids to replay (default notebook list).")
    parser.add_argument("--baseten-api-key", type=str, default=None, help="Override BASETEN_API_KEY env var.")
    parser.add_argument("--replicate-api-token", type=str, default=None, help="Override REPLICATE_API_TOKEN env var.")
    parser.add_argument("--allow-non-english", action="store_true", help="Include non-English WildChat conversations.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Shuffle seed for WildChat streaming sampler.")
    parser.add_argument("--results-dir", type=Path, default=an.DEFAULT_RESULTS_DIR, help="Where to cache json results.")
    parser.add_argument("--outputs-dir", type=Path, default=an.DEFAULT_OUTPUTS_DIR, help="Where to write figures and summaries.")
    parser.add_argument("--force-reality", action="store_true", help="Ignore cached WildChat results and rerun.")
    parser.add_argument("--force-benchmark", action="store_true", help="Ignore cached TruthfulQA results and rerun.")
    parser.add_argument("--skip-reality", action="store_true", help="Skip WildChat replay pipeline.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip TruthfulQA benchmark pipeline.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--test-endpoints", action="store_true", help="Run short smoke tests against Replicate and Baseten before analysis.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> an.Config:
    an.load_env()

    cfg = an.Config(
        n_conversations=args.n_conversations or int(os.getenv("N_CONVERSATIONS", 2)),
        n_truthfulqa=args.n_truthfulqa or int(os.getenv("N_TRUTHFULQA", 2)),
        min_turns=args.min_turns or int(os.getenv("MIN_TURNS", 3)),
        max_workers=args.max_workers or int(os.getenv("MAX_WORKERS", 4)),
        english_only=not args.allow_non_english,
        baseten_api_key=args.baseten_api_key or os.getenv("BASETEN_API_KEY", ""),
        replicate_api_token=args.replicate_api_token or os.getenv("REPLICATE_API_TOKEN", ""),
        models=[m.strip() for m in args.models.split(",")] if args.models else an.DEFAULT_MODELS.copy(),
        results_dir=args.results_dir,
        outputs_dir=args.outputs_dir,
        sample_seed=args.sample_seed,
    )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    cfg.validate()
    return cfg


def main() -> None:
    args = parse_args()
    an.setup_logging(args.log_level)
    an.apply_matplotlib_style()

    try:
        cfg = build_config(args)
    except Exception as exc:  # pragma: no cover
        logging.error("Configuration error: %s", exc)
        sys.exit(1)

    logging.info("Starting analysis with %s conversations and %s TruthfulQA questions", cfg.n_conversations, cfg.n_truthfulqa)

    if args.test_endpoints:
        an.smoke_test_endpoints(cfg)

    benchmark_scores: dict = {}
    reality_df = None

    if not args.skip_reality:
        reality_df = an.run_reality_pipeline(cfg, force=args.force_reality)
    else:
        logging.info("Skipping WildChat reality pipeline per --skip-reality")
        reality_df = None

    if not args.skip_benchmark:
        benchmark_scores = an.run_benchmark_pipeline(cfg, force=args.force_benchmark)
    else:
        logging.info("Skipping TruthfulQA benchmark per --skip-benchmark")

    score_columns: list[str] = []
    reality_scores_by_model = None
    if reality_df is not None and not reality_df.empty:
        score_columns, reality_scores_by_model = an.compute_score_tables(reality_df)

    figure_paths: list[Path] = []
    if not args.skip_plots:
        figure_paths.extend(
            path
            for path in [
                an.plot_benchmark(benchmark_scores, cfg.outputs_dir),
                an.plot_reality_overview(score_columns, reality_scores_by_model or an.pd.DataFrame(), cfg.outputs_dir),
                an.plot_manipulation_heatmap(score_columns, reality_scores_by_model or an.pd.DataFrame(), cfg.outputs_dir),
                an.plot_domain_distribution(reality_df if reality_df is not None else an.pd.DataFrame(), cfg.outputs_dir),
                an.plot_correlation(benchmark_scores, reality_scores_by_model or an.pd.DataFrame(), cfg.outputs_dir),
            ]
            if path
        )
    else:
        logging.info("Skipping figure generation per --skip-plots")

    summary_path = an.write_summary(benchmark_scores, reality_df if reality_df is not None else an.pd.DataFrame(), score_columns, cfg)

    logging.info("Analysis complete.")
    logging.info("Summary: %s", summary_path)
    if figure_paths:
        logging.info("Figures: %s", ", ".join(str(p) for p in figure_paths))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        logging.error("Fatal error: %s", exc)
        sys.exit(1)
