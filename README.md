# manipulation-cli

CLI wrapper for the Benchmark vs Reality manipulation analysis. It replays WildChat conversations through Baseten models, scores manipulation via Claude on Replicate, runs TruthfulQA as a benchmark, and saves figures plus summaries to the repo’s `data/results` and `outputs` folders.

## Requirements
- Python 3.10+
- `uv` for installs/runs (https://github.com/astral-sh/uv)
- Env vars: `BASETEN_API_KEY`, `REPLICATE_API_TOKEN`
- Network access for HuggingFace `allenai/WildChat`, Replicate, and Baseten

## Install with uv
```bash
cd cli
uv sync           # or: uv pip install -e .
uv run manipulation-cli --help
```

## Quickstart
Run a tiny smoke sample (2 conversations, 2 TruthfulQA items), generate plots, and write cached results to `../data/results`:
```bash
uv run manipulation-cli \
  --n-conversations 2 \
  --n-truthfulqa 2
```

Key outputs (relative to repo root):
- `data/results/wildchat_poc_results.json` – cached reality scores
- `data/results/benchmark_poc_results.json` – cached TruthfulQA scores
- `outputs/*.png` – figures
- `outputs/analysis_summary.md` – text summary

## Common flags
- `--models zai-org/GLM-4.6,openai/gpt-oss-120b` – override model list
- `--allow-non-english` – include non-English WildChat samples
- `--force-reality` / `--force-benchmark` – ignore caches and rerun
- `--skip-reality` / `--skip-benchmark` – opt out of pipelines
- `--skip-plots` – skip figure generation
- `--test-endpoints` – short smoke test against Replicate + Baseten
- `--results-dir` / `--outputs-dir` – override default paths
- `MANIPULATION_REPO_ROOT` – override the base folder for `data/` and `outputs/` if you install the package elsewhere

Defaults come from env vars when present:
- `N_CONVERSATIONS` (default 2)
- `N_TRUTHFULQA` (default 2)
- `MIN_TURNS` (default 3)
- `MAX_WORKERS` (default 4)

## Notes
- The CLI uses streaming sampling from `allenai/WildChat`; increase `--n-conversations` gradually to avoid long runs.
- Figures are rendered headlessly via matplotlib; no GUI required.
- Retries with exponential backoff are built in for model/judge calls.
