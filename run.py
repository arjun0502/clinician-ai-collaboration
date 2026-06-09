"""
Entry point for the clinician-AI anchoring experiment.

Models are specified as "provider:model" strings. Supported providers:
  openai     — requires OPENAI_API_KEY
  gemini     — requires GOOGLE_API_KEY
  anthropic  — requires ANTHROPIC_API_KEY

Usage:
  python run.py                                            # defaults: Gemini gen, OpenAI eval
  python run.py --condition anchored                       # single condition only
  python run.py --cases 2 --runs 1                         # quick smoke test
  python run.py --gen-model "anthropic:claude-sonnet-4-6"  # use Claude for generation
  python run.py --eval-model "openai:gpt-4o"               # use GPT-4o as judge
  python run.py --critique-rounds 3                        # 3 LLM self-critique cycles
  python run.py --skip-eval                                # generate only
  python run.py --eval-only                                # evaluate existing results only
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.clients import LLMClient, make_client, provider_of, PROVIDER_ENV_VARS
from src.config import CONDITIONS, CRITIQUE_ROUNDS, EVAL_MODEL, GEN_MODEL, RAW_DIR, RUNS
from src.data import load_cases, load_unique_cases
from src.pipeline import (
    collect_all_eval_results,
    evaluate_result,
    run_anchored,
    run_critique_clinician,
    run_critique_combined,
    run_critique_llm,
    write_summary_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the clinician-AI anchoring experiment.")
    p.add_argument("--condition", choices=CONDITIONS, default=None,
                   help="Run a single condition (default: all).")
    p.add_argument("--cases", type=int, default=None, metavar="N",
                   help="Limit to first N unique cases.")
    p.add_argument("--runs", type=int, default=RUNS, metavar="K",
                   help="Number of independent runs per condition/case (default: %(default)s).")
    p.add_argument("--critique-rounds", type=int, default=CRITIQUE_ROUNDS, metavar="N",
                   help="Critique-then-regenerate cycles for critique_llm/critique_combined (default: %(default)s).")
    p.add_argument("--gen-model", default=GEN_MODEL, metavar="PROVIDER:MODEL",
                   help=f"Generation model (default: {GEN_MODEL}).")
    p.add_argument("--eval-model", default=EVAL_MODEL, metavar="PROVIDER:MODEL",
                   help=f"Judge/evaluation model (default: {EVAL_MODEL}).")
    p.add_argument("--skip-eval", action="store_true",
                   help="Generate outputs but skip evaluation.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip generation; evaluate existing raw results only.")
    return p.parse_args()


def _resolve_api_key(provider_model: str) -> str:
    """Look up the API key env var for the given provider and return its value."""
    provider = provider_of(provider_model)
    env_var  = PROVIDER_ENV_VARS.get(provider)
    if not env_var:
        logger.error("Unknown provider '%s' in model string '%s'.", provider, provider_model)
        sys.exit(1)
    key = os.environ.get(env_var)
    if not key:
        logger.error("%s is not set (required for provider '%s').", env_var, provider)
        sys.exit(1)
    return key


async def run_generation(
    gen_client: LLMClient,
    conditions: list[str],
    max_cases: Optional[int],
    k_runs: int,
    critique_rounds: int,
) -> list[dict]:
    """Dispatch all generation tasks concurrently and return raw results."""
    all_cases = load_cases(max_cases)
    tasks = []
    for cond in conditions:
        for case in all_cases:
            for run in range(k_runs):
                if cond == "anchored":
                    tasks.append(run_anchored(gen_client, case, run))
                elif cond == "critique_clinician":
                    tasks.append(run_critique_clinician(gen_client, case, run))
                elif cond == "critique_llm":
                    tasks.append(run_critique_llm(gen_client, case, run, critique_rounds))
                elif cond == "critique_combined":
                    tasks.append(run_critique_combined(gen_client, case, run, critique_rounds))
    logger.info("Dispatching %d generation tasks...", len(tasks))
    results = await asyncio.gather(*tasks)
    logger.info("Generation complete.")
    return list(results)


async def run_evaluation(
    eval_client: LLMClient,
    raw_results: list[dict],
    max_cases: Optional[int],
) -> list[dict]:
    """Dispatch all evaluation tasks concurrently and return eval records."""
    all_cases = load_cases(max_cases)
    case_meta = {
        c.case_id: {"final_diagnosis": c.final_diagnosis, "vignette": c.vignette}
        for c in all_cases
    }
    tasks = [
        evaluate_result(
            eval_client,
            r,
            case_meta[r["case_id"]]["final_diagnosis"],
            case_meta[r["case_id"]]["vignette"],
        )
        for r in raw_results
        if r["case_id"] in case_meta
    ]
    logger.info("Dispatching %d evaluation tasks...", len(tasks))
    eval_results = await asyncio.gather(*tasks)
    logger.info("Evaluation complete.")
    return list(eval_results)


async def main() -> None:
    args = parse_args()

    # Build clients — key lookup is skipped for the role being bypassed
    gen_client  = None
    eval_client = None

    if not args.eval_only:
        gen_client = make_client(args.gen_model, _resolve_api_key(args.gen_model))
        logger.info("Generation model: %s", args.gen_model)

    if not args.skip_eval:
        eval_client = make_client(args.eval_model, _resolve_api_key(args.eval_model))
        logger.info("Evaluation model: %s", args.eval_model)

    conditions = [args.condition] if args.condition else CONDITIONS

    # --- Generation ---
    if not args.eval_only:
        await run_generation(gen_client, conditions, args.cases, args.runs, args.critique_rounds)

    # --- Evaluation ---
    if not args.skip_eval:
        # Collect all raw results on disk (not just current session) so --eval-only works.
        raw_results = []
        for f in sorted(RAW_DIR.rglob("*.json")):
            r = json.loads(f.read_text())
            if r["condition"] not in conditions:
                continue
            raw_results.append(r)

        if args.cases is not None:
            allowed = {c.case_id for c in load_unique_cases(args.cases)}
            raw_results = [r for r in raw_results if r["case_id"] in allowed]

        raw_results = [r for r in raw_results if r["run"] < args.runs]

        await run_evaluation(eval_client, raw_results, args.cases)

        all_eval = collect_all_eval_results()
        csv_path = write_summary_csv(all_eval)
        logger.info("Summary CSV: %s", csv_path)

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
