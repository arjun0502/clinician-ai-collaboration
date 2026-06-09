"""
Condition runners and evaluation logic for the k-run experiment.

Four experimental conditions — all receive the clinician's assessment alongside
the clinical vignette, with two variants (helpful / harmful clinician input):

  anchored           — 1 call:       generate directly after seeing clinician input
  critique_clinician — 2 calls:      critique clinician input, then generate
  critique_llm       — 1 + 2N calls: generate initial response, then N rounds of self-critique
  critique_combined  — 2 + 2N calls: critique clinician input, generate, then N rounds of self-critique

Generation and evaluation models are configurable via --gen-model / --eval-model CLI args.

Raw outputs: results/raw/{condition}/{case_id}_{variant}_run{k}.json
Eval outputs: results/eval/{condition}/{case_id}_{variant}_run{k}.json
"""

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Optional

from src.clients import LLMClient
from src.config import (
    CONCURRENCY,
    CRITIQUE_ROUNDS,
    EVAL_DIR,
    HARM_LEVELS,
    RAW_DIR,
    RESULTS_DIR,
)
from src.data import Case, load_harm_table_by_case
from src.prompts import (
    EVALUATOR_SYSTEM,
    PHYSICIAN_SYSTEM,
    anchored_user,
    critique_clinician_call_a_user,
    critique_clinician_call_b_user,
    critique_llm_call_b_user,
    critique_llm_call_c_user,
    eval_differential_user,
    eval_harm_reproduction_user,
)

logger = logging.getLogger(__name__)

# Module-level singletons to avoid re-creating on every call
_semaphore: Optional[asyncio.Semaphore] = None
_harm_table: Optional[dict] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(CONCURRENCY)
    return _semaphore


def _get_harm_table() -> dict:
    """Lazily load and cache the ground-truth harm labels keyed by case_id."""
    global _harm_table
    if _harm_table is None:
        _harm_table = load_harm_table_by_case()
    return _harm_table


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------

async def _chat(
    client: LLMClient,
    messages: list[dict],
    json_mode: bool = False,
) -> str:
    """Make a single chat completion call, respecting the concurrency semaphore."""
    async with _get_semaphore():
        return await client.chat(messages, json_mode=json_mode)


async def _generate(client: LLMClient, user_prompt: str) -> dict:
    """Call the generation model; return parsed {differential_diagnoses, next_steps}."""
    messages = [
        {"role": "system", "content": PHYSICIAN_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    raw = await _chat(client, messages, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Generation JSON parse failed: %s", raw[:200])
        return {"differential_diagnoses": [], "next_steps": []}


async def _judge(client: LLMClient, user_prompt: str) -> dict:
    """Call the judge/eval model; return parsed JSON."""
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    raw = await _chat(client, messages, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Eval JSON parse failed: %s", raw[:200])
        return {}


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _raw_path(condition: str, case_id: str, variant: str, run: int, suffix: str = "") -> Path:
    return RAW_DIR / condition / f"{case_id}_{variant}_run{run}{suffix}.json"


def _eval_path(condition: str, case_id: str, variant: str, run: int, suffix: str = "") -> Path:
    return EVAL_DIR / condition / f"{case_id}_{variant}_run{run}{suffix}.json"


def _load_raw(condition: str, case_id: str, variant: str, run: int, suffix: str = "") -> Optional[dict]:
    p = _raw_path(condition, case_id, variant, run, suffix)
    return json.loads(p.read_text()) if p.exists() else None


def _save_raw(result: dict) -> None:
    suffix = result.get("_path_suffix", "")
    p = _raw_path(result["condition"], result["case_id"], result["variant"], result["run"], suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    # _path_suffix is internal bookkeeping only — don't write it to the JSON file
    result_to_write = {k: v for k, v in result.items() if k != "_path_suffix"}
    p.write_text(json.dumps(result_to_write, indent=2))


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

async def run_anchored(client: LLMClient, case: Case, run: int) -> dict:
    """
    Anchored (baseline): one call.
    AI sees vignette + clinician input and generates directly.
    """
    cached = _load_raw("anchored", case.case_id, case.variant, run)
    if cached:
        return cached

    logger.info("anchored | %s | %s | run=%d", case.case_id, case.variant, run)
    parsed = await _generate(
        client,
        anchored_user(case.vignette, case.clinician_differential, case.clinician_next_steps),
    )

    result = {
        "condition": "anchored",
        "case_id": case.case_id,
        "variant": case.variant,
        "run": run,
        "intermediate": None,     # no intermediate step for this condition
        "initial_response": None, # no initial response for this condition
        "differential_diagnoses": parsed.get("differential_diagnoses", []),
        "next_steps": parsed.get("next_steps", []),
    }
    _save_raw(result)
    return result


async def run_critique_clinician(client: LLMClient, case: Case, run: int) -> dict:
    """
    Critique-Clinician: two calls.
    Call A: AI critiques the clinician's differential and next steps.
    Call B: AI generates its response given the vignette + clinician input + critique.
    """
    cached = _load_raw("critique_clinician", case.case_id, case.variant, run)
    if cached:
        return cached

    logger.info("critique_clinician | %s | %s | run=%d", case.case_id, case.variant, run)

    # Call A: critique the clinician's assessment
    critique_messages = [
        {"role": "system", "content": PHYSICIAN_SYSTEM},
        {
            "role": "user",
            "content": critique_clinician_call_a_user(
                case.vignette,
                case.clinician_differential,
                case.clinician_next_steps,
            ),
        },
    ]
    critique_text = await _chat(client,critique_messages, json_mode=False)

    # Call B: generate final response informed by the critique
    parsed = await _generate(
        client,
        critique_clinician_call_b_user(
            case.vignette,
            case.clinician_differential,
            case.clinician_next_steps,
            critique_text,
        ),
    )

    result = {
        "condition": "critique_clinician",
        "case_id": case.case_id,
        "variant": case.variant,
        "run": run,
        "intermediate": critique_text,  # clinician critique from Call A
        "initial_response": None,
        "differential_diagnoses": parsed.get("differential_diagnoses", []),
        "next_steps": parsed.get("next_steps", []),
    }
    _save_raw(result)
    return result


async def run_critique_llm(
    client: LLMClient,
    case: Case,
    run: int,
    critique_rounds: int = CRITIQUE_ROUNDS,
) -> dict:
    """
    Critique-LLM: 1 + (2 × critique_rounds) sequential calls.

    Step 0:       Initial anchored generation (same prompt as anchored condition).
    Each round:   (a) AI critiques its current response, including reflecting on
                      clinician influence. (b) AI regenerates given current response
                      + critique.

    critique_rounds=1 (default): one critique-then-regenerate cycle (3 calls total).
    critique_rounds=N:           N critique-then-regenerate cycles (1 + 2N calls total).

    Results are cached per (condition, case, variant, run, critique_rounds) so that
    different round counts are stored independently and don't overwrite each other.
    """
    # Include critique_rounds in the filename so configs don't collide in cache
    suffix = f"_r{critique_rounds}"
    cached = _load_raw("critique_llm", case.case_id, case.variant, run, suffix)
    if cached:
        return cached

    logger.info(
        "critique_llm | %s | %s | run=%d | rounds=%d",
        case.case_id, case.variant, run, critique_rounds,
    )

    # Step 0: initial anchored generation
    initial_parsed = await _generate(
        client,
        anchored_user(case.vignette, case.clinician_differential, case.clinician_next_steps),
    )
    current_diff  = initial_parsed.get("differential_diagnoses", [])
    current_steps = initial_parsed.get("next_steps", [])

    # Iterate for the requested number of critique-then-regenerate rounds
    rounds_log = []
    for round_idx in range(critique_rounds):
        # Critique current response
        critique_messages = [
            {"role": "system", "content": PHYSICIAN_SYSTEM},
            {
                "role": "user",
                "content": critique_llm_call_b_user(
                    case.vignette,
                    case.clinician_differential,
                    case.clinician_next_steps,
                    current_diff,
                    current_steps,
                ),
            },
        ]
        critique_text = await _chat(client,critique_messages, json_mode=False)

        # Regenerate based on current response + critique
        regenerated = await _generate(
            client,
            critique_llm_call_c_user(
                case.vignette,
                case.clinician_differential,
                case.clinician_next_steps,
                current_diff,
                current_steps,
                critique_text,
            ),
        )
        current_diff  = regenerated.get("differential_diagnoses", [])
        current_steps = regenerated.get("next_steps", [])

        rounds_log.append({
            "round": round_idx + 1,
            "critique": critique_text,
            "differential_diagnoses": current_diff,
            "next_steps": current_steps,
        })

    result = {
        "condition": "critique_llm",
        "case_id": case.case_id,
        "variant": case.variant,
        "run": run,
        "critique_rounds": critique_rounds,
        "initial_response": {                       # generation before any critique
            "differential_diagnoses": initial_parsed.get("differential_diagnoses", []),
            "next_steps": initial_parsed.get("next_steps", []),
        },
        "rounds": rounds_log,                       # one entry per critique-regenerate cycle
        "intermediate": rounds_log[-1]["critique"],  # last critique (for consistency)
        "differential_diagnoses": current_diff,     # final output after all rounds
        "next_steps": current_steps,
        "_path_suffix": suffix,                     # used by _save_raw, not written to file
    }
    _save_raw(result)
    return result


# ---------------------------------------------------------------------------
# Condition 4 — Critique-Combined
# ---------------------------------------------------------------------------

async def run_critique_combined(
    client: LLMClient,
    case: Case,
    run: int,
    critique_rounds: int = CRITIQUE_ROUNDS,
) -> dict:
    """
    Critique-Combined: 2 + (2 × critique_rounds) sequential calls.

    Chains both critique strategies:
      Step 0a: Critique the clinician's assessment (critique_clinician_call_a_user).
      Step 0b: Generate initial response informed by clinician critique (critique_clinician_call_b_user).
      Each round: (a) AI critiques its own current response. (b) AI regenerates.

    This tests whether addressing clinician anchoring first, then LLM self-anchoring,
    produces better outcomes than either intervention alone.
    """
    suffix = f"_r{critique_rounds}"
    cached = _load_raw("critique_combined", case.case_id, case.variant, run, suffix)
    if cached:
        return cached

    logger.info(
        "critique_combined | %s | %s | run=%d | rounds=%d",
        case.case_id, case.variant, run, critique_rounds,
    )

    # Step 0a: critique the clinician's assessment
    clinician_critique_messages = [
        {"role": "system", "content": PHYSICIAN_SYSTEM},
        {
            "role": "user",
            "content": critique_clinician_call_a_user(
                case.vignette,
                case.clinician_differential,
                case.clinician_next_steps,
            ),
        },
    ]
    clinician_critique_text = await _chat(client,clinician_critique_messages, json_mode=False)

    # Step 0b: generate initial response informed by clinician critique
    initial_parsed = await _generate(
        client,
        critique_clinician_call_b_user(
            case.vignette,
            case.clinician_differential,
            case.clinician_next_steps,
            clinician_critique_text,
        ),
    )
    current_diff  = initial_parsed.get("differential_diagnoses", [])
    current_steps = initial_parsed.get("next_steps", [])

    # N rounds of LLM self-critique and regeneration
    rounds_log = []
    for round_idx in range(critique_rounds):
        self_critique_messages = [
            {"role": "system", "content": PHYSICIAN_SYSTEM},
            {
                "role": "user",
                "content": critique_llm_call_b_user(
                    case.vignette,
                    case.clinician_differential,
                    case.clinician_next_steps,
                    current_diff,
                    current_steps,
                ),
            },
        ]
        self_critique_text = await _chat(client,self_critique_messages, json_mode=False)

        regenerated = await _generate(
            client,
            critique_llm_call_c_user(
                case.vignette,
                case.clinician_differential,
                case.clinician_next_steps,
                current_diff,
                current_steps,
                self_critique_text,
            ),
        )
        current_diff  = regenerated.get("differential_diagnoses", [])
        current_steps = regenerated.get("next_steps", [])

        rounds_log.append({
            "round": round_idx + 1,
            "critique": self_critique_text,
            "differential_diagnoses": current_diff,
            "next_steps": current_steps,
        })

    result = {
        "condition": "critique_combined",
        "case_id": case.case_id,
        "variant": case.variant,
        "run": run,
        "critique_rounds": critique_rounds,
        "clinician_critique": clinician_critique_text,  # critique of clinician input (step 0a)
        "initial_response": {                            # generation after clinician critique (step 0b)
            "differential_diagnoses": initial_parsed.get("differential_diagnoses", []),
            "next_steps": initial_parsed.get("next_steps", []),
        },
        "rounds": rounds_log,                            # one entry per self-critique cycle
        "intermediate": rounds_log[-1]["critique"],      # last self-critique (for consistency)
        "differential_diagnoses": current_diff,          # final output after all rounds
        "next_steps": current_steps,
        "_path_suffix": suffix,
    }
    _save_raw(result)
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate_result(
    eval_client: LLMClient,
    raw: dict,
    final_diagnosis: str,
    vignette: str,
) -> dict:
    """
    Evaluate a single generation result using the OpenAI judge.

    Two parallel sub-tasks:
      1. Diagnostic accuracy: did the correct diagnosis appear in the differential?
      2. Harm reproduction: did any next step reproduce a pre-labeled harmful step?
    """
    condition      = raw["condition"]
    case_id        = raw["case_id"]
    variant        = raw["variant"]
    run            = raw["run"]
    critique_rounds = raw.get("critique_rounds")

    # Use the same filename suffix as the raw file so eval and raw stay in sync
    suffix   = f"_r{critique_rounds}" if critique_rounds is not None else ""
    out_path = _eval_path(condition, case_id, variant, run, suffix)
    if out_path.exists():
        return json.loads(out_path.read_text())

    logger.info("eval | %s | %s | %s | run=%d", condition, case_id, variant, run)

    differential         = raw.get("differential_diagnoses", [])
    next_steps           = raw.get("next_steps", [])
    ground_truth_harmful = _get_harm_table().get(case_id, [])

    async def _empty_harm() -> dict:
        return {"matches": []}

    # Run differential and harm evaluation concurrently
    diff_task = asyncio.create_task(
        _judge(eval_client, eval_differential_user(final_diagnosis, differential))
    )
    harm_coro = (
        _judge(eval_client, eval_harm_reproduction_user(ground_truth_harmful, next_steps))
        if next_steps and ground_truth_harmful
        else _empty_harm()
    )
    harm_task = asyncio.create_task(harm_coro)
    diff_result, harm_result = await asyncio.gather(diff_task, harm_task)

    # Count reproduced steps by severity level
    matches     = harm_result.get("matches", [])
    harm_counts = {level: 0 for level in HARM_LEVELS}
    for m in matches:
        if m.get("reproduced"):
            level = m.get("harm_level", "None")
            if level in harm_counts:
                harm_counts[level] += 1

    eval_record = {
        "condition": condition,
        "case_id": case_id,
        "variant": variant,
        "run": run,
        "critique_rounds": critique_rounds,
        "final_diagnosis": final_diagnosis,
        "final_diagnosis_included": diff_result.get("included", False),
        "matching_diagnosis": diff_result.get("matching_diagnosis"),
        "differential_reasoning": diff_result.get("reasoning"),
        "total_ground_truth_harmful": len(ground_truth_harmful),
        "harm_reproductions": matches,
        "harm_counts": harm_counts,
        "total_reproduced": sum(harm_counts.values()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_record, indent=2))
    return eval_record


# ---------------------------------------------------------------------------
# Results collection and CSV output
# ---------------------------------------------------------------------------

def collect_all_eval_results() -> list[dict]:
    """Read all eval JSON files from disk and return as a flat list."""
    return [json.loads(f.read_text()) for f in sorted(EVAL_DIR.rglob("*.json"))]


_SUMMARY_FIELDNAMES = [
    "condition", "case_id", "variant", "run",
    "final_diagnosis", "final_diagnosis_included", "matching_diagnosis",
    "total_ground_truth_harmful", "total_reproduced",
    "reproduced_mild", "reproduced_moderate", "reproduced_severe", "reproduced_death",
    "any_severe_or_death_reproduced",
]


def write_summary_csv(eval_results: list[dict]) -> Path:
    """Write results/summary.csv — one row per (condition, case_id, variant, run)."""
    out_path = RESULTS_DIR / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for r in eval_results:
            counts          = r.get("harm_counts", {})
            severe_or_death = counts.get("Severe", 0) + counts.get("Death", 0)
            writer.writerow({
                "condition": r["condition"],
                "case_id": r["case_id"],
                "variant": r["variant"],
                "run": r["run"],
                "final_diagnosis": r.get("final_diagnosis", ""),
                "final_diagnosis_included": r.get("final_diagnosis_included", False),
                "matching_diagnosis": r.get("matching_diagnosis", ""),
                "total_ground_truth_harmful": r.get("total_ground_truth_harmful", 0),
                "total_reproduced": r.get("total_reproduced", 0),
                "reproduced_mild": counts.get("Mild", 0),
                "reproduced_moderate": counts.get("Moderate", 0),
                "reproduced_severe": counts.get("Severe", 0),
                "reproduced_death": counts.get("Death", 0),
                "any_severe_or_death_reproduced": int(severe_or_death > 0),
            })
    logger.info("Summary written to %s (%d rows)", out_path, len(eval_results))
    return out_path
