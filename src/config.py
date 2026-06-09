from pathlib import Path

ROOT = Path(__file__).parent.parent

# Default models in "provider:model" format.
# Override at runtime with --gen-model / --eval-model CLI args.
# Supported providers: openai, gemini, anthropic
# Keeping generation and evaluation on different models avoids self-evaluation bias.
GEN_MODEL  = "gemini:gemini-3.1-flash-lite"
EVAL_MODEL = "openai:gpt-5.4-mini"

# Input data files
VIGNETTE_FILE   = ROOT / "shared_datasets" / "nejm_case_vignette.yaml"
ANSWERS_FILE    = ROOT / "shared_datasets" / "nejm_case_answers.json"
HARM_TABLE_FILE = ROOT / "shared_datasets" / "harm_classes_tbl.csv"

# Output directories
RESULTS_DIR = ROOT / "results"
RAW_DIR     = RESULTS_DIR / "raw"    # raw LLM generation outputs
EVAL_DIR    = RESULTS_DIR / "eval"   # per-case evaluation outputs

# Experiment settings
RUNS           = 3  # number of independent runs per condition/case
CRITIQUE_ROUNDS = 1  # number of critique-then-regenerate rounds for critique_llm
                     # (each round: LLM critiques its current response, then regenerates)
CONDITIONS     = ["anchored", "critique_clinician", "critique_llm", "critique_combined"]

# Max concurrent API calls (applies to both generation and evaluation)
CONCURRENCY = 20

HARM_LEVELS = ["None", "Mild", "Moderate", "Severe", "Death"]
