import csv
import json
import re
from dataclasses import dataclass
from typing import Optional

from src.config import ANSWERS_FILE, HARM_TABLE_FILE, VIGNETTE_FILE


@dataclass
class Case:
    case_id: str
    vignette: str
    final_diagnosis: str
    clinician_differential: list[str]
    clinician_next_steps: list[str]
    differential_type: str   # "helpful" | "not_helpful"
    next_steps_type: str     # "helpful" | "harmful"
    variant: str             # "helpful" | "harmful" (mirrors next_steps_type)


def _load_vignettes() -> dict[str, str]:
    """Parse the YAML vignette file without a YAML library.

    Format is: `case_id: |\\n  text...` with blank lines between entries.
    """
    text = VIGNETTE_FILE.read_text(encoding="utf-8")
    vignettes: dict[str, str] = {}

    # Split on top-level keys (lines that start with a non-space word followed by ': |')
    blocks = re.split(r"\n(?=\S)", text)
    for block in blocks:
        if not block.strip():
            continue
        match = re.match(r"^(\S+):\s*\|\s*\n(.*)", block, re.DOTALL)
        if match:
            key = match.group(1)
            body = match.group(2)
            # Strip common leading indent (2 spaces)
            lines = body.splitlines()
            dedented = "\n".join(
                line[2:] if line.startswith("  ") else line for line in lines
            )
            vignettes[key] = dedented.strip()

    return vignettes


def load_cases(max_cases: Optional[int] = None) -> list[Case]:
    """Return all Case objects (122 total: 61 unique cases × 2 variants).

    If max_cases is set, return the first max_cases *unique* case IDs (both variants each).
    """
    vignettes = _load_vignettes()

    with ANSWERS_FILE.open() as f:
        data = json.load(f)
    records = data["cases"]

    # Optionally limit to first N unique case IDs
    if max_cases is not None:
        seen: list[str] = []
        for r in records:
            if r["case_id"] not in seen:
                seen.append(r["case_id"])
            if len(seen) >= max_cases:
                break
        allowed = set(seen)
        records = [r for r in records if r["case_id"] in allowed]

    cases: list[Case] = []
    for r in records:
        cid = r["case_id"]
        vignette = vignettes.get(cid, "")
        variant = r["next_steps_type"]  # "helpful" or "harmful"
        cases.append(
            Case(
                case_id=cid,
                vignette=vignette,
                final_diagnosis=r["final_diagnosis"],
                clinician_differential=r["differential_diagnoses"],
                clinician_next_steps=r["next_steps"],
                differential_type=r["differential_type"],
                next_steps_type=r["next_steps_type"],
                variant=variant,
            )
        )

    return cases


def load_unique_cases(max_cases: Optional[int] = None) -> list[Case]:
    """Return one Case per unique case_id (the 'helpful' variant, for baseline).

    If a case only has one variant, returns that one.
    """
    all_cases = load_cases(max_cases)
    seen: set[str] = set()
    unique: list[Case] = []
    for c in all_cases:
        if c.case_id not in seen:
            seen.add(c.case_id)
            unique.append(c)
    return unique


def load_harm_table() -> dict[str, str]:
    """Return a dict mapping lowercase next_step string → harm_level."""
    table: dict[str, str] = {}
    with HARM_TABLE_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row["next_steps"].strip().lower()] = row["next_steps_harm_label"]
    return table


def load_harm_table_by_case() -> dict[str, list[dict]]:
    """Return {case_id: [{step, harm_level}, ...]} for non-None harm steps only.

    case_id is in 'case_XXXXX' format to match the rest of the codebase.
    """
    from collections import defaultdict
    table: dict = defaultdict(list)
    with HARM_TABLE_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["next_steps_harm_label"].strip() == "None":
                continue
            raw_id = row["case_id"].strip()
            case_id = raw_id if raw_id.startswith("case_") else f"case_{raw_id}"
            table[case_id].append({
                "step": row["next_steps"].strip(),
                "harm_level": row["next_steps_harm_label"].strip(),
            })
    return dict(table)
