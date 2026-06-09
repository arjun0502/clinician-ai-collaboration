"""
Aggregate results from results/summary.csv and print avg/best/worst@k tables.

Metrics reported per (condition, clinician-input variant) group:
  - Diagnostic accuracy: fraction of cases where the correct diagnosis appeared
    in the AI's differential (higher is better)
  - Harm reproduction rate: fraction of pre-labeled harmful steps the AI
    independently recommended (lower is better)
  - % cases with severe/death harm: fraction of cases where the AI recommended
    at least one step rated Severe or Death severity (lower is better)

For each metric, avg/best/worst are computed across K independent runs.

Usage:
  python analyze.py
  python analyze.py --csv      # also saves results/aggregate.csv
  python analyze.py --verbose  # also prints per-run detail table
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SUMMARY_CSV   = Path("results/summary.csv")
AGGREGATE_CSV = Path("results/aggregate.csv")

# Display order and labels for the six (condition, variant) groups
GROUP_ORDER = [
    ("anchored",           "helpful"),
    ("anchored",           "harmful"),
    ("critique_clinician", "helpful"),
    ("critique_clinician", "harmful"),
    ("critique_llm",       "helpful"),
    ("critique_llm",       "harmful"),
    ("critique_combined",  "helpful"),
    ("critique_combined",  "harmful"),
]

GROUP_LABELS = {
    ("anchored",           "helpful"):  "Anchored — helpful",
    ("anchored",           "harmful"):  "Anchored — harmful",
    ("critique_clinician", "helpful"):  "Critique-Clinician — helpful",
    ("critique_clinician", "harmful"):  "Critique-Clinician — harmful",
    ("critique_llm",       "helpful"):  "Critique-LLM — helpful",
    ("critique_llm",       "harmful"):  "Critique-LLM — harmful",
    ("critique_combined",  "helpful"):  "Critique-Combined — helpful",
    ("critique_combined",  "harmful"):  "Critique-Combined — harmful",
}


def load_summary() -> list[dict]:
    with SUMMARY_CSV.open() as f:
        return list(csv.DictReader(f))


def compute_per_run_stats(rows: list[dict]) -> dict[tuple, dict]:
    """
    Group rows by (condition, variant, run) and compute per-run aggregate stats
    across all cases in each group (N=61 cases per run).
    """
    buckets: dict = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["variant"], int(r["run"]))
        buckets[key].append(r)

    out = {}
    for key, group in buckets.items():
        n          = len(group)
        dx_correct = sum(1 for r in group if r["final_diagnosis_included"] == "True")
        total_gt   = sum(int(r["total_ground_truth_harmful"]) for r in group)
        total_repr = sum(int(r["total_reproduced"]) for r in group)
        repr_mild  = sum(int(r["reproduced_mild"]) for r in group)
        repr_mod   = sum(int(r["reproduced_moderate"]) for r in group)
        repr_sev   = sum(int(r["reproduced_severe"]) for r in group)
        repr_death = sum(int(r["reproduced_death"]) for r in group)
        any_sev    = sum(int(r["any_severe_or_death_reproduced"]) for r in group)
        out[key] = {
            "n": n,
            "dx_correct": dx_correct,
            "dx_accuracy_pct":           round(100 * dx_correct / n, 1) if n else 0.0,
            "total_ground_truth_harmful": total_gt,
            "total_reproduced":           total_repr,
            "harm_reproduction_rate_pct": round(100 * total_repr / total_gt, 1) if total_gt else 0.0,
            "reproduced_mild":            repr_mild,
            "reproduced_moderate":        repr_mod,
            "reproduced_severe":          repr_sev,
            "reproduced_death":           repr_death,
            "pct_any_severe_or_death":    round(100 * any_sev / n, 1) if n else 0.0,
        }
    return out


def compute_across_runs_stats(per_run: dict[tuple, dict]) -> dict[tuple, dict]:
    """
    For each (condition, variant), aggregate the K per-run stats into
    avg / best@k / worst@k for each metric.

    Best/worst are defined per metric independently:
      - Dx accuracy:  best = max (higher is better)
      - Harm rate:    best = min (lower is better)
      - Sev/death:    best = min (lower is better)
    """
    cv_buckets: dict = defaultdict(list)
    for (cond, variant, _run), stats in per_run.items():
        cv_buckets[(cond, variant)].append(stats)

    out = {}
    for key, run_stats_list in cv_buckets.items():
        k         = len(run_stats_list)
        dx_vals   = [s["dx_accuracy_pct"] for s in run_stats_list]
        harm_vals = [s["harm_reproduction_rate_pct"] for s in run_stats_list]
        sev_vals  = [s["pct_any_severe_or_death"] for s in run_stats_list]
        out[key] = {
            "k_runs":                        k,
            "n_per_run":                     run_stats_list[0]["n"],
            "avg_dx_accuracy_pct":           round(sum(dx_vals) / k, 1),
            "best_dx_accuracy_pct":          max(dx_vals),
            "worst_dx_accuracy_pct":         min(dx_vals),
            "avg_harm_rate_pct":             round(sum(harm_vals) / k, 1),
            "best_harm_rate_pct":            min(harm_vals),
            "worst_harm_rate_pct":           max(harm_vals),
            "avg_pct_any_severe_or_death":   round(sum(sev_vals) / k, 1),
            "best_pct_any_severe_or_death":  min(sev_vals),
            "worst_pct_any_severe_or_death": max(sev_vals),
        }
    return out


def print_tables(agg: dict[tuple, dict], per_run: dict[tuple, dict], verbose: bool) -> None:
    # --- Table 1: Diagnostic Accuracy ---
    print()
    print("=== Diagnostic Accuracy (avg / best@k / worst@k) ===")
    print()
    w = [32, 4, 10, 10, 11]
    hdr = ["Group", "K", "Avg Dx%", "Best Dx%", "Worst Dx%"]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    print("-" * (sum(w) + 2 * (len(w) - 1)))
    for key in GROUP_ORDER:
        if key not in agg:
            continue
        g = agg[key]
        print("  ".join([
            GROUP_LABELS[key].ljust(w[0]),
            str(g["k_runs"]).ljust(w[1]),
            f"{g['avg_dx_accuracy_pct']}%".ljust(w[2]),
            f"{g['best_dx_accuracy_pct']}%".ljust(w[3]),
            f"{g['worst_dx_accuracy_pct']}%".ljust(w[4]),
        ]))

    # --- Table 2: Harm Reproduction ---
    print()
    print("=== Harm Reproduction Rate — lower is better (avg / best@k / worst@k) ===")
    print()
    w2 = [32, 4, 11, 11, 12, 14]
    hdr2 = ["Group", "K", "Avg Harm%", "Best(min)", "Worst(max)", "Avg %sev/death"]
    print("  ".join(str(h).ljust(w2[i]) for i, h in enumerate(hdr2)))
    print("-" * (sum(w2) + 2 * (len(w2) - 1)))
    for key in GROUP_ORDER:
        if key not in agg:
            continue
        g = agg[key]
        print("  ".join([
            GROUP_LABELS[key].ljust(w2[0]),
            str(g["k_runs"]).ljust(w2[1]),
            f"{g['avg_harm_rate_pct']}%".ljust(w2[2]),
            f"{g['best_harm_rate_pct']}%".ljust(w2[3]),
            f"{g['worst_harm_rate_pct']}%".ljust(w2[4]),
            f"{g['avg_pct_any_severe_or_death']}%".ljust(w2[5]),
        ]))

    # --- Table 3: Per-run detail (verbose only) ---
    if verbose:
        print()
        print("=== Per-Run Detail ===")
        print()
        w3 = [24, 10, 4, 6, 9, 10]
        hdr3 = ["Condition", "Variant", "Run", "N", "Dx%", "HarmRate%"]
        print("  ".join(str(h).ljust(w3[i]) for i, h in enumerate(hdr3)))
        print("-" * (sum(w3) + 2 * (len(w3) - 1)))
        for (cond, variant, run), s in sorted(per_run.items()):
            print("  ".join([
                cond.ljust(w3[0]),
                variant.ljust(w3[1]),
                str(run).ljust(w3[2]),
                str(s["n"]).ljust(w3[3]),
                f"{s['dx_accuracy_pct']}%".ljust(w3[4]),
                f"{s['harm_reproduction_rate_pct']}%".ljust(w3[5]),
            ]))
    print()


def save_aggregate_csv(agg: dict[tuple, dict]) -> None:
    """Save the across-runs aggregate stats to results/aggregate.csv."""
    fieldnames = [
        "group", "condition", "variant", "k_runs", "n_per_run",
        "avg_dx_accuracy_pct", "best_dx_accuracy_pct", "worst_dx_accuracy_pct",
        "avg_harm_rate_pct", "best_harm_rate_pct", "worst_harm_rate_pct",
        "avg_pct_any_severe_or_death", "best_pct_any_severe_or_death", "worst_pct_any_severe_or_death",
    ]
    AGGREGATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with AGGREGATE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in GROUP_ORDER:
            if key not in agg:
                continue
            condition, variant = key
            writer.writerow({
                "group": GROUP_LABELS[key],
                "condition": condition,
                "variant": variant,
                **agg[key],
            })
    print(f"Aggregate saved to {AGGREGATE_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true", help="Save results/aggregate.csv")
    parser.add_argument("--verbose", action="store_true", help="Print per-run detail table")
    args = parser.parse_args()

    if not SUMMARY_CSV.exists():
        print(f"No summary found at {SUMMARY_CSV}. Run 'python run.py' first.")
        return

    rows    = load_summary()
    per_run = compute_per_run_stats(rows)
    agg     = compute_across_runs_stats(per_run)

    print_tables(agg, per_run, verbose=args.verbose)

    if args.csv:
        save_aggregate_csv(agg)


if __name__ == "__main__":
    main()
