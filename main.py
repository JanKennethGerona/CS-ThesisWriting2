"""
main.py — Full multi-commodity MILP pipeline entry point.

Runs the complete pipeline:
  1. Load data (with provisional status detection)
  2. Validate data integrity
  3. Primary solve (multi-commodity deprivation-cost minimization)
  4. Compute phi_j post-solve (min_r phi_{jr})
  5. Verify solution (bounds, under-approximation, limiting-resource check)
  6. Baseline comparison (proportional allocation)
  7. Post-hoc fairness evaluation
  8. Alternate-objective re-solves
  9. Sensitivity sweep
 10. Disaster scenarios
 11. Tangent approximation experiment
 12. Export results to CSV and JSON

Usage:
    python main.py
"""

import os
import sys
import json
import csv
import math
from typing import Any

# Add src/ directory to sys.path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pyomo.environ as pyo

from data_loader import load_data, print_data_summary
from model import (
    build_model, build_tangent_segments, solve_model,
    verify_and_report,
)
from davao_rap_milp import (
    compute_overall_fulfillment, sensitivity_sweep,
)
from fairness import run_fairness_comparison, print_fairness_results
from scenarios import get_all_scenarios, apply_scenario
from validation import compare_breakpoint_counts

# Output directory
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pipeline(xlsx_path=None, mode="development"):
    """Run the full multi-commodity MILP pipeline."""

    print("=" * 70)
    print("  Multi-Commodity Two-Stage MILP -- Davao City Resource Allocation")
    print("  (Two-stage deterministic MILP: hub activation + resource allocation)")
    print("=" * 70)

    # -- 1. Load data --
    print("\n[1/11] Loading data...")
    data = load_data(xlsx_path, mode=mode)
    print_data_summary(data)

    hubs = data["hubs"]
    barangays = data["barangays"]
    resources = data["resources"]
    D = data["D"]
    S = data["S"]
    P = data["P"]
    params = data["params"]
    is_provisional = data["is_provisional"]
    lambda_dict = params["lambda_"]

    # -- 2. Build tangent segments --
    print("\n[2/11] Building tangent segments (21 breakpoints per resource)...")
    segments = build_tangent_segments(lambda_dict, num_breakpoints=21)
    for r in resources:
        print(f"  {r}: {len(segments[r])} segments "
              f"(lambda={lambda_dict[r]}, PROVISIONAL)")

    # -- 3. Primary solve --
    print("\n[3/11] Building and solving primary model...")
    model: Any = build_model(
        hubs, barangays, resources, D, S, P, params, segments
    )
    result, solver_used = solve_model(model)
    status = str(result.solver.termination_condition)
    print(f"  Solver: {solver_used}")
    print(f"  Status: {status}")
    print(f"  Objective (vulnerability-weighted deprivation): "
          f"{pyo.value(model.OBJ):.4f}")

    if status != "optimal":
        print("\n  PRIMARY SOLVE FAILED -- model is infeasible or unbounded.")
        print("  Check supply capacities vs demand + anti-starvation floor.")
        return

    # -- 4. Compute phi_j post-solve --
    print("\n[4/11] Computing overall fulfillment (post-solve)...")
    phi_j, limiting = compute_overall_fulfillment(model, resources)

    # Summary statistics
    print(f"  Min overall fulfillment: {min(phi_j.values()):.4f}")
    print(f"  Max overall fulfillment: {max(phi_j.values()):.4f}")
    print(f"  Mean overall fulfillment: "
          f"{sum(phi_j.values())/len(phi_j):.4f}")

    # Per-resource summary
    for r in resources:
        vals = [pyo.value(model.phi_r[j, r]) for j in model.J]
        print(f"  {r}: min={min(vals):.4f}  max={max(vals):.4f}  "
              f"mean={sum(vals)/len(vals):.4f}")

    # Limiting resource distribution
    from collections import Counter
    lim_counts = Counter(limiting.values())
    print(f"  Limiting resource distribution: {dict(lim_counts)}")

    # -- 5. Verify --
    print("\n[5/11] Verifying solution...")
    phi_j, limiting, val_results = verify_and_report(
        model, lambda_dict, resources
    )

    if not val_results["passed"]:
        print("\n  SOLUTION VERIFICATION FAILED -- see errors above.")
        print("  Results should not be trusted.")

    # -- 6. Active hubs --
    active_hubs = [i for i in model.I if pyo.value(model.y[i]) > 0.5]
    print(f"\n  Active hubs ({len(active_hubs)}/{len(hubs)}): {active_hubs}")

    # -- 7. Fairness comparison --
    print("\n[6/11] Running fairness comparison...")
    fairness_results = run_fairness_comparison(
        hubs, barangays, resources, D, S, P, params,
        segments, phi_j, is_provisional
    )
    print_fairness_results(fairness_results)

    # -- 8. Sensitivity sweep --
    print("\n[7/11] Sensitivity sweep...")
    sweep = sensitivity_sweep(
        hubs, barangays, resources, D, S, P, params, segments
    )
    print(f"\n{'Factor':>8} {'Status':>10} {'Objective':>12} "
          f"{'Min phi':>8} {'Mean phi':>8}")
    print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")
    for row in sweep:
        if row["feasible"]:
            print(f"{row['supply_factor']:>8.1f} {'optimal':>10} "
                  f"{row['objective']:>12.2f} "
                  f"{row['min_phi_overall']:>8.4f} "
                  f"{row['mean_phi_overall']:>8.4f}")
        else:
            print(f"{row['supply_factor']:>8.1f} {'INFEASIBLE':>10} "
                  f"{'---':>12} {'---':>8} {'---':>8}")

    # -- 9. Disaster scenarios --
    print("\n[8/11] Running disaster scenarios...")
    scenario_results = []
    for scenario in get_all_scenarios():
        D_sc, S_sc = apply_scenario(scenario, D, S, barangays, resources)
        segs_sc = build_tangent_segments(lambda_dict, num_breakpoints=21)
        m_sc: Any = build_model(
            hubs, barangays, resources, D_sc, S_sc, P, params, segs_sc
        )
        try:
            res_sc, _ = solve_model(m_sc)
            feasible = str(res_sc.solver.termination_condition) == "optimal"
        except Exception:
            feasible = False

        if feasible:
            phi_sc, lim_sc = compute_overall_fulfillment(m_sc, resources)
            sc_result = {
                "name": scenario["name"],
                "feasible": True,
                "objective": pyo.value(m_sc.OBJ),
                "min_phi": min(phi_sc.values()),
                "mean_phi": sum(phi_sc.values()) / len(phi_sc),
            }
        else:
            sc_result = {
                "name": scenario["name"],
                "feasible": False,
                "objective": None,
                "min_phi": None,
                "mean_phi": None,
            }
        scenario_results.append(sc_result)
        status_str = f"obj={sc_result['objective']:.2f}  min_phi={sc_result['min_phi']:.4f}" if feasible else "INFEASIBLE"
        print(f"  {scenario['name']}: {status_str}")

    # -- 10. Tangent approximation experiment --
    print("\n[9/11] Tangent approximation experiment...")
    for r, lam in lambda_dict.items():
        print(f"\n  Resource: {r} (lambda={lam})")
        bp_results = compare_breakpoint_counts(lam, counts=(5, 11, 21, 41))
        print(f"  {'Breakpoints':>12} {'Max Abs Err':>12} {'Max Rel Err':>12} "
              f"{'Avg Abs Err':>12}")
        for n, bp_stat in bp_results.items():
            print(f"  {n:>12} {bp_stat['max_abs_error']:>12.6f} "
                  f"{bp_stat['max_rel_error']:>12.6f} "
                  f"{bp_stat['avg_abs_error']:>12.6f}")

    # -- 11. Export --
    print("\n[10/11] Exporting results...")
    ensure_results_dir()

    # CSV export
    csv_path = os.path.join(RESULTS_DIR, "barangay_allocations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if is_provisional:
            f.write("# WARNING: DEMONSTRATION ONLY -- PLACEHOLDER "
                    "VULNERABILITY DATA\n")
            f.write("# Results must NOT be presented as thesis findings.\n")

        writer = csv.writer(f)
        header = ["Barangay", "District"]
        for r in resources:
            header.extend([
                f"phi_{r}", f"z_{r}", f"w_{r}",
            ])
        header.extend(["phi_overall", "limiting_resource"])
        writer.writerow(header)

        df = data["barangay_details"]
        for j in barangays:
            district = df.loc[df["Barangay"] == j, "District"].iloc[0]
            row = [j, district]
            for r in resources:
                row.extend([
                    round(pyo.value(model.phi_r[j, r]), 6),
                    round(pyo.value(model.z[j, r]), 6),
                    round(pyo.value(model.w[j, r]), 2),
                ])
            row.extend([round(phi_j[j], 6), limiting[j]])
            writer.writerow(row)
    print(f"  CSV: {csv_path}")

    # JSON export
    json_path = os.path.join(RESULTS_DIR, "summary_metrics.json")
    summary = {
        "data_status": ("DEMONSTRATION ONLY -- PLACEHOLDER VULNERABILITY DATA"
                        if is_provisional
                        else "VALIDATED RESEARCH RUN"),
        "data_mode": data["data_mode"],
        "is_provisional": is_provisional,
        "solver": solver_used,
        "objective": pyo.value(model.OBJ),
        "resources": resources,
        "barangay_count": len(barangays),
        "hub_count": len(hubs),
        "active_hubs": len(active_hubs),
        "hmax": params["Hmax"],
        "eps_min": params["eps_min"],
        "lambda": lambda_dict,
        "lambda_note": "PROVISIONAL SENSITIVITY-ANALYSIS VALUES, NOT CALIBRATED",
        "fulfillment_summary": {
            "overall": {
                "min": min(phi_j.values()),
                "max": max(phi_j.values()),
                "mean": sum(phi_j.values()) / len(phi_j),
            },
        },
        "fairness": {
            "min_phi": fairness_results["primary_fairness"]["phi_min (Max-Min proxy)"],
            "nash_welfare_score": fairness_results["primary_fairness"]["nash_welfare_score"],
            "disparity_violations": fairness_results["primary_fairness"]["disparity_count"],
        },
        "sensitivity_sweep": sweep,
        "scenario_results": scenario_results,
        "validation_passed": val_results["passed"],
    }
    # Add per-resource summary
    for r in resources:
        vals = [pyo.value(model.phi_r[j, r]) for j in model.J]
        summary["fulfillment_summary"][r] = {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # -- Done --
    print(f"\n[11/11] Pipeline complete.")
    if is_provisional:
        print("\n" + "=" * 70)
        print("  WARNING: DEMONSTRATION ONLY -- PLACEHOLDER VULNERABILITY DATA")
        print("  These results must NOT be presented as thesis findings.")
        print("  Replace placeholder P_j with real MGB hazard-derived weights")
        print("  and re-run in 'research' mode before reporting.")
        print("=" * 70)

    return {
        "model": model,
        "phi_j": phi_j,
        "limiting": limiting,
        "fairness": fairness_results,
        "sweep": sweep,
        "scenarios": scenario_results,
        "validation": val_results,
    }


if __name__ == "__main__":
    mode = "development"
    xlsx_path = None

    # Simple arg parsing
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in ("development", "research"):
                mode = arg
            elif arg.endswith(".xlsx"):
                xlsx_path = arg

    run_pipeline(xlsx_path=xlsx_path, mode=mode)
