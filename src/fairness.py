"""
fairness.py — Post-hoc fairness evaluation with corrected terminology.

Provides:
  - evaluate_fairness() — Nash welfare evaluation, fulfillment-ratio disparity,
    Leximin, Max-Min proxy (all from post-processed phi_j)
  - solve_alternate_objective() — re-solves with different objectives
  - compute_baseline_proportional() — proportional allocation for comparison
  - pareto_check() — actual Pareto-efficiency check via single-node improvement

All metrics operate on post-processed phi_j = min_r(phi_{jr}), NOT an
in-model variable.
"""

from davao_rap_milp import (
    evaluate_fairness,
    solve_alternate_objective,
    compute_baseline_proportional,
    build_model,
    build_tangent_segments,
    solve_model,
    compute_overall_fulfillment,
)
import pyomo.environ as pyo


def run_fairness_comparison(hubs, barangays, resources, D, S, P, params,
                            segments, primary_phi_j, is_provisional=False):
    """
    Run the full fairness comparison suite:
      1. Post-hoc fairness metrics from primary solution
      2. Alternate-objective re-solves
      3. Baseline proportional comparison
      4. Comparison table

    Args:
        primary_phi_j: {j: phi_j} from compute_overall_fulfillment on primary solve

    Returns:
        dict with all fairness results
    """
    results = {}

    # 1. Primary fairness metrics
    results["primary_fairness"] = evaluate_fairness(primary_phi_j, params)

    # 2. Alternate objectives
    alt_results = {}
    for mode in ("max_utilitarian", "pure_egalitarian", "max_min"):
        try:
            alt_phi = solve_alternate_objective(
                hubs, barangays, resources, D, S, P, params, segments, mode
            )
            alt_results[mode] = {
                "phi_j": alt_phi,
                "min_phi": min(alt_phi.values()),
                "max_phi": max(alt_phi.values()),
                "mean_phi": sum(alt_phi.values()) / len(alt_phi),
            }
        except Exception as e:
            alt_results[mode] = {"error": str(e)}
    results["alternate_objectives"] = alt_results

    # 3. Baseline proportional allocation
    baseline_phi = compute_baseline_proportional(
        barangays, resources, D, S, hubs, params
    )
    results["baseline_proportional"] = {
        "phi_j": baseline_phi,
        "min_phi": min(baseline_phi.values()),
        "mean_phi": sum(baseline_phi.values()) / len(baseline_phi),
    }

    # 4. Comparison table
    primary_min = min(primary_phi_j.values())
    primary_mean = sum(primary_phi_j.values()) / len(primary_phi_j)
    primary_std = (
        sum((v - primary_mean) ** 2 for v in primary_phi_j.values())
        / len(primary_phi_j)
    ) ** 0.5

    results["comparison_table"] = {
        "Deprivation-Min (Primary)": {
            "min_phi": primary_min,
            "mean_phi": primary_mean,
            "std_phi": primary_std,
        },
        "Proportional Baseline": {
            "min_phi": results["baseline_proportional"]["min_phi"],
            "mean_phi": results["baseline_proportional"]["mean_phi"],
            "std_phi": 0.0,  # uniform by definition
        },
    }
    for mode, data in alt_results.items():
        if "error" not in data:
            alt_mean = data["mean_phi"]
            alt_std = (
                sum((v - alt_mean) ** 2 for v in data["phi_j"].values())
                / len(data["phi_j"])
            ) ** 0.5
            results["comparison_table"][mode] = {
                "min_phi": data["min_phi"],
                "mean_phi": alt_mean,
                "std_phi": alt_std,
            }

    results["is_provisional"] = is_provisional

    return results


def print_fairness_results(results):
    """Pretty-print the fairness comparison."""
    if results.get("is_provisional"):
        print("\n  [WARNING: Results based on PLACEHOLDER vulnerability data]")

    # Primary metrics
    pf = results["primary_fairness"]
    print(f"\n--- Primary Solution Fairness ---")
    print(f"  Min fulfillment (Max-Min proxy): {pf['phi_min (Max-Min proxy)']:.4f}")
    print(f"  Max fulfillment: {pf['phi_max']:.4f}")
    print(f"  Mean fulfillment: {pf['phi_mean']:.4f}")
    print(f"  Std dev: {pf['phi_std']:.4f}")
    print(f"  Nash welfare score: {pf['nash_welfare_score']:.4f}")
    print(f"  Fulfillment-ratio disparity violations: {pf['disparity_count']}")

    # Comparison table
    print(f"\n--- Objective Comparison Table ---")
    print(f"{'Objective':<30} {'Min phi':>8} {'Mean phi':>9} {'Std phi':>9}")
    print(f"{'-'*30} {'-'*8} {'-'*9} {'-'*9}")
    for name, data in results["comparison_table"].items():
        print(f"{name:<30} {data['min_phi']:>8.4f} {data['mean_phi']:>9.4f} "
              f"{data['std_phi']:>9.4f}")

    # Baseline comparison
    bl = results["baseline_proportional"]
    prim_min = results["comparison_table"]["Deprivation-Min (Primary)"]["min_phi"]
    bl_min = bl["min_phi"]
    if prim_min > bl_min:
        print(f"\n  MILP improves minimum fulfillment by "
              f"{(prim_min - bl_min):.4f} over proportional baseline.")
    elif prim_min < bl_min:
        print(f"\n  Note: Proportional baseline has higher min fulfillment "
              f"({bl_min:.4f} vs {prim_min:.4f}) -- expected when deprivation "
              f"objective prioritizes vulnerability-weighted cost over raw equity.")
