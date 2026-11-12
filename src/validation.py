"""
validation.py — Strict solution validation layer for the multi-commodity MILP.

Validates both data integrity (pre-solve) and solution correctness (post-solve).
If validation fails, the solve should not proceed or results should not be trusted.

Usage:
    from validation import validate_solution, validate_tangent_approximation
"""

import math
from typing import Any
import pyomo.environ as pyo


class SolutionValidationError(Exception):
    """Raised when solution validation fails."""
    pass


# -- Function: Validate mathematical constraints and epigraph conditions of solved model --
def validate_solution(model: Any, lambda_dict, resources, post_processed_phi,
                      tol=1e-6):
    """
    Comprehensive post-solve validation.

    Args:
        model: solved Pyomo model
        lambda_dict: {"water": λ_w, "food": λ_f}
        resources: list of resource names ["water", "food"]
        post_processed_phi: {j: min_r(φ_{jr})} computed in Python
        tol: numerical tolerance

    Returns:
        dict with validation results and any warnings/errors
    """
    results = {
        "passed": True,
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    # --- Check 1: φ_{jr} bounds ---
    eps_min = pyo.value(model.eps_min)
    for j in model.J:
        for r in model.R:
            phi_jr = pyo.value(model.phi_r[j, r])
            if phi_jr < eps_min - tol:
                results["errors"].append(
                    f"Anti-starvation violated: phi[{j},{r}] = {phi_jr:.6f} "
                    f"< eps_min = {eps_min}"
                )
                results["passed"] = False
            if phi_jr > 1.0 + tol:
                results["errors"].append(
                    f"No-hoarding violated: phi[{j},{r}] = {phi_jr:.6f} > 1.0"
                )
                results["passed"] = False
    results["checks"].append("phi_jr bounds (eps_min <= phi_jr <= 1.0)")

    # --- Check 2: Deprivation under-approximation ---
    max_gap = 0.0
    for j in model.J:
        for r in model.R:
            phi_jr = pyo.value(model.phi_r[j, r])
            z_jr = pyo.value(model.z[j, r])
            lam = lambda_dict[r]
            true_val = math.exp(lam * (1.0 - phi_jr)) - 1.0
            if z_jr > true_val + tol:
                results["errors"].append(
                    f"Over-approximation: z[{j},{r}] = {z_jr:.6f} > "
                    f"e^(lambda*(1-phi))-1 = {true_val:.6f}"
                )
                results["passed"] = False
            gap = true_val - z_jr
            max_gap = max(max_gap, gap)
    results["checks"].append("z_jr <= e^(lambda_r*(1-phi_jr)) [under-approximation]")
    results["max_approximation_gap"] = max_gap

    # --- Check 3: Post-processed φ_j = min_r(φ_{jr}) ---
    for j in model.J:
        phi_values = {r: pyo.value(model.phi_r[j, r]) for r in model.R}
        expected_min = min(phi_values.values())
        reported = post_processed_phi[j]
        if abs(reported - expected_min) > tol:
            results["errors"].append(
                f"Post-processed phi[{j}] = {reported:.6f} != "
                f"min_r(phi_jr) = {expected_min:.6f}"
            )
            results["passed"] = False
    results["checks"].append("phi_j = min_r(phi_jr) [post-processed]")

    # --- Check 4: Resource conservation ---
    for i in model.I:
        for r in model.R:
            total_shipped = sum(
                pyo.value(model.x[i, j, r]) for j in model.J
            )
            capacity = pyo.value(model.S[i, r]) * pyo.value(model.y[i])
            if total_shipped > capacity + tol:
                results["errors"].append(
                    f"Resource conservation violated: hub {i}, resource {r}: "
                    f"shipped {total_shipped:.1f} > capacity {capacity:.1f}"
                )
                results["passed"] = False
    results["checks"].append("resource conservation (shipped <= capacity)")

    # --- Check 5: Demand conservation (w_jr = phi_jr * D_jr) ---
    for j in model.J:
        for r in model.R:
            w_jr = pyo.value(model.w[j, r])
            phi_jr = pyo.value(model.phi_r[j, r])
            D_jr = pyo.value(model.D[j, r])
            expected = phi_jr * D_jr
            if abs(w_jr - expected) > tol * max(1, abs(expected)):
                results["errors"].append(
                    f"Demand conservation: w[{j},{r}] = {w_jr:.2f} != "
                    f"phi * D = {expected:.2f}"
                )
                results["passed"] = False
    results["checks"].append("demand conservation (w_jr = phi_jr * D_jr)")

    # --- Check 6: Activation link ---
    for i in model.I:
        y_i = pyo.value(model.y[i])
        if y_i < 0.5:  # not activated
            for j in model.J:
                for r in model.R:
                    x_val = pyo.value(model.x[i, j, r])
                    if x_val > tol:
                        results["errors"].append(
                            f"Activation link violated: y[{i}]=0 but "
                            f"x[{i},{j},{r}]={x_val:.2f}"
                        )
                        results["passed"] = False
    results["checks"].append("activation link (y_i=0 => x_ijr=0)")

    # --- Check 7: Hub count ---
    active_hubs = sum(1 for i in model.I if pyo.value(model.y[i]) > 0.5)
    hmax = pyo.value(model.Hmax)
    if active_hubs > hmax:
        results["errors"].append(
            f"Hub count violated: {active_hubs} active > Hmax = {hmax}"
        )
        results["passed"] = False
    results["checks"].append(f"hub count ({active_hubs} <= Hmax={hmax})")

    # --- Check 8: Non-negativity ---
    for i in model.I:
        for j in model.J:
            for r in model.R:
                x_val = pyo.value(model.x[i, j, r])
                if x_val < -tol:
                    results["errors"].append(
                        f"Non-negativity: x[{i},{j},{r}] = {x_val:.6f} < 0"
                    )
                    results["passed"] = False
    results["checks"].append("non-negativity (x_ijr >= 0)")

    return results


# -- Function: Validate tangent approximation accuracy against exact exponential --
def validate_tangent_approximation(lambda_, breakpoints, test_points=None):
    """
    Compute tangent approximation error at test points.

    Args:
        lambda_: deprivation escalation rate
        breakpoints: tuple of φ values where tangent lines are constructed
        test_points: optional list of φ values to test (default: 0.00 to 1.00 by 0.01)

    Returns:
        list of dicts with phi, exact, approx, abs_error, rel_error
    """
    if test_points is None:
        test_points = [i / 100.0 for i in range(101)]

    # Build tangent segments
    segments = []
    for phi0 in breakpoints:
        u0 = 1 - phi0
        f_u0 = math.exp(lambda_ * u0) - 1.0
        slope_u = lambda_ * math.exp(lambda_ * u0)
        m_k = slope_u
        b_k = f_u0 - slope_u * u0
        segments.append((m_k, b_k))

    results = []
    for phi in test_points:
        u = 1 - phi
        exact = math.exp(lambda_ * u) - 1.0
        # Approximation = max over all tangent lines
        approx = max(m * u + b for m, b in segments)
        abs_err = exact - approx  # should be >= 0 (under-approximation)
        rel_err = abs_err / exact if exact > 1e-12 else 0.0
        results.append({
            "phi": phi,
            "exact": exact,
            "approx": approx,
            "abs_error": abs_err,
            "rel_error": rel_err,
        })

    return results


# -- Function: Compare tangent approximation errors across multiple breakpoint counts --
def compare_breakpoint_counts(lambda_, counts=(5, 11, 21, 41)):
    """
    Compare tangent approximation accuracy across different breakpoint counts.

    Returns dict mapping count -> {max_abs_error, max_rel_error, avg_abs_error}
    """
    results = {}
    for n in counts:
        breakpoints = tuple(i / (n - 1) for i in range(n))
        errors = validate_tangent_approximation(lambda_, breakpoints)
        abs_errors = [e["abs_error"] for e in errors]
        rel_errors = [e["rel_error"] for e in errors]
        results[n] = {
            "breakpoints": n,
            "max_abs_error": max(abs_errors),
            "max_rel_error": max(rel_errors),
            "avg_abs_error": sum(abs_errors) / len(abs_errors),
            "avg_rel_error": sum(rel_errors) / len(rel_errors),
        }
    return results


# -- Function: Pretty-print solution validation results --
def print_validation_results(results):
    """Pretty-print validation results."""
    status = "PASSED" if results["passed"] else "FAILED"
    print(f"\n{'=' * 50}")
    print(f"  Solution Validation: {status}")
    print(f"{'=' * 50}")

    print(f"\nChecks performed ({len(results['checks'])}):")
    for check in results["checks"]:
        print(f"  [OK] {check}")

    if results.get("max_approximation_gap") is not None:
        print(f"\n  Max tangent approximation gap: "
              f"{results['max_approximation_gap']:.6f}")

    if results["warnings"]:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for w in results["warnings"]:
            print(f"  [WARN] {w}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"  [FAIL] {e}")

    return results["passed"]
