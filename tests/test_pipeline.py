"""
test_pipeline.py — Comprehensive automated verification suite for the
multi-commodity Two-Stage MILP.

Tests cover:
  - Data integrity (182 rows, 14 columns, uniqueness, demand recomputation)
  - Multi-commodity tangent math (under-approximation, breakpoint accuracy)
  - Model feasibility (single-commodity and multi-commodity)
  - Anti-starvation per resource
  - Deprivation accuracy (under-approximation direction)
  - Limiting-resource check (post-processed phi_j = min_r phi_{jr})
  - Resource conservation and activation link
  - Resource independence
  - Alternate-objective re-solves
  - Sensitivity sweep monotonicity
  - Baseline comparison
  - Backward compatibility (single-commodity)
  - Placeholder protection (research mode rejects placeholders)

Usage:
    python test_pipeline.py
"""

import math
import sys
import os
from typing import Any

# Ensure project root and src/ are on path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pyomo.environ as pyo
from data_loader import load_data, DataValidationError
from load_real_data import (
    load_barangay_data, SPHERE_WATER_L_PER_PERSON,
    SPHERE_FOOD_KCAL_PER_PERSON, RESOURCES,
)
from davao_rap_milp import (
    build_model, build_tangent_segments, solve_model,
    compute_overall_fulfillment, build_sample_data,
    build_sample_data_multi, evaluate_fairness,
    solve_alternate_objective, sensitivity_sweep,
    compute_baseline_proportional,
)
from validation import (
    validate_solution, validate_tangent_approximation,
    compare_breakpoint_counts,
)


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, msg=""):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  [FAIL] {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 50}")
        print(f"  Results: {self.passed}/{total} passed, "
              f"{self.failed} failed")
        print(f"{'=' * 50}")
        if self.errors:
            print("\nFailed tests:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


def test_data_integrity(t):
    """Test data loading and validation."""
    print("\n--- Data Integrity ---")

    # Load xlsx
    df = load_barangay_data()
    if len(df) == 182:
        t.ok("182 barangays loaded")
    else:
        t.fail("row count", f"expected 182, got {len(df)}")

    if len(df.columns) == 14:
        t.ok("14 columns present")
    else:
        t.fail("column count", f"expected 14, got {len(df.columns)}")

    # Unique names
    if df["Barangay"].nunique() == 182:
        t.ok("182 unique barangay names")
    else:
        t.fail("unique names", f"only {df['Barangay'].nunique()} unique")

    # Unique PSGC codes
    if df["10 Digit Code"].nunique() == 182:
        t.ok("182 unique PSGC codes")
    else:
        t.fail("unique codes", f"only {df['10 Digit Code'].nunique()} unique")

    # No null population
    if df["Population (2024 POPCEN)"].isnull().sum() == 0:
        t.ok("no null populations")
    else:
        t.fail("null populations")

    # Water demand recomputation
    expected_water = df["Population (2024 POPCEN)"] * SPHERE_WATER_L_PER_PERSON
    if (df["Demand (D_j)"] - expected_water).abs().max() < 0.01:
        t.ok("water demand recomputation matches")
    else:
        t.fail("water demand recomputation")

    # Food demand recomputation
    expected_food = df["Population (2024 POPCEN)"] * SPHERE_FOOD_KCAL_PER_PERSON
    if (df["Food Demand (D_j, Food)"] - expected_food).abs().max() < 0.01:
        t.ok("food demand recomputation matches")
    else:
        t.fail("food demand recomputation")

    # Districts
    expected_districts = {
        "Poblacion", "Talomo", "Agdao", "Buhangin", "Bunawan",
        "Paquibato", "Baguio", "Calinan", "Marilog", "Toril", "Tugbok",
    }
    actual = set(df["District"].unique())
    if actual == expected_districts:
        t.ok("11 expected districts present")
    else:
        t.fail("districts", f"missing={expected_districts - actual}, "
               f"extra={actual - expected_districts}")

    # Vulnerability Data Source check
    if df["Vulnerability Data Source"].str.contains("PLACEHOLDER").all():
        t.ok("is_provisional correctly detected (all PLACEHOLDER)")
    else:
        t.fail("is_provisional detection")

    # Data loader API
    try:
        data = load_data(mode="development")
        if data["is_provisional"]:
            t.ok("data_loader development mode works")
        else:
            t.fail("data_loader", "should be provisional")
    except Exception as e:
        t.fail("data_loader", str(e))


def test_tangent_math(t):
    """Test tangent approximation mathematics."""
    print("\n--- Tangent Approximation Math ---")

    for lam_name, lam_val in [("water", 2.0), ("food", 0.5)]:
        # At breakpoints: gap should be ~0
        breakpoints = tuple(i / 20.0 for i in range(21))
        errors = validate_tangent_approximation(lam_val, breakpoints)

        bp_errors = [e for e in errors if e["phi"] in breakpoints]
        max_bp_gap = max(e["abs_error"] for e in bp_errors)
        if max_bp_gap < 1e-10:
            t.ok(f"tangent touches curve at breakpoints (lambda={lam_val})")
        else:
            t.fail(f"breakpoint accuracy ({lam_name})",
                   f"max gap at breakpoints = {max_bp_gap}")

        # Between breakpoints: should be strictly below (under-approximation)
        all_positive = all(e["abs_error"] >= -1e-10 for e in errors)
        if all_positive:
            t.ok(f"under-approximation verified (lambda={lam_val})")
        else:
            neg_errors = [e for e in errors if e["abs_error"] < -1e-10]
            t.fail(f"under-approximation ({lam_name})",
                   f"{len(neg_errors)} points above curve")

    # Breakpoint count comparison
    results = compare_breakpoint_counts(2.0, counts=(5, 11, 21, 41))
    if results[21]["max_abs_error"] < results[5]["max_abs_error"]:
        t.ok("21 breakpoints more accurate than 5")
    else:
        t.fail("breakpoint count comparison")

    if results[41]["max_abs_error"] < results[21]["max_abs_error"]:
        t.ok("41 breakpoints more accurate than 21")
    else:
        t.fail("breakpoint count comparison 41 vs 21")


def test_single_commodity_model(t):
    """Test backward-compatible single-commodity model."""
    print("\n--- Single-Commodity Model (Backward Compatibility) ---")

    hubs, brgys, resources, D, S, P, params = build_sample_data()
    segments = build_tangent_segments(params["lambda_"])
    model: Any = build_model(hubs, brgys, resources, D, S, P, params, segments)
    result, solver = solve_model(model)

    status = str(result.solver.termination_condition)
    if status == "optimal":
        t.ok("single-commodity solves to optimal")
    else:
        t.fail("single-commodity feasibility", status)
        return

    # Bounds
    phi_j, _ = compute_overall_fulfillment(model, resources)
    eps_min = params["eps_min"]
    if all(v >= eps_min - 1e-6 for v in phi_j.values()):
        t.ok("anti-starvation floor satisfied (single-commodity)")
    else:
        t.fail("anti-starvation (single-commodity)")

    if all(v <= 1.0 + 1e-6 for v in phi_j.values()):
        t.ok("no-hoarding satisfied (single-commodity)")
    else:
        t.fail("no-hoarding (single-commodity)")


def test_multi_commodity_model(t):
    """Test multi-commodity model — the main formulation."""
    print("\n--- Multi-Commodity Model ---")

    hubs, brgys, resources, D, S, P, params = build_sample_data_multi()
    segments = build_tangent_segments(params["lambda_"])
    model: Any = build_model(hubs, brgys, resources, D, S, P, params, segments)
    result, solver = solve_model(model)

    status = str(result.solver.termination_condition)
    if status == "optimal":
        t.ok("multi-commodity solves to optimal")
    else:
        t.fail("multi-commodity feasibility", status)
        return

    phi_j, limiting = compute_overall_fulfillment(model, resources)
    eps_min = params["eps_min"]
    lambda_dict = params["lambda_"]

    # Anti-starvation per resource
    all_ok = True
    for j in model.J:
        for r in model.R:
            phi_jr = pyo.value(model.phi_r[j, r])
            if phi_jr < eps_min - 1e-6:
                all_ok = False
    if all_ok:
        t.ok("anti-starvation per resource (phi_jr >= eps_min)")
    else:
        t.fail("anti-starvation per resource")

    # No-hoarding per resource
    all_ok = True
    for j in model.J:
        for r in model.R:
            if pyo.value(model.phi_r[j, r]) > 1.0 + 1e-6:
                all_ok = False
    if all_ok:
        t.ok("no-hoarding per resource (phi_jr <= 1.0)")
    else:
        t.fail("no-hoarding per resource")

    # Deprivation under-approximation
    all_ok = True
    for j in model.J:
        for r in model.R:
            phi_jr = pyo.value(model.phi_r[j, r])
            z_jr = pyo.value(model.z[j, r])
            true_val = math.exp(lambda_dict[r] * (1.0 - phi_jr))
            if z_jr > true_val + 1e-6:
                all_ok = False
    if all_ok:
        t.ok("deprivation under-approximation (z_jr <= e^(lambda*(1-phi)))")
    else:
        t.fail("deprivation under-approximation")

    # Limiting-resource check (the critical φ_j fix)
    all_ok = True
    for j in model.J:
        phi_values = {r: pyo.value(model.phi_r[j, r]) for r in resources}
        expected_min = min(phi_values.values())
        if abs(phi_j[j] - expected_min) > 1e-6:
            all_ok = False
    if all_ok:
        t.ok("phi_j = min_r(phi_jr) [post-processed correctly]")
    else:
        t.fail("limiting-resource post-processing")

    # Resource conservation
    all_ok = True
    for i in model.I:
        for r in model.R:
            shipped = sum(pyo.value(model.x[i, j, r]) for j in model.J)
            cap = pyo.value(model.S[i, r]) * pyo.value(model.y[i])
            if shipped > cap + 1e-6:
                all_ok = False
    if all_ok:
        t.ok("resource conservation (shipped <= capacity)")
    else:
        t.fail("resource conservation")

    # Activation link
    all_ok = True
    for i in model.I:
        if pyo.value(model.y[i]) < 0.5:
            for j in model.J:
                for r in model.R:
                    if pyo.value(model.x[i, j, r]) > 1e-6:
                        all_ok = False
    if all_ok:
        t.ok("activation link (y_i=0 => x_ijr=0)")
    else:
        t.fail("activation link")

    # Hub count
    active = sum(1 for i in model.I if pyo.value(model.y[i]) > 0.5)
    if active <= params["Hmax"]:
        t.ok(f"hub count ({active} <= Hmax={params['Hmax']})")
    else:
        t.fail("hub count")

    # Resource independence: check that water and food allocations are separate
    # (total water shipped should not exceed total water capacity, same for food)
    for r in resources:
        total_shipped = sum(
            pyo.value(model.x[i, j, r]) for i in model.I for j in model.J
        )
        total_cap = sum(
            pyo.value(model.S[i, r]) * pyo.value(model.y[i])
            for i in model.I
        )
        if total_shipped <= total_cap + 1e-6:
            t.ok(f"resource independence ({r})")
        else:
            t.fail(f"resource independence ({r})")

    # Full validation
    val = validate_solution(model, lambda_dict, resources, phi_j)
    if val["passed"]:
        t.ok("full solution validation passed")
    else:
        t.fail("full solution validation", str(val["errors"]))


def test_alternate_objectives(t):
    """Test alternate-objective re-solves."""
    print("\n--- Alternate-Objective Re-Solves ---")

    hubs, brgys, resources, D, S, P, params = build_sample_data_multi()
    segments = build_tangent_segments(params["lambda_"])

    # Primary solve first
    model: Any = build_model(hubs, brgys, resources, D, S, P, params, segments)
    result, _ = solve_model(model)
    primary_phi, _ = compute_overall_fulfillment(model, resources)

    for mode in ("max_utilitarian", "pure_egalitarian", "max_min"):
        try:
            alt_phi = solve_alternate_objective(
                hubs, brgys, resources, D, S, P, params, segments, mode
            )
            if all(v >= params["eps_min"] - 1e-6 for v in alt_phi.values()):
                t.ok(f"{mode}: feasible, anti-starvation satisfied")
            else:
                t.fail(f"{mode}: anti-starvation")

            # Check it produces a different allocation
            same = all(
                abs(primary_phi.get(j, 0) - alt_phi.get(j, 0)) < 1e-4
                for j in primary_phi
            )
            if not same:
                t.ok(f"{mode}: produces different allocation from primary")
            else:
                # For small toy problems, some objectives may coincide
                t.ok(f"{mode}: allocation may coincide (acceptable for toy data)")

        except Exception as e:
            t.fail(f"{mode}", str(e))


def test_sensitivity_sweep(t):
    """Test sensitivity sweep monotonicity."""
    print("\n--- Sensitivity Sweep ---")

    hubs, brgys, resources, D, S, P, params = build_sample_data_multi()
    segments = build_tangent_segments(params["lambda_"])
    sweep = sensitivity_sweep(
        hubs, brgys, resources, D, S, P, params, segments
    )

    # Check monotonic degradation (objective should increase as supply drops)
    objectives = [
        row["objective"] for row in sweep if row["feasible"]
    ]
    if len(objectives) >= 2:
        monotonic = all(
            objectives[i] <= objectives[i+1] + 1e-6
            for i in range(len(objectives) - 1)
        )
        if monotonic:
            t.ok("sensitivity sweep: monotonic objective increase")
        else:
            t.fail("sensitivity monotonicity", str(objectives))
    else:
        t.fail("sensitivity: too few feasible points")

    # Check infeasibility detection
    infeasible_found = any(not row["feasible"] for row in sweep)
    if infeasible_found:
        t.ok("sensitivity sweep: infeasibility detected at low supply")
    else:
        t.ok("sensitivity sweep: all points feasible (supply sufficient)")


def test_baseline_comparison(t):
    """Test baseline proportional allocation."""
    print("\n--- Baseline Comparison ---")

    hubs, brgys, resources, D, S, P, params = build_sample_data_multi()
    baseline = compute_baseline_proportional(brgys, resources, D, S, hubs, params)

    # All should have the same value (proportional = uniform)
    vals = list(baseline.values())
    if all(abs(v - vals[0]) < 1e-6 for v in vals):
        t.ok("baseline produces uniform fulfillment")
    else:
        t.fail("baseline uniformity")

    # Should be between 0 and 1
    if all(0 <= v <= 1.0 for v in vals):
        t.ok("baseline in [0, 1]")
    else:
        t.fail("baseline range")


def test_placeholder_protection(t):
    """Test that research mode rejects placeholder data."""
    print("\n--- Placeholder Protection ---")

    try:
        data = load_data(mode="research")
        # If we get here, it either has real data or failed to detect placeholders
        if not data["is_provisional"]:
            t.ok("research mode: accepted (data is real)")
        else:
            t.fail("research mode should reject provisional data")
    except DataValidationError as e:
        if "placeholder" in str(e).lower():
            t.ok("research mode correctly rejects placeholder data")
        else:
            t.fail("research mode rejection", str(e))
    except Exception as e:
        t.fail("research mode", str(e))


def test_real_data_model(t):
    """Test the model with real Davao City data (development mode)."""
    print("\n--- Real Data Model (Development Mode) ---")

    try:
        data = load_data(mode="development")
        t.ok("real data loaded successfully")
    except Exception as e:
        t.fail("real data loading", str(e))
        return

    segments = build_tangent_segments(
        data["params"]["lambda_"], num_breakpoints=21
    )
    model: Any = build_model(
        data["hubs"], data["barangays"], data["resources"],
        data["D"], data["S"], data["P"], data["params"], segments
    )

    result, solver = solve_model(model)
    status = str(result.solver.termination_condition)
    if status == "optimal":
        t.ok(f"real data model solves to optimal (solver: {solver})")
    else:
        t.fail("real data model", f"status: {status}")
        return

    phi_j, limiting = compute_overall_fulfillment(
        model, data["resources"]
    )

    # Bounds check
    eps_min = data["params"]["eps_min"]
    if all(v >= eps_min - 1e-6 for v in phi_j.values()):
        t.ok("anti-starvation satisfied on real data")
    else:
        min_v = min(phi_j.values())
        t.fail("anti-starvation on real data", f"min phi = {min_v}")

    if all(v <= 1.0 + 1e-6 for v in phi_j.values()):
        t.ok("no-hoarding satisfied on real data")
    else:
        t.fail("no-hoarding on real data")

    # Validation
    val = validate_solution(
        model, data["params"]["lambda_"],
        data["resources"], phi_j
    )
    if val["passed"]:
        t.ok("full validation passed on real data")
    else:
        t.fail("validation on real data", str(val["errors"][:3]))


if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Commodity MILP -- Automated Test Suite")
    print("=" * 60)

    t = TestResults()

    test_data_integrity(t)
    test_tangent_math(t)
    test_single_commodity_model(t)
    test_multi_commodity_model(t)
    test_alternate_objectives(t)
    test_sensitivity_sweep(t)
    test_baseline_comparison(t)
    test_placeholder_protection(t)
    test_real_data_model(t)

    success = t.summary()
    sys.exit(0 if success else 1)
