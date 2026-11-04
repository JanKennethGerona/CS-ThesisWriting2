"""
Multi-Commodity Two-Stage MILP for Disaster Relief Resource Allocation (Davao City)

Extends the original single-commodity formulation (Section 3.4 of Misa, Labor
& Gerona, 2026) to a multi-commodity (water + food) Resource Allocation Problem:

  - Stage 1: hub activation (y_i)
  - Stage 2: flow allocation (x_{ijr}), per-resource fulfillment (phi_{jr}),
    per-resource deprivation penalty (z_{jr})
  - Objective: minimize vulnerability-weighted multi-commodity deprivation cost
    via piecewise-linear tangent approximation of e^{lambda_r*(1-phi_{jr})}
  - Post-hoc fairness metrics: Max-Min, Leximin, Nash welfare evaluation,
    fulfillment-ratio disparity, Pareto check, plus alternate-objective runs
    for Max Utilitarian and Pure Egalitarian.

CRITICAL DESIGN DECISION (v6):
  phi_j (overall fulfillment) is NOT a decision variable in the model.
  It is computed post-solve as min_r(phi_{jr}) in Python. This avoids the
  underdetermination bug: phi_j would only appear in phi_j <= phi_{jr} and
  phi_j >= eps_min, but never in the objective — so the solver has no
  incentive to push it above eps_min. Anti-starvation is enforced directly
  per resource: phi_{jr} >= eps_min for all j, r.

Requires: pyomo, and a MILP solver (appsi_highs / highs / cbc / glpk /
          gurobi / cplex).
"""

import math
import pyomo.environ as pyo


# ======================================================================
# STEP 1 — SAMPLE DATA (for regression testing)
# ======================================================================

def build_sample_data():
    """
    Single-commodity toy instance — preserved for backward compatibility
    and regression testing. Matches the original Section 3.4 formulation.
    """
    hubs = ["Hub_Talomo", "Hub_Buhangin", "Hub_Agdao", "Hub_Toril"]
    barangays = ["Brgy_A", "Brgy_B", "Brgy_C", "Brgy_D", "Brgy_E", "Brgy_F"]
    resources = ["water"]

    D = {("Brgy_A", "water"): 500, ("Brgy_B", "water"): 800,
         ("Brgy_C", "water"): 300, ("Brgy_D", "water"): 650,
         ("Brgy_E", "water"): 420, ("Brgy_F", "water"): 900}

    S = {("Hub_Talomo", "water"): 900, ("Hub_Buhangin", "water"): 700,
         ("Hub_Agdao", "water"): 600, ("Hub_Toril", "water"): 800}

    P = {"Brgy_A": 1.2, "Brgy_B": 1.0, "Brgy_C": 1.8,
         "Brgy_D": 1.0, "Brgy_E": 1.5, "Brgy_F": 1.0}

    params = {
        "Hmax": 3,
        "eps_min": 0.10,
        "eps_env": 0.15,
        "lambda_": {"water": 2.0},
    }
    return hubs, barangays, resources, D, S, P, params


def build_sample_data_multi():
    """
    Multi-commodity toy instance (water + food) for testing the
    extended formulation.
    """
    hubs = ["Hub_A", "Hub_B", "Hub_C", "Hub_D"]
    barangays = ["Brgy_1", "Brgy_2", "Brgy_3", "Brgy_4", "Brgy_5"]
    resources = ["water", "food"]

    # Water: L/day; Food: kcal/day (different scales by design)
    D = {
        ("Brgy_1", "water"): 600,  ("Brgy_1", "food"): 84000,
        ("Brgy_2", "water"): 900,  ("Brgy_2", "food"): 126000,
        ("Brgy_3", "water"): 400,  ("Brgy_3", "food"): 56000,
        ("Brgy_4", "water"): 750,  ("Brgy_4", "food"): 105000,
        ("Brgy_5", "water"): 500,  ("Brgy_5", "food"): 70000,
    }

    # Hub capacities per resource (total supply ≈ 80% of total demand per resource)
    S = {
        ("Hub_A", "water"): 700,  ("Hub_A", "food"): 98000,
        ("Hub_B", "water"): 600,  ("Hub_B", "food"): 84000,
        ("Hub_C", "water"): 500,  ("Hub_C", "food"): 70000,
        ("Hub_D", "water"): 700,  ("Hub_D", "food"): 98000,
    }

    P = {"Brgy_1": 1.2, "Brgy_2": 1.0, "Brgy_3": 1.8,
         "Brgy_4": 1.0, "Brgy_5": 1.5}

    params = {
        "Hmax": 3,
        "eps_min": 0.10,
        "eps_env": 0.15,
        "lambda_": {"water": 2.0, "food": 0.5},
    }
    return hubs, barangays, resources, D, S, P, params


# ======================================================================
# STEP 2 — DEPRIVATION COST CALIBRATION
# ======================================================================

def build_tangent_segments(lambda_dict,
                           breakpoints=None,
                           num_breakpoints=21):
    """
    Build piecewise-linear tangent segments for each resource's deprivation
    curve f(phi) = e^{lambda_r*(1-phi)}.

    Args:
        lambda_dict: {"water": lambda_w, "food": lambda_f, ...}
        breakpoints: explicit breakpoint tuple, or None to auto-generate
        num_breakpoints: number of evenly-spaced breakpoints if breakpoints=None

    Returns:
        dict: {resource: [(m_k, b_k), ...]}
    """
    if breakpoints is None:
        breakpoints = tuple(
            i / (num_breakpoints - 1) for i in range(num_breakpoints)
        )

    segments = {}
    for resource, lam in lambda_dict.items():
        segs = []
        for phi0 in breakpoints:
            u0 = 1 - phi0
            f_u0 = math.exp(lam * u0) - 1.0  # ZERO-BASELINE SHIFT (Chapter 3, Eq 3.3)
            slope_u = lam * math.exp(lam * u0)  # d f / d u
            # z >= f(u0) + slope_u*(u - u0)  where u = 1 - phi_jr
            m_k = slope_u
            b_k = f_u0 - slope_u * u0
            segs.append((m_k, b_k))
        segments[resource] = segs
    return segments


# ======================================================================
# STEP 3 — BUILD THE MULTI-COMMODITY TWO-STAGE MILP
# ======================================================================

def build_model(hubs, barangays, resources, D, S, P, params, segments):
    """
    Build the multi-commodity two-stage MILP.

    φ_j (overall fulfillment) is NOT included as a decision variable.
    It is computed post-solve via compute_overall_fulfillment().
    Anti-starvation is enforced per-resource: φ_{jr} >= ε_min.

    Args:
        hubs: list of hub names
        barangays: list of barangay names
        resources: list of resource names (e.g. ["water", "food"])
        D: dict (j, r) -> demand
        S: dict (i, r) -> capacity
        P: dict j -> vulnerability weight
        params: dict with Hmax, eps_min, lambda_ (dict), eps_env
        segments: dict r -> [(m_k, b_k), ...] from build_tangent_segments

    Returns:
        Pyomo ConcreteModel
    """
    m = pyo.ConcreteModel(name="Davao_MultiCommodity_RAP")

    # --- Sets ---
    m.I = pyo.Set(initialize=hubs)
    m.J = pyo.Set(initialize=barangays)
    m.R = pyo.Set(initialize=resources)

    # Segment count may differ per resource; use max for indexing
    max_segs = max(len(segments[r]) for r in resources)
    m.K = pyo.Set(initialize=range(max_segs))

    # --- Parameters ---
    m.D = pyo.Param(m.J, m.R, initialize=D)
    m.S = pyo.Param(m.I, m.R, initialize=S)
    m.P = pyo.Param(m.J, initialize=P)
    m.Hmax = pyo.Param(initialize=params["Hmax"])
    m.eps_min = pyo.Param(initialize=params["eps_min"])

    # Tangent coefficients per resource per segment
    mk_init = {}
    bk_init = {}
    for r in resources:
        for k, (mk, bk) in enumerate(segments[r]):
            mk_init[(k, r)] = mk
            bk_init[(k, r)] = bk
    m.mk = pyo.Param(m.K, m.R, initialize=mk_init, default=0)
    m.bk = pyo.Param(m.K, m.R, initialize=bk_init, default=0)

    # --- Decision Variables ---
    m.y = pyo.Var(m.I, domain=pyo.Binary)                          # Stage 1
    m.x = pyo.Var(m.I, m.J, m.R, domain=pyo.NonNegativeReals)      # Stage 2
    m.w = pyo.Var(m.J, m.R, domain=pyo.NonNegativeReals)
    m.phi_r = pyo.Var(m.J, m.R, bounds=(0, 1))
    m.z = pyo.Var(m.J, m.R, domain=pyo.NonNegativeReals)

    # NOTE: NO m.phi (overall) variable — computed post-solve

    # --- Objective: minimize vulnerability-weighted deprivation ---
    def obj_rule(m):
        return sum(m.P[j] * m.z[j, r] for j in m.J for r in m.R)
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- Tangent cuts per resource: z_{jr} >= m_k*(1-phi_{jr}) + b_k ---
    def tangent_rule(m, j, r, k):
        if k >= len(segments[r]):
            return pyo.Constraint.Skip
        return m.z[j, r] >= m.mk[k, r] * (1 - m.phi_r[j, r]) + m.bk[k, r]
    m.TangentCuts = pyo.Constraint(m.J, m.R, m.K, rule=tangent_rule)

    # --- Hub activation cap ---
    def hub_count_rule(m):
        return sum(m.y[i] for i in m.I) <= m.Hmax
    m.HubCount = pyo.Constraint(rule=hub_count_rule)

    # --- Capacity per hub per resource ---
    def capacity_rule(m, i, r):
        return sum(m.x[i, j, r] for j in m.J) <= m.S[i, r] * m.y[i]
    m.Capacity = pyo.Constraint(m.I, m.R, rule=capacity_rule)

    # --- Total allocation per resource ---
    def total_alloc_rule(m, j, r):
        return m.w[j, r] == sum(m.x[i, j, r] for i in m.I)
    m.TotalAlloc = pyo.Constraint(m.J, m.R, rule=total_alloc_rule)

    # --- Fulfillment ratio per resource ---
    def fulfillment_rule(m, j, r):
        return m.w[j, r] == m.phi_r[j, r] * m.D[j, r]
    m.Fulfillment = pyo.Constraint(m.J, m.R, rule=fulfillment_rule)

    # --- Anti-starvation per resource (direct, not transitive) ---
    def min_fulfillment_rule(m, j, r):
        return m.phi_r[j, r] >= m.eps_min
    m.MinFulfillment = pyo.Constraint(m.J, m.R, rule=min_fulfillment_rule)

    return m


# ======================================================================
# STEP 4 — SOLVE (with auto-detection)
# ======================================================================

# Solver preference order
_SOLVER_CHAIN = [
    "appsi_highs", "highs", "cbc", "glpk", "gurobi", "cplex",
]


def solve_model(m, solver_name=None, tee=False):
    """
    Solve the model, with automatic solver detection if no solver specified.

    Args:
        m: Pyomo model
        solver_name: explicit solver name, or None for auto-detection
        tee: whether to print solver output

    Returns:
        (result, solver_name_used)
    """
    if solver_name is not None:
        solver = pyo.SolverFactory(solver_name)
        result = solver.solve(m, tee=tee)
        return result, solver_name

    # Auto-detect
    for name in _SOLVER_CHAIN:
        try:
            solver = pyo.SolverFactory(name)
            if solver.available():
                result = solver.solve(m, tee=tee)
                return result, name
        except Exception:
            continue

    raise RuntimeError(
        f"No solver found. Tried: {_SOLVER_CHAIN}. "
        "Install at least one (e.g., 'pip install highspy')."
    )


# ======================================================================
# STEP 5 — POST-SOLVE COMPUTATION
# ======================================================================

def compute_overall_fulfillment(m, resources=None):
    """
    Compute overall fulfillment φ_j = min_r(φ_{jr}) from the solved model.
    This is NOT an in-model variable — it is computed here in Python.

    Returns:
        phi_j: {j: overall_fulfillment}
        limiting_resource: {j: resource_name}
    """
    if resources is None:
        resources = list(m.R)

    phi_j = {}
    limiting_resource = {}
    for j in m.J:
        phi_values = {r: pyo.value(m.phi_r[j, r]) for r in resources}
        min_r = min(phi_values, key=phi_values.get)
        phi_j[j] = phi_values[min_r]
        limiting_resource[j] = min_r

    return phi_j, limiting_resource


# ======================================================================
# STEP 6 — POST-HOC FAIRNESS EVALUATION
# ======================================================================

def evaluate_fairness(phi_j, params):
    """
    Compute fairness metrics from the post-processed φ_j vector.

    NOTE: φ_j here is the Python-computed min_r(φ_{jr}), NOT an in-model
    variable. This ensures all fairness metrics reflect the actual
    limiting-resource fulfillment.

    Args:
        phi_j: dict {j: overall_fulfillment} (from compute_overall_fulfillment)
        params: dict with eps_env

    Returns:
        dict with fairness metrics
    """
    sorted_phi = sorted(phi_j.values())

    report = {}
    report["fulfillment_by_node"] = {j: round(v, 6) for j, v in phi_j.items()}
    report["phi_min (Max-Min proxy)"] = min(phi_j.values())
    report["phi_max"] = max(phi_j.values())
    report["phi_mean"] = sum(phi_j.values()) / len(phi_j)
    report["phi_std"] = (
        sum((v - report["phi_mean"]) ** 2 for v in phi_j.values())
        / len(phi_j)
    ) ** 0.5
    report["leximin_sorted_vector"] = sorted_phi

    # Nash welfare evaluation (NOT "Nash-optimal") — sum of ln(phi_j)
    report["nash_welfare_score"] = sum(
        math.log(max(v, 1e-9)) for v in phi_j.values()
    )

    # Fulfillment-ratio disparity (NOT "envy-freeness")
    eps = params.get("eps_env", 0.15)
    disparity_violations = []
    js = list(phi_j.keys())
    for a in range(len(js)):
        for b in range(a + 1, len(js)):
            diff = abs(phi_j[js[a]] - phi_j[js[b]])
            if diff > eps:
                disparity_violations.append(
                    (js[a], js[b], round(diff, 4))
                )
    report["disparity_violations (> eps)"] = disparity_violations
    report["disparity_count"] = len(disparity_violations)

    return report


def solve_alternate_objective(hubs, barangays, resources, D, S, P, params,
                              segments, mode):
    """
    Re-solve with a different objective to compare against the primary
    deprivation-minimizing solution.

    mode: 'max_utilitarian' | 'pure_egalitarian' | 'max_min'

    For max_min, we DO use an in-model φ_j and Φ_min — but only for this
    specific re-solve, not the primary model.

    Returns:
        dict: {j: overall_phi_j} (post-processed)
    """
    m = pyo.ConcreteModel()

    m.I = pyo.Set(initialize=hubs)
    m.J = pyo.Set(initialize=barangays)
    m.R = pyo.Set(initialize=resources)
    m.D = pyo.Param(m.J, m.R, initialize=D)
    m.S = pyo.Param(m.I, m.R, initialize=S)
    m.Hmax = pyo.Param(initialize=params["Hmax"])
    m.eps_min = pyo.Param(initialize=params["eps_min"])

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, m.R, domain=pyo.NonNegativeReals)
    m.w = pyo.Var(m.J, m.R, domain=pyo.NonNegativeReals)
    m.phi_r = pyo.Var(m.J, m.R, bounds=(0, 1))

    # Shared constraints
    m.HubCount = pyo.Constraint(
        expr=sum(m.y[i] for i in m.I) <= m.Hmax
    )
    m.Capacity = pyo.Constraint(
        m.I, m.R,
        rule=lambda m, i, r: (
            sum(m.x[i, j, r] for j in m.J) <= m.S[i, r] * m.y[i]
        )
    )
    m.TotalAlloc = pyo.Constraint(
        m.J, m.R,
        rule=lambda m, j, r: m.w[j, r] == sum(m.x[i, j, r] for i in m.I)
    )
    m.Fulfillment = pyo.Constraint(
        m.J, m.R,
        rule=lambda m, j, r: m.w[j, r] == m.phi_r[j, r] * m.D[j, r]
    )
    m.MinFulfillment = pyo.Constraint(
        m.J, m.R,
        rule=lambda m, j, r: m.phi_r[j, r] >= m.eps_min
    )

    if mode == "max_utilitarian":
        # Maximize total per-resource fulfillment
        m.OBJ = pyo.Objective(
            expr=sum(m.phi_r[j, r] for j in m.J for r in m.R),
            sense=pyo.maximize
        )

    elif mode == "pure_egalitarian":
        # Force equal fulfillment across barangays per resource
        m.EqualPairs = pyo.ConstraintList()
        js = list(barangays)
        for r in resources:
            for a in range(len(js) - 1):
                m.EqualPairs.add(m.phi_r[js[a], r] == m.phi_r[js[a + 1], r])
        m.OBJ = pyo.Objective(
            expr=sum(m.phi_r[j, r] for j in m.J for r in m.R),
            sense=pyo.maximize
        )

    elif mode == "max_min":
        # Maximize the minimum overall fulfillment
        # Here we DO need in-model phi_j for the max-min formulation
        m.phi = pyo.Var(m.J, bounds=(0, 1))
        m.LimitingFulfillment = pyo.Constraint(
            m.J, m.R,
            rule=lambda m, j, r: m.phi[j] <= m.phi_r[j, r]
        )
        m.PhiMin = pyo.Var(bounds=(0, 1))
        m.MinLink = pyo.Constraint(
            m.J,
            rule=lambda m, j: m.phi[j] >= m.PhiMin
        )
        m.OBJ = pyo.Objective(expr=m.PhiMin, sense=pyo.maximize)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    result, solver_used = solve_model(m)

    # Post-process overall fulfillment
    phi_j, limiting = compute_overall_fulfillment(m, resources)
    return phi_j


def compute_baseline_proportional(barangays, resources, D, S, hubs, params):
    """
    Compute a proportional allocation baseline for comparison.
    Each barangay receives the same fulfillment ratio, determined by
    the resource with the tightest supply/demand ratio.

    Returns:
        dict: {j: phi_j}
    """
    # For each resource, compute achievable uniform fulfillment
    hmax = params["Hmax"]

    # Pick the Hmax hubs with the highest total capacity
    hub_total_cap = {}
    for i in hubs:
        hub_total_cap[i] = sum(S.get((i, r), 0) for r in resources)
    active_hubs = sorted(hub_total_cap, key=hub_total_cap.get, reverse=True)[:hmax]

    uniform_phi = {}
    for r in resources:
        total_supply = sum(S.get((i, r), 0) for i in active_hubs)
        total_demand = sum(D.get((j, r), 0) for j in barangays)
        if total_demand > 0:
            uniform_phi[r] = min(1.0, total_supply / total_demand)
        else:
            uniform_phi[r] = 1.0

    # Overall fulfillment is the limiting resource
    phi_overall = min(uniform_phi.values())

    return {j: phi_overall for j in barangays}


# ======================================================================
# STEP 7 — SENSITIVITY SWEEP
# ======================================================================

def sensitivity_sweep(hubs, barangays, resources, D, S, P, params,
                      segments,
                      shrink_factors=(1.0, 0.8, 0.6, 0.4, 0.2, 0.1)):
    """
    Progressively shrink supply to test anti-starvation binding and
    infeasibility detection.

    Returns:
        list of dicts with per-factor results
    """
    results = []
    for f in shrink_factors:
        S_scaled = {(i, r): v * f for (i, r), v in S.items()}
        segs = build_tangent_segments(
            params["lambda_"],
            num_breakpoints=21
        )
        m = build_model(
            hubs, barangays, resources, D, S_scaled, P, params, segs
        )
        try:
            res, solver_used = solve_model(m)
            feasible = str(res.solver.termination_condition) == "optimal"
        except Exception:
            feasible = False
            res = None

        if feasible:
            phi_j, limiting = compute_overall_fulfillment(m, resources)
            row = {
                "supply_factor": f,
                "feasible": True,
                "objective": pyo.value(m.OBJ),
                "min_phi_overall": min(phi_j.values()),
                "mean_phi_overall": sum(phi_j.values()) / len(phi_j),
            }
            # Per-resource min fulfillment
            for r in resources:
                vals = [pyo.value(m.phi_r[j, r]) for j in m.J]
                row[f"min_phi_{r}"] = min(vals)
                row[f"mean_phi_{r}"] = sum(vals) / len(vals)
        else:
            row = {
                "supply_factor": f,
                "feasible": False,
                "objective": None,
                "min_phi_overall": None,
                "mean_phi_overall": None,
            }
            for r in resources:
                row[f"min_phi_{r}"] = None
                row[f"mean_phi_{r}"] = None

        results.append(row)
    return results


# ======================================================================
# MAIN — end-to-end smoke test
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SMOKE TEST: Single-commodity (backward compatibility)")
    print("=" * 60)
    hubs, barangays, resources, D, S, P, params = build_sample_data()
    segments = build_tangent_segments(params["lambda_"])
    model = build_model(hubs, barangays, resources, D, S, P, params, segments)
    result, solver_used = solve_model(model)

    print(f"Solver: {solver_used}")
    print(f"Status: {result.solver.termination_condition}")
    print(f"Objective: {pyo.value(model.OBJ):.4f}")

    phi_j, limiting = compute_overall_fulfillment(model, resources)
    for j in model.J:
        phi_w = pyo.value(model.phi_r[j, "water"])
        print(f"  {j}: phi_water={phi_w:.3f}  phi_overall={phi_j[j]:.3f}  "
              f"w_water={pyo.value(model.w[j, 'water']):.1f}")

    print(f"\n{'=' * 60}")
    print("  SMOKE TEST: Multi-commodity (water + food)")
    print(f"{'=' * 60}")
    hubs2, brgys2, res2, D2, S2, P2, params2 = build_sample_data_multi()
    segments2 = build_tangent_segments(params2["lambda_"])
    model2 = build_model(hubs2, brgys2, res2, D2, S2, P2, params2, segments2)
    result2, solver2 = solve_model(model2)

    print(f"Solver: {solver2}")
    print(f"Status: {result2.solver.termination_condition}")
    print(f"Objective: {pyo.value(model2.OBJ):.4f}")

    phi_j2, limiting2 = compute_overall_fulfillment(model2, res2)
    for j in model2.J:
        phi_w = pyo.value(model2.phi_r[j, "water"])
        phi_f = pyo.value(model2.phi_r[j, "food"])
        print(f"  {j}: phi_water={phi_w:.3f}  phi_food={phi_f:.3f}  "
              f"phi_overall={phi_j2[j]:.3f}  limiting={limiting2[j]}")

    print("\nPost-hoc fairness (multi-commodity):")
    fairness = evaluate_fairness(phi_j2, params2)
    for k, v in fairness.items():
        if k != "fulfillment_by_node":
            print(f"  {k}: {v}")

    print("\nAlternate-objective comparisons (multi-commodity):")
    for mode in ("max_utilitarian", "pure_egalitarian", "max_min"):
        alt_phi = solve_alternate_objective(
            hubs2, brgys2, res2, D2, S2, P2, params2, segments2, mode
        )
        print(f"  {mode}: min_phi={min(alt_phi.values()):.3f}  "
              f"mean_phi={sum(alt_phi.values())/len(alt_phi):.3f}")

    print("\nSensitivity sweep (multi-commodity):")
    sweep = sensitivity_sweep(
        hubs2, brgys2, res2, D2, S2, P2, params2, segments2
    )
    for row in sweep:
        if row["feasible"]:
            print(f"  factor={row['supply_factor']:.1f}  "
                  f"obj={row['objective']:.2f}  "
                  f"min_phi={row['min_phi_overall']:.3f}")
        else:
            print(f"  factor={row['supply_factor']:.1f}  INFEASIBLE")
