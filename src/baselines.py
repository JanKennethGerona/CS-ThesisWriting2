"""
baselines.py — Accessibility-driven and decentralized allocation baselines.

Provides two counterfactual baselines for comparison with the primary
deprivation-minimizing MILP:

  1. Accessibility-driven baseline:
     Allocation share depends on geographic accessibility (urban/rural,
     district remoteness). Models the observed status-quo bias documented
     by Samson (2024) and Cayamanda et al. (2021).

  2. Decentralized request-based baseline:
     Each barangay inflates its request; the hub fills proportionally.
     Simulates bullwhip-type distortion over multiple Monte Carlo runs.
     Also serves as the accessibility-biased baseline (urban barangays
     inflate more due to better access / louder voice).

Both baselines are STYLIZED ASSUMPTIONS, not real policies. They exist
to demonstrate what the MILP avoids, not to model actual mechanisms.

Usage:
    from baselines import (
        compute_baseline_accessibility,
        compute_baseline_decentralized,
        compute_accessibility_scores,
    )
"""

from typing import Any
from collections import defaultdict
import numpy as np


# Districts classified as upland/remote (Davao City context)
UPLAND_DISTRICTS = {"Marilog", "Paquibato"}

# Districts classified as urban-core
URBAN_CORE_DISTRICTS = {"Poblacion", "Agdao"}


# -- Function: Compute stylized accessibility scores based on district and urbanization --
def compute_accessibility_scores(barangays, district_map, urban_rural_map):
    """
    Assign a stylized accessibility score to each barangay.

    Scoring logic (all values are STYLIZED ASSUMPTIONS):
      - Urban barangay in urban-core district: 1.0
      - Urban barangay in non-upland district:  0.8
      - Rural barangay in non-upland district:  0.4
      - Urban barangay in upland district:      0.3
      - Rural barangay in upland district:      0.2

    Args:
        barangays: list of barangay names
        district_map: {barangay_name: district_name}
        urban_rural_map: {barangay_name: "Urban" or "Rural"}

    Returns:
        dict: {barangay_name: accessibility_score}
    """
    scores = {}
    for j in barangays:
        district = district_map.get(j, "")
        is_urban = urban_rural_map.get(j, "Rural") == "Urban"
        is_upland = district in UPLAND_DISTRICTS
        is_core = district in URBAN_CORE_DISTRICTS

        if is_upland:
            scores[j] = 0.3 if is_urban else 0.2
        elif is_core:
            scores[j] = 1.0 if is_urban else 0.6
        else:
            scores[j] = 0.8 if is_urban else 0.4

    return scores


# -- Function: Compute accessibility-driven allocation baseline --
def compute_baseline_accessibility(barangays, resources, D, S, hubs, params,
                                   district_map, urban_rural_map):
    """
    Accessibility-driven allocation baseline.

    Supply is distributed proportional to (accessibility_score × D_jr),
    so accessible barangays receive a larger fraction of their demand.
    This models the observed bias where urban/accessible communities
    receive disproportionate relief.

    Labeled as a "stylized accessibility-driven allocation" — not a real
    policy, but a counterfactual illustrating the bias the MILP avoids.

    Args:
        barangays: list of barangay names
        resources: list of resource names
        D: demand dict {(j, r): value}
        S: supply dict {(i, r): value}
        hubs: list of hub names
        params: dict with Hmax
        district_map: {barangay_name: district_name}
        urban_rural_map: {barangay_name: "Urban" or "Rural"}

    Returns:
        dict: {
            "phi_jr": {(j, r): fulfillment_ratio},
            "phi_j": {j: overall_fulfillment},
            "accessibility_scores": {j: score},
            "metrics": dict with summary statistics,
        }
    """
    hmax = params["Hmax"]

    # Pick the Hmax hubs with highest total capacity (same logic as
    # compute_baseline_proportional in davao_rap_milp.py)
    hub_total_cap = {}
    for i in hubs:
        hub_total_cap[i] = sum(S.get((i, r), 0) for r in resources)
    active_hubs = sorted(hub_total_cap, key=hub_total_cap.get,
                         reverse=True)[:hmax]

    # Accessibility scores
    acc_scores = compute_accessibility_scores(
        barangays, district_map, urban_rural_map
    )

    # Allocate per resource: each barangay's share is proportional to
    # accessibility_score × D_jr
    phi_jr = {}
    for r in resources:
        total_supply = sum(S.get((i, r), 0) for i in active_hubs)

        # Weighted requests
        weighted = {j: acc_scores[j] * D.get((j, r), 0) for j in barangays}
        total_weighted = sum(weighted.values())

        if total_weighted <= 0:
            for j in barangays:
                phi_jr[(j, r)] = 0.0
            continue

        for j in barangays:
            share = (weighted[j] / total_weighted) * total_supply
            demand = D.get((j, r), 0)
            if demand > 0:
                phi_jr[(j, r)] = min(1.0, share / demand)
            else:
                phi_jr[(j, r)] = 1.0

    # Overall fulfillment: min across resources (Liebig's Law)
    phi_j = {}
    for j in barangays:
        phi_j[j] = min(phi_jr.get((j, r), 0) for r in resources)

    # Compute diagnostic metrics
    metrics = _compute_baseline_metrics(
        barangays, resources, phi_jr, phi_j, acc_scores, district_map
    )

    return {
        "phi_jr": phi_jr,
        "phi_j": phi_j,
        "accessibility_scores": acc_scores,
        "metrics": metrics,
    }


# -- Function: Compute decentralized Monte Carlo allocation baseline with demand inflation --
def compute_baseline_decentralized(barangays, resources, D, S, hubs, params,
                                   district_map, urban_rural_map,
                                   n_runs=30, seed=42):
    """
    Decentralized request-based allocation baseline with bullwhip-type
    request inflation.

    Each Monte Carlo run:
      1. Each barangay draws inflation_j ~ Uniform(0, 1.0)
      2. Urban barangays get +0.3 inflation bonus (better access = louder
         voice in requesting), capped at 1.0
      3. Inflated request: R_jr = D_jr × (1 + inflation_j)
      4. Hub fills proportionally: w_jr = R_jr × min(1, supply_r / Σ R_jr)
      5. Fulfillment: phi_jr = w_jr / D_jr (can exceed 1.0 = over-allocation)

    Over n_runs, reports mean ± std of key disparity metrics.

    Args:
        n_runs: number of Monte Carlo runs (default 30)
        seed: random seed for reproducibility

    Returns:
        dict: {
            "mean_phi_j": {j: mean_phi across runs},
            "run_results": list of per-run dicts,
            "aggregate_metrics": dict with mean/std of key metrics,
            "accessibility_scores": {j: score},
        }
    """
    rng = np.random.default_rng(seed)

    hmax = params["Hmax"]
    hub_total_cap = {}
    for i in hubs:
        hub_total_cap[i] = sum(S.get((i, r), 0) for r in resources)
    active_hubs = sorted(hub_total_cap, key=hub_total_cap.get,
                         reverse=True)[:hmax]

    acc_scores = compute_accessibility_scores(
        barangays, district_map, urban_rural_map
    )

    run_results = []
    phi_j_accum = defaultdict(list)

    for run_idx in range(n_runs):
        # Draw per-barangay inflation
        inflation = {}
        for j in barangays:
            base = rng.uniform(0, 1.0)
            is_urban = urban_rural_map.get(j, "Rural") == "Urban"
            bonus = 0.3 if is_urban else 0.0
            inflation[j] = min(1.0, base + bonus)

        # Compute inflated requests and proportional rationing
        phi_jr = {}
        over_alloc = 0.0
        under_alloc = 0.0

        for r in resources:
            total_supply = sum(S.get((i, r), 0) for i in active_hubs)

            # Inflated requests
            R = {j: D.get((j, r), 0) * (1 + inflation[j]) for j in barangays}
            total_R = sum(R.values())

            # Proportional rationing
            ration_factor = min(1.0, total_supply / total_R) if total_R > 0 else 0

            for j in barangays:
                w_jr = R[j] * ration_factor
                d_jr = D.get((j, r), 0)

                if d_jr > 0:
                    phi_jr[(j, r)] = w_jr / d_jr
                else:
                    phi_jr[(j, r)] = 1.0

                # Over/under-allocation relative to true need
                if w_jr > d_jr:
                    over_alloc += w_jr - d_jr
                else:
                    under_alloc += d_jr - w_jr

        # Overall fulfillment per barangay (cap at 1.0 for comparability)
        phi_j = {}
        phi_j_uncapped = {}
        for j in barangays:
            raw = min(phi_jr.get((j, r), 0) for r in resources)
            phi_j_uncapped[j] = raw
            phi_j[j] = min(1.0, raw)
            phi_j_accum[j].append(phi_j[j])

        # Per-run metrics
        vals = list(phi_j.values())
        vals_uncapped = list(phi_j_uncapped.values())
        mean_phi = sum(vals) / len(vals)

        run_result = {
            "run": run_idx,
            "min_phi": min(vals),
            "max_phi": max(vals),
            "mean_phi": mean_phi,
            "std_phi": (sum((v - mean_phi) ** 2 for v in vals) / len(vals)) ** 0.5,
            "spread": max(vals) - min(vals),
            "total_over_allocation": over_alloc,
            "total_under_allocation": under_alloc,
            "max_uncapped_phi": max(vals_uncapped),
            "gini": _gini_coefficient(vals),
        }
        run_results.append(run_result)

    # Mean fulfillment per barangay across runs
    mean_phi_j = {j: np.mean(phi_j_accum[j]) for j in barangays}

    # Aggregate metrics across runs
    aggregate = {}
    for key in ("min_phi", "max_phi", "mean_phi", "std_phi", "spread",
                "total_over_allocation", "total_under_allocation", "gini"):
        values = [r[key] for r in run_results]
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # Correlation between mean fulfillment and accessibility
    acc_list = [acc_scores[j] for j in barangays]
    phi_list = [mean_phi_j[j] for j in barangays]
    aggregate["corr_phi_accessibility"] = float(np.corrcoef(acc_list, phi_list)[0, 1])

    return {
        "mean_phi_j": mean_phi_j,
        "run_results": run_results,
        "aggregate_metrics": aggregate,
        "accessibility_scores": acc_scores,
    }


# ======================================================================
# District-level analysis helpers
# ======================================================================

# -- Function: Compute mean fulfillment grouped by administrative district --
def compute_district_fulfillment(barangays, phi_j, district_map):
    """
    Compute mean fulfillment per district.

    Returns:
        dict: {district_name: mean_phi}
    """
    district_phis = defaultdict(list)
    for j in barangays:
        district = district_map.get(j, "Unknown")
        district_phis[district].append(phi_j[j])

    return {d: sum(vs) / len(vs) for d, vs in district_phis.items()}


# -- Function: Compare urban-core vs upland district fulfillment equity --
def compare_urban_vs_upland(barangays, phi_j, district_map):
    """
    Compare mean fulfillment of urban-core districts (Poblacion, Agdao)
    vs upland districts (Marilog, Paquibato).

    Returns:
        dict with urban_core_mean, upland_mean, difference
    """
    urban_vals = []
    upland_vals = []

    for j in barangays:
        d = district_map.get(j, "")
        if d in URBAN_CORE_DISTRICTS:
            urban_vals.append(phi_j[j])
        elif d in UPLAND_DISTRICTS:
            upland_vals.append(phi_j[j])

    urban_mean = sum(urban_vals) / len(urban_vals) if urban_vals else 0
    upland_mean = sum(upland_vals) / len(upland_vals) if upland_vals else 0

    return {
        "urban_core_mean": urban_mean,
        "urban_core_count": len(urban_vals),
        "upland_mean": upland_mean,
        "upland_count": len(upland_vals),
        "difference": urban_mean - upland_mean,
    }


# ======================================================================
# Internal helpers
# ======================================================================

# -- Function: Calculate baseline statistical and equity summary metrics --
def _compute_baseline_metrics(barangays, resources, phi_jr, phi_j,
                              acc_scores, district_map):
    """Compute summary metrics for a baseline result."""
    vals = list(phi_j.values())
    mean_phi = sum(vals) / len(vals)

    # Correlation between fulfillment and accessibility
    acc_list = [acc_scores[j] for j in barangays]
    phi_list = [phi_j[j] for j in barangays]
    corr = float(np.corrcoef(acc_list, phi_list)[0, 1])

    # Urban-core vs upland comparison
    urban_upland = compare_urban_vs_upland(barangays, phi_j, district_map)

    # Per-district means
    district_means = compute_district_fulfillment(barangays, phi_j, district_map)

    return {
        "min_phi": min(vals),
        "max_phi": max(vals),
        "mean_phi": mean_phi,
        "std_phi": (sum((v - mean_phi) ** 2 for v in vals) / len(vals)) ** 0.5,
        "gini": _gini_coefficient(vals),
        "corr_phi_accessibility": corr,
        "urban_vs_upland": urban_upland,
        "district_fulfillment": district_means,
    }


# -- Function: Calculate statistical Gini inequality coefficient --
def _gini_coefficient(values):
    """Compute the Gini coefficient of a list of values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    return (2 * cumsum) / (n * total) - (n + 1) / n


# -- Function: Compute hazard-weighted proportional allocation baseline --
def compute_baseline_weighted_proportional(barangays, resources, D, S, P, hubs, params, H=None):
    r"""
    Hazard-Weighted Proportional Baseline (Chapter 3, Eq 3.28 revised):
    Apportions available supplies in proportion to hazard-weighted demand:
    w_{jr} \propto H_j D_{jr} \implies \phi_{jr} \propto H_j,
    subject to anti-starvation floor and ceiling:
        \phi_{jr}^{wprop} = \min(1.0, \max(\varepsilon_{\min}, (H_j \cdot S_r^{\text{active}}) / \sum_k H_k D_{kr}))

    This ensures fulfillment scales with hazard severity rather than population size.

    Args:
        barangays: list of barangay names
        resources: list of commodity names (e.g. ['water', 'food'])
        D: demand dict {(j, r): value}
        S: supply dict {(i, r): value}
        P: composite vulnerability weight dict {j: P_j} (used if H is None)
        hubs: list of candidate hub names
        params: parameter dict with Hmax and eps_min
        H: optional dict of hazard tiers {j: H_j}; if None, inferred from P_j / D_{jr}

    Returns:
        phi_j: {j: overall_fulfillment}
        phi_jr: {(j, r): fulfillment_ratio}
    """
    hmax = params["Hmax"]
    eps_min = params.get("eps_min", 0.10)

    # Infer hazard tier H_j if not explicitly provided
    if H is None:
        r_ref = resources[0]
        ratios = {j: P[j] / D.get((j, r_ref), 1.0) for j in barangays}
        min_r = min(ratios.values()) if ratios else 1.0
        H = {j: ratios[j] / min_r if min_r > 0 else 1.0 for j in barangays}

    # Pick top Hmax hubs by capacity
    hub_total_cap = {}
    for i in hubs:
        hub_total_cap[i] = sum(S.get((i, r), 0) for r in resources)
    active_hubs = sorted(hub_total_cap, key=hub_total_cap.get, reverse=True)[:hmax]

    phi_jr = {}
    for r in resources:
        total_supply = sum(S.get((i, r), 0) for i in active_hubs)
        total_hazard_demand = sum(H[j] * D.get((j, r), 0) for j in barangays)

        for j in barangays:
            if total_hazard_demand > 0:
                raw_ratio = (H[j] * total_supply) / total_hazard_demand
            else:
                raw_ratio = 1.0
            phi_jr[(j, r)] = min(1.0, max(eps_min, raw_ratio))

    # Overall fulfillment under Liebig's Law of the Minimum
    phi_j = {j: min(phi_jr[(j, r)] for r in resources) for j in barangays}
    return phi_j, phi_jr

