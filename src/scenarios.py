"""
scenarios.py — Disaster scenario engine for the multi-commodity MILP.

Provides 7 pre-defined disaster scenarios that modify supply and/or demand
to test the model under realistic conditions.

Usage:
    from scenarios import get_all_scenarios, apply_scenario
"""


# -- Function: Retrieve disaster scenario definitions --
def get_all_scenarios():
    """
    Returns a list of scenario definitions.
    Each scenario is a dict with:
      - name: human-readable name
      - description: what it tests
      - supply_factor: dict {resource: factor} or float for uniform
      - demand_factor: dict {resource: factor} or float for uniform
      - demand_surge_barangays: optional list of barangay names with 1.5× demand
    """
    return [
        {
            "name": "Scenario 1 -- Normal",
            "description": "100% demand, 100% supply (baseline)",
            "supply_factor": 1.0,
            "demand_factor": 1.0,
        },
        {
            "name": "Scenario 2 -- Moderate Disaster",
            "description": "100% demand, 75% supply",
            "supply_factor": 0.75,
            "demand_factor": 1.0,
        },
        {
            "name": "Scenario 3 -- Severe Disaster",
            "description": "100% demand, 50% supply",
            "supply_factor": 0.50,
            "demand_factor": 1.0,
        },
        {
            "name": "Scenario 4 -- Extreme Disaster",
            "description": "100% demand, 25% supply",
            "supply_factor": 0.25,
            "demand_factor": 1.0,
        },
        {
            "name": "Scenario 5 -- Uneven Impact",
            "description": "First 20% of barangays at 1.5x demand, rest at baseline",
            "supply_factor": 1.0,
            "demand_factor": 1.0,
            "demand_surge_fraction": 0.20,
            "demand_surge_multiplier": 1.5,
        },
        {
            "name": "Scenario 6 -- Water-Critical Event",
            "description": "Water supply at 40%, food supply at 100%",
            "supply_factor": {"water": 0.40, "food": 1.0},
            "demand_factor": 1.0,
        },
        {
            "name": "Scenario 7 -- Food Shortage",
            "description": "Food supply at 40%, water supply at 100%",
            "supply_factor": {"water": 1.0, "food": 0.40},
            "demand_factor": 1.0,
        },
    ]


# -- Function: Apply disaster scenario scaling to supply and demand --
def apply_scenario(scenario, D, S, barangays, resources):
    """
    Apply a scenario to the demand and supply dicts.

    Args:
        scenario: dict from get_all_scenarios()
        D: original demand dict {(j, r): value}
        S: original supply dict {(i, r): value}
        barangays: list of barangay names
        resources: list of resource names

    Returns:
        D_new, S_new — modified copies of the demand/supply dicts
    """
    D_new = dict(D)
    S_new = dict(S)

    # Apply supply factor
    sf = scenario["supply_factor"]
    if isinstance(sf, (int, float)):
        S_new = {k: v * sf for k, v in S.items()}
    elif isinstance(sf, dict):
        S_new = {}
        for (i, r), v in S.items():
            factor = sf.get(r, 1.0)
            S_new[(i, r)] = v * factor

    # Apply demand factor
    df = scenario["demand_factor"]
    if isinstance(df, (int, float)) and df != 1.0:
        D_new = {k: v * df for k, v in D.items()}
    elif isinstance(df, dict):
        D_new = {}
        for (j, r), v in D.items():
            factor = df.get(r, 1.0)
            D_new[(j, r)] = v * factor

    # Apply demand surge to subset of barangays
    if "demand_surge_fraction" in scenario:
        frac = scenario["demand_surge_fraction"]
        mult = scenario.get("demand_surge_multiplier", 1.5)
        n_surge = max(1, int(len(barangays) * frac))
        surge_brgys = set(barangays[:n_surge])
        for (j, r), v in list(D_new.items()):
            if j in surge_brgys:
                D_new[(j, r)] = v * mult

    return D_new, S_new
