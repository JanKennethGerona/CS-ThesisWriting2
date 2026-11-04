"""
Data loader: bridges the real PSA/PSGC barangay workbook into the
multi-commodity two-stage MILP (davao_rap_milp.py), using clearly-flagged
PLACEHOLDER values wherever real data (vulnerability, hub capacity, lambda)
is not yet available. Nothing here is fabricated to look final -- every
placeholder is named and easy to grep/replace once CDRRMO / hazard-map
data arrives.

Multi-commodity extension (v6): loads both water and food demands from
the xlsx, constructs per-resource supply capacities, and provides
per-resource lambda placeholders.

Usage:
    python load_real_data.py
"""

import os
import pandas as pd
import math

# Default xlsx path -- search data/ directory, parent data/, or local dir
_XLSX_FILENAME = "Davao_City_Barangays_PSA_PSGC_2026_with_Model_Columns.xlsx"
_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", _XLSX_FILENAME),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", _XLSX_FILENAME),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), _XLSX_FILENAME),
    os.path.join(os.getcwd(), "data", _XLSX_FILENAME),
    os.path.join(os.getcwd(), _XLSX_FILENAME),
]

_DEFAULT_XLSX = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])

# ----------------------------------------------------------------------
# PLACEHOLDER REGISTRY
# Every value below is a stand-in, not a real figure. Search for
# "PLACEHOLDER" before trusting any output of this script for real
# decision-making.
# ----------------------------------------------------------------------

# Sphere standards — these ARE real values from Sphere_Standard.txt
SPHERE_WATER_L_PER_PERSON = 15.0       # L/person/day
SPHERE_FOOD_KCAL_PER_PERSON = 2100.0   # kcal/person/day

# Deprivation escalation rates — THEORETICALLY GROUNDED SENSITIVITY PARAMETERS.
# Water deprivation escalates significantly faster than food deprivation in short-term
# disaster relief (acute dehydration risks severe morbidity within 48-72h vs caloric
# deficit progression over weeks), hence lambda_water (2.0) > lambda_food (0.5).
# Robustness is evaluated across parametric sensitivity sweeps lambda in [0.1, 5.0].
LAMBDA_WATER = 2.0
LAMBDA_FOOD = 0.5
PLACEHOLDER_LAMBDA_WATER = LAMBDA_WATER
PLACEHOLDER_LAMBDA_FOOD = LAMBDA_FOOD

# Hub infrastructure — entirely unknown, pure placeholders
PLACEHOLDER_HUB_COUNT = 6           # PLACEHOLDER: assumed number of DRRMOC hubs
PLACEHOLDER_HMAX = 4                # PLACEHOLDER: assumed simultaneous-activation cap
PLACEHOLDER_SUPPLY_COVERAGE_RATIO = 0.5  # PLACEHOLDER: hubs cover 50% of city demand

# Resource list
RESOURCES = ["water", "food"]


def load_barangay_data(path=None):
    """
    Read barangay data from the xlsx using pandas named-column access.

    Returns a DataFrame with all columns, plus computed verification columns.
    """
    if path is None:
        path = _DEFAULT_XLSX

    df = pd.read_excel(path, sheet_name="Davao City Barangays")

    # Verify expected columns exist
    required_cols = [
        "Barangay", "District", "10 Digit Code", "Correspondence Code",
        "Urban/Rural", "Population (2024 POPCEN)", "Sphere Minimum per Person",
        "Demand (D_j)", "Vulnerability Indicator", "Vulnerability Weight (P_j)",
        "Household Count", "Vulnerability Data Source",
        "Food Minimum per Person", "Food Demand (D_j, Food)",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in xlsx: {missing}")

    # Drop any fully-empty rows (safety)
    df = df.dropna(subset=["Barangay"]).reset_index(drop=True)

    return df


def detect_provisional_status(df):
    """
    Check the Vulnerability Data Source column for the substring "PLACEHOLDER".
    Returns True if ANY row contains it.
    """
    return df["Vulnerability Data Source"].str.contains(
        "PLACEHOLDER", case=False, na=False
    ).any()


def build_dataset(df, hub_count=None, hmax=None, supply_coverage=None):
    """
    Produces the full multi-commodity parameter set the MILP needs:
    D[(j,r)], S[(i,r)], P[j], params, is_provisional.

    Uses real population for D_{jr} and PLACEHOLDERS for hub infrastructure.
    """
    if hub_count is None:
        hub_count = PLACEHOLDER_HUB_COUNT
    if hmax is None:
        hmax = PLACEHOLDER_HMAX
    if supply_coverage is None:
        supply_coverage = PLACEHOLDER_SUPPLY_COVERAGE_RATIO

    barangay_names = df["Barangay"].tolist()
    J = barangay_names

    # --- D_{jr}: REAL population × REAL Sphere standard per resource ---
    D = {}
    for _, row in df.iterrows():
        name = row["Barangay"]
        pop = row["Population (2024 POPCEN)"]
        D[(name, "water")] = pop * SPHERE_WATER_L_PER_PERSON
        D[(name, "food")] = pop * SPHERE_FOOD_KCAL_PER_PERSON

    # --- P_j: read directly from xlsx (may be placeholder) ---
    P = {row["Barangay"]: row["Vulnerability Weight (P_j)"]
         for _, row in df.iterrows()}

    # --- is_provisional ---
    is_provisional = detect_provisional_status(df)

    # --- Hubs: PLACEHOLDER — picks N largest-population barangays as
    # candidate hub sites, purely so the model has something to solve.
    # Replace with actual DRRMOC facility list + real capacities. ---
    sorted_by_pop = df.sort_values(
        "Population (2024 POPCEN)", ascending=False
    ).head(hub_count)
    hub_names = [f"HUB_{name}" for name in sorted_by_pop["Barangay"]]
    I = hub_names

    # Per-resource hub capacities (placeholder: fraction of total demand / hub_count)
    total_demand_water = sum(D[(j, "water")] for j in J)
    total_demand_food = sum(D[(j, "food")] for j in J)
    per_hub_water = (total_demand_water * supply_coverage) / hub_count
    per_hub_food = (total_demand_food * supply_coverage) / hub_count

    S = {}
    for hub in I:
        S[(hub, "water")] = per_hub_water
        S[(hub, "food")] = per_hub_food

    params = {
        "Hmax": hmax,
        "eps_min": 0.10,       # policy choice, reasonable to fix now
        "eps_env": 0.15,       # policy choice, reasonable to fix now
        "lambda_": {
            "water": PLACEHOLDER_LAMBDA_WATER,
            "food": PLACEHOLDER_LAMBDA_FOOD,
        },
    }

    return I, J, D, S, P, params, is_provisional


if __name__ == "__main__":
    df = load_barangay_data()
    print(f"Loaded {len(df)} barangays from real PSA data.")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    I, J, D, S, P, params, is_provisional = build_dataset(df)

    if is_provisional:
        print("\n" + "=" * 70)
        print("  DEMONSTRATION ONLY — PLACEHOLDER VULNERABILITY DATA")
        print("=" * 70)

    print(f"\nResources: {RESOURCES}")
    print(f"Candidate hubs (PLACEHOLDER selection): {len(I)}")
    print(f"Total water demand (real pop x Sphere): "
          f"{sum(D[(j, 'water')] for j in J):,.0f} L/day")
    print(f"Total food demand (real pop x Sphere):  "
          f"{sum(D[(j, 'food')] for j in J):,.0f} kcal/day")
    print(f"Total placeholder water supply: "
          f"{sum(S[(i, 'water')] for i in I):,.0f} L/day")
    print(f"Total placeholder food supply:  "
          f"{sum(S[(i, 'food')] for i in I):,.0f} kcal/day")
    print(f"is_provisional: {is_provisional}")

    print("\nFirst 5 barangays with computed D_jr:")
    for name in J[:5]:
        pop = df.loc[df["Barangay"] == name, "Population (2024 POPCEN)"].iloc[0]
        print(f"  {name}: pop={pop:,}  "
              f"D_water={D[(name, 'water')]:,.0f}  "
              f"D_food={D[(name, 'food')]:,.0f}  "
              f"P_j={P[name]:.6f}")

    print("\n--- Ready to feed into build_model() from davao_rap_milp.py ---")
