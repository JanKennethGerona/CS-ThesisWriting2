"""
data_loader.py — Clean API for the multi-commodity MILP pipeline.

Provides a single load_data() entry point with:
  - Named-column xlsx reading via load_real_data
  - Data readiness gate (DEVELOPMENT vs RESEARCH mode)
  - Comprehensive validation before solving
  - Multi-commodity parameter structures

Usage:
    from data_loader import load_data
    data = load_data()  # development mode (default)
    data = load_data(mode="research")  # fails if placeholders detected
"""

import os
import pandas as pd
from load_real_data import (
    load_barangay_data, build_dataset, detect_provisional_status,
    RESOURCES, SPHERE_WATER_L_PER_PERSON, SPHERE_FOOD_KCAL_PER_PERSON,
)

# Expected districts in Davao City (11 administrative districts)
EXPECTED_DISTRICTS = {
    "Poblacion", "Talomo", "Agdao", "Buhangin", "Bunawan",
    "Paquibato", "Baguio", "Calinan", "Marilog", "Toril", "Tugbok",
}

EXPECTED_ROW_COUNT = 182


class DataValidationError(Exception):
    """Raised when data validation fails — model should NOT be solved."""
    pass


# -- Function: Validate data integrity and schema --
def validate_data(df):
    """
    Strict data validation. Returns list of error strings.
    If any errors, the model should not be solved.
    """
    errors = []

    # Row count
    if len(df) != EXPECTED_ROW_COUNT:
        errors.append(f"Expected {EXPECTED_ROW_COUNT} rows, got {len(df)}")

    # Unique barangay names
    if df["Barangay"].nunique() != len(df):
        dupes = df[df["Barangay"].duplicated(keep=False)]["Barangay"].tolist()
        errors.append(f"Duplicate barangay names: {dupes}")

    # Unique PSGC codes
    if df["10 Digit Code"].nunique() != len(df):
        errors.append("Duplicate PSGC codes detected")

    # No missing population
    null_pop = df["Population (2024 POPCEN)"].isnull().sum()
    if null_pop > 0:
        errors.append(f"{null_pop} rows with missing population")

    # No negative population
    neg_pop = (df["Population (2024 POPCEN)"] < 0).sum()
    if neg_pop > 0:
        errors.append(f"{neg_pop} rows with negative population")

    # Water demand recomputation
    expected_water = df["Population (2024 POPCEN)"] * SPHERE_WATER_L_PER_PERSON
    water_mismatch = (df["Demand (D_j)"] - expected_water).abs() > 0.01
    if water_mismatch.any():
        errors.append(
            f"{water_mismatch.sum()} rows where Demand (D_j) != "
            f"Population * {SPHERE_WATER_L_PER_PERSON}"
        )

    # Food demand recomputation
    expected_food = df["Population (2024 POPCEN)"] * SPHERE_FOOD_KCAL_PER_PERSON
    food_mismatch = (df["Food Demand (D_j, Food)"] - expected_food).abs() > 0.01
    if food_mismatch.any():
        errors.append(
            f"{food_mismatch.sum()} rows where Food Demand != "
            f"Population * {SPHERE_FOOD_KCAL_PER_PERSON}"
        )

    # Districts match expected set
    actual_districts = set(df["District"].unique())
    missing_districts = EXPECTED_DISTRICTS - actual_districts
    extra_districts = actual_districts - EXPECTED_DISTRICTS
    if missing_districts:
        errors.append(f"Missing expected districts: {missing_districts}")
    if extra_districts:
        errors.append(f"Unexpected districts: {extra_districts}")

    # No null vulnerability weights
    null_vuln = df["Vulnerability Weight (P_j)"].isnull().sum()
    if null_vuln > 0:
        errors.append(f"{null_vuln} rows with missing vulnerability weight")

    # No null vulnerability data source
    null_src = df["Vulnerability Data Source"].isnull().sum()
    if null_src > 0:
        errors.append(f"{null_src} rows with missing vulnerability data source")

    return errors


# -- Function: Load and validate dataset with mode gating --
def load_data(xlsx_path=None, mode="development"):
    """
    Load and validate all data for the multi-commodity MILP pipeline.

    Args:
        xlsx_path: Path to xlsx file (default: auto-detect relative to load_real_data.py)
        mode: "development" (allows placeholders) or "research" (rejects placeholders)

    Returns:
        dict with keys:
            hubs, barangays, resources, D, S, P, params, is_provisional,
            data_mode, barangay_details (DataFrame)

    Raises:
        DataValidationError if validation fails or if mode="research"
        and placeholder data is detected.
    """
    if mode not in ("development", "research"):
        raise ValueError(f"mode must be 'development' or 'research', got '{mode}'")

    # Load raw data
    df = load_barangay_data(xlsx_path)

    # Validate
    errors = validate_data(df)
    if errors:
        msg = "DATA VALIDATION FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        raise DataValidationError(msg)

    # Build multi-commodity dataset
    I, J, D, S, P, params, is_provisional = build_dataset(df)

    # Research mode gate
    if mode == "research" and is_provisional:
        raise DataValidationError(
            "DATA VALIDATION FAILED: Research mode requires all parameters "
            "to be from validated sources, but placeholder vulnerability data "
            "was detected. Replace placeholder P_j values with real hazard-"
            "derived vulnerability weights before running in research mode."
        )

    return {
        "hubs": I,
        "barangays": J,
        "resources": RESOURCES,
        "D": D,
        "S": S,
        "P": P,
        "params": params,
        "is_provisional": is_provisional,
        "data_mode": mode,
        "barangay_details": df,
    }


# -- Function: Print loaded dataset summary --
def print_data_summary(data):
    """Print a human-readable summary of the loaded data."""
    J = data["barangays"]
    I = data["hubs"]
    D = data["D"]
    S = data["S"]

    if data["is_provisional"]:
        print("\n" + "=" * 70)
        print("  WARNING: DEMONSTRATION ONLY -- PLACEHOLDER VULNERABILITY DATA")
        print("  Results from this run must NOT be presented as thesis findings.")
        print("=" * 70)

    print(f"\nData mode: {data['data_mode'].upper()}")
    print(f"Barangays: {len(J)}")
    print(f"Candidate hubs: {len(I)}")
    print(f"Resources: {data['resources']}")
    print(f"Total water demand: {sum(D[(j, 'water')] for j in J):,.0f} L/day")
    print(f"Total food demand:  {sum(D[(j, 'food')] for j in J):,.0f} kcal/day")
    print(f"Total water supply: {sum(S[(i, 'water')] for i in I):,.0f} L/day")
    print(f"Total food supply:  {sum(S[(i, 'food')] for i in I):,.0f} kcal/day")
    print(f"Hmax: {data['params']['Hmax']}")
    print(f"eps_min: {data['params']['eps_min']}")
    print(f"lambda_water: {data['params']['lambda_']['water']} "
          f"(theoretically grounded parameter)")
    print(f"lambda_food:  {data['params']['lambda_']['food']} "
          f"(theoretically grounded parameter)")
    print(f"is_provisional: {data['is_provisional']}")


if __name__ == "__main__":
    data = load_data(mode="development")
    print_data_summary(data)
    print("\nData loaded and validated successfully.")
