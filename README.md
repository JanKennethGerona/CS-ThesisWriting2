# Davao City Multi-Commodity Resource Allocation Problem (RAP)

A two-stage deterministic Mixed-Integer Linear Programming (MILP) optimization model for post-disaster humanitarian resource allocation across the 182 barangays of Davao City, Philippines.

The formulation optimizes the allocation of multiple essential relief commodities (water and food) from candidate Disaster Risk Reduction and Management Operation Center (DRRMOC) hubs to affected barangays, minimizing total vulnerability-weighted deprivation cost subject to humanitarian standards, hub activation limits, and anti-starvation policy constraints.

---

## Repository Structure

```
CS-ThesisWriting2/
│
├── data/                                             # Raw input datasets & standards
│   ├── Davao_City_Barangays_PSA_PSGC_2026_with_Model_Columns.xlsx  # 182 barangays (PSA 2024 POPCEN)
│   ├── GovPH.pdf                                    # MGB hazard & susceptibility reference
│   └── Sphere_Standard.txt                          # Humanitarian minimum standards
│
├── docs/                                             # Research briefs and implementation notes
│   ├── Implementation_Brief.md                      # Thesis technical specifications
│   └── implementation_plan_review_and_recommendations.md  # Formulation review & recommendations
│
├── src/                                              # Optimization codebase
│   ├── __init__.py
│   ├── data_loader.py                               # Validated data loader & mode gate
│   ├── load_real_data.py                            # Parameter extraction & Sphere calculation
│   ├── davao_rap_milp.py                            # Pyomo MILP formulation & solver detection
│   ├── model.py                                     # Model interface & solution runner
│   ├── validation.py                                # Strict post-solve validation suite
│   ├── fairness.py                                  # Fairness metrics, re-solves, & baselines
│   └── scenarios.py                                 # 7 disaster scenario configurations
│
├── tests/                                            # Verification test suite
│   ├── __init__.py
│   └── test_pipeline.py                             # 46 automated unit and integration tests
│
├── results/                                          # Exported optimization results
│   ├── barangay_allocations.csv                     # 182-barangay per-commodity allocation table
│   └── summary_metrics.json                         # Structured run summary & metrics
│
├── .gitignore
├── ASSUMPTIONS.md                                    # Registry of all placeholder parameters
├── main.py                                           # Top-level executable pipeline
└── README.md
```

---

## Mathematical Formulation Summary

### Indices & Sets
- $i \in I$: Candidate relief hubs (e.g., Bucana, Buhangin, Ma-a, Tibungco, Calinan, Toril)
- $j \in J$: 182 Davao City barangays
- $r \in R$: Commodities, $R = \{\text{water}, \text{food}\}$

### Parameters
- $D_{jr}$: Demand for commodity $r$ in barangay $j$, computed from real 2024 POPCEN census data and Sphere minimum standards:
  - $S_{\text{water}} = 15\text{ L/person/day}$
  - $S_{\text{food}} = 2,100\text{ kcal/person/day}$
- $S_{ir}$: Supply capacity of hub $i$ for commodity $r$
- $P_j$: Hazard-derived vulnerability weight for barangay $j$
- $H_{\max}$: Maximum number of hubs activated simultaneously ($H_{\max} = 4$)
- $\varepsilon_{\min}$: Anti-starvation minimum fulfillment threshold ($\varepsilon_{\min} = 0.10$)
- $\lambda_r$: Deprivation escalation rate ($\lambda_{\text{water}} = 2.0$, $\lambda_{\text{food}} = 0.5$, provisional)

### Decision Variables
- $y_i \in \{0, 1\}$: Hub activation status (Stage 1)
- $x_{ijr} \ge 0$: Units of commodity $r$ shipped from hub $i$ to barangay $j$ (Stage 2)
- $w_{jr} = \sum_{i \in I} x_{ijr}$: Total commodity $r$ delivered to barangay $j$
- $\phi_{jr} = \frac{w_{jr}}{D_{jr}} \in [\varepsilon_{\min}, 1.0]$: Per-commodity fulfillment ratio
- $z_{jr} \ge 0$: Linearized exponential deprivation cost for commodity $r$

### Objective Function
$$\min \sum_{j \in J} P_j \sum_{r \in R} z_{jr}$$

### Post-Solve Overall Fulfillment
To eliminate distortion in the LP relaxation, overall fulfillment is computed post-solve:
$$\phi_j = \min_{r \in R} \phi_{jr}$$

---

## Getting Started

### Prerequisites
- Python 3.10+
- Dependencies: `pandas`, `openpyxl`, `pyomo`, `highs` (or `cbc` / `glpk`)

Install dependencies:
```bash
pip install pandas openpyxl pyomo highspy
```

### Running the Full Pipeline
Run the end-to-end pipeline in **development** mode (uses provisional/placeholder vulnerability weights and hub estimates):
```bash
python main.py
```

Run in **research** mode (enforces strict validation and will block execution if placeholder data is present):
```bash
python main.py research
```

### Running the Test Suite
Execute the 46 automated unit and integration tests:
```bash
python tests/test_pipeline.py
```

---

## Output Files

Executing `main.py` generates the following files in the `results/` directory:

1. **`results/barangay_allocations.csv`**:
   - `Barangay`, `District`
   - `phi_water`, `z_water`, `w_water` (Water fulfillment, deprivation, quantity)
   - `phi_food`, `z_food`, `w_food` (Food fulfillment, deprivation, quantity)
   - `phi_overall`: Post-solve minimum fulfillment $\phi_j = \min(\phi_{j,\text{water}}, \phi_{j,\text{food}})$
   - `limiting_resource`: Which commodity constrained overall fulfillment (`water` or `food`)

2. **`results/summary_metrics.json`**:
   - Optimal objective value
   - Active hub roster ($H_{\max} = 4$)
   - City-wide fulfillment statistics (min, max, mean per resource and overall)
   - Fairness metrics (Nash welfare score, fulfillment-ratio disparity count)
   - Baseline proportional comparison
   - 6-point sensitivity sweep ($100\%$ down to $10\%$)
   - 7 disaster scenario results
   - Tangent piecewise linear approximation error audit

---

## Data Provenance & Assumptions

For a complete audit of all provisional constants, empirical data sources, and guidelines for transitioning from demonstration to thesis reporting, refer to:
- [ASSUMPTIONS.md](file:///c:/Users/kenge/Documents/GitHub/CS-ThesisWriting2/ASSUMPTIONS.md)
