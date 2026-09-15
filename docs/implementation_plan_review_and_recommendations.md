# Implementation Plan Review and Recommendations

## Research Project

**Exact Optimization of City-Wide Disaster Relief: A Two-Stage MILP for Resource Allocation in Davao City**

## Overall Assessment

### Overall Rating: 8.0 / 10

The current implementation plan is strong and substantially better than the existing single-commodity implementation. The transition from water-only to water + food is a meaningful structural improvement, and the proposed software architecture, validation suite, fairness evaluation, and sensitivity analysis provide a solid foundation.

However, several mathematical and methodological issues should be corrected before the final implementation begins. The most important concern is the current definition and implementation of the overall fulfillment variable `φ_j`. Other important improvements involve placeholder data, the interpretation of the two-stage structure, fairness metrics, baseline comparisons, disaster scenarios, and parameter calibration.

| Area | Rating | Assessment |
|---|---:|---|
| Alignment with research paper | 9/10 | Very strong |
| Mathematical formulation | 8/10 | Good, but `φ_j` needs correction |
| Multi-commodity extension | 8.5/10 | Strong direction |
| Data implementation | 6.5/10 | Still heavily dependent on placeholders |
| Software architecture | 8.5/10 | Clean and maintainable |
| Validation/testing | 8.5/10 | Strong |
| Research defensibility | 7.5/10 | Assumptions need stronger justification |
| Practical disaster-relief realism | 7/10 | Allocation is good, but transportation is intentionally excluded |
| **Overall** | **8/10** | **Good implementation plan; revise before final coding** |

The research paper establishes a centralized Hub-and-Spoke framework, a two-stage MILP, deprivation costs, and fairness evaluation. The paper also explicitly separates the primary deprivation-minimization objective from post-hoc fairness evaluation.

---

# 1. What the Implementation Plan Gets Right

## 1.1 Moving from Water-Only to Water + Food

This is a good improvement and should be retained.

The research discusses food and water as life-sustaining resources, making a multi-commodity formulation more consistent with the stated humanitarian problem.

The implementation correctly changes the allocation variable from:

```text
x_ij
```

to:

```text
x_ijr
```

and introduces resource-indexed:

```text
D_jr
S_ir
φ_jr
z_jr
```

This is a genuine structural extension rather than simply adding another spreadsheet column.

The Sphere standards file currently provides:

- Water = **15 L/person/day**
- Food = **2,100 kcal/person/day**

### Recommendation

**Keep water + food.**

Do not add shelter and waste yet. Those are better treated as future extensions because the current thesis already has a sufficiently ambitious mathematical and implementation scope.

---

# 2. Most Important Mathematical Issue: `φ_j`

The implementation plan defines:

```text
φ_j = min_r φ_jr
```

and imposes:

```text
φ_j ≤ φ_jr
```

for every resource.

This is not sufficient by itself to guarantee:

```text
φ_j = min(φ_j,water, φ_j,food)
```

It only guarantees:

```text
φ_j ≤ min(φ_j,water, φ_j,food)
```

## Why this is a problem

Suppose:

```text
φ_j,water = 0.80
φ_j,food  = 0.60
```

The current constraints allow:

```text
φ_j = 0.10
```

or:

```text
φ_j = 0.40
```

or:

```text
φ_j = 0.60
```

All of these satisfy:

```text
φ_j ≤ 0.80
φ_j ≤ 0.60
```

The implementation plan says that the solver will push `φ_j` upward, but the current primary objective is:

```text
Minimize Σ_j Σ_r P_j z_jr
```

and `φ_j` does not appear directly in that objective.

Therefore, there is no guarantee that the solver will maximize `φ_j`.

## Recommended Fix

### Preferred approach: calculate `φ_j` after solving

Do not make `φ_j` a decision variable.

Keep:

```text
φ_jr
```

inside the MILP and calculate:

```python
phi_j = min(
    phi_jr[j, "water"],
    phi_jr[j, "food"]
)
```

after the model is solved.

The anti-starvation requirement can then be written directly as:

```text
φ_jr ≥ ε_min    for all j,r
```

This is simpler, easier to verify, and avoids unnecessary model complexity.

### Alternative approach

If `φ_j` must remain a decision variable, use a lexicographic objective:

1. Minimize deprivation cost.
2. Among solutions with minimum deprivation cost, maximize `Σ_j φ_j`.

This would force:

```text
φ_j = min_r φ_jr
```

at the final solution.

### Recommendation

**Use the first approach.**

---

# 3. Clarify the Meaning of "Two-Stage MILP"

The paper describes:

- Stage 1: hub activation
- Stage 2: resource allocation

The implementation has:

```text
y_i = hub activation
x_ijr = resource allocation
```

However, these decisions are currently optimized together in one deterministic MILP.

This is different from classical two-stage stochastic programming, where first-stage decisions are made before uncertainty is realized and second-stage decisions respond to scenarios.

This does not invalidate the research because the paper explicitly defines its own two-stage structure in terms of activation followed by allocation.

## Recommendation

Use terminology such as:

> **Two-stage deterministic MILP consisting of hub activation and resource allocation decisions.**

This is more precise and reduces the risk of a panel question about where uncertainty or recourse is represented.

---

# 4. Placeholder Data Is the Biggest Implementation Risk

The current data loader contains placeholders for important parameters such as:

```python
PLACEHOLDER_VULNERABILITY_WEIGHT = 1.0
PLACEHOLDER_LAMBDA = 2.0
PLACEHOLDER_HUB_COUNT = 6
PLACEHOLDER_HMAX = 4
PLACEHOLDER_SUPPLY_COVERAGE_RATIO = 0.5
```

The current implementation clearly identifies these as stand-ins, which is good practice.

However, there must be a strict distinction between:

### Development/Prototype Runs

Placeholder values may be used to verify that the software works.

### Final Research Runs

Critical real-world parameters should not remain placeholders if the resulting allocation is presented as a realistic Davao City decision.

## Recommendation: Data Readiness Gate

Create two run states.

### Development Mode

```text
Real population: YES
Sphere standards: YES
Vulnerability: PLACEHOLDER
Hub capacity: PLACEHOLDER
Lambda: PLACEHOLDER

RESULT STATUS:
DEMONSTRATION ONLY
```

### Research Mode

```text
Real population: YES
Sphere standards: YES
Vulnerability: YES
Hub capacity: YES
Lambda: CALIBRATED

RESULT STATUS:
VALIDATED RESEARCH RUN
```

The `is_provisional` flag should be more than an output field. It should become a validation state that prevents the system from presenting provisional results as final results.

---

# 5. Hub Selection Is Currently Too Artificial

The current loader chooses hypothetical hub locations by selecting the largest-population barangays.

This is acceptable for a software smoke test, but it is not a defensible representation of actual disaster-relief infrastructure.

## Recommended Final Hub Data

Use an explicit structure such as:

```text
Hub_ID
Hub_Name
Hub_Type
Location
Water_Capacity
Food_Capacity
Activation_Status
```

Candidate hubs should ideally come from CDRRMO or another authoritative government source.

Do not finalize the hub-selection logic until actual hub information is available.

---

# 6. Reconsider How `λ` Is Handled

The current plan proposes:

```text
λ_water = 2.0
λ_food = 0.5
```

The conceptual rationale—that water deprivation may escalate faster than food deprivation during short-term crises—is reasonable as a modeling hypothesis.

However, these are still assumed values.

The research methodology indicates that deprivation parameters should be calibrated using evidence such as time-to-criticality or relevant response information.

## Recommendation

Until empirical calibration is available, describe these values as:

> **provisional sensitivity-analysis values**

rather than final calibrated parameters.

---

# 7. Test Whether Resource-Specific `λ` Values Actually Matter

The use of:

```text
λ_water
λ_food
```

is theoretically useful, but it also introduces additional assumptions.

A strong research experiment would compare:

```text
λ_water = λ_food
```

against cases where:

```text
λ_water > λ_food
```

and:

```text
λ_water < λ_food
```

This allows the research to answer an important question:

> **How sensitive is the optimal allocation to the relative urgency of water versus food deprivation?**

This is a stronger empirical contribution than simply selecting one assumed pair of values.

---

# 8. The Deprivation Approximation Is Good, but Validate It Quantitatively

The implementation uses tangent lines to approximate:

```text
e^(λ(1-φ))
```

This is a good way to preserve the MILP formulation.

The planned increase from five to 21 breakpoints is also a useful improvement.

However, the thesis should measure the approximation error rather than simply stating that 21 breakpoints were used.

## Recommended Validation

For each test point, report:

```text
Exact exponential value
MILP approximation
Absolute error
Relative error
```

at:

```text
φ = 0.00
φ = 0.05
φ = 0.10
...
φ = 1.00
```

## Even Better

Compare:

```text
5 breakpoints
11 breakpoints
21 breakpoints
41 breakpoints
```

and examine how:

- approximation error changes,
- objective value changes,
- allocation changes.

This provides a strong numerical validation experiment.

---

# 9. Fairness Metrics Need Careful Operational Definitions

The paper makes an important distinction between the primary optimization objective and fairness metrics.

The implementation should preserve that distinction.

## Max-Min

The planned approach is appropriate:

```text
maximize Φ_min
subject to φ_j ≥ Φ_min
```

## Pure Egalitarian

The proposed re-solve with equal fulfillment levels is also consistent with the intended definition.

## Nash

The current implementation calculates:

```python
sum(log(phi_j))
```

after solving.

That is a **Nash welfare score**, not a Nash optimization.

Unless the model actually re-solves with:

```text
maximize Σ ln(φ_j)
```

the output should be described as:

> **Nash welfare evaluation**

rather than a Nash-optimal solution.

## Envy-Freeness

The current code checks:

```python
abs(phi_a - phi_b) <= eps
```

This is more precisely a fulfillment-ratio disparity test than full economic/game-theoretic envy-freeness.

A transparent description would be:

> **Operationalized envy-freeness based on fulfillment-ratio disparity within a specified tolerance.**

This is more defensible than claiming a complete envy-freeness formulation.

---

# 10. Pareto-Efficiency Must Actually Be Implemented

The implementation plan mentions Pareto-efficiency, but the current `evaluate_fairness()` function does not implement a real Pareto test.

The current evaluator reports:

- fulfillment by node,
- minimum fulfillment,
- sorted fulfillment vector,
- Nash log sum,
- envy violations.

A proper Pareto check should therefore be added or the claimed metric should be removed from the implementation description.

This is an important consistency issue between:

```text
Research paper
        ↓
Implementation plan
        ↓
Actual code
```

---

# 11. Alternate-Objective Runs Are One of the Strongest Parts

The plan to compare different allocation objectives is very good.

The implementation should produce solutions for:

### Model A
Deprivation minimization

### Model B
Max Utilitarian

### Model C
Pure Egalitarian

### Model D
Max-Min

Then compare:

```text
Total/weighted deprivation
Average fulfillment
Minimum fulfillment
Maximum fulfillment
Standard deviation
Water fulfillment
Food fulfillment
Number of underserved barangays
```

This directly demonstrates the trade-offs between different interpretations of fairness.

It also makes the research more than a demonstration that an optimizer can produce a solution.

---

# 12. Add a Baseline Model

This is one of the strongest recommended additions.

A proposed MILP should be compared against a simpler allocation rule.

At minimum, implement:

### Baseline 1 — Proportional Allocation

Each barangay receives an equivalent fulfillment ratio subject to available supply.

### Baseline 2 — Population-Proportional Allocation

Allocation is driven primarily by population.

### Proposed Model — MILP

The deprivation/fairness-oriented allocation model.

## Example Comparison

| Metric | Proportional | MILP |
|---|---:|---:|
| Total fulfillment | | |
| Minimum fulfillment | | |
| Average fulfillment | | |
| Worst barangay | | |
| Weighted deprivation | | |
| Water fulfillment | | |
| Food fulfillment | | |
| Equity metric | | |

This allows the thesis to make a much stronger claim:

> The proposed MILP produces a measurably different and potentially more equitable allocation than conventional proportional allocation.

---

# 13. Add Disaster Scenarios

The current supply sensitivity experiment is useful, but a disaster-relief model should also be tested under distinct operational scenarios.

Recommended scenarios:

### Scenario 1 — Normal Shortage

```text
100% demand
100% supply
```

### Scenario 2 — Moderate Disaster

```text
100% demand
75% supply
```

### Scenario 3 — Severe Disaster

```text
100% demand
50% supply
```

### Scenario 4 — Extreme Disaster

```text
100% demand
25% supply
```

### Scenario 5 — Uneven Impact

Some barangays experience increased demand, such as:

```text
D_j × 1.5
```

while other barangays remain at baseline demand.

### Scenario 6 — Water-Critical Event

Water supply is sharply reduced while food supply remains relatively available.

### Scenario 7 — Food Shortage

Food supply is reduced while water remains relatively available.

This makes the multi-commodity extension substantially more meaningful.

---

# 14. Recommended Implementation Order

The existing implementation order is reasonable but can be improved.

## Phase 1 — Mathematical Core

```text
Water-only
    ↓
Validate MILP
    ↓
Verify tangent approximation
```

## Phase 2 — Multi-Commodity

```text
Water + Food
    ↓
Validate resource-indexed constraints
    ↓
Validate limiting-resource calculation
```

## Phase 3 — Real Data

```text
PSA population
    ↓
Sphere standards
    ↓
CDRRMO hubs
    ↓
CDRRMO inventory
```

## Phase 4 — Vulnerability

```text
P_j
```

## Phase 5 — Deprivation Calibration

```text
λ_water
λ_food
```

## Phase 6 — Validation

```text
Baseline comparison
    ↓
MILP
    ↓
Fairness comparison
    ↓
Sensitivity
    ↓
Disaster scenarios
```

## Phase 7 — Optional Future Extension

```text
Transportation capacity
```

Do not implement transportation before the core model is completely validated.

---

# 15. Do Not Add Transportation Yet

The current plan is correct to defer:

```text
Σ_r v_r x_ijr ≤ C_ij
```

and vehicle-routing decisions.

The paper explicitly limits the study to allocation/assignment and excludes physical routing and transportation logistics.

Adding routing now could turn the project into a much larger combination of:

- location-allocation,
- capacitated transportation,
- vehicle routing,
- multi-commodity flow.

The current research already has enough mathematical complexity for a thesis.

---

# 16. Recommended Software Architecture

The proposed architecture is already good:

```text
project/
│
├── data/
│
├── data_loader.py
├── model.py
├── fairness.py
├── main.py
├── test_pipeline.py
├── ASSUMPTIONS.md
│
└── results/
```

I recommend adding two files:

```text
scenarios.py
validation.py
```

### `scenarios.py`

Responsible for:

```text
normal
moderate shortage
severe shortage
water shortage
food shortage
uneven disaster impact
```

### `validation.py`

Responsible for:

```text
constraint verification
tangent approximation error
solution feasibility
baseline comparison
data integrity
```

This prevents `main.py` from becoming overly large.

---

# 17. Expand the Automated Test Suite

The proposed test suite is already strong.

Add the following:

## Resource Conservation

For every hub and resource:

```text
allocated ≤ available inventory
```

## Demand Conservation

For every barangay/resource:

```text
received = φ_jr × demand
```

## Non-Negativity

```text
x_ijr ≥ 0
```

## Activation Link

If:

```text
y_i = 0
```

then:

```text
x_ijr = 0
```

## Hub Activation Limit

```text
Σ y_i ≤ Hmax
```

## Resource Independence

A food shortage should not create additional water supply, and vice versa.

## Limiting Resource

```text
φ_j = min(φ_jwater, φ_jfood)
```

## Placeholder Protection

A research-mode run should fail or be explicitly marked provisional when required real-world parameters remain placeholders.

---

# 18. Make the Data Loader Strict

The current data loader extracts population and generates some parameters.

The revised loader should validate:

```text
182 barangays
182 unique barangay names
182 unique PSGC codes
no missing population
no negative population
water demand = population × 15
food demand = population × 2100
no duplicate records
all expected districts present
all hub IDs unique
all resource names valid
```

If validation fails:

```text
DO NOT SOLVE
```

Return:

```text
DATA VALIDATION FAILED
```

rather than producing a mathematically valid model from invalid input data.

---

# 19. Clarify the Meaning of `P_j`

The model uses:

```text
Σ_j Σ_r P_j z_jr
```

The implementation should define exactly what `P_j` means.

Possible interpretations include:

### A. Population

Population is already included in demand, so using it again as an objective weight may double-count population.

### B. Vulnerability

This is cleaner if population is already embedded in demand.

### C. Population × Vulnerability

This explicitly weights both factors.

## Recommended Definition

Use:

```text
P_j = vulnerability priority weight
```

and keep population as part of:

```text
D_jr
```

This gives a clear separation:

```text
Demand = population
Priority = vulnerability
```

This is easier to explain and defend.

---

# 20. Avoid Claims That the Model Guarantees Survival

The thesis should avoid implying that the optimization model itself guarantees survival.

The model can guarantee:

> compliance with specified mathematical constraints and optimization criteria.

It cannot guarantee actual survival because real disaster outcomes also depend on:

- delivery time,
- road accessibility,
- inventory accuracy,
- disaster severity,
- population movement,
- infrastructure damage,
- transportation,
- implementation decisions.

Prefer wording such as:

> **supports equitable resource allocation**

rather than:

> **guarantees survival**

This will make the research more academically defensible.

---

# 21. Recommended Final Architecture

```text
                  REAL DATA
                     │
                     ▼
             ┌─────────────────┐
             │  Data Loader    │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Data Validation │
             └────────┬────────┘
                      │
              ┌───────┴────────┐
              │                │
              ▼                ▼
          WATER DATA       FOOD DATA
              │                │
              └───────┬────────┘
                      ▼
             ┌─────────────────┐
             │ Scenario Engine │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Two-Stage MILP │
             │                 │
             │ Stage 1: y_i    │
             │ Stage 2: x_ijr  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Solution Verify │
             └────────┬────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Fairness     Baseline    Sensitivity
      Analysis     Compare      Analysis
          │           │           │
          └───────────┼───────────┘
                      ▼
              FINAL RESULTS
```

---

# 22. Recommended Final Research Experiments

## Experiment 1 — Model Validation

Water-only toy model with a known/simple solution.

**Purpose:** demonstrate that the mathematical implementation works.

## Experiment 2 — Multi-Commodity Validation

Water + food toy model.

**Purpose:** verify the resource dimension and limiting-resource logic.

## Experiment 3 — Real Davao Baseline

Population-proportional or proportional allocation.

**Purpose:** provide a comparison point.

## Experiment 4 — Real Davao MILP

Run the proposed model using validated data.

**Purpose:** demonstrate the proposed optimization.

## Experiment 5 — Fairness Comparison

Compare:

```text
Deprivation minimization
Max Utilitarian
Pure Egalitarian
Max-Min
Nash welfare
```

**Purpose:** show fairness trade-offs.

## Experiment 6 — Supply Sensitivity

```text
100%
80%
60%
40%
20%
```

**Purpose:** study scarcity behavior and the point at which the system becomes infeasible.

## Experiment 7 — Commodity Sensitivity

Compare water-shortage and food-shortage scenarios.

**Purpose:** demonstrate why a multi-commodity formulation is useful.

## Experiment 8 — Vulnerability Sensitivity

Change:

```text
P_j
```

and observe allocation changes.

**Purpose:** determine how strongly vulnerability weighting influences who receives aid.

## Experiment 9 — Tangent Approximation

Compare:

```text
5
11
21
41
```

breakpoints.

**Purpose:** demonstrate numerical accuracy and justify the selected number of breakpoints.

---

# 23. Final Recommended Changes for Version 6

## Keep

- Water + food
- 21 tangent breakpoints
- Per-resource supply
- Per-resource deprivation
- Centralized hub model
- Fairness re-solves
- Sensitivity analysis
- Automated tests
- `is_provisional`
- Future transportation extension
- CSV/JSON results

## Modify

- `φ_j` definition
- `λ_water` / `λ_food` treatment
- terminology around the two-stage structure
- envy-freeness terminology
- Pareto implementation
- placeholder handling
- hub selection
- vulnerability-weight definition
- interpretation of Nash

## Add

- Baseline allocation
- Disaster scenarios
- Approximation-error experiment
- Data readiness gate
- Strict data validation
- Resource-specific shortage experiments
- Clear separation of development vs research runs

---

# 24. Final Assessment

The implementation plan is on the right track and should not be discarded or redesigned from scratch.

The strongest improvement is the recognition that moving from water-only to water + food is a structural mathematical extension rather than merely adding another spreadsheet column.

However, the implementation should **not begin as a blind line-by-line patch of the current code**. The current Python model is still fundamentally single-commodity, with variables such as `x[i,j]`, `w[j]`, `phi[j]`, and `z[j]`.

The v5 implementation plan is correct to treat the multi-commodity version as a structural rewrite.

### Highest-Priority Fixes

1. **Fix the mathematical definition of `φ_j`.**
2. **Separate development/placeholder runs from final research runs.**
3. **Add baseline and disaster-scenario experiments.**
4. **Tighten the terminology and definitions of two-stage MILP, Nash, envy-freeness, `P_j`, and λ calibration.**
5. **Implement a strict validation layer before solving.**

### Target Rating After Revision

With these changes, the implementation plan could reasonably reach approximately:

**9.0–9.2 / 10**

for an undergraduate CS/IT research project.

The next implementation should be based on a revised **Version 6** of the plan that incorporates these mathematical, methodological, and validation changes.
