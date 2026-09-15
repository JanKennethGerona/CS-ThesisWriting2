# ASSUMPTIONS.md — Placeholder Registry and Data Provenance

## Purpose

This document records every placeholder constant used in the multi-commodity
Two-Stage MILP implementation, what real data source should eventually replace
each, and which parts of the output are therefore provisional.

**Any result computed with one or more PLACEHOLDER values is a demonstration
output, not a thesis finding.**

---

## Model Description

**Two-stage deterministic MILP consisting of hub activation and resource
allocation decisions.** This is distinct from classical two-stage stochastic
programming — there is no uncertainty realization or recourse. "Two-stage"
refers to the decision structure: Stage 1 activates hubs, Stage 2 allocates
resources from active hubs to barangays.

The model supports equitable resource allocation. It does **not** guarantee
survival — real disaster outcomes also depend on delivery time, road
accessibility, inventory accuracy, disaster severity, population movement,
infrastructure damage, and implementation decisions.

---

## Placeholder Registry

| Constant | Value | Status | Real Source |
|---|---|---|---|
| `PLACEHOLDER_LAMBDA_WATER` | `2.0` | 🔴 Provisional | CDRRMO "time-to-criticality" rescue logs — water deprivation escalation rate |
| `PLACEHOLDER_LAMBDA_FOOD` | `0.5` | 🔴 Provisional | CDRRMO rescue logs — food/caloric deprivation escalation rate |
| `PLACEHOLDER_HUB_COUNT` | `6` | 🔴 Provisional | CDRRMO facility inventory — actual number of designated relief hubs |
| `PLACEHOLDER_HMAX` | `4` | 🔴 Provisional | CDRRMO personnel/logistics capacity — max simultaneous hub activations |
| `PLACEHOLDER_SUPPLY_COVERAGE_RATIO` | `0.5` | 🔴 Provisional | CDRRMO stock reports — actual warehouse inventory levels |
| `Vulnerability Weight (P_j)` | Per-barangay (xlsx) | 🟠 Synthetic | MGB flood/landslide susceptibility maps (GovPH.pdf) — requires manual visual read-off or GIS overlay |
| `Vulnerability Indicator` | Integer 1–5 (xlsx) | 🟠 Synthetic | Same as P_j — the indicator is a uniform-random-looking assignment, not a measured hazard value |
| Hub locations | Top-N population barangays | 🔴 Synthetic | Actual CDRRMO/DSWD designated hub locations and capacities |

### Notes on λ values

`λ_water = 2.0` and `λ_food = 0.5` are **provisional sensitivity-analysis
values, not calibrated magnitudes.** The ordering (`λ_water > λ_food`) reflects
a plausible humanitarian assumption: acute water deprivation escalates faster
than caloric deficit in short-term crises (dehydration can be life-threatening
within 3 days; caloric deficit has a slower onset). However, the specific
magnitudes are not derived from empirical data and should not be presented as
calibrated parameters.

The sensitivity sweep and λ-comparison experiments are designed to test whether
the relative magnitude of `λ_water` vs `λ_food` materially affects the optimal
allocation — answering the question: *How sensitive is the allocation to the
relative urgency of water versus food deprivation?*

### Notes on P_j

The `Vulnerability Weight (P_j)` column in the xlsx has real-looking numbers
(per-barangay values computed as `Indicator × 0.0018484288`), but:

1. The companion `Vulnerability Data Source` column is uniformly
   `"PLACEHOLDER - FOR DEVELOPMENT"` for every row.
2. The Indicator distribution (38/36/35/39/34 across values 1–5) is
   near-uniform-random, not reflective of real flood/landslide clustering.

**`P_j` is defined as a vulnerability priority weight, not population.** Population
is already embedded in `D_{jr} = Population × Sphere_Standard`. Using P_j as
population would double-count it. The separation is:
- **Demand = population** (how much each barangay needs)
- **Priority = vulnerability** (how urgently that need should be met)

---

## Real Data Available

| Parameter | Status | Source |
|---|---|---|
| Population per barangay | ✅ Real | PSA 2024 POPCEN (182 barangays) |
| Sphere water standard | ✅ Real | Sphere Association: 15 L/person/day |
| Sphere food standard | ✅ Real | Sphere Association: 2,100 kcal/person/day |
| Administrative districts | ✅ Real | PSA/PSGC (11 districts, verified) |
| `ε_min` (anti-starvation) | 🟢 Policy choice | 10% — thesis-stated policy decision |
| `ε_env` (disparity tolerance) | 🟢 Policy choice | 15% — thesis-stated policy decision |

---

## Data Readiness Gate

The pipeline enforces two modes:

- **DEVELOPMENT mode** (`is_provisional = True`): Allows placeholder data.
  All output carries `⚠️ DEMONSTRATION ONLY` banners. Suitable for software
  testing and methodology validation.

- **RESEARCH mode** (`is_provisional = False`): Rejects placeholder data.
  Requires all parameters to be from validated sources. Output carries
  `VALIDATED RESEARCH RUN` status. Required before any result can be
  presented as a thesis finding.

---

## Scope Boundaries

### What this implementation does

- Multi-commodity (water + food) optimization with limiting-resource fulfillment
- Single live optimization objective (deprivation-cost minimization) with
  post-hoc fairness evaluation
- Per-resource deprivation escalation via separate `λ_water`, `λ_food`
- Post-processed `φ_j = min_r(φ_{jr})` — NOT an in-model variable
- Baseline and disaster-scenario comparisons
- Tangent approximation accuracy experiment
- Strict data validation gate

### What it does NOT do (future work)

- **Shared transportation capacity** (`Σ_r v_r x_{ijr} ≤ C_{ij}`): allocating
  food and water mathematically is stage two; routing the physical trucks that
  carry both is stage three. Deferred to avoid turning the project into a
  combined location-allocation / capacitated transportation / vehicle routing /
  multi-commodity flow problem.

- **Truck routing / vehicle assignment** (`y_{ijt} ∈ {0,1}`): a third-stage
  extension beyond the thesis scope.

- **Localized `λ_j` per barangay**: the formulation defines λ as a global
  (per-resource) parameter. Per-barangay λ_j would change the formulation and
  require extracting per-barangay hazard zones from the image-based MGB maps
  — the same GIS/vision wall as real P_j.

- **Real P_j from MGB maps**: `GovPH.pdf` contains image-based flood/landslide
  susceptibility maps (12 map images). Extracting per-barangay values requires
  either manual visual reading of 182 barangay locations against colored
  susceptibility zones, or a proper GIS overlay with georeferenced map data.
  Neither is a simple programmatic parse.

- **Shelter or waste resources**: `Sphere_Standard.txt` lists standards for
  shelter (3.5 m²/person) and waste management (100 L/10 households/day),
  but these are not yet in the xlsx and have different operational constraints.

- **Classical two-stage stochastic programming with recourse**: the "two-stage"
  structure here refers to hub activation followed by allocation, not
  uncertainty realization with recourse decisions.

---

## Solver Notes

The auto-detection fallback chain is:
`appsi_highs → highs → cbc → glpk → gurobi → cplex`

Different solvers may give slightly different numerical tie-breaking on
degenerate LPs. If reporting solve times in the thesis methodology, note
which solver was used — HiGHS and CPLEX can produce different degenerate
solutions that are equally optimal.

---

## Version History

- **v6** (current): Multi-commodity (water + food), φ_j post-solve fix,
  data readiness gate, baseline comparisons, disaster scenarios, corrected
  fairness terminology, tangent approximation experiment.
- **v5**: Multi-commodity extension proposal.
- **v4**: Column-index bug fix, 21 breakpoints, localized λ_j deferred.
- **v1–v3**: Single-commodity (water only).
