# AI Implementation Brief
## Two-Stage MILP for Disaster Relief Resource Allocation — Davao City

**Purpose of this document:** you (the AI receiving this) are being asked to implement the computational engine for an undergraduate thesis: *"Exact Optimization of City-Wide Disaster Relief: A Two-Stage MILP for Resource Allocation in Davao City"* (Misa, Labor & Gerona, Ateneo de Davao University, 2026). This brief is self-contained — it does not assume you have read the thesis or any prior conversation. Follow it literally; where the thesis is ambiguous, this brief tells you explicitly what to assume and what to flag instead of guessing.

**Before doing anything else, re-read Section 3 status table against the actual attached files** — this brief was written at a point in time; if a column that's marked empty now has data in it, or a value marked "not available" now exists somewhere in the attached files, trust the file over this document and note the discrepancy in your output.

**Files that should be attached alongside this brief:**
1. `Davao_City_Barangays_PSA_PSGC_2026_with_Model_Columns.xlsx` — barangay-level data (full schema in Section 3a).
2. `Sphere_Standard.txt` — per-capita relief minimums (full content in Section 3a).
3. `GovPH.pdf` — MGB flood/landslide susceptibility maps (image-based; see Section 3 and 3a on limitations).
4. `davao_rap_milp.py` and `load_real_data.py` — draft Pyomo implementation and data loader already written and hand-verified for logic (not solver-tested — no internet/solver was available when they were written). **Read both files in full before writing anything new.** Treat them as a verified-correct starting scaffold: validate against Section 2 below line by line, then extend rather than rewrite from scratch, unless you find a genuine bug. **Known stale piece:** `load_real_data.py` currently hardcodes a uniform `PLACEHOLDER_VULNERABILITY_WEIGHT = 1.0` for every barangay, because at the time it was written the xlsx had no vulnerability data at all. That's no longer accurate — the xlsx now has a real (if placeholder) `Vulnerability Weight (P_j)` column with per-barangay variation (see Section 3). Update the loader to read that column directly instead of using the hardcoded constant.

---

## 0. Environment & Tooling

The thesis itself specifies the tooling — use it, don't substitute:

```
pip install pyomo pandas openpyxl
```

Plus one MILP solver:
- **CBC** (free, fine for iteration/testing): `conda install -c conda-forge coincbc`, or on Debian/Ubuntu `apt-get install coinor-cbc`, or macOS `brew install cbc`.
- **Gurobi or CPLEX** — the thesis names these as the target production solvers (Section 3.6 of the thesis). Use one of these for any run whose results are meant to go in the final paper, if a license is available; CBC is acceptable for development/testing.

Python 3.9+ is assumed; no other version constraints are known.

---

## 1. The Non-Negotiable Rule

**Never fabricate a value for a parameter this brief marks "NOT AVAILABLE," and never present a result computed from a placeholder as if it were a real finding.** Several parameters (listed in Section 3) genuinely do not exist yet — they depend on a data request to Davao City's CDRRMO that has not been completed. It is correct and expected to build and test the full model using clearly-labeled placeholder values for these — that is real, useful engineering work. It is not acceptable to invent a "realistic-sounding" number and use it without a loud label, or to report any output derived from a placeholder as a thesis result. Every placeholder must be a named constant (e.g., `PLACEHOLDER_LAMBDA`) with a comment explaining what real source should eventually replace it.

---

## 2. Exact Mathematical Specification

This is copied directly from the thesis (Section 3.4) — implement it exactly as stated, not a reinterpretation.

### 2.1 Sets
- `I`: supply hubs / consolidated inventory pools, indexed by `i`
- `J`: demand points (barangays), indexed by `j`
- `K`: piecewise-linear tangent segments (your own construction, see 2.4), indexed by `k`

### 2.2 Parameters
- `D_j`: baseline demand for the resource at node `j`
- `S_i`: maximum available inventory at hub `i`
- `P_j`: population/vulnerability weight of node `j`
- `λ` (lambda): deprivation escalation rate
- `Hmax`: max number of hubs simultaneously activatable
- `ε_min`: minimum fulfillment threshold (anti-starvation floor), e.g. 0.10
- `ε`: envy-freeness tolerance (post-hoc evaluation only, not a live constraint)

### 2.3 Decision Variables
- `y_i ∈ {0,1}` — Stage 1: hub activation
- `x_ij ≥ 0` — Stage 2: quantity allocated from hub `i` to node `j`
- `w_j ≥ 0` — total quantity received at node `j`
- `φ_j ∈ [0,1]` — fulfillment ratio at node `j`
- `z_j ≥ 0` — linearized deprivation penalty at node `j`
- `Φ_min ≥ 0` — network-wide minimum fulfillment ratio (used only in post-hoc Max-Min evaluation, Section 2.6)

### 2.4 Objective Function — the only thing actually optimized
Minimize total population-weighted deprivation:

```
Minimize Z = Σ_j  P_j · z_j
```

The "true" deprivation cost per node is `e^(λ·(1-φ_j))`, which is convex and nonlinear (would make this a MINLP). Approximate it with a **piecewise-linear tangent under-approximation**: pick breakpoints (the thesis suggests φ = 0, 0.25, 0.5, 0.75, 1.0), and at each breakpoint `φ0`, compute the tangent line to `f(u) = e^(λu)` where `u = 1-φ`:

```
slope m_k = λ · e^(λ·u0)
intercept b_k = e^(λ·u0) − m_k·u0        (where u0 = 1 − φ0)
```

Add one constraint per node per segment:

```
z_j ≥ m_k·(1 − φ_j) + b_k     for all j, for all k
```

Because the objective minimizes `z_j`, the solver pushes each `z_j` down to the *highest* active tangent line at that `φ_j`. Since tangent lines to a convex function always lie *below* the function except exactly at the point of tangency, this makes `z_j` a valid **under-approximation** of the true curve — exact at the chosen breakpoints, and strictly less than the true value everywhere else. (This matches the thesis's own wording: the segments "under-approximate the convex exponential curve.") Verify this behaves correctly by hand-checking a few `(φ_j, z_j)` pairs against `e^(λ(1-φ_j))` after solving — `z_j` should be very close to the true value and never exceed it. If you find `z_j` coming out *above* the true curve anywhere, something in the tangent construction is wrong.

### 2.5 Operational Constraints
```
Σ_i y_i ≤ Hmax                                    (hub activation cap)
Σ_j x_ij ≤ S_i · y_i        for all i             (capacity, linked to activation)
w_j = Σ_i x_ij              for all j             (total allocation definition)
w_j = φ_j · D_j              for all j             (fulfillment ratio — note: this is
                                                     multiplication by a PARAMETER D_j,
                                                     not two variables, so it stays linear)
φ_j ≤ 1                     for all j             (no hoarding — enforce via variable
                                                     bound, not a separate constraint)
x_ij ≥ 0                    for all i, j          (non-negativity)
φ_j ≥ ε_min                 for all j             (anti-starvation floor)
```

### 2.6 Post-Hoc Fairness Metrics — NOT simultaneous constraints
**Critical scope limitation, stated explicitly in the thesis (Section 1.5):** the model has exactly one live objective (2.4). The following are evaluated *after* solving, or via *separate* re-solves with a different objective — never added as constraints alongside the deprivation objective:

- **Pareto-efficiency**: a structural property of the optimal solution — assert it, don't separately verify unless you want a proof-of-concept check that no `φ_j` can be raised without lowering another under the current `S_i`.
- **Max Utilitarian** (separate solve): `Maximize Σ_j φ_j`, subject to the same operational constraints (2.5) minus the deprivation objective.
- **Pure Egalitarian** (separate solve): add `φ_j = φ_k` for all pairs, then `Maximize Σ_j φ_j`.
- **Max-Min Fair Share** (separate solve): `Maximize Φ_min` subject to `φ_j ≥ Φ_min` for all `j`.
- **Max Leximin**: computed by sorting the *primary solution's* `φ_j` vector ascending — no re-solve needed, just report the sorted vector.
- **Max Nash**: computed as `Σ_j ln(φ_j)` from the primary solution (report only — the thesis's formal Nash objective, `Maximize Σ ln(φ_j)`, is a separate optional re-solve if you want it).
- **Envy-freeness**: pairwise check `|φ_j − φ_k| ≤ ε` on the primary solution; report all violating pairs.

---

## 3. Data Status (as of the brief's writing — re-verify before trusting, per the note at the top)

| Parameter | Status | Detail |
|---|---|---|
| `D_j` | ✅ **Real, complete** | Resource modeled = **water**. `Demand (D_j)` = `Population × 15` (L/person/day), verified with zero mismatches across all 182 barangays. City-wide total ≈ 27,734,205 L/day. |
| Barangay → administrative district | ✅ **Real, complete, independently verified** | `District` column holds the 11 administrative districts (Poblacion, Talomo, Agdao, Buhangin, Bunawan, Paquibato, Baguio, Calinan, Marilog, Toril, Tugbok) that the MGB hazard maps in `GovPH.pdf` are organized by. Cross-checked against an external source (Wikipedia's *Districts of Davao City*, citing NSCB/Davao City Directory) — all 182 rows matched exactly. **Do not confuse this with Davao's 3 congressional districts** (District I/II/III) — a different, incompatible grouping that cannot tell you which of the 11 MGB maps applies to a barangay. |
| `P_j` (vulnerability weight) | 🟠 **Filled, but explicitly PLACEHOLDER (182/182)** | All three vulnerability columns are now populated for development purposes: `Vulnerability Indicator` holds an integer 1–5 per barangay, `Vulnerability Weight (P_j)` = `Indicator × 0.0018484288` exactly (a straight linear rescaling of the indicator, not a real hazard-derived value — verified across all 182 rows), and `Vulnerability Data Source` literally reads `"PLACEHOLDER - FOR DEVELOPMENT"` for every row. The Indicator distribution (38/36/35/39/34 across the 5 values) looks close to uniform-random rather than reflecting real flood/landslide clustering — treat it as synthetic. **Use it as-is for building and testing the pipeline; never as a real vulnerability signal.** The real version of this column still needs the MGB-map read-off described in Section 3a/Task 1, using the now-solved district lookup, before any result is reportable as a thesis finding. |
| `S_i` (hub capacity) | 🔴 **Not available** | No hub location list exists at all. Needs a real CDRRMO request. Build with a placeholder, flagged everywhere. |
| `Hmax` | 🔴 **Not available** | CDRRMO personnel-capacity fact. Placeholder-only. |
| `λ` | 🔴 **Not available** | Needs CDRRMO "time-to-criticality" rescue-log data. Placeholder fine for testing mechanics; never for reporting a deprivation-cost result. |
| `ε_min`, `ε` | 🟢 **Policy choice, fixable now** | Thesis-stated policy decisions, not empirical facts (`ε_min`=0.10 is the thesis's own example). Fix and document. |

---

## 3a. Full Data Schema Reference

**`Davao_City_Barangays_PSA_PSGC_2026_with_Model_Columns.xlsx`**
- Sheet `Davao City Barangays`: 182 data rows (+ 1 header row), columns in order:
  1. `Barangay` (text)
  2. `District` (text — one of the 11 administrative districts; see Section 3)
  3. `10 Digit Code` (text, PSGC code)
  4. `Correspondence Code` (text, alternate/legacy code)
  5. `Urban/Rural` (text: "Urban" or "Rural")
  6. `Population (2024 POPCEN)` (integer)
  7. `Sphere Minimum per Person` (numeric — currently 15 for all rows, the water standard)
  8. `Demand (D_j)` (numeric — `Population × Sphere Minimum per Person`; this is a static computed value, not a live formula, so it won't auto-update if population is edited — recompute in your own code rather than trusting the cell if you change any input)
  9. `Vulnerability Indicator` (integer 1–5, filled for all 182 rows — **placeholder**, see Section 3)
  10. `Vulnerability Weight (P_j)` (numeric, filled for all 182 rows — **placeholder**, `= Indicator × 0.0018484288` exactly)
  11. `Household Count` (integer, filled for all 182 rows — provenance unclear: ratios of Population/Household Count range ~3.8–4.3 across sampled rows, which is plausible for real Philippine household-size data, but it's not confirmed whether this is genuine PSA data or another synthetic fill. Currently irrelevant either way — the water resource doesn't use it; only the waste multiplier would.)
  12. `Vulnerability Data Source` (text, filled for all 182 rows with the literal string `"PLACEHOLDER - FOR DEVELOPMENT"` — use this as a reliable, self-documenting flag: check for the substring `"PLACEHOLDER"` at load time rather than assuming any particular status)
- Sheet `Source`: documentation/metadata only, not model data — do not parse this as input. It may contain stale notes from earlier revisions of this file (e.g., about columns that were temporarily removed and later re-added); don't treat anything in this sheet as authoritative over the actual column headers and data in the main sheet.

**`Sphere_Standard.txt`** — plain text, four candidate multipliers:
```
Water Supply (WASH): 15 L / person / day
Food & Nutrition: 2,100 kcal / person / day
Shelter & Settlement: 3.5 m² / person
Solid Waste Management: 100 L / 10 households / day
```
Only the water figure is currently wired into the xlsx (see above). If the modeled resource ever changes, the others are available but nothing currently uses them.

**`GovPH.pdf`** — 12 map images total: 1 city-and-region flood-susceptibility overview, plus 11 detailed per-district maps (one per administrative district; note the Agdao map appears to be duplicated in the source PDF — harmless, just don't double-count it). Each district map shows flood susceptibility (Very High/High/Moderate/Low) and landslide susceptibility (same 4 levels) as colored zones, plus barangay center-point markers. This is image data, not a structured table — extracting per-barangay values requires either visual reading (if you have vision capability) or a proper GIS overlay.

---

## 4. Required Implementation Tasks, in Order

1. **Data loader.** Read the xlsx directly per the schema in Section 3a. `D_j`, `District`, and `P_j` are all currently populated — load `P_j` directly rather than falling back to a uniform value. **At load time, check the `Vulnerability Data Source` column for the substring `"PLACEHOLDER"` and set an `is_provisional` flag accordingly**; propagate that flag through the pipeline so it can't silently disappear into a final-looking result. Build the loader to read whatever is actually in the columns rather than hardcoding today's placeholder scheme — when real vulnerability data eventually replaces it (via the MGB map read-off in Section 3a), the same loader should keep working unchanged, and the `is_provisional` flag should naturally flip to false once the source string no longer says "PLACEHOLDER". For `S_i`/`Hmax`: construct a placeholder hub set (e.g., the N largest-population barangays as candidate sites, capacity as some fraction of total demand) — label every constant `PLACEHOLDER_*`, following the same self-labeling discipline already present in the xlsx.

2. **Build the MILP** exactly per Section 2, using Pyomo. Read and extend `davao_rap_milp.py` if attached — validate its constraint set against Section 2.5 line by line before trusting it.

3. **Solve** with CBC for iteration; note Gurobi/CPLEX as the thesis's target production solver (Section 0).

4. **Run post-hoc fairness evaluation** (Section 2.6). Report all metrics together, each clearly labeled with whether it came from the primary solve or a separate re-solve.

5. **Sensitivity sweep**: progressively shrink placeholder `S_i` (e.g., 100%, 80%, 60%, 40%) and confirm `ε_min` remains satisfied until genuine infeasibility. Report that breaking point as a structural finding, re-runnable once real `S_i` exists.

6. **Write an assumptions log** listing every placeholder constant used, what real data source should eventually replace it, and which parts of the output are therefore provisional. This should travel with any results you produce.

---

## 5. Expected Deliverables

Produce a small, runnable project — not just inline code fragments:

- `data_loader.py` — reads the xlsx (and, if attempted, the MGB maps) into the parameter structures the model needs.
- `model.py` — the Pyomo model builder (Sections 2.1–2.5) and solver call.
- `fairness.py` — post-hoc metrics and alternate-objective re-solves (Section 2.6).
- `main.py` — a single entry point that runs the full pipeline end-to-end (load → solve → evaluate → sensitivity sweep → print/export results) given the file paths as arguments or constants at the top.
- `ASSUMPTIONS.md` — the assumptions log from Task 6.
- Results should be exportable in a plain format (CSV or JSON) as well as printed to console — don't lock the output inside notebook cells or ad hoc print statements only.

---

## 6. Acceptance Criteria — how to know the implementation is correct

- Solver returns an optimal (not infeasible) status on the placeholder-plus-real-demand dataset.
- Every `φ_j` in the solution is ≥ `ε_min` and ≤ 1.
- Spot-check 3–5 `(φ_j, z_j)` pairs by hand against `e^(λ(1-φ_j))` — `z_j` should track just *below* the true curve (a valid under-approximation) and should never exceed it; the gap should shrink to zero exactly at the chosen breakpoints.
- The three post-hoc/alternate-objective re-solves (Max Utilitarian, Pure Egalitarian, Max-Min) each return a feasible, different allocation from the primary solve — if identical, something is likely wired wrong.
- The sensitivity sweep shows monotonically non-improving fairness/feasibility as supply shrinks, with a clear infeasibility point once `ε_min` can no longer be met network-wide.
- The `is_provisional` flag (Task 1) correctly reflects the current data: true for this dataset's `P_j` (it's placeholder), and any exported result carries this flag visibly rather than presenting numbers unqualified.
- Nothing in the final report or output text asserts a specific number (e.g., "Barangay X should receive Y liters") as a *recommendation* without a visible caveat that it rests on placeholder `S_i`/`Hmax`/`λ`/`P_j`.

---

## 7. If You Get Stuck or Something Is Ambiguous

Do not silently guess and proceed on anything that would materially change a reported result. Specifically stop and ask (or clearly flag in your output) if:
- The vulnerability-weight numeric scale isn't obvious from context and you're about to invent one — state your choice explicitly instead of picking silently.
- You're about to treat a placeholder value as if it were sourced from CDRRMO.
- The xlsx structure or fill-status has changed from what Sections 3/3a describe — re-verify against the actual file rather than assuming this brief is still accurate (per the note at the top of this document).
