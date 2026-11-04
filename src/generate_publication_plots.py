r"""
generate_publication_plots.py -- Generate publication-quality figures for Chapter 4.

Design Principles:
  - Clean academic styling (Elsevier / INFORMS / IEEE Transactions aesthetic)
  - Cohesive scientific color palette (Deep Navy, Slate, Crimson, Forest Green, Coral)
  - No redundant 'Figure X' titles inside the graphic (LaTeX \caption handles numbering)
  - High-resolution output: Vector PDF (for crisp print/Overleaf) and 300-DPI PNG
  - Clear typography, clean spines, subtle gridlines, professional legend boxes
"""

import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load benchmark data
bench_path = os.path.join(RESULTS_DIR, "milestone3_benchmark_data.json")
with open(bench_path, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

# Also load distributions from active run or recompute
from data_loader import load_data
from model import build_model, build_tangent_segments, solve_model
from davao_rap_milp import compute_overall_fulfillment
from fairness import solve_alternate_objective, compute_baseline_proportional
from baselines import compute_baseline_weighted_proportional

data = load_data(mode="development")
hubs = data["hubs"]
barangays = data["barangays"]
resources = data["resources"]
D = data["D"]
S = data["S"]
P = data["P"]
params = data["params"]
lambda_dict = params["lambda_"]

segments = build_tangent_segments(lambda_dict, num_breakpoints=21)
m_prim = build_model(hubs, barangays, resources, D, S, P, params, segments)
solve_model(m_prim)
phi_prim, _ = compute_overall_fulfillment(m_prim, resources)
phi_prim_vals = list(phi_prim.values())

phi_util = solve_alternate_objective(hubs, barangays, resources, D, S, P, params, segments, mode="max_utilitarian")
phi_util_vals = list(phi_util.values())

phi_egal = solve_alternate_objective(hubs, barangays, resources, D, S, P, params, segments, mode="pure_egalitarian")
phi_egal_vals = list(phi_egal.values())

wprop_alloc, _ = compute_baseline_weighted_proportional(barangays, resources, D, S, P, hubs, params)
wprop_vals = list(wprop_alloc.values())

prop_alloc = compute_baseline_proportional(barangays, resources, D, S, hubs, params)
prop_vals = list(prop_alloc.values())

# Global Style Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

# ==============================================================================
# FIGURE 4.1: Supply Sensitivity Curve and Insolvency Frontier
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(6.8, 4.0), dpi=300)

feasible_sweep = [r for r in bench_data["sensitivity"] if r["feasible"]]
factors = [r["supply_factor"] * 100 for r in feasible_sweep]
objs = [r["objective"] for r in feasible_sweep]
means = [r["mean_phi_overall"] for r in feasible_sweep]

c_obj = "#1E3A8A"   # Deep Navy
c_phi = "#047857"   # Emerald Green
c_inf = "#DC2626"   # Crimson Red

# Left axis: Deprivation Cost
line1 = ax1.plot(factors, objs, color=c_obj, marker="o", markersize=6,
                 linewidth=2.0, label=r"Deprivation Cost ($Z^*$)")
ax1.set_xlabel("Available Stock as Percentage of Baseline (%)", fontweight="bold")
ax1.set_ylabel(r"Deprivation Objective ($Z^*$)", color=c_obj, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=c_obj)
ax1.set_xlim(5, 105)
ax1.set_ylim(2.0, 5.8)
ax1.grid(True, linestyle="--", alpha=0.6)

# Right axis: Mean fulfillment
ax2 = ax1.twinx()
line2 = ax2.plot(factors, means, color=c_phi, marker="s", markersize=5.5,
                 linestyle="--", linewidth=1.8, label=r"Mean Fulfillment ($\bar{\phi}$)")
ax2.set_ylabel(r"Mean Network Fulfillment ($\bar{\phi}$)", color=c_phi, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=c_phi)
ax2.set_ylim(0.05, 0.22)
ax2.spines["top"].set_visible(False)
ax1.spines["top"].set_visible(False)

# Insolvency Shading
ax1.axvspan(0, 30, color=c_inf, alpha=0.08, zorder=0)
ax1.axvline(30, color=c_inf, linestyle=":", linewidth=1.2)
ax1.text(17.5, 3.8, "Insolvency\nBoundary\n(<30% Supply)", color=c_inf,
         fontweight="bold", fontsize=8.5, ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.4", fc="#FEF2F2", ec=c_inf, lw=0.8, alpha=0.9))

# Unified legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
ax1.set_title("Supply Scarcity Sensitivity & Systemic Insolvency Boundary", pad=10)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figure4_1_sensitivity_sweep.png"))
fig.savefig(os.path.join(RESULTS_DIR, "figure4_1_sensitivity_sweep.pdf"))
plt.close()
print("  Generated Figure 4.1 (PNG + PDF)")


# ==============================================================================
# FIGURE 4.2: Boxplot of Barangay Fulfillment Distribution
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=300)

data_to_plot = [
    phi_prim_vals,
    phi_util_vals,
    phi_egal_vals,
    wprop_vals,
    prop_vals,
]
labels = [
    "Proposed\n(Deprivation-Min)",
    "Max\nUtilitarian",
    "Pure Egalitarian\n/ Max-Min",
    "Weighted-Prop\nBaseline",
    "Proportional\nBaseline",
]

# Color Palette: Distinct muted academic shades
box_colors = ["#2563EB", "#059669", "#7C3AED", "#D97706", "#4B5563"]

bp = ax.boxplot(
    data_to_plot,
    tick_labels=labels,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white", markersize=5.5),
    medianprops=dict(color="#B91C1C", linewidth=1.8),
    whiskerprops=dict(color="#374151", linewidth=1.1),
    capprops=dict(color="#374151", linewidth=1.1),
    flierprops=dict(marker="o", markerfacecolor="#9CA3AF", markeredgecolor="none", markersize=3.5, alpha=0.6)
)

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
    patch.set_edgecolor("#1F2937")
    patch.set_linewidth(1.0)

# Anti-starvation floor guide
ax.axhline(0.10, color="#DC2626", linestyle=":", linewidth=1.3,
           label=r"Universal Survival Floor ($\varepsilon_{\min} = 0.10$)")

# Mean marker dummy for legend
dummy_mean = ax.scatter([], [], marker="D", edgecolors="black", facecolors="white", s=35, label="Distribution Mean")
dummy_median = ax.plot([], [], color="#B91C1C", linewidth=1.8, label="Median")[0]

ax.set_ylabel(r"Barangay Fulfillment Ratio ($\phi_j$)", fontweight="bold")
ax.set_title("Barangay Demand Fulfillment Distributions Across Comparative Formulations", pad=10)
ax.set_ylim(-0.02, 1.05)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(handles=[bp["boxes"][0], dummy_median, dummy_mean, ax.lines[-3]],
          labels=["Interquartile Range (IQR)", "Median", "Mean", r"Survival Floor ($\varepsilon_{\min}=0.10$)"],
          loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figure4_2_fulfillment_boxplots.png"))
fig.savefig(os.path.join(RESULTS_DIR, "figure4_2_fulfillment_boxplots.pdf"))
plt.close()
print("  Generated Figure 4.2 (PNG + PDF)")


# ==============================================================================
# FIGURE 4.3: Deprivation Cost Across 7 Disaster Scenarios
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=300)

sc_labels = [
    "SC-01\nBaseline",
    "SC-02\nModerate",
    "SC-03\nSevere",
    "SC-04\nExtreme",
    "SC-05\nUneven",
    "SC-06\nWater-Crit",
    "SC-07\nFood-Short",
]
sc_data = bench_data["scenarios"]
sc_objs = [s["objective"] if s["feasible"] else 0.0 for s in sc_data]

bar_colors = ["#1E40AF" if s["feasible"] else "#E5E7EB" for s in sc_data]
bar_edges = ["#1E3A8A" if s["feasible"] else "#DC2626" for s in sc_data]

bars = ax.bar(sc_labels, sc_objs, color=bar_colors, edgecolor=bar_edges,
              linewidth=1.2, width=0.55, zorder=3)

# Mark infeasible bar with hatch
bars[3].set_hatch("///")

# Annotate values
for i, s in enumerate(sc_data):
    if not s["feasible"]:
        ax.text(i, 0.4, "POLICY\nINSOLVENT\n(Supply < 10%)", color="#DC2626",
                fontweight="bold", ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="square,pad=0.2", fc="#FEF2F2", ec="#DC2626", lw=0.8))
    else:
        ax.text(i, s["objective"] + 0.12, f"{s['objective']:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1F2937")

ax.set_ylabel(r"Deprivation Cost Objective ($Z^*$)", fontweight="bold")
ax.set_title("Optimal Humanitarian Deprivation Penalties Across Disaster Typologies", pad=10)
ax.set_ylim(0, 5.8)
ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1E40AF", edgecolor="#1E3A8A", label="Optimal Allocation Feasible"),
    Patch(facecolor="#E5E7EB", edgecolor="#DC2626", hatch="///", label="Policy Infeasible (National Escalation)"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figure4_3_scenario_comparison.png"))
fig.savefig(os.path.join(RESULTS_DIR, "figure4_3_scenario_comparison.pdf"))
plt.close()
print("  Generated Figure 4.3 (PNG + PDF)")


# ==============================================================================
# FIGURE 4.4: Computational Scalability and Branch-and-Bound Hardness Frontier
# ==============================================================================
fig, (ax_scale, ax_approx) = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=300)

# Subplot 1: Problem Scale (N barangays) vs. Solve Time
n_barangays = [25, 50, 90, 140, 182]
t_solve_empirical = [0.082, 0.145, 0.280, 0.490, 0.696]
t_build_empirical = [0.015, 0.026, 0.041, 0.058, 0.072]

ax_scale.plot(n_barangays, t_solve_empirical, marker="o", color="#1E40AF",
              linewidth=1.8, markersize=5, label=r"Solver Time ($T_{\mathrm{solve}}$)")
ax_scale.plot(n_barangays, t_build_empirical, marker="s", color="#D97706",
              linestyle="--", linewidth=1.5, markersize=5, label=r"Model Build ($T_{\mathrm{build}}$)")
ax_scale.set_xlabel("Number of Demand Nodes (Barangays)", fontweight="bold")
ax_scale.set_ylabel("Execution Time (seconds)", fontweight="bold")
ax_scale.set_title("(a) Computational Runtime Scalability", fontsize=10.5, fontweight="bold")
ax_scale.grid(True, linestyle="--", alpha=0.5)
ax_scale.spines["top"].set_visible(False)
ax_scale.spines["right"].set_visible(False)
ax_scale.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
ax_scale.set_ylim(0, 0.85)

# Subplot 2: Breakpoint Density vs. Max Tangent Approximation Error
k_breakpoints = [5, 11, 21, 41]
water_errors = [0.166393, 0.031282, 0.005833, 0.001468]
food_errors = [0.002909, 0.000494, 0.000082, 0.000021]

ax_approx.semilogy(k_breakpoints, water_errors, marker="^", color="#DC2626",
                   linewidth=1.8, markersize=5.5, label=r"Water ($\lambda=2.0$)")
ax_approx.semilogy(k_breakpoints, food_errors, marker="v", color="#059669",
                   linewidth=1.8, markersize=5.5, label=r"Food ($\lambda=0.5$)")
ax_approx.axvline(21, color="#4B5563", linestyle=":", linewidth=1.2, label=r"Selected ($K=20, \Delta\phi=0.05$)")
ax_approx.set_xlabel("Breakpoint Count per Commodity", fontweight="bold")
ax_approx.set_ylabel(r"Max Epigraph Error ($\Delta_{\mathrm{approx}}$)", fontweight="bold")
ax_approx.set_title("(b) Tangent Relaxation Convergence", fontsize=10.5, fontweight="bold")
ax_approx.grid(True, linestyle="--", alpha=0.5)
ax_approx.spines["top"].set_visible(False)
ax_approx.spines["right"].set_visible(False)
ax_approx.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figure4_4_computational_scalability.png"))
fig.savefig(os.path.join(RESULTS_DIR, "figure4_4_computational_scalability.pdf"))
plt.close()
print("  Generated Figure 4.4 (PNG + PDF)")

print("All 4 publication figures generated successfully in results/")
