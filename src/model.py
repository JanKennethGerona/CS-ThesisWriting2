"""
model.py — Re-exports from davao_rap_milp.py with solution verification.

Provides a clean import surface for the pipeline:
    from model import build_model, solve_model, verify_and_report
"""

from davao_rap_milp import (
    build_model,
    build_tangent_segments,
    solve_model,
    compute_overall_fulfillment,
    build_sample_data,
    build_sample_data_multi,
)
from validation import (
    validate_solution,
    validate_tangent_approximation,
    compare_breakpoint_counts,
    print_validation_results,
)
import pyomo.environ as pyo


# -- Function: Verify solution and report validation results --
def verify_and_report(model, lambda_dict, resources, tol=1e-6):
    """
    Full post-solve verification: compute φ_j, validate, print results.

    Returns:
        phi_j: {j: overall_fulfillment}
        limiting_resource: {j: resource_name}
        validation_results: dict
    """
    # Compute overall fulfillment (post-solve, NOT in-model)
    phi_j, limiting_resource = compute_overall_fulfillment(model, resources)

    # Validate
    val_results = validate_solution(
        model, lambda_dict, resources, phi_j, tol=tol
    )

    # Print
    print_validation_results(val_results)

    return phi_j, limiting_resource, val_results
