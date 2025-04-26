# pyem/bic.py
import numpy as np
from typing import Dict
# from .model_interface import FitResult  # Import FitResult for type hinting if needed


def calculate_bic_int(
    log_marginal_likelihood_approx: float,
    total_trials: int,
    group_theta: Dict[str, Dict[str, float]],
) -> float:
    """
    Calculates the Integrated Bayesian Information Criterion (BIC_int).

    Args:
        log_marginal_likelihood_approx: An approximation of the log marginal
                                         likelihood log p(A|theta_ML). Often approximated
                                         by the sum of log posteriors at the MAP estimates
                                         under the final group parameters.
        total_trials: The total number of trials across all subjects.
        group_theta: The final estimated group hyperparameters. Used to determine
                     the number of free group parameters.

    Returns:
        The BIC_int score. Returns np.inf if inputs are invalid.
    """
    if not np.isfinite(log_marginal_likelihood_approx):
        print("Warning: Cannot calculate BIC_int with non-finite log marginal likelihood approximation.")
        return np.inf
    if total_trials <= 0:
        print("Warning: Cannot calculate BIC_int with zero or negative total trials.")
        return np.inf
    if not group_theta:
        print("Warning: Cannot calculate BIC_int without group_theta to determine parameter count.")
        return np.inf

    # Determine number of free group hyperparameters (mu and var for each param)
    num_hyperparams = len(group_theta) * 2

    if num_hyperparams <= 0:
        print("Warning: Cannot calculate BIC_int with zero hyperparameters.")
        return np.inf

    # BIC_int formula
    bic = -2 * log_marginal_likelihood_approx + num_hyperparams * np.log(total_trials)

    if not np.isfinite(bic):
        print("Warning: BIC_int calculation resulted in non-finite value.")
        return np.inf

    return bic


# Example of how it might be called (usually within fit_em_hierarchical)
# bic_value = calculate_bic_int(log_marginal_likelihood_approx,
#                               num_subjects,
#                               total_trials,
#                               final_group_theta)
