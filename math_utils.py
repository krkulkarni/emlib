import numpy as np

def softmax(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Compute softmax values for each set of scores in x, scaled by beta.
    Handles 1D or 2D input (applies softmax row-wise for 2D).
    Ensures numerical stability.

    Args:
        x: Array of scores (e.g., Q-values). Shape (n_choices,) or (n_trials, n_choices).
        beta: Inverse temperature scaling factor.

    Returns:
        Array of probabilities with the same shape as x.
    """
    if beta < 0:
        beta = 1e-6 # Beta must be non-negative for interpretation

    if x.ndim == 1:
        # Numerically stable softmax for 1D input
        scaled_x = beta * x
        stable_x = scaled_x - np.max(scaled_x)
        probs = np.exp(stable_x) / np.sum(np.exp(stable_x))
    elif x.ndim == 2:
        # Apply row-wise for 2D input
        scaled_x = beta * x
        stable_x = scaled_x - np.max(scaled_x, axis=1, keepdims=True)
        exp_x = np.exp(stable_x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        raise ValueError("Input array must be 1D or 2D for softmax.")

    # Handle potential NaNs resulting from extreme values if necessary
    probs = np.nan_to_num(probs, nan=0.0)
    # Renormalize slightly if probabilities don't sum exactly to 1 due to numerical issues
    if probs.ndim == 1 and not np.isclose(np.sum(probs), 1.0):
         probs /= np.sum(probs)
    elif probs.ndim == 2:
         row_sums = np.sum(probs, axis=1, keepdims=True)
         # Avoid division by zero for rows that were all NaN initially
         safe_sums = np.where(row_sums == 0, 1.0, row_sums)
         probs /= safe_sums

    return probs