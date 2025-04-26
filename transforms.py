import numpy as np
from scipy.special import logit, expit

EPSILON = 1e-7  # Small value to prevent log(0) or division by zero

# --- Individual Transformations ---


def identity(x: float) -> float:
    """Identity transformation (for untransformed parameters)."""
    return x


def transform_alpha(alpha: float) -> float:
    """Logit transform for alpha (0, 1)."""
    alpha_clipped = np.clip(alpha, EPSILON, 1.0 - EPSILON)
    return logit(alpha_clipped)


def inverse_transform_alpha(alpha_prime: float) -> float:
    """Inverse logit transform for alpha."""
    return expit(alpha_prime)


def transform_beta(beta: float) -> float:
    """Log transform for beta > 0."""
    beta_clipped = np.maximum(beta, EPSILON)
    return np.log(beta_clipped)


def inverse_transform_beta(beta_prime: float) -> float:
    """Inverse log transform (exponential) for beta."""
    return np.exp(beta_prime)


# --- Dictionary for Easy Access ---
# Add more parameters and their functions as needed
param_transformations = {
    "alpha": {"forward": transform_alpha, "inverse": inverse_transform_alpha},
    "beta": {"forward": transform_beta, "inverse": inverse_transform_beta},
    "eps": {"forward": transform_alpha, "inverse": inverse_transform_alpha},
    "mod": {"forward": transform_alpha, "inverse": inverse_transform_alpha},
}
