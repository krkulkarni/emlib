from typing import Protocol, List, Dict, Tuple, Callable, NamedTuple
import numpy as np
import pandas as pd

# --- Data Structures ---


class EStepResult(NamedTuple):
    """Holds the results of the E-step for a single subject."""

    pid_index: int  # Original index for re-sorting
    map_estimate: np.ndarray  # MAP estimate in *transformed* space
    hessian_inv: np.ndarray  # Approx covariance in *transformed* space
    neg_log_posterior: float  # Value of the objective function at MAP


class FitResult(NamedTuple):
    """Holds the final results of the EM fitting procedure."""

    group_params: Dict[str, Dict[str, float]]  # Final group mu & var per param
    individual_params: List[EStepResult]  # List of EStepResult for each subject
    bic_int: float  # Integrated BIC score
    convergence_iterations: int  # Number of iterations run
    converged: bool  # Did the algorithm converge?
    final_objective: float  # Final value of EM objective proxy


# --- Model Interface Definition ---


class ModelLikelihoodInfo(NamedTuple):
    """Information returned by a model required by the EM algorithm."""

    param_names: List[str]  # List of parameter names in order
    transform_funcs: Dict[str, Dict[str, Callable]]  # {'param': {'forward': f, 'inverse': inv_f}}
    log_likelihood_func: Callable[[np.ndarray, pd.DataFrame], float]  # Takes transformed_params, data -> loglik


class ModelProtocol(Protocol):
    """Defines the interface required for any model used with the EM fitter."""

    def get_likelihood_info(self) -> ModelLikelihoodInfo:
        """
        Returns information needed to calculate likelihood and handle parameters.
        """
        ...

    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Optional: Returns reasonable bounds for parameters in their *native* space.
        Can be used for optimizer constraints if not transforming, or for sanity checks.
        Return None or empty dict if not applicable.
        Example: {'alpha': (0, 1), 'beta': (0, None)} # None means no upper/lower bound
        """
        # Default implementation if not needed by a specific model
        return {}
