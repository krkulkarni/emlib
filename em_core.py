import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import norm
import numdifftools as nd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import time
from copy import deepcopy
import warnings
from typing import List, Dict, Any, Tuple, Callable, NamedTuple, Protocol

# Import from other modules within the pyem package
from .model_interface import ModelProtocol, ModelLikelihoodInfo, EStepResult, FitResult
from .bic import calculate_bic_int

# Define EPSILON or import it if defined elsewhere (e.g., transforms.py)
EPSILON = 1e-7

# --- Log Prior Calculation ---
def log_prior(transformed_params: np.ndarray,
              group_theta: Dict[str, Dict[str, float]],
              param_names: List[str]) -> float:
    """
    Calculates the log prior probability density for a set of transformed parameters.
    (Implementation is the same as before)
    """
    total_log_prior = 0.0
    for i, name in enumerate(param_names):
        if name not in group_theta:
            raise ValueError(f"Group parameters (theta) missing for parameter: {name}")
        mu = group_theta[name]['mu']
        variance = group_theta[name]['var']
        if variance <= 0: return -np.inf
        scale = np.sqrt(variance)
        if scale <= 0 or not np.isfinite(scale): return -np.inf
        try:
            lp = norm.logpdf(transformed_params[i], loc=mu, scale=scale)
            if not np.isfinite(lp): lp = -1e6 # Avoid -inf
            total_log_prior += lp
        except ValueError: return -np.inf
    return total_log_prior

# --- Negative Log Posterior Calculation ---
def negative_log_posterior(transformed_params: np.ndarray,
                           data: Dict[str, Any], # Passed as args to minimize
                           group_theta: Dict[str, Dict[str, float]],
                           model_info: ModelLikelihoodInfo) -> float:
    """
    Calculates the negative log posterior probability (objective function).
    (Implementation is the same as before)
    """
    log_lik = model_info.log_likelihood_func(transformed_params, data)
    log_pri = log_prior(transformed_params, group_theta, model_info.param_names)
    if not np.isfinite(log_lik) or not np.isfinite(log_pri): return np.inf
    return -(log_lik + log_pri)

# --- E-Step for Single Subject ---
def run_e_step_for_subject(pid_index: int,
                           subject_data: Dict[str, Any],
                           group_theta: Dict[str, Dict[str, float]],
                           model_info: ModelLikelihoodInfo,
                           optimizer_options: Dict = None,
                           initial_guess: np.ndarray = None,
                           verbose: bool = False) -> EStepResult:
    """
    Performs the E-step for a single subject.
    (Implementation is the same as before, including warnings)
    """
    if optimizer_options is None:
        optimizer_options = {'method': 'L-BFGS-B', 'options': {'maxiter': 1000}}

    # Pass static args needed by the objective function via 'args' tuple
    obj_args = (subject_data, group_theta, model_info)
    objective_func = lambda p, *args: negative_log_posterior(p, *args) # Ensure correct signature

    if initial_guess is None:
        initial_guess = np.array([group_theta[name]['mu'] for name in model_info.param_names])

    # --- Optimization ---
    try:
        opt_result = minimize(objective_func, initial_guess, args=obj_args, **optimizer_options)
    except Exception as e:
        warnings.warn(f"Optimizer Error for subject index {pid_index}: {e}")
        failed_hess_inv = np.eye(len(model_info.param_names)) * 1e6
        return EStepResult(pid_index=pid_index, map_estimate=initial_guess,
                           hessian_inv=failed_hess_inv, neg_log_posterior=np.inf)

    if not opt_result.success:
        warnings.warn(f"Optimization failed to converge for subject index {pid_index}. Message: {opt_result.message}")

    map_estimate = opt_result.x
    neg_log_post_val = opt_result.fun

    # --- Hessian Calculation ---
    hessian_inv = np.eye(len(map_estimate)) * 1e6 # Default fallback
    try:
        # Recalculate objective function with args for numdifftools
        hessian_calculator = nd.Hessian(lambda p: objective_func(p, *obj_args), step=1e-4, method='central')
        hessian_matrix = hessian_calculator(map_estimate)

        if not np.all(np.isfinite(hessian_matrix)):
             warnings.warn(f"Non-finite values in Hessian for subject index {pid_index}. Using high-variance identity.")
        else:
            ridge = 1e-6
            hessian_matrix += np.eye(len(map_estimate)) * ridge
            try:
                hessian_inv = np.linalg.inv(hessian_matrix)
                hessian_inv[np.diag_indices_from(hessian_inv)] = np.maximum(
                    np.diag(hessian_inv), EPSILON
                 )
            except np.linalg.LinAlgError:
                warnings.warn(f"Hessian inversion failed for subject index {pid_index}. Using high-variance identity.")
    except Exception as e:
        warnings.warn(f"Error during Hessian calculation/inversion for subject index {pid_index}: {e}")

    return EStepResult(pid_index=pid_index, map_estimate=map_estimate,
                       hessian_inv=hessian_inv, neg_log_posterior=neg_log_post_val)


# --- Parallel E-Step Wrapper ---
def run_e_step_parallel(all_subject_data: List[Dict[str, Any]],
                        group_theta: Dict[str, Dict[str, float]],
                        model_info: ModelLikelihoodInfo,
                        optimizer_options: Dict = None,
                        n_jobs: int = -1,
                        verbose: bool = False) -> List[EStepResult]:
    """
    Runs the E-step for all subjects in parallel using joblib.
    (Implementation is the same as before)
    """
    num_subjects = len(all_subject_data)
    # Start message is now handled by the main loop's tqdm bar
    # if verbose: tqdm.write(f"Running E-step for {num_subjects} subjects...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_e_step_for_subject)(
            pid_index=i,
            subject_data=all_subject_data[i],
            group_theta=group_theta,
            model_info=model_info,
            optimizer_options=optimizer_options,
            verbose=verbose # Pass verbose flag down
        ) for i in range(num_subjects)
    )
    results.sort(key=lambda res: res.pid_index if res else -1) # Sort safely if None possible
    # if verbose: tqdm.write("E-step finished.")
    return results


# --- M-Step Function ---
def run_m_step(e_step_results: List[EStepResult],
               param_names: List[str],
               verbose: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Performs the M-step: updates group hyperparameters (mu, var).
    (Implementation is the same as before)
    """
    valid_results = [res for res in e_step_results if res is not None and np.all(np.isfinite(res.map_estimate))]
    if not valid_results:
        raise ValueError("M-step received no valid E-step results.")

    num_subjects = len(valid_results)
    num_params = len(param_names)
    new_group_theta = {}

    all_maps = np.array([res.map_estimate for res in valid_results])
    all_vars = np.array([np.diag(res.hessian_inv) for res in valid_results])

    if all_maps.ndim != 2 or all_maps.shape[1] != num_params or \
       all_vars.ndim != 2 or all_vars.shape[1] != num_params:
         raise ValueError(f"Shape mismatch in aggregated E-step results. MAPs: {all_maps.shape}, Vars: {all_vars.shape}, Params: {num_params}")

    for i, name in enumerate(param_names):
        new_mu = np.mean(all_maps[:, i])
        mean_squared_param = np.mean(all_maps[:, i]**2 + all_vars[:, i])
        new_var = mean_squared_param - new_mu**2
        new_var = np.maximum(new_var, EPSILON**2)

        if not np.isfinite(new_mu) or not np.isfinite(new_var):
            raise RuntimeError(f"Non-finite group parameters calculated for {name} in M-step.")

        new_group_theta[name] = {'mu': new_mu, 'var': new_var}

    # Print summary only outside the loop, controlled by main function's verbose
    # if verbose: print("M-step finished.")
    return new_group_theta


# --- Main EM Fitting Function ---
def fit_em_hierarchical(all_subject_data: List[Dict[str, Any]],
                        model: ModelProtocol,
                        initial_group_theta: Dict[str, Dict[str, float]],
                        max_iter: int = 100,
                        tolerance: float = 1e-4,
                        n_jobs: int = -1,
                        optimizer_options: Dict = None,
                        verbose: bool = True) -> FitResult:
    """
    Fits a hierarchical model using Expectation-Maximization with TQDM progress bar.
    (Implementation is the same as the last version)
    """
    start_time = time.time()
    model_info = model.get_likelihood_info()
    param_names = model_info.param_names
    num_params = len(param_names)
    num_subjects = len(all_subject_data)

    group_theta = deepcopy(initial_group_theta)
    group_theta_old = deepcopy(group_theta)
    converged = False
    em_objective_sequence = []

    if verbose:
        print(f"Starting EM fitting for {num_subjects} subjects, {num_params} parameters...")
        print(f"Initial Theta (Transformed): {group_theta}")

    e_step_results = None
    pbar = tqdm(range(max_iter), desc="EM Iteration", disable=not verbose, leave=True)
    iteration = 0
    current_objective = np.inf

    for iteration in pbar:
        group_theta_old = deepcopy(group_theta)

        # --- E-Step ---
        e_step_results = run_e_step_parallel(
            all_subject_data, group_theta, model_info, optimizer_options, n_jobs,
            verbose=False # Suppress internal prints from parallel step
        )

        valid_e_step_results = [res for res in e_step_results if res is not None and np.all(np.isfinite(res.map_estimate))]
        if not valid_e_step_results:
             warnings.warn("E-step failed for all subjects. Aborting.")
             converged = False
             break

        # --- M-Step ---
        try:
            group_theta = run_m_step(
                valid_e_step_results, param_names,
                verbose=False # Suppress internal prints
            )
        except Exception as e:
            warnings.warn(f"Error during M-step (iteration {iteration + 1}): {e}. Aborting.")
            converged = False
            break

        # --- Calculate EM Objective & Check Convergence ---
        current_objective = sum(res.neg_log_posterior for res in valid_e_step_results if np.isfinite(res.neg_log_posterior))
        em_objective_sequence.append(current_objective)

        delta_theta = 0.0
        for name in param_names:
            if name in group_theta and name in group_theta_old:
                 delta_theta += np.abs(group_theta[name]['mu'] - group_theta_old[name]['mu'])
                 delta_theta += np.abs(group_theta[name]['var'] - group_theta_old[name]['var'])
            else:
                 warnings.warn(f"Parameter {name} missing in theta comparison.")
                 delta_theta = np.inf

        pbar.set_postfix({ 'Objective': f"{current_objective:.4f}", 'dTheta': f"{delta_theta:.6f}" })

        if delta_theta < tolerance and iteration > 0:
            converged = True
            break

    else: # Loop finished without break
        if verbose and max_iter > 0 : warnings.warn(f"EM did not converge after {max_iter} iterations.")

    if hasattr(pbar,'last_print_t') and pbar.last_print_t: pbar.close()

    if converged and verbose: print(f"\nConvergence reached after {iteration + 1} iterations.")

    # --- Post-Convergence Calculations ---
    final_log_posterior_sum = 0.0
    final_log_likelihood_sum = 0.0
    if valid_e_step_results:
        for res in valid_e_step_results:
            subj_idx = res.pid_index
            log_lik = model_info.log_likelihood_func(res.map_estimate, all_subject_data[subj_idx])
            log_pri = log_prior(res.map_estimate, group_theta, param_names)
            if np.isfinite(log_lik) and np.isfinite(log_pri):
                 final_log_posterior_sum += (log_lik + log_pri)
                 final_log_likelihood_sum += log_lik
            else:
                 if verbose: warnings.warn(f"Could not calculate final posterior/likelihood for subject index {subj_idx}")
                 final_log_posterior_sum = -np.inf; break

        log_marginal_likelihood_approx = final_log_posterior_sum
        if verbose:
            print(f"Approximated Log Marginal Likelihood (Sum Log Posteriors): {log_marginal_likelihood_approx:.4f}")
            print(f"Sum of Individual Log Likelihoods at MAPs: {final_log_likelihood_sum:.4f}")
    else:
        log_marginal_likelihood_approx = -np.inf

    total_trials = sum(len(subj['choices'].ravel()) for subj in all_subject_data) # Use ravel for 1D/2D
    bic_int = calculate_bic_int(log_marginal_likelihood_approx, num_subjects, total_trials, group_theta)

    if verbose:
        print(f"\nBIC_int: {bic_int:.4f}")
        print("Final Group Parameters (Transformed Space):")
        # Pretty print theta
        for name, params in group_theta.items():
            print(f"  {name}: mu={params['mu']:.4f}, var={params['var']:.4f}")


    end_time = time.time()
    if verbose: print(f"Total fitting time: {end_time - start_time:.2f} seconds")

    final_iterations = iteration + 1 if converged or iteration == max_iter - 1 else iteration

    fit_result = FitResult(
        group_params=group_theta,
        individual_params=e_step_results if e_step_results else [],
        bic_int=bic_int,
        convergence_iterations=final_iterations,
        converged=converged,
        final_objective=current_objective if e_step_results else np.inf
    )

    return fit_result