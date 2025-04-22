# emlib: Hierarchical Model Fitting with Expectation-Maximization

`emlib` is a Python library designed for fitting hierarchical computational models to behavioral data using the Expectation-Maximization (EM) algorithm. It focuses on estimating group-level parameter distributions (means and variances) and individual-level Maximum A Posteriori (MAP) parameter estimates.

The library aims to be modular, allowing users to define their own computational models and easily fit them to grouped data (e.g., multiple participants), where each participant's data is represented as a pandas DataFrame. It includes utilities for saving and loading fit results to avoid re-computation.

## Features

*   **Hierarchical Modeling:** Estimates group-level Gaussian distributions for model parameters.
*   **Expectation-Maximization:** Implements the EM algorithm with Laplacian approximation for the E-step to find group-level Maximum Likelihood estimates and individual MAP estimates.
*   **Modular Model Definition:** Define your own computational models by implementing a simple protocol (`ModelProtocol`). Models declare required parameters and transformations, and provide a likelihood function that accepts subject data as a pandas DataFrame.
*   **Flexible Data Input:** Expects subject data as a `List[pd.DataFrame]`, allowing models to access various data types (choices, rewards, reaction times, ratings, conditions, etc.) using standard DataFrame column access.
*   **Parameter Transformations:** Handles common parameter transformations (logit/expit, log/exp, identity) to map bounded parameters to an unbounded space suitable for Gaussian priors. Can add custom transformations as well.
*   **Parallel Processing:** Utilizes `joblib` to parallelize the E-step across subjects for faster fitting.
*   **Model Comparison:** Calculates the Integrated Bayesian Information Criterion (BIC_int) for comparing the fit of different hierarchical models.
*   **Result Persistence:** Includes utilities (`save_fit_result`, `load_fit_result`) using `joblib` to efficiently save and load complex `FitResult` objects, preventing redundant computations.
*   **Visualization Utilities:** Includes functions to plot the distributions of fitted individual parameters and compare model BIC values.
*   **Progress Tracking:** Uses `tqdm` for progress bars during fitting.

## Installation

Currently, `emlib` is intended to be used as a local library. Clone or download the repository and ensure the `emlib` directory is in your Python path.

**Dependencies:**

*   numpy
*   scipy
*   pandas
*   joblib (for parallelization and saving/loading results)
*   tqdm (for progress bars)
*   numdifftools (for numerical Hessian calculation)
*   matplotlib & seaborn (Optional, required for plotting utilities)

You can install dependencies using pip:
```bash
pip install numpy scipy pandas joblib tqdm numdifftools matplotlib seaborn
```

## Library Structure

```
emlib/
├── __init__.py         # Makes emlib a package, exports key functions/classes
├── model_interface.py  # Defines protocols (ModelProtocol) and data structures (FitResult, etc.)
├── transforms.py       # Parameter transformation functions (logit, log, identity, etc.)
├── math_utils.py       # Helper math functions (e.g., stable softmax)
├── em_core.py          # Core EM logic (fit_em_hierarchical, E-step, M-step)
├── bic.py              # BIC_int calculation function
├── plotting_utils.py   # Plotting functions (parameter distributions, model comparison)
└── save_utils.py       # Functions for saving/loading FitResult objects using joblib
```

## Usage

1.  **Prepare Data:** Load your data into a list of pandas DataFrames `List[pd.DataFrame]`. Each DataFrame represents one subject. Ensure necessary columns are present.

    ```python
    # Example using a hypothetical loading function
    # from your_utils import load_data_as_list_of_dataframes
    # all_subject_data_list = load_data_as_list_of_dataframes("path/to/data.csv", ...)
    ```

2.  **Define Your Model:** Create a Python class implementing `emlib.ModelProtocol`. Each model will require (at least) the following:
    *   `get_likelihood_info()`: Returns `ModelLikelihoodInfo` (param names, transforms, likelihood function).
    *   `log_likelihood_func`: Accepts `(transformed_params: np.ndarray, data: pd.DataFrame)` and returns log-likelihood (float).
    *   `get_param_bounds()` (Optional).

3.  **Define Initial Group Parameters:** Provide starting guesses for `mu` and `var` for each parameter *in the transformed space*.

    ```python
    initial_theta = {
        'beta': {'mu': np.log(4.0), 'var': 10**2}, # broad prior @ μ=log(4), σ=10
        'lr':   {'mu': logit(0.5),  'var': 10**2}  # broad prior @ μ=0, σ=10
    }
    ```

4.  **Example Run Fitting (with Load/Save):** Check if results exist before running `fit_em_hierarchical` and save results after fitting.

    ```python
    import os
    from emlib import fit_em_hierarchical, save_fit_result, load_fit_result
    from model_lib import QLearningRW
    from your_utils import load_data_as_list_of_dataframes

    # ------ Load Data ------
    all_subject_dataframes = load_data_as_list_of_dataframes("path/to/data.csv", ...)

    # ------ Define Save Directory ------
    SAVE_DIR = "fit_results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_name = "QLearningRW" # Example model
    save_filename = os.path.join(SAVE_DIR, f"fit_result_{model_name}.joblib")

    # ------ Define Model and Fitting Options ------
    model_instance = QLearningRW()
     initial_theta = {
        'beta': {'mu': np.log(4.0), 'var': 10**2},      # broad prior @ μ=log(4), σ=10
        'lr':   {'mu': logit(0.5),  'var': 10**2}       # broad prior @ μ=0, σ=10
    }
    optimizer_options = {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-5}
    }

    # ------ Fit Model ------
    fit_result = load_fit_result(save_filename)         # Try loading first
    if fit_result is None:
        print(f"Fitting model {model_name}...")
        fit_result = fit_em_hierarchical(
            all_subject_data=all_subject_dataframes,    # List of DataFrames
            model=model,                                # Model instance
            initial_group_theta=initial_theta,          # Initial group parameters
            max_iter=100,                               # Max iterations for EM
            tolerance=1e-3,                             # Convergence tolerance
            n_jobs=-1,                                  # Use all available cores
            optimizer_options=optimizer_options,        # Options for optimizer
            verbose=True                                # Show progress for fitting runs
        )
        if fit_result:                                  # Save only if fitting was successful
            save_fit_result(fit_result, save_filename)
    else:
        print(f"Loaded existing results for model {model_name}.")

    ```

5.  **Analyze Results:** If `fit_result` is not `None`, access its attributes (`converged`, `group_params`, `individual_params`, `bic_int`).

6.  **Plot Results (Optional):** Use the plotting utilities with the loaded or computed `fit_result`.

    ```python
    from emlib import plot_parameter_distributions, plot_model_comparison_bic

    if fit_result and fit_result.converged:
        plot_parameter_distributions(fit_result, model_instance)

    # Example comparing multiple fits (load or fit each one first)
    # results_list = [load_or_fit_model_A(), load_or_fit_model_B()]
    # names_list = ["Model A", "Model B"]
    # plot_model_comparison_bic([r for r in results_list if r], names_list) # Filter None results
    ```

## Adding New Models

1.  Create a new model class implementing `emlib.ModelProtocol` (see step 2 in Usage). Ensure the `log_likelihood_func` correctly accesses columns from the input `pd.DataFrame`.
2.  Define its `initial_theta` in your fitting script.
3.  Add it to your fitting loop, including appropriate saving/loading logic using a unique filename.

## Acknowledgments

This library is inspired by the work of several researchers in the field of computational modeling and reinforcement learning. The original code was adapted from MATLAB implementations and has been modified for Python to enhance usability and flexibility. 

Many thanks to <a href="https://shawnrhoads.github.io/">Dr. Shawn Rhoads</a> in particular and his terrific <a href="https://github.com/shawnrhoads/pyEM">pyEM library</a> which inspired the design of this library.

**See References:**
- Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP estimation in Python. Zenodo. https://doi.org/10.5281/zenodo.10415396
- Wittmann, M. K., Fouragnan, E., Folloni, D., Klein-Flügge, M. C., Chau, B. K., Khamassi, M., & Rushworth, M. F. (2020). Global reward state affects learning and activity in raphe nucleus and anterior insula in monkeys. Nature Communications, 11(1), 3771. https://doi.org/10.1038/s41467-020-17343-w
- Cutler, J., Wittmann, M. K., Abdurahman, A., Hargitai, L. D., Drew, D., Husain, M., & Lockwood, P. L. (2021). Ageing is associated with disrupted reinforcement learning whilst learning to help others is preserved. Nature Communications, 12(1), 4440. https://doi.org/10.1038/s41467-021-24576-w
- Rhoads, S. A., Gan, L., Berluti, K., OConnell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2023). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. PsyArXiv. https://doi.org/10.31234/osf.io/rf4x9
- Daw, N. D. (2011). Trial-by-trial data analysis using computational models. Decision making, affect, and learning: Attention and performance XXIII, 23(1). https://doi.org/10.1093/acprof:oso/9780199600434.003.0001 [<a href="https://www.princeton.edu/~ndaw/d10.pdf">pdf</a>]
- Huys, Q. J., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, R. J., & Dayan, P. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS computational biology, 7(4), e1002028. https://doi.org/10.1371/journal.pcbi.1002028 

**For MATLAB flavors of this algorithm:**
- https://github.com/sjgershm/mfit
- https://github.com/mpc-ucl/emfit
- https://osf.io/s7z6j