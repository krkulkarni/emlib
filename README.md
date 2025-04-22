# emlib: Hierarchical Model Fitting with Expectation-Maximization

`emlib` is a Python library designed for fitting hierarchical computational models to behavioral data using the Expectation-Maximization (EM) algorithm. It focuses on estimating group-level parameter distributions (means and variances) and individual-level Maximum A Posteriori (MAP) parameter estimates.

The library aims to be modular, allowing users to define their own computational models (requiring specific data fields) and easily fit them to grouped data (e.g., multiple participants), where each participant's data is represented as a dictionary.

## Features

*   **Hierarchical Modeling:** Estimates group-level Gaussian distributions for model parameters.
*   **Expectation-Maximization:** Implements the EM algorithm with Laplacian approximation for the E-step to find group-level Maximum Likelihood estimates and individual MAP estimates.
*   **Modular Model Definition:** Define your own computational models by implementing a simple protocol (`ModelProtocol`). Models declare required parameters and transformations, and provide a likelihood function that accepts subject data as a dictionary.
*   **Flexible Data Input:** Expects subject data as a list of dictionaries, allowing models to access various data types (choices, rewards, reaction times, ratings, conditions, etc.) by key.
*   **Parameter Transformations:** Handles common parameter transformations (logit, log/exp, identity) to map bounded parameters to an unbounded space suitable for Gaussian priors. Easily extendable.
*   **Parallel Processing:** Utilizes `joblib` to parallelize the E-step across subjects for faster fitting on multi-core machines.
*   **Model Comparison:** Calculates the Integrated Bayesian Information Criterion (BIC_int) for comparing the fit of different hierarchical models.
*   **Visualization Utilities:** Includes functions to plot the distributions of fitted individual parameters and compare model BIC values.
*   **Progress Tracking:** Uses `tqdm` for progress bars during fitting.

## Installation

Currently, `emlib` is intended to be used as a local library. Clone or download the repository and ensure the `emlib` directory is in your Python path or installed locally (e.g., using `pip install .` from the parent directory).

**Dependencies:**

*   numpy
*   scipy
*   pandas (recommended for data loading/preprocessing utilities and used in some model examples)
*   joblib (for parallelization)
*   tqdm (for progress bars)
*   numdifftools (for numerical Hessian calculation)
*   matplotlib & seaborn (Optional, required for plotting utilities)

You can install dependencies using pip:
```bash
pip install numpy scipy pandas joblib tqdm numdifftools matplotlib seaborn
```

## Library Structure

```
emlib/                  # The EM library
├── __init__.py         # Makes emlib a package, exports key functions/classes
├── model_interface.py  # Defines protocols (ModelProtocol) and data structures (FitResult, etc.)
├── transforms.py       # Parameter transformation functions (logit, log, identity, etc.)
├── math_utils.py       # Helper math functions (e.g., stable softmax)
├── em_core.py          # Core EM logic (fit_em_hierarchical, E-step, M-step)
├── bic.py              # BIC_int calculation function
└── plotting_utils.py   # Plotting functions (parameter distributions, model comparison)
```

## Usage

1.  **Prepare Data:** Load your data into a `List[pd.DataFrame]`. Each DataFrame in the list represents one subject's data in long format (one row per trial). Ensure the DataFrames contain all necessary columns required by the models you intend to fit (e.g., `'Action'`, `'Reward'`, `'Craving Rating'`). Consistent column naming is recommended.

    ```python
    import pandas as pd
    from your_utils import load_data_as_list_of_dataframes # Example helper

    # Example: Load data ensuring each subject is a separate DataFrame
    all_subject_data_list = load_data_as_list_of_dataframes("path/to/data.csv", pid_column='PID')
    # all_subject_data_list = [subject1_df, subject2_df, ...]
    ```
    *(A helper function like `load_and_preprocess_data_df` shown previously can be used or adapted).*

2.  **Define Your Model:** Create a Python class implementing `emlib.ModelProtocol`.
    *   `get_likelihood_info()`: Returns `ModelLikelihoodInfo` specifying `param_names`, `transform_funcs`, and `log_likelihood_func`.
    *   **`log_likelihood_func`**: Must now accept `(transformed_params: np.ndarray, data: pd.DataFrame)` and return the log-likelihood (float). Access data using DataFrame column indexing (e.g., `data['Reward'].values`).
    *   `get_param_bounds()` (Optional): Define native-space bounds.

    ```python
    from models import QLearningRW # Adjust import path

    model_instance = QLearningRW()
    ```

2.  **Define Your Model:** Create a Python class for your computational model that implements the `emlib.ModelProtocol`. This involves defining:
    *   `get_likelihood_info()`: Returns a `ModelLikelihoodInfo` tuple containing:
        *   `param_names`: List of parameter names (e.g., `['beta', 'lr']`).
        *   `transform_funcs`: Dictionary mapping parameter names to their forward/inverse transformation functions (e.g., from `emlib.transforms`). Use `identity` for untransformed parameters.
        *   `log_likelihood_func`: A static or class method that takes `transformed_params` (NumPy array) and `data` (the subject's dictionary) and returns the total log-likelihood (float). This function must access required data fields using dictionary keys (e.g., `data['choices']`).
    *   `get_param_bounds()` (Optional): Returns native-space bounds.

    ```python
    from models import QLearningRW # Example model implementation (adjust import path)

    model_instance = QLearningRW()
    ```
    *(See separate `models/` directory for examples)*

3.  **Define Initial Group Parameters:** Provide starting guesses for the group-level mean (`mu`) and variance (`var`) for each parameter *in the transformed space*.

    ```python
    import numpy as np
    from emlib.transforms import logit # Import necessary transforms

    initial_theta = {
        'beta': {'mu': np.log(4.0), 'var': 1.0**2},
        'lr':   {'mu': logit(0.5),  'var': 1.5**2}
    }
    ```

4.  **Run Fitting:** Call the main fitting function `fit_em_hierarchical`.

    ```python
    from emlib import fit_em_hierarchical

    fit_result = fit_em_hierarchical(
        all_subject_data=all_subject_data_list,
        model=model_instance,
        initial_group_theta=initial_theta,
        max_iter=100,
        tolerance=1e-4,
        n_jobs=-1, # Use all cores
        verbose=True # Show progress bar and output
    )
    ```

5.  **Analyze Results:** The returned `FitResult` object contains fit details (see `emlib/model_interface.py` for fields). Use the `group_params` (transformed) and `individual_params` (transformed MAPs + Hessians).

6.  **Plot Results (Optional):** Use the plotting utilities.

    ```python
    from emlib import plot_parameter_distributions, plot_model_comparison_bic

    if fit_result.converged:
        # Plot distributions for this fit
        plot_parameter_distributions(fit_result, model_instance)

    # Example comparing multiple fits
    # results_list = [fit_result_model1, fit_result_model2]
    # names_list = ["Model A", "Model B"]
    # plot_model_comparison_bic(results_list, names_list)
    ```

## Adding New Models

1.  Create a new Python file (e.g., `models/my_cool_model.py`).
2.  Define a class implementing `emlib.ModelProtocol`.
3.  Implement `get_likelihood_info`:
    *   Specify `param_names`.
    *   Define `transform_funcs` using `emlib.transforms`.
    *   Implement `_calculate_log_likelihood(transformed_params, data_dict)`: Access necessary data using keys from the `data_dict` (e.g., `choices = data_dict['choices']`, `rts = data_dict.get('rt', None)`). Ensure robustness if optional data keys are missing.
4.  Implement `get_param_bounds` (optional).
5.  Import your model and use it in your fitting script, ensuring your data loading provides the required dictionary keys.

## License

This project is licensed under the MIT License - see the LICENSE file for details (or include the text below).

```text
MIT License

Copyright (c) 2025 Kaustubh R. Kulkarni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```