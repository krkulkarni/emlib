import numpy as np
import pandas as pd
import math
from typing import List, Dict, Optional, Sequence

# Import necessary components from the library
from .model_interface import FitResult, ModelProtocol
from .transforms import param_transformations # Use the defined transformations

# Optional imports - check if available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _plotting_enabled = True
except ImportError:
    _plotting_enabled = False
    # Define placeholder classes if libraries are missing, allowing the function to exist
    plt = None
    sns = None

def plot_parameter_distributions(fit_result: FitResult,
                                 model: ModelProtocol,
                                 max_beta_plot: Optional[float] = 50.0):
    """
    Plots the distributions of individual MAP parameter estimates from EM results.

    Generates boxplots overlaid with stripplots for each parameter.

    Args:
        fit_result: The FitResult object returned by fit_em_hierarchical.
        model: The model instance (adhering to ModelProtocol) that was fitted.
               Used to get parameter names and inverse transformations.
        max_beta_plot: Optional upper limit for the y-axis of the beta plot
                       to prevent extreme values from dominating the scale. Set to
                       None for no limit.
    """
    if not _plotting_enabled:
        print("Plotting libraries (matplotlib, seaborn, pandas) not found. Skipping plot.")
        print("Please install them: pip install matplotlib seaborn pandas")
        return

    # if not fit_result.converged:
    #     print("EM algorithm did not converge. Skipping parameter distribution plots.")
    #     return

    if not fit_result.individual_params:
        print("No valid individual parameter estimates found in fit_result. Skipping plots.")
        return

    model_info = model.get_likelihood_info()
    param_names = model_info.param_names
    inv_transforms = {name: info['inverse'] for name, info in model_info.transform_funcs.items()}

    # Extract native-space MAP estimates for all parameters
    native_maps: Dict[str, List[float]] = {name: [] for name in param_names}
    subject_indices: List[int] = []

    valid_results_count = 0
    for res in fit_result.individual_params:
        if res is not None and np.all(np.isfinite(res.map_estimate)):
            subject_indices.append(res.pid_index)
            for i, name in enumerate(param_names):
                try:
                    native_val = inv_transforms[name](res.map_estimate[i])
                    native_maps[name].append(native_val)
                except Exception as e:
                    print(f"Warning: Could not inverse transform parameter '{name}' for subject index {res.pid_index}. Error: {e}")
                    # Handle error, e.g., append NaN or skip subject for this param?
                    # Appending NaN might be better for consistent DataFrame length if possible
                    native_maps[name].append(np.nan)

            valid_results_count += 1
        else:
            print(f"Skipping plotting for subject index {res.pid_index if res else 'Unknown'} due to invalid result.")

    if valid_results_count == 0:
        print("No subjects with valid parameter estimates found to plot.")
        return

    # Create DataFrame - handle potential NaNs if transforms failed
    try:
        plot_data = {'Subject Index': subject_indices}
        plot_data.update({f'MAP {name}': native_maps[name] for name in param_names})
        individual_params_df = pd.DataFrame(plot_data)
        individual_params_df = individual_params_df.dropna() # Drop rows where any transform failed
        if individual_params_df.empty:
             print("DataFrame is empty after removing NaNs from failed transformations. Skipping plots.")
             return
    except ValueError as e:
        print(f"Error creating DataFrame for plotting, likely due to inconsistent list lengths: {e}")
        return


    # Determine plot layout (e.g., 2 columns)
    n_params = len(param_names)
    n_cols = n_params
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5), squeeze=False) # Ensure axes is always 2D
    axes_flat = axes.flatten() # Flatten for easy iteration

    print(f"\nGenerating plots for {len(individual_params_df)} subjects with valid estimates...")

    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        col_name = f'MAP {name}'

        # Use specific y-limits for known parameters if desired
        ylim = None
        ylabel = f'{name}'
        if name == 'lr' or name == 'alpha':
             ylim = (0, 1)
             ylabel = f'Learning Rate ({name})'
        elif name == 'beta':
             ylabel = f'Inv. Temp. ({name})'
             if max_beta_plot is not None:
                 ylim = (0, max_beta_plot) # Use provided limit
             else:
                 ylim = (0, None) # Only lower limit

        sns.boxplot(y=col_name, data=individual_params_df, ax=ax, color='skyblue', width=0.3, showfliers=False)
        sns.stripplot(y=col_name, data=individual_params_df, ax=ax, color='black', alpha=0.6, size=4)
        ax.set_title(f'Distribution of MAP {name} Estimates')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("") # Remove x-label as it's clear from y

        if ylim:
            ax.set_ylim(ylim)

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_model_comparison_bic(fit_results_list: Sequence[FitResult],
                              model_names: Sequence[str]):
    """
    Creates a bar plot comparing the BIC_int values of multiple model fits.

    Lower BIC_int indicates a better model fit, penalizing for complexity.
    Also plots Delta BIC (difference relative to the best model).

    Args:
        fit_results_list: A sequence (list or tuple) of FitResult objects,
                          one for each model fit to compare.
        model_names: A sequence of strings corresponding to the names of the
                     models in fit_results_list. Must be the same length.
    """
    if not _plotting_enabled:
        print("Plotting libraries (matplotlib, seaborn, pandas) not found. Skipping plot.")
        print("Please install them: pip install matplotlib seaborn pandas")
        return

    if len(fit_results_list) != len(model_names):
        raise ValueError("Length of fit_results_list must match length of model_names.")

    if len(fit_results_list) < 2:
        print("Need at least two model fits to compare.")
        return

    bic_values = [res.bic_int if res else np.inf for res in fit_results_list]

    # Handle potential infinities (e.g., if fitting failed or BIC couldn't be computed)
    valid_indices = [i for i, bic in enumerate(bic_values) if np.isfinite(bic)]
    if not valid_indices:
        print("No valid BIC values found to plot.")
        return
    if len(valid_indices) < len(bic_values):
        print("Warning: Some models have invalid BIC values and will be excluded from the plot.")

    valid_bics = [bic_values[i] for i in valid_indices]
    valid_names = [model_names[i] for i in valid_indices]

    # Find the best model (lowest BIC) among valid results
    min_bic = min(valid_bics)
    delta_bics = [bic - min_bic for bic in valid_bics]

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Model': valid_names,
        'BIC_int': valid_bics,
        'Delta BIC': delta_bics
    })
    plot_df = plot_df.sort_values('BIC_int') # Sort by BIC for clearer comparison

    # --- Create Plot ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=False) # Don't share y-axis

    # Plot 1: Absolute BIC_int values
    sns.barplot(x='BIC_int', y='Model', hue='Model', data=plot_df, ax=ax[0], legend=False)
    ax[0].set_title('Model Comparison (Lower BIC is Better)')
    ax[0].set_xlabel('Integrated BIC (BIC_int)')
    ax[0].set_ylabel('Model')

    # Plot 2: Delta BIC values (relative to best model)
    sns.barplot(x='Delta BIC', y='Model', hue='Model', data=plot_df, ax=ax[1], legend=False)
    ax[1].set_title('Model Comparison (Delta BIC)')
    ax[1].set_xlabel('Delta BIC (Relative to Best Model)')
    ax[1].set_ylabel('') # Remove redundant y-label

    # Add text labels for Delta BIC
    for container in ax[1].containers:
        ax[1].bar_label(container, fmt='%.1f', padding=3)

    sns.despine()
    plt.tight_layout()
    plt.show()