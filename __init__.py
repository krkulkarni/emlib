# pyem/__init__.py
from .model_interface import ModelProtocol, ModelLikelihoodInfo, EStepResult, FitResult
from .transforms import param_transformations, transform_alpha, inverse_transform_alpha, transform_beta, inverse_transform_beta # Expose common ones
from .math_utils import softmax
from .em_core import fit_em_hierarchical # Expose the main fitting function
from .bic import calculate_bic_int
from .plotting_utils import plot_parameter_distributions, plot_model_comparison_bic