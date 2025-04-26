# pyem/save_utils.py
import joblib
import os
import warnings
from typing import Optional

# Import FitResult for type hinting
from .model_interface import FitResult


def save_fit_result(fit_result: FitResult, filename: str, compress: bool = True):
    """
    Saves a FitResult object using joblib.

    Args:
        fit_result: The FitResult object to save.
        filename: The path (including filename) to save the file to.
                  Should ideally end with '.joblib'.
        compress: Whether to use compression (default True).
    """
    try:
        # Ensure directory exists if filename includes path
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        joblib.dump(fit_result, filename, compress=compress)
        print(f"Fit result saved successfully to: {filename}")
    except Exception as e:
        warnings.warn(f"Error saving fit result to {filename}: {e}")


def load_fit_result(filename: str) -> Optional[FitResult]:
    """
    Loads a FitResult object using joblib.

    Args:
        filename: The path to the joblib file.

    Returns:
        The loaded FitResult object, or None if loading fails or file not found.
    """
    if not os.path.exists(filename):
        # Don't print an error here, just return None - file might not exist yet
        # Calling function can decide if this is an error or expected.
        return None
    try:
        fit_result = joblib.load(filename)
        # Basic type check after loading
        if isinstance(fit_result, FitResult):
            # print(f"Fit result loaded successfully from: {filename}") # Optional load message
            return fit_result
        else:
            warnings.warn(f"Loaded object from {filename} is not of type FitResult.")
            return None
    except Exception as e:
        warnings.warn(f"Error loading fit result from {filename}: {e}")
        return None
