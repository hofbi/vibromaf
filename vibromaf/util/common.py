"""Common utilities"""

import numpy as np
from sklearn.metrics import mean_squared_error


def print_metirc(text: str, score: float):
    """Print metric score in predefined format"""
    print(f"{text:40s} {score:.3f}")


def print_mse_and_pc(name: str, y_true, y_pred):
    """Print MSE and Pearson Correlation for signal"""
    mse_test = mean_squared_error(y_true, y_pred)
    cor_test = np.corrcoef(y_true, y_pred)[0, 1]

    print_metirc(f"{name} MSE", mse_test)
    print_metirc(f"{name} PC", cor_test)
