"""Utility functions for the vibromaf model"""

import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def make_vibromaf_pipeline() -> Pipeline:
    """Create the vibromaf SVM regressor pipeline"""
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=3000, epsilon=0.1))


def save_model(model, model_path: Path) -> None:
    """Save the model into a file"""
    model_path.write_bytes(pickle.dumps(model))


def load_model(model_path: Path):
    """Load the model from a file"""
    return pickle.loads(model_path.read_bytes())
