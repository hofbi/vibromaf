"""Utility functions for MATLAB files"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import io

from vibromaf import config


def load_signal_from_mat(mat_file: Path, signal_name: str) -> np.array:
    """Load .mat file and parse signal from it"""
    mat = io.loadmat(str(mat_file))
    try:
        return mat[signal_name]
    except KeyError:
        raise KeyError(f"Available keys: {mat.keys()}")


def load_data_for_metric(
    metric: str, test_indices: List[int]
) -> Tuple[np.array, np.array]:
    """Load and concatenate the training and test data"""
    vcpwq = load_signal_from_mat(
        config.DATA_PATH / f"{metric}_VCPWQ.mat", f"{metric}_VCPWQ"
    )
    pvcslp = load_signal_from_mat(
        config.DATA_PATH / f"{metric}_PVCSLP.mat", f"{metric}_PVCSLP"
    )
    vpcds = load_signal_from_mat(
        config.DATA_PATH / f"{metric}_VPCDS.mat", f"{metric}_VPCDS"
    )

    train_indices = [
        element for element in range(0, vcpwq.shape[1]) if element not in test_indices
    ]

    return np.concatenate(
        [
            vcpwq[:, train_indices].flatten(),
            pvcslp[:, train_indices].flatten(),
            vpcds[:, train_indices].flatten(),
        ]
    ), np.concatenate(
        [
            vcpwq[:, test_indices].flatten(),
            pvcslp[:, test_indices].flatten(),
            vpcds[:, test_indices].flatten(),
        ]
    )


def split_per_codec(data: np.array, number_of_codecs: int = 3) -> np.array:
    """
    Split the data into equal pieces:
    As we concatenate them per codec this is a split per codec
    """
    return np.split(data, number_of_codecs)


def reshape_per_compression_rate(
    data: np.array, number_of_compression_levels: int = 17
) -> np.array:
    """
    Reshape the data into same compression level per row:
    """
    number_of_columns = int(data.size / number_of_compression_levels)
    return data.reshape((number_of_compression_levels, number_of_columns))
