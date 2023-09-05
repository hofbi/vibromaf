"""Utility functions for MATLAB files."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import io

from vibromaf import config


def load_signal_from_mat(mat_file: Path, signal_name: str) -> np.array:
    """Load .mat file and parse signal from it."""
    mat = io.loadmat(str(mat_file))
    try:
        return mat[signal_name]
    except KeyError as exc:
        raise KeyError(f"Available keys: {mat.keys()}") from exc


def load_data_for_metric(
    metric: str, test_indices: List[int]
) -> Tuple[np.array, np.array]:
    """Load and concatenate the training and test data."""
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
        element for element in range(vcpwq.shape[1]) if element not in test_indices
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
    """Split the data into equal pieces:

    As we concatenate them per codec this is a split per codec
    """
    return np.split(data, number_of_codecs)


def reshape_per_compression_rate(
    data: np.array, number_of_compression_levels: int = 17
) -> np.array:
    """Reshape the data into same compression level per row."""
    number_of_columns = int(data.size / number_of_compression_levels)
    return data.reshape((number_of_compression_levels, number_of_columns))


class MatSignalLoader:
    """Helper class to load test signals from mat files."""

    def __init__(self, metric: str, codec: str = "VCPWQ") -> None:
        self.__reference = load_signal_from_mat(
            config.DATA_PATH / "Signals.mat", "Signals"
        )
        self.__distorted = load_signal_from_mat(
            config.DATA_PATH / f"recsig_{codec}.mat", f"recsig_{codec}"
        )
        self.__metric_scores = load_signal_from_mat(
            config.DATA_PATH / f"{metric}_{codec}.mat", f"{metric}_{codec}"
        )

    def signal_ids(self):
        return range(self.__reference.shape[1])

    def compression_levels(self):
        return range(self.__distorted.shape[0])

    def load_reference_signal(self, signal_id: int):
        return self.__reference[:, signal_id]

    def load_distorted_signal(self, signal_id: int, compression_level: int):
        return self.__distorted[compression_level, signal_id].reshape(
            -1,
        )

    def load_quality_score(self, signal_id: int, compression_level: int):
        return self.__metric_scores[compression_level, signal_id]
