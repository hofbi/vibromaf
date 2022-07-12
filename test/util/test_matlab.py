"""Test matlab utilities."""

import unittest
from pathlib import Path

import numpy as np

from vibromaf.util import matlab


class MatlabTest(unittest.TestCase):
    """Matlab Test."""

    RES_PATH = Path(__file__).parent / "res"

    def test_load_signal_from_mat__valid_mat__should_load_signal(self):
        test_signal = np.array([[0, 1, 2, 3]])

        result = matlab.load_signal_from_mat(MatlabTest.RES_PATH / "test.mat", "test")

        self.assertTrue(np.array_equal(test_signal, result))

    def test_load_signal_from_mat__invalid_key__should_raise_key_error(self):
        with self.assertRaises(KeyError):
            matlab.load_signal_from_mat(MatlabTest.RES_PATH / "test.mat", "foo")

    def test_load_signal_from_mat__invalid_file__should_raise_file_error(self):
        with self.assertRaises(FileNotFoundError):
            matlab.load_signal_from_mat(Path("test.mat"), "test")

    def test_split_per_codec__vector_should_be_split_into_three(self):
        data = np.arange(9)

        results = matlab.split_per_codec(data)

        self.assertListEqual([0, 1, 2], list(results[0]))
        self.assertListEqual([3, 4, 5], list(results[1]))
        self.assertListEqual([6, 7, 8], list(results[2]))

    def test_reshape_per_compression_rate__one_compression_level_per_row(self):
        data = np.arange(9)

        results = matlab.reshape_per_compression_rate(
            data, number_of_compression_levels=3
        )

        self.assertListEqual([0, 1, 2], list(results[0]))
        self.assertListEqual([3, 4, 5], list(results[1]))
        self.assertListEqual([6, 7, 8], list(results[2]))

    def test_reshape_per_compression_rate__invalid_size_should_throw(self):
        data = np.arange(10)

        with self.assertRaises(ValueError):
            matlab.reshape_per_compression_rate(data, number_of_compression_levels=3)
