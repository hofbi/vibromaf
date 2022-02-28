"""Tests for spectrum module"""

import unittest

import numpy as np

from vibromaf.signal.spectrum import (
    compute_normalized_spectral_difference,
    compute_spectral_support,
    db2pow,
    mag2db,
    pow2db,
)


class SpectrumTest(unittest.TestCase):
    """Spectrum Tests"""

    def test_pow2db(self):
        signal = np.array([100, 100, 10])
        result = pow2db(signal)
        self.assertListEqual([20, 20, 10], list(result))

    def test_db2pow(self):
        signal = np.array([20, 20, 10])
        result = db2pow(signal)
        self.assertListEqual([100, 100, 10], list(result))

    def test_mag2db(self):
        signal = np.array([100, 100, 10])
        result = mag2db(signal)
        self.assertListEqual([40, 40, 20], list(result))

    def test_compute_normalized_spectral_difference__same_signals_should_be_minus_inf(
        self,
    ):
        signal = np.ones((10, 2))
        result = compute_normalized_spectral_difference(signal, signal)
        self.assertListEqual([-np.inf] * 10, list(result))

    def test_compute_normalized_spectral_difference__different_signals_should_be_positive(
        self,
    ):
        signal_one = np.ones((2, 10))
        signal_two = np.zeros((2, 10))
        result = compute_normalized_spectral_difference(signal_one, signal_two)
        self.assertGreaterEqual(0, result[0])
        self.assertEqual(2, result.size)

    def test_compute_spectral_support__zeros_array__array_with_0p5(self):
        spectrum = np.zeros((2, 4))
        result = compute_spectral_support(spectrum)
        self.assertListEqual([0.5] * 4, list(result[0]))
