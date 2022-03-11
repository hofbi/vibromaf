"""Spectral Perceptual Quality Index Tests"""

import unittest

import numpy as np

from vibromaf.metrics.spqi import spqi


class SPQITest(unittest.TestCase):
    """SPQI Test"""

    def test_spqi_wrapper__dist_and_ref_identical__should_be_one(self):
        signal = np.linspace(0, 1, 1000)
        result = spqi(signal, signal)
        self.assertEqual(1, result)

    def test_spqi_wrapper__sample_signals(self):
        sample_reference_signal = np.ones(1000) * 1000 + np.random.randn(1000)
        sample_distorted_signal = sample_reference_signal + np.random.randn(1000)

        result = spqi(sample_distorted_signal, sample_reference_signal)

        self.assertGreaterEqual(1, result)
        self.assertGreaterEqual(result, 0)

    def test_spqi__truncated_signals_identical__dist_should_be_truncated(self):
        signal = np.linspace(0, 1, 2800)
        dist = np.append(np.linspace(0, 1, 2800), np.ones(300))
        with self.assertWarnsRegex(RuntimeWarning, r"Truncating distorted signal"):
            result = spqi(dist, signal)
        self.assertEqual(1, result)

    def test_spqi__dist_larger_than_ref__dist_should_be_truncated(self):
        signal = np.linspace(0, 1, 2800)
        dist = np.append(np.ones(300), np.linspace(0, 1, 2800))
        with self.assertWarnsRegex(RuntimeWarning, r"Truncating distorted signal"):
            result = spqi(dist, signal)
        self.assertGreaterEqual(1, result)
        self.assertGreaterEqual(result, 0)

    def test_spqi__dist_shorter_than_ref__should_throw(self):
        signal = np.append(np.linspace(0, 1, 2800), np.ones(300))
        dist = np.linspace(0, 1, 2800)
        with self.assertRaisesRegex(ValueError, r"Distorted .* must not be shorter"):
            spqi(dist, signal)

    def test_spqi__input_signals_identical_zero__should_be_valid(self):
        signal = np.zeros(1024)
        distorted = np.zeros(1024)
        signal[-1] = 0.00001
        result = spqi(distorted, signal)
        self.assertGreaterEqual(1, result)
        self.assertGreaterEqual(result, 0)
