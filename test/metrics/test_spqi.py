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

        self.assertAlmostEqual(1.0, result)
