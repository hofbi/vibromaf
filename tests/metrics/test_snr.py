"""Signal to Noise Ratio Tests."""

import unittest

import numpy as np

from vibromaf.metrics.snr import nsnr, snr


class SNRTest(unittest.TestCase):
    """SNR Test."""

    def test_snr__dist_and_ref_identical__should_be_inf(self):
        signal = np.array([0, 1])
        result = snr(signal, signal)
        self.assertEqual(np.inf, result)

    def test_snr__nonzero_signal_and_zero_signal__should_be_zero(self):
        signal = np.linspace(1, 10, 20)
        zero_signal = np.zeros(20)
        result = snr(zero_signal, signal)
        self.assertEqual(0, result)

    def test_snr__ones_signal_0p9_signal__should_be_20(self):
        signal = np.ones(10)
        distorted = 0.9 * np.ones(10)
        result = snr(distorted, signal)
        self.assertAlmostEqual(20, result)

    def test_snr__sample_signals(self):
        sample_reference_signal = np.ones(1000) * 1000 + np.random.randn(1000)
        sample_distorted_signal = sample_reference_signal + np.random.randn(1000)

        result = snr(sample_distorted_signal, sample_reference_signal)

        self.assertAlmostEqual(60, result, delta=2.0)

    def test_snr__dist_larger_than_ref__dist_should_be_truncated_and_warn(self):
        signal = np.array([0, 1])
        dist = np.array([0, 1, 2])
        with self.assertWarnsRegex(RuntimeWarning, r"Truncating distorted signal"):
            result = snr(dist, signal)
        self.assertEqual(np.inf, result)

    def test_snr__dist_shorter_than_ref__should_throw(self):
        signal = np.array([0, 1, 2])
        dist = np.array([0, 1])
        with self.assertRaisesRegex(ValueError, r"Distorted .* must not be shorter"):
            snr(dist, signal)

    def test_nsnr__snr_larger_than_max__should_be_1(self):
        signal = np.array([0, 1])
        result = nsnr(signal, signal)
        self.assertEqual(1, result)

    def test_nsnr__snr_negative__should_be_0(self):
        signal = np.array([0, 1])
        result = nsnr(-signal, signal)
        self.assertEqual(0, result)

    def test_nsnr__ones_signal_0p9_signal__should_be_quarter(self):
        signal = np.ones(10)
        distorted = 0.9 * np.ones(10)
        result = nsnr(distorted, signal, normalization_db=80)
        self.assertAlmostEqual(0.25, result)
