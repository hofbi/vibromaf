"""Spectral Temporal SIMilarity Tests."""

# pylint: disable=duplicate-code

import math
import unittest

import numpy as np

from vibromaf.metrics.stsim import STSIM, st_sim


class STSIMTest(unittest.TestCase):
    """STSIM Test."""

    def test_st_sim_wrapper__dist_and_ref_identical__should_be_one(self):
        signal = np.array([0, 1])
        result = st_sim(signal, signal)
        self.assertEqual(1, result)

    def test_st_sim_wrapper__sample_signals(self):
        sample_reference_signal = np.ones(1000) * 1000 + np.random.randn(1000)
        sample_distorted_signal = sample_reference_signal + np.random.randn(1000)

        result = st_sim(sample_distorted_signal, sample_reference_signal)

        self.assertGreaterEqual(1, result)
        self.assertGreaterEqual(result, 0)

    def test_st_sim__truncated_signals_identical__dist_should_be_truncated(self):
        signal = np.linspace(0, 1, 2800)
        dist = np.append(np.linspace(0, 1, 2800), np.ones(300))
        with self.assertWarnsRegex(RuntimeWarning, r"Truncating distorted signal"):
            result = st_sim(dist, signal)
        self.assertEqual(1, result)

    def test_st_sim__dist_larger_than_ref__dist_should_be_truncated(self):
        signal = np.linspace(0, 1, 2800)
        dist = np.append(np.ones(300), np.linspace(0, 1, 2800))
        with self.assertWarnsRegex(RuntimeWarning, r"Truncating distorted signal"):
            result = st_sim(dist, signal)
        self.assertGreaterEqual(1, result)
        self.assertGreaterEqual(result, 0)

    def test_st_sim__dist_shorter_than_ref__should_throw(self):
        signal = np.append(np.linspace(0, 1, 2800), np.ones(300))
        dist = np.linspace(0, 1, 2800)
        with self.assertRaisesRegex(ValueError, r"Distorted .* must not be shorter"):
            st_sim(dist, signal)

    def test_compute_sim__one_block_zero__zero(self):
        ref_block = np.ones((4, 2))
        dist_block = np.zeros((4, 2))
        result = STSIM.compute_sim(ref_block, dist_block)
        self.assertEqual(0, result)

    def test_compute_sim__blocks_identical__one(self):
        block = np.ones((4, 3))
        result = STSIM.compute_sim(block, block)
        self.assertEqual(1, result)

    def test_compute_sim__values_larger_than_one_possible(self):
        ref_block = np.array([[0.5] * 4])
        dist_block = np.array([[0.5, 0.5, 0.5, 1]])
        result = STSIM.compute_sim(ref_block, dist_block)
        self.assertAlmostEqual(1.25, result)

    def test_compute_sim__input_signals_identical_zero__should_be_not_nan(self):
        signal = np.zeros((4, 1024))
        distorted = np.zeros((4, 1024))
        result = STSIM.compute_sim(distorted, signal)
        self.assertFalse(math.isnan(result))

    def test_st_sim_init__eta_grater_one__should_throw(self):
        with self.assertRaisesRegex(ValueError, "Eta must be between 0 and 1."):
            STSIM(eta=1.1)

    def test_st_sim_init__eta_negative__should_throw(self):
        with self.assertRaisesRegex(ValueError, "Eta must be between 0 and 1."):
            STSIM(eta=-0.1)
