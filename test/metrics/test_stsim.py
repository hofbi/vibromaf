"""Spectral Temporal SIMilarity Tests"""

import unittest

import numpy as np

from vibromaf.metrics.stsim import STSIM, st_sim


class STSIMTest(unittest.TestCase):
    """STSIM Test"""

    def test_st_sim_wrapper__dist_and_ref_identical__should_be_one(self):
        signal = np.array([0, 1])
        result = st_sim(signal, signal)
        self.assertEqual(1, result)

    def test_st_sim_wrapper__sample_signals(self):
        sample_reference_signal = np.ones(1000) * 1000 + np.random.randn(1000)
        sample_distorted_signal = sample_reference_signal + np.random.randn(1000)

        result = st_sim(sample_distorted_signal, sample_reference_signal)

        self.assertAlmostEqual(0.85, result, delta=0.1)

    def test_compute_block_sim__one_block_zero__zero(self):
        ref_block = np.ones(4)
        dist_block = np.zeros(4)
        result = STSIM.compute_block_sim(ref_block, dist_block)
        self.assertEqual(0, result)

    def test_compute_block_sim__blocks_identical__one(self):
        block = np.array([1, 2, 3, 4])
        result = STSIM.compute_block_sim(block, block)
        self.assertEqual(1, result)
