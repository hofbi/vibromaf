"""Perception module tests."""

import unittest

from vibromaf.signal.perception import PerceptualThreshold


class PerceptualThresholdTest(unittest.TestCase):
    """Perceptual Threshold Test."""

    def test_perceptual_threshold__block_length_ten__output_array_length_ten(self):
        block_length = 10
        sampling_frequency = 0
        result = PerceptualThreshold(sampling_frequency).calculate(block_length)
        self.assertEqual(10, result.size)

    def test_perceptual_threshold__block_length_10_frequency_2800__limit_cut_off_correctly(
        self,
    ):
        block_length = 10
        sampling_frequency = 2800
        result = PerceptualThreshold(sampling_frequency).calculate(block_length)
        self.assertListEqual([22.253170255179413] * 4, list(result[-4:]))

    def test_perceptual_threshold__block_length_10_frequency_0__should_be_constant_array(
        self,
    ):
        block_length = 10
        sampling_frequency = 0
        result = PerceptualThreshold(sampling_frequency).calculate(block_length)
        self.assertListEqual([-15.0] * 10, list(result))
