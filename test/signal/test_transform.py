"""Transform module tests"""

import unittest

import numpy as np

from vibromaf.signal.perception import PerceptualThreshold
from vibromaf.signal.transform import (
    BlockBuilder,
    PerceptualSpectrumBuilder,
    compute_block_dct,
    compute_block_dft,
    cut_off_strategy,
    preprocess_input_signal,
    zero_padding_strategy,
)


class TransformTest(unittest.TestCase):
    """Transform Test"""

    def test_block_dft__ones_block__constant_block(self):
        input_block = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = compute_block_dft(input_block)
        self.assertListEqual([-10] * 10, list(result))

    def test_block_dft__block_with_legth_10__block_with_same_length(self):
        input_block = np.linspace(1, 1, 10)
        result = compute_block_dft(input_block)
        self.assertEqual(10, np.size(result))

    def test_block_dft__parseval_theorem(self):
        input_block = np.linspace(1, 1, 16)
        spectrum = compute_block_dft(input_block)
        spectrum = np.power(10, spectrum / 20)
        result = np.sum(np.power(spectrum, 2))
        self.assertAlmostEqual(24, float(result), delta=0.05)

    def test_block_dct__block_of_length_10__block_of_length_10(self):
        input_block = np.linspace(1, 1, 10)
        result = compute_block_dct(input_block)
        self.assertEqual(10, np.size(result))

    def test_block_dct__ones_block_of_length_10__block_with_2_and_zeros(self):
        input_block = np.ones(10)
        result = compute_block_dct(input_block)
        self.assertListEqual([-np.inf] * 9, list(result[1:]))
        self.assertAlmostEqual(10, result[0])

    def test_cut_off_strategy__signal_more_than_block_length__cut_off(self):
        input_signal = np.array([1, 2, 3, 4, 5])
        result = cut_off_strategy(input_signal, 3)
        self.assertListEqual([1, 2, 3], list(result))

    def test_cut_off_strategy__signal_more_than_multiple_block_length__cut_off(self):
        input_signal = np.array([1, 2, 3, 4, 5])
        result = cut_off_strategy(input_signal, 2)
        self.assertListEqual([1, 2, 3, 4], list(result))

    def test_cut_off_strategy__signal_smaller_than_block_length__empty(self):
        input_signal = np.array([1, 2])
        result = cut_off_strategy(input_signal, 3)
        self.assertListEqual([], list(result))

    def test_cut_off_strategy__signal_equal_to_block_length__unchanged(self):
        input_signal = np.array([1, 2, 3])
        result = cut_off_strategy(input_signal, 3)
        self.assertListEqual([1, 2, 3], list(result))

    def test_zero_padding_strategy__signal_more_than_block_length__fill_with_zeros(
        self,
    ):
        input_signal = np.array([1, 2, 3, 4])
        result = zero_padding_strategy(input_signal, 3)
        self.assertListEqual([1, 2, 3, 4, 0, 0], list(result))

    def test_zero_padding_strategy__signal_smaller_than_block_length__zeros_only(
        self,
    ):
        input_signal = np.array([1, 2])
        result = zero_padding_strategy(input_signal, 3)
        self.assertListEqual([1, 2, 0], list(result))

    def test_zero_padding_strategy__signal_equal_to_block_length__unchanged(
        self,
    ):
        input_signal = np.array([1, 2, 3])
        result = zero_padding_strategy(input_signal, 3)
        self.assertListEqual([1, 2, 3], list(result))

    def test_preprocess_input_signal__dist_larger_than_ref__dist_should_be_truncated_and_warn(
        self,
    ):
        signal = np.array([0, 1])
        dist = np.array([0, 1, 2])
        with self.assertWarnsRegex(
            RuntimeWarning,
            r"Truncating distorted signal .* since longer than reference",
        ):
            result = preprocess_input_signal(dist, signal)
        self.assertListEqual(list(signal), list(result))

    def test_preprocess_input_signal__dist_shorter_than_ref__should_throw(self):
        signal = np.array([0, 1, 2])
        dist = np.array([0, 1])
        with self.assertRaisesRegex(ValueError, r"Distorted .* must not be shorter"):
            preprocess_input_signal(dist, signal)


class BlockBuilderTest(unittest.TestCase):
    """Block Builder Test"""

    def test_divide__empty_array__should_throw(self):
        unit = BlockBuilder(10)
        with self.assertRaises(ValueError):
            unit.divide(np.array([]))

    def test_divide__array_shorter_than_block_length__should_throw(self):
        unit = BlockBuilder(10)
        with self.assertRaises(ValueError):
            unit.divide(np.array([1, 2, 3]))

    def test_divide__perfect_matching_array__perfect_division(self):
        input_signal = np.array([1, 2, 3, 4, 5, 6])
        unit = BlockBuilder(2)
        result = unit.divide(input_signal)
        self.assertListEqual([1, 2], list(result[0]))
        self.assertListEqual([3, 4], list(result[1]))
        self.assertListEqual([5, 6], list(result[2]))

    def test_divide__perfect_not_matching_array__division_truncated(self):
        input_signal = np.array([1, 2, 3, 4, 5])
        unit = BlockBuilder(2)
        result = unit.divide(input_signal)
        self.assertListEqual([1, 2], list(result[0]))
        self.assertListEqual([3, 4], list(result[1]))
        self.assertEqual((2, 2), result.shape)

    def test_divide_and_normalize__periodic_array__array_divided_by_two(self):
        input_signal = np.array([-2, 2, -2, 2])
        unit = BlockBuilder(4)
        result = unit.divide_and_normalize(input_signal)
        self.assertListEqual([-1, 1, -1, 1], list(result[0]))

    def test_divide_and_normalize__periodic_array__array_divided_by_two_and_shifted(
        self,
    ):
        input_signal = np.array([0, 4, 0, 4])
        unit = BlockBuilder(4)
        result = unit.divide_and_normalize(input_signal)
        self.assertListEqual([-1, 1, -1, 1], list(result[0]))

    def test_divide_and_normalize__multiple_blocks__correct_reshaped(self):
        input_signal = np.array([-2, 2, -2, 2, -2, 2, -2, 2])
        unit = BlockBuilder(2)
        result = unit.divide_and_normalize(input_signal)
        self.assertTrue(
            np.array_equal(np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]]), result)
        )

    def test_divide_and_normalize__input_signals_identical_zero__should_be_not_nan(
        self,
    ):
        input_signal = np.zeros(10)
        unit = BlockBuilder(5)
        result = unit.divide_and_normalize(input_signal)
        self.assertFalse(np.isnan(result).any())


class PerceptualSpectrumBuilderTest(unittest.TestCase):
    """Perceptual Spectrum Builder Test"""

    def test_compute_perceptual_spectrum__signal_length_equals_block_length__signal_shape_unchanged(
        self,
    ):
        input_signal = np.array([1, 2, 3, 4])
        unit = PerceptualSpectrumBuilder(BlockBuilder(4))
        result = unit.compute_perceptual_spectrum(input_signal)
        self.assertEqual(input_signal.size, result.size)
        self.assertEqual(result.shape, (1, 4))

    def test_compute_perceptual_spectrum__signal_length_equals_twice_block_length__signal_shape_changed(
        self,
    ):
        input_signal = np.array([1, 2, 3, 4])
        unit = PerceptualSpectrumBuilder(BlockBuilder(2))
        result = unit.compute_perceptual_spectrum(input_signal)
        self.assertEqual(input_signal.size, result.size)
        self.assertEqual(result.shape, (2, 2))

    def test_compute_perceptual_spectrum__signal_with_ones__block_wise_dct(self):
        input_signal = np.array([1, 1])
        threshold = PerceptualThreshold(0)

        unit = PerceptualSpectrumBuilder(
            block_builder=BlockBuilder(2),
            block_transform_strategy=compute_block_dct,
            perceptual_threshold=threshold,
        )
        result = unit.compute_perceptual_spectrum(input_signal)

        result += threshold.calculate(2)
        self.assertAlmostEqual(3.01, result[0, 0], delta=0.001)  # sqrt(2) in dB
        self.assertEqual(-np.inf, result[0, 1])
