"""Transform module"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.fftpack import dct

from vibromaf.signal.perception import PerceptualThreshold
from vibromaf.signal.spectrum import mag2db


def compute_block_dft(block: np.array) -> np.array:
    """Compute DFT spectrum using FFT"""
    block_length = np.size(block)
    extended_block = np.zeros(
        2 * block_length,
    )
    extended_block[0:block_length] = block
    spectrum = mag2db(abs(1 / math.sqrt(block_length) * np.fft.fft(extended_block)))[
        0:block_length
    ]
    return spectrum


def compute_block_dct(block: np.array) -> np.array:
    """Compute block transform using DCT"""
    return mag2db(abs(dct(block, norm="ortho")))


def cut_off_strategy(signal: np.array, block_length: int) -> np.array:
    """Prepare signal for block building by cutting off"""
    num_blocks = int(signal.size / block_length)
    return signal[: num_blocks * block_length]


def zero_padding_strategy(signal: np.array, block_length: int) -> np.array:
    """Prepare signal for block building by padding with zeros"""
    num_padding = block_length * math.ceil(signal.size / block_length) - signal.size
    return np.pad(signal, (0, num_padding), "constant")


@dataclass(frozen=True)
class BlockBuilder:
    """Split signal into blocks"""

    block_length: int
    truncation_strategy: Callable[[np.array], np.array] = cut_off_strategy

    def divide(self, signal: np.array) -> np.array:
        if signal.size < self.block_length:
            raise ValueError(
                "Signal is too short! The signal must be at least as long as the block length."
            )
        signal = self.truncation_strategy(signal, self.block_length)
        return np.reshape(signal, (-1, self.block_length))

    def divide_and_normalize(self, signal: np.array) -> np.array:
        blocks = self.divide(signal)
        means = np.apply_along_axis(np.mean, 1, blocks)
        stds = np.apply_along_axis(np.std, 1, blocks)
        return (blocks - means) / stds


@dataclass(frozen=True)
class PerceptualSpectrumBuilder:
    """Calculate perceptual spectrum"""

    block_builder: BlockBuilder = BlockBuilder(512)
    perceptual_threshold: PerceptualThreshold = PerceptualThreshold(2800)
    block_transform_strategy: Callable[[np.array], np.array] = compute_block_dct

    def compute_perceptual_spectrum(self, signal: np.array) -> np.array:
        blocks = self.block_builder.divide(signal)
        spectrum = np.apply_along_axis(self.block_transform_strategy, 1, blocks)
        return spectrum - self.perceptual_threshold.calculate(
            self.block_builder.block_length
        )
