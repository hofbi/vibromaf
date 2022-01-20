"""Spectral Perceptual Quality Index"""

from dataclasses import dataclass

import numpy as np

from vibromaf.signal.spectrum import compute_normalized_spectral_difference
from vibromaf.signal.transform import PerceptualSpectrumBuilder


def spqi(
    distorted: np.array, reference: np.array, eta: float = 0.3, threshold: float = -2.0
) -> float:
    """Wrapper function to calculate the SPQI score

    Parameters
    ------
    * `distorted: np.array` Distorted vibrotactile signal.
    * `reference: np.array` Reference vibrotactile signal.
    * `eta: float` Slope of the mapping function between error and score.
    * `threshold: float` Offset of the mapping function.

    Returns
    -------
    * `float` The SPQI score.
    """
    metric = SPQI(eta, threshold)
    return metric.calculate(distorted, reference)


@dataclass(frozen=True)
class SPQI:
    """Spectral Perceptual Quality Index"""

    eta: float
    threshold: float
    perceptual_spectrum_builder = PerceptualSpectrumBuilder()

    def calculate(self, distorted: np.array, reference: np.array) -> float:
        if np.array_equal(distorted, reference):
            return 1

        ref_perceptual_spectrum = (
            self.perceptual_spectrum_builder.compute_perceptual_spectrum(reference)
        )
        dist_perceptual_spectrum = (
            self.perceptual_spectrum_builder.compute_perceptual_spectrum(distorted)
        )

        norm_perceptual_difference = compute_normalized_spectral_difference(
            ref_perceptual_spectrum, dist_perceptual_spectrum
        )

        block_spqi_scores = self.__compute_block_spqi(norm_perceptual_difference)

        return np.mean(block_spqi_scores)

    def __compute_block_spqi(self, normalized_perceptual_difference: np.array) -> float:
        return (
            1 - np.tanh(self.eta * normalized_perceptual_difference - self.threshold)
        ) / 2
