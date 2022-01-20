"""Spectral Temporal SIMilarity"""

from dataclasses import dataclass

import numpy as np

from vibromaf.signal.spectrum import compute_spectral_support
from vibromaf.signal.transform import PerceptualSpectrumBuilder


def st_sim(distorted: np.array, reference: np.array, eta: float = 2 / 3) -> float:
    """Wrapper function to calculate the ST-SIM score

    Parameters
    ------
    * `distorted: np.array` Distorted vibrotactile signal.
    * `reference: np.array` Reference vibrotactile signal.
    * `eta: float` Importance of temporal component compared to spectral component. Should be between 0 and 1.

    Returns
    -------
    * `float` The ST-SIM score.
    """
    metric = STSIM(eta)
    return metric.calculate(distorted, reference)


@dataclass(frozen=True)
class STSIM:
    """Spectral Temporal SIMilarity"""

    eta: float
    perceptual_spectrum_builder = PerceptualSpectrumBuilder()

    def calculate(self, distorted: np.array, reference: np.array) -> float:
        if np.array_equal(distorted, reference):
            return 1

        ref_spectral_support = compute_spectral_support(
            self.perceptual_spectrum_builder.compute_perceptual_spectrum(reference)
        )
        dist_spectral_support = compute_spectral_support(
            self.perceptual_spectrum_builder.compute_perceptual_spectrum(distorted)
        )
        spectral_sim = STSIM.compute_sim(ref_spectral_support, dist_spectral_support)

        ref_normalized_blocks = (
            self.perceptual_spectrum_builder.block_builder.divide_and_normalize(
                reference
            )
        )
        dist_normalized_blocks = (
            self.perceptual_spectrum_builder.block_builder.divide_and_normalize(
                distorted
            )
        )
        temporal_sim = STSIM.compute_sim(ref_normalized_blocks, dist_normalized_blocks)

        return pow(temporal_sim, self.eta) * pow(spectral_sim, 1 - self.eta)

    @staticmethod
    def compute_block_sim(
        reference_block: np.array, distorted_block: np.array
    ) -> float:
        return np.sum(reference_block * distorted_block) / np.sum(
            np.power(reference_block, 2)
        )

    @staticmethod
    def compute_sim(reference: np.array, distorted: np.array) -> float:
        block_sim = np.apply_along_axis(
            STSIM.compute_block_sim, 1, reference, distorted
        )
        return np.mean(block_sim)
