"""Spectral Temporal SIMilarity."""

from dataclasses import dataclass

import numpy as np

from vibromaf.signal.spectrum import compute_spectral_support
from vibromaf.signal.transform import PerceptualSpectrumBuilder, preprocess_input_signal


def st_sim(distorted: np.array, reference: np.array, eta: float = 2 / 3) -> float:
    """Wrapper function to calculate the ST-SIM score.

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
    """Spectral Temporal SIMilarity."""

    eta: float
    perceptual_spectrum_builder = PerceptualSpectrumBuilder()

    def calculate(self, distorted: np.array, reference: np.array) -> float:
        distorted = preprocess_input_signal(distorted, reference)
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
    def compute_sim(reference: np.array, distorted: np.array) -> float:
        return float(
            np.mean(
                np.sum(reference * distorted, axis=1)
                / (np.sum(np.power(reference, 2), axis=1) + np.finfo(float).eps)
            )
        )

    def __post_init__(self):
        if not 0.0 < self.eta < 1.0:  # noqa: PLR2004
            raise ValueError("Eta must be between 0 and 1.")
