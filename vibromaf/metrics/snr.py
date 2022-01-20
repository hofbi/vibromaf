"""Signal to Noise Ratio"""

import numpy as np

from vibromaf.signal.spectrum import pow2db, signal_energy


def snr(distorted: np.array, reference: np.array) -> float:
    """Calculate the signal-to-noise ratio

    Parameters
    ------
    * `distorted: np.array` Distorted signal.
    * `reference: np.array` Reference signal.

    Returns
    -------
    * `float` The SNR.
    """
    return pow2db(signal_energy(reference) / signal_energy(distorted - reference))


def nsnr(
    distorted: np.array, reference: np.array, normalization_db: float = 75
) -> float:
    """Calculate the normalized signal-to-noise ratio restricted to range of 0 to 1"""
    return max(0.0, min(1.0, snr(distorted, reference) / normalization_db))
