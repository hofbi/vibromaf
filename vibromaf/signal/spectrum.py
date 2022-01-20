"""Spectrum module"""

import numpy as np
from scipy.stats import norm


def pow2db(power: np.array) -> np.array:
    """
    Convert power to decibels
    https://de.mathworks.com/help/signal/ref/pow2db.html
    """
    return 10.0 * np.log10(power)


def db2pow(decibel: np.array) -> np.array:
    """
    Convert decibel to power
    https://de.mathworks.com/help/signal/ref/db2pow.html
    """
    return np.power(10.0, decibel / 10.0)


def mag2db(power: np.array) -> np.array:
    """
    Convert magnitude to decibels
    https://de.mathworks.com/help/signal/ref/mag2db.html
    """
    return 2 * pow2db(power)


def signal_energy(signal: np.array) -> np.array:
    """Calculate the signal energy"""
    return np.sum(np.power(signal, 2))


def compute_normalized_spectral_difference(
    reference_spectrum: np.array, distorted_spectrum: np.array
) -> np.array:
    """Compute the normalized difference of two spectrums"""
    difference = np.sum(np.abs(db2pow(reference_spectrum) - db2pow(distorted_spectrum)))
    return pow2db(difference / np.sum(np.abs(db2pow(difference))))


def compute_spectral_support(spectrum: np.array, scale: float = 12) -> np.array:
    """Compute the spectral support of perceptual spectrum using a normal distribution cdf"""
    return np.apply_along_axis(norm.cdf, 1, spectrum, scale=scale)
