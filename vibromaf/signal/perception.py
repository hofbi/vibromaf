"""Module human vibrotactile perception"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerceptualThreshold:
    """Human perceptual threshold"""

    sampling_frequency: int
    x_scaler: float = 62
    y_scaler: float = 1 / 550
    x_offset: float = 1 - 250 * y_scaler
    y_offset: float = 77

    def calculate(self, block_length: int) -> np.array:
        frequency_grid = np.linspace(0, self.sampling_frequency, 2 * block_length)
        frequency_grid = frequency_grid[0:block_length]

        perceptual_threshold = (
            abs(
                self.x_scaler
                / pow((math.log10(self.x_offset)), 2)
                * np.power(
                    (np.log10(self.y_scaler * frequency_grid + self.x_offset)), 2
                )
            )
            - self.y_offset
        )
        limit = np.argmax(perceptual_threshold > 0)
        perceptual_threshold[limit:] = perceptual_threshold[limit]
        return perceptual_threshold
