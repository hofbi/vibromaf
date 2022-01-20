"""Vibrotactile Multi-Method Assessment Fusion"""

from pathlib import Path

import numpy as np

from vibromaf.metrics.snr import nsnr
from vibromaf.metrics.spqi import SPQI
from vibromaf.metrics.stsim import STSIM
from vibromaf.util import model


def vibro_maf(distorted: np.array, reference: np.array, model_path: Path) -> float:
    """Wrapper function to calculate the SPQI score"""
    metric = VibroMAF(model_path, SPQI(0.3, -2.0), STSIM(2 / 3))
    return metric.calculate(distorted, reference)


class VibroMAF:
    """Vibrotactile Multi-Method Assessment Fusion"""

    def __init__(self, model_path: Path, spqi: SPQI, st_sim: STSIM) -> None:
        self.__model = model.load_model(model_path)
        self.__spqi = spqi
        self.__st_sim = st_sim

    def calculate(self, distorted: np.array, reference: np.array) -> float:
        signal_to_noise_ratio = nsnr(distorted, reference)
        st_sim = self.__st_sim.calculate(distorted, reference)
        spqi = self.__spqi.calculate(distorted, reference)
        return float(self.__model.predict([[signal_to_noise_ratio, st_sim, spqi]]))
