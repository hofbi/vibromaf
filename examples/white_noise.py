"""Simple get started example with white noise signals"""

import sys
from pathlib import Path

import numpy as np

# This is to make the local vibromaf package available
try:
    sys.path.append(str(Path(__file__).absolute().parents[1]))
except IndexError:
    pass

from vibromaf.metrics.snr import snr
from vibromaf.metrics.spqi import spqi
from vibromaf.metrics.stsim import st_sim

# Define sample signals
sample_reference_signal = np.ones(1000) * 1000 + np.random.randn(1000)
sample_distorted_signal = sample_reference_signal + np.random.randn(1000)

# Calculate metric scores
snr_score = snr(sample_distorted_signal, sample_reference_signal)
st_sim_score = st_sim(sample_distorted_signal, sample_reference_signal)
spqi_score = spqi(sample_distorted_signal, sample_reference_signal)

# Print individual metric scores
print(f"SNR score:    {snr_score}")
print(f"ST-SIM score: {st_sim_score}")
print(f"SPQI score:   {spqi_score}")
