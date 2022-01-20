# Get Started

Create a sample signal or read one from your database.
Here, we create a constant signal of 1000 with length 1000 and add random white noise.
For the distorted signal, we add again random white noise to the reference signal.

```python
import numpy as np

sample_reference_signal = np.ones(1000) * 1000 + np.random.randn(1000)
sample_distorted_signal = sample_reference_signal + np.random.randn(1000)
```

## Calculate SNR

Calculate the [SNR](metrics/snr.md) from the signals defined above.

```python
from vibromaf.metrics.snr import snr

snr_score = snr(sample_distorted_signal, sample_reference_signal)

print(snr_score)  # Should be around 60dB
```

Find further details how to use this metric at [SNR](metrics/snr.md).

## Calculate ST-SIM

Calculate the [ST-SIM](metrics/stsim.md) from the signals defined above.

```python
from vibromaf.metrics.stsim import st_sim

st_sim_score = st_sim(sample_distorted_signal, sample_reference_signal)

print(st_sim_score)  # Should be around 0.85
```

Find further details how to use this metric at [ST-SIM](metrics/stsim.md).

## Calculate SPQI

Calculate the [SPQI](metrics/spqi.md) from the signals defined above.

```python
from vibromaf.metrics.spqi import spqi

spqi_score = spqi(sample_distorted_signal, sample_reference_signal)

print(spqi_score)  # Should be around 1.0
```

Find further details how to use this metric at [SPQI](metrics/spqi.md).
