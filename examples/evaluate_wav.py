"""Example to evaluate WAV files"""

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
from pathlib import Path

from scipy.io import wavfile

# This is to make the local vibromaf package available
try:
    sys.path.append(str(Path(__file__).absolute().parents[1]))
except IndexError:
    pass

from vibromaf.metrics.snr import snr
from vibromaf.metrics.spqi import spqi
from vibromaf.metrics.stsim import st_sim


def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "distorted",
        type=FileType("r"),
        help="Distorted .wav file",
    )
    parser.add_argument(
        "reference",
        type=FileType("r"),
        help="Undistorted reference .wav file",
    )
    return parser.parse_args()


def main():
    """main"""
    args = parse_arguments()

    distorted_signal = wavfile.read(args.distorted.name)[1]
    reference_signal = wavfile.read(args.reference.name)[1]

    # Calculate metric scores
    snr_score = snr(distorted_signal, reference_signal)
    st_sim_score = st_sim(distorted_signal, reference_signal)
    spqi_score = spqi(distorted_signal, reference_signal)

    # Print individual metric scores
    print(f"SNR score:    {snr_score}")
    print(f"ST-SIM score: {st_sim_score}")
    print(f"SPQI score:   {spqi_score}")


if __name__ == "__main__":
    main()
