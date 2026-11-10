import argparse
import re
from pathlib import Path

import h5py
import numpy as np

DEVICE_RE = re.compile(r"chip_formatted_(\d+)\.csv")
N_OSCILLATORS = 512
N_SAMPLES = 100
N_BITS = N_OSCILLATORS // 2


def device_id(path: Path) -> int:
    match = DEVICE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected Maiti filename: {path.name}")
    return int(match.group(1))


def read_frequencies(path: Path) -> np.ndarray:
    frequencies = np.loadtxt(
        path,
        delimiter=",",
        usecols=range(N_SAMPLES),
        dtype=np.int32,
    )
    if frequencies.shape != (N_OSCILLATORS, N_SAMPLES):
        raise ValueError(
            f"{path}: expected {(N_OSCILLATORS, N_SAMPLES)}, "
            f"got {frequencies.shape}"
        )
    return frequencies.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing chip_formatted_<device>.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Maiti_dataset.h5"),
    )
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("chip_formatted_*.csv"), key=device_id)
    if not paths:
        raise FileNotFoundError(f"No Maiti CSV files found in {args.input_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with h5py.File(temporary_path, "w") as h5:
        frequencies_out = h5.create_dataset(
            "frequencies",
            shape=(len(paths), N_SAMPLES, N_OSCILLATORS),
            dtype=np.int32,
            chunks=(1, N_SAMPLES, N_OSCILLATORS),
            compression="gzip",
        )
        responses_out = h5.create_dataset(
            "crps",
            shape=(len(paths), N_SAMPLES, N_BITS),
            dtype=np.uint8,
            chunks=(1, N_SAMPLES, N_BITS),
            compression="gzip",
        )
        reference_out = h5.create_dataset(
            "reference",
            shape=(len(paths), N_BITS),
            dtype=np.uint8,
            compression="gzip",
        )

        for index, path in enumerate(paths):
            frequencies = read_frequencies(path)
            responses = (frequencies[:, 0::2] > frequencies[:, 1::2]).astype(
                np.uint8
            )
            frequencies_out[index] = frequencies
            responses_out[index] = responses
            reference_out[index] = responses.mean(axis=0) >= 0.5

        h5.create_dataset(
            "device_ids",
            data=np.asarray([device_id(path) for path in paths], dtype=np.int32),
        )
        h5.attrs["n_devices"] = len(paths)
        h5.attrs["n_samples"] = N_SAMPLES
        h5.attrs["n_oscillators"] = N_OSCILLATORS
        h5.attrs["n_bits"] = N_BITS
        h5.attrs["pairing"] = "disjoint_adjacent_oscillators"

    temporary_path.replace(args.output)
    print(f"Wrote {args.output} with {len(paths)} devices")


if __name__ == "__main__":
    main()
