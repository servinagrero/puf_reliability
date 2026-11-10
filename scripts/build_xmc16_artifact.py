import argparse
import re
from pathlib import Path

import h5py
import numpy as np

CYCLE_RE = re.compile(r"board([0-9A-Fa-f]{4})cycle(\d{4})")
N_SAMPLES = 101
RAW_N_BYTES = 160 * 1024
N_BITS = 256
N_SELECTED_BYTES = N_BITS // 8


def normal_cycle_files(board_dir: Path) -> list[Path]:
    indexed_paths = []
    for path in board_dir.iterdir():
        match = CYCLE_RE.fullmatch(path.name)
        if match is not None:
            indexed_paths.append((int(match.group(2)), path))
    indexed_paths.sort()

    cycles = [cycle for cycle, _ in indexed_paths]
    if cycles != list(range(1, N_SAMPLES + 1)):
        raise ValueError(
            f"{board_dir}: expected normal cycles 1..{N_SAMPLES}, got {cycles}"
        )
    return [path for _, path in indexed_paths]


def read_response(path: Path) -> np.ndarray:
    if path.stat().st_size != RAW_N_BYTES:
        raise ValueError(
            f"{path}: expected {RAW_N_BYTES} bytes, got {path.stat().st_size}"
        )
    packed = np.fromfile(path, dtype=np.uint8, count=N_SELECTED_BYTES)
    return np.unpackbits(packed, bitorder="big")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing board<id> directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/XMC16_dataset.h5"),
    )
    args = parser.parse_args()

    board_dirs = sorted(
        path for path in args.input_dir.glob("board*") if path.is_dir()
    )
    if not board_dirs:
        raise FileNotFoundError(
            f"No XMC board directories found in {args.input_dir}"
        )

    files_by_board = [normal_cycle_files(board_dir) for board_dir in board_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with h5py.File(temporary_path, "w") as h5:
        responses_out = h5.create_dataset(
            "crps",
            shape=(len(board_dirs), N_SAMPLES, N_BITS),
            dtype=np.uint8,
            chunks=(1, N_SAMPLES, N_BITS),
            compression="gzip",
        )
        reference_out = h5.create_dataset(
            "reference",
            shape=(len(board_dirs), N_BITS),
            dtype=np.uint8,
            compression="gzip",
        )

        for device_index, paths in enumerate(files_by_board):
            responses = np.stack([read_response(path) for path in paths])
            responses_out[device_index] = responses
            reference_out[device_index] = responses.mean(axis=0) >= 0.5

        h5.create_dataset(
            "device_ids",
            data=np.asarray(
                [path.name.removeprefix("board") for path in board_dirs],
                dtype="S4",
            ),
        )
        h5.attrs["n_devices"] = len(board_dirs)
        h5.attrs["n_samples"] = N_SAMPLES
        h5.attrs["n_raw_bits"] = RAW_N_BYTES * 8
        h5.attrs["n_bits"] = N_BITS
        h5.attrs["selected_bit_start"] = 0
        h5.attrs["bit_order"] = "big"
        h5.attrs["environment"] = "normal"
        h5.attrs["excluded_file_prefix"] = "heat"

    temporary_path.replace(args.output)
    print(f"Wrote {args.output} with {len(board_dirs)} devices")


if __name__ == "__main__":
    main()
