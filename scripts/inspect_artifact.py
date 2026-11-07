import argparse
from pathlib import Path

import numpy as np

from pufre import DATA_DIR
from pufre.data import PUFObservations
from pufre.datasets import load_hdf5_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DATA_DIR / "puf_artifact.h5")
    args = parser.parse_args()

    families = load_hdf5_artifact(args.artifact)
    if not families:
        raise SystemExit(f"No families found in {args.artifact}")

    for family in families:
        observations = PUFObservations.from_family(family)
        flip_rate = observations.error_binary.mean(axis=(0, 1))
        per_device_flip_rate = observations.error_binary.mean(axis=1)
        variable_error_bits = np.sum((flip_rate > 0.0) & (flip_rate < 1.0))
        constant_error_bits = observations.n_bits - variable_error_bits
        unique_error_rows = np.unique(observations.flat_errors(), axis=0).shape[0]
        physical = observations.physical_bit_ids
        print(f"{family.family}")
        print(f"  responses: {family.responses.shape}")
        print(f"  reference: {family.reference.shape}")
        print(f"  local bit IDs: 0..{observations.n_bits - 1}")
        print(f"  physical bit IDs: {physical[:5]} ... {physical[-5:]}")
        print(f"  variable error bits: {variable_error_bits}/{observations.n_bits}")
        print(f"  constant error bits: {constant_error_bits}/{observations.n_bits}")
        print(
            f"  unique error rows: {unique_error_rows}/{observations.flat_errors().shape[0]}"
        )
        print(
            "  flip rate: "
            f"min={np.min(flip_rate):.4f}, "
            f"mean={np.mean(flip_rate):.4f}, "
            f"max={np.max(flip_rate):.4f}"
        )
        print(
            "  per-device flip rate: "
            f"min={np.min(per_device_flip_rate):.4f}, "
            f"mean={np.mean(per_device_flip_rate):.4f}, "
            f"max={np.max(per_device_flip_rate):.4f}"
        )


if __name__ == "__main__":
    main()
