import argparse
from functools import partial
from pathlib import Path

import numpy as np

from pufre import DATA_DIR
from pufre.datasets import (
    DatasetConfig,
    generate_gen_ideal,
    load_differential_room,
    load_ro_maiti,
    load_ro_fraunhofer,
    load_ro_infineon,
    load_sram_stm32,
    load_sram_verona,
    load_sram_xmc16,
    save_hdf5_artifact,
)

DATASETS = [
    DatasetConfig(
        family="GEN_Ideal",
        path=None,
        loader=generate_gen_ideal,
        enabled=True,
    ),
    DatasetConfig(
        family="SRAM_ESP32_freezer",
        path=DATA_DIR / "ESP32_dataset.h5",
        loader=partial(load_sram_verona, environment="freezer"),
        enabled=True,
    ),
    DatasetConfig(
        family="SRAM_ESP32_roomT",
        path=DATA_DIR / "ESP32_dataset.h5",
        loader=partial(load_sram_verona, environment="roomT"),
        enabled=True,
    ),
    DatasetConfig(
        family="SRAM_ESP32_hot",
        path=DATA_DIR / "ESP32_dataset.h5",
        loader=partial(load_sram_verona, environment="hot"),
        enabled=True,
    ),
    DatasetConfig(
        family="SRAM_STM32",
        path=DATA_DIR / "STM32_dataset.h5",
        loader=load_sram_stm32,
        enabled=True,
    ),
    DatasetConfig(
        family="RO_Fraunhofer_LEFT_LOWER",
        path=DATA_DIR / "Fraunhofer_dataset.h5",
        loader=load_ro_fraunhofer,
        enabled=True,
    ),
    DatasetConfig(
        family="RO_Infineon",
        path=DATA_DIR / "Infineon_dataset.h5",
        loader=load_ro_infineon,
        enabled=True,
    ),
    DatasetConfig(
        family="RO_Spartan7_roomT",
        path=DATA_DIR / "raw" / "RO_PUF.h5",
        loader=load_differential_room,
        enabled=True,
    ),
    DatasetConfig(
        family="RO_Maiti_Spartan3E",
        path=DATA_DIR / "Maiti_dataset.h5",
        loader=load_ro_maiti,
        enabled=True,
    ),
    DatasetConfig(
        family="SRAM_XMC4500",
        path=DATA_DIR / "XMC16_dataset.h5",
        loader=load_sram_xmc16,
        enabled=True,
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR / "puf_artifact.h5")
    parser.add_argument("--n-bits", type=int, default=256)
    parser.add_argument("--bit-selection", choices=["best", "chunk"], default="chunk")
    parser.add_argument("--only-generated", action="store_true")
    args = parser.parse_args()

    configs = DATASETS
    if args.only_generated:
        configs = [config for config in DATASETS if config.path is None]

    missing = []
    for config in configs:
        if not config.enabled or config.path is None:
            continue
        if not config.path.exists():
            missing.append(config.path)

    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Refusing to build the artifact because enabled datasets are missing:\n"
            f"{missing_list}"
        )

    families = []
    for config in configs:
        if not config.enabled:
            continue
        family_data = config.loader(
            config.family,
            config.path,
            n_bits=args.n_bits,
            bit_selection=args.bit_selection,
        )
        _validate_family_data(family_data, expected_bits=args.n_bits)
        print(
            f"{family_data.family}: responses={family_data.responses.shape}, "
            f"reference={family_data.reference.shape}"
        )
        families.append(family_data)

    if not families:
        raise SystemExit("No datasets were loaded. Edit DATASETS in this script.")

    save_hdf5_artifact(args.output, families)
    print(f"Wrote {args.output}")


def _validate_family_data(family_data, *, expected_bits: int):
    responses = family_data.responses
    reference = family_data.reference
    if responses.ndim != 3:
        raise ValueError(f"{family_data.family}: responses must be 3D")
    if reference.shape != (responses.shape[0], responses.shape[2]):
        raise ValueError(
            f"{family_data.family}: reference shape {reference.shape} does not "
            f"match response shape {responses.shape}"
        )
    if responses.shape[2] != expected_bits:
        raise ValueError(
            f"{family_data.family}: expected {expected_bits} bits, got "
            f"{responses.shape[2]}"
        )
    if not np.all(np.isin(responses, [0, 1])):
        raise ValueError(f"{family_data.family}: responses must be binary 0/1")
    if not np.all(np.isin(reference, [0, 1])):
        raise ValueError(f"{family_data.family}: reference must be binary 0/1")


if __name__ == "__main__":
    main()
