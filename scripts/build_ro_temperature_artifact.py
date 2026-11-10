import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import polars as pl

N_BITS = 256
N_SAMPLES = 100

DIR_RE = re.compile(r"or_lut1_en_uni_rep_t(\d+)")
TEMP_RE = re.compile(r"RO_PUF_or_lut1_en_uni_rep_t(\d+)_(\w+).csv")
REF_RE = re.compile(r"RO_PUF_or_lut1_en_uni_rep_(\w+).csv")

def parse_csv(file):
    df = pl.read_csv(file)
    expected_rows = N_BITS * N_SAMPLES
    if len(df) != expected_rows:
        raise ValueError(f"{file}: expected {expected_rows} rows, got {len(df)}")
    bits = df["CRP"].to_numpy().reshape(N_SAMPLES, N_BITS)
    count_fst = df["freq1"].to_numpy().reshape(N_SAMPLES, N_BITS)
    count_snd = df["freq2"].to_numpy().reshape(N_SAMPLES, N_BITS)
    return bits, count_fst, count_snd


def parse_temperature_dir(directory):
    data_by_uid = {}
    for csv in sorted(directory.iterdir()):
        if csv.suffix != ".csv":
            continue
        match = TEMP_RE.fullmatch(csv.name)
        if match is None:
            raise ValueError(f"Unexpected temperature filename: {csv.name}")
        uid = match.group(2)
        bits, count_fst, count_snd = parse_csv(csv)
        data_by_uid[uid] = {
            "crp": bits.astype(np.uint8),
            "freq1": count_fst.astype(np.int32),
            "freq2": count_snd.astype(np.int32),
        }
    return data_by_uid


def parse_reference(reference_dir):
    data_by_uid = {}
    for csv in sorted(
        csv for csv in reference_dir.iterdir() if csv.suffix == ".csv"
    ):
        match = REF_RE.fullmatch(csv.name)
        if match is None:
            raise ValueError(f"Unexpected reference filename: {csv.name}")
        uid = match.group(1)
        bits, count_fst, count_snd = parse_csv(csv)
        data_by_uid[uid] = {
            "crp": bits.astype(np.uint8),
            "freq1": count_fst.astype(np.int32),
            "freq2": count_snd.astype(np.int32),
        }
    return data_by_uid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory containing room-temperature reference CSV files",
    )
    parser.add_argument(
        "--temperature-dir",
        type=Path,
        required=True,
        help="Directory containing or_lut1_en_uni_rep_t<T> directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/RO_PUF.h5"),
    )
    args = parser.parse_args()

    reference_by_uid = parse_reference(args.reference_dir)
    if not reference_by_uid:
        raise FileNotFoundError(
            f"No reference CSV files found in {args.reference_dir}"
        )

    temperature_dirs = sorted(
        directory
        for directory in args.temperature_dir.iterdir()
        if directory.is_dir()
    )
    if not temperature_dirs:
        raise FileNotFoundError(
            f"No temperature directories found in {args.temperature_dir}"
        )
    temperature_data = {}

    common_uids = set(reference_by_uid)
    for directory in temperature_dirs:
        match = DIR_RE.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"Unexpected temperature directory: {directory.name}")
        temp = int(match.group(1))
        data_by_uid = parse_temperature_dir(directory)
        missing = sorted(set(reference_by_uid) - set(data_by_uid))
        if missing:
            print(f"t={temp}: removing missing reference UIDs {missing}")
        common_uids &= set(data_by_uid)
        temperature_data[temp] = data_by_uid

    uids = sorted(common_uids)
    if not uids:
        raise ValueError("No device UID is common to all temperature datasets")
    reference = stack_uid_field(reference_by_uid, uids, "crp")
    reference_freq1 = stack_uid_field(reference_by_uid, uids, "freq1")
    reference_freq2 = stack_uid_field(reference_by_uid, uids, "freq2")
    reference_majority = (reference.mean(axis=1) >= 0.5).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with h5py.File(temporary_path, "w") as h5:
        h5.create_dataset("uids", data=np.asarray(uids, dtype="S"))
        h5.create_dataset("reference", data=reference, compression="gzip")
        h5.create_dataset("reference_freq1", data=reference_freq1, compression="gzip")
        h5.create_dataset("reference_freq2", data=reference_freq2, compression="gzip")
        h5.create_dataset(
            "reference_majority",
            data=reference_majority,
            compression="gzip",
        )
        h5.attrs["n_devices"] = len(uids)
        h5.attrs["n_samples"] = N_SAMPLES
        h5.attrs["n_bits"] = N_BITS
        h5.attrs["missing_reference_uids"] = ",".join(
            sorted(set(reference_by_uid) - set(uids))
        )

        counts = h5.create_group("counts")
        freq1_group = counts.create_group("freq1")
        freq2_group = counts.create_group("freq2")
        for temp, crps_by_uid in sorted(temperature_data.items()):
            crps = stack_uid_field(crps_by_uid, uids, "crp")
            freq1 = stack_uid_field(crps_by_uid, uids, "freq1")
            freq2 = stack_uid_field(crps_by_uid, uids, "freq2")
            h5.create_dataset(str(temp), data=crps, compression="gzip")
            freq1_group.create_dataset(str(temp), data=freq1, compression="gzip")
            freq2_group.create_dataset(str(temp), data=freq2, compression="gzip")

    temporary_path.replace(args.output)
    print(f"Wrote {args.output} with {len(uids)} devices")


def stack_uid_field(data_by_uid, uids, field):
    return np.stack([data_by_uid[uid][field] for uid in uids], axis=0)


if __name__ == "__main__":
    main()
