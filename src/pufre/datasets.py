from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import numpy.typing as npt

from pufre.metrics import reliable_entropy_score


@dataclass(frozen=True)
class FamilyResponses:
    """Responses and references for one PUF family or condition."""

    family: str
    responses: npt.NDArray
    reference: npt.NDArray
    labels: npt.NDArray
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class DatasetConfig:
    family: str
    path: Optional[Path]
    loader: Callable[..., FamilyResponses]
    enabled: bool = True


def default_sample_device_labels(family: str, responses: npt.NDArray):
    """Create per-row labels for flattened ``(device, sample, bit)`` responses."""
    n_devices, n_samples, _ = responses.shape
    device = np.repeat(np.arange(n_devices), n_samples)
    return np.array([f"{family}:device-{idx}" for idx in device], dtype=object)


def majority_reference(responses: npt.NDArray):
    """Binary majority reference for responses shaped ``(device, sample, bit)``."""
    return np.where(np.mean(responses, axis=1) >= 0.5, 1, 0).astype(np.uint8)


def hdf5_rows_in_order(dataset: h5py.Dataset, indices: npt.NDArray):
    """Read HDF5 rows while preserving the order supplied by the caller."""
    indices = np.asarray(indices, dtype=int)
    order = np.argsort(indices)
    sorted_rows = dataset[indices[order]]
    rows = np.empty_like(sorted_rows)
    rows[order] = sorted_rows
    return rows


def normalize_binary_crps(
    responses: npt.NDArray,
    *,
    dataset_name: str,
    bit_axis: Optional[int] = None,
):
    """
    Return binary CRPs with canonical shape ``(device, sample, bit)``.

    SRAM HDF5 files commonly store arrays as ``(device, bit, repeat)`` while the
    modeling code expects ``(device, repeat, bit)``. If ``bit_axis`` is omitted,
    the largest non-device axis is treated as the bit axis.
    """
    responses = np.asarray(responses, dtype=np.uint8)
    if responses.ndim != 3:
        raise ValueError(f"{dataset_name} CRPs must be a 3D array")

    if bit_axis is None:
        bit_axis = 1 if responses.shape[1] >= responses.shape[2] else 2
    if bit_axis not in (1, 2):
        raise ValueError("bit_axis must be 1 or 2 for 3D CRPs")

    if bit_axis == 1:
        responses = np.transpose(responses, (0, 2, 1))
    return responses


def best_bits_by_rel_entropy(
    reference: npt.NDArray,
    responses: npt.NDArray,
    *,
    n_bits: int = 256,
):
    """Return bit indices with highest Reliable Entropy score."""
    scores = reliable_entropy_score(responses, reference)
    if scores.size < n_bits:
        raise ValueError(
            f"cannot select {n_bits} bits from only {scores.size} available bits; "
            "check the input tensor axes"
        )
    n_select = n_bits
    selected = np.argsort(scores)[-n_select:]
    selected.sort()
    return selected


def select_response_bits(
    family: str,
    responses: npt.NDArray,
    reference: npt.NDArray,
    *,
    n_bits: int = 256,
    mode: str = "best",
    chunk_index: int = 0,
    original_bit_ids: Optional[npt.NDArray] = None,
):
    """Select either best Reliable Entropy bits or a raw contiguous chunk."""
    if original_bit_ids is None:
        original_bit_ids = np.arange(responses.shape[2])
    else:
        original_bit_ids = np.asarray(original_bit_ids)

    if mode == "best":
        selected = best_bits_by_rel_entropy(reference, responses, n_bits=n_bits)
        metadata = {"bit_selection": "best_bits"}
    elif mode == "chunk":
        start = chunk_index * n_bits
        stop = start + n_bits
        selected = np.arange(start, stop)
        if stop > responses.shape[2]:
            raise ValueError("selected chunk exceeds available bit count")
        metadata = {"bit_selection": f"raw_chunk_{chunk_index}"}
    else:
        raise ValueError(f"unknown bit selection mode: {mode}")

    metadata["selected_bit_indices"] = original_bit_ids[selected]
    metadata["n_bits"] = int(selected.size)
    return FamilyResponses(
        family=family,
        responses=responses[:, :, selected],
        reference=reference[:, selected],
        labels=default_sample_device_labels(family, responses[:, :, selected]),
        metadata=metadata,
    )


def load_sram_verona(
    family: str,
    path: Optional[Path],
    *,
    environment: str = "roomT",
    reference_environment: str = "roomT",
    n_bits: int = 256,
    bit_selection: str = "best",
    chunk_index: int = 0,
):
    with h5py.File(path, "r") as h5:
        responses = h5[f"crps/{environment}"][:]
        reference_responses = h5[f"crps/{reference_environment}"][:]

    responses = normalize_binary_crps(
        responses,
        dataset_name=family,
        bit_axis=2,
    )
    reference_responses = normalize_binary_crps(
        reference_responses,
        dataset_name=f"{family} reference",
        bit_axis=2,
    )
    reference = majority_reference(reference_responses)

    result = select_response_bits(
        family,
        responses,
        reference,
        n_bits=n_bits,
        mode=bit_selection,
        chunk_index=chunk_index,
        original_bit_ids=np.arange(responses.shape[2]),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "source": "SRAM_ESP32",
            "environment": environment,
            "reference_environment": reference_environment,
        }
    )
    return FamilyResponses(
        result.family, result.responses, result.reference, result.labels, metadata
    )


def load_sram_stm32(
    family: str,
    path: Optional[Path],
    *,
    n_bits: int = 256,
    bit_selection: str = "chunk",
    chunk_index: int = 0,
    z_thresh: float = 2.5,
):
    if path is None:
        raise ValueError("STM32 loader requires a path")

    with h5py.File(path, "r") as h5:
        shape = h5["crps"].shape
        if shape[1] >= shape[2]:
            n_raw_bits = shape[1]
            chunks = np.split(np.arange(n_raw_bits), n_raw_bits // 256)
            base_bit_ids = np.concatenate(chunks[1180:1380])
            raw = np.array(h5["crps"][:, base_bit_ids, :], dtype=np.uint8)
            responses = normalize_binary_crps(
                raw,
                dataset_name=family,
                bit_axis=1,
            )
        else:
            n_raw_bits = shape[2]
            chunks = np.split(np.arange(n_raw_bits), n_raw_bits // 256)
            base_bit_ids = np.concatenate(chunks[1180:1380])
            raw = np.array(h5["crps"][:, :, base_bit_ids], dtype=np.uint8)
            responses = normalize_binary_crps(
                raw,
                dataset_name=family,
                bit_axis=2,
            )

    # Sample 13 (12 in 0th-based index) has a median of bitflips much higher than the rest, so we remove it
    valid_samples = np.array(list(set(range(15)) - {12}))
    responses = responses[:, valid_samples, :]
    sample_means = responses.mean(axis=2)
    mean = sample_means.mean(axis=1, keepdims=True)
    std = sample_means.std(axis=1, keepdims=True)
    valid_mask = np.abs((sample_means - mean) / (std + 1e-8)) < z_thresh

    reference_mean = np.zeros((responses.shape[0], responses.shape[2]), dtype=float)
    for device_idx in range(responses.shape[0]):
        valid_samples = valid_mask[device_idx]
        if np.any(valid_samples):
            reference_mean[device_idx] = responses[device_idx, valid_samples].mean(
                axis=0
            )
    reference = np.where(reference_mean >= 0.5, 1, 0).astype(np.uint8)

    result = select_response_bits(
        family,
        responses,
        reference,
        n_bits=n_bits,
        mode=bit_selection,
        chunk_index=chunk_index,
        original_bit_ids=base_bit_ids,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "source": "SRAM_STM32",
            "z_thresh": z_thresh,
            "n_outlier_samples": int(np.sum(~valid_mask)),
        }
    )
    return FamilyResponses(
        result.family, result.responses, result.reference, result.labels, metadata
    )


def load_sram_xmc16(
    family: str,
    path: Optional[Path],
    *,
    n_bits: int = 256,
    bit_selection: str = "chunk",
    chunk_index: int = 0,
):
    """Load the first 256 XMC4500 SRAM bits extracted by the ETL."""
    if path is None:
        raise ValueError("SRAM_XMC4500 loader requires a path")

    with h5py.File(path, "r") as h5:
        responses = np.asarray(h5["crps"], dtype=np.uint8)
        reference = np.asarray(h5["reference"], dtype=np.uint8)

    result = select_response_bits(
        family,
        responses,
        reference,
        n_bits=n_bits,
        mode=bit_selection,
        chunk_index=chunk_index,
        original_bit_ids=np.arange(responses.shape[2]),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "source": "SRAM_XMC4500",
            "environment": "normal",
            "raw_bit_offset": 0,
        }
    )
    return FamilyResponses(
        result.family, result.responses, result.reference, result.labels, metadata
    )


def load_ro_maiti(
    family: str,
    path: Optional[Path],
    *,
    n_bits: int = 256,
    bit_selection: str = "chunk",
    chunk_index: int = 0,
):
    """Load response bits derived from disjoint adjacent Maiti RO pairs."""
    if path is None:
        raise ValueError("RO_Maiti_Spartan3E loader requires a path")

    with h5py.File(path, "r") as h5:
        responses = np.asarray(h5["crps"], dtype=np.uint8)
        reference = np.asarray(h5["reference"], dtype=np.uint8)

    result = select_response_bits(
        family,
        responses,
        reference,
        n_bits=n_bits,
        mode=bit_selection,
        chunk_index=chunk_index,
        original_bit_ids=np.arange(responses.shape[2]),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "source": "RO_Maiti_Spartan3E",
            "pairing": "disjoint_adjacent_oscillators",
        }
    )
    return FamilyResponses(
        result.family, result.responses, result.reference, result.labels, metadata
    )


def load_ro_infineon(
    family: str,
    path: Optional[Path],
    *,
    n_bits: int = 256,
    bit_selection: str = "best",
    chunk_index: int = 0,
):
    if path is None:
        raise ValueError("RO_Infineon loader requires a path")

    with h5py.File(path, "r") as h5fp:
        indices_399 = h5fp["common/indices"][:]
        common_metadata = h5fp["common/metadata"][:]
        dataset_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5fp["common/dataset_names"][:]
        ]

        responses = []
        for idx, dataset_name in enumerate(dataset_names):
            dataset = h5fp[f"datasets/{dataset_name}"]
            indices = indices_399[idx]
            selected_metadata = hdf5_rows_in_order(dataset["metadata"], indices)
            for field in ("wafer_id", "lot_id", "x", "y"):
                if not np.array_equal(
                    selected_metadata[field], common_metadata[field]
                ):
                    raise ValueError(
                        f"{dataset_name}: {field} does not match common device order"
                    )
            freq_diffs = hdf5_rows_in_order(dataset["frequency_diffs"], indices)
            responses.append((freq_diffs > 0).astype(np.int8))

    responses = np.transpose(np.array(responses), (1, 0, 2))
    reference = majority_reference(responses)

    result = select_response_bits(
        family,
        responses,
        reference,
        n_bits=n_bits,
        mode=bit_selection,
        chunk_index=chunk_index,
        original_bit_ids=np.arange(responses.shape[2]),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "source": "RO_Infineon",
            "n_ro_groups": 6,
            "n_ro_designs": 255,
            "response_order": "group_pair_then_design",
        }
    )
    return FamilyResponses(
        result.family, result.responses, result.reference, result.labels, metadata
    )


def load_ro_fraunhofer(
    family: str,
    path: Optional[Path],
    *,
    n_bits: int = 256,
    bit_selection: str = "best",
    chunk_index: int = 0,
):
    if path is None:
        raise ValueError("RO_Fraunhofer loader requires a path")

    with h5py.File(path, "r") as h5:
        responses = np.array(h5["crps"], dtype=np.uint8)[:]

    responses = np.transpose(np.array(responses), (1, 0, 2))
    reference = majority_reference(responses)

    result = select_response_bits(
        family,
        responses,
        reference,
        n_bits=n_bits,
        mode=bit_selection,
        chunk_index=chunk_index,
        original_bit_ids=np.arange(responses.shape[2]),
    )
    metadata = dict(result.metadata)
    metadata["source"] = "RO_Fraunhofer"
    return FamilyResponses(
        result.family, result.responses, result.reference, result.labels, metadata
    )


def load_differential_room(
    family: str,
    path: Optional[Path],
    *,
    n_bits: int = 256,
    bit_selection: str = "chunk",
    chunk_index: int = 0,
):
    """Load the room condition from the differential temperature artifact."""
    if path is None:
        raise ValueError("differential room loader requires a path")

    with h5py.File(path, "r") as h5:
        responses = np.asarray(h5["reference"], dtype=np.uint8)
        reference = np.asarray(h5["reference_majority"], dtype=np.uint8)

    result = select_response_bits(
        family,
        responses,
        reference,
        n_bits=n_bits,
        mode=bit_selection,
        chunk_index=chunk_index,
        original_bit_ids=np.arange(responses.shape[2]),
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "source": "differential_temperature_artifact",
            "environment": "roomT",
        }
    )
    return FamilyResponses(
        result.family, result.responses, result.reference, result.labels, metadata
    )


def generate_gen_ideal(
    family: str,
    path: Optional[Path],
    *,
    n_devices: int = 100,
    n_samples: int = 20,
    n_bits: int = 256,
    bit_selection: str = "first",
    seed: int = 0,
) -> FamilyResponses:
    """
    Generate an IID near-ideal binary PUF with mild population imbalance and
    independent temporal bit flips.
    """
    rng = np.random.default_rng(seed)
    p_bits = rng.beta(a=5, b=5, size=n_bits)
    reference = rng.binomial(n=1, p=p_bits, size=(n_devices, n_bits)).astype(np.uint8)

    flips = rng.binomial(1, 0.05, size=(n_devices, n_samples, n_bits)).astype(np.uint8)
    responses = reference[:, np.newaxis, :] ^ flips

    metadata = {
        "generator": "GEN_IDEAL",
        "description": "IID bits, Beta-Binomial reference, Bernoulli noise",
        "bit_selection": bit_selection,
        "selected_bit_indices": np.arange(n_bits),
    }
    return FamilyResponses(
        family,
        responses,
        reference,
        default_sample_device_labels(family, responses),
        metadata,
    )


def save_hdf5_artifact(path: Path, families: list[FamilyResponses]):
    """Write a dataset artifact with one HDF5 group per family."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for family in families:
            group = h5.create_group(family.family)
            group.create_dataset("responses", data=family.responses, compression="gzip")
            group.create_dataset("reference", data=family.reference, compression="gzip")
            group.create_dataset(
                "labels",
                data=family.labels.astype(str).astype("S"),
                compression="gzip",
            )
            for key, value in family.metadata.items():
                if isinstance(value, np.ndarray):
                    group.create_dataset(f"metadata/{key}", data=value)
                else:
                    group.attrs[key] = value


def load_hdf5_artifact(path: Path) -> list[FamilyResponses]:
    """Load all family groups from a dataset artifact."""
    families = []
    with h5py.File(path, "r") as h5:
        for family_name, group in h5.items():
            metadata = dict(group.attrs)
            if "metadata" in group:
                for key, dataset in group["metadata"].items():
                    metadata[key] = np.array(dataset)
            families.append(
                FamilyResponses(
                    family=family_name,
                    responses=np.array(group["responses"]),
                    reference=np.array(group["reference"]),
                    labels=np.array(group["labels"]).astype(str),
                    metadata=metadata,
                )
            )
    return families


def families_by_name(path: Path):
    """Load an artifact as a dictionary keyed by family name."""
    return {family.family: family for family in load_hdf5_artifact(path)}
