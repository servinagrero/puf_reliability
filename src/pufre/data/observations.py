from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from pufre.data.encoding import binary_to_polar, polar_to_binary


@dataclass(frozen=True)
class PUFObservations:
    """Artifact-backed PUF observations with separated reference and noise channel.

    Attributes:
        - responses: 3D array of responses in polar {-1, 1}, of shape (devices, responses, samples)
        - reference: 2D array of responses in polar {-1, 1} of shape (devices, responses). It should have the shape ``(responses.shape[0], responses.shape[2])
        - bit_ids: List of the bits that are selected
        - device_ids: List of the device labels. Should have length responses.shape[0]
        - sample_ids: List of the sample labels. Should have length responses.shape[1]
        - challenges: Optional list of challenges applied
        - metadata: Optional dictionary to store metadata

    The error spin is ``e = reference * response`` so +1 means agreement and -1 means a flip.
    Bit IDs are local artifact coordinates. Physical source bit IDs are metadata.
    """

    responses: npt.NDArray
    reference: npt.NDArray
    bit_ids: npt.NDArray
    device_ids: npt.NDArray
    sample_ids: npt.NDArray
    challenges: Optional[npt.NDArray] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_binary(
        cls,
        responses: npt.NDArray,
        reference: npt.NDArray,
        *,
        challenges: Optional[npt.NDArray] = None,
        metadata: Optional[dict] = None,
    ):
        responses = np.asarray(responses)
        reference = np.asarray(reference)
        if responses.ndim != 3:
            raise ValueError("responses must have shape (device, sample, bit)")
        if reference.shape != (responses.shape[0], responses.shape[2]):
            raise ValueError("reference must have shape (device, bit)")

        n_devices, n_samples, n_bits = responses.shape
        bit_ids = np.arange(n_bits)
        device_ids = np.repeat(np.arange(n_devices), n_samples)
        sample_ids = np.tile(np.arange(n_samples), n_devices)

        meta = {} if metadata is None else dict(metadata)
        meta["n_bits"] = int(n_bits)
        return cls(
            responses=binary_to_polar(responses),
            reference=binary_to_polar(reference),
            bit_ids=bit_ids,
            device_ids=device_ids,
            sample_ids=sample_ids,
            challenges=None if challenges is None else binary_to_polar(challenges),
            metadata=meta,
        )

    @classmethod
    def from_family(cls, family):
        """Create local-coordinate observations from one HDF5 artifact family."""
        return cls.from_binary(
            family.responses,
            family.reference,
            metadata=family.metadata,
        )

    @property
    def n_devices(self):
        return int(self.responses.shape[0])

    @property
    def n_samples(self):
        return int(self.responses.shape[1])

    @property
    def n_bits(self):
        return int(self.responses.shape[2])

    @property
    def physical_bit_ids(self):
        return np.asarray(self.metadata.get("selected_bit_indices", self.bit_ids))

    @property
    def response_binary(self):
        return polar_to_binary(self.responses).astype(np.uint8)

    @property
    def reference_binary(self):
        return polar_to_binary(self.reference).astype(np.uint8)

    @property
    def error_spins(self):
        return self.responses * self.reference[:, np.newaxis, :]

    @property
    def error_binary(self):
        return (self.error_spins < 0).astype(np.uint8)

    def flat_errors(self):
        return self.error_spins.reshape(-1, self.n_bits)

    def flat_error_binary(self):
        return self.error_binary.reshape(-1, self.n_bits)

    def flat_responses(self):
        return self.responses.reshape(-1, self.n_bits)

    def select_bits(self, bit_indices: npt.ArrayLike):
        bit_indices = np.asarray(bit_indices, dtype=int)
        if np.any(bit_indices < 0) or np.any(bit_indices >= self.n_bits):
            raise ValueError("bit selection is outside local artifact width")

        metadata = dict(self.metadata)
        metadata["selected_bit_indices"] = self.physical_bit_ids[bit_indices]
        metadata["n_bits"] = int(bit_indices.size)
        return PUFObservations(
            responses=self.responses[:, :, bit_indices],
            reference=self.reference[:, bit_indices],
            bit_ids=np.arange(bit_indices.size),
            device_ids=self.device_ids.copy(),
            sample_ids=self.sample_ids.copy(),
            challenges=self.challenges,
            metadata=metadata,
        )


@dataclass(frozen=True)
class DifferentialObservations:
    """
    Repeated differential measurements where binary responses are value1 > value2.

    This covers RO-style measurements and any PUF where the observed bit is
    derived from two competing analog/count channels.
    """

    value1: npt.NDArray
    value2: npt.NDArray
    reference: npt.NDArray
    bit_ids: npt.NDArray
    device_ids: npt.NDArray
    sample_ids: npt.NDArray
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_counts(
        cls,
        value1: npt.NDArray,
        value2: npt.NDArray,
        *,
        reference: Optional[npt.NDArray] = None,
        metadata: Optional[dict] = None,
    ):
        value1 = np.asarray(value1)
        value2 = np.asarray(value2)
        if value1.shape != value2.shape:
            raise ValueError("value1 and value2 must have the same shape")
        if value1.ndim != 3:
            raise ValueError(
                "differential values must have shape (device, sample, bit)"
            )

        responses = (value1 > value2).astype(np.uint8)
        if reference is None:
            reference = (responses.mean(axis=1) >= 0.5).astype(np.uint8)
        else:
            reference = np.asarray(reference)

        if reference.shape != (responses.shape[0], responses.shape[2]):
            raise ValueError("reference must have shape (device, bit)")

        n_devices, n_samples, n_bits = responses.shape
        meta = {} if metadata is None else dict(metadata)
        meta["n_bits"] = int(n_bits)
        return cls(
            value1=value1,
            value2=value2,
            reference=reference.astype(np.uint8),
            bit_ids=np.arange(n_bits),
            device_ids=np.repeat(np.arange(n_devices), n_samples),
            sample_ids=np.tile(np.arange(n_samples), n_devices),
            metadata=meta,
        )

    @property
    def n_devices(self):
        return int(self.value1.shape[0])

    @property
    def n_samples(self):
        return int(self.value1.shape[1])

    @property
    def n_bits(self):
        return int(self.value1.shape[2])

    @property
    def responses_binary(self):
        return (self.value1 > self.value2).astype(np.uint8)

    @property
    def margin(self):
        return np.asarray(self.value1, dtype=float) - np.asarray(
            self.value2, dtype=float
        )

    @property
    def normalized_margin(self):
        value1 = np.asarray(self.value1, dtype=float)
        value2 = np.asarray(self.value2, dtype=float)
        return (value1 - value2) / np.maximum(value1 + value2, 1.0)

    def to_puf_observations(self):
        return PUFObservations.from_binary(
            self.responses_binary,
            self.reference,
            metadata=self.metadata,
        )

    def select_bits(self, bit_indices: npt.ArrayLike):
        bit_indices = np.asarray(bit_indices, dtype=int)
        if np.any(bit_indices < 0) or np.any(bit_indices >= self.n_bits):
            raise ValueError("bit selection is outside local observation width")
        metadata = dict(self.metadata)
        metadata["selected_bit_indices"] = self.bit_ids[bit_indices]
        metadata["n_bits"] = int(bit_indices.size)
        return DifferentialObservations(
            value1=self.value1[:, :, bit_indices],
            value2=self.value2[:, :, bit_indices],
            reference=self.reference[:, bit_indices],
            bit_ids=np.arange(bit_indices.size),
            device_ids=self.device_ids.copy(),
            sample_ids=self.sample_ids.copy(),
            metadata=metadata,
        )
