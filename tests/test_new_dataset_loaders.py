import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pufre.datasets import load_ro_infineon, load_ro_maiti, load_sram_xmc16


class NewDatasetLoaderTests(unittest.TestCase):
    def test_infineon_loader_preserves_common_device_order(self):
        metadata_dtype = np.dtype(
            [
                ("row_id", np.int32),
                ("wafer_id", np.int16),
                ("lot_id", np.int16),
                ("x", np.int16),
                ("y", np.int16),
            ]
        )
        common_metadata = np.array(
            [(0, 1, 1, 10, 20), (1, 1, 1, 11, 20)], dtype=metadata_dtype
        )
        reversed_metadata = common_metadata[::-1].copy()
        reversed_metadata["row_id"] = [4, 5]
        first = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=float)
        second_common_order = np.array(
            [[1, 1, -1, -1], [-1, -1, 1, 1]], dtype=float
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "infineon.h5"
            with h5py.File(path, "w") as h5:
                common = h5.create_group("common")
                common.create_dataset("metadata", data=common_metadata)
                common.create_dataset("indices", data=np.array([[0, 1], [1, 0]]))
                common.create_dataset(
                    "dataset_names",
                    data=np.array(
                        ["condition_0", "condition_1"], dtype=h5py.string_dtype()
                    ),
                )
                datasets = h5.create_group("datasets")
                condition_0 = datasets.create_group("condition_0")
                condition_0.create_dataset("metadata", data=common_metadata)
                condition_0.create_dataset("frequency_diffs", data=first)
                condition_1 = datasets.create_group("condition_1")
                condition_1.create_dataset("metadata", data=reversed_metadata)
                condition_1.create_dataset(
                    "frequency_diffs", data=second_common_order[::-1]
                )

            family = load_ro_infineon(
                "Infineon",
                path,
                n_bits=4,
                bit_selection="chunk",
            )

        expected = np.stack(
            [(first > 0).astype(np.int8), (second_common_order > 0).astype(np.int8)],
            axis=1,
        )
        np.testing.assert_array_equal(family.responses, expected)

    def test_xmc_loader_selects_requested_raw_chunk(self):
        responses = np.arange(48, dtype=np.uint8).reshape(2, 3, 8) % 2
        reference = (responses.mean(axis=1) >= 0.5).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xmc.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset("crps", data=responses)
                h5.create_dataset("reference", data=reference)

            family = load_sram_xmc16(
                "XMC",
                path,
                n_bits=4,
                bit_selection="chunk",
                chunk_index=1,
            )

        np.testing.assert_array_equal(family.responses, responses[:, :, 4:8])
        np.testing.assert_array_equal(family.reference, reference[:, 4:8])

    def test_maiti_loader_selects_requested_raw_chunk(self):
        responses = np.arange(48, dtype=np.uint8).reshape(2, 3, 8) % 2
        reference = (responses.mean(axis=1) >= 0.5).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maiti.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset("crps", data=responses)
                h5.create_dataset("reference", data=reference)

            family = load_ro_maiti(
                "Maiti",
                path,
                n_bits=4,
                bit_selection="chunk",
                chunk_index=1,
            )

        np.testing.assert_array_equal(family.responses, responses[:, :, 4:8])
        np.testing.assert_array_equal(family.reference, reference[:, 4:8])


if __name__ == "__main__":
    unittest.main()
