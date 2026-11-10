# Testing PUF Error Independence Beyond Marginal Reliability

Research code for testing whether PUF bit errors observed in the same readout are compatible with device-conditioned independent marginals. The framework combines model-free permutation tests, correction-capacity diagnostics, sparse directed error graphs, held-out device validation, bootstrap stability analysis, and an RO-PUF temperature case study.

## Scope

The repository focuses on:

- PUF response and reference artifacts;
- contemporaneous error-dependence tests;
- sparse first-order directed error graphs;
- graph roles and matched independence-null analysis;
- empirical and Poisson-binomial error-count distributions;
- held-out cross-device predictive validation;
- room-temperature and RO temperature-sweep studies.

Energy-based, Hamiltonian, RBM, LASSO-PUF, and Hopfield experiments are not part of the public package.

## Installation

Python 3.11 or newer and `uv` are required.

```sh
uv sync
```

Run the unit tests with:

```sh
uv run python -m unittest discover -s tests -v
```

## Data

No datasets or generated artifacts are committed to the repository. See [`data/README.md`](data/README.md) for the expected artifact paths.

Available artifact builders:

```sh
uv run python scripts/build_maiti_artifact.py \
    --input-dir /path/to/ropuf_host2010

uv run python scripts/build_xmc16_artifact.py \
    --input-dir /path/to/SRAMxmc16/data

uv run python scripts/build_ro_temperature_artifact.py \
    --reference-dir /path/to/reference/csvs \
    --temperature-dir /path/to/temperature/directories
```

Once all source artifacts are present, build and inspect the unified room-temperature artifact:

```sh
uv run python scripts/build_dataset_artifact.py
uv run python scripts/inspect_artifact.py
```

## Room-temperature study

```sh
./run_all.sh
```

Existing stage outputs are reused. Recompute every stage with:

```sh
./run_all.sh --force
```

The artifact and worker count are configurable:

```sh
PUFRE_ARTIFACT=/path/to/puf_artifact.h5 \
PUFRE_N_JOBS=4 \
./run_all.sh
```

## Temperature study

```sh
./run_temperature.sh
```

To select another artifact or worker count:

```sh
PUFRE_TEMPERATURE_ARTIFACT=/path/to/RO_PUF.h5 \
PUFRE_N_JOBS=4 \
./run_temperature.sh --force
```

## Generated outputs

The workflows write intermediate data under `data/`, figures under `figures/`, and publication tables under `tables/`. These directories are ignored because all outputs can be regenerated from the supplied artifacts.

The sparse graph captures first-order conditional association. It does not establish causality, and failure to reject the matched independence null is not proof that all forms of dependence are absent.
