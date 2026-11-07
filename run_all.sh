#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pufre-matplotlib}"

ARTIFACT="data/puf_artifact.h5"
QC_DIR="data/error_qc"
DEPENDENCY_DIR="data/error_dependency"
GRAPH_DIR="data/error_graph"
HYPOTHESIS_DIR="data/error_hypothesis"

FORCE=0
case "${1:-}" in
    --force)
        FORCE=1
        shift
        ;;
    -h|--help)
        printf 'Usage: %s [--force]\n' "$0"
        exit 0
        ;;
esac
if (( $# > 0 )); then
    printf 'Unknown argument: %s\n' "$1" >&2
    printf 'Usage: %s [--force]\n' "$0" >&2
    exit 2
fi

run_stage() {
    local label="$1"
    local sentinel="$2"
    shift 2
    if (( FORCE == 0 )) && [[ -s "${sentinel}" ]]; then
        printf 'Skipping %s; found %s\n' "${label}" "${sentinel}"
        return
    fi
    printf 'Running %s\n' "${label}"
    "$@"
}

if [[ ! -f "${ARTIFACT}" ]]; then
    printf 'Missing artifact: %s\n' "${ARTIFACT}" >&2
    exit 1
fi

run_stage "error QC and repeated predictive validation" \
    "${QC_DIR}/validation_trials.csv" \
    uv run python scripts/run_error_qc.py \
    --artifact "${ARTIFACT}" \
    --output-dir "${QC_DIR}" \
    --block-size 256 \
    --correction-capacity 8 \
    --n-blocks 1000 \
    --compute-mi \
    --run-validation \
    --validation-repeats 20 \
    --repeated-validation-families \
        RO_Infineon \
        RO_Fraunhofer_LEFT_LOWER \
        RO_Maiti_Spartan3E \
        RO_Spartan7_roomT \
        SRAM_ESP32_roomT \
        SRAM_STM32 \
        SRAM_XMC4500 \
    --max-iter 500 \
    --max-tuning-bits 8 \
    --cv-splits 2 \
    --n-jobs 8

run_stage "conditional-independence tests" \
    "${HYPOTHESIS_DIR}/permutation_null.csv" \
    uv run python scripts/run_error_independence_tests.py \
    --artifact "${ARTIFACT}" \
    --output-dir "${HYPOTHESIS_DIR}" \
    --families GEN_Ideal \
    --families RO_Infineon \
    --families RO_Fraunhofer_LEFT_LOWER \
    --families RO_Spartan7_roomT \
    --families RO_Maiti_Spartan3E \
    --families SRAM_ESP32_roomT \
    --families SRAM_STM32 \
    --families SRAM_XMC4500 \
    --n-permutations 999 \
    --n-jobs 8

run_stage "directed dependency models" "${DEPENDENCY_DIR}/summary.csv" \
    uv run python scripts/run_error_dependency_studies.py \
    --artifact "${ARTIFACT}" \
    --study-name error_dependency \
    --output-dir "${DEPENDENCY_DIR}" \
    --regularization-csv "${QC_DIR}/validation.csv" \
    --n-jobs 8

run_stage "graph studies" "${GRAPH_DIR}/graph_summary.csv" \
    uv run python scripts/run_error_graph_studies.py \
    --artifact "${ARTIFACT}" \
    --study-dir "${DEPENDENCY_DIR}" \
    --qc-dir "${QC_DIR}" \
    --output-dir "${GRAPH_DIR}" \
    --bootstrap-replicates 50 \
    --bootstrap-min-probability 0.5 \
    --max-iter 500 \
    --n-jobs 8

run_stage "graph independence null" \
    "${GRAPH_DIR}/independence_null_summary.csv" \
    uv run python scripts/run_graph_independence_null.py \
    --artifact "${ARTIFACT}" \
    --study-dir "${DEPENDENCY_DIR}" \
    --graph-dir "${GRAPH_DIR}" \
    --families RO_Infineon \
    --families RO_Fraunhofer_LEFT_LOWER \
    --families RO_Spartan7_roomT \
    --families RO_Maiti_Spartan3E \
    --families SRAM_ESP32_roomT \
    --families SRAM_STM32 \
    --families SRAM_XMC4500 \
    --bootstrap-replicates 50 \
    --bootstrap-min-probability 0.5 \
    --max-iter 500 \
    --n-jobs 8

uv run python scripts/plot_error_qc.py \
    --qc-dir "${QC_DIR}" \
    --figures-dir figures/error_qc

uv run python scripts/plot_error_graph_studies.py \
    --graph-dir "${GRAPH_DIR}" \
    --figures-dir figures/error_graph

uv run python scripts/make_host_room_temperature_assets.py

