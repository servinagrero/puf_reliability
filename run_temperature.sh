#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pufre-matplotlib}"

ARTIFACT="${PUFRE_TEMPERATURE_ARTIFACT:-data/raw/RO_PUF.h5}"
N_JOBS="${PUFRE_N_JOBS:-8}"
FORCE_ARGS=()

case "${1:-}" in
    --force)
        FORCE_ARGS=(--force)
        shift
        ;;
    -h|--help)
        printf 'Usage: %s [--force]\n' "$0"
        printf 'Environment:\n'
        printf '  PUFRE_TEMPERATURE_ARTIFACT  Temperature HDF5 artifact path\n'
        printf '  PUFRE_N_JOBS                Parallel worker count (default: 8)\n'
        exit 0
        ;;
esac

if (( $# > 0 )); then
    printf 'Unknown argument: %s\n' "$1" >&2
    printf 'Usage: %s [--force]\n' "$0" >&2
    exit 2
fi

if ! [[ "${N_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PUFRE_N_JOBS must be a positive integer: %s\n' "${N_JOBS}" >&2
    exit 2
fi

if [[ ! -f "${ARTIFACT}" ]]; then
    printf 'Missing temperature artifact: %s\n' "${ARTIFACT}" >&2
    exit 1
fi

export PUFRE_TEMPERATURE_ARTIFACT="${ARTIFACT}"
export PUFRE_N_JOBS="${N_JOBS}"

uv run python scripts/run_ro_temperature_study.py "${FORCE_ARGS[@]}"
uv run python scripts/make_host_ro_temperature_assets.py
