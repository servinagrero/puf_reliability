import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl

from pufre import DATA_DIR, FIGURES_DIR
from pufre.plotting.qc import (  # noqa: E402
    plot_block_failure,
    plot_block_failure_curves,
    plot_effective_independent_errors,
    plot_flip_rate,
    plot_mi_heatmaps,
    plot_unique_error_rows,
    plot_validation_delta,
)

try:
    plt.style.use("publications")
except OSError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-dir", type=Path, default=DATA_DIR / "error_qc")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR / "error_qc")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    qc = pl.read_csv(args.qc_dir / "artifact_qc.csv")
    plot_unique_error_rows(qc, args.figures_dir / "unique_error_rows.png")
    plot_effective_independent_errors(
        qc, args.figures_dir / "effective_independent_error_bits.png"
    )
    plot_flip_rate(qc, args.figures_dir / "mean_flip_rate.png")
    plot_block_failure(qc, args.figures_dir / "block_failure.png")

    curve_path = args.qc_dir / "block_failure_curve.csv"
    if curve_path.exists():
        plot_block_failure_curves(
            pl.read_csv(curve_path),
            args.figures_dir / "block_failure_curves.png",
        )
    plot_mi_heatmaps(args.qc_dir, args.figures_dir)

    validation_path = args.qc_dir / "validation.csv"
    if validation_path.exists():
        validation = pl.read_csv(validation_path)
        plot_validation_delta(validation, args.figures_dir / "validation_log_loss.png")


if __name__ == "__main__":
    main()
