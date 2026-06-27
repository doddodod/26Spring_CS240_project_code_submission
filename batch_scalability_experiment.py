"""Scalability experiment for batched seam extraction (improvement #1).

Reuses the shared timing framework (``TEST_SIZES``, ``REPEATS``, input prep) from
``runtime_scalability_experiment`` and compares several batch sizes. For each
test size the image width is halved with batched seam removal; ``batch_size=1``
reproduces the baseline one-seam-per-pass behavior.

Outputs:
    results/timing/batch_runtime.csv
    results/timing/batch_runtime_loglog.png
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from batch_seam_carving import batch_resize
from runtime_scalability_experiment import (
    INPUT_PATH,
    REPEATS,
    TEST_SIZES,
    prepare_scaled_inputs,
)

BATCH_SIZES = [1, 5, 20]
RESULT_DIR = Path("results/timing")
CSV_PATH = RESULT_DIR / "batch_runtime.csv"
PLOT_PATH = RESULT_DIR / "batch_runtime_loglog.png"

COLORS = {1: "#08519c", 5: "#3182bd", 20: "#6baed6"}
MARKERS = {1: "o", 5: "s", 20: "^"}


def time_batch_resize(image: np.ndarray, target_width: int, batch_size: int) -> float:
    working = image.copy()
    start = time.perf_counter()
    batch_resize(working, target_width=target_width, batch_size=batch_size)
    return time.perf_counter() - start


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), std


def run_experiment() -> list[dict[str, float | int]]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scaled_inputs = prepare_scaled_inputs(INPUT_PATH)
    rows: list[dict[str, float | int]] = []

    for width, height, image in scaled_inputs:
        target_width = width // 2
        row: dict[str, float | int] = {
            "width": width,
            "height": height,
            "pixels": width * height,
        }
        for batch_size in BATCH_SIZES:
            times = []
            for repeat in range(1, REPEATS + 1):
                seconds = time_batch_resize(image, target_width, batch_size)
                times.append(seconds)
                print(
                    f"{width}x{height} batch={batch_size:2d} "
                    f"repeat={repeat}: {seconds:.4f}s"
                )
            mean, std = summarize(times)
            row[f"batch{batch_size}_mean"] = mean
            row[f"batch{batch_size}_std"] = std
        rows.append(row)

    write_csv(rows)
    return rows


def write_csv(rows: list[dict[str, float | int]]) -> None:
    fieldnames = ["width", "height", "pixels"]
    for batch_size in BATCH_SIZES:
        fieldnames += [f"batch{batch_size}_mean", f"batch{batch_size}_std"]
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fit_slope(pixels: np.ndarray, times: np.ndarray) -> float:
    return float(np.polyfit(np.log(pixels), np.log(times), 1)[0])


def plot_results(rows: list[dict[str, float | int]]) -> dict[int, float]:
    pixels = np.asarray([row["pixels"] for row in rows], dtype=np.float64)
    sorted_pixels = np.sort(pixels)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    slopes: dict[int, float] = {}

    for batch_size in BATCH_SIZES:
        means = np.asarray([row[f"batch{batch_size}_mean"] for row in rows], dtype=np.float64)
        slope = fit_slope(pixels, means)
        slopes[batch_size] = slope
        ax.loglog(
            pixels,
            means,
            color=COLORS[batch_size],
            marker=MARKERS[batch_size],
            linestyle="-",
            linewidth=2,
            label=f"batch size = {batch_size} (slope={slope:.2f})",
        )

    # Anchor the reference line to the smallest-image runtime of the first batch.
    anchor = rows[0][f"batch{BATCH_SIZES[0]}_mean"]
    reference = anchor * (sorted_pixels / sorted_pixels[0]) ** 1.5
    ax.loglog(
        sorted_pixels,
        reference,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Theoretical slope = 1.5",
    )

    ax.set_title("Batched Seam Extraction: Runtime Scalability")
    ax.set_xlabel("Pixels (width x height), log scale")
    ax.set_ylabel("Average runtime (seconds), log scale")
    ax.grid(True, which="both", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    return slopes


def main() -> None:
    rows = run_experiment()
    slopes = plot_results(rows)
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved plot: {PLOT_PATH}")
    for batch_size in BATCH_SIZES:
        print(f"batch size {batch_size:2d} fitted log-log slope: {slopes[batch_size]:.3f}")
    print(f"Completed {len(rows)} image sizes with {REPEATS} repeats each.")


if __name__ == "__main__":
    main()
