"""Scalability experiment for multi-scale seam carving (improvement #3).

Reuses the shared timing framework (``TEST_SIZES``, ``REPEATS``, input prep) from
``runtime_scalability_experiment`` and compares the baseline backward-energy
carving (full-resolution DP) with the multi-scale variant (coarse-to-fine band
DP). For each test size the width is halved.

A visual-quality comparison image (baseline vs multi-scale) is also saved for the
largest test size so the quality trade-off can be judged by eye.

Outputs:
    results/timing/multiscale_runtime.csv
    results/timing/multiscale_comparison.png
    results/timing/multiscale_quality_<W>x<H>.png
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backward_energy_seam_carving import resize as baseline_resize
from backward_energy_seam_carving import save_image
from multiscale_carving import resize as multiscale_resize
from runtime_scalability_experiment import (
    INPUT_PATH,
    REPEATS,
    TEST_SIZES,
    prepare_scaled_inputs,
)
from visualization import make_comparison

SCALE_FACTOR = 2
BAND_WIDTH = 5
RESULT_DIR = Path("results/timing")
CSV_PATH = RESULT_DIR / "multiscale_runtime.csv"
PLOT_PATH = RESULT_DIR / "multiscale_comparison.png"

BASELINE_COLOR = "#9e3d22"
MULTISCALE_COLOR = "#756bb1"


def time_method(method: str, image: np.ndarray, target_width: int) -> float:
    working = image.copy()
    start = time.perf_counter()
    if method == "baseline":
        baseline_resize(working, target_width=target_width, target_height=None)
    elif method == "multiscale":
        multiscale_resize(
            working,
            target_width=target_width,
            target_height=None,
            scale_factor=SCALE_FACTOR,
            band_width=BAND_WIDTH,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return time.perf_counter() - start


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), std


def save_quality_comparison(image: np.ndarray, width: int, height: int) -> Path:
    """Save a baseline-vs-multiscale visual comparison for one image size."""
    target_width = width // 2
    baseline_out = baseline_resize(image.copy(), target_width=target_width)
    multiscale_out = multiscale_resize(
        image.copy(),
        target_width=target_width,
        scale_factor=SCALE_FACTOR,
        band_width=BAND_WIDTH,
    )

    baseline_path = RESULT_DIR / f"multiscale_quality_baseline_{width}x{height}.png"
    multiscale_path = RESULT_DIR / f"multiscale_quality_multiscale_{width}x{height}.png"
    save_image(baseline_out, baseline_path)
    save_image(multiscale_out, multiscale_path)

    comparison_path = RESULT_DIR / f"multiscale_quality_{width}x{height}.png"
    make_comparison(
        [baseline_path, multiscale_path],
        ["Baseline (full DP)", "Multi-scale (band DP)"],
        comparison_path,
    )
    return comparison_path


def run_experiment() -> list[dict[str, float | int]]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scaled_inputs = prepare_scaled_inputs(INPUT_PATH)
    rows: list[dict[str, float | int]] = []

    for width, height, image in scaled_inputs:
        target_width = width // 2
        baseline_times = []
        multiscale_times = []
        for repeat in range(1, REPEATS + 1):
            baseline_seconds = time_method("baseline", image, target_width)
            multiscale_seconds = time_method("multiscale", image, target_width)
            baseline_times.append(baseline_seconds)
            multiscale_times.append(multiscale_seconds)
            print(
                f"{width}x{height} repeat={repeat}: "
                f"baseline={baseline_seconds:.4f}s multiscale={multiscale_seconds:.4f}s"
            )

        baseline_mean, baseline_std = summarize(baseline_times)
        multiscale_mean, multiscale_std = summarize(multiscale_times)
        rows.append(
            {
                "width": width,
                "height": height,
                "pixels": width * height,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "multiscale_mean": multiscale_mean,
                "multiscale_std": multiscale_std,
                "speedup": baseline_mean / multiscale_mean if multiscale_mean > 0 else float("nan"),
            }
        )

    # Save a visual-quality comparison for the largest test size.
    largest_width, largest_height, largest_image = scaled_inputs[-1]
    comparison_path = save_quality_comparison(largest_image, largest_width, largest_height)
    print(f"Saved quality comparison: {comparison_path}")

    write_csv(rows)
    return rows


def write_csv(rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "width",
        "height",
        "pixels",
        "baseline_mean",
        "baseline_std",
        "multiscale_mean",
        "multiscale_std",
        "speedup",
    ]
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fit_slope(pixels: np.ndarray, times: np.ndarray) -> float:
    return float(np.polyfit(np.log(pixels), np.log(times), 1)[0])


def plot_results(rows: list[dict[str, float | int]]) -> tuple[float, float]:
    pixels = np.asarray([row["pixels"] for row in rows], dtype=np.float64)
    baseline_mean = np.asarray([row["baseline_mean"] for row in rows], dtype=np.float64)
    multiscale_mean = np.asarray([row["multiscale_mean"] for row in rows], dtype=np.float64)

    baseline_slope = fit_slope(pixels, baseline_mean)
    multiscale_slope = fit_slope(pixels, multiscale_mean)

    sorted_pixels = np.sort(pixels)
    reference = baseline_mean[0] * (sorted_pixels / sorted_pixels[0]) ** 1.5

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    ax.loglog(
        pixels,
        baseline_mean,
        color=BASELINE_COLOR,
        marker="o",
        linestyle="-",
        linewidth=2,
        label=f"Baseline full DP (slope={baseline_slope:.2f})",
    )
    ax.loglog(
        pixels,
        multiscale_mean,
        color=MULTISCALE_COLOR,
        marker="s",
        linestyle="-",
        linewidth=2,
        label=f"Multi-scale band DP (slope={multiscale_slope:.2f})",
    )
    ax.loglog(
        sorted_pixels,
        reference,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Theoretical slope = 1.5",
    )

    ax.set_title("Multi-scale Seam Carving: Runtime Scalability")
    ax.set_xlabel("Pixels (width x height), log scale")
    ax.set_ylabel("Average runtime (seconds), log scale")
    ax.grid(True, which="both", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    return baseline_slope, multiscale_slope


def main() -> None:
    rows = run_experiment()
    baseline_slope, multiscale_slope = plot_results(rows)
    mean_speedup = float(np.mean([row["speedup"] for row in rows]))
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved plot: {PLOT_PATH}")
    print(f"Baseline fitted log-log slope: {baseline_slope:.3f}")
    print(f"Multi-scale fitted log-log slope: {multiscale_slope:.3f}")
    print(f"Mean speedup (baseline / multiscale): {mean_speedup:.2f}x")
    print(f"Completed {len(rows)} image sizes with {REPEATS} repeats each.")


if __name__ == "__main__":
    main()
