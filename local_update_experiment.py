"""Scalability experiment for local energy updates (improvement #2).

Reuses the shared timing framework (``TEST_SIZES``, ``REPEATS``, input prep) from
``runtime_scalability_experiment`` and compares the baseline backward-energy
carving (full energy recomputation every seam) with the local-update variant
(band recomputation). For each test size the width is halved.

Outputs:
    results/timing/local_update_runtime.csv
    results/timing/local_update_comparison.png
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backward_energy_seam_carving import resize as baseline_resize
from local_update_carving import resize as local_resize
from runtime_scalability_experiment import (
    INPUT_PATH,
    REPEATS,
    TEST_SIZES,
    prepare_scaled_inputs,
)

RESULT_DIR = Path("results/timing")
CSV_PATH = RESULT_DIR / "local_update_runtime.csv"
PLOT_PATH = RESULT_DIR / "local_update_comparison.png"

BASELINE_COLOR = "#9e3d22"
LOCAL_COLOR = "#2ca25f"


def time_method(method: str, image: np.ndarray, target_width: int) -> float:
    working = image.copy()
    start = time.perf_counter()
    if method == "baseline":
        baseline_resize(working, target_width=target_width, target_height=None)
    elif method == "local":
        local_resize(working, target_width=target_width, target_height=None)
    else:
        raise ValueError(f"unknown method: {method}")
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
        baseline_times = []
        local_times = []
        for repeat in range(1, REPEATS + 1):
            baseline_seconds = time_method("baseline", image, target_width)
            local_seconds = time_method("local", image, target_width)
            baseline_times.append(baseline_seconds)
            local_times.append(local_seconds)
            print(
                f"{width}x{height} repeat={repeat}: "
                f"baseline={baseline_seconds:.4f}s local={local_seconds:.4f}s"
            )

        baseline_mean, baseline_std = summarize(baseline_times)
        local_mean, local_std = summarize(local_times)
        rows.append(
            {
                "width": width,
                "height": height,
                "pixels": width * height,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "local_mean": local_mean,
                "local_std": local_std,
                "speedup": baseline_mean / local_mean if local_mean > 0 else float("nan"),
            }
        )

    write_csv(rows)
    return rows


def write_csv(rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "width",
        "height",
        "pixels",
        "baseline_mean",
        "baseline_std",
        "local_mean",
        "local_std",
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
    local_mean = np.asarray([row["local_mean"] for row in rows], dtype=np.float64)

    baseline_slope = fit_slope(pixels, baseline_mean)
    local_slope = fit_slope(pixels, local_mean)

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
        label=f"Baseline full recompute (slope={baseline_slope:.2f})",
    )
    ax.loglog(
        pixels,
        local_mean,
        color=LOCAL_COLOR,
        marker="s",
        linestyle="-",
        linewidth=2,
        label=f"Local energy update (slope={local_slope:.2f})",
    )
    ax.loglog(
        sorted_pixels,
        reference,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Theoretical slope = 1.5",
    )

    ax.set_title("Local Energy Update: Runtime Scalability")
    ax.set_xlabel("Pixels (width x height), log scale")
    ax.set_ylabel("Average runtime (seconds), log scale")
    ax.grid(True, which="both", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    return baseline_slope, local_slope


def main() -> None:
    rows = run_experiment()
    baseline_slope, local_slope = plot_results(rows)
    mean_speedup = float(np.mean([row["speedup"] for row in rows]))
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved plot: {PLOT_PATH}")
    print(f"Baseline fitted log-log slope: {baseline_slope:.3f}")
    print(f"Local-update fitted log-log slope: {local_slope:.3f}")
    print(f"Mean speedup (baseline / local): {mean_speedup:.2f}x")
    print(f"Completed {len(rows)} image sizes with {REPEATS} repeats each.")


if __name__ == "__main__":
    main()
