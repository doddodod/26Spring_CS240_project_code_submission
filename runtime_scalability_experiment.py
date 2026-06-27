"""Runtime and scalability experiment for backward and forward seam carving.

The experiment uses one source image, rescales it to several fixed test sizes,
then removes half of the width with backward-energy and forward-energy seam
carving. Timing excludes image loading, image saving, and the initial LANCZOS
rescaling step.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from backward_energy_seam_carving import resize as bw_resize
from forward_energy_seam_carving import resize as fw_resize


INPUT_PATH = Path("picture/input/bench.png")
TEST_SIZES = [
    (100, 67),
    (160, 107),
    (256, 171),
    (360, 240),
    (440, 294),
    (512, 342),
]
REPEATS = 5
RESULT_DIR = Path("results/timing")
INPUT_DIR = RESULT_DIR / "inputs"
CSV_PATH = RESULT_DIR / "runtime.csv"
PLOT_PNG_PATH = RESULT_DIR / "runtime_plot.png"
PLOT_PDF_PATH = RESULT_DIR / "runtime_plot.pdf"
PLOT_LOGLOG_PATH = RESULT_DIR / "runtime_plot_loglog.png"

BACKWARD_COLOR = "#1f77b4"
FORWARD_COLOR = "#6baed6"


def prepare_scaled_inputs(source_path: Path) -> list[tuple[int, int, np.ndarray]]:
    """Create resized input arrays and save them for experiment traceability."""
    source = Image.open(source_path).convert("RGB")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    scaled_inputs = []
    for width, height in TEST_SIZES:
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(INPUT_DIR / f"bench_{width}x{height}.png")
        image = np.asarray(resized, dtype=np.float64)
        scaled_inputs.append((width, height, image))
    return scaled_inputs


def time_resize(method: str, image: np.ndarray, target_width: int) -> float:
    """Measure only the seam-carving resize call."""
    working = image.copy()
    start = time.perf_counter()
    if method == "backward":
        bw_resize(working, target_width=target_width, target_height=None)
    elif method == "forward":
        fw_resize(working, target_width=target_width, target_height=None)
    else:
        raise ValueError(f"Unknown method: {method}")
    return time.perf_counter() - start


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def linear_fit(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Return slope, intercept, R^2, and fitted y values for y = ax + b."""
    slope, intercept = np.polyfit(x_values, y_values, 1)
    fitted = slope * x_values + intercept
    residual = np.sum((y_values - fitted) ** 2)
    total = np.sum((y_values - np.mean(y_values)) ** 2)
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    return float(slope), float(intercept), float(r_squared), fitted


def run_experiment() -> list[dict[str, float | int]]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scaled_inputs = prepare_scaled_inputs(INPUT_PATH)
    rows: list[dict[str, float | int]] = []

    for width, height, image in scaled_inputs:
        target_width = width // 2
        backward_times = []
        forward_times = []

        for repeat in range(1, REPEATS + 1):
            backward_seconds = time_resize("backward", image, target_width)
            forward_seconds = time_resize("forward", image, target_width)
            backward_times.append(backward_seconds)
            forward_times.append(forward_seconds)
            print(
                f"{width}x{height} repeat={repeat}: "
                f"backward={backward_seconds:.4f}s forward={forward_seconds:.4f}s"
            )

        backward_mean, backward_std = summarize(backward_times)
        forward_mean, forward_std = summarize(forward_times)
        rows.append(
            {
                "width": width,
                "height": height,
                "pixels": width * height,
                "backward_mean": backward_mean,
                "backward_std": backward_std,
                "forward_mean": forward_mean,
                "forward_std": forward_std,
            }
        )

    write_csv(rows)
    plot_results(rows)
    return rows


def write_csv(rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "width",
        "height",
        "pixels",
        "backward_mean",
        "backward_std",
        "forward_mean",
        "forward_std",
    ]
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(rows: list[dict[str, float | int]]) -> tuple[float, float]:
    pixels = np.asarray([row["pixels"] for row in rows], dtype=np.float64)
    backward_mean = np.asarray([row["backward_mean"] for row in rows], dtype=np.float64)
    backward_std = np.asarray([row["backward_std"] for row in rows], dtype=np.float64)
    forward_mean = np.asarray([row["forward_mean"] for row in rows], dtype=np.float64)
    forward_std = np.asarray([row["forward_std"] for row in rows], dtype=np.float64)

    bw_slope, bw_intercept, bw_r2, bw_fit = linear_fit(pixels, backward_mean)
    fw_slope, fw_intercept, fw_r2, fw_fit = linear_fit(pixels, forward_mean)

    plt.rcParams.update(
        {
            "axes.edgecolor": "#4f6f8f",
            "axes.labelcolor": "#1f3b57",
            "axes.titlecolor": "#1f3b57",
            "xtick.color": "#1f3b57",
            "ytick.color": "#1f3b57",
            "grid.color": "#d7e3ef",
            "font.size": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    ax.errorbar(
        pixels,
        backward_mean,
        yerr=backward_std,
        color=BACKWARD_COLOR,
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=4,
        label="Backward energy",
    )
    ax.errorbar(
        pixels,
        forward_mean,
        yerr=forward_std,
        color=FORWARD_COLOR,
        marker="s",
        linestyle="-",
        linewidth=2,
        capsize=4,
        label="Forward energy",
    )

    ax.plot(pixels, bw_fit, "k--", linewidth=1.5, label="O(N) linear fit")
    ax.plot(pixels, fw_fit, "k--", linewidth=1.0, alpha=0.55)

    bw_text = f"Backward: T = {bw_slope:.2e}*N + {bw_intercept:.3f}  (R^2={bw_r2:.3f})"
    fw_text = f"Forward:  T = {fw_slope:.2e}*N + {fw_intercept:.3f}  (R^2={fw_r2:.3f})"
    ax.text(
        0.03,
        0.97,
        bw_text + "\n" + fw_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef6ff", "edgecolor": "#9ecae1"},
    )

    ax.set_title("Runtime Scalability of Seam Carving")
    ax.set_xlabel("Pixels (width x height)")
    ax.set_ylabel("Average runtime (seconds)")
    ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.9)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    fig.savefig(PLOT_PNG_PATH)
    fig.savefig(PLOT_PDF_PATH)
    plt.close(fig)

    bw_log_slope, fw_log_slope = plot_runtime_loglog(pixels, backward_mean, forward_mean)
    return bw_log_slope, fw_log_slope


def plot_runtime_loglog(
    pixels: np.ndarray,
    backward_mean: np.ndarray,
    forward_mean: np.ndarray,
) -> tuple[float, float]:
    """Plot log-log runtime scaling and fit empirical power-law slopes."""
    log_pixels = np.log(pixels)
    log_backward = np.log(backward_mean)
    log_forward = np.log(forward_mean)

    bw_slope, bw_intercept = np.polyfit(log_pixels, log_backward, 1)
    fw_slope, fw_intercept = np.polyfit(log_pixels, log_forward, 1)

    sorted_pixels = np.sort(pixels)
    bw_fit = np.exp(bw_intercept) * sorted_pixels**bw_slope
    fw_fit = np.exp(fw_intercept) * sorted_pixels**fw_slope

    reference_anchor = backward_mean[0]
    reference = reference_anchor * (sorted_pixels / sorted_pixels[0]) ** 1.5

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    ax.loglog(
        pixels,
        backward_mean,
        color=BACKWARD_COLOR,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Backward energy",
    )
    ax.loglog(
        pixels,
        forward_mean,
        color=FORWARD_COLOR,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Forward energy",
    )
    ax.loglog(sorted_pixels, bw_fit, color=BACKWARD_COLOR, linestyle=":", linewidth=1.8)
    ax.loglog(sorted_pixels, fw_fit, color=FORWARD_COLOR, linestyle=":", linewidth=1.8)
    ax.loglog(
        sorted_pixels,
        reference,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Theoretical slope = 1.5",
    )

    slope_text = f"Backward slope = {bw_slope:.2f}\nForward slope = {fw_slope:.2f}"
    ax.text(
        0.04,
        0.96,
        slope_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef6ff", "edgecolor": "#9ecae1"},
    )

    ax.set_title("Log-Log Runtime Scalability")
    ax.set_xlabel("Pixels (width x height), log scale")
    ax.set_ylabel("Average runtime (seconds), log scale")
    ax.grid(True, which="both", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_LOGLOG_PATH)
    plt.close(fig)

    return float(bw_slope), float(fw_slope)


def main() -> None:
    rows = run_experiment()
    pixels = np.asarray([row["pixels"] for row in rows], dtype=np.float64)
    backward_mean = np.asarray([row["backward_mean"] for row in rows], dtype=np.float64)
    forward_mean = np.asarray([row["forward_mean"] for row in rows], dtype=np.float64)
    backward_slope, _ = np.polyfit(np.log(pixels), np.log(backward_mean), 1)
    forward_slope, _ = np.polyfit(np.log(pixels), np.log(forward_mean), 1)
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved plot: {PLOT_PNG_PATH}")
    print(f"Saved plot: {PLOT_PDF_PATH}")
    print(f"Saved plot: {PLOT_LOGLOG_PATH}")
    print(f"Backward log-log fitted slope: {backward_slope:.3f}")
    print(f"Forward log-log fitted slope: {forward_slope:.3f}")
    print(f"Completed {len(rows)} image sizes with {REPEATS} repeats each.")


if __name__ == "__main__":
    main()
