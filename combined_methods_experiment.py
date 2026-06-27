"""Combined runtime comparison for baseline and three extension methods.

This experiment uses the same scalability inputs as the main runtime experiment:
the source image is resized to six fixed resolutions, and each method reduces
width by half while keeping height unchanged. Each method/size pair is repeated
five times.

Methods:
    - backward-energy baseline
    - forward-energy baseline
    - batch seam extraction, batch_size=5
    - local DP update, delta=5
    - multi-scale seam carving, scale_factor=2, band_width=5

Outputs:
    results/timing/combined_methods_runtime.csv
    results/timing/combined_methods_runtime_loglog.png
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backward_energy_seam_carving import resize as backward_resize
from batch_seam_carving import batch_resize
from forward_energy_seam_carving import resize as forward_resize
from local_dp_update_carving import resize_local_dp
from multiscale_carving import resize as multiscale_resize

REPEATS = 5
INPUT_PATH = Path("picture/input/bench.png")
TEST_SIZES = [
    (100, 67),
    (160, 107),
    (256, 171),
    (360, 240),
    (440, 294),
    (512, 342),
]
RESULT_DIR = Path("results/timing")
INPUT_DIR = RESULT_DIR / "inputs"
CSV_PATH = RESULT_DIR / "combined_methods_runtime.csv"
PLOT_PATH = RESULT_DIR / "combined_methods_runtime_loglog.png"

BATCH_SIZE = 5
LOCAL_DELTA = 5
MULTISCALE_FACTOR = 2
MULTISCALE_BAND = 5

METHODS = [
    "backward",
    "forward",
    "batch5",
    "local_dp_delta5",
    "multiscale_s2_band5",
]

DISPLAY_NAMES = {
    "backward": "Backward energy",
    "forward": "Forward energy",
    "batch5": "Batch seams (b=5)",
    "local_dp_delta5": "Local DP (delta=5)",
    "multiscale_s2_band5": "Multi-scale (s=2, band=5)",
}

COLORS = {
    "backward": (31, 119, 180),
    "forward": (107, 174, 214),
    "batch5": (44, 160, 44),
    "local_dp_delta5": (214, 39, 40),
    "multiscale_s2_band5": (117, 107, 177),
}


def prepare_scaled_inputs(source_path: Path) -> list[tuple[int, int, np.ndarray]]:
    """Create resized input arrays without importing the matplotlib-based script."""
    source = Image.open(source_path).convert("RGB")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    scaled_inputs: list[tuple[int, int, np.ndarray]] = []
    for width, height in TEST_SIZES:
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(INPUT_DIR / f"bench_{width}x{height}.png")
        scaled_inputs.append((width, height, np.asarray(resized, dtype=np.float64)))
    return scaled_inputs


def time_method(method: str, image: np.ndarray, target_width: int) -> float:
    working = image.copy()
    start = time.perf_counter()
    if method == "backward":
        backward_resize(working, target_width=target_width, target_height=None)
    elif method == "forward":
        forward_resize(working, target_width=target_width, target_height=None)
    elif method == "batch5":
        batch_resize(working, target_width=target_width, batch_size=BATCH_SIZE)
    elif method == "local_dp_delta5":
        resize_local_dp(
            working,
            target_width=target_width,
            target_height=None,
            mask=None,
            delta=LOCAL_DELTA,
        )
    elif method == "multiscale_s2_band5":
        multiscale_resize(
            working,
            target_width=target_width,
            target_height=None,
            scale_factor=MULTISCALE_FACTOR,
            band_width=MULTISCALE_BAND,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return time.perf_counter() - start


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


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
            "target_width": target_width,
            "repeats": REPEATS,
        }
        print(f"\n=== {width}x{height} -> {target_width}x{height} ===")
        for method in METHODS:
            times: list[float] = []
            for repeat in range(1, REPEATS + 1):
                seconds = time_method(method, image, target_width)
                times.append(seconds)
                print(f"{DISPLAY_NAMES[method]} repeat={repeat}: {seconds:.4f}s")
            mean, std = summarize(times)
            row[f"{method}_mean"] = mean
            row[f"{method}_std"] = std
        rows.append(row)

    write_csv(rows)
    plot_loglog(rows)
    return rows


def write_csv(rows: list[dict[str, float | int]]) -> None:
    fields = ["width", "height", "pixels", "target_width", "repeats"]
    for method in METHODS:
        fields += [f"{method}_mean", f"{method}_std"]
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fit_log_slope(pixels: list[float], times: list[float]) -> float:
    return float(np.polyfit(np.log(np.asarray(pixels)), np.log(np.asarray(times)), 1)[0])


def plot_loglog(rows: list[dict[str, float | int]]) -> None:
    """Draw a dependency-free log-log line plot using PIL."""
    pixels = [float(row["pixels"]) for row in rows]
    all_times = [
        float(row[f"{method}_mean"])
        for row in rows
        for method in METHODS
    ]

    log_x = [math.log10(x) for x in pixels]
    log_y_values = [math.log10(y) for y in all_times]
    min_x, max_x = min(log_x), max(log_x)
    min_y, max_y = min(log_y_values), max(log_y_values)
    pad_y = (max_y - min_y) * 0.08
    min_y -= pad_y
    max_y += pad_y

    width, height = 1300, 820
    left, right, top, bottom = 115, 330, 70, 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def to_xy(x_value: float, y_value: float) -> tuple[int, int]:
        x = left + int((math.log10(x_value) - min_x) / (max_x - min_x) * plot_w)
        y = top + int((max_y - math.log10(y_value)) / (max_y - min_y) * plot_h)
        return x, y

    # Grid and axes.
    for row in rows:
        x, _ = to_xy(float(row["pixels"]), all_times[0])
        draw.line([(x, top), (x, top + plot_h)], fill=(230, 230, 230), width=1)
        label = f'{int(row["width"])}x{int(row["height"])}'
        draw.text((x - 22, top + plot_h + 14), label, fill=(0, 0, 0), font=font)

    y_ticks = [0.3, 1, 3, 10, 30, 100, 200]
    for tick in y_ticks:
        if min_y <= math.log10(tick) <= max_y:
            _, y = to_xy(pixels[0], tick)
            draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225), width=1)
            draw.text((48, y - 6), f"{tick:g}s", fill=(0, 0, 0), font=font)

    draw.line([(left, top), (left, top + plot_h)], fill=(0, 0, 0), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(0, 0, 0), width=2)

    # Lines.
    for method in METHODS:
        points = [
            to_xy(float(row["pixels"]), float(row[f"{method}_mean"]))
            for row in rows
        ]
        draw.line(points, fill=COLORS[method], width=3)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS[method])

    draw.text((left, 25), "Runtime Comparison Across Seam-Carving Methods (log-log)", fill=(0, 0, 0), font=font)
    draw.text((left + plot_w // 2 - 40, height - 38), "Input image size", fill=(0, 0, 0), font=font)
    draw.text((16, 28), "Average runtime\n(seconds, log)", fill=(0, 0, 0), font=font)

    # Legend with fitted slopes.
    legend_x = left + plot_w + 35
    legend_y = top + 20
    for idx, method in enumerate(METHODS):
        y = legend_y + idx * 45
        times = [float(row[f"{method}_mean"]) for row in rows]
        slope = fit_log_slope(pixels, times)
        draw.line([(legend_x, y + 7), (legend_x + 32, y + 7)], fill=COLORS[method], width=4)
        draw.text(
            (legend_x + 42, y),
            f"{DISPLAY_NAMES[method]}\nslope={slope:.2f}",
            fill=(0, 0, 0),
            font=font,
        )

    image.save(PLOT_PATH)


def main() -> None:
    rows = run_experiment()
    print(f"\nSaved CSV: {CSV_PATH}")
    print(f"Saved plot: {PLOT_PATH}")
    for method in METHODS:
        largest = rows[-1]
        print(
            f"{DISPLAY_NAMES[method]} largest mean: "
            f"{float(largest[f'{method}_mean']):.3f}s"
        )


if __name__ == "__main__":
    main()
