"""Runtime and scalability experiments for the local seam carving implementation."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from PIL import Image

import backward_energy_seam_carving as backward
import forward_energy_seam_carving as forward


def prepare_image(input_path: str | Path, size: int) -> Path:
    source = Image.open(input_path).convert("RGB")
    resized = source.resize((size, size), Image.Resampling.LANCZOS)
    output_path = Path("results/timing") / f"input_{size}x{size}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    return output_path


def time_method(method: str, input_path: str | Path, target_width: int) -> float:
    if method == "backward":
        image = backward.load_image(input_path)
        start = time.perf_counter()
        backward.resize(image, target_width, None)
    elif method == "forward":
        image = forward.load_image(input_path)
        start = time.perf_counter()
        forward.resize(image, target_width, None)
    else:
        raise ValueError(f"unknown method: {method}")
    return time.perf_counter() - start


def run_timing(input_path: str | Path, sizes: list[int], removal_ratio: float, repeats: int) -> Path:
    output_csv = Path("results/timing/runtime.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "size",
                "pixels",
                "original_width",
                "target_width",
                "removed_seams",
                "theoretical_work",
                "repeat",
                "seconds",
            ],
        )
        writer.writeheader()

        for size in sizes:
            sized_input = prepare_image(input_path, size)
            removed = max(1, int(size * removal_ratio))
            target_width = size - removed
            for method in ("backward", "forward"):
                for repeat in range(1, repeats + 1):
                    seconds = time_method(method, sized_input, target_width)
                    writer.writerow(
                        {
                            "method": method,
                            "size": size,
                            "pixels": size * size,
                            "original_width": size,
                            "target_width": target_width,
                            "removed_seams": removed,
                            "theoretical_work": removed * size * size,
                            "repeat": repeat,
                            "seconds": f"{seconds:.6f}",
                        }
                    )
                    print(f"{method:8s} size={size} repeat={repeat} seconds={seconds:.3f}")

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run runtime scalability experiment")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 300, 400])
    parser.add_argument("--removal-ratio", type=float, default=0.10)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = run_timing(args.input, args.sizes, args.removal_ratio, args.repeats)
    print(f"Saved timing CSV: {output_csv}")


if __name__ == "__main__":
    main()
