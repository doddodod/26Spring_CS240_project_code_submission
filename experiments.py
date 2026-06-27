"""Run visual seam carving experiments on one input image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import backward_energy_seam_carving as backward
import forward_energy_seam_carving as forward
from baselines import center_crop, standard_resize
from visualization import make_comparison


def stem(path: str | Path) -> str:
    return Path(path).stem


def parse_target_size(input_path: str | Path, width: int | None, height: int | None) -> tuple[int, int]:
    image = Image.open(input_path)
    original_width, original_height = image.size
    target_width = width if width is not None else original_width
    target_height = height if height is not None else original_height
    if target_width > original_width or target_height > original_height:
        raise ValueError("target size must not exceed the input size for this experiment script")
    return target_width, target_height


def run_experiment(input_path: str | Path, target_width: int, target_height: int) -> None:
    name = stem(input_path)
    original = Image.open(input_path).convert("RGB")

    backward_path = Path("results/backward") / f"{name}_backward_{target_width}x{target_height}.jpg"
    forward_path = Path("results/forward") / f"{name}_forward_{target_width}x{target_height}.jpg"
    resize_path = Path("results/baseline_resize") / f"{name}_resize_{target_width}x{target_height}.jpg"
    crop_path = Path("results/baseline_crop") / f"{name}_crop_{target_width}x{target_height}.jpg"
    comparison_path = Path("results/comparisons") / f"{name}_comparison_{target_width}x{target_height}.jpg"

    backward_path.parent.mkdir(parents=True, exist_ok=True)
    forward_path.parent.mkdir(parents=True, exist_ok=True)
    resize_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)

    backward_image = backward.resize(backward.load_image(input_path), target_width, target_height)
    forward_image = forward.resize(forward.load_image(input_path), target_width, target_height)

    backward.save_image(backward_image, backward_path)
    forward.save_image(forward_image, forward_path)
    standard_resize(original, target_width, target_height).save(resize_path)
    center_crop(original, target_width, target_height).save(crop_path)

    make_comparison(
        [input_path, resize_path, crop_path, backward_path, forward_path],
        ["Original", "Scaling", "Cropping", "Backward", "Forward"],
        comparison_path,
    )

    print(f"Saved comparison: {comparison_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seam carving visual experiment")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--width", type=int, help="Target width")
    parser.add_argument("--height", type=int, help="Target height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_width, target_height = parse_target_size(args.input, args.width, args.height)
    run_experiment(args.input, target_width, target_height)


if __name__ == "__main__":
    main()
