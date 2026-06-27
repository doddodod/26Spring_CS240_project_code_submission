"""Standard resizing and cropping baselines for seam carving experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def standard_resize(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def center_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = image.size
    if target_width > width or target_height > height:
        raise ValueError("crop target must not be larger than the input image")

    left = (width - target_width) // 2
    upper = (height - target_height) // 2
    right = left + target_width
    lower = upper + target_height
    return image.crop((left, upper, right, lower))


def save_array(image: np.ndarray, path: str | Path) -> None:
    ensure_parent(path)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline resize/crop outputs")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--width", type=int, required=True, help="Target width")
    parser.add_argument("--height", type=int, required=True, help="Target height")
    parser.add_argument("--resize-output", default="results/baseline_resize/resize_output.jpg")
    parser.add_argument("--crop-output", default="results/baseline_crop/crop_output.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.input)

    resized = standard_resize(image, args.width, args.height)
    cropped = center_crop(image, args.width, args.height)

    ensure_parent(args.resize_output)
    ensure_parent(args.crop_output)
    resized.save(args.resize_output)
    cropped.save(args.crop_output)


if __name__ == "__main__":
    main()
