"""Generate poster comparisons against scaling and cropping baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import forward_energy_seam_carving as forward
from baselines import center_crop, standard_resize
from visualization import make_comparison


INPUT_ROOT = Path("picture/input")
RESULT_ROOT = Path("results/poster_demos")
INPUT_DIR = RESULT_ROOT / "baseline_inputs"
OUTPUT_DIR = RESULT_ROOT / "baseline_outputs"
COMPARISON_DIR = RESULT_ROOT / "baseline_comparisons"


def save_array(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def resize_input(source_path: Path, output_path: Path, size: tuple[int, int]) -> Path:
    image = Image.open(source_path).convert("RGB")
    resized = image.resize(size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    return output_path


def run_case(
    name: str,
    source_file: str,
    input_size: tuple[int, int],
    target_size: tuple[int, int],
    include_crop: bool = True,
) -> None:
    input_path = resize_input(INPUT_ROOT / source_file, INPUT_DIR / f"{name}_input.jpg", input_size)
    original = Image.open(input_path).convert("RGB")
    target_width, target_height = target_size

    scaling_path = OUTPUT_DIR / f"{name}_scaling.jpg"
    cropping_path = OUTPUT_DIR / f"{name}_cropping.jpg"
    seam_path = OUTPUT_DIR / f"{name}_forward_seam.jpg"
    comparison_path = COMPARISON_DIR / f"{name}_baseline_comparison.jpg"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    standard_resize(original, target_width, target_height).save(scaling_path)
    if include_crop:
        center_crop(original, target_width, target_height).save(cropping_path)

    image_array = np.asarray(original, dtype=np.float64)
    seam_result = forward.resize(image_array, target_width=target_width, target_height=target_height)
    save_array(seam_result, seam_path)

    if include_crop:
        image_paths = [input_path, scaling_path, cropping_path, seam_path]
        labels = ["Original", "Scaling", "Cropping", "Forward Seam"]
    else:
        image_paths = [input_path, scaling_path, seam_path]
        labels = ["Original", "Scaling", "Forward Seam"]

    make_comparison(image_paths, labels, comparison_path, tile_height=220)
    print(f"Saved: {comparison_path}", flush=True)


def main() -> None:
    cases = [
        ("vertical_removal_castle", "castle.jpg", (360, 244), (240, 244), True),
        ("horizontal_removal_museum", "museum.jpg", (260, 330), (260, 220), True),
        ("seam_insertion_shore", "shore.jpg", (240, 320), (320, 320), False),
    ]
    for case in cases:
        print(f"Running baseline comparison: {case[0]}", flush=True)
        run_case(*case)
    print(f"Saved baseline comparisons under: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
