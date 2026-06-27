"""Generate poster-ready application demos for seam carving.

The demos use medium-sized copies of the original images so that the pure
Python implementation can finish in a reasonable amount of time while still
showing the qualitative behavior of each operation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import forward_energy_seam_carving as forward
from visualization import make_comparison


ROOT = Path(".")
INPUT_ROOT = ROOT / "picture" / "input"
MASK_ROOT = ROOT / "picture" / "masks"
DEMO_ROOT = ROOT / "results" / "poster_demos"
DEMO_INPUTS = DEMO_ROOT / "inputs"
DEMO_OUTPUTS = DEMO_ROOT / "outputs"
DEMO_COMPARISONS = DEMO_ROOT / "comparisons"


def ensure_dirs() -> None:
    for directory in (DEMO_INPUTS, DEMO_OUTPUTS, DEMO_COMPARISONS):
        directory.mkdir(parents=True, exist_ok=True)


def load_rgb_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def save_array(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def resize_image_file(input_path: Path, output_path: Path, size: tuple[int, int]) -> Path:
    image = Image.open(input_path).convert("RGB")
    resized = image.resize(size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    return output_path


def resize_mask_file(input_path: Path, output_path: Path, size: tuple[int, int]) -> Path:
    mask = Image.open(input_path).convert("L")
    resized = mask.resize(size, Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    return output_path


def run_vertical_removal() -> None:
    input_path = resize_image_file(INPUT_ROOT / "castle.jpg", DEMO_INPUTS / "castle_420x285.jpg", (420, 285))
    output_path = DEMO_OUTPUTS / "vertical_removal_castle_forward.jpg"
    result = forward.resize(load_rgb_array(input_path), target_width=280, target_height=None)
    save_array(result, output_path)
    make_comparison(
        [input_path, output_path],
        ["Original", "Vertical Removal"],
        DEMO_COMPARISONS / "01_vertical_removal_castle.jpg",
        tile_height=220,
    )


def run_horizontal_removal() -> None:
    input_path = resize_image_file(INPUT_ROOT / "museum.jpg", DEMO_INPUTS / "museum_315x400.jpg", (315, 400))
    output_path = DEMO_OUTPUTS / "horizontal_removal_museum_forward.jpg"
    result = forward.resize(load_rgb_array(input_path), target_width=None, target_height=270)
    save_array(result, output_path)
    make_comparison(
        [input_path, output_path],
        ["Original", "Horizontal Removal"],
        DEMO_COMPARISONS / "02_horizontal_removal_museum.jpg",
        tile_height=220,
    )


def run_protective_mask() -> None:
    size = (480, 270)
    input_path = resize_image_file(INPUT_ROOT / "ratatouille.jpg", DEMO_INPUTS / "ratatouille_480x270.jpg", size)
    mask_path = resize_mask_file(
        MASK_ROOT / "ratatouille_mask.jpg", DEMO_INPUTS / "ratatouille_mask_480x270.png", size
    )
    image = load_rgb_array(input_path)
    mask = forward.load_mask(mask_path)

    unprotected_path = DEMO_OUTPUTS / "protective_mask_ratatouille_unprotected.jpg"
    protected_path = DEMO_OUTPUTS / "protective_mask_ratatouille_protected.jpg"
    save_array(forward.resize(image.copy(), target_width=340, target_height=None), unprotected_path)
    save_array(forward.resize(image.copy(), target_width=340, target_height=None, mask=mask), protected_path)

    make_comparison(
        [input_path, mask_path, unprotected_path, protected_path],
        ["Original", "Mask", "No Mask", "Protected"],
        DEMO_COMPARISONS / "03_protective_mask_ratatouille.jpg",
        tile_height=220,
    )


def run_seam_insertion() -> None:
    input_path = resize_image_file(INPUT_ROOT / "shore.jpg", DEMO_INPUTS / "shore_300x400.jpg", (300, 400))
    output_path = DEMO_OUTPUTS / "seam_insertion_shore_forward.jpg"
    result = forward.resize(load_rgb_array(input_path), target_width=380, target_height=None)
    save_array(result, output_path)
    make_comparison(
        [input_path, output_path],
        ["Original", "Seam Insertion"],
        DEMO_COMPARISONS / "04_seam_insertion_shore.jpg",
        tile_height=220,
    )


def run_object_removal() -> None:
    size = (480, 260)
    input_path = resize_image_file(INPUT_ROOT / "eiffel.jpg", DEMO_INPUTS / "eiffel_480x260.jpg", size)
    mask_path = resize_mask_file(MASK_ROOT / "eiffel_mask.jpg", DEMO_INPUTS / "eiffel_mask_480x260.png", size)
    image = load_rgb_array(input_path)
    remove_mask = forward.load_mask(mask_path)

    output_path = DEMO_OUTPUTS / "object_removal_eiffel_forward.jpg"
    result = forward.object_removal(image, remove_mask)
    save_array(result, output_path)
    make_comparison(
        [input_path, mask_path, output_path],
        ["Original", "Removal Mask", "Object Removal"],
        DEMO_COMPARISONS / "05_object_removal_eiffel.jpg",
        tile_height=220,
    )


def main() -> None:
    ensure_dirs()
    jobs = [
        ("Vertical Removal", run_vertical_removal),
        ("Horizontal Removal", run_horizontal_removal),
        ("Protective Mask", run_protective_mask),
        ("Seam Insertion", run_seam_insertion),
        ("Object Removal", run_object_removal),
    ]
    for label, job in jobs:
        print(f"Running demo: {label}", flush=True)
        job()
    print(f"Saved poster demos under: {DEMO_ROOT}")


if __name__ == "__main__":
    main()
