"""Generate visualizations of seam-carving energy maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import backward_energy_seam_carving as backward
import forward_energy_seam_carving as forward


def normalize_energy_map(energy: np.ndarray) -> np.ndarray:
    """Normalize an energy map to 8-bit grayscale for visualization."""
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        return np.zeros_like(energy, dtype=np.uint8)

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(energy, dtype=np.uint8)

    norm = (energy - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def save_energy_image(energy: np.ndarray, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalize_energy_map(energy), mode="L").save(output_path)


def forward_cumulative_energy(image: np.ndarray) -> np.ndarray:
    """Compute cumulative forward energy used by the DP seam search."""
    energy = forward.base_energy(image)
    cost_left, cost_up, cost_right = forward.forward_costs(image)
    height, width = energy.shape

    cumulative = energy.copy()
    for row in range(1, height):
        previous = cumulative[row - 1]
        for col in range(width):
            candidates: list[float] = []
            if col > 0:
                candidates.append(previous[col - 1] + cost_left[row, col])
            candidates.append(previous[col] + cost_up[row, col])
            if col < width - 1:
                candidates.append(previous[col + 1] + cost_right[row, col])
            cumulative[row, col] = energy[row, col] + min(candidates)

    # Log compression makes high dynamic-range cumulative maps readable.
    return np.log1p(cumulative - np.min(cumulative))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize seam-carving energy maps")
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "--method",
        choices=["backward", "forward", "both"],
        default="both",
        help="Which energy maps to export",
    )
    parser.add_argument(
        "--out-prefix",
        default="results/energy",
        help="Output path prefix (suffixes are added automatically)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = backward.load_image(args.input)

    prefix = Path(args.out_prefix)
    generated: list[Path] = []

    if args.method in {"backward", "both"}:
        backward_energy = backward.backward_energy(image)
        path = prefix.with_name(prefix.name + "_backward.png")
        save_energy_image(backward_energy, path)
        generated.append(path)

    if args.method in {"forward", "both"}:
        forward_energy = forward_cumulative_energy(image)
        path = prefix.with_name(prefix.name + "_forward.png")
        save_energy_image(forward_energy, path)
        generated.append(path)

    for path in generated:
        print(f"Saved energy map: {path}")


if __name__ == "__main__":
    main()
