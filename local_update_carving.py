"""Local energy-update seam carving (algorithmic improvement #2).

The baseline recomputes the *entire* gradient energy map after every seam
removal. However, deleting one vertical seam only changes pixel adjacencies in a
narrow band around the seam path: pixels far from the seam keep the same
neighbors (their column index merely shifts), so their energy is unchanged.

This module implements a hybrid optimization:

* **Local energy update** -- after a seam is removed, the energy map is shrunk by
  deleting the seam column per row, and Sobel energy is recomputed only for a
  conservative band (``+/- radius`` columns, ``radius >= 2``) around the seam.
* **Global DP** -- the dynamic program still runs over the full (updated) energy
  map, so seam optimality is identical to the baseline.

Because the local update reproduces exactly the same energy values the baseline
would compute, the carved output is bit-for-bit identical to the baseline; only
the redundant full-map energy recomputation is avoided.

The baseline files are not modified; their primitives are imported directly.
"""

from __future__ import annotations

import argparse

import numpy as np

from backward_energy_seam_carving import (
    backward_energy,
    find_vertical_seam,
    grayscale,
    load_image,
    remove_vertical_seam,
    save_image,
)

_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
_SOBEL_Y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64)

# Conservative half-width (in columns) of the band recomputed after each removal.
# A vertical seam only alters adjacencies at its own column +/- 1; the Sobel
# stencil widens that by one more, so radius 2 is sufficient for exact energy.
DEFAULT_RADIUS = 2


def _sobel_energy_strip(gray: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Recompute Sobel energy for columns ``[lo, hi)`` of ``gray``.

    One real neighbor column is included on each side (except at the true image
    border, where edge padding matches the baseline), so the returned values for
    columns ``[lo, hi)`` are identical to a full-image recomputation.
    """
    height, width = gray.shape
    a = max(0, lo - 1)
    b = min(width, hi + 1)
    strip = gray[:, a:b]

    padded = np.pad(strip, ((1, 1), (1, 1)), mode="edge")

    def convolve(kernel: np.ndarray) -> np.ndarray:
        return (
            kernel[0, 0] * padded[:-2, :-2]
            + kernel[0, 1] * padded[:-2, 1:-1]
            + kernel[0, 2] * padded[:-2, 2:]
            + kernel[1, 0] * padded[1:-1, :-2]
            + kernel[1, 1] * padded[1:-1, 1:-1]
            + kernel[1, 2] * padded[1:-1, 2:]
            + kernel[2, 0] * padded[2:, :-2]
            + kernel[2, 1] * padded[2:, 1:-1]
            + kernel[2, 2] * padded[2:, 2:]
        )

    grad_x = convolve(_SOBEL_X)
    grad_y = convolve(_SOBEL_Y)
    strip_energy = np.abs(grad_x) + np.abs(grad_y)
    # Map strip-local columns back to the requested global range.
    return strip_energy[:, lo - a : hi - a]


def _delete_seam_columns(array2d: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Delete one column per row from a 2D array (energy map or grayscale)."""
    height, width = array2d.shape
    carved = np.empty((height, width - 1), dtype=array2d.dtype)
    for row in range(height):
        carved[row] = np.delete(array2d[row], int(seam[row]))
    return carved


def local_update_resize(
    image: np.ndarray,
    target_width: int,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    """Reduce image width to ``target_width`` using local energy updates.

    The full energy map and grayscale image are computed once. After each seam
    removal only a band around the seam is recomputed, while the DP remains
    global.

    Args:
        image: Image array of shape ``(height, width, channels)``.
        target_width: Desired width (1 <= target_width <= current width).
        radius: Half-width of the recomputed band (must be >= 2 for correctness).

    Returns:
        The width-reduced image (identical to the baseline output).
    """
    if not 1 <= target_width <= image.shape[1]:
        raise ValueError("target_width must be between 1 and the current width")
    if radius < 2:
        raise ValueError("radius must be at least 2 to guarantee correctness")

    gray = grayscale(image)
    energy = backward_energy(image)

    while image.shape[1] > target_width:
        seam = find_vertical_seam(energy)
        image = remove_vertical_seam(image, seam)

        gray = _delete_seam_columns(gray, seam)
        energy = _delete_seam_columns(energy, seam)

        new_width = gray.shape[1]
        lo = max(0, int(seam.min()) - radius)
        hi = min(new_width, int(seam.max()) + radius)
        energy[:, lo:hi] = _sobel_energy_strip(gray, lo, hi)

    return image


def resize(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    """Resize width and/or height using local energy updates (removal only)."""
    if target_width is not None and target_width < image.shape[1]:
        image = local_update_resize(image, target_width, radius)
    if target_height is not None and target_height < image.shape[0]:
        image = np.transpose(image, (1, 0, 2))
        image = local_update_resize(image, target_height, radius)
        image = np.transpose(image, (1, 0, 2))
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-update seam carving")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, help="Target output width")
    parser.add_argument("--height", type=int, help="Target output height")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.input)
    result = resize(image, args.width, args.height, args.radius)
    save_image(result, args.output)


if __name__ == "__main__":
    main()
