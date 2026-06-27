"""Multi-scale (coarse-to-fine) seam carving (algorithmic improvement #3).

Running the seam-finding DP on a full-resolution image is expensive because the
DP visits every pixel. A classic divide-and-conquer speedup is to:

1. Downsample the image (here by simple stride decimation, so the step is fully
   transparent and uses no external resizing library).
2. Run the DP on the small image to locate the seam coarsely.
3. Map the coarse seam back to full resolution (multiply indices by the scale).
4. Re-run the DP on the full image but only inside a narrow band around the
   mapped path, producing the final fine seam.

The coarse DP is cheap (few pixels) and the fine DP visits only
``O(height * band_width)`` cells instead of ``O(height * width)``, which is the
source of the speedup. The trade-off is a small loss of optimality / visual
quality, because the fine seam is constrained to a band and cannot deviate far
from the coarse estimate.

The baseline files are not modified; their primitives are imported directly.
"""

from __future__ import annotations

import argparse

import numpy as np

from backward_energy_seam_carving import (
    backward_energy,
    find_vertical_seam,
    load_image,
    remove_vertical_seam,
    save_image,
)

DEFAULT_SCALE = 2
DEFAULT_BAND = 5


def downsample(image: np.ndarray, scale: int) -> np.ndarray:
    """Decimate an image by ``scale`` using stride slicing (nearest neighbor)."""
    return image[::scale, ::scale, :]


def map_seam_to_full(
    low_seam: np.ndarray,
    full_height: int,
    full_width: int,
    scale: int,
) -> np.ndarray:
    """Upscale a low-resolution seam to a per-row center column at full resolution."""
    low_height = low_seam.shape[0]
    center = np.empty(full_height, dtype=np.int32)
    for row in range(full_height):
        low_row = min(row // scale, low_height - 1)
        center[row] = min(int(low_seam[low_row]) * scale, full_width - 1)
    return center


def find_seam_in_band(
    energy: np.ndarray,
    center: np.ndarray,
    band: int,
) -> np.ndarray:
    """Run the seam DP restricted to ``center +/- band`` columns per row.

    Only the cells inside the band are evaluated, so the DP costs
    ``O(height * (2*band + 1))`` instead of ``O(height * width)``.
    """
    height, width = energy.shape
    cost = np.full((height, width), np.inf, dtype=np.float64)
    parent = np.zeros((height, width), dtype=np.int32)

    lo = max(0, int(center[0]) - band)
    hi = min(width, int(center[0]) + band + 1)
    cost[0, lo:hi] = energy[0, lo:hi]

    for row in range(1, height):
        lo = max(0, int(center[row]) - band)
        hi = min(width, int(center[row]) + band + 1)
        for col in range(lo, hi):
            left = max(col - 1, 0)
            right = min(col + 2, width)
            previous = cost[row - 1, left:right]
            offset = int(np.argmin(previous))
            parent[row, col] = left + offset
            cost[row, col] = energy[row, col] + previous[offset]

    last_lo = max(0, int(center[height - 1]) - band)
    last_hi = min(width, int(center[height - 1]) + band + 1)
    seam = np.zeros(height, dtype=np.int32)
    seam[-1] = last_lo + int(np.argmin(cost[height - 1, last_lo:last_hi]))
    for row in range(height - 2, -1, -1):
        seam[row] = parent[row + 1, seam[row + 1]]
    return seam


def find_multiscale_seam(
    image: np.ndarray,
    scale_factor: int = DEFAULT_SCALE,
    band_width: int = DEFAULT_BAND,
) -> np.ndarray:
    """Locate one vertical seam using the coarse-to-fine strategy."""
    height, width = image.shape[:2]
    # Fall back to a full DP when the image is too small to downsample usefully.
    if scale_factor < 2 or height < 2 * scale_factor or width < 2 * scale_factor:
        return find_vertical_seam(backward_energy(image))

    low = downsample(image, scale_factor)
    low_seam = find_vertical_seam(backward_energy(low))
    center = map_seam_to_full(low_seam, height, width, scale_factor)
    energy = backward_energy(image)
    return find_seam_in_band(energy, center, band_width)


def multiscale_resize(
    image: np.ndarray,
    target_width: int,
    scale_factor: int = DEFAULT_SCALE,
    band_width: int = DEFAULT_BAND,
) -> np.ndarray:
    """Reduce image width to ``target_width`` using multi-scale seam search."""
    if not 1 <= target_width <= image.shape[1]:
        raise ValueError("target_width must be between 1 and the current width")
    if band_width < 3:
        raise ValueError("band_width must be at least 3 to cover the DP stencil")

    while image.shape[1] > target_width:
        seam = find_multiscale_seam(image, scale_factor, band_width)
        image = remove_vertical_seam(image, seam)
    return image


def resize(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
    scale_factor: int = DEFAULT_SCALE,
    band_width: int = DEFAULT_BAND,
) -> np.ndarray:
    """Resize width and/or height using multi-scale seam search (removal only)."""
    if target_width is not None and target_width < image.shape[1]:
        image = multiscale_resize(image, target_width, scale_factor, band_width)
    if target_height is not None and target_height < image.shape[0]:
        image = np.transpose(image, (1, 0, 2))
        image = multiscale_resize(image, target_height, scale_factor, band_width)
        image = np.transpose(image, (1, 0, 2))
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-scale seam carving")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, help="Target output width")
    parser.add_argument("--height", type=int, help="Target output height")
    parser.add_argument("--scale-factor", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--band-width", type=int, default=DEFAULT_BAND)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.input)
    result = resize(
        image,
        args.width,
        args.height,
        scale_factor=args.scale_factor,
        band_width=args.band_width,
    )
    save_image(result, args.output)


if __name__ == "__main__":
    main()
