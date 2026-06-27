"""Batch seam extraction for seam carving (algorithmic improvement #1).

The baseline implementation removes a single seam per dynamic-programming (DP)
pass: it recomputes the energy map, runs the DP, removes one seam, and repeats.
Reducing the width by half therefore requires ``W/2`` separate energy
recomputations and seam-removal operations.

This module extracts ``k`` non-crossing seams from a *single* energy map using a
penalty method, then removes all ``k`` seams in one vectorized pass. This avoids
recomputing the energy map and re-running per-seam removal for every column,
trading a small amount of optimality (later seams are slightly sub-optimal
because the energy map is not refreshed within a batch) for fewer full passes.

The module does not modify the baseline files; it imports their energy and seam
primitives so the core DP / backtracking / removal logic stays identical.
"""

from __future__ import annotations

import argparse

import numpy as np

from backward_energy_seam_carving import (
    backward_energy,
    find_vertical_seam,
    load_image,
    save_image,
)


def find_multiple_seams(
    energy: np.ndarray,
    k: int,
    penalty_radius: int = 2,
    penalty_weight: float = 1e6,
) -> list[np.ndarray]:
    """Extract ``k`` non-crossing vertical seams from one energy map.

    After each seam is located, a penalty is added to the energy values within
    ``+/- penalty_radius`` columns of the seam path. The penalty makes those
    pixels prohibitively expensive, so the next seam is pushed away and the
    extracted seams do not cross.

    Args:
        energy: 2D energy map of shape ``(height, width)``.
        k: Number of seams to extract.
        penalty_radius: Half-width (in columns) of the penalty band per row.
        penalty_weight: Energy added to penalized pixels.

    Returns:
        A list of ``k`` seams, each a length-``height`` array of column indices,
        all expressed in the coordinate frame of the input ``energy`` map.
    """
    if k <= 0:
        return []

    height, width = energy.shape
    if k > width:
        raise ValueError("cannot extract more seams than the image width")

    work = energy.astype(np.float64, copy=True)
    rows = np.arange(height)
    seams: list[np.ndarray] = []

    for _ in range(k):
        seam = find_vertical_seam(work)
        seams.append(seam)
        # Soft penalty in a band around the seam discourages near-crossings and
        # keeps the extracted seams visually separated.
        for offset in range(-penalty_radius, penalty_radius + 1):
            cols = np.clip(seam + offset, 0, width - 1)
            work[rows, cols] += penalty_weight
        # Hard block of the exact seam pixels guarantees that no later seam can
        # reuse the same pixel in a row, so all seams are removable together.
        work[rows, seam] = np.inf

    return seams


def _build_keep_mask(seams: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Return a boolean keep-mask with exactly ``len(seams)`` False per row."""
    keep = np.ones((height, width), dtype=bool)
    rows = np.arange(height)
    for seam in seams:
        keep[rows, seam] = False

    removed_per_row = (~keep).sum(axis=1)
    if not np.all(removed_per_row == len(seams)):
        # Two seams share a pixel in some row (penalty band too small). The caller
        # should increase penalty_radius; we fail loudly to avoid silent corruption.
        raise ValueError(
            "batch seams overlap in at least one row; increase penalty_radius"
        )
    return keep


def remove_multiple_seams(
    image: np.ndarray,
    seams: list[np.ndarray],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove several non-crossing seams from an image in one pass.

    Removal is done with a per-row boolean keep-mask, which is equivalent to
    deleting the seam columns from right to left (the largest column index in a
    row is dropped first, so smaller indices stay valid).

    Args:
        image: Image array of shape ``(height, width, channels)``.
        seams: Non-crossing seams in the image's coordinate frame.
        mask: Optional 2D mask kept aligned with the image.

    Returns:
        The carved image and the (optionally) carved mask.
    """
    if not seams:
        return image, mask

    height, width, channels = image.shape
    keep = _build_keep_mask(seams, height, width)
    num_remove = len(seams)

    carved = image[keep].reshape(height, width - num_remove, channels)
    carved_mask = None
    if mask is not None:
        carved_mask = mask[keep].reshape(height, width - num_remove)
    return carved, carved_mask


def batch_resize(
    image: np.ndarray,
    target_width: int,
    batch_size: int = 10,
    mask: np.ndarray | None = None,
    penalty_radius: int = 2,
    penalty_weight: float = 1e6,
) -> np.ndarray:
    """Reduce image width to ``target_width`` using batched seam removal.

    Each iteration extracts ``min(batch_size, remaining)`` seams from a single
    energy map and removes them together. ``batch_size == 1`` reproduces the
    baseline one-seam-per-pass behavior.

    Args:
        image: Image array of shape ``(height, width, channels)``.
        target_width: Desired width (must be between 1 and the current width).
        batch_size: Number of seams removed per energy recomputation.
        mask: Optional protective mask kept aligned with the image.
        penalty_radius: Penalty band half-width passed to ``find_multiple_seams``.
        penalty_weight: Penalty magnitude passed to ``find_multiple_seams``.

    Returns:
        The width-reduced image.
    """
    if not 1 <= target_width <= image.shape[1]:
        raise ValueError("target_width must be between 1 and the current width")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    while image.shape[1] > target_width:
        remaining = image.shape[1] - target_width
        k = min(batch_size, remaining)
        energy = backward_energy(image)
        if mask is not None:
            from backward_energy_seam_carving import apply_masks

            energy = apply_masks(energy, mask)
        seams = find_multiple_seams(energy, k, penalty_radius, penalty_weight)
        image, mask = remove_multiple_seams(image, seams, mask)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch seam-carving width reduction")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, required=True, help="Target width")
    parser.add_argument("--batch-size", type=int, default=10, help="Seams per batch")
    parser.add_argument("--penalty-radius", type=int, default=2)
    parser.add_argument("--penalty-weight", type=float, default=1e6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.input)
    result = batch_resize(
        image,
        target_width=args.width,
        batch_size=args.batch_size,
        penalty_radius=args.penalty_radius,
        penalty_weight=args.penalty_weight,
    )
    save_image(result, args.output)


if __name__ == "__main__":
    main()
