"""Forward-energy seam carving for content-aware image resizing.

Usage:
    python forward_energy_seam_carving.py picture/input.jpg output.jpg --width 500 --height 350

Forward energy penalizes the new neighboring pixel differences created after a
seam is removed, which usually reduces visual artifacts compared with backward
energy seam carving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Energy offsets used to steer seams through masked regions. These are large
# relative to real gradient magnitudes so that protective regions are avoided
# and removal regions are always emptied first.
PROTECT_ENERGY = 1e6
REMOVE_ENERGY = -1e8
MASK_THRESHOLD = 10.0


def load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def save_image(image: np.ndarray, path: str | Path) -> None:
    output = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(output).save(path)


def grayscale(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def pixel_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.abs(left - right)))


def forward_costs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute forward costs for moving up-left, up, and up-right."""
    gray = grayscale(image)
    height, width = gray.shape
    cost_left = np.zeros((height, width), dtype=np.float64)
    cost_up = np.zeros((height, width), dtype=np.float64)
    cost_right = np.zeros((height, width), dtype=np.float64)

    for row in range(1, height):
        for col in range(width):
            left_col = max(col - 1, 0)
            right_col = min(col + 1, width - 1)
            middle_cost = abs(gray[row, right_col] - gray[row, left_col])
            up_pixel = gray[row - 1, col]

            cost_left[row, col] = middle_cost + abs(up_pixel - gray[row, left_col])
            cost_up[row, col] = middle_cost
            cost_right[row, col] = middle_cost + abs(up_pixel - gray[row, right_col])

    # Bending toward a non-existent neighbour is illegal at the borders. The DP
    # already skips these directions, but we mark them explicitly so the cost
    # arrays are self-consistent and safe to reuse.
    cost_left[:, 0] = np.inf
    cost_right[:, width - 1] = np.inf

    return cost_left, cost_up, cost_right


def base_energy(image: np.ndarray) -> np.ndarray:
    gray = grayscale(image)
    grad_y, grad_x = np.gradient(gray)
    return np.abs(grad_x) + np.abs(grad_y)


def apply_masks(
    energy: np.ndarray,
    mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Bias the energy map so seams avoid protected pixels and prefer removal pixels."""
    energy = energy.copy()
    if mask is not None:
        energy[mask > MASK_THRESHOLD] = PROTECT_ENERGY
    if remove_mask is not None:
        energy[remove_mask > MASK_THRESHOLD] = REMOVE_ENERGY
    return energy


def find_vertical_seam_forward(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return column indices of the minimum forward-energy vertical seam."""
    energy = apply_masks(base_energy(image), mask, remove_mask)
    cost_left, cost_up, cost_right = forward_costs(image)
    height, width = energy.shape

    cumulative = energy.copy()
    parent = np.zeros((height, width), dtype=np.int32)
    # Explicit DP base case for readability.
    cumulative[0, :] = energy[0, :]
    parent[0, :] = 0

    for row in range(1, height):
        previous = cumulative[row - 1]
        for col in range(width):
            candidates: list[tuple[float, int]] = []
            if col > 0:
                candidates.append((previous[col - 1] + cost_left[row, col], col - 1))
            candidates.append((previous[col] + cost_up[row, col], col))
            if col < width - 1:
                candidates.append((previous[col + 1] + cost_right[row, col], col + 1))

            best_cost, best_parent = min(candidates, key=lambda item: item[0])
            cumulative[row, col] = energy[row, col] + best_cost
            parent[row, col] = best_parent

    seam = np.zeros(height, dtype=np.int32)
    seam[-1] = int(np.argmin(cumulative[-1]))
    for row in range(height - 2, -1, -1):
        seam[row] = parent[row + 1, seam[row + 1]]
    return seam


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    height, width, channels = image.shape
    carved = np.empty((height, width - 1, channels), dtype=image.dtype)
    for row in range(height):
        col = seam[row]
        carved[row] = np.delete(image[row], col, axis=0)
    return carved


def remove_seam_from_mask(mask: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Drop one column per row from a 2D mask to keep it aligned with the image."""
    height, width = mask.shape
    carved = np.empty((height, width - 1), dtype=mask.dtype)
    for row in range(height):
        carved[row] = np.delete(mask[row], int(seam[row]))
    return carved


def insert_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Insert one vertical seam, filling it with the average of its neighbours."""
    height, width, channels = image.shape
    expanded = np.zeros((height, width + 1, channels), dtype=image.dtype)
    for row in range(height):
        col = min(int(seam[row]), width - 1)
        if col == 0:
            new_pixel = (image[row, 0] + image[row, 1]) / 2.0
        else:
            new_pixel = (image[row, col - 1] + image[row, col]) / 2.0
        expanded[row, :col] = image[row, :col]
        expanded[row, col] = new_pixel
        expanded[row, col + 1:] = image[row, col:]
    return expanded


def insert_seam_into_mask(mask: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Insert one column per row into a 2D mask, mirroring image seam insertion."""
    height, width = mask.shape
    expanded = np.zeros((height, width + 1), dtype=mask.dtype)
    for row in range(height):
        col = min(int(seam[row]), width - 1)
        expanded[row, :col] = mask[row, :col]
        expanded[row, col] = mask[row, col]
        expanded[row, col + 1:] = mask[row, col:]
    return expanded


def find_seam(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one minimum forward-energy vertical seam, honouring optional masks."""
    return find_vertical_seam_forward(image, mask, remove_mask)


def remove_seams(
    image: np.ndarray, num_remove: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    for _ in range(num_remove):
        seam = find_seam(image, mask)
        image = remove_vertical_seam(image, seam)
        if mask is not None:
            mask = remove_seam_from_mask(mask, seam)
    return image, mask


def insert_seams(
    image: np.ndarray, num_add: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Enlarge the image by re-inserting the lowest-energy seams.

    The seams are first located on a temporary copy (removing them one by one),
    then inserted back into the original image in reverse order. Indices of the
    not-yet-inserted seams are shifted to account for the newly added column.
    """
    record: list[np.ndarray] = []
    temp_image = image.copy()
    temp_mask = mask.copy() if mask is not None else None
    for _ in range(num_add):
        seam = find_seam(temp_image, temp_mask)
        record.append(seam)
        temp_image = remove_vertical_seam(temp_image, seam)
        if temp_mask is not None:
            temp_mask = remove_seam_from_mask(temp_mask, seam)

    record.reverse()
    for _ in range(num_add):
        seam = record.pop()
        image = insert_vertical_seam(image, seam)
        if mask is not None:
            mask = insert_seam_into_mask(mask, seam)
        # Each recorded seam lives in the coordinate frame where the earlier
        # seams were *removed*. Re-inserting one earlier seam shifts every
        # not-yet-inserted index at/after it by two: +1 to undo the removal
        # frame and +1 for the freshly inserted column.
        for remaining in record:
            remaining[remaining >= seam] += 2
    return image, mask


def _resize_width(
    image: np.ndarray, target_width: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    delta = target_width - image.shape[1]
    if delta < 0:
        return remove_seams(image, -delta, mask)
    if delta > 0:
        return insert_seams(image, delta, mask)
    return image, mask


def resize(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Resize width and/or height by removing or inserting seams.

    Targets larger than the current size trigger seam insertion; smaller targets
    trigger seam removal. Height changes are handled by transposing the image so
    the same vertical-seam machinery applies to rows.
    """
    if target_width is not None:
        image, mask = _resize_width(image, target_width, mask)
    if target_height is not None and target_height != image.shape[0]:
        image = np.transpose(image, (1, 0, 2))
        t_mask = mask.T if mask is not None else None
        image, t_mask = _resize_width(image, target_height, t_mask)
        image = np.transpose(image, (1, 0, 2))
        mask = t_mask.T if t_mask is not None else None
    return image


def object_removal(
    image: np.ndarray,
    remove_mask: np.ndarray,
    mask: np.ndarray | None = None,
    horizontal: bool = False,
) -> np.ndarray:
    """Remove the masked object by repeatedly carving seams through it.

    Seams are carved until no removal pixels remain, then the image is grown back
    to its original size with seam insertion. Setting ``horizontal`` removes the
    object using horizontal seams instead of vertical ones.
    """
    target_height, target_width = image.shape[:2]
    if horizontal:
        image = np.transpose(image, (1, 0, 2))
        remove_mask = remove_mask.T
        if mask is not None:
            mask = mask.T

    while np.count_nonzero(remove_mask > MASK_THRESHOLD) > 0:
        seam = find_seam(image, mask, remove_mask)
        image = remove_vertical_seam(image, seam)
        remove_mask = remove_seam_from_mask(remove_mask, seam)
        if mask is not None:
            mask = remove_seam_from_mask(mask, seam)

    restore = (target_height if horizontal else target_width) - image.shape[1]
    if restore > 0:
        image, _ = insert_seams(image, restore, mask)
    if horizontal:
        image = np.transpose(image, (1, 0, 2))
    return image


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-energy seam carving")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, help="Target output width")
    parser.add_argument("--height", type=int, help="Target output height")
    parser.add_argument("--mask", help="Protective mask path (white = keep)")
    parser.add_argument("--remove-mask", help="Removal mask path (enables object removal)")
    parser.add_argument(
        "--horizontal-removal",
        action="store_true",
        help="Carve horizontal seams during object removal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.input)
    mask = load_mask(args.mask) if args.mask else None
    if args.remove_mask:
        remove_mask = load_mask(args.remove_mask)
        result = object_removal(image, remove_mask, mask, args.horizontal_removal)
    else:
        result = resize(image, args.width, args.height, mask)
    save_image(result, args.output)


if __name__ == "__main__":
    main()
