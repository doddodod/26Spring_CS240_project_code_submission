"""Local dynamic-programming (DP) update seam carving (algorithmic improvement #3).

The baseline re-runs the *entire* O(H*W) dynamic program after every seam
removal. However, deleting one vertical seam only perturbs the cumulative DP
matrix ``M`` inside a narrow "funnel" around the seam path: every pixel whose
upward dependency cone never touches the perturbed band keeps exactly the same
``M`` value (its column index merely shifts when the seam column is dropped).

This module reuses the previous round's ``M`` / ``parent`` tables and only
recomputes the affected funnel:

* ``compute_affected_region`` -- marks the band that must be recomputed.
* ``partial_dp``              -- recomputes only the marked pixels, copying the
                                rest from the (index-aligned) previous tables.
* ``find_seam_local_dp``      -- glues the two together for one seam.
* ``resize_local_dp``         -- a drop-in replacement for ``resize`` that uses
                                the local DP update for width/height reduction.

Correctness is *identical* to the baseline global DP: the energy map is still
recomputed exactly (full Sobel is local, so far-from-seam values are unchanged),
and the funnel is a conservative super-set of all genuinely changed ``M`` cells.
If anything goes wrong the code falls back to a full DP, so it never crashes.

The baseline files are not modified; their primitives are imported directly.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from backward_energy_seam_carving import (
    apply_masks,
    backward_energy,
    insert_seams,
    load_image,
    load_mask,
    remove_seam_from_mask,
    remove_vertical_seam,
    resize as baseline_resize,
    save_image,
)

# Default influence half-width (in columns). The seam removal perturbs energy
# within about +/-2 columns; delta=3 leaves a safe margin. Larger = more
# conservative (recomputes more, never less correct).
DEFAULT_DELTA = 3


# ---------------------------------------------------------------------------
# Full DP (used for round 0 and as a safety fallback). Numerically identical to
# ``backward_energy_seam_carving.find_vertical_seam``.
# ---------------------------------------------------------------------------
def _full_dp(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the complete vertical-seam DP, returning ``(M, parent)``."""
    height, width = energy.shape
    cost = energy.astype(np.float64).copy()
    parent = np.zeros((height, width), dtype=np.int32)

    for row in range(1, height):
        for col in range(width):
            left = max(col - 1, 0)
            right = min(col + 2, width)
            previous = cost[row - 1, left:right]
            offset = int(np.argmin(previous))
            parent[row, col] = left + offset
            cost[row, col] += previous[offset]
    return cost, parent


def _backtrack(M: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """Recover the minimum-cost seam from filled ``M`` / ``parent`` tables."""
    height = M.shape[0]
    seam = np.zeros(height, dtype=np.int32)
    seam[-1] = int(np.argmin(M[-1]))
    for row in range(height - 2, -1, -1):
        seam[row] = parent[row + 1, seam[row + 1]]
    return seam


# ---------------------------------------------------------------------------
# Index-alignment helpers: bring the previous round's tables (width W) into the
# current round's coordinate frame (width W-1) by dropping the removed seam.
# ---------------------------------------------------------------------------
def _delete_seam_columns(array2d: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Delete one column per row (used for the scalar-valued ``M`` matrix)."""
    height, width = array2d.shape
    carved = np.empty((height, width - 1), dtype=array2d.dtype)
    for row in range(height):
        carved[row] = np.delete(array2d[row], int(seam[row]))
    return carved


def _shift_parent_columns(parent: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Re-index a ``parent`` table after removing ``seam``.

    Two adjustments are required:

    * **position** -- drop the removed column ``seam[r]`` from row ``r``;
    * **value**    -- ``parent[r, c]`` stores a column index *into row r-1*, so a
      stored value ``p`` must be remapped through that row's deletion:
      ``p -> p - (p > seam[r-1])``. Unaffected pixels never point at the deleted
      column, so no information is lost.
    """
    height, width = parent.shape
    out = np.empty((height, width - 1), dtype=parent.dtype)
    for row in range(height):
        shifted = np.delete(parent[row], int(seam[row]))
        if row > 0:
            prev_removed = int(seam[row - 1])
            shifted = shifted - (shifted > prev_removed).astype(parent.dtype)
        out[row] = shifted
    return out


# ---------------------------------------------------------------------------
# 1. Affected-region (funnel) computation.
# ---------------------------------------------------------------------------
def compute_affected_region(
    seam: np.ndarray, height: int, width: int, delta: int = DEFAULT_DELTA
) -> np.ndarray:
    """Boolean ``H x W`` mask of pixels whose ``M`` value may have changed.

    Each row ``r`` is seeded with ``[seam[r] - delta, seam[r] + delta]``. Because
    ``M(i, j)`` depends on the three cells ``(i-1, j-1..j+1)``, the band expands
    by one column per row going downward. The running band is therefore::

        lo = min(lo_prev - 1, seam[r] - delta)
        hi = max(hi_prev + 1, seam[r] + delta)

    which is a conservative super-set of the true dependency funnel (over-marking
    is always safe; under-marking would corrupt the result).
    """
    region = np.zeros((height, width), dtype=bool)
    seam = np.clip(np.asarray(seam, dtype=np.int64), 0, width - 1)

    lo = int(seam[0]) - delta
    hi = int(seam[0]) + delta
    for row in range(height):
        if row > 0:
            lo = min(lo - 1, int(seam[row]) - delta)
            hi = max(hi + 1, int(seam[row]) + delta)
        a = max(0, lo)
        b = min(width - 1, hi)
        if a <= b:
            region[row, a : b + 1] = True
    return region


# ---------------------------------------------------------------------------
# 2. Partial DP: recompute only the affected funnel, copy the rest.
# ---------------------------------------------------------------------------
def partial_dp(
    image: np.ndarray,
    energy: np.ndarray,
    affected_region: np.ndarray,
    prev_M: np.ndarray | None = None,
    prev_parent: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Update ``(M, parent)`` recomputing only ``affected_region``.

    ``prev_M`` / ``prev_parent`` must already be aligned to the *current* width
    (i.e. the previous round's seam has been removed from them). For
    ``affected_region == False`` the cells are copied verbatim; for ``True`` they
    are recomputed with the standard recurrence, reading row ``i-1`` from the
    in-progress tables (which hold up-to-date values for affected neighbours and
    correct copied values for unaffected ones).

    If no previous tables are supplied this degenerates to a full DP.
    """
    if prev_M is None or prev_parent is None:
        return _full_dp(energy)

    height, width = energy.shape
    M = prev_M.astype(np.float64).copy()
    parent = prev_parent.astype(np.int32).copy()

    # Row 0 base case: M = energy, parent = 0 (matches the baseline).
    row0 = affected_region[0]
    if np.any(row0):
        M[0, row0] = energy[0, row0]
        parent[0, row0] = 0

    for row in range(1, height):
        cols = np.flatnonzero(affected_region[row])
        for col in cols:
            col = int(col)
            left = max(col - 1, 0)
            right = min(col + 2, width)
            previous = M[row - 1, left:right]
            offset = int(np.argmin(previous))
            parent[row, col] = left + offset
            M[row, col] = energy[row, col] + previous[offset]
    return M, parent


# ---------------------------------------------------------------------------
# 3. One seam via local DP update (with full-DP fallback).
# ---------------------------------------------------------------------------
def find_seam_local_dp(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    prev_M: np.ndarray | None = None,
    prev_parent: np.ndarray | None = None,
    prev_seam: np.ndarray | None = None,
    delta: int = DEFAULT_DELTA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find one vertical seam, reusing the previous DP tables when possible.

    Returns ``(seam, M, parent)`` for the *current* image. ``prev_M`` /
    ``prev_parent`` are the previous round's tables already re-indexed to the
    current width, and ``prev_seam`` is the seam that was removed (used only to
    locate the affected band). On the first round (any of them ``None``) a full
    DP is run. Any failure during the local update falls back to a full DP so the
    pipeline never crashes.

    Note: ``prev_seam`` is an extra argument beyond the original spec -- the
    affected-region rule is defined in terms of "上一轮的 seam", which is not
    otherwise recoverable from ``prev_M`` / ``prev_parent`` alone.
    """
    energy = apply_masks(backward_energy(image), mask, remove_mask)
    height, width = energy.shape

    if prev_M is None or prev_parent is None or prev_seam is None:
        M, parent = _full_dp(energy)
        return _backtrack(M, parent), M, parent

    try:
        region = compute_affected_region(prev_seam, height, width, delta)
        M, parent = partial_dp(image, energy, region, prev_M, prev_parent)
        seam = _backtrack(M, parent)
        return seam, M, parent
    except Exception:  # pragma: no cover - safety net, must not crash
        M, parent = _full_dp(energy)
        return _backtrack(M, parent), M, parent


# ---------------------------------------------------------------------------
# 4. Width reduction loop + public resize entry point.
# ---------------------------------------------------------------------------
def _reduce_width_local_dp(
    image: np.ndarray,
    target_width: int,
    mask: np.ndarray | None = None,
    delta: int = DEFAULT_DELTA,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove vertical seams down to ``target_width`` using local DP updates."""
    prev_M = prev_parent = prev_seam = None

    while image.shape[1] > target_width:
        seam, M, parent = find_seam_local_dp(
            image, mask, None, prev_M, prev_parent, prev_seam, delta
        )

        image = remove_vertical_seam(image, seam)
        if mask is not None:
            mask = remove_seam_from_mask(mask, seam)

        # Re-index this round's tables into the next round's (narrower) frame.
        prev_M = _delete_seam_columns(M, seam)
        prev_parent = _shift_parent_columns(parent, seam)
        prev_seam = seam

    return image, mask


def resize_local_dp(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
    mask: np.ndarray | None = None,
    delta: int = DEFAULT_DELTA,
) -> np.ndarray:
    """Resize using local DP updates for reductions (drop-in for ``resize``).

    Width/height *reduction* uses the accelerated local DP update. *Enlargement*
    reuses the baseline ``insert_seams`` (which records seams on a shrinking
    copy), keeping behaviour identical to the baseline.
    """
    if target_width is not None:
        if target_width < image.shape[1]:
            image, mask = _reduce_width_local_dp(image, target_width, mask, delta)
        elif target_width > image.shape[1]:
            image, mask = insert_seams(image, target_width - image.shape[1], mask)

    if target_height is not None and target_height != image.shape[0]:
        image = np.transpose(image, (1, 0, 2))
        t_mask = mask.T if mask is not None else None
        if target_height < image.shape[1]:
            image, t_mask = _reduce_width_local_dp(image, target_height, t_mask, delta)
        elif target_height > image.shape[1]:
            image, t_mask = insert_seams(image, target_height - image.shape[1], t_mask)
        image = np.transpose(image, (1, 0, 2))
        mask = t_mask.T if t_mask is not None else None

    return image


# ---------------------------------------------------------------------------
# Correctness self-test: local DP must match global DP exactly.
# ---------------------------------------------------------------------------
def _self_test(height: int = 100, width: int = 100, target_width: int = 60) -> None:
    rng = np.random.default_rng(240)
    image = rng.uniform(0, 255, size=(height, width, 3)).astype(np.float64)

    # --- Reference: global DP every round (accumulate optimal seam cost). ---
    full_cost = 0.0
    img_full = image.copy()
    t0 = time.perf_counter()
    while img_full.shape[1] > target_width:
        energy = apply_masks(backward_energy(img_full), None, None)
        M, parent = _full_dp(energy)
        full_cost += float(M[-1].min())
        img_full = remove_vertical_seam(img_full, _backtrack(M, parent))
    full_time = time.perf_counter() - t0

    # --- Local DP update (accumulate optimal seam cost the same way). ---
    local_cost = 0.0
    img_local = image.copy()
    prev_M = prev_parent = prev_seam = None
    t0 = time.perf_counter()
    while img_local.shape[1] > target_width:
        seam, M, parent = find_seam_local_dp(
            img_local, None, None, prev_M, prev_parent, prev_seam
        )
        local_cost += float(M[-1].min())
        img_local = remove_vertical_seam(img_local, seam)
        prev_M = _delete_seam_columns(M, seam)
        prev_parent = _shift_parent_columns(parent, seam)
        prev_seam = seam
    local_time = time.perf_counter() - t0

    image_diff = float(np.abs(img_full.astype(np.float64) - img_local.astype(np.float64)).max())
    cost_diff = abs(full_cost - local_cost)

    print(f"Full DP cost: {full_cost:.6f}, Local DP cost: {local_cost:.6f}, "
          f"差异: {cost_diff:.6e}")
    print(f"图像逐像素最大差异: {image_diff:.6e}")
    print(f"耗时  Full DP: {full_time:.3f}s   Local DP: {local_time:.3f}s   "
          f"加速比: {full_time / local_time:.2f}x")

    if image_diff == 0.0 and cost_diff == 0.0:
        print("[PASS] 局部 DP 更新结果与全局 DP 完全一致。")
    else:
        print("[FAIL] 出现差异，请检查 affected_region 是否覆盖了所有依赖链。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local DP-update seam carving")
    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output image path")
    parser.add_argument("--width", type=int, help="Target output width")
    parser.add_argument("--height", type=int, help="Target output height")
    parser.add_argument("--mask", help="Protective mask path (white = keep)")
    parser.add_argument("--delta", type=int, default=DEFAULT_DELTA,
                        help="Influence half-width of the recomputed band")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the correctness self-test and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test or not args.input:
        _self_test()
        return
    image = load_image(args.input)
    mask = load_mask(args.mask) if args.mask else None
    result = resize_local_dp(image, args.width, args.height, mask, args.delta)
    save_image(result, args.output)


if __name__ == "__main__":
    main()
