"""Create a Backward-vs-Forward seam carving pipeline visualization on bench.png."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import backward_energy_seam_carving as backward
import forward_energy_seam_carving as forward


INPUT_PATH = Path("picture/input/bench.png")
OUTPUT_DIR = Path("results/poster_demos/pipeline")
OUTPUT_PNG = OUTPUT_DIR / "bench_backward_forward_pipeline.png"
OUTPUT_PDF = OUTPUT_DIR / "bench_backward_forward_pipeline.pdf"


def normalize_heatmap(energy: np.ndarray) -> np.ndarray:
    clipped = np.clip(energy, np.percentile(energy, 2), np.percentile(energy, 98))
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-12)
    # Lightweight magma-like heatmap without requiring matplotlib.
    stops = np.asarray(
        [
            [20, 13, 55],
            [82, 18, 92],
            [150, 38, 92],
            [221, 84, 58],
            [252, 194, 89],
            [252, 253, 191],
        ],
        dtype=np.float64,
    )
    scaled = normalized * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    weight = scaled - lower
    rgb = (stops[lower] * (1.0 - weight[..., None]) + stops[upper] * weight[..., None]).astype(np.uint8)
    return rgb


def overlay_seam(image: np.ndarray, seam: np.ndarray) -> Image.Image:
    canvas = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    points = [(int(col), row) for row, col in enumerate(seam)]
    for offset in (-1, 0, 1):
        shifted = [(x + offset, y) for x, y in points]
        draw.line(shifted, fill=(255, 0, 0), width=3)
    return canvas


def ensure_resized_outputs(image: np.ndarray) -> tuple[Path, Path]:
    backward_path = Path("results/backward/bench_backward_350x342.jpg")
    forward_path = Path("results/forward/bench_forward_350x342.jpg")
    backward_path.parent.mkdir(parents=True, exist_ok=True)
    forward_path.parent.mkdir(parents=True, exist_ok=True)

    if not backward_path.exists():
        backward.save_image(backward.resize(image.copy(), target_width=350, target_height=342), backward_path)
    if not forward_path.exists():
        forward.save_image(forward.resize(image.copy(), target_width=350, target_height=342), forward_path)
    return backward_path, forward_path


def make_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = backward.load_image(INPUT_PATH)

    bw_energy = backward.backward_energy(image)
    bw_seam = backward.find_vertical_seam(bw_energy)
    bw_overlay = overlay_seam(image, bw_seam)

    fw_seam = forward.find_vertical_seam_forward(image)
    fw_overlay = overlay_seam(image, fw_seam)

    bw_result_path, fw_result_path = ensure_resized_outputs(image)
    bw_result = Image.open(bw_result_path).convert("RGB")
    fw_result = Image.open(fw_result_path).convert("RGB")
    original = Image.open(INPUT_PATH).convert("RGB")
    bw_heatmap = Image.open("results/energy/bench_backward.png").convert("RGB")
    fw_heatmap = Image.open("results/energy/bench_forward.png").convert("RGB")

    canvas = Image.new("RGB", (1600, 1250), "#F7FAFF")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 44)
    section_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)

    draw.text((50, 30), "Seam Carving Pipeline on Bench Image", fill="#153A6B", font=title_font)
    draw.text(
        (50, 88),
        "Original Image -> Energy Map -> Optimal Seam -> Seam Removed -> Resized Image",
        fill="#4B6584",
        font=label_font,
    )

    def fit_image(img: Image.Image | np.ndarray, size: tuple[int, int]) -> Image.Image:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        fitted = img.convert("RGB").copy()
        fitted.thumbnail(size, Image.Resampling.LANCZOS)
        frame = Image.new("RGB", size, "white")
        frame.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
        return frame

    def paste_panel(img: Image.Image | np.ndarray, x: int, y: int, w: int, h: int, label: str) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill="white", outline="#B8C9E8", width=3)
        draw.text((x + 18, y + 14), label, fill="#1F5C99", font=section_font)
        panel = fit_image(img, (w - 36, h - 64))
        canvas.paste(panel, (x + 18, y + 52))

    original_panel = fit_image(original, (620, 230))
    draw.rounded_rectangle((490, 132, 1110, 382), radius=18, fill="white", outline="#B8C9E8", width=3)
    draw.text((700, 150), "Original Image", fill="#1F5C99", font=section_font)
    canvas.paste(original_panel, (490, 182))

    draw.text((245, 420), "Backward Energy", fill="#153A6B", font=section_font)
    draw.text((1060, 420), "Forward Energy", fill="#153A6B", font=section_font)

    left_x, right_x = 70, 850
    panel_w, panel_h = 680, 230
    row_y = [465, 720, 975]
    paste_panel(bw_heatmap, left_x, row_y[0], panel_w, panel_h, "Energy Map")
    paste_panel(fw_heatmap, right_x, row_y[0], panel_w, panel_h, "Energy Map")
    paste_panel(bw_overlay, left_x, row_y[1], panel_w, panel_h, "Optimal Seam (red curve)")
    paste_panel(fw_overlay, right_x, row_y[1], panel_w, panel_h, "Optimal Seam (red curve)")
    paste_panel(bw_result, left_x, row_y[2], panel_w, panel_h, "Result A: Resized after Seam Removal")
    paste_panel(fw_result, right_x, row_y[2], panel_w, panel_h, "Result B: Resized after Seam Removal")

    canvas.save(OUTPUT_PNG)
    canvas.save(OUTPUT_PDF, "PDF", resolution=180.0)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    make_pipeline()
