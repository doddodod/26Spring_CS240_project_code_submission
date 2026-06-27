"""Visualization helpers for comparing seam carving outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_comparison(
    image_paths: list[str | Path],
    labels: list[str],
    output_path: str | Path,
    tile_height: int = 240,
) -> None:
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length")

    images = []
    for path in image_paths:
        image = load_rgb(path)
        ratio = tile_height / image.height
        tile_width = max(1, int(image.width * ratio))
        images.append(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS))

    label_height = 32
    total_width = sum(image.width for image in images)
    canvas = Image.new("RGB", (total_width, tile_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x_offset = 0
    for image, label in zip(images, labels):
        canvas.paste(image, (x_offset, label_height))
        text_x = x_offset + max(4, (image.width - len(label) * 6) // 2)
        draw.text((text_x, 10), label, fill="black", font=font)
        x_offset += image.width

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side result comparison")
    parser.add_argument("--output", required=True, help="Output comparison image path")
    parser.add_argument("--height", type=int, default=240, help="Tile height")
    parser.add_argument(
        "items",
        nargs="+",
        help="Alternating label and image path values, e.g. Original original.jpg Backward b.jpg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.items) % 2 != 0:
        raise ValueError("items must be alternating label/path pairs")

    labels = args.items[0::2]
    image_paths = args.items[1::2]
    make_comparison(image_paths, labels, args.output, args.height)


if __name__ == "__main__":
    main()
