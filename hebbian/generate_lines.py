"""Generate a dataset of 28x28 straight-line images (NIST-ish).

Each image has a single line crossing the canvas at a random angle in
[0, 180) degrees, with a small position jitter. Lines are drawn at 4x
resolution and downscaled with antialiasing (PIL) for smooth edges.

Saved as an ``.npz`` with ``images`` uint8 of shape ``(N, 28, 28)``.
Optionally also writes negatives (``255 - image``) and a PNG preview grid.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw


def make_line_image(rng: np.random.Generator, size: int = 28, scale: int = 4,
                    jitter: float = 4.0, width: int = 2) -> np.ndarray:
    """Draw one antialiased line image; returns uint8 ``(size, size)``.

    Background is black (0), the line is white (255).
    """
    S = size * scale
    img = Image.new("L", (S, S), color=0)
    draw = ImageDraw.Draw(img)

    angle = rng.uniform(0.0, math.pi)  # [0, 180) degrees
    dx, dy = math.cos(angle), math.sin(angle)
    # center with jitter (in 28px space, scaled up)
    cx = (size / 2 + rng.uniform(-jitter, jitter)) * scale
    cy = (size / 2 + rng.uniform(-jitter, jitter)) * scale
    # extend the line well past the canvas so it always crosses it fully
    L = S * 1.5
    x0, y0 = cx - dx * L, cy - dy * L
    x1, y1 = cx + dx * L, cy + dy * L
    draw.line((x0, y0, x1, y1), fill=255, width=width * scale)

    small = img.resize((size, size), Image.LANCZOS)
    return np.asarray(small, dtype=np.uint8)


def generate(n: int, seed: int = 0, size: int = 28) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([make_line_image(rng, size=size) for _ in range(n)])


def save_preview(images: np.ndarray, path: str, cols: int = 10, rows: int = 5) -> None:
    """Write a PNG grid preview of the first rows*cols images."""
    n = min(len(images), cols * rows)
    size = images.shape[1]
    grid = np.zeros((rows * size, cols * size), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * size:(r + 1) * size, c * size:(c + 1) * size] = images[i]
    Image.fromarray(grid, mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 28x28 line images -> .npz")
    ap.add_argument("--n", type=int, default=1000, help="number of images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--out", default="data/processed/lines_hebbian/lines.npz")
    ap.add_argument("--negatives", action="store_true",
                    help="also store negatives (255 - image) as 'images_neg'")
    ap.add_argument("--preview", action="store_true",
                    help="write a <out>.preview.png grid")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    images = generate(args.n, seed=args.seed, size=args.size)

    payload = {"images": images}
    if args.negatives:
        payload["images_neg"] = (255 - images).astype(np.uint8)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}  images={images.shape} dtype={images.dtype}"
          + (" (+negatives)" if args.negatives else ""))

    if args.preview:
        ppath = os.path.splitext(args.out)[0] + ".preview.png"
        save_preview(images, ppath)
        print(f"wrote {ppath}")


if __name__ == "__main__":
    main()
