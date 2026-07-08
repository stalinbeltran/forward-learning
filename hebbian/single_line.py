"""Generate a dataset with a SINGLE centered straight line.

Draws one antialiased line (black background, white line) at a fixed angle,
centered with no jitter, and saves it as an ``.npz`` with ``images`` uint8 of
shape ``(1, size, size)``. Handy for watching the competitive layer learn a
single fixed input step by step in ``webapp_evolution.py``.

    python hebbian/single_line.py --angle 0 --out data/processed/hline/hline.npz
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw


def make_centered_line(angle_deg: float, size: int = 28, scale: int = 4,
                       width: int = 2) -> np.ndarray:
    """One centered line at ``angle_deg`` (0 = horizontal). uint8 (size, size)."""
    S = size * scale
    img = Image.new("L", (S, S), color=0)
    draw = ImageDraw.Draw(img)

    angle = math.radians(angle_deg)
    dx, dy = math.cos(angle), math.sin(angle)
    cx = cy = S / 2  # centered, no jitter
    L = S * 1.5  # extend past the canvas so it crosses fully
    draw.line((cx - dx * L, cy - dy * L, cx + dx * L, cy + dy * L),
              fill=255, width=width * scale)

    small = img.resize((size, size), Image.LANCZOS)
    return np.asarray(small, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a single centered line -> .npz")
    ap.add_argument("--angle", type=float, default=0.0,
                    help="line angle in degrees (0 = horizontal, 90 = vertical)")
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--out", default="data/processed/hline/hline.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img = make_centered_line(args.angle, size=args.size, width=args.width)
    images = img[None, ...]  # shape (1, size, size)
    np.savez_compressed(args.out, images=images)
    print(f"wrote {args.out}  images={images.shape} angle={args.angle} deg")


if __name__ == "__main__":
    main()
