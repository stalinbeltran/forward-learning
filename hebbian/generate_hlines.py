"""Generate a SET of N horizontal lines at different vertical positions.

Every image is a horizontal line (angle 0) centered horizontally but shifted
vertically by a different offset, evenly spread across the canvas. Saved as an
``.npz`` with ``images`` uint8 ``(N, size, size)`` plus the ``offsets`` used, so
the sequential trainer can present each one, one at a time.

    python hebbian/generate_hlines.py --n 10 --out data/processed/hlines_set/hlines.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Support running as a module (python -m hebbian.generate_hlines) or as a script.
try:
    from .single_line import make_centered_line
except ImportError:  # pragma: no cover - script execution fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from single_line import make_centered_line


def generate(n: int, size: int = 28, width: int = 2,
             spread: float = 11.0) -> tuple[np.ndarray, np.ndarray]:
    """N horizontal lines with vertical offsets evenly spread in [-spread, spread].

    Returns ``(images uint8 (n, size, size), offsets float (n,))`` ordered from
    the topmost line (most negative offset_y = up) to the bottom one.
    """
    offsets = np.linspace(-spread, spread, n)
    images = np.stack([
        make_centered_line(0.0, size=size, width=width, offset_y=float(off))
        for off in offsets
    ])
    return images, offsets


def save_preview(images: np.ndarray, path: str) -> None:
    """Write a single-row PNG strip of every line for a quick eyeball check."""
    from PIL import Image
    n, size = images.shape[0], images.shape[1]
    strip = np.zeros((size, n * size), dtype=np.uint8)
    for i in range(n):
        strip[:, i * size:(i + 1) * size] = images[i]
    Image.fromarray(strip, mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate N horizontal lines at different heights -> .npz")
    ap.add_argument("--n", type=int, default=10, help="number of lines")
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--spread", type=float, default=11.0,
                    help="max vertical offset in px (lines span [-spread, spread])")
    ap.add_argument("--out", default="data/processed/hlines_set/hlines.npz")
    ap.add_argument("--preview", action="store_true", help="write a <out>.preview.png strip")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    images, offsets = generate(args.n, size=args.size, width=args.width, spread=args.spread)
    np.savez_compressed(args.out, images=images, offsets=offsets.astype(np.float32))
    print(f"wrote {args.out}  images={images.shape} dtype={images.dtype}")
    print(f"  vertical offsets (px): {np.round(offsets, 2).tolist()}")

    if args.preview:
        ppath = os.path.splitext(args.out)[0] + ".preview.png"
        save_preview(images, ppath)
        print(f"wrote {ppath}")


if __name__ == "__main__":
    main()
