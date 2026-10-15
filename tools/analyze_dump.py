"""Analyzes two consecutive frames of upscaler inputs written by the F8 capture (d2enhanced/captures/dump*.bin).

Measures the sub-pixel shift between the two color frames (phase correlation on static image content)
and compares it with the jitter the mod sent to FSR, plus motion vector / depth statistics.
Writes PNG previews next to the dumps.
Usage: python analyze_dump.py <d2enhanced/captures dir>
"""
import pathlib
import re
import sys

import numpy as np
from PIL import Image

game = pathlib.Path(sys.argv[1])
A, B = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)


def load(kind, index):
    files = sorted(game.glob(f"dump{index}_{kind}_*.bin"))
    if not files:
        sys.exit(f"missing dump {index} {kind}")
    w, h = map(int, re.search(r"_(\d+)x(\d+)\.bin$", files[0].name).groups())
    raw = files[0].read_bytes()
    if kind.startswith("color"):
        a = np.frombuffer(raw, np.float16).reshape(h, w, 4).astype(np.float32)
    elif kind.startswith("motion"):
        a = np.frombuffer(raw, np.float16).reshape(h, w, 2).astype(np.float32)
    else:
        a = np.frombuffer(raw, np.float32).reshape(h, w)
    return a


def phase_shift(a, b):
    """Sub-pixel translation of b relative to a (dx, dy) via phase correlation with parabolic peak fit."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-12
    c = np.fft.ifft2(r).real
    py, px = np.unravel_index(np.argmax(c), c.shape)

    def fit(m1, z, p1):
        d = m1 - 2 * z + p1
        return 0.0 if abs(d) < 1e-12 else 0.5 * (m1 - p1) / d

    h, w = c.shape
    dy = py + fit(c[(py - 1) % h, px], c[py, px], c[(py + 1) % h, px])
    dx = px + fit(c[py, (px - 1) % w], c[py, px], c[py, (px + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return -dx, -dy


c0, c1 = load("color_rgba16f", A), load("color_rgba16f", B)
m0, d0 = load("motion_rg16f", B), load("depth_r32f", B)
h, w = c0.shape[:2]
lum0 = c0[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
lum1 = c1[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)

print(f"render size {w}x{h}")
print(f"color range {lum0.min():.4f}..{lum0.max():.4f}, mean abs frame diff {np.abs(lum1 - lum0).mean():.6f}")
print(f"identical pixels between frames: {np.mean(lum0 == lum1) * 100:.2f}%")

# Static-looking region: center crop (avoids HUD/weapon at the edges as much as possible).
y0, y1, x0, x1 = h // 6, h // 2, w // 4, 3 * w // 4
dx, dy = phase_shift(lum0[y0:y1, x0:x1], lum1[y0:y1, x0:x1])
print(f"measured content shift frame1 vs frame0 (center-top crop): dx={dx:+.4f} px dy={dy:+.4f} px")

mpx = m0 * np.array([w, h], np.float32)
mag = np.hypot(mpx[..., 0], mpx[..., 1])
print(f"motion vectors (px, viewport UV * render size): mean |mv| {mag.mean():.4f}, median {np.median(mag):.4f}, "
      f"crop mean mv ({mpx[y0:y1, x0:x1, 0].mean():+.4f}, {mpx[y0:y1, x0:x1, 1].mean():+.4f})")
print(f"depth range {d0.min():.6f}..{d0.max():.6f}, zero (sky) {np.mean(d0 == 0) * 100:.1f}%")

for i, c in enumerate((c0, c1)):
    srgb = np.clip(c[..., :3], 0, 1) ** (1 / 2.2)
    Image.fromarray((srgb * 255).astype(np.uint8)).save(game / f"dump{i}_color.png")
Image.fromarray((np.clip(mag / 4.0, 0, 1) * 255).astype(np.uint8)).save(game / "dump0_motion.png")
dn = d0 / (d0.max() + 1e-9)
Image.fromarray((np.clip(dn, 0, 1) ** 0.25 * 255).astype(np.uint8)).save(game / "dump0_depth.png")
diff = np.abs(lum1 - lum0)
Image.fromarray((np.clip(diff * 20, 0, 1) * 255).astype(np.uint8)).save(game / "dump_diff.png")
print("previews written")

# FSR output vs a plain bilinear stretch of its input: does the upscaler add or lose detail?
outs = sorted(game.glob(f"dump{B}_output_rgba16f_*.bin"))
if outs:
    ow, oh = map(int, re.search(r"_(\d+)x(\d+)\.bin$", outs[0].name).groups())
    out = np.frombuffer(outs[0].read_bytes(), np.float16).reshape(oh, ow, 4).astype(np.float32)[..., :3]

    def to8(img):
        return (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)

    stretched = np.asarray(Image.fromarray(to8(c1[..., :3])).resize((ow, oh), Image.BILINEAR)).astype(np.float32)
    fsr8 = to8(out).astype(np.float32)

    def sharpness(img):
        g = img.mean(axis=2)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return np.abs(lap).mean()

    print(f"edge energy (mean |laplacian|, 8-bit): FSR output {sharpness(fsr8):.3f}, bilinear stretch {sharpness(stretched):.3f}")
    # Same region as the input zoom, at output scale.
    sx, sy = ow / w, oh / h
    box = (int((w // 2 - 120) * sx), int((h // 2 - 170) * sy), int((w // 2 + 120) * sx), int((h // 2 - 30) * sy))
    for name, img in (("fsr", fsr8), ("bilinear", stretched)):
        Image.fromarray(img.astype(np.uint8)).crop(box).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3),
                                                             Image.NEAREST).save(game / f"dump_zoom_{name}.png")
    print("output zooms written")
