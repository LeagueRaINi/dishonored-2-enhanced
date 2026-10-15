"""Compares the actual sub-pixel shift between consecutive upscaler input frames with the jitter sent to FSR.

FSR's convention: the jitter offset is how far the image content moved, in render pixels (x right, y down).
So for consecutive frames, measured content shift should equal FSR jitter[i+1] - jitter[i].
Reads dump<N>_color_*.bin and dump<N>_info.txt written by the F8 capture.
Usage: python measure_jitter.py <d2enhanced/captures dir>
"""
import pathlib
import re
import sys

import numpy as np

game = pathlib.Path(sys.argv[1])


def color_luma(i):
    f = sorted(game.glob(f"dump{i}_color_rgba16f_*.bin"))[0]
    w, h = map(int, re.search(r"_(\d+)x(\d+)\.bin$", f.name).groups())
    c = np.frombuffer(f.read_bytes(), np.float16).reshape(h, w, 4).astype(np.float64)[..., :3]
    return c @ [0.2126, 0.7152, 0.0722]


def fsr_jitter(i):
    text = (game / f"dump{i}_info.txt").read_text()
    m = re.search(r"FSR jitter px \(([-\d.]+), ([-\d.]+)\)", text)
    return float(m.group(1)), float(m.group(2))


def frame_number(i):
    return float(re.search(r"frame ([\d.]+)", (game / f"dump{i}_info.txt").read_text()).group(1))


def register(a, b, up=40):
    """Shift (dx, dy) of b relative to a: phase correlation refined by an upsampled matrix DFT."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-12
    c = np.fft.ifft2(r).real
    py, px = np.unravel_index(np.argmax(c), c.shape)
    h, w = c.shape
    py = py - h if py > h // 2 else py
    px = px - w if px > w // 2 else px
    n = 3 * up
    ky = np.fft.ifftshift(np.arange(h)) - h // 2
    kx = np.fft.ifftshift(np.arange(w)) - w // 2
    oy = py - 1.5 + np.arange(n) / up
    ox = px - 1.5 + np.arange(n) / up
    cc = (np.exp(2j * np.pi * np.outer(oy, ky) / h) @ r @ np.exp(2j * np.pi * np.outer(kx, ox) / w)).real
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    return -ox[ix], -oy[iy]


count = len(list(game.glob("dump*_info.txt")))
frames = [color_luma(i) for i in range(count)]
h, w = frames[0].shape
regions = {"center-top": (h // 6, h // 2, w // 4, 3 * w // 4), "upper-left": (h // 10, h // 2, w // 10, w // 2),
           "upper-right": (h // 10, h // 2, w // 2, 9 * w // 10)}
for i in range(count - 1):
    if frame_number(i + 1) != frame_number(i) + 1:
        continue  # different F8 capture
    j0, j1 = fsr_jitter(i), fsr_jitter(i + 1)
    expected = (j1[0] - j0[0], j1[1] - j0[1])
    print(f"frames {i}->{i + 1}: FSR jitter delta (content shift FSR expects) dx={expected[0]:+.4f} dy={expected[1]:+.4f}")
    for name, (y0, y1, x0, x1) in regions.items():
        dx, dy = register(frames[i][y0:y1, x0:x1], frames[i + 1][y0:y1, x0:x1])
        print(f"    {name:12s} measured dx={dx:+.4f} dy={dy:+.4f}")
