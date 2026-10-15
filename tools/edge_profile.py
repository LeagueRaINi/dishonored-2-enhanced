"""Luminance of the last columns / rows of an F8 dump image (RGBA16F), to find edge artifacts such as a bright
border: prints the mean luminance of the 6 outermost columns on each side and rows at the top/bottom.

Usage: python edge_profile.py <dumpN_xxx_rgba16f_WxH.bin>
"""
import re
import struct
import sys

path = sys.argv[1]
w, h = map(int, re.search(r"_(\d+)x(\d+)\.bin$", path).groups())
data = open(path, "rb").read()
assert len(data) == w * h * 8, "expected RGBA16F"


def luminance(x, y):
    r, g, b, _ = struct.unpack_from("<4e", data, (y * w + x) * 8)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def column(x):
    return sum(luminance(x, y) for y in range(0, h, 2)) / len(range(0, h, 2))


def row(y):
    return sum(luminance(x, y) for x in range(0, w, 2)) / len(range(0, w, 2))


print(f"{w}x{h}")
print("left  cols 0..5 :", " ".join(f"{column(x):.4f}" for x in range(6)))
print("right cols -6..-1:", " ".join(f"{column(x):.4f}" for x in range(w - 6, w)))
print("top   rows 0..5 :", " ".join(f"{row(y):.4f}" for y in range(6)))
print("bottom rows -6..-1:", " ".join(f"{row(y):.4f}" for y in range(h - 6, h)))
