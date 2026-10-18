"""Convert a raw RGBA16F capture (d2enhanced F8 dump) to an 8-bit PNG preview.

usage: f16png.py <in.bin> <width> <height> <out.png> [--scale N] [--exposure E]
Preview only: simple Reinhard tone curve and sRGB encode, not the game's tonemapper.
"""
import struct, sys, zlib, argparse

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("width", type=int); ap.add_argument("height", type=int); ap.add_argument("out")
ap.add_argument("--scale", type=int, default=2); ap.add_argument("--exposure", type=float, default=1.0)
a = ap.parse_args()
data = open(a.src, "rb").read()
assert len(data) == a.width * a.height * 8, "size mismatch"
w, h = a.width // a.scale, a.height // a.scale
rows = bytearray()
def enc(v):
    v = v * a.exposure
    v = v / (1.0 + v) if v > 0 else 0.0
    v = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return max(0, min(255, int(v * 255 + 0.5)))
for y in range(h):
    rows.append(0)
    for x in range(w):
        o = ((y * a.scale) * a.width + x * a.scale) * 8
        r, g, b, _ = struct.unpack_from("<4e", data, o)
        rows += bytes((enc(r), enc(g), enc(b)))
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
open(a.out, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                       + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))
