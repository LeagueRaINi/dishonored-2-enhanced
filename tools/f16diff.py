"""Mean absolute and max difference between two RGBA16F dumps (stdlib only).

usage: f16diff.py <a.bin> <b.bin> <width> <height> [--stride N]
Prints mean |a-b| per channel, mean |a| for scale, and the fraction of sampled
pixels whose RGB differs by more than 1 % of the mean luminance.
"""
import argparse, struct
ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("width", type=int); ap.add_argument("height", type=int)
ap.add_argument("--stride", type=int, default=4)
x = ap.parse_args()
A = open(x.a, "rb").read(); B = open(x.b, "rb").read()
assert len(A) == len(B) == x.width * x.height * 8
s = [0.0] * 3; m = [0.0] * 3; n = 0; samples = []
for y in range(0, x.height, x.stride):
    for xx in range(0, x.width, x.stride):
        o = (y * x.width + xx) * 8
        a = struct.unpack_from("<3e", A, o); b = struct.unpack_from("<3e", B, o)
        d = [abs(a[i] - b[i]) for i in range(3)]
        for i in range(3):
            s[i] += d[i]; m[i] += abs(a[i])
        samples.append(sum(d)); n += 1
mean_lum = sum(m) / (3 * n)
big = sum(1 for d in samples if d > 0.03 * mean_lum) / n
print("mean|a| %.5f  mean|a-b| rgb (%.5f %.5f %.5f)  changed>1%%: %.2f%%" % (mean_lum, s[0] / n, s[1] / n, s[2] / n, 100 * big))
