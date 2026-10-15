"""Surveys the dumped pixel shaders: which ones are forward-lighting shaders (bind the tiled light lists) and what
texture bindings they use, to find how base colour reaches them.

Usage: python survey_forward.py <shaders dir>
"""
import collections
import pathlib
import struct
import sys


def rdef_bindings(code):
    if code[:4] != b"DXBC":
        return []
    count = struct.unpack_from("<I", code, 28)[0]
    for i in range(count):
        offset = struct.unpack_from("<I", code, 32 + i * 4)[0]
        if code[offset:offset + 4] != b"RDEF":
            continue
        rdef = code[offset + 8:]
        bind_count, bind_offset = struct.unpack_from("<II", rdef, 8)
        out = []
        for b in range(bind_count):
            name_offset, kind, _ret, dim, _samples, slot, _count, _flags = struct.unpack_from("<8I", rdef, bind_offset + b * 32)
            name = rdef[name_offset:rdef.index(b"\0", name_offset)].decode(errors="replace")
            out.append((kind, slot, dim, name))
        return out
    return []


folder = pathlib.Path(sys.argv[1])
forward = []
texture_names = collections.Counter()
for path in folder.glob("ps_*.dxbc"):
    bindings = rdef_bindings(path.read_bytes())
    names = [b[3] for b in bindings]
    if "ro_tlf_ll_heads" not in names:
        continue
    forward.append((path.name, bindings))
    for kind, slot, dim, name in bindings:
        if kind == 2:  # texture
            texture_names[name] += 1

print(f"{len(forward)} forward-lighting pixel shaders")
print("most common texture bindings:")
for name, n in texture_names.most_common(40):
    print(f"  {n:5d}  {name}")
with_diffuse = [f for f, b in forward if any("diffuse" in x[3] or "albedo" in x[3] or "basecolor" in x[3] for x in b)]
print(f"{len(with_diffuse)} of them bind a diffuse/albedo/basecolor texture")
