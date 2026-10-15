"""Counts, per forward-lighting pixel shader, how the IBL cubemap array and the metallic map are sampled (from the
disassembly), to check the SSR material patch can rely on one specular IBL sample with an explicit LOD.

Usage: python survey_ibl.py <shaders dir>
"""
import collections
import ctypes
import pathlib
import re
import struct
import sys

compiler = ctypes.WinDLL("d3dcompiler_47.dll")
D3DDisassemble = compiler.D3DDisassemble
D3DDisassemble.restype = ctypes.c_long
D3DDisassemble.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]


def blob_text(blob):
    vtbl = ctypes.cast(ctypes.cast(blob, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    get_ptr = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vtbl[3])
    get_size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vtbl[4])
    data = ctypes.string_at(get_ptr(blob), get_size(blob))
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])(blob)
    return data.decode(errors="replace")


def slot_of(text, name):
    m = re.search(rf"^// {name}\s+texture\s+\S+\s+\S+\s+t(\d+)", text, re.M)
    return int(m.group(1)) if m else None


stats = collections.Counter()
lod_kinds = collections.Counter()
for path in sorted(pathlib.Path(sys.argv[1]).glob("ps_*.dxbc")):
    code = path.read_bytes()
    if b"ro_tlf_ll_heads" not in code:
        continue
    out = ctypes.c_void_p()
    if D3DDisassemble(code, len(code), 0, None, ctypes.byref(out)) < 0:
        continue
    text = blob_text(out)
    ibl = slot_of(text, "ro_ibl_texcubemaparray")
    metal = slot_of(text, "ro_metallicmap")
    stats["forward"] += 1
    if ibl is None:
        stats["no ibl"] += 1
        continue
    samples = re.findall(rf"^\s*sample_l_indexable\(texturecubearray\)\S*\s+(r\d+)\.(\w+), [^,]+, t{ibl}\.\w+, s\d+, (\S+)$", text, re.M)
    stats[f"ibl samples: {len(samples)}"] += 1
    for dest, mask, lod in samples[:1]:
        lod_kinds[re.sub(r"\d+", "N", lod)] += 1
    if metal is not None:
        m = re.findall(rf"^\s*sample\w*\S*\s+(r\d+)\.(\w+), [^,]+, t{metal}\.", text, re.M)
        stats[f"metallic samples: {len(m)}"] += 1
print(dict(stats))
print("first IBL sample LOD operand kinds:", dict(lod_kinds))
