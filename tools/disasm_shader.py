"""Disassembles dumped shaders (d2enhanced/captures/shaders/*.dxbc, [Debug] DumpShaders=1) with d3dcompiler_47.

Usage: python disasm_shader.py <shaders dir> <hash or file name part> [more ...]
Writes <name>.asm next to each match and prints its path.
"""
import ctypes
import pathlib
import sys

compiler = ctypes.WinDLL("d3dcompiler_47.dll")
D3DDisassemble = compiler.D3DDisassemble
D3DDisassemble.restype = ctypes.c_long
D3DDisassemble.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_char_p,
                           ctypes.POINTER(ctypes.c_void_p)]


def blob_bytes(blob):
    vtbl = ctypes.cast(ctypes.cast(blob, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    get_ptr = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vtbl[3])
    get_size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vtbl[4])
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])
    data = ctypes.string_at(get_ptr(blob), get_size(blob))
    release(blob)
    return data


folder = pathlib.Path(sys.argv[1])
for needle in sys.argv[2:]:
    matches = sorted(folder.glob(f"*{needle}*.dxbc"))
    if not matches:
        print(f"{needle}: no dump")
    for path in matches:
        code = path.read_bytes()
        text = ctypes.c_void_p()
        hr = D3DDisassemble(code, len(code), 0, None, ctypes.byref(text))
        if hr < 0:
            print(f"{path.name}: disassembly failed 0x{hr & 0xffffffff:08x}")
            continue
        out = path.with_suffix(".asm")
        out.write_bytes(blob_bytes(text))
        print(out)
