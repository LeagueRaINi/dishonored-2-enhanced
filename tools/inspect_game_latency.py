"""Read-only PE inspection for the latency investigation. Requires capstone.

Prints matching embedded strings, RIP-relative references, and instructions.
The function mode uses PE exception-directory ranges, which can describe only
one fragment of a function. Use explicit verified ranges for split functions.
Does not execute the game. Reference scans are bounded pattern/linear scans,
not a complete control-flow-aware disassembler database.
"""
import bisect
import pathlib
import re
import struct
import sys
from capstone import Cs, CS_ARCH_X86, CS_MODE_64

path = pathlib.Path(__file__).resolve().parents[2] / 'Dishonored2.exe'
data = path.read_bytes()
pe = struct.unpack_from('<I', data, 0x3c)[0]
nsec, optsize = struct.unpack_from('<H', data, pe + 6)[0], struct.unpack_from('<H', data, pe + 20)[0]
opt = pe + 24
base = struct.unpack_from('<Q', data, opt + 24)[0]
sections = []
for i in range(nsec):
    pos = opt + optsize + i * 40
    name = data[pos:pos+8].rstrip(b'\0').decode()
    vs, rva, size, raw = struct.unpack_from('<IIII', data, pos + 8)
    sections.append((name, rva, size, raw))

def off(va):
    for _, rva, size, raw in sections:
        if rva <= va-base < rva+size:
            return raw + va-base-rva
    raise ValueError(hex(va))

def va_at(offset):
    for _, rva, size, raw in sections:
        if raw <= offset < raw+size:
            return base+rva+offset-raw
    return 0

er, es = struct.unpack_from('<II', data, opt+112+3*8)
funcs = [struct.unpack_from('<III', data, off(base+er)+i)[:2] for i in range(0, es, 12)]
starts = [f[0]+base for f in funcs]
def bounds(va):
    ix = bisect.bisect_right(starts, va)-1
    start, end = funcs[ix]
    if start+base <= va < end+base:
        return start+base, end+base
    return None

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.skipdata = True
def annotated(addr, size, mnem, oper):
    text = f'{addr:#x}: {mnem} {oper}'
    m = re.search(r'\[rip ([+-]) (0x[0-9a-f]+)\]', oper)
    if m:
        target = addr+size + (1 if m[1]=='+' else -1)*int(m[2],16)
        text += f' ; -> {target:#x}'
        try:
            p=off(target); value=data[p:p+100].split(b'\0')[0]
            if value and all(32 <= x < 127 for x in value): text += ' '+repr(value.decode())
        except ValueError: pass
    return text
mode = sys.argv[1]
if mode == 'strings':
    pattern = re.compile(sys.argv[2], re.I)
    for m in re.finditer(rb'[\x20-\x7e]{4,}\x00', data):
        s=m.group()[:-1].decode('ascii')
        if pattern.search(s):
            print(hex(va_at(m.start())), repr(s[:350]))
elif mode == 'function':
    for arg in sys.argv[2:]:
        va=int(arg,0); b=bounds(va)
        if not b:
            print('No unwind bounds:', arg); continue
        start,end=b
        print('FUNCTION',hex(start),hex(end))
        for addr,size,mnem,oper in md.disasm_lite(data[off(start):off(start)+end-start],start):
            print(annotated(addr,size,mnem,oper))
elif mode == 'range':
    start,end=(int(x,0) for x in sys.argv[2:4])
    for addr,size,mnem,oper in md.disasm_lite(data[off(start):off(start)+end-start],start):
        print(annotated(addr,size,mnem,oper))
elif mode == 'memrefs':
    targets={int(arg,0) for arg in sys.argv[2:]}
    for name,rva,size,raw in sections:
        if name != '.text': continue
        for addr,n,mnem,oper in md.disasm_lite(data[raw:raw+size],base+rva):
            m=re.search(r'\[rip ([+-]) (0x[0-9a-f]+)\]',oper)
            if m:
                target=addr+n+(1 if m[1]=='+' else -1)*int(m[2],16)
                if target in targets:
                    print(annotated(addr,n,mnem,oper),'unwind-range',tuple(hex(x) for x in bounds(addr)) if bounds(addr) else None)
elif mode == 'refs':
    targets={int(arg,0) for arg in sys.argv[2:]}
    # RIP-relative LEA, common code pattern for cvar registration and object access.
    for name,rva,size,raw in sections:
        if name != '.text': continue
        code=data[raw:raw+size]
        for m in re.finditer(rb'[\x48-\x4f]\x8d[\x05\x0d\x15\x1d\x25\x2d\x35\x3d]',code):
            p=m.start(); addr=base+rva+p
            if p+7 > len(code): continue
            target=addr+7+struct.unpack_from('<i',code,p+3)[0]
            if target in targets:
                print(hex(target),'referenced at',hex(addr),'function', tuple(hex(x) for x in bounds(addr)) if bounds(addr) else None)
