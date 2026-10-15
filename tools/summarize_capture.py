"""Summarizes the F8 draw-pass capture in d2enhanced.log: one line per distinct pass (pixel shader + targets).

Usage: python summarize_capture.py <d2enhanced.log> [capture index (0 = first)] [filter substring]
"""
import re
import sys
from collections import OrderedDict

log = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
needle = sys.argv[3].lower() if len(sys.argv) > 3 else ''

starts = [i for i, l in enumerate(log) if l.startswith('=== capture begin')]
ends = [i for i, l in enumerate(log) if l.startswith('=== capture end')]
lines = log[starts[index]:ends[index] + 1]

entries = []  # (kind, header, body lines)
current = None
for line in lines:
    if line.startswith('  pass #') or line.startswith('  #'):
        current = [line.strip(), []]
        entries.append(current)
    elif current and line.startswith('     '):
        current[1].append(line.strip())

FORMATS = {2: 'RGBA32F', 10: 'RGBA16F', 11: 'RGBA16', 24: 'RGB10A2', 26: 'R11G11B10F', 28: 'RGBA8', 29: 'RGBA8_SRGB',
           34: 'RG16F', 35: 'RG16', 41: 'R32F', 44: 'R24G8', 45: 'D24S8', 46: 'R24X8', 54: 'R16F', 56: 'R16',
           61: 'R8', 49: 'RG8', 87: 'BGRA8', 91: 'BGRA8_SRGB', 16: 'RG32F', 19: 'D32S8', 40: 'D32F', 55: 'D16'}


def fmt(n):
    return FORMATS.get(int(n), 'fmt' + n)


summary = OrderedDict()
for header, body in entries:
    if not header.startswith('pass'):
        continue
    draws = int(re.search(r'draws=(\d+)', header).group(1))
    ps = next((b for b in body if b.startswith('PS=')), 'PS=?')
    psBind = re.sub(r'\s+', ' ', ps.split('[', 1)[1].rstrip(']').strip()) if '[' in ps else ''
    rts = [f"rt{m.group(1)}:{fmt(m.group(2))}@{m.group(3)}" for b in body
           for m in [re.match(r'rt(\d) viewfmt=(\d+) res=\S+ (\d+x\d+)', b)] if m]
    depth = next((f"depth:{fmt(m.group(1))}@{m.group(2)}" for b in body
                  for m in [re.match(r'depth viewfmt=(\d+) res=\S+ (\d+x\d+)', b)] if m), '')
    srvs = [f"t{m.group(1)}:{fmt(m.group(2))}@{m.group(3)}" for b in body
            for m in [re.match(r't(\d+) \S+ viewfmt=(\d+) res=\S+ (\d+x\d+)', b)] if m]
    viewport = next((b.split()[1] for b in body if b.startswith('viewport')), '?')
    key = (ps.split()[0], psBind, ' '.join(rts), depth, viewport)
    item = summary.setdefault(key, {'passes': 0, 'draws': 0, 'srvs': srvs})
    item['passes'] += 1
    item['draws'] += draws

for (ps, binds, rts, depth, viewport), item in summary.items():
    text = f"{item['draws']:5d} draws {item['passes']:3d}x  vp {viewport:10s} {rts} {depth}  {ps} [{binds}]  srv {' '.join(item['srvs'][:6])}"
    if needle in text.lower():
        print(text[:420])
print(f"-- {len(summary)} distinct passes, {sum(i['draws'] for i in summary.values())} draws")
