"""Lists the textures the display-resolution post chain reads that were rendered at render resolution.

Post passes are the draws/dispatches recorded on the TAA context after the TAA dispatch, up to the first
render-resolution viewport (the next frame). For each texture they sample, the last earlier writer is printed
with its viewport, so inputs that only cover the render-scaled corner stand out.

Usage: python post_inputs.py <d2enhanced.log> [capture index]
"""
import re
import sys

log = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
starts = [i for i, l in enumerate(log) if l.startswith('=== capture begin')]
ends = [i for i, l in enumerate(log) if l.startswith('=== capture end')]
lines = log[starts[index]:ends[index] + 1]

entries = []
for line in lines:
    if line.startswith('  pass #') or line.startswith('  #'):
        entries.append({'header': line.strip(), 'body': []})
    elif entries and line.startswith('     '):
        entries[-1]['body'].append(line.strip())

for e in entries:
    e['ctx'] = re.search(r'ctx=(\S+)', e['header']).group(1)
    e['vp'] = next((b.split()[1] for b in e['body'] if b.startswith('viewport')), '-')
    e['ps'] = next((b.split()[0] + ' ' + b.split('[', 1)[-1][:110] for b in e['body'] if b.startswith(('PS=', 'bindings'))), '')
    e['writes'] = [m.group(1) for b in e['body'] for m in [re.match(r'(?:rt\d|u\d+)\s+viewfmt=\d+\s+res=(\S+)', b)] if m]
    e['reads'] = [(m.group(1), m.group(2), m.group(3)) for b in e['body']
                  for m in [re.match(r'(t\d+)\s+\S+\s+viewfmt=\d+\s+res=(\S+) (\d+x\d+)', b)] if m]

taa = next(i for i, e in enumerate(entries) if '[TAA]' in e['header'])
renderVp = re.search(r'dither\+viewport=\(\S+ \S+ (\d+)\.\d+ (\d+)\.', ' '.join(entries[taa]['body'])).groups()
renderVp = f"{renderVp[0]}x{renderVp[1]}"
ctx = entries[taa]['ctx']
print(f"TAA ctx {ctx}, render viewport {renderVp}")

post = []
for i in range(taa + 1, len(entries)):
    e = entries[i]
    if e['ctx'] != ctx:
        continue
    if e['vp'] == renderVp:
        break
    post.append(i)

seen = set()
for i in post:
    e = entries[i]
    for slot, res, size in e['reads']:
        if res in seen or any(res in entries[j]['writes'] for j in post if j < i):
            continue
        seen.add(res)
        writer = next((entries[j] for j in range(i - 1, -1, -1) if res in entries[j]['writes']), None)
        where = f"written vp {writer['vp']:10s} by {writer['ps'][:90]}" if writer else 'not written in capture'
        print(f"{e['header'][:12]:12s} {slot:3s} {size:10s} {res}  {where}")
        print(f"{'':16s}reader {e['ps'][:120]}")
