# sass.py DIR OUT: for each kernel's SASS (DIR/*.sass, cuobjdump -sass of ptxas's sm_120 build) and its ptxas log
# (DIR/*.ptxas.txt): registers, spills, and the main loop's instruction mix (the loop holding the most matrix-unit
# instructions: from a backward branch's target to the branch), instructions per matrix-unit instruction, by opcode.
import os, re, sys, collections
d, out = sys.argv[1], sys.argv[2]
line_re = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]\s+)?([A-Z0-9_.]+)([^;]*);')
rows = []
for f in sorted(os.listdir(d)):
    if not f.endswith('.sass'):
        continue
    name = f[:-5]
    ins = []
    for line in open(os.path.join(d, f), encoding='utf-8', errors='replace'):
        m = line_re.search(line)
        if m:
            ins.append((int(m.group(1), 16), m.group(3), m.group(4)))
    addr = {a: i for i, (a, _, _) in enumerate(ins)}
    best = None
    for i, (a, op, args) in enumerate(ins):
        if op.startswith('BRA'):
            t = re.search(r'0x([0-9a-f]+)', args)
            if t and int(t.group(1), 16) < a and int(t.group(1), 16) in addr:
                body = ins[addr[int(t.group(1), 16)]:i + 1]
                mma = sum(1 for _, o, _ in body if o.startswith(('HMMA', 'IMMA')))
                if best is None or mma > best[0]:
                    best = (mma, body)
    log = open(os.path.join(d, name + '.ptxas.txt'), encoding='utf-8', errors='replace').read() if os.path.exists(os.path.join(d, name + '.ptxas.txt')) else ''
    regs = re.search(r'Used (\d+) registers', log)
    spill = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', log)
    if best is None or best[0] == 0:
        rows.append(f'{name}: no loop with matrix-unit instructions; {len(ins)} instructions; registers {regs.group(1) if regs else "?"}')
        continue
    mma, body = best
    ops = collections.Counter(o.split('.')[0] for _, o, _ in body)
    top = ', '.join(f'{o} {c}' for o, c in ops.most_common(14))
    rows.append(f'{name}: registers {regs.group(1) if regs else "?"}, spill stores/loads {spill.group(1) + "/" + spill.group(2) if spill else "?"}; '
                f'main loop {len(body)} instructions, {mma} matrix-unit, {len(body) / mma:.2f} a matrix-unit instruction; {top}')
open(out, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
