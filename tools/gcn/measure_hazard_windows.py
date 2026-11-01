"""How far each gfx10/gfx11 hazard reaches (plan 35, G8a-2): case A, then n fillers (a VALU, a transcendental, an SALU,
a wait), then case B, for n = 0 to 6; prints where llc 18 inserts its fix (@i: before the i-th given instruction).
    LLC=<llvm-18 llc> python3 -I tools/gcn/measure_hazard_windows.py [cases]
"""
import os, subprocess, sys, tempfile
LLC = os.environ.get("LLC", "llc")
V = '$vgpr10 = V_MOV_B32_e32 0, implicit $exec'
T = '$vgpr11 = V_SQRT_F32_e32 $vgpr12, implicit $mode, implicit $exec'
SA = '$sgpr20 = S_MOV_B32 0'
W0 = 'S_WAITCNT 0'
CASES = {  # name: (cpu, first, second, filler kinds)
 'vmem_salu': ('gfx1010', ['$vgpr1 = GLOBAL_LOAD_DWORD_SADDR $sgpr2_sgpr3, $vgpr0, 0, 0, implicit $exec'], ['$sgpr2 = S_MOV_B32 0'], [SA, V, W0]),
 'smem_valu': ('gfx1010', ['$sgpr0 = S_LOAD_DWORD_IMM $sgpr4_sgpr5, 0, 0'], ['$sgpr4 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec'], [SA, V, W0, 'S_WAITCNT 49279']),
 'trans_valu': ('gfx1100', ['$vgpr0 = V_EXP_F32_e32 $vgpr1, implicit $mode, implicit $exec'], ['$vgpr2 = V_ADD_F32_e32 $vgpr0, $vgpr0, implicit $mode, implicit $exec'], [V, T, SA]),
 'mask_salu': ('gfx1100', ['$vgpr0 = V_CNDMASK_B32_e64 0, $vgpr1, 0, $vgpr2, $sgpr0_sgpr1, implicit $exec'], ['$sgpr0_sgpr1 = S_MOV_B64 0', '$vgpr3 = V_ADD_U32_e32 $sgpr0, $vgpr0, implicit $exec'], [V, SA]),
 'mask_vcc': ('gfx1100', ['$vgpr0 = V_CNDMASK_B32_e32 $vgpr1, $vgpr2, implicit $vcc, implicit $exec'], ['$vcc = S_MOV_B64 0'], [V, SA]),
 'partial': ('gfx1100', ['$vgpr0 = V_MOV_B32_e32 0, implicit $exec', '$exec = S_MOV_B64 -1', '$vgpr1 = V_MOV_B32_e32 1, implicit $exec'], ['$vgpr2 = V_ADD_U32_e32 $vgpr0, $vgpr1, implicit $exec'], [V, SA]),
}
def run(cpu, lines):
    text = "---\nname: t\ntracksRegLiveness: false\nbody: |\n  bb.0:\n" + ''.join(f'    {l}\n' for l in lines) + "    S_ENDPGM 0\n...\n"
    with tempfile.NamedTemporaryFile('w', suffix='.mir', delete=False) as f: f.write(text); p = f.name
    r = subprocess.run([LLC, '-march=amdgcn', f'-mcpu={cpu}', '-mattr=+wavefrontsize64', '-run-pass=post-RA-hazard-rec', '-o', '-', p], capture_output=True, text=True)
    os.unlink(p)
    if r.returncode: return 'ERR ' + r.stderr.strip().splitlines()[0][:150]
    body = [l.strip() for l in r.stdout.split('body:', 1)[1].splitlines() if l.strip() and not l.strip().startswith(('bb.', '...'))]
    # positions of inserted lines (not in the given ones)
    given = list(lines); res = []
    i = 0
    for b in body:
        if i < len(given) and b.replace('undef ', '').split(',')[0][:25] == given[i].split(',')[0][:25]: i += 1; continue
        if b.startswith('S_ENDPGM'): continue
        res.append(f'@{i}:{b.split(" implicit")[0][:40]}')
    return ' '.join(res) or '-'
which = sys.argv[1:] or list(CASES)
for name in which:
    cpu, first, second, fills = CASES[name]
    for fill in fills:
        row = []
        for n in range(0, 7):
            row.append(f'{n}:{run(cpu, first + [fill] * n + second)}')
        print(f'{name} [{fill.split(" = ")[-1][:18]}]: ' + ' | '.join(row))
