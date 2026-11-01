"""The gfx10 and gfx11 hazards as LLVM 18 fixes them (plan 35, G8a-2): which cases its post-RA hazard recognizer
fixes on gfx1010, gfx1030, gfx1100 and gfx1102 in wave64, and what it inserts. Development only, as make_gcn_golden.py:
    LLC=<llvm-18 llc> python3 -I tools/gcn/measure_hazards.py [cpus] [cases]
"""
import os, subprocess, sys, tempfile
LLC = os.environ.get("LLC", "llc")
CPUS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['gfx1010', 'gfx1030', 'gfx1100', 'gfx1102']
CASES = {
 'vmem_sgpr_after_valu': ['$sgpr2 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec',
                          '$vgpr1 = GLOBAL_LOAD_DWORD_SADDR $sgpr2_sgpr3, $vgpr0, 0, 0, implicit $exec'],
 'vmem_exec_after_vcmpx': ['V_CMPX_GT_U32_nosdst_e32 $vgpr0, $vgpr1, implicit-def $exec, implicit $exec',
                           '$vgpr1 = GLOBAL_LOAD_DWORD_SADDR $sgpr2_sgpr3, $vgpr0, 0, 0, implicit $exec'],
 'lane_select_after_valu': ['$sgpr4 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec',
                            '$sgpr0 = V_READLANE_B32 $vgpr1, $sgpr4'],
 'lane_select_vcc_after_vcmp': ['$vcc = V_CMP_GT_U32_e64 $vgpr0, $vgpr1, implicit $exec',
                                '$sgpr0 = V_READLANE_B32 $vgpr1, $vcc_lo'],
 'div_fmas_after_valu_vcc': ['$vgpr0, $vcc = V_DIV_SCALE_F32_e64 0, $vgpr1, 0, $vgpr2, 0, $vgpr1, 0, 0, implicit $mode, implicit $exec',
                             '$vgpr3 = V_DIV_FMAS_F32_e64 0, $vgpr1, 0, $vgpr2, 0, $vgpr3, 0, 0, implicit $mode, implicit $vcc, implicit $exec'],
 'wide_store_then_valu': ['GLOBAL_STORE_DWORDX4_SADDR $vgpr0, $vgpr4_vgpr5_vgpr6_vgpr7, $sgpr2_sgpr3, 0, 0, implicit $exec',
                          '$vgpr4 = V_MOV_B32_e32 0, implicit $exec'],
 'smem_sgpr_after_salu': ['$sgpr4 = S_MOV_B32 0', '$sgpr0 = S_LOAD_DWORD_IMM $sgpr4_sgpr5, 0, 0'],
 'smem_sgpr_after_valu': ['$sgpr4 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec', '$sgpr0 = S_LOAD_DWORD_IMM $sgpr4_sgpr5, 0, 0'],
 'salu_after_valu_sgpr': ['$sgpr4 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec', '$sgpr5 = S_ADD_U32 $sgpr4, 1, implicit-def $scc'],
 'valu_after_valu_sgpr': ['$sgpr4 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec', '$vgpr1 = V_ADD_U32_e32 $sgpr4, $vgpr0, implicit $exec'],
 'vmem_then_salu_writes_sgpr': ['$vgpr1 = GLOBAL_LOAD_DWORD_SADDR $sgpr2_sgpr3, $vgpr0, 0, 0, implicit $exec', '$sgpr2 = S_MOV_B32 0'],
 'ds_then_salu_writes_m0': ['$vgpr1 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec', '$m0 = S_MOV_B32 0'],
 'smem_then_valu_writes_sgpr': ['$sgpr0 = S_LOAD_DWORD_IMM $sgpr4_sgpr5, 0, 0', '$sgpr4 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec'],
 'smem_then_vcmp_writes_sgpr': ['$sgpr0 = S_LOAD_DWORD_IMM $sgpr4_sgpr5, 0, 0', '$sgpr4_sgpr5 = V_CMP_GT_U32_e64 $vgpr0, $vgpr1, implicit $exec'],
 'readlane_exec_then_vcmpx': ['$sgpr0 = V_READLANE_B32 $vgpr1, $exec_lo', 'V_CMPX_GT_U32_nosdst_e32 $vgpr0, $vgpr1, implicit-def $exec, implicit $exec'],
 'salu_reads_exec_then_vcmpx': ['$sgpr0 = S_MOV_B32 $exec_lo', 'V_CMPX_GT_U32_nosdst_e32 $vgpr0, $vgpr1, implicit-def $exec, implicit $exec'],
 'trans_then_valu_reads': ['$vgpr0 = V_EXP_F32_e32 $vgpr1, implicit $mode, implicit $exec', '$vgpr2 = V_ADD_F32_e32 $vgpr0, $vgpr0, implicit $mode, implicit $exec'],
 'trans_then_trans_reads': ['$vgpr0 = V_EXP_F32_e32 $vgpr1, implicit $mode, implicit $exec', '$vgpr2 = V_LOG_F32_e32 $vgpr0, implicit $mode, implicit $exec'],
 'trans_then_mem_reads': ['$vgpr0 = V_RCP_F32_e32 $vgpr1, implicit $mode, implicit $exec', 'GLOBAL_STORE_DWORD_SADDR $vgpr2, $vgpr0, $sgpr2_sgpr3, 0, 0, implicit $exec'],
 'cndmask_mask_salu_write_valu': ['$vgpr0 = V_CNDMASK_B32_e64 0, $vgpr1, 0, $vgpr2, $sgpr0_sgpr1, implicit $exec',
                                  '$sgpr0_sgpr1 = S_MOV_B64 0', '$vgpr3 = V_ADD_U32_e32 $sgpr0, $vgpr0, implicit $exec'],
 'cndmask_mask_salu_write': ['$vgpr0 = V_CNDMASK_B32_e64 0, $vgpr1, 0, $vgpr2, $sgpr0_sgpr1, implicit $exec', '$sgpr0_sgpr1 = S_MOV_B64 0'],
 'cndmask_vcc_salu_write': ['$vgpr0 = V_CNDMASK_B32_e32 $vgpr1, $vgpr2, implicit $vcc, implicit $exec', '$vcc = S_MOV_B64 0'],
 'partial_forwarding': ['$vgpr0 = V_MOV_B32_e32 0, implicit $exec', '$exec = S_MOV_B64 -1', '$vgpr1 = V_MOV_B32_e32 1, implicit $exec',
                        '$vgpr2 = V_ADD_U32_e32 $vgpr0, $vgpr1, implicit $exec'],
 'valu_writes_vgpr_then_exec_salu': ['$vgpr0 = V_MOV_B32_e32 0, implicit $exec', '$exec = S_MOV_B64 -1', '$vgpr2 = V_ADD_U32_e32 $vgpr0, $vgpr0, implicit $exec'],
 'lds_branch_vmem': None,
 'getreg_after_setreg': ['S_SETREG_IMM32_B32 0, 2049, implicit-def $mode, implicit $mode', '$sgpr0 = S_GETREG_B32 2049, implicit $mode'],
 'setreg_after_setreg': ['S_SETREG_IMM32_B32 0, 2049, implicit-def $mode, implicit $mode', 'S_SETREG_IMM32_B32 1, 2049, implicit-def $mode, implicit $mode'],
}
BRANCH = """    $vgpr1 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec
    S_CBRANCH_SCC1 %bb.2, implicit $scc
  bb.1:
    successors: %bb.2
    S_NOP 0
  bb.2:
    $vgpr2 = GLOBAL_LOAD_DWORD_SADDR $sgpr2_sgpr3, $vgpr0, 0, 0, implicit $exec
    S_ENDPGM 0
"""
def mir(lines):
    body = BRANCH if lines is None else ''.join(f'    {l}\n' for l in lines) + '    S_ENDPGM 0\n'
    head = "  bb.0:\n" + ("    successors: %bb.1, %bb.2\n" if lines is None else "")
    return f"---\nname: t\ntracksRegLiveness: false\nbody: |\n{head}{body}...\n"
only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
for name, lines in CASES.items():
    if only and name not in only: continue
    out = []
    for cpu in CPUS:
        with tempfile.NamedTemporaryFile('w', suffix='.mir', delete=False) as f:
            f.write(mir(lines)); path = f.name
        r = subprocess.run([LLC, '-march=amdgcn', f'-mcpu={cpu}', '-mattr=+wavefrontsize64', '-run-pass=post-RA-hazard-rec', '-o', '-', path],
                           capture_output=True, text=True)
        os.unlink(path)
        if r.returncode:
            out.append(f'{cpu}: ERROR {r.stderr.strip().splitlines()[0][:200]}'); continue
        body = r.stdout.split('body:', 1)[1]
        got = [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith(('bb.', 'successors', 'liveins', '...', 'S_ENDPGM'))]
        given = len(lines) if lines else 0
        inserted = [g for g in got if not any(g.split(' ')[0] in (x.split(' = ')[-1].split(' ')[0] if ' = ' in x else x.split(' ')[0]) for x in (lines or [])) ] if lines else got
        out.append(f'{cpu}: ' + ' ; '.join(g for g in got if g.split()[0] in ('S_NOP', 'S_WAITCNT_DEPCTR', 'V_NOP_e32', 'S_MOV_B32', 'S_WAITCNT_VSCNT', 'S_WAITCNT') or g.startswith(('$sgpr_null', 'S_WAITCNT'))))
    print(f'{name}: ' + ' | '.join(out))
