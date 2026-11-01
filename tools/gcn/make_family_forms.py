"""The gfx10 and gfx11 forms.s sources (plan 35, G8a-1): the instruction forms Idrak's GCN kernels use, in each family's
own syntax, from which make_gcn_golden.py makes the goldens. Development only, as make_gcn_golden.py.

    python3 -I tools/gcn/make_family_forms.py tests/Idrak.Tests/data/gcn <folder of the kernels' printed text>

The kernels' printed text is what `idrak kernels dump gcn` writes for a gfx9 processor (one .s file a kernel, physical
registers). Each family's forms.s holds:
- the lines of gfx9/forms.s whose mnemonic a kernel uses (DPP and SDWA forms left out: no kernel uses them), each
  rewritten by the family's rules below where gfx9's spelling or operand is not the family's, then printed by llvm-mc
  as the family's canonical form (so each line is what the family's kernels emit), in wave64, the waves Idrak's kernels
  run in on every family;
- the same forms' wave32 spellings where they differ (VCC as vcc_lo, a compare's or select's 64-bit mask as one
  register, EXEC's 64-bit operations as exec_lo's), printed for wave32.
A form llvm-mc refuses in both modes after the rules stops the script: a rule is missing. Labels are kept where they
are, so branches resolve as in gfx9's file.
"""
import os
import re
import subprocess
import sys

TRIPLE = "amdgcn-amd-amdhsa"
LLVM_MC = os.environ.get("LLVM_MC", "llvm-mc")
LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):$")
SUFFIX = re.compile(r"_(e32|e64)$")
DIAGNOSTIC = re.compile(r"^<stdin>:(\d+):\d+: (error|warning): (.*)$")

# Per family: the CPU llvm-mc prints its canonical forms for, and the rewrites from gfx9's spelling (pattern, replacement;
# None drops the line). gfx940's cache bits (sc0, sc1, nt) are no family's here; gfx10's global offsets are 12-bit signed.
FAMILIES = {
    "gfx10": ("gfx1030", [
        (r"\b(sc0|sc1|nt)\b", None),
        (r"^v_add_u32", "v_add_nc_u32"), (r"^v_sub_u32", "v_sub_nc_u32"), (r"^v_subrev_u32", "v_subrev_nc_u32"),
        (r"offset:4095\b", "offset:2047"), (r"offset:-4096\b", "offset:-2048"),
    ]),
    "gfx11": ("gfx1100", [
        (r"\b(sc0|sc1|nt)\b", None),
        (r"^v_add_u32", "v_add_nc_u32"), (r"^v_sub_u32", "v_sub_nc_u32"), (r"^v_subrev_u32", "v_subrev_nc_u32"),
    ]),
}

# wave32 spellings of a wave64 form: VCC's low half, one register for a 64-bit lane mask, exec_lo for EXEC.
WAVE32 = [
    (r"\bvcc\b", "vcc_lo"),
    (r"\bexec\b", "exec_lo"),
    (r"^(v_cmp\w*_e64|v_cmp_class_f32_e64) s\[(\d+):\d+\]", r"\1 s\2"),
    (r"^(v_cndmask_b32_e64 .*), s\[(\d+):\d+\]$", r"\1, s\2"),
    (r"^s_(and_saveexec|or_saveexec|andn2_saveexec)_b64 s\[(\d+):\d+\]", r"s_\1_b32 s\2"),
]


def run(arguments, stdin):
    done = subprocess.run(arguments, input=stdin, capture_output=True, text=True, encoding="utf-8",
                          env=dict(os.environ, LC_ALL="C", LANG="C"))
    return done.returncode, done.stdout, done.stderr


def kernel_mnemonics(folder):
    used = set()
    for name in os.listdir(folder):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            for line in f:
                code = line.split(";", 1)[0].strip()
                if code and not code.startswith(".") and not code.endswith(":"):
                    used.add(SUFFIX.sub("", code.split()[0]))
    return used


def canonical(cpu, wave, lines):
    """llvm-mc's canonical text for each line (None where it refuses one), labels as they are."""
    base = [LLVM_MC, f"-triple={TRIPLE}", f"-mcpu={cpu}", f"-mattr=+wavefrontsize{wave}"]
    source = "\n".join(lines) + "\n"
    _, _, err = run(base + ["-show-encoding", "-o", os.devnull], source)
    refused = {int(d.group(1)) for d in map(DIAGNOSTIC.match, err.split("\n")) if d}
    kept = [(i, line) for i, line in enumerate(lines, start=1) if i not in refused]
    code, out, err = run(base + ["-show-encoding"], "\n".join(line for _, line in kept) + "\n")
    if code != 0:
        sys.exit(f"{cpu} wave{wave}: llvm-mc failed on the accepted lines:\n{err}")
    printed = []
    for line in out.split("\n"):
        text = line.strip()
        if not text or text.startswith(";") or text.startswith("."):
            continue
        printed.append(text.split("; encoding:", 1)[0].strip())
    if len(printed) != len(kept):
        sys.exit(f"{cpu} wave{wave}: {len(kept)} lines accepted, {len(printed)} printed")
    result = [None] * len(lines)
    for (i, _), text in zip(kept, printed):
        result[i - 1] = text
    return result


def rewrite(line, rules):
    for pattern, replacement in rules:
        if re.search(pattern, line):
            if replacement is None:
                return None
            line = re.sub(pattern, replacement, line)
    return line


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python3 -I tools/gcn/make_family_forms.py <tests/Idrak.Tests/data/gcn> <the kernels' printed text>")
    folder, printed = sys.argv[1:]
    used = kernel_mnemonics(printed)
    with open(os.path.join(folder, "gfx9", "forms.s"), encoding="utf-8") as f:
        gfx9 = [line.split(";", 1)[0].strip() for line in f]
    chosen = [line for line in gfx9 if line and (LABEL.match(line) or SUFFIX.sub("", line.split()[0]) in used)]
    for family, (cpu, rules) in FAMILIES.items():
        wave64 = [line if LABEL.match(line) else rewrite(line, rules) for line in chosen]
        wave64 = [line for line in wave64 if line is not None]
        printed64 = canonical(cpu, 64, wave64)
        wave32 = [line if LABEL.match(line) else rewrite(line, WAVE32) for line in wave64]
        printed32 = canonical(cpu, 32, wave32)
        out, missing = [], []
        for source, p64, p32 in zip(wave64, printed64, printed32):
            if LABEL.match(source):
                out.append(source)
                continue
            if p64 is None and p32 is None:
                missing.append(source)
                continue
            for text in dict.fromkeys(t for t in (p64, p32) if t is not None):
                out.append(text)
        if missing:
            sys.exit(f"{family}: forms neither wave size accepts (a rule is missing):\n  " + "\n  ".join(missing))
        header = [
            f"; The {family} instruction forms Idrak's GCN kernels use (plan 35, G8a), in {family}'s own syntax as llvm-mc prints",
            f"; it for {cpu}: made by tools/gcn/make_family_forms.py from gfx9/forms.s and the kernels' mnemonics; rerun it,",
            "; do not edit. Wave64 forms first, each followed by its wave32 spelling where that differs; make_gcn_golden.py",
            "; assembles the file for each CPU and wave size and lists the lines each refuses.",
        ]
        os.makedirs(os.path.join(folder, family), exist_ok=True)
        with open(os.path.join(folder, family, "forms.s"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(header + out) + "\n")
        print(f"{family}: {sum(1 for l in out if not LABEL.match(l))} forms ({cpu})")


if __name__ == "__main__":
    main()
