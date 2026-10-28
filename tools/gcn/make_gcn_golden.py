"""The goldens for Idrak's GCN assembler and code object writer (plan 35, phase G0b): LLVM's bytes for every gfx9
instruction form Idrak's GCN kernels use, and LLVM's code objects for the gate kernel, fill_f32. The tests compare
Idrak's bytes with these files in C#, without LLVM; LLVM is a development tool here, never a dependency of the library
or the tests.

    python3 -I tools/gcn/make_gcn_golden.py tests/Idrak.Tests/data/gcn
    LLVM_MC=llvm-mc-18 LLD=ld.lld-18 python3 -I tools/gcn/make_gcn_golden.py tests/Idrak.Tests/data/gcn

Needs Python 3 and LLVM with the AMDGPU target (made with LLVM 18.1): llvm-mc, ld.lld, llvm-readelf and llvm-objdump,
named by LLVM_MC, LLD, LLVM_READELF and LLVM_OBJDUMP (defaults: those plain names on PATH). Nothing is downloaded.

Reads, from the folder (both checked in):
- gfx9/forms.s: the gfx9 instruction forms (GCN 5 and CDNA), one per line, with the labels its branches name;
- kernels/fill_f32.s: the gate kernel, with its .amdhsa_kernel descriptor and .amdgpu_metadata.

Writes, into the folder:
- gfx9/forms.golden: one record per instruction per CPU, "<mcpu>\\t<the instruction as llvm-mc prints it>\\t<its
  bytes as lowercase hex>", for gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx940 and gfx942 (UTF-8, '#' comments).
  Each CPU's records are forms.s assembled as one block from offset 0, less the lines that CPU rejects; a record
  "<label>:" with no bytes is a label at its offset, and branches are resolved against it (llvm-mc -show-encoding
  leaves branch bytes as fixups, so the bytes come from the assembled object's .text, checked against every byte
  -show-encoding does give). The lines a CPU rejects are listed in a "# rejected:" comment section for that CPU, with
  llvm-mc's reason. A line that no CPU accepts stops the script: it is a mistake in forms.s;
- kernels/fill_f32.gfx90c.v4.co and kernels/fill_f32.gfx90c.v5.co: llvm-mc -filetype=obj with code object version 4
  or 5, linked by ld.lld -shared (the HIP runtime loads such an object). The one source serves both: for v5 the script
  writes the metadata's "amdhsa.version: [ 1, 2 ]" in place of v4's "[ 1, 1 ]";
- next to each code object, a .txt dump to read it as text: llvm-readelf -h -l -S -s -d --notes, llvm-objdump -d,
  and llvm-objdump -s -j .rodata (the 64-byte kernel descriptor).
Reruns write the same bytes with the same LLVM release (no time, path or host is written); the LLVM version is written
in forms.golden's header and the README names it.
"""
import os
import re
import subprocess
import sys
import tempfile

TRIPLE = "amdgcn-amd-amdhsa"
GFX9_CPUS = ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx90c", "gfx940", "gfx942"]
KERNEL_CPU = "gfx90c"
KERNEL_VERSIONS = [4, 5]
METADATA_VERSIONS = {4: "amdhsa.version: [ 1, 1 ]", 5: "amdhsa.version: [ 1, 2 ]"}

LLVM_MC = os.environ.get("LLVM_MC", "llvm-mc")
LLD = os.environ.get("LLD", "ld.lld")
LLVM_READELF = os.environ.get("LLVM_READELF", "llvm-readelf")
LLVM_OBJDUMP = os.environ.get("LLVM_OBJDUMP", "llvm-objdump")

DIAGNOSTIC = re.compile(r"^<stdin>:(\d+):\d+: (error|warning): (.*)$")
LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):$")


def run(arguments, stdin=None, cwd=None):
    """Runs a tool with a fixed locale; returns (exit code, stdout, stderr) as text."""
    environment = dict(os.environ, LC_ALL="C", LANG="C")
    done = subprocess.run(arguments, input=stdin, cwd=cwd, env=environment, capture_output=True, text=True,
                          encoding="utf-8")
    return done.returncode, done.stdout, done.stderr


def checked(arguments, stdin=None, cwd=None):
    code, out, err = run(arguments, stdin, cwd)
    if code != 0 or err.strip():
        sys.exit(f"{' '.join(arguments)} failed ({code}):\n{err}")
    return out


def llvm_version():
    out = checked([LLVM_MC, "--version"])
    for line in out.splitlines():
        if "LLVM version" in line:
            return line.strip()
    sys.exit(f"{LLVM_MC} --version names no LLVM version")


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_text(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if "\r" in text:
        sys.exit(f"{path}: carriage returns; the sources are kept with LF line ends")
    return text


def elf_section(data, wanted):
    """The bytes of a named section of a little-endian ELF64 file, and the names of all its sections."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        sys.exit("not a little-endian ELF64 object")
    shoff = int.from_bytes(data[0x28:0x30], "little")
    shentsize = int.from_bytes(data[0x3A:0x3C], "little")
    shnum = int.from_bytes(data[0x3C:0x3E], "little")
    shstrndx = int.from_bytes(data[0x3E:0x40], "little")
    headers = []
    for i in range(shnum):
        h = data[shoff + i * shentsize: shoff + (i + 1) * shentsize]
        name = int.from_bytes(h[0:4], "little")
        offset = int.from_bytes(h[0x18:0x20], "little")
        size = int.from_bytes(h[0x20:0x28], "little")
        headers.append((name, offset, size))
    strings_offset = headers[shstrndx][1]

    def name_of(index):
        end = data.index(b"\0", strings_offset + index)
        return data[strings_offset + index:end].decode("ascii")

    names = [name_of(h[0]) for h in headers]
    for name, (_, offset, size) in zip(names, headers):
        if name == wanted:
            return data[offset:offset + size], names
    sys.exit(f"no {wanted} section")


def source_items(text):
    """forms.s as (line number, 'label' or 'instruction', text without its comment), in order."""
    items = []
    for number, line in enumerate(text.split("\n"), start=1):
        code = line.split(";", 1)[0].strip()
        if not code:
            continue
        label = LABEL.match(code)
        items.append((number, "label", label.group(1)) if label else (number, "instruction", code))
    return items


def printed_items(out):
    """llvm-mc -show-encoding's output as ('label', name, None) and ('instruction', text, byte tokens)."""
    items = []
    for line in out.split("\n"):
        code = line.strip()
        if not code or code.startswith(";") or code.startswith("."):
            continue                                                     # fixup notes and directives
        label = LABEL.match(code)
        if label:
            items.append(("label", label.group(1), None))
            continue
        text, marker, encoding = code.partition("; encoding: [")
        if not marker or not encoding.endswith("]"):
            sys.exit(f"llvm-mc printed an instruction without its encoding: {code}")
        tokens = encoding[:-1].split(",")
        items.append(("instruction", text.strip(), tokens))
    return items


def assemble_forms(cpu, source, workdir):
    """One CPU's records and rejected lines: [(text, bytes or None)], [(line number, source text, reason)]."""
    base = [LLVM_MC, f"-triple={TRIPLE}", f"-mcpu={cpu}"]
    _, _, err = run(base + ["-show-encoding", "-o", os.devnull], source)
    rejected = {}
    for line in err.split("\n"):
        diagnostic = DIAGNOSTIC.match(line)
        if diagnostic is None:
            continue
        number, kind, reason = int(diagnostic.group(1)), diagnostic.group(2), diagnostic.group(3)
        if kind == "warning":
            sys.exit(f"{cpu}: forms.s:{number}: llvm-mc warned: {reason}")
        rejected.setdefault(number, reason)

    lines = source.split("\n")
    items = source_items(source)
    texts = {number: code for number, _, code in items}
    for number in rejected:
        if number not in texts or any(n == number and kind == "label" for n, kind, _ in items):
            sys.exit(f"{cpu}: forms.s:{number}: llvm-mc rejected a line that is not an instruction: {rejected[number]}")
        lines[number - 1] = ""                                           # blank, so line numbers stay
    accepted_source = "\n".join(lines)
    accepted = [item for item in items if item[0] not in rejected]

    printed = printed_items(checked(base + ["-show-encoding"], accepted_source))
    object_path = os.path.join(workdir, f"forms.{cpu}.o")
    checked(base + ["-filetype=obj", "-o", object_path], accepted_source)
    with open(object_path, "rb") as f:
        data = f.read()
    text, names = elf_section(data, ".text")
    if any(name.startswith(".rel") for name in names):
        sys.exit(f"{cpu}: the forms' object has relocations ({names}); every label must be resolved in the block")

    if len(printed) != len(accepted):
        sys.exit(f"{cpu}: {len(accepted)} lines accepted but llvm-mc printed {len(printed)}")
    records = []
    offset = 0
    for (number, kind, code), (printed_kind, printed_text, tokens) in zip(accepted, printed):
        if kind != printed_kind or (kind == "label" and code != printed_text):
            sys.exit(f"{cpu}: forms.s:{number}: '{code}' printed as '{printed_text}'")
        if kind == "label":
            records.append((printed_text + ":", None))
            continue
        encoded = text[offset:offset + len(tokens)]
        if len(encoded) != len(tokens):
            sys.exit(f"{cpu}: forms.s:{number}: .text ends inside '{printed_text}'")
        for i, token in enumerate(tokens):
            if token.startswith("0x") and int(token, 16) != encoded[i]:
                sys.exit(f"{cpu}: forms.s:{number}: byte {i} of '{printed_text}' is {encoded[i]:#04x} in .text, "
                         f"{token} in -show-encoding")
        records.append((printed_text, encoded))
        offset += len(tokens)
    if offset != len(text):
        sys.exit(f"{cpu}: .text holds {len(text)} bytes, the records {offset}")
    return records, [(number, texts[number], rejected[number]) for number in sorted(rejected)]


def make_forms(folder, version, workdir):
    source_path = os.path.join(folder, "gfx9", "forms.s")
    source = read_text(source_path)
    instructions = sum(1 for _, kind, _ in source_items(source) if kind == "instruction")
    results = {cpu: assemble_forms(cpu, source, workdir) for cpu in GFX9_CPUS}

    everywhere = set.intersection(*({number for number, _, _ in rejected} for _, rejected in results.values()))
    if everywhere:
        sys.exit(f"forms.s lines no gfx9 CPU accepts (mistakes in forms.s): {sorted(everywhere)}")

    out = [
        "# The gfx9 instruction forms of forms.s (GCN 5 and CDNA) as LLVM assembles them: the goldens Idrak's GCN",
        "# assembler is checked against. Made by tools/gcn/make_gcn_golden.py with " + version + "; do not edit,",
        "# rerun the script.",
        "#",
        "# UTF-8, one record a line, three fields split by tabs: the CPU (llvm-mc's -mcpu), the instruction as",
        "# llvm-mc -show-encoding prints it (its canonical form, which llvm-mc reads back), and its bytes in memory",
        "# order as lowercase hex without separators (4 or 8 bytes on gfx9).",
        "#",
        "# Each CPU's records are one block: forms.s assembled as a whole, less the lines that CPU rejects, from offset",
        "# 0, so a record's offset is the sum of the sizes of the records before it for that CPU. A record whose",
        "# instruction field is a name followed by ':' and whose bytes field is empty is a label at that offset.",
        "# Branches (s_branch, s_cbranch_*) to a label are resolved against it: the SOPP simm16 is",
        "# (label offset - (branch offset + 4)) / 4, as a signed 16-bit count of dwords. Branches written with a number",
        "# hold that number as the simm16.",
        "#",
        "# Lines starting with '#' are comments. Each CPU's records follow its \"# rejected:\" section, which lists the",
        "# forms.s lines that CPU refuses: '#', a tab, the forms.s line number, a tab, the line, a tab, llvm-mc's reason.",
        "#",
        f"# forms.s: {instructions} instruction lines. CPUs: {', '.join(GFX9_CPUS)}.",
    ]
    for cpu in GFX9_CPUS:
        records, rejected = results[cpu]
        count = sum(1 for _, encoded in records if encoded is not None)
        out.append(f"# {cpu}: {count} forms, {len(rejected)} rejected")
    for cpu in GFX9_CPUS:
        records, rejected = results[cpu]
        out.append("")
        out.append(f"# rejected: {cpu} ({len(rejected)} of {instructions})")
        for number, code, reason in rejected:
            out.append(f"#\t{number}\t{code}\t{reason}")
        out.append(f"# {cpu}")
        for printed, encoded in records:
            out.append(f"{cpu}\t{printed}\t{encoded.hex() if encoded is not None else ''}")
    write_text(os.path.join(folder, "gfx9", "forms.golden"), "\n".join(out) + "\n")
    return {cpu: (sum(1 for _, e in results[cpu][0] if e is not None), results[cpu][1]) for cpu in GFX9_CPUS}


def make_kernel(folder, name, workdir):
    kernels = os.path.join(folder, "kernels")
    source = read_text(os.path.join(kernels, name + ".s"))
    v4 = METADATA_VERSIONS[4]
    if source.count(v4) != 1:
        sys.exit(f"{name}.s: '{v4}' must appear once (the v5 object is made by replacing it)")
    made = []
    for code_object in KERNEL_VERSIONS:
        text = source.replace(v4, METADATA_VERSIONS[code_object])
        stem = f"{name}.{KERNEL_CPU}.v{code_object}"
        object_path = os.path.join(workdir, stem + ".o")
        checked([LLVM_MC, f"-triple={TRIPLE}", f"-mcpu={KERNEL_CPU}", f"--amdhsa-code-object-version={code_object}",
                 "-filetype=obj", "-o", object_path], text)
        checked([LLD, "-shared", object_path, "-o", os.path.join(kernels, stem + ".co")])
        file = stem + ".co"
        dump = []
        for arguments in ([LLVM_READELF, "-h", "-l", "-S", "-s", "-d", "--notes", file],
                          [LLVM_OBJDUMP, "-d", file],
                          [LLVM_OBJDUMP, "-s", "-j", ".rodata", file]):
            dump.append("$ " + " ".join([os.path.basename(arguments[0])] + arguments[1:]))
            dump.append(checked(arguments, cwd=kernels).rstrip("\n"))
            dump.append("")
        write_text(os.path.join(kernels, stem + ".txt"), "\n".join(dump))
        made.append(file)
    return made


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 -I tools/gcn/make_gcn_golden.py <the data folder, tests/Idrak.Tests/data/gcn>")
    folder = sys.argv[1]
    version = llvm_version()
    with tempfile.TemporaryDirectory(prefix="gcn-golden-") as workdir:
        forms = make_forms(folder, version, workdir)
        made = make_kernel(folder, "fill_f32", workdir)
    print(version)
    for cpu, (count, rejected) in forms.items():
        print(f"{cpu}: {count} forms, {len(rejected)} rejected")
    print("code objects: " + ", ".join(made))


if __name__ == "__main__":
    main()
