# GCN goldens

LLVM's bytes for the AMD GCN code Idrak assembles itself (plan 35). The tests compare Idrak's GCN assembler and code
object writer with these files in C#, without LLVM: LLVM is a development tool, never a dependency of the library or
the tests.

Made by `tools/gcn/make_gcn_golden.py` with **LLVM 18.1.3** (Ubuntu's build: `llvm-mc`, `ld.lld`, `llvm-readelf`,
`llvm-objdump`). To remake them after a source changes:

```sh
python3 -I tools/gcn/make_gcn_golden.py tests/Idrak.Tests/data/gcn
```

The script writes the same bytes on every run with the same LLVM release. The `.s` files are the sources, kept with
LF line ends; the rest is made by the script and never edited by hand.

| File | What it is |
|---|---|
| `gfx9/forms.s` | Source: the gfx9 instruction forms (GCN 5 and CDNA) the GCN kernels use, one per line, with the labels its branches name |
| `gfx9/forms.golden` | Each form as llvm-mc prints it and its bytes, per CPU: gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx940, gfx942. Tab-separated `<mcpu>`, `<instruction>`, `<hex bytes>`; `#` comments. Each CPU's records are one block from offset 0, labels are records with no bytes, and branches are resolved against them (the header states the rule). Each CPU's `# rejected:` section lists the lines it refuses and why |
| `kernels/fill_f32.s` | Source: the gate kernel (`out[i] = value` for `i < count`, blocks of 256), with its `.amdhsa_kernel` descriptor and `.amdgpu_metadata` |
| `kernels/fill_f32.gfx90c.v4.co` | The kernel as a gfx90c code object, version 4 (`llvm-mc -filetype=obj`, then `ld.lld -shared`) |
| `kernels/fill_f32.gfx90c.v4.txt` | The v4 object as text: `llvm-readelf -h -l -S -s -d --notes`, `llvm-objdump -d`, and the kernel descriptor (`llvm-objdump -s -j .rodata`) |
| `kernels/fill_f32.gfx90c.v5.co` | The same kernel as code object version 5 (metadata version 1.2) |
| `kernels/fill_f32.gfx90c.v5.txt` | The v5 object as text |
