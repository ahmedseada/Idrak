# GCN goldens

LLVM's bytes for the AMD GCN code Idrak assembles itself (plan 35). The tests compare Idrak's GCN assembler and code
object writer with these files in C#, without LLVM: LLVM is a development tool, never a dependency of the library or
the tests.

Made by `tools/gcn/make_gcn_golden.py` with **LLVM 18.1.3** (Ubuntu's build: `llvm-mc`, `ld.lld`, `llvm-readelf`,
`llvm-objdump`). To remake them after a source changes:

```sh
python3 -I tools/gcn/make_gcn_golden.py tests/Idrak.Tests/data/gcn
```

The gfx10 and gfx11 goldens (plan 35, G8a) were made with **LLVM 18.1.8** (Ubuntu's packages `llvm-18`,
`llvm-18-linker-tools`, `lld-18`, `libllvm18`, `liblld-18`, unpacked with `dpkg -x` into a user folder: LLVM alone, no
SDK or ROCm), each family's `forms.s` first from the kernels' printed text (what `idrak kernels dump gcn` writes for a
gfx9 processor), then its goldens, without touching gfx9's:

```sh
python3 -I tools/gcn/make_family_forms.py tests/Idrak.Tests/data/gcn <the kernels' printed gfx9 text>
python3 -I tools/gcn/make_gcn_golden.py tests/Idrak.Tests/data/gcn --families gfx10,gfx11
```

The script writes the same bytes on every run with the same LLVM release. The `.s` files are the sources, kept with
LF line ends; the rest is made by the script and never edited by hand.

| File | What it is |
|---|---|
| `gfx9/forms.s` | Source: the gfx9 instruction forms (GCN 5 and CDNA) the GCN kernels use, one per line, with the labels its branches name |
| `gfx9/forms.golden` | Each form as llvm-mc prints it and its bytes, per CPU: gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx940, gfx942. Tab-separated `<mcpu>`, `<instruction>`, `<hex bytes>`; `#` comments. Each CPU's records are one block from offset 0, labels are records with no bytes, and branches are resolved against them (the header states the rule). Each CPU's `# rejected:` section lists the lines it refuses and why |
| `gfx10/forms.s`, `gfx11/forms.s` | Sources (made by `make_family_forms.py`): the instruction forms the GCN kernels use, in the family's own syntax as llvm-mc prints it for gfx1030 or gfx1100; wave64 forms (the kernels' waves), each followed by its wave32 spelling where that differs |
| `gfx10/forms.golden`, `gfx11/forms.golden` | gfx9's format, per CPU and wave size: `gfx1010-w32`, `gfx1010-w64`, `gfx1030-w32`, `gfx1030-w64`; `gfx1100-w32`, `gfx1100-w64`, `gfx1102-w32`, `gfx1102-w64` (`-mattr=+wavefrontsize32` or `+wavefrontsize64`) |
| `kernels/fill_f32.s` | Source: the gate kernel (`out[i] = value` for `i < count`, blocks of 256), with its `.amdhsa_kernel` descriptor and `.amdgpu_metadata` |
| `kernels/fill_f32.gfx90c.v4.co` | The kernel as a gfx90c code object, version 4 (`llvm-mc -filetype=obj`, then `ld.lld -shared`) |
| `kernels/fill_f32.gfx90c.v4.txt` | The v4 object as text: `llvm-readelf -h -l -S -s -d --notes`, `llvm-objdump -d`, and the kernel descriptor (`llvm-objdump -s -j .rodata`) |
| `kernels/fill_f32.gfx90c.v5.co` | The same kernel as code object version 5 (metadata version 1.2) |
| `kernels/fill_f32.gfx90c.v5.txt` | The v5 object as text |
