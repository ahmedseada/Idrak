// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using Idrak;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using GcnCodeObjectReader = Idrak.Tests.Gcn.GcnCodeObjectReader;
using GcnCpu = Idrak.Tests.Gcn.GcnCpu;
using GcnDecoder = Idrak.Tests.Gcn.GcnDecoder;
using GcnDispatch = Idrak.Tests.Gcn.GcnDispatch;
using GcnFailure = Idrak.Tests.Gcn.GcnFailure;
using GcnInterpreter = Idrak.Tests.Gcn.GcnInterpreter;
using GcnInterpreterException = Idrak.Tests.Gcn.GcnInterpreterException;
using GcnKernelImage = Idrak.Tests.Gcn.GcnKernelImage;
using GcnMemory = Idrak.Tests.Gcn.GcnMemory;

// GCN code objects written by Idrak (plan 35, G0a): the target families chosen by the version a device reports, and the
// ELF code object the HIP runtime loads, read back here by a reader of its own (the ELF, its tables, the metadata's
// MessagePack). Every expected number was read from LLVM 18 (llvm-mc, ld.lld, llvm-readelf) for the same kernels on
// 2026-10-28; nothing here calls LLVM. The tests run anywhere: no device, no runtime.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnGroup =
    [
        ("gcn targets: target strings parse from the span into major, minor and stepping (one hex digit each: gfx90c is 9.0.12, gfx1030 10.3.0) and the xnack and sramecc settings, with nothing allocated; generic targets, unknown or repeated features and malformed strings are refused", GcnTargetStrings),
        ("gcn targets: gfx9 is the library's default and handles 9.0.x without a unified register file; an app's family is asked first; a target no family handles fails as not registered, naming the reported target", GcnTargetFamilies),
        ("gcn targets: ELF flags as LLVM 18 writes them for every processor (the machine value, then xnack and sramecc any, off or on); a feature the processor lacks and format 6 are refused", GcnTargetElfFlags),
        ("gcn targets: gfx9's register files, reserved scalar registers, descriptor blocks and waves per SIMD as LLVM counts them; counts past the register file are refused", GcnTargetRegisters),
        ("gcn code object: fill_f32 for gfx90c read back: the ELF header and flags, segments, sections, symbols, the dynamic and hash tables, the metadata note; the same code, flags, note and descriptor as the golden (ld.lld's objects, formats 5 and 4)", GcnCodeObjectFill),
        ("gcn code object: several kernels: code 256-byte aligned with s_nop between, descriptors 64-byte aligned, each entry offset reaching its code, every name found through the hash table, a long name in the metadata", GcnCodeObjectKernels),
        ("gcn code object: formats 5 and 4 differ only in the ABI version and amdhsa.version; format 6, a target the family does not handle and kernels it cannot describe are refused with their reason", GcnCodeObjectFormats),
        ("gcn assembler: each gfx9 form fill_f32 uses, encoded from typed operands (scalar and vector registers, VCC, EXEC, M0, off, inline and literal constants, offsets, glc), equals its forms.golden record for gfx90c; branches reach their labels; operands a form cannot take are refused, naming them; nothing is allocated per instruction", GcnAssemblerForms),
        ("gcn assembler: every forms.golden record of each registered CPU (gfx900, gfx906, gfx908, gfx90c), its text parsed from UTF-8 and encoded as one block, equals llvm-mc's bytes, labels and branches included, and each form alone too; forms.s itself (LLVM's input syntax: no suffixes, bound_ctrl:0, s_waitcnt 0) less each CPU's rejected lines gives the same block", GcnAssemblerGolden),
        ("gcn assembler: every line forms.golden lists as rejected for a registered CPU is refused for that CPU's version, naming the instruction, feature or operand (features decided by the reported version: v_fmac_f32 on 9.0.6, not 9.0.0 or 9.0.12; without a version, none)", GcnAssemblerRejected),
        ("gcn assembler: malformed or impossible text (unknown mnemonics, operands and modifiers, missing operands, bad ranges, virtual registers, directives, literals in VOP3, two scalar reads, labels defined twice or never) is refused with its line and column", GcnAssemblerMalformed),
        ("gcn assembler: assembling a whole forms block (gfx90c's 759 forms, labels and branches) allocates nothing once warm, and its code array alone after; fill_f32 written as text equals the typed fill_f32", GcnAssemblerAllocations),
        ("gcn assembler: fill_f32 written with virtual registers allocates to the golden's registers (s[4:7] for its arguments, s2 and v0 rewritten in place, s[0:1] for the saved EXEC) with its one wait: llvm-mc's code byte for byte, the golden's instruction lines printed, read back to the same bytes, and it fills in the interpreter", GcnAssemblerFillVirtual),
        ("gcn assembler: allocation of a straight-line kernel and a branching one (a bounds check, a nested if restored by s_or_b64, a loop): registers within their counts, the printed text reads back to the same bytes, and both run in the interpreter as the CPU computes them", GcnAssemblerAllocation),
        ("gcn assembler: a vector register written under a narrowed EXEC keeps its other lanes: where a value reaches the write it is live from before (a value set after it on the path there, laid out later, takes another register, and the lanes left out keep theirs in the interpreter); where none reaches, the write starts it", GcnAssemblerPartialDefinition),
        ("gcn assembler: ranges are placed as the hardware takes them (scalar pairs even, quads and more 4-aligned, vector ranges adjacent, a register named alone is its range's, overlapping ranges one run); impossible alignments, registers read but never written, physical registers holding no system value or written, misplaced directives and flows that run off the end are refused with line and column", GcnAssemblerRanges),
        ("gcn assembler: running out of registers fails naming the kernel and the count (65 vector registers live in a block of 1024, where 64 fit; 103 scalar ones, where 102 fit; 40 at 8 waves a SIMD, where 32 fit), with no spilling; the same kernels fit a block of 256 and 102 scalar registers", GcnAssemblerBudget),
        ("gcn assembler: waits, each proved in the interpreter (the kernel runs right; each wait taken out, the interpreter reports the missing wait or the race): vmcnt in order (vmcnt(1) for the first of two loads), SMEM's lgkmcnt(0), LDS in order (lgkmcnt(1)), FLAT on both counters, lgkmcnt(0) before s_barrier (without it the other wave reads an unretired LDS store), a block's start waiting for a predecessor's load and a loop's back edge; a written wait kept and counted; GDS's expcnt(0) before its data is overwritten", GcnAssemblerWaits),
        ("gcn assembler: gfx9's hazards get the s_nop counts LLVM 18 inserts (its post-RA hazard recognizer): VMEM after a VALU's SGPR or EXEC 4, readlane and writelane lane selects 3, v_div_fmas after VCC 3, DPP after a VGPR 1 and after EXEC 4, s_getreg and s_setreg after s_setreg 1, a wide store's data 0, GDS after M0 0, soft clauses 0 with xnack; a wait counts as a wait state and the nearest write on any path counts; the cases LLVM leaves alone, and the clauses with xnack off, get none", GcnAssemblerHazards),
        ("gcn assembler: the printer: every forms.golden record of each registered CPU (gfx900, gfx906, gfx908, gfx90c), printed as Assemble encoded it, reads back to the same bytes", GcnAssemblerPrinter),
        ("gcn assembler: a kernel with virtual registers allocates nothing per instruction once warm: four times the instructions allocate only the longer code", GcnAssemblerKernelAllocations),
        ("gcn gate: Idrak's fill_f32, written through the encoder, is the golden's code byte for byte, and its code object for gfx90c (formats 5 and 4) equals the golden's where the loader reads: the ELF header and flags, the code, the metadata note, the kernel descriptor (its code's offset aside), the kernel's symbols and segments", GcnGateGolden),
        ("gcn gate: on a HIP device, the graphics driver's runtime alone loads Idrak's own code object for the target the device reports, and fill_f32 fills 1,000,003 floats with 2.5, the memory past them untouched; a refused object fails with the runtime's reason, never a skip (on HIP devices)", GcnGateDevice),
    ];

    // The golden fill_f32 (tests/Idrak.Tests/data/gcn/kernels/fill_f32.s: out[i] = value for i < count, blocks of 256)
    // as llvm-mc encodes it for gfx90c. Its kernel arguments' address arrives in s[0:1] (the only user SGPRs), the
    // workgroup id x in s2; it uses s0-s7 and VCC.
    private static readonly uint[] FillWords =
        [0xC00A0100, 0x00000000, 0x8E028802, 0x68000002, 0xBF8CC07F, 0x7D980007, 0xBE80206A, 0xBF880004, 0x24000082, 0x7E020206, 0xDC708000, 0x00040100, 0xBF810000];

    private static GcnKernelCode FillKernel(string name = "fill_f32") => new()
    {
        Name = name,
        Code = Words(FillWords),
        Vgprs = 2,
        Sgprs = 8,
        UsesVcc = true,
        Arguments = [GcnArgument.Buffer(0, "out"), GcnArgument.Value(8, 4, "value"), GcnArgument.Value(12, 4, "count")],
        RequiredWorkgroupSize = (256, 1, 1),
    };

    private static string GoldenPath(string file) => RepositoryPath("tests", "Idrak.Tests", "data", "gcn", "kernels", file);

    private static byte[] Words(ReadOnlySpan<uint> words)
    {
        var bytes = new byte[words.Length * 4];
        for (int i = 0; i < words.Length; i++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4), words[i]);
        }

        return bytes;
    }

    internal static void GcnTargetStrings(Device device)
    {
        _ = device;
        var xnackOff = GcnIsaVersion.Parse("gfx90c:xnack-");
        Check(xnackOff == new GcnIsaVersion(9, 0, 12, GcnFeatureSetting.Off), $"gfx90c:xnack- is 9.0.12 with xnack off: {xnackOff.Major}.{xnackOff.Minor}.{xnackOff.Stepping} {xnackOff.Xnack}");
        Check(xnackOff.ToString() == "gfx90c:xnack-" && xnackOff.Processor == "gfx90c", $"written back as reported: {xnackOff}");
        var both = GcnIsaVersion.Parse("gfx942:sramecc+:xnack-");
        Check(both == new GcnIsaVersion(9, 4, 2, GcnFeatureSetting.Off, GcnFeatureSetting.On) && both.ToString() == "gfx942:sramecc+:xnack-", $"gfx942:sramecc+:xnack-: {both}");
        Check(GcnIsaVersion.Parse("gfx1100") == new GcnIsaVersion(11, 0, 0) && GcnIsaVersion.Parse("gfx1030") == new GcnIsaVersion(10, 3, 0), "gfx1100 is 11.0.0, gfx1030 10.3.0");
        Check(GcnIsaVersion.Parse("gfx90a:xnack+:sramecc-").ToString() == "gfx90a:sramecc-:xnack+", "features in either order, written in LLVM's (sramecc first)");
        Check(GcnIsaVersion.Parse("gfx906:sramecc+") is { Major: 9, Minor: 0, Stepping: 6, Sramecc: GcnFeatureSetting.On, Xnack: GcnFeatureSetting.Any }, "a feature left out is any");

        foreach (string bad in (string[])["", "gfx", "gfx9", "gfx90", "gfx9-generic", "gfx10-3-generic", "gfx90C", "sm_90", "gfx90c:", "gfx90c::xnack-", "gfx90c:xnack", "gfx90c:xnack-:xnack+",
                     "gfx90c:tgsplit+", "gfx90c:XNACK-", " gfx90c", "gfx0900"])
        {
            Check(!GcnIsaVersion.TryParse(bad, out _), $"'{bad}' is refused");
        }

        ThrowsNaming<FormatException>(() => GcnIsaVersion.Parse("gfx9-generic"), "a generic target", "'gfx9-generic'");

        // From the span: nothing allocated per part (warm, so the tiered JIT's first code is not what is counted).
        ReadOnlySpan<char> reported = "gfx942:sramecc+:xnack-";
        for (int i = 0; i < 64; i++)
        {
            _ = GcnIsaVersion.TryParse(reported, out _);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            _ = GcnIsaVersion.TryParse(reported, out _);
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0, $"parsing allocated {allocated} bytes for 100 target strings");
    }

    internal static void GcnTargetFamilies(Device device)
    {
        _ = device;
        Check(GcnTargets.Names.Contains("gfx9") && GcnTargets.Default("gfx9") is { Family: "gfx9" } && GcnTargets.Origin("gfx9") == Overrides.Library,
            $"gfx9 is the library's default: {string.Join(", ", GcnTargets.Names)}, from {GcnTargets.Origin("gfx9")}");
        foreach (string handled in (string[])["gfx90c:xnack-", "gfx90c", "gfx900", "gfx902:xnack+", "gfx904", "gfx906:sramecc+:xnack-", "gfx908", "gfx909"])
        {
            Check(GcnTargets.For(handled).Family == "gfx9", $"{handled} is gfx9's");
        }

        // 9.0.10 and 9.4.x (one file of vector and accumulation registers) and every later version are other families'.
        foreach (string other in (string[])["gfx90a:xnack-", "gfx942:sramecc+:xnack-", "gfx1030", "gfx1100", "gfx1201"])
        {
            Check(!GcnTargets.TryFor(GcnIsaVersion.Parse(other), out _), $"{other} is not gfx9's");
            ThrowsNaming<NotRegisteredException>(() => GcnTargets.For(other), $"{other} on its own", other, "not registered", "gfx9");
        }

        Check(!GcnTargets.TryFor(new GcnIsaVersion(9, 0, 5), out _), "a 9.0.x version with no known processor is not handled");
        try
        {
            _ = GcnTargets.For("gfx1030");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == "GcnTargets" && e.Name == "gfx1030" && e.Registered.Contains("gfx9"), $"the refusal names the registry, the target and the families: {e.Registry}, {e.Name}");
        }

        // An app's family is asked before the library's, and leaves when unregistered.
        var app = new AppFamily();
        GcnTargets.Register(app);
        try
        {
            Check(ReferenceEquals(GcnTargets.For("gfx1030"), app) && GcnTargets.Names[0] == app.Family, $"the app's family answers for 10.3.x, asked first: {string.Join(", ", GcnTargets.Names)}");
            Check(GcnTargets.Origin(app.Family) != Overrides.Library && GcnTargets.For("gfx90c:xnack-").Family == "gfx9", "the app's family is the app's; gfx9 still answers for 9.0.12");
        }
        finally
        {
            GcnTargets.Unregister(app.Family);
        }

        Check(!GcnTargets.Names.Contains(app.Family) && !GcnTargets.TryFor(GcnIsaVersion.Parse("gfx1030"), out _), "unregistered, 10.3.x is nobody's again");
    }

    // A family an app registers (a stand-in: it answers for 10.3.x and encodes nothing).
    private sealed class AppFamily : IGcnTarget
    {
        public string Family => "app-family-test";

        public IReadOnlyList<int> WaveSizes { get; } = [32, 64];

        public uint PaddingWord => 0;

        public bool Handles(GcnIsaVersion isa) => isa is { Major: 10, Minor: 3 };

        public GcnRegisterFile Registers(GcnIsaVersion isa, int waveSize) => throw new NotSupportedException();

        public int SgprCount(GcnIsaVersion isa, int sgprs, bool vcc, bool flatScratch) => sgprs;

        public int WavesPerSimd(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize) => 1;

        public GcnDescriptorBits Descriptor(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize) => default;

        public uint ElfFlags(GcnIsaVersion isa, int codeObjectVersion) => 0;
    }

    internal static void GcnTargetElfFlags(Device device)
    {
        _ = device;

        // llvm-readelf -h's flags for `.amdgcn_target "amdgcn-amd-amdhsa--<processor>"` assembled by llvm-mc 18.1.3.
        const string Llvm = "gfx600 0x20 gfx601 0x21 gfx602 0x3A gfx700 0x22 gfx701 0x23 gfx702 0x24 gfx703 0x25 gfx704 0x26 gfx705 0x3B gfx801 0x128 gfx802 0x29 gfx803 0x2A "
                            + "gfx805 0x3C gfx810 0x12B gfx900 0x12C gfx902 0x12D gfx904 0x12E gfx906 0x52F gfx908 0x530 gfx909 0x131 gfx90a 0x53F gfx90c 0x132 gfx940 0x540 "
                            + "gfx941 0x54B gfx942 0x54C gfx1010 0x133 gfx1011 0x134 gfx1012 0x135 gfx1013 0x142 gfx1030 0x36 gfx1031 0x37 gfx1032 0x38 gfx1033 0x39 "
                            + "gfx1034 0x3E gfx1035 0x3D gfx1036 0x45 gfx1100 0x41 gfx1101 0x46 gfx1102 0x47 gfx1103 0x44 gfx1150 0x43 gfx1151 0x4A gfx1200 0x48 gfx1201 0x4E";
        string[] parts = Llvm.Split(' ');
        var processors = GcnTargets.Processors.ToList();
        Check(processors.Count == parts.Length / 2, $"every processor LLVM 18 assembles is known: {processors.Count} of {parts.Length / 2}");
        for (int i = 0; i < parts.Length; i += 2)
        {
            uint expected = uint.Parse(parts[i + 1].AsSpan(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            uint flags = GcnTargets.ElfFlagsOf(GcnIsaVersion.Parse(parts[i]), 5);
            Check(flags == expected && GcnTargets.ElfFlagsOf(GcnIsaVersion.Parse(parts[i]), 4) == expected, $"{parts[i]}: 0x{flags:X} where LLVM writes 0x{expected:X}");
        }

        // With features (llvm-mc -mattr=±xnack,±sramecc), the same in formats 4 and 5.
        foreach (var (target, expected) in (ReadOnlySpan<(string, uint)>)[("gfx90c:xnack-", 0x232), ("gfx90c:xnack+", 0x332), ("gfx942:sramecc+:xnack-", 0xE4C),
                     ("gfx942:sramecc-:xnack+", 0xB4C), ("gfx906:sramecc+", 0xD2F), ("gfx1010:xnack-", 0x233)])
        {
            uint flags = GcnTargets.ElfFlagsOf(GcnIsaVersion.Parse(target), 4);
            Check(flags == expected && GcnTargets.ElfFlagsOf(GcnIsaVersion.Parse(target), 5) == expected, $"{target}: 0x{flags:X} where LLVM writes 0x{expected:X}");
        }

        var gfx90c = GcnIsaVersion.Parse("gfx90c:xnack-");
        Check(GcnTargets.For(gfx90c).ElfFlags(gfx90c, 5) == 0x232, "gfx9's flags are the shared ones");
        ThrowsNaming<ArgumentException>(() => GcnTargets.ElfFlagsOf(GcnIsaVersion.Parse("gfx1030:xnack+"), 5), "xnack on a processor without it", "gfx1030:xnack+", "xnack");
        ThrowsNaming<ArgumentException>(() => GcnTargets.ElfFlagsOf(GcnIsaVersion.Parse("gfx900:sramecc-"), 5), "sramecc on a processor without it", "sramecc");
        ThrowsNaming<ArgumentException>(() => GcnTargets.ElfFlagsOf(new GcnIsaVersion(9, 0, 5), 5), "an unknown processor", "gfx905");
        ThrowsNaming<NotSupportedException>(() => GcnTargets.ElfFlagsOf(gfx90c, 6), "format 6", "format 6");
    }

    internal static void GcnTargetRegisters(Device device)
    {
        _ = device;
        var off = GcnIsaVersion.Parse("gfx90c:xnack-");
        var on = GcnIsaVersion.Parse("gfx90c:xnack+");
        var any = GcnIsaVersion.Parse("gfx90c");
        var gfx9 = GcnTargets.For(off);
        Check(gfx9.WaveSizes.SequenceEqual([64]) && gfx9.PaddingWord == 0xBF800000, "wave64, padded with s_nop 0");
        Check(gfx9.Registers(off, 64) == new GcnRegisterFile(Vgprs: 256, VgprGranule: 4, Sgprs: 102, SgprGranule: 8, MaxWavesPerSimd: 10), $"the register files: {gfx9.Registers(off, 64)}");
        ThrowsNaming<ArgumentOutOfRangeException>(() => gfx9.Registers(off, 32), "wave32 on gfx9", "64 lanes");

        // VCC, the xnack mask and flat scratch at the top of the scalar registers (llvm-mc's .amdhsa_reserve_* blocks).
        Check(gfx9.SgprCount(off, 8, vcc: false, flatScratch: false) == 8 && gfx9.SgprCount(off, 8, vcc: true, flatScratch: false) == 10
              && gfx9.SgprCount(off, 8, vcc: false, flatScratch: true) == 14 && gfx9.SgprCount(off, 8, vcc: true, flatScratch: true) == 14, "xnack off: VCC 2, flat scratch 6");
        Check(gfx9.SgprCount(on, 3, false, false) == 7 && gfx9.SgprCount(any, 5, true, false) == 9 && gfx9.SgprCount(on, 1, false, true) == 7, "xnack on or any: 4, flat scratch 6");

        // COMPUTE_PGM_RSRC1's blocks as llvm-mc wrote them for .amdhsa_next_free_vgpr and _sgpr (DX10_CLAMP and IEEE_MODE on).
        foreach (var (vgprs, sgprs, blocks) in (ReadOnlySpan<(int, int, uint)>)[(0, 0, 0x000u), (1, 1, 0x000), (5, 8, 0x001), (9, 9, 0x042), (256, 102, 0x33F), (2, 108, 0x340)])
        {
            var bits = gfx9.Descriptor(off, vgprs, sgprs, 64);
            Check(bits == new GcnDescriptorBits(0x00A00000 | blocks, 0, 0), $"{vgprs} VGPRs and {sgprs} SGPRs: RSRC1 0x{bits.Rsrc1:X8}, expected 0x{0x00A00000 | blocks:X8}");
        }

        ThrowsNaming<ArgumentOutOfRangeException>(() => gfx9.Descriptor(off, 257, 8, 64), "257 VGPRs", "vgprs");
        ThrowsNaming<ArgumentOutOfRangeException>(() => gfx9.Descriptor(off, 8, 109, 64), "109 SGPRs", "sgprs");

        // Waves per SIMD as clang reported them (Occupancy) for those VGPR and SGPR counts.
        foreach (var (vgprs, sgprs, waves) in (ReadOnlySpan<(int, int, int)>)[(4, 0, 10), (25, 0, 9), (28, 0, 9), (29, 0, 8), (36, 0, 7), (64, 0, 4), (65, 0, 3), (86, 0, 2),
                     (128, 0, 2), (129, 0, 1), (256, 0, 1), (0, 80, 10), (0, 81, 9), (0, 88, 9), (0, 89, 8), (0, 100, 8), (0, 102, 7), (36, 89, 7)])
        {
            int found = gfx9.WavesPerSimd(off, vgprs, sgprs, 64);
            Check(found == waves, $"{vgprs} VGPRs, {sgprs} SGPRs: {found} waves a SIMD, LLVM says {waves}");
        }
    }

    internal static void GcnCodeObjectFill(Device device)
    {
        _ = device;
        var isa = GcnIsaVersion.Parse("gfx90c");
        var kernel = FillKernel();
        byte[] image = GcnCodeObject.Write(isa, 5, kernel);
        var elf = new ElfReader(image);

        // The header: ELF64, little-endian, OS ABI AMDGPU-HSA, ABI version 3 (format 5), a shared object for EM_AMDGPU.
        Check(image.AsSpan(0, 9).SequenceEqual((byte[])[0x7F, (byte)'E', (byte)'L', (byte)'F', 2, 1, 1, 64, 3]) && image.AsSpan(9, 7).IndexOfAnyExcept((byte)0) < 0,
            $"e_ident: {Convert.ToHexString(image.AsSpan(0, 16))}");
        Check(elf.U16(16) == 3 && elf.U16(18) == 224 && elf.U32(20) == 1 && elf.U64(24) == 0 && elf.U64(32) == 64, "type DYN, machine EM_AMDGPU, version 1, no entry, program headers after the header");
        Check(elf.U32(48) == 0x132, $"e_flags 0x{elf.U32(48):X}: gfx90c (0x32) with xnack any (0x100)");
        Check(elf.U16(52) == 64 && elf.U16(54) == 56 && elf.U16(56) == 8 && elf.U16(58) == 64 && elf.U16(60) == 11 && elf.Sections[elf.U16(62)].Name == ".shstrtab", "header sizes and counts");
        Check(image.Length == (int)elf.U64(40) + 11 * 64, $"the section headers end the file ({image.Length} bytes)");

        // Sections, as ld.lld lays them out (without .gnu.hash, .relro_padding and .comment).
        string names = string.Join(" ", elf.Sections.Select(s => s.Name));
        Check(names == " .note .dynsym .hash .dynstr .rodata .text .dynamic .symtab .shstrtab .strtab", $"sections: {names}");
        var text = elf.Section(".text");
        var rodata = elf.Section(".rodata");
        var dynamicSection = elf.Section(".dynamic");
        Check(text is { Type: 1, Flags: 6, Align: 256 } && text.Address % 256 == 0 && text.Size == 52 && image.AsSpan((int)text.Offset, 52).SequenceEqual(kernel.Code.Span), "the code, 256-byte aligned, read and execute");
        Check(rodata is { Type: 1, Flags: 2, Align: 64, Size: 64 } && rodata.Address % 64 == 0, "one 64-byte descriptor, 64-byte aligned");
        Check(elf.Section(".dynsym") is { Type: 11, EntrySize: 24, Info: 1 } dynsymSection && elf.Sections[(int)dynsymSection.Link].Name == ".dynstr", ".dynsym names its strings, one local symbol");
        Check(elf.Section(".hash") is { Type: 5, EntrySize: 4 } hashSection && elf.Sections[(int)hashSection.Link].Name == ".dynsym", ".hash names .dynsym");
        Check(dynamicSection is { Type: 6, Flags: 3, EntrySize: 16 } && elf.Sections[(int)dynamicSection.Link].Name == ".dynstr", ".dynamic: read and write, names .dynstr");
        Check(elf.Section(".symtab") is { Type: 2, EntrySize: 24, Info: 2 } symtabSection && elf.Sections[(int)symtabSection.Link].Name == ".strtab", ".symtab: _DYNAMIC local, then the globals");

        // Segments: the headers, three loads (read; read and execute; read and write), the dynamic table, RELRO, stack, note.
        var types = elf.Segments.Select(s => s.Type).ToArray();
        Check(types.SequenceEqual((uint[])[6, 1, 1, 1, 2, 0x6474E552, 0x6474E551, 4]), $"segment types: {string.Join(", ", types.Select(t => $"0x{t:X}"))}");
        var loads = elf.Segments.Where(s => s.Type == 1).ToArray();
        Check(loads[0] is { Flags: 4, Offset: 0, Address: 0 } && loads[1].Flags == 5 && loads[2].Flags == 6 && loads.All(l => l.Align == 0x1000 && l.Offset % 0x1000 == l.Address % 0x1000),
            "loads: read, read and execute, read and write; offsets agree with addresses modulo the page");
        Check(loads[0].FileSize == rodata.Offset + rodata.Size && loads[1].Offset == text.Offset && loads[1].Address == text.Address && loads[2].Address == dynamicSection.Address,
            "the first load ends with the descriptors, the second is the code, the third the dynamic table");
        foreach (var section in elf.Sections.Where(s => (s.Flags & 2) != 0))
        {
            Check(loads.Count(l => section.Address >= l.Address && section.Address + section.Size <= l.Address + l.MemorySize && section.Offset - l.Offset == section.Address - l.Address) == 1,
                $"{section.Name} lies in one load, at the same place in the file and in memory");
        }

        // Symbols: the code (a protected function) and its descriptor (an object), as ld.lld writes them; _DYNAMIC in the static table.
        var dynsym = elf.Symbols(".dynsym");
        Check(dynsym.Count == 3 && dynsym[0] == new ElfSymbol("", 0, 0, 0, 0, 0), "the null symbol first");
        Check(dynsym[1] == new ElfSymbol("fill_f32", 0x12, 3, (ushort)elf.IndexOf(".text"), text.Address, 52), $"the code symbol: {dynsym[1]}");
        Check(dynsym[2] == new ElfSymbol("fill_f32.kd", 0x11, 0, (ushort)elf.IndexOf(".rodata"), rodata.Address, 64), $"the descriptor symbol: {dynsym[2]}");
        var symtab = elf.Symbols(".symtab");
        Check(symtab.Count == 4 && symtab[1] == new ElfSymbol("_DYNAMIC", 0, 2, (ushort)elf.IndexOf(".dynamic"), dynamicSection.Address, 0) && symtab.Skip(2).SequenceEqual(dynsym.Skip(1)),
            "the static symbols: _DYNAMIC, then the dynamic ones");

        // The dynamic table and the hash table.
        var entries = elf.Dynamic();
        Check(entries.SequenceEqual([(6L, (long)elf.Section(".dynsym").Address), (11, 24), (5, (long)elf.Section(".dynstr").Address), (10, (long)elf.Section(".dynstr").Size),
            (4, (long)elf.Section(".hash").Address), (0, 0)]), $"dynamic: {string.Join(", ", entries)}");
        Check(elf.Lookup("fill_f32") == 1 && elf.Lookup("fill_f32.kd") == 2 && elf.Lookup("fill") < 0, "the hash table finds both names and not another");

        // The note's metadata, decoded: the keys in LLVM's order, the kernel as the golden's YAML gives it.
        var note = elf.Section(".note");
        Check(elf.U32((int)note.Offset) == 7 && elf.U32((int)note.Offset + 8) == 32 && image.AsSpan((int)note.Offset + 12, 8).SequenceEqual("AMDGPU\0\0"u8), "NT_AMDGPU_METADATA, named AMDGPU");
        int metadataBytes = (int)elf.U32((int)note.Offset + 4);
        int at = 0;
        var root = (List<KeyValuePair<string, object>>)Unpack(image.AsSpan((int)note.Offset + 20, metadataBytes), ref at);
        Check(at == metadataBytes, "the MessagePack fills the note");
        Check(string.Join(" ", root.Select(e => e.Key)) == "amdhsa.kernels amdhsa.target amdhsa.version", "the top keys in LLVM's order");
        Check((string)Value(root, "amdhsa.target") == $"{GcnTargets.Triple}--gfx90c" && ((List<object>)Value(root, "amdhsa.version")).SequenceEqual([1UL, 2UL]), "the target id and version 1.2");
        var fill = (List<KeyValuePair<string, object>>)((List<object>)Value(root, "amdhsa.kernels"))[0];
        Check(string.Join(" ", fill.Select(e => e.Key)) == ".args .group_segment_fixed_size .kernarg_segment_align .kernarg_segment_size .max_flat_workgroup_size .name "
              + ".private_segment_fixed_size .reqd_workgroup_size .sgpr_count .sgpr_spill_count .symbol .vgpr_count .vgpr_spill_count .wavefront_size", "the kernel's keys, sorted as LLVM writes them");
        Check((string)Value(fill, ".name") == "fill_f32" && (string)Value(fill, ".symbol") == "fill_f32.kd" && (ulong)Value(fill, ".kernarg_segment_size") == 16
              && (ulong)Value(fill, ".sgpr_count") == 12 && (ulong)Value(fill, ".vgpr_count") == 2 && (ulong)Value(fill, ".wavefront_size") == 64
              && (ulong)Value(fill, ".max_flat_workgroup_size") == 256 && ((List<object>)Value(fill, ".reqd_workgroup_size")).SequenceEqual([256UL, 1UL, 1UL]),
            "name, symbol, arguments' size, registers (8, and 4 for VCC and the xnack mask), wave and workgroup sizes");
        var args = ((List<object>)Value(fill, ".args")).Cast<List<KeyValuePair<string, object>>>().ToList();
        Check(args.Count == 3 && (string)Value(args[0], ".value_kind") == "global_buffer" && (string)Value(args[0], ".address_space") == "global" && (string)Value(args[0], ".name") == "out"
              && (ulong)Value(args[2], ".offset") == 12 && (string)Value(args[2], ".value_kind") == "by_value" && args[2].All(e => e.Key != ".address_space"), "a global buffer, then two values");

        // The golden, ld.lld's objects for the same source in formats 5 and 4: the same flags, code, metadata, descriptor
        // and symbols (ld.lld adds .gnu.hash, .relro_padding and .comment, which move the addresses).
        foreach (int format in (int[])[5, 4])
        {
            CheckAgainstGolden(format == 5 ? image : GcnCodeObject.Write(isa, 4, kernel), format);
        }
    }

    // A gfx90c code object of fill_f32 against the golden of its format (ld.lld's), where the loader reads it: the same
    // flags, code, metadata, descriptor and symbols (ld.lld adds .gnu.hash, .relro_padding and .comment, which move the
    // addresses, so the descriptor's offset to its code is checked to reach the code instead).
    private static void CheckAgainstGolden(byte[] mine, int format)
    {
        var ours = new ElfReader(mine);
        var golden = new ElfReader(File.ReadAllBytes(GoldenPath($"fill_f32.gfx90c.v{format}.co")));
        string what = $"format {format} against fill_f32.gfx90c.v{format}.co";
        Check(mine.AsSpan(0, 16).SequenceEqual(golden.Bytes(0, 16)) && ours.U32(48) == golden.U32(48) && ours.U16(18) == golden.U16(18) && ours.U16(16) == golden.U16(16),
            $"{what}: e_ident, type, machine and flags");
        Check(ours.SectionBytes(".text").SequenceEqual(golden.SectionBytes(".text")), $"{what}: the code");
        Check(ours.SectionBytes(".note").SequenceEqual(golden.SectionBytes(".note")),
            $"{what}: the note, byte for byte: {Convert.ToHexStringLower(ours.SectionBytes(".note"))} against {Convert.ToHexStringLower(golden.SectionBytes(".note"))}");
        byte[] descriptor = ours.SectionBytes(".rodata"), expected = golden.SectionBytes(".rodata");
        Check(BinaryPrimitives.ReadInt64LittleEndian(descriptor.AsSpan(16)) == (long)ours.Section(".text").Address - (long)ours.Section(".rodata").Address
              && BinaryPrimitives.ReadInt64LittleEndian(expected.AsSpan(16)) == (long)golden.Section(".text").Address - (long)golden.Section(".rodata").Address,
            $"{what}: each entry offset reaches its code");
        descriptor.AsSpan(16, 8).Clear();
        expected.AsSpan(16, 8).Clear();
        Check(descriptor.SequenceEqual(expected), $"{what}: the descriptor, the code's offset aside: {Convert.ToHexStringLower(descriptor)} against {Convert.ToHexStringLower(expected)}");
        Check(ours.Symbols(".dynsym").Select(s => (s.Name, s.Info, s.Other, s.Size)).SequenceEqual(golden.Symbols(".dynsym").Select(s => (s.Name, s.Info, s.Other, s.Size))),
            $"{what}: the dynamic symbols' names, kinds, visibility and sizes");
        Check(ours.Segments.Select(s => (s.Type, s.Flags, s.Align)).SequenceEqual(golden.Segments.Select(s => (s.Type, s.Flags, s.Align))), $"{what}: the segments' kinds");
        Check(ours.Dynamic().Select(e => e.Tag).SequenceEqual(golden.Dynamic().Select(e => e.Tag).Where(t => t != 0x6FFFFEF5)), $"{what}: the dynamic tags (without GNU_HASH)");
    }

    internal static void GcnCodeObjectKernels(Device device)
    {
        _ = device;
        var isa = GcnIsaVersion.Parse("gfx90c:xnack-");
        string longName = "a_kernel_whose_name_is_longer_than_thirty_one_bytes";
        GcnKernelCode[] kernels =
        [
            new() { Name = "end", Code = Words([0xBF810000]), Vgprs = 1, Sgprs = 1 },
            FillKernel(),
            FillKernel(longName) with { LdsBytes = 1024, MaxFlatWorkgroupSize = 1024, WorkgroupIds = 3, WorkitemIds = 3, RequiredWorkgroupSize = null },
        ];
        byte[] image = GcnCodeObject.Write(isa, 5, kernels);
        var elf = new ElfReader(image);
        var text = elf.Section(".text");
        var rodata = elf.Section(".rodata");
        var dynsym = elf.Symbols(".dynsym");
        Check(dynsym.Count == 7 && dynsym.Skip(1).Select(s => s.Name).SequenceEqual(["end", "fill_f32", longName, "end.kd", "fill_f32.kd", longName + ".kd"]),
            $"code symbols, then descriptor symbols: {string.Join(", ", dynsym.Select(s => s.Name))}");
        for (int i = 0; i < kernels.Length; i++)
        {
            var code = dynsym[1 + i];
            var kd = dynsym[4 + i];
            Check(code.Value % 256 == 0 && code.Size == (ulong)kernels[i].Code.Length, $"{code.Name}: 256-byte aligned");
            int at = (int)(text.Offset + (code.Value - text.Address));
            Check(image.AsSpan(at, kernels[i].Code.Length).SequenceEqual(kernels[i].Code.Span), $"{code.Name}: its code");
            Check(kd.Value == rodata.Address + (ulong)(64 * i) && kd.Size == 64, $"{kd.Name}: the {i}th descriptor");
            long entry = BinaryPrimitives.ReadInt64LittleEndian(image.AsSpan((int)(rodata.Offset + (ulong)(64 * i) + 16)));
            Check(entry == (long)code.Value - (long)kd.Value, $"{kd.Name}: its entry offset reaches {code.Name}");
            Check(elf.Lookup(code.Name) == 1 + i && elf.Lookup(kd.Name) == 4 + i, $"the hash table finds {code.Name} and {kd.Name}");
        }

        // Between kernels: s_nop 0 up to the next 256-byte boundary.
        for (ulong address = dynsym[1].Value + 4; address < dynsym[2].Value; address += 4)
        {
            Check(elf.U32((int)(text.Offset + (address - text.Address))) == 0xBF800000, $"padding at 0x{address:X} is s_nop 0");
        }

        // The third kernel's descriptor: LDS, workgroup ids x, y, z, work-item ids up to z, VCC's registers counted.
        var third = image.AsSpan((int)rodata.Offset + 128, 64);
        uint rsrc2 = BinaryPrimitives.ReadUInt32LittleEndian(third[52..]);
        Check(BinaryPrimitives.ReadUInt32LittleEndian(third) == 1024 && rsrc2 == (0x04u | (7u << 7) | (2u << 11)), $"LDS 1024, RSRC2 0x{rsrc2:X}");
        Check((BinaryPrimitives.ReadUInt32LittleEndian(third[48..]) >> 6 & 0xF) == 1, "8 + VCC's 2 SGPRs: two blocks");

        int offset = 0;
        var note = elf.Section(".note");
        var root = (List<KeyValuePair<string, object>>)Unpack(image.AsSpan((int)note.Offset + 20, (int)elf.U32((int)note.Offset + 4)), ref offset);
        var listed = ((List<object>)Value(root, "amdhsa.kernels")).Cast<List<KeyValuePair<string, object>>>().ToList();
        Check(listed.Select(k => (string)Value(k, ".name")).SequenceEqual(["end", "fill_f32", longName]) && (string)Value(listed[2], ".symbol") == longName + ".kd",
            "the metadata lists the kernels in order, a long name whole");
        Check(listed[0].All(e => e.Key != ".args") && (ulong)Value(listed[0], ".kernarg_segment_size") == 0, "a kernel without arguments has no .args");
        Check((ulong)Value(listed[2], ".group_segment_fixed_size") == 1024 && (ulong)Value(listed[2], ".max_flat_workgroup_size") == 1024 && (ulong)Value(listed[2], ".sgpr_count") == 10
              && listed[2].All(e => e.Key != ".reqd_workgroup_size"), "LDS, workgroup size, the SGPRs with VCC's (xnack off), no required shape");
    }

    internal static void GcnCodeObjectFormats(Device device)
    {
        _ = device;
        var isa = GcnIsaVersion.Parse("gfx90c:xnack-");
        byte[] five = GcnCodeObject.Write(isa, 5, FillKernel());
        byte[] four = GcnCodeObject.Write(isa, 4, FillKernel());
        var differ = Enumerable.Range(0, five.Length).Where(i => five[i] != four[i]).ToList();
        var reader = new ElfReader(four);
        var note = reader.Section(".note");
        int lastMetadataByte = (int)note.Offset + 20 + (int)reader.U32((int)note.Offset + 4) - 1;
        Check(four.Length == five.Length && differ.Count == 2 && differ[0] == 8 && four[8] == 2 && five[8] == 3, $"the ABI version (2 for format 4, 3 for 5): bytes {string.Join(", ", differ)} differ");
        Check(differ.Count == 2 && differ[1] == lastMetadataByte && four[differ[1]] == 1 && five[differ[1]] == 2, "amdhsa.version 1.1 for format 4, 1.2 for 5");
        Check(GcnCodeObject.Write(isa, 5, FillKernel()).AsSpan().SequenceEqual(five), "the same kernel gives the same bytes");

        ThrowsNaming<NotSupportedException>(() => GcnCodeObject.Write(isa, 6, FillKernel()), "format 6", "format 6", "IDRAK_HIP_CODE_OBJECT");
        ThrowsNaming<NotSupportedException>(() => GcnCodeObject.Write(isa, 3, FillKernel()), "format 3", "format 3");
        ThrowsNaming<NotRegisteredException>(() => GcnCodeObject.Write(GcnIsaVersion.Parse("gfx1030"), 5, FillKernel()), "a target no family handles", "gfx1030", "not registered");
        var gfx9 = GcnTargets.For(isa);
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(gfx9, GcnIsaVersion.Parse("gfx1030"), 5, [FillKernel()]), "gfx9 asked for 10.3.0", "gfx9", "gfx1030");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5), "no kernels", "at least one kernel");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel(), FillKernel()), "two kernels of one name", "'fill_f32'");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { Code = new byte[6] }), "6 bytes of code", "whole dwords");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { WaveSize = 32 }), "wave32 on gfx9", "waves of 32");
        ThrowsNaming<ArgumentOutOfRangeException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { Vgprs = 257 }), "257 VGPRs", "vgprs");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { Name = "" }), "no name", "name");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { Arguments = [GcnArgument.Buffer(0), GcnArgument.Value(4, 4)] }), "overlapping arguments", "at 4");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { Arguments = [GcnArgument.Value(2, 4)] }), "a misaligned argument", "aligned");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { KernargBytes = 8 }), "arguments past the size given", "end at 16");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { WorkitemIds = 4 }), "four work-item ids", "out of its field's range");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.Write(isa, 5, FillKernel() with { RequiredWorkgroupSize = (512, 1, 1) }), "a shape past the most", "out of its field's range");

        // IDRAK_HIP_CODE_OBJECT: unset or auto, every format newest first; a version, that one; 6 and others refused, named.
        Check(GcnCodeObject.FormatsFor(null).SequenceEqual([5, 4]) && GcnCodeObject.FormatsFor("").SequenceEqual([5, 4]) && GcnCodeObject.FormatsFor("auto").SequenceEqual([5, 4])
              && GcnCodeObject.FormatsFor("default").SequenceEqual([5, 4]), "unset, auto or default: 5, then 4");
        Check(GcnCodeObject.FormatsFor(" 4 ").SequenceEqual([4]) && GcnCodeObject.FormatsFor("5").SequenceEqual([5]), "a version alone");
        ThrowsNaming<NotSupportedException>(() => GcnCodeObject.FormatsFor("6"), "IDRAK_HIP_CODE_OBJECT=6", "format 6", "5 or 4");
        ThrowsNaming<ArgumentException>(() => GcnCodeObject.FormatsFor("newest"), "IDRAK_HIP_CODE_OBJECT=newest", "IDRAK_HIP_CODE_OBJECT=newest");
        string? saved = Environment.GetEnvironmentVariable("IDRAK_HIP_CODE_OBJECT");
        try
        {
            Environment.SetEnvironmentVariable("IDRAK_HIP_CODE_OBJECT", "4");
            Check(GcnCodeObject.FormatsFromEnvironment().SequenceEqual([4]), "the environment's IDRAK_HIP_CODE_OBJECT=4 is read");
        }
        finally
        {
            Environment.SetEnvironmentVariable("IDRAK_HIP_CODE_OBJECT", saved);
        }
    }

    internal static void GcnAssemblerForms(Device device)
    {
        _ = device;
        var gfx9 = GcnTargets.For(GcnIsaVersion.Parse("gfx90c"));

        // gfx90c's records of forms.golden: llvm-mc's text of each form and its bytes (a text seen twice has one encoding).
        var golden = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(RepositoryPath("tests", "Idrak.Tests", "data", "gcn", "gfx9", "forms.golden")))
        {
            string[] fields = line.Split('\t');
            if (fields is ["gfx90c", var text, { Length: > 0 } hex] && !golden.TryAdd(text, hex))
            {
                Check(golden[text] == hex, $"'{text}' has one encoding in the golden: {golden[text]} and {hex}");
            }
        }

        string Encoded(Action<GcnAssembler> write)
        {
            using var asm = new GcnAssembler(gfx9);
            write(asm);
            return Convert.ToHexStringLower(asm.ToArray());
        }

        // Each record, written through the typed API.
        (string Text, Action<GcnAssembler> Write)[] forms =
        [
            ("s_load_dwordx4 s[0:3], s[4:5], 0x0", a => a.Smem(GcnOp.SLoadDwordx4, GcnOperand.S(0, 4), GcnOperand.S(4, 2), 0)),
            ("s_load_dwordx4 s[8:11], s[4:5], 0x20", a => a.Smem(GcnOp.SLoadDwordx4, GcnOperand.S(8, 4), GcnOperand.S(4, 2), 0x20)),
            ("s_lshl_b32 s0, s1, 2", a => a.Sop2(GcnOp.SLshlB32, GcnOperand.S(0), GcnOperand.S(1), GcnOperand.Int(2))),
            ("s_lshl_b32 s0, s1, s2", a => a.Sop2(GcnOp.SLshlB32, GcnOperand.S(0), GcnOperand.S(1), GcnOperand.S(2))),
            ("s_and_saveexec_b64 s[0:1], vcc", a => a.Sop1(GcnOp.SAndSaveexecB64, GcnOperand.S(0, 2), GcnOperand.Vcc)),
            ("s_and_saveexec_b64 s[0:1], s[2:3]", a => a.Sop1(GcnOp.SAndSaveexecB64, GcnOperand.S(0, 2), GcnOperand.S(2, 2))),
            ("s_and_saveexec_b64 s[4:5], vcc", a => a.Sop1(GcnOp.SAndSaveexecB64, GcnOperand.S(4, 2), GcnOperand.Vcc)),
            ("s_waitcnt lgkmcnt(0)", a => a.Waitcnt(lgkmcnt: 0)),
            ("s_waitcnt lgkmcnt(1)", a => a.Waitcnt(lgkmcnt: 1)),
            ("s_waitcnt vmcnt(0)", a => a.Waitcnt(vmcnt: 0)),
            ("s_waitcnt vmcnt(15)", a => a.Waitcnt(vmcnt: 15)),
            ("s_waitcnt vmcnt(16)", a => a.Waitcnt(vmcnt: 16)),
            ("s_waitcnt expcnt(0)", a => a.Waitcnt(expcnt: 0)),
            ("s_waitcnt vmcnt(2) lgkmcnt(1)", a => a.Waitcnt(vmcnt: 2, lgkmcnt: 1)),
            ("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)", a => a.Waitcnt(0, 0, 0)),
            ("s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(15)", a => a.Waitcnt()),
            ("s_endpgm", a => a.Sopp(GcnOp.SEndpgm)),
            ("v_add_u32_e32 v0, v1, v2", a => a.Vop2(GcnOp.VAddU32, GcnOperand.V(0), GcnOperand.V(1), GcnOperand.V(2))),
            ("v_add_u32_e32 v0, s1, v2", a => a.Vop2(GcnOp.VAddU32, GcnOperand.V(0), GcnOperand.S(1), GcnOperand.V(2))),
            ("v_add_u32_e32 v0, 16, v2", a => a.Vop2(GcnOp.VAddU32, GcnOperand.V(0), GcnOperand.Int(16), GcnOperand.V(2))),
            ("v_cmp_gt_u32_e32 vcc, s1, v2", a => a.Vopc(GcnOp.VCmpGtU32, GcnOperand.S(1), GcnOperand.V(2))),
            ("v_cmp_gt_u32_e32 vcc, s2, v0", a => a.Vopc(GcnOp.VCmpGtU32, GcnOperand.S(2), GcnOperand.V(0))),
            ("v_lshlrev_b32_e32 v0, 2, v1", a => a.Vop2(GcnOp.VLshlrevB32, GcnOperand.V(0), GcnOperand.Int(2), GcnOperand.V(1))),
            ("v_lshlrev_b32_e32 v0, 4, v1", a => a.Vop2(GcnOp.VLshlrevB32, GcnOperand.V(0), GcnOperand.Int(4), GcnOperand.V(1))),
            ("v_lshlrev_b32_e32 v0, s1, v2", a => a.Vop2(GcnOp.VLshlrevB32, GcnOperand.V(0), GcnOperand.S(1), GcnOperand.V(2))),
            ("v_mov_b32_e32 v0, v1", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.V(1))),
            ("v_mov_b32_e32 v0, s1", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.S(1))),
            ("v_mov_b32_e32 v255, v0", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(255), GcnOperand.V(0))),
            ("v_mov_b32_e32 v0, 0", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Int(0))),
            ("v_mov_b32_e32 v0, -1", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Int(-1))),
            ("v_mov_b32_e32 v0, 64", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Int(64))),
            ("v_mov_b32_e32 v0, -16", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Int(-16))),
            ("v_mov_b32_e32 v0, 1.0", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Float(1f))),
            ("v_mov_b32_e32 v0, -2.0", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Float(-2f))),
            ("v_mov_b32_e32 v0, 0.5", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Float(0.5f))),
            ("v_mov_b32_e32 v0, 4.0", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Float(4f))),
            ("v_mov_b32_e32 v0, 0.15915494", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Float((float)(1 / (2 * Math.PI))))),
            ("v_mov_b32_e32 v0, 0x3fb8aa3b", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Bits(0x3FB8AA3B))),
            ("v_mov_b32_e32 v0, 0x7fffffff", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Int(int.MaxValue))),
            ("v_mov_b32_e32 v0, 0x80000000", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Bits(0x80000000))),
            ("v_mov_b32_e32 v0, 0xff800000", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.Float(float.NegativeInfinity))),
            ("v_mov_b32_e32 v0, vcc_lo", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.VccLo)),
            ("v_mov_b32_e32 v0, exec_lo", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.ExecLo)),
            ("v_mov_b32_e32 v0, m0", a => a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.M0)),
            ("global_store_dword v1, v0, s[2:3]", a => a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(1), GcnOperand.V(0), GcnOperand.S(2, 2))),
            ("global_store_dword v1, v0, s[2:3] offset:4", a => a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(1), GcnOperand.V(0), GcnOperand.S(2, 2), offset: 4)),
            ("global_store_dword v[2:3], v0, off", a => a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(2, 2), GcnOperand.V(0), GcnOperand.Off)),
            ("global_store_dword v[2:3], v0, off offset:-4", a => a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(2, 2), GcnOperand.V(0), GcnOperand.Off, offset: -4)),
            ("global_store_dword v[2:3], v0, off glc", a => a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(2, 2), GcnOperand.V(0), GcnOperand.Off, glc: true)),
        ];
        foreach (var (text, write) in forms)
        {
            Check(golden.TryGetValue(text, out string? expected), $"'{text}' is a gfx90c record of forms.golden");
            string encoded = Encoded(write);
            Check(encoded == expected, $"{text}: {encoded}, llvm-mc writes {expected}");
        }

        // Branches: the 16-bit field counts dwords from the next instruction (forms.golden's s_cbranch_execz loop_end is 4
        // dwords before its label; a backward branch over two instructions and itself is -3).
        string forward = Encoded(a =>
        {
            var end = a.NewLabel();
            a.Branch(GcnOp.SCbranchExecz, end);
            for (int i = 0; i < 4; i++)
            {
                a.Sopp(GcnOp.SEndpgm);
            }

            a.Bind(end);
        });
        Check(forward[..8] == golden["s_cbranch_execz loop_end"], $"a branch 4 dwords on: {forward[..8]}, llvm-mc writes {golden["s_cbranch_execz loop_end"]}");
        string backward = Encoded(a =>
        {
            var top = a.NewLabel();
            a.Bind(top);
            a.Waitcnt(lgkmcnt: 0);
            a.Vop1(GcnOp.VMovB32, GcnOperand.V(0), GcnOperand.S(1));
            a.Branch(GcnOp.SCbranchExecz, top);
        });
        Check(backward[16..] == "fdff88bf", $"a branch 3 dwords back: {backward[16..]}");

        // Refusals, each naming what it refused.
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Vop2(GcnOp.VAddU32, GcnOperand.V(0), GcnOperand.V(1), GcnOperand.S(1))), "a scalar second source of a VOP2", "s1", "v_add_u32");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Sop2(GcnOp.SLshlB32, GcnOperand.S(0), GcnOperand.V(1), GcnOperand.S(2))), "a vector source of a scalar instruction", "v1");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Smem(GcnOp.SLoadDwordx4, GcnOperand.S(1, 4), GcnOperand.S(4, 2), 0)), "a misaligned quad", "s[1:4]");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Smem(GcnOp.SLoadDwordx4, GcnOperand.S(0, 2), GcnOperand.S(4, 2), 0)), "two registers for four dwords", "s[0:1]", "4 aligned");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Smem(GcnOp.SLoadDwordx4, GcnOperand.S(0, 4), GcnOperand.S(3, 2), 0)), "an odd base pair", "s[3:4]");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(1), GcnOperand.V(0), GcnOperand.Off)), "a 32-bit address with off", "v1", "pair");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Sop2(GcnOp.SLshlB32, GcnOperand.S(0), GcnOperand.Bits(0x12345678), GcnOperand.Bits(0x9ABCDEF0))), "two literals", "0x9abcdef0", "one literal");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Sop1(GcnOp.SAndSaveexecB64, GcnOperand.S(0), GcnOperand.Vcc)), "one register for a 64-bit destination", "s0", "2 scalar");
        ThrowsNaming<ArgumentException>(() => Encoded(a => a.Vop1(GcnOp.VAddU32, GcnOperand.V(0), GcnOperand.V(1))), "a VOP2 instruction written as VOP1", "VAddU32", "Vop2");
        ThrowsNaming<ArgumentOutOfRangeException>(() => Encoded(a => a.Waitcnt(vmcnt: 64)), "vmcnt(64)", "vmcnt");
        ThrowsNaming<ArgumentOutOfRangeException>(() => Encoded(a => a.Smem(GcnOp.SLoadDwordx4, GcnOperand.S(0, 4), GcnOperand.S(4, 2), 0x100000)), "an SMEM offset past 20 bits", "0xFFFFF");
        ThrowsNaming<ArgumentOutOfRangeException>(() => Encoded(a => a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(1), GcnOperand.V(0), GcnOperand.S(2, 2), offset: 4096)), "a global offset of 4096", "4095");
        ThrowsNaming<InvalidOperationException>(() => Encoded(a => a.Branch(GcnOp.SCbranchExecz, a.NewLabel())), "a branch to a label never bound", "never bound");
        ThrowsNaming<NotSupportedException>(() => _ = new GcnAssembler(new AppFamily()), "a family without encodings", "app-family-test", "IGcnInstructionSet");

        // Nothing allocated per instruction: 1,000 instructions into a buffer sized for them (warm, so the tiered JIT's
        // first code is not what is counted).
        static void Thousand(GcnAssembler a)
        {
            for (int i = 0; i < 200; i++)
            {
                a.Smem(GcnOp.SLoadDwordx4, GcnOperand.S(4, 4), GcnOperand.S(0, 2), 16 * i);
                a.Sop2(GcnOp.SLshlB32, GcnOperand.S(2), GcnOperand.S(2), GcnOperand.Int(i % 65));   // inline
                a.Waitcnt(lgkmcnt: 0);
                a.Vop1(GcnOp.VMovB32, GcnOperand.V(1), GcnOperand.Float(i + 0.25f));         // a literal
                a.GlobalStore(GcnOp.GlobalStoreDword, GcnOperand.V(0), GcnOperand.V(1), GcnOperand.S(4, 2), offset: i);
            }
        }

        for (int warm = 0; warm < 3; warm++)
        {
            using var warmUp = new GcnAssembler(gfx9, 16 * 1024);
            Thousand(warmUp);
        }

        using var asm = new GcnAssembler(gfx9, 16 * 1024);
        long before = GC.GetAllocatedBytesForCurrentThread();
        Thousand(asm);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0 && asm.Length == 200 * 32, $"1,000 instructions ({asm.Length} bytes) allocated {allocated} bytes");
    }

    // The gfx9 CPUs whose versions the registered gfx9 family handles (gfx90a and gfx94x are another family's, plan 35's G8).
    private static readonly string[] GcnRegisteredCpus = ["gfx900", "gfx906", "gfx908", "gfx90c"];

    // One CPU's block of forms.golden: its records (llvm-mc's text and bytes, labels with none) and the forms.s lines it rejects.
    private sealed record GcnFormsBlock(List<(string Text, byte[] Bytes)> Records, List<(int Line, string Text)> Rejected);

    private static string FormsPath(string file) => RepositoryPath("tests", "Idrak.Tests", "data", "gcn", "gfx9", file);

    private static Dictionary<string, GcnFormsBlock> ReadFormsGolden()
    {
        var blocks = new Dictionary<string, GcnFormsBlock>(StringComparer.Ordinal);
        GcnFormsBlock Block(string cpu) => blocks.TryGetValue(cpu, out var block) ? block : blocks[cpu] = new([], []);
        string? rejecting = null;
        foreach (string line in File.ReadLines(FormsPath("forms.golden")))
        {
            if (line.StartsWith("# rejected: ", StringComparison.Ordinal))
            {
                rejecting = line.Split(' ')[2];
            }
            else if (line.StartsWith("#\t", StringComparison.Ordinal) && rejecting is not null)
            {
                string[] fields = line.Split('\t');
                Block(rejecting).Rejected.Add((int.Parse(fields[1], CultureInfo.InvariantCulture), fields[2]));
            }
            else if (line.StartsWith('#') || line.Length == 0)
            {
                rejecting = null;
            }
            else
            {
                string[] fields = line.Split('\t');
                Block(fields[0]).Records.Add((fields[1], Convert.FromHexString(fields[2])));
            }
        }

        return blocks;
    }

    private static byte[] AssembleGcn(IGcnTarget target, GcnIsaVersion isa, string text)
    {
        using var asm = new GcnAssembler(target, isa);
        asm.Assemble(Encoding.UTF8.GetBytes(text));
        return asm.ToArray();
    }

    // A branch to a label (its target needs the block): s_branch or s_cbranch_* and a name, not a number.
    private static bool IsLabelBranch(string text) =>
        (text.StartsWith("s_branch ", StringComparison.Ordinal) || text.StartsWith("s_cbranch_", StringComparison.Ordinal))
        && text.Split(' ')[1] is { Length: > 0 } target && !char.IsAsciiDigit(target[0]) && target[0] != '-';

    internal static void GcnAssemblerGolden(Device device)
    {
        _ = device;
        var golden = ReadFormsGolden();
        string[] source = File.ReadAllLines(FormsPath("forms.s"));
        foreach (string cpu in GcnRegisteredCpus)
        {
            var block = golden[cpu];
            var isa = GcnIsaVersion.Parse(cpu);
            var target = GcnTargets.For(isa);
            byte[] expected = [.. block.Records.SelectMany(r => r.Bytes)];
            int forms = block.Records.Count(r => r.Bytes.Length > 0), labels = block.Records.Count - forms, branches = block.Records.Count(r => IsLabelBranch(r.Text));

            // The block as llvm-mc prints it, one text: each record's bytes at its offset.
            byte[] code = AssembleGcn(target, isa, string.Join('\n', block.Records.Select(r => r.Text)));
            var wrong = new List<string>();
            int matched = 0, at = 0;
            foreach (var (text, bytes) in block.Records.Where(r => r.Bytes.Length > 0))
            {
                if (at + bytes.Length <= code.Length && code.AsSpan(at, bytes.Length).SequenceEqual(bytes))
                {
                    matched++;
                }
                else if (wrong.Count < 12)
                {
                    wrong.Add($"'{text}' at {at}: {Convert.ToHexStringLower(code.AsSpan(Math.Min(at, code.Length), Math.Clamp(code.Length - at, 0, bytes.Length)))}, llvm-mc {Convert.ToHexStringLower(bytes)}");
                }

                at += bytes.Length;
            }

            Check(code.Length == expected.Length && matched == forms, $"{cpu}: {matched} of {forms} forms equal llvm-mc's bytes ({code.Length} bytes, {expected.Length} expected): {string.Join("; ", wrong)}");

            // Each form alone (the branches to labels aside), so a wrong one is named wherever it sits.
            var alone = new List<string>();
            foreach (var (text, bytes) in block.Records.Where(r => r.Bytes.Length > 0 && !IsLabelBranch(r.Text)))
            {
                try
                {
                    byte[] one = AssembleGcn(target, isa, text);
                    if (!one.AsSpan().SequenceEqual(bytes))
                    {
                        alone.Add($"'{text}': {Convert.ToHexStringLower(one)}, llvm-mc {Convert.ToHexStringLower(bytes)}");
                    }
                }
                catch (Exception e) when (e is GcnSyntaxException or ArgumentException or NotSupportedException or InvalidOperationException)
                {
                    alone.Add($"'{text}': refused: {e.Message}");
                }
            }

            Check(alone.Count == 0, $"{cpu}: {alone.Count} forms alone differ from llvm-mc's bytes: {string.Join("; ", alone.Take(12))}");

            // forms.s itself, LLVM's input (no suffixes, s_waitcnt 0, bound_ctrl:0, 65 for 0x41...), less the lines this
            // CPU rejects: the same block.
            var rejected = block.Rejected.Select(r => r.Line).ToHashSet();
            byte[] fromSource = AssembleGcn(target, isa, string.Join('\n', source.Select((line, n) => rejected.Contains(n + 1) ? "" : line)));
            int differs = fromSource.AsSpan().CommonPrefixLength(expected);
            bool same = differs == expected.Length && fromSource.Length == expected.Length;
            Check(same, $"{cpu}: forms.s less its {rejected.Count} rejected lines differs from the golden block at byte {differs}, in '{RecordAt(block, differs)}' "
                        + $"({fromSource.Length} bytes, {expected.Length} expected)");
            Console.WriteLine($"    {cpu}: {matched} of {forms} forms ({labels} labels, {branches} branches to them) equal llvm-mc's bytes, as a block and alone; "
                              + $"forms.s less its {rejected.Count} rejected lines: {(same ? "the same block" : "differs")}");
        }

        static string RecordAt(GcnFormsBlock block, int offset)
        {
            int at = 0;
            foreach (var (text, bytes) in block.Records)
            {
                if (offset < at + bytes.Length)
                {
                    return text;
                }

                at += bytes.Length;
            }

            return "(past the end)";
        }
    }

    internal static void GcnAssemblerRejected(Device device)
    {
        _ = device;
        var golden = ReadFormsGolden();
        var counts = new List<string>();
        foreach (string cpu in GcnRegisteredCpus)
        {
            var isa = GcnIsaVersion.Parse(cpu);
            var target = GcnTargets.For(isa);
            var taken = new List<string>();
            var unnamed = new List<string>();
            int refused = 0;
            foreach (var (line, text) in golden[cpu].Rejected)
            {
                try
                {
                    _ = AssembleGcn(target, isa, text);
                    taken.Add($"{line}: {text}");
                }
                catch (GcnSyntaxException e)
                {
                    refused++;

                    // The reason names the instruction, or the operand or modifier refused (sc0, row_newbcast).
                    string[] words = text.Split([' ', ','], StringSplitOptions.RemoveEmptyEntries);
                    if (e.Line != 1 || e.Column < 1 || !words.Any(w => e.Reason.Contains(w.Split(':')[0], StringComparison.Ordinal)))
                    {
                        unnamed.Add($"{line}: {text}: {e.Message}");
                    }
                }
            }

            Check(taken.Count == 0, $"{cpu}: forms llvm-mc rejects for it were taken: {string.Join("; ", taken)}");
            Check(unnamed.Count == 0, $"{cpu}: refusals that name nothing of the line: {string.Join("; ", unnamed)}");
            counts.Add($"{cpu} {refused} of {golden[cpu].Rejected.Count}");
        }

        Console.WriteLine($"    rejected forms refused: {string.Join(", ", counts)}");

        // The features follow the reported version, never a card: v_fmac_f32 is 9.0.6's and 9.0.8's, not 9.0.0's or 9.0.12's;
        // v_mad_mix_f32 9.0.0's, v_fma_mix_f32 9.0.4's (not in the goldens: its bytes are gfx906's record's).
        var gfx9 = GcnTargets.For(GcnIsaVersion.Parse("gfx906"));
        Check(Convert.ToHexStringLower(AssembleGcn(gfx9, GcnIsaVersion.Parse("gfx906"), "v_fmac_f32 v0, v1, v2")) == "01050076", "v_fmac_f32 on 9.0.6");
        ThrowsNaming<GcnSyntaxException>(() => AssembleGcn(gfx9, GcnIsaVersion.Parse("gfx900"), "v_fmac_f32 v0, v1, v2"), "v_fmac_f32 on 9.0.0", "v_fmac_f32", "dl instructions", "9.0.0");
        ThrowsNaming<GcnSyntaxException>(() => AssembleGcn(gfx9, GcnIsaVersion.Parse("gfx90c:xnack-"), "v_fmac_f32 v0, v1, v2"), "v_fmac_f32 on 9.0.12", "9.0.12");
        Check(Convert.ToHexStringLower(AssembleGcn(gfx9, GcnIsaVersion.Parse("gfx904"), "v_fma_mix_f32 v0, v1, v2, v3")) == "0000a0d301050e04", "v_fma_mix_f32 on 9.0.4");
        ThrowsNaming<GcnSyntaxException>(() => AssembleGcn(gfx9, GcnIsaVersion.Parse("gfx904"), "v_mad_mix_f32 v0, v1, v2, v3"), "v_mad_mix_f32 on 9.0.4", "mad-mix", "9.0.4");
        ThrowsNaming<GcnSyntaxException>(() =>
        {
            using var asm = new GcnAssembler(gfx9);
            asm.Assemble("v_dot2_f32_f16 v0, v1, v2, v3"u8);
        }, "a feature without a version", "v_dot2_f32_f16", "without a version");
        ThrowsNaming<NotSupportedException>(() =>
        {
            using var asm = new GcnAssembler(gfx9);
            asm.Vop2(GcnOp.VFmacF32, GcnOperand.V(0), GcnOperand.V(1), GcnOperand.V(2));
        }, "a typed feature without a version", "v_fmac_f32", "without a version");
        ThrowsNaming<ArgumentException>(() => _ = new GcnAssembler(gfx9, GcnIsaVersion.Parse("gfx90a")), "a version the family does not handle", "gfx90a");
    }

    internal static void GcnAssemblerMalformed(Device device)
    {
        _ = device;
        var isa = GcnIsaVersion.Parse("gfx90c");
        var gfx9 = GcnTargets.For(isa);
        (string Text, int Line, int Column, string Names)[] cases =
        [
            ("v_bogus_f32 v0, v1", 1, 1, "'v_bogus_f32' is not an instruction"),
            ("s_mov_b32 s0,", 1, 14, "expected a scalar source"),
            ("v_add_f32 v0, v1", 1, 17, "expected ','"),
            ("s_mov_b32 s0, s1, s2", 1, 17, "more are written"),
            ("v_add_f32 v0, v1, v2 clamp:1", 1, 22, "clamp takes no value"),
            ("s_nop 0\ns_nop 1 ; a comment\n  v_mov_b32 v0, q7", 3, 17, "'q7' is not an operand"),
            ("s_nop 0\r\nv_mov_b32 v0, q7", 2, 15, "'q7'"),
            ("v_mov_b32 v0, %v12", 1, 15, "virtual register"),
            (".amdhsa_kernel fill", 1, 1, "directives"),
            ("v_add_f32 v0, v1, v[2:", 1, 20, "ends with ']'"),
            ("v_add_f32 v0, v1, v[3:2]", 1, 20, "not a register range"),
            ("v_mov_b32 v0, v1 row_newbcast:1", 1, 18, "not a modifier"),
            ("ds_read_b32 v0, v1 glc", 1, 20, "takes no glc"),
            ("s_waitcnt vmcnt(64)", 1, 17, "vmcnt is 0 to 63"),
            ("s_waitcnt vmcnt(1) vmcnt(2)", 1, 20, "named twice"),
            ("s_branch", 1, 9, "expected a label"),
            ("loop:\nloop:", 2, 1, "defined twice"),
            ("s_mov_b32 s0, 0x100000000", 1, 15, "does not fit 32 bits"),
            ("s_movk_i32 s0, 0x10000", 1, 16, "-32768 to 65535"),
            ("s_load_dword s0, s[3:4], 0x0", 1, 18, "even scalar register pair"),
            ("global_load_dword v0, v[2:3], off offset:4096", 1, 1, "-4096 to 4095"),
            ("global_load_dword v0, v1, off", 1, 23, "vector register pair"),
            ("global_atomic_add v0, v[2:3], v1, off", 1, 1, "only with glc"),
            ("v_add_u32_e32 v0, v1, s2", 1, 23, "reads src1 from a vector register"),
            ("v_add_f32_e64 v0, v1, 0x3fb8aa3b", 1, 23, "no literal"),
            ("v_add_f32 v0, s1, s2", 1, 19, "at most 1 scalar value"),
            ("v_cndmask_b32 v0, s1, v2, vcc", 1, 19, "at most 1 scalar value"),
            ("v_add_u32_e64 v0, -v1, v2", 1, 19, "no negation"),
            ("v_div_scale_f32 v0, vcc, v1, |v2|, v3", 1, 1, "no abs"),
            ("v_readfirstlane_b32_e64 s0, v1", 1, 1, "no VOP3 form"),
            ("v_cmp_lt_f32_dpp vcc, v1, v2 row_shr:1", 1, 30, "takes no row_shr"),
            ("v_mov_b32_dpp v0, v1", 1, 1, "needs a control"),
            ("ds_swizzle_b32 v0, v1 offset:swizzle(SWAP,3)", 1, 38, "SWAP and a group of 1-16"),
            ("v_pk_add_f16 v0, v1, v2 op_sel_hi:[1,0,1,1]", 1, 42, "at most 3 values"),
            ("v_pk_add_f16 v0, v1, v2 op_sel:[1,0,1,1]", 1, 1, "at most 3 sources"),
        ];
        foreach (var (text, line, column, names) in cases)
        {
            try
            {
                _ = AssembleGcn(gfx9, isa, text);
                Check(false, $"'{text.ReplaceLineEndings(" | ")}' is refused");
            }
            catch (GcnSyntaxException e)
            {
                Check(e.Line == line && e.Column == column && e.Reason.Contains(names, StringComparison.Ordinal) && e.Message.StartsWith($"GCN line {line}, column {column}: ", StringComparison.Ordinal),
                    $"'{text.ReplaceLineEndings(" | ")}': line {e.Line}, column {e.Column} ({line}, {column} expected), naming '{names}': {e.Message}");
            }
        }

        // A label branched to and never defined: refused when the code is made, naming it and the line.
        ThrowsNaming<InvalidOperationException>(() => AssembleGcn(gfx9, isa, "s_nop 0\ns_branch nowhere"), "a label never defined", "'nowhere'", "line 2");
    }

    internal static void GcnAssemblerAllocations(Device device)
    {
        _ = device;
        var isa = GcnIsaVersion.Parse("gfx90c");
        var gfx9 = GcnTargets.For(isa);
        var block = ReadFormsGolden()["gfx90c"];
        byte[] text = Encoding.UTF8.GetBytes(string.Join('\n', block.Records.Select(r => r.Text)));
        int expected = block.Records.Sum(r => r.Bytes.Length);
        using var asm = new GcnAssembler(gfx9, isa, 2 * expected);
        void Block()
        {
            asm.Reset();
            asm.Assemble(text);
        }

        for (int warm = 0; warm < 40; warm++)
        {
            Block();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        Block();
        long assembled = GC.GetAllocatedBytesForCurrentThread() - before;
        byte[] code = asm.ToArray();
        long withArray = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(assembled == 0 && asm.Length == expected, $"assembling gfx90c's block ({asm.Length} bytes of code) allocated {assembled} bytes once warm{(assembled == 0 ? "" : $" ({AllocatedTypes(Block)})")}");
        Check(withArray - assembled <= code.Length + 32, $"the code array: {withArray - assembled} bytes allocated for {code.Length} bytes of code");

        // fill_f32 as text: the same code as the typed fill_f32 (and so as llvm-mc's, GcnGateGolden).
        byte[] fill = AssembleGcn(gfx9, isa, """
            s_load_dwordx4 s[4:7], s[0:1], 0x0   ; s[4:5] out, s6 value, s7 count
            s_lshl_b32 s2, s2, 8                 ; the block's first element
            v_add_u32_e32 v0, s2, v0
            s_waitcnt lgkmcnt(0)
            v_cmp_gt_u32_e32 vcc, s7, v0         // element < count
            s_and_saveexec_b64 s[0:1], vcc
            s_cbranch_execz done
            v_lshlrev_b32_e32 v0, 2, v0
            v_mov_b32_e32 v1, s6
            global_store_dword v0, v1, s[4:5]
            done:
            s_endpgm
            """);
        Check(fill.AsSpan().SequenceEqual(GcnKernels.Fill(gfx9).Code.Span), $"fill_f32 as text: {Convert.ToHexStringLower(fill)}, typed {Convert.ToHexStringLower(GcnKernels.Fill(gfx9).Code.Span)}");
    }

    internal static void GcnGateGolden(Device device)
    {
        _ = device;
        var isa = GcnIsaVersion.Parse("gfx90c");
        var fill = GcnKernels.Fill(GcnTargets.For(isa));
        byte[] goldenCode = new ElfReader(File.ReadAllBytes(GoldenPath("fill_f32.gfx90c.v5.co"))).SectionBytes(".text");
        Check(fill.Code.Span.SequenceEqual(goldenCode), $"fill_f32's code: {Convert.ToHexStringLower(fill.Code.Span)}, llvm-mc's {Convert.ToHexStringLower(goldenCode)}");
        Check(fill.Code.Span.SequenceEqual(Words(FillWords)), "the same words as the code object writer's tests");
        Check(fill.Arguments.SequenceEqual(GcnKernels.FillArguments) && GcnCodeObject.KernargBytes(fill) == GcnKernels.FillArgumentBytes && fill is { Vgprs: 2, Sgprs: 8, UsesVcc: true },
            "its arguments (16 bytes), registers and VCC as the golden's descriptor says");
        foreach (int format in (int[])[5, 4])
        {
            CheckAgainstGolden(GcnKernels.FillCodeObject(isa, format), format);
        }

        // The arguments a launch packs, at the metadata's offsets.
        Span<byte> arguments = stackalloc byte[GcnKernels.FillArgumentBytes];
        GcnKernels.PackFill(arguments, 0x0000_7F12_3456_7800, 2.5f, 1_000_003);
        Check(Convert.ToHexStringLower(arguments) == "00785634127f0000" + "00002040" + "43420f00", $"packed: out at 0, value at 8, count at 12: {Convert.ToHexStringLower(arguments)}");

        // The device's own target: its features change the flags and the target id, never the code.
        var xnackOff = GcnIsaVersion.Parse("gfx90c:xnack-");
        var off = new ElfReader(GcnKernels.FillCodeObject(xnackOff, 5));
        Check(off.U32(48) == 0x232 && off.SectionBytes(".text").SequenceEqual(goldenCode), $"gfx90c:xnack-: flags 0x{off.U32(48):X}, the same code");
        ThrowsNaming<NotRegisteredException>(() => GcnKernels.FillCodeObject(GcnIsaVersion.Parse("gfx1030"), 5), "a target no family handles", "gfx1030", "not registered");
    }

    internal static void GcnGateDevice(Device device)
    {
        if (device.Type != DeviceType.Hip)
        {
            return;
        }

        // The refusal, when the runtime takes no format, is the test's failure: its message names the target, each
        // format with the runtime's error, and the runtime.
        var hip = (HipBackend)device.Backend;
        using var fill = hip.LoadGcn(GcnKernels.Fill);
        Console.WriteLine($"    {device}: Idrak's fill_f32 loaded from code object format {fill.Format} for {fill.Target}"
                          + (fill.Refused.Length > 0 ? $" (refused first: {fill.Refused})" : "")
                          + $"; HIP runtime {HipBackend.FormatVersion(hip.RuntimeVersion)}, driver {hip.DriverVersion}, from {HipRuntime.LoadedFrom}");

        // 1,000,003 elements: 3,907 blocks, the last with 67 lanes in bounds; the floats past them (beyond the last block's
        // reach, 1,000,192) keep what was uploaded.
        const int Count = 1_000_003, Past = 1024;
        float[] initial = [.. Enumerable.Range(0, Count + Past).Select(i => -1f - i % 251)];
        var storage = hip.Allocate(Count + Past, zeroed: false);
        try
        {
            hip.Upload(initial, storage);
            hip.FillGcn(fill, storage, Count, 2.5f);
            var result = new float[Count + Past];
            hip.Download(storage, result);
            int wrong = Array.FindIndex(result, 0, Count, v => v != 2.5f);
            Check(wrong < 0, $"element {wrong} is {(wrong < 0 ? 0 : result[wrong])}, not 2.5 ({result.AsSpan(0, Count).Count(2.5f)} of {Count} filled)");
            int touched = FirstChanged(result, initial, Count);
            Check(touched < 0, $"element {touched}, past the count, was written: {(touched < 0 ? 0 : result[touched])} where {(touched < 0 ? 0 : initial[touched])} was");
        }
        finally
        {
            storage.Release();
        }

        static int FirstChanged(float[] result, float[] initial, int from)
        {
            for (int i = from; i < result.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(result[i]) != BitConverter.SingleToInt32Bits(initial[i]))
                {
                    return i;
                }
            }

            return -1;
        }
    }

    // ---- The register allocator, waits and hazards (G2a) --------------------------------------------------------

    private static readonly GcnIsaVersion KernelIsa = GcnIsaVersion.Parse("gfx90c");

    // A kernel's text assembled for gfx90c (or isa): its code and its printed text.
    private static (GcnKernelCode Code, string Printed) KernelOf(string text, GcnIsaVersion? isa = null, int? waves = null)
    {
        var version = isa ?? KernelIsa;
        using var asm = new GcnAssembler(GcnTargets.For(version), version);
        var code = asm.AssembleKernel(Encoding.UTF8.GetBytes(text), waves);
        return (code, asm.PrintedText);
    }

    // The printed text, read back by Assemble, gives the kernel's bytes.
    private static void CheckReadsBack(string what, GcnKernelCode code, string printed)
    {
        byte[] again = AssembleGcn(GcnTargets.For(KernelIsa), KernelIsa, printed);
        Check(again.AsSpan().SequenceEqual(code.Code.Span), $"{what}: its printed text reads back to {Convert.ToHexStringLower(again)}, not {Convert.ToHexStringLower(code.Code.Span)}:\n{printed}");
    }

    private static GcnKernelImage ImageOf(GcnKernelCode code) =>
        GcnCodeObjectReader.Read(GcnCodeObject.Write(GcnTargets.For(KernelIsa), KernelIsa, 5, [code]), code.Name);

    private static string[] PrintedLines(string printed) => [.. printed.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0)];

    // The printed line before the `occurrence`-th line (from 0) that starts with `prefix` and ends with `suffix`.
    private static string Before(string printed, string prefix, string suffix = "", int occurrence = 0)
    {
        var lines = PrintedLines(printed);
        for (int at = 0, seen = 0; at < lines.Length; at++)
        {
            if (lines[at].StartsWith(prefix, StringComparison.Ordinal) && lines[at].EndsWith(suffix, StringComparison.Ordinal) && seen++ == occurrence)
            {
                return at > 0 ? lines[at - 1] : "(the first line)";
            }
        }

        return $"(no line '{prefix}...{suffix}')";
    }

    // The kernel with its n-th s_waitcnt (as the interpreter's decoder reads the code) made s_nop 0; null past the last.
    private static GcnKernelCode? WithoutWait(GcnKernelCode code, int n)
    {
        byte[] bytes = code.Code.ToArray();
        var decoder = new GcnDecoder(GcnCpu.Gfx90c);
        for (int at = 0, seen = 0; at < bytes.Length;)
        {
            var instruction = decoder.Decode(bytes, at);
            if (instruction.Text.StartsWith("s_waitcnt", StringComparison.Ordinal) && seen++ == n)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(at), 0xBF800000);
                return code with { Code = bytes };
            }

            at += instruction.Size;
        }

        return null;
    }

    private static byte[] KernelArguments(int bytes, params (int Offset, ulong Value, int Size)[] values)
    {
        var arguments = new byte[bytes];
        foreach (var (offset, value, size) in values)
        {
            if (size == 8)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(arguments.AsSpan(offset), value);
            }
            else
            {
                BinaryPrimitives.WriteUInt32LittleEndian(arguments.AsSpan(offset), (uint)value);
            }
        }

        return arguments;
    }

    private static ulong FloatBits(float value) => BitConverter.SingleToUInt32Bits(value);

    private static int FirstDifferent(float[] got, Func<int, float> expected, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected(i)))
            {
                return i;
            }
        }

        return -1;
    }

    // Runs the kernel, then each of its waits taken out in turn: every one must be missed (a missing wait, or a race).
    // What the interpreter reported, in order.
    private static List<(GcnFailure Kind, string Message)> RunWithoutEachWait(string what, GcnKernelCode code, Func<GcnMemory> memory, GcnDispatch dispatch)
    {
        GcnInterpreter.Run(ImageOf(code), memory(), dispatch);
        var kinds = new List<(GcnFailure, string)>();
        for (int n = 0; WithoutWait(code, n) is { } without; n++)
        {
            try
            {
                GcnInterpreter.Run(ImageOf(without), memory(), dispatch);
                Check(false, $"{what}: without its wait {n}, the interpreter reports nothing");
            }
            catch (GcnInterpreterException e)
            {
                Check(e.Kind is GcnFailure.MissingWait or GcnFailure.Race, $"{what}: without its wait {n}: {e.Kind}: {e.Message}");
                kinds.Add((e.Kind, e.Message));
            }
        }

        Check(kinds.Count > 0, $"{what}: it has no s_waitcnt to take out");
        return kinds;
    }

    // out[i] = x[i] * a + y[i]: two global loads in flight (vmcnt(1), then vmcnt(0)), three SMEM loads in one clause.
    private const string AxpyText = """
        .kernel axpy
        .arg x, 8, global
        .arg y, 8, global
        .arg out, 8, global
        .arg a, 4, value
        .reqd_workgroup_size 64, 1, 1
        .user_sgpr kernarg_segment_ptr
        .system_sgpr workgroup_id_x
          s_load_dwordx4 %s[0:3], s[0:1], 0x0      ; x, y
          s_load_dwordx2 %s[4:5], s[0:1], 0x10     ; out
          s_load_dword %s6, s[0:1], 0x18           ; a
          s_lshl_b32 %s7, s2, 6
          v_add_u32_e32 %v0, %s7, v0
          v_lshlrev_b32_e32 %v1, 2, %v0
          global_load_dword %v2, %v1, %s[0:1]
          global_load_dword %v3, %v1, %s[2:3]
          v_mul_f32_e32 %v4, %s6, %v2
          v_add_f32_e32 %v5, %v4, %v3
          global_store_dword %v1, %v5, %s[4:5]
          s_endpgm
        """;

    // For i < n: out[i] = x[i] > 0 ? x[i] added k times : x[i]. A bounds check, a nested if restored by s_or_b64, a loop.
    private const string BranchyText = """
        .kernel branchy
        .arg x, 8, global
        .arg out, 8, global
        .arg n, 4, value
        .arg k, 4, value
        .max_flat_workgroup_size 128
        .user_sgpr kernarg_segment_ptr
        .system_sgpr workgroup_id_x
          s_load_dwordx4 %s[0:3], s[0:1], 0x0
          s_load_dwordx2 %s[4:5], s[0:1], 0x10
          s_lshl_b32 %s6, s2, 7
          v_add_u32_e32 %v0, %s6, v0
          v_cmp_gt_u32_e32 vcc, %s4, %v0
          s_and_saveexec_b64 %s[8:9], vcc
          s_cbranch_execz .Lend
          v_lshlrev_b32_e32 %v1, 2, %v0
          global_load_dword %v2, %v1, %s[0:1]
          v_mov_b32_e32 %v3, %v2
          v_cmp_lt_f32_e32 vcc, 0, %v2
          s_and_saveexec_b64 %s[10:11], vcc
          s_cbranch_execz .Ljoin
          s_mov_b32 %s12, %s5
          v_mov_b32_e32 %v3, 0
        .Lloop:
          v_add_f32_e32 %v3, %v3, %v2
          s_add_u32 %s12, %s12, -1
          s_cmp_lg_u32 %s12, 0
          s_cbranch_scc1 .Lloop
        .Ljoin:
          s_or_b64 exec, exec, %s[10:11]
          global_store_dword %v1, %v3, %s[2:3]
        .Lend:
          s_endpgm
        """;

    internal static void GcnAssemblerFillVirtual(Device device)
    {
        _ = device;
        using var asm = new GcnAssembler(GcnTargets.For(KernelIsa), KernelIsa);
        var fill = asm.AssembleKernel(GcnKernels.FillText);
        string printed = asm.PrintedText;
        byte[] golden = new ElfReader(File.ReadAllBytes(GoldenPath("fill_f32.gfx90c.v5.co"))).SectionBytes(".text");
        Check(fill.Code.Span.SequenceEqual(golden), $"fill_f32 through virtual registers: {Convert.ToHexStringLower(fill.Code.Span)}, llvm-mc's {Convert.ToHexStringLower(golden)}:\n{printed}");

        // The golden's instruction lines (fill_f32.s, comments and indentation aside) are the printed text's.
        string[] source = [.. File.ReadAllLines(GoldenPath("fill_f32.s")).Select(l => (l.IndexOf(';', StringComparison.Ordinal) is >= 0 and var c ? l[..c] : l).Trim())];
        string[] expected = [.. source[(Array.IndexOf(source, "fill_f32:") + 1)..Array.IndexOf(source, ".Lfill_f32_end:")].Where(l => l.Length > 0)];
        Check(PrintedLines(printed).SequenceEqual(expected), $"the printed text is the golden's:\n{printed}\nexpected:\n{string.Join('\n', expected)}");
        Check(fill is { Name: "fill_f32", Vgprs: 2, Sgprs: 8, UsesVcc: true, LdsBytes: 0, UserSgprs: GcnUserSgprs.KernargSegmentPtr, WorkgroupIds: 1, WorkitemIds: 1, MaxFlatWorkgroupSize: 256 }
              && fill.RequiredWorkgroupSize == (256, 1, 1) && fill.Arguments.SequenceEqual(GcnKernels.FillArguments) && GcnCodeObject.KernargBytes(fill) == 16,
            $"from use and the directives: {fill.Vgprs} VGPRs, {fill.Sgprs} SGPRs, VCC {fill.UsesVcc}, {fill.UserSgprs}, {fill.WorkgroupIds} workgroup ids, block {fill.MaxFlatWorkgroupSize}");
        Check(asm.PhysicalRegister('s', 0) == 4 && asm.PhysicalRegister('s', 3) == 7 && asm.PhysicalRegister('s', 4) == 2 && asm.PhysicalRegister('s', 6) == 0
              && asm.PhysicalRegister('v', 0) == 0 && asm.PhysicalRegister('v', 1) == 0 && asm.PhysicalRegister('v', 2) == 1,
            $"%s[0:3] in s[4:7], %s4 in s2 and %v0 in v0 in place, %s[6:7] in s[0:1], %v1 in v0 (written under the narrowed EXEC with no value reaching it), %v2 in v1: %s0 s{asm.PhysicalRegister('s', 0)}, %s4 s{asm.PhysicalRegister('s', 4)}, %s6 s{asm.PhysicalRegister('s', 6)}, %v1 v{asm.PhysicalRegister('v', 1)}, %v2 v{asm.PhysicalRegister('v', 2)}");
        CheckReadsBack("fill_f32", fill, printed);

        // 300 elements in two blocks of 256: the second block's last 212 lanes are past the count.
        var memory = new GcnMemory();
        ulong output = memory.Map<float>("out", Enumerable.Repeat(-7.25f, 512).ToArray());
        GcnInterpreter.Run(ImageOf(fill), memory, new GcnDispatch
        {
            Workgroups = (2, 1, 1), WorkgroupSize = (256, 1, 1), Arguments = KernelArguments(16, (0, output, 8), (8, FloatBits(2.5f), 4), (12, 300, 4)),
        });
        float[] result = memory.Read<float>(output, 512);
        Check(result.AsSpan(0, 300).IndexOfAnyExcept(2.5f) < 0 && result.AsSpan(300).IndexOfAnyExcept(-7.25f) < 0,
            $"filled: out[{result.AsSpan(0, 300).IndexOfAnyExcept(2.5f)}] not 2.5, or out[{300 + result.AsSpan(300).IndexOfAnyExcept(-7.25f)}] past the count written");
    }

    internal static void GcnAssemblerAllocation(Device device)
    {
        _ = device;

        // Straight-line: out = x * a + y over two workgroups of 64.
        var (axpy, printed) = KernelOf(AxpyText);
        CheckReadsBack("axpy", axpy, printed);
        Check(axpy is { Vgprs: 3, Sgprs: 10, UsesVcc: false, Arguments.Count: 4 } && GcnCodeObject.KernargBytes(axpy) == 28,
            $"axpy: 3 VGPRs and 10 SGPRs (v0 and s2 reused in place, each value's register freed at its last use), 28 bytes of arguments: {axpy.Vgprs}, {axpy.Sgprs}, {GcnCodeObject.KernargBytes(axpy)}:\n{printed}");
        const int N = 128;
        float[] x = [.. Enumerable.Range(0, N).Select(i => 0.25f * i - 3)], y = [.. Enumerable.Range(0, N).Select(i => 100f - i)];
        var memory = new GcnMemory();
        ulong xs = memory.Map<float>("x", x), ys = memory.Map<float>("y", y), output = memory.Map("out", 4 * N);
        GcnInterpreter.Run(ImageOf(axpy), memory, new GcnDispatch
        {
            Workgroups = (2, 1, 1), WorkgroupSize = (64, 1, 1), Arguments = KernelArguments(28, (0, xs, 8), (8, ys, 8), (16, output, 8), (24, FloatBits(1.5f), 4)),
        });
        float[] got = memory.Read<float>(output, N);
        int wrong = FirstDifferent(got, i => x[i] * 1.5f + y[i], N);
        Check(wrong < 0, $"axpy: out[{wrong}] is {(wrong < 0 ? 0 : got[wrong])}, the CPU's {(wrong < 0 ? 0 : x[wrong] * 1.5f + y[wrong])}");

        // Branching: a bounds check, a nested if and a loop, 200 elements in workgroups of 128, k = 3.
        var (branchy, branchyText) = KernelOf(BranchyText);
        CheckReadsBack("branchy", branchy, branchyText);
        Check(branchy.Vgprs <= 4 && branchy.Sgprs <= 14 && branchy.UsesVcc, $"branchy: {branchy.Vgprs} VGPRs (at most 4), {branchy.Sgprs} SGPRs (at most 14), VCC used:\n{branchyText}");
        const int Count = 200, K = 3;
        float[] values = [.. Enumerable.Range(0, 256).Select(i => (i * 37 % 11) - 5.25f)];
        memory = new GcnMemory();
        xs = memory.Map<float>("x", values);
        output = memory.Map<float>("out", Enumerable.Repeat(-7f, 256).ToArray());
        GcnInterpreter.Run(ImageOf(branchy), memory, new GcnDispatch
        {
            Workgroups = (2, 1, 1), WorkgroupSize = (128, 1, 1), Arguments = KernelArguments(24, (0, xs, 8), (8, output, 8), (16, Count, 4), (20, K, 4)),
        });
        got = memory.Read<float>(output, 256);
        float Expected(int i)
        {
            if (i >= Count)
            {
                return -7f;
            }

            float sum = 0;
            for (int t = 0; t < K; t++)
            {
                sum += values[i];
            }

            return values[i] > 0 ? sum : values[i];
        }

        wrong = FirstDifferent(got, Expected, 256);
        Check(wrong < 0, $"branchy: out[{wrong}] is {(wrong < 0 ? 0 : got[wrong])}, the CPU's {(wrong < 0 ? 0 : Expected(wrong))}:\n{branchyText}");
    }

    // %v1 is 1.0 in every lane, then 2.0 in lanes 0-31 under a narrowed EXEC, in a block laid out before the one that
    // sets it: %v1 is live out of that block, so %v9, written after %v1 there, takes another register.
    private const string PartialText = """
        .kernel partial
        .arg out, 8, global
        .reqd_workgroup_size 64, 1, 1
        .user_sgpr kernarg_segment_ptr
          s_branch .Lsetup
        .Lnarrowed:
          v_mov_b32_e32 %v1, 2.0                 ; lanes 0-31: 2.0; the others keep 1.0
          s_or_b64 exec, exec, %s[4:5]
          v_lshlrev_b32_e32 %v3, 2, v0
          global_store_dword %v3, %v1, %s[0:1]
          s_endpgm
        .Lsetup:
          s_load_dwordx2 %s[0:1], s[0:1], 0x0
          v_mov_b32_e32 %v1, 1.0
          v_add_u32_e32 %v9, 100, v0
          v_cmp_gt_u32_e32 vcc, 0x84, %v9        ; lanes 0-31
          s_and_saveexec_b64 %s[4:5], vcc
          s_branch .Lnarrowed
        """;

    internal static void GcnAssemblerPartialDefinition(Device device)
    {
        _ = device;
        using var asm = new GcnAssembler(GcnTargets.For(KernelIsa), KernelIsa);
        var partial = asm.AssembleKernel(Encoding.UTF8.GetBytes(PartialText));
        string printed = asm.PrintedText;
        CheckReadsBack("partial", partial, printed);
        int v1 = asm.PhysicalRegister('v', 1), v9 = asm.PhysicalRegister('v', 9);
        Check(v1 != v9, $"%v1, written under the narrowed EXEC with its value reaching there, is live through %v9's life: v{v1} and v{v9}:\n{printed}");
        var memory = new GcnMemory();
        ulong output = memory.Map("out", 4 * 64);
        GcnInterpreter.Run(ImageOf(partial), memory, new GcnDispatch { WorkgroupSize = (64, 1, 1), Arguments = KernelArguments(8, (0, output, 8)) });
        float[] got = memory.Read<float>(output, 64);
        int wrong = FirstDifferent(got, lane => lane < 32 ? 2f : 1f, 64);
        Check(wrong < 0, $"lane {wrong} holds {(wrong < 0 ? 0 : got[wrong])}: lanes 0-31 take 2.0, the others keep 1.0:\n{printed}");

        // Where no value reaches the narrowed write, it starts the register: %v6 takes the register %v5 left.
        var (fresh, freshText) = KernelOf("""
            .kernel fresh
            .arg out, 8, global
            .reqd_workgroup_size 64, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_load_dwordx2 %s[0:1], s[0:1], 0x0
              v_lshlrev_b32_e32 %v0, 2, v0
              v_mov_b32_e32 %v5, 2.0
              global_store_dword %v0, %v5, %s[0:1]
              v_cmp_gt_u32_e32 vcc, 32, v0
              s_and_saveexec_b64 %s[2:3], vcc
              v_mov_b32_e32 %v6, 4.0
              global_store_dword %v0, %v6, %s[0:1] offset:256
              s_endpgm
            """);
        Check(fresh.Vgprs == 3 && PrintedLines(freshText).Any(l => l.StartsWith("v_mov_b32_e32 v0, 4.0", StringComparison.Ordinal)),
            $"%v6 starts in v0, the work-item id's register, free once the compare read it: 3 VGPRs, not a fourth for a %v6 live from the start ({fresh.Vgprs}):\n{freshText}");
    }

    internal static void GcnAssemblerRanges(Device device)
    {
        _ = device;
        using var asm = new GcnAssembler(GcnTargets.For(KernelIsa), KernelIsa);
        asm.AssembleKernel("""
            .kernel ranges
            .arg p, 8, global
            .reqd_workgroup_size 64, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_mov_b32 %s1, 7
              s_load_dwordx2 %s[2:3], s[0:1], 0x0
              s_load_dwordx8 %s[8:15], s[0:1], 0x20
              v_mov_b32_e32 %v1, %s9
              v_mov_b32_e32 %v2, %s15
              v_add_u32_e32 %v3, %s1, %v1
              v_mov_b32_e32 %v4, 0
              v_mov_b32_e32 %v5, 1
              v_mov_b32_e32 %v6, 2
              v_add_f64 %v[8:9], %v[4:5], %v[5:6]
              global_load_dwordx4 %v[10:13], %v3, %s[2:3]
              v_add_u32_e32 %v14, %v10, %v13
              v_add_u32_e32 %v14, %v14, %v2
              global_store_dwordx2 %v3, %v[8:9], %s[2:3]
              global_store_dword %v3, %v14, %s[2:3]
              s_endpgm
            """u8);
        int P(char file, int id) => asm.PhysicalRegister(file, id);
        Check(P('s', 2) % 2 == 0 && P('s', 3) == P('s', 2) + 1, $"%s[2:3] is an even pair: s{P('s', 2)}");
        Check(P('s', 8) % 4 == 0 && P('s', 9) == P('s', 8) + 1 && P('s', 15) == P('s', 8) + 7, $"%s[8:15] is 4-aligned and %s9, %s15 are its registers: s{P('s', 8)}, s{P('s', 9)}, s{P('s', 15)}");
        Check(P('v', 5) == P('v', 4) + 1 && P('v', 6) == P('v', 4) + 2 && P('v', 9) == P('v', 8) + 1, $"%v[4:5] and %v[5:6] are one run: v{P('v', 4)}, v{P('v', 5)}, v{P('v', 6)}");
        Check(P('v', 11) == P('v', 10) + 1 && P('v', 13) == P('v', 10) + 3, $"%v[10:13] is adjacent: v{P('v', 10)}..v{P('v', 13)}");
        Check(P('s', 1) != P('s', 2) && P('s', 1) != P('s', 3), $"%s1, live across the pair's load, is not in it: s{P('s', 1)}");

        // Refusals, each with its line, the column of what it names and its reason.
        const string Head = ".kernel bad\n.arg p, 8, global\n.user_sgpr kernarg_segment_ptr\n";
        (string Text, int Line, string At, string Reason)[] cases =
        [
            (Head + "  s_load_dwordx4 %s[0:3], s[0:1], 0x0\n  s_load_dwordx2 %s[4:5], %s[1:2], 0x0\n  s_endpgm", 5, "%s[1:2]", "cannot be aligned"),
            (Head + "  v_mov_b32_e32 %v1, %v7\n  s_endpgm", 4, "%v7", "%v7 is read but never written"),
            (Head + "  v_mov_b32_e32 %v1, s5\n  s_endpgm", 4, "s5", "holds no system value"),
            (Head + "  s_mov_b32 s0, 1\n  s_endpgm", 4, "s0", "reads only"),
            (Head + "  s_nop 0\n.lds 16\n  s_endpgm", 5, ".lds", "directives come before"),
            (Head + ".bogus 1\n  s_endpgm", 4, ".bogus", "not a directive"),
            (Head + "  s_mov_b32 %s0, ttmp0\n  s_endpgm", 4, "ttmp0", "not allocated"),
            ("  s_endpgm", 1, "", "starts with .kernel"),
            (Head + "  s_branch .Lx\n.Lx:", 4, "", "ends the kernel"),
            (Head + "  v_mov_b32_e32 %v0, 1", 4, "", "runs past its last instruction"),
            (Head + "  s_movrels_b32 %s0, %s1\n  s_endpgm", 4, "s_movrels_b32", "not taken in a kernel's text"),
            (Head + "  s_branch .Lnowhere\n  s_endpgm", 4, ".Lnowhere", "never defined"),
            (Head + "  s_branch 1\n  s_endpgm", 4, "1", "names a label"),
            (Head + "  v_mov_b32_e32 %s0, 1\n  s_endpgm", 4, "%s0", "vector register"),
            (Head + ".arg q, 4, global\n  s_endpgm", 4, "4", "8 bytes"),
            (Head + "  v_mov_b32_e32 %v[0:16], 0\n  s_endpgm", 4, "%v[0:16]", "1 to 16 virtual registers"),
        ];
        foreach (var (text, line, at, reason) in cases)
        {
            string[] lines = text.Split('\n');
            int column = at.Length == 0 ? 1 : lines[line - 1].IndexOf(at, StringComparison.Ordinal) + 1;
            try
            {
                _ = KernelOf(text);
                Check(false, $"'{text.ReplaceLineEndings(" | ")}' is refused");
            }
            catch (GcnSyntaxException e)
            {
                Check(e.Line == line && e.Column == column && e.Reason.Contains(reason, StringComparison.Ordinal),
                    $"'{text.ReplaceLineEndings(" | ")}': line {e.Line}, column {e.Column} ({line}, {column} expected), naming '{reason}': {e.Message}");
            }
        }
    }

    // `count` registers of one file set, then summed: all of them live at once before the sum.
    private static string ManyRegisters(int count, int block, bool scalar)
    {
        char file = scalar ? 's' : 'v';
        string move = scalar ? "s_mov_b32" : "v_mov_b32_e32", add = scalar ? "s_add_u32" : "v_add_u32_e32";
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $".kernel many\n.max_flat_workgroup_size {block}\n");
        for (int i = 0; i < count; i++)
        {
            text.Append(CultureInfo.InvariantCulture, $"  {move} %{file}{i}, {i}\n");
        }

        text.Append(CultureInfo.InvariantCulture, $"  {add} %{file}1000, %{file}0, %{file}1\n");
        for (int i = 2; i < count; i++)
        {
            text.Append(CultureInfo.InvariantCulture, $"  {add} %{file}1000, %{file}1000, %{file}{i}\n");
        }

        return text.Append(scalar ? "  v_mov_b32_e32 %v0, %s1000\n  s_endpgm\n" : "  s_endpgm\n").ToString();
    }

    internal static void GcnAssemblerBudget(Device device)
    {
        _ = device;
        foreach (var (scalar, count, block, budget) in (ReadOnlySpan<(bool, int, int, int)>)[(false, 65, 1024, 64), (true, 103, 1024, 102), (true, 103, 64, 102)])
        {
            try
            {
                _ = KernelOf(ManyRegisters(count, block, scalar));
                Check(false, $"{count} {(scalar ? "scalar" : "vector")} registers live in a block of {block} are refused");
            }
            catch (GcnAllocationException e)
            {
                string what = scalar ? "scalar" : "vector";
                Check(e is { Kernel: "many" } && e.File == (scalar ? 's' : 'v') && e.Needed == count && e.Budget == budget
                      && e.Message.Contains("'many'", StringComparison.Ordinal) && e.Message.Contains($"needs {count} {what} registers at once", StringComparison.Ordinal)
                      && e.Message.Contains($"{budget} fit", StringComparison.Ordinal) && e.Message.Contains("spill", StringComparison.Ordinal),
                    $"{count} {what} registers in a block of {block}: {e.File} {e.Needed} of {e.Budget}: {e.Message}");
            }
        }

        var (fits, _) = KernelOf(ManyRegisters(65, 256, scalar: false));
        Check(fits.Vgprs == 65, $"65 vector registers fit a block of 256 (256 a lane): {fits.Vgprs}");
        var (scalars, _) = KernelOf(ManyRegisters(102, 1024, scalar: true));
        Check(scalars.Sgprs == 102, $"102 scalar registers fit: {scalars.Sgprs}");

        // The occupancy aimed for, when given, sets the budget: 8 waves a SIMD leave a lane 32 vector registers.
        ThrowsNaming<GcnAllocationException>(() => KernelOf(ManyRegisters(40, 256, scalar: false), waves: 8), "40 live registers at 8 waves a SIMD", "needs 40 vector registers", "32 fit");
        Check(KernelOf(ManyRegisters(32, 256, scalar: false), waves: 8).Code.Vgprs == 32, "32 fit at 8 waves a SIMD");
    }

    // A fresh interpreter memory with copies of these storages mapped in order: the same addresses every time.
    private static GcnMemory Storages(out ulong[] addresses, params byte[][] storages)
    {
        var memory = new GcnMemory();
        addresses = new ulong[storages.Length];
        for (int i = 0; i < storages.Length; i++)
        {
            addresses[i] = memory.Map($"storage {i}", (byte[])storages[i].Clone());
        }

        return memory;
    }

    private static byte[] FloatBytes(float[] values)
    {
        var bytes = new byte[4 * values.Length];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    internal static void GcnAssemblerWaits(Device device)
    {
        _ = device;
        float[] x = [.. Enumerable.Range(0, 64).Select(i => 0.5f * i - 7)];
        byte[] xBytes = FloatBytes(x), outBytes = new byte[512];

        // vmcnt in order: the first of two global loads used with vmcnt(1), the second with vmcnt(0); SMEM with lgkmcnt(0).
        var (axpy, axpyText) = KernelOf(AxpyText);
        Check(Before(axpyText, "v_mul_f32") == "s_waitcnt vmcnt(1)" && Before(axpyText, "v_add_f32") == "s_waitcnt vmcnt(0)" && Before(axpyText, "global_load_dword") == "s_waitcnt lgkmcnt(0)",
            $"axpy's waits: vmcnt(1) before the first load's use, vmcnt(0) before the second's, lgkmcnt(0) before the scalar loads' first use:\n{axpyText}");
        _ = Storages(out ulong[] three, xBytes, xBytes, outBytes);
        var axpyKinds = RunWithoutEachWait("axpy", axpy, () => Storages(out _, xBytes, xBytes, outBytes), new GcnDispatch
        {
            WorkgroupSize = (64, 1, 1), Arguments = KernelArguments(28, (0, three[0], 8), (8, three[1], 8), (16, three[2], 8), (24, FloatBits(2f), 4)),
        });
        Check(axpyKinds.Count == 3 && axpyKinds.TrueForAll(k => k.Kind == GcnFailure.MissingWait), $"axpy: each of its 3 waits missed: {string.Join(", ", axpyKinds.Select(k => k.Kind))}");

        // LDS in order: two writes, two reads, the first read used with lgkmcnt(1).
        var (lds, ldsText) = KernelOf("""
            .kernel lds_order
            .arg out, 8, global
            .lds 512
            .reqd_workgroup_size 64, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_load_dwordx2 %s[0:1], s[0:1], 0x0
              v_mov_b32_e32 %v7, %s0
              v_lshlrev_b32_e32 %v0, 2, v0
              v_cvt_f32_u32_e32 %v1, v0
              v_add_f32_e32 %v2, 1.0, %v1
              ds_write_b32 %v0, %v1
              ds_write_b32 %v0, %v2 offset:256
              ds_read_b32 %v3, %v0
              ds_read_b32 %v4, %v0 offset:256
              v_mul_f32_e32 %v5, %v3, %v3
              v_add_f32_e32 %v6, %v5, %v4
              global_store_dword %v0, %v6, %s[0:1]
              s_endpgm
            """);
        Check(Before(ldsText, "v_mul_f32") == "s_waitcnt lgkmcnt(1)" && Before(ldsText, "v_add_f32", "", 1) == "s_waitcnt lgkmcnt(0)",
            $"LDS completes in order: lgkmcnt(1) for the first of two reads, lgkmcnt(0) for the second:\n{ldsText}");
        var memory = Storages(out ulong[] one, outBytes);
        var ldsDispatch = new GcnDispatch { WorkgroupSize = (64, 1, 1), Arguments = KernelArguments(8, (0, one[0], 8)) };
        GcnInterpreter.Run(ImageOf(lds), memory, ldsDispatch);
        float[] got = memory.Read<float>(one[0], 64);
        int wrong = FirstDifferent(got, i => (float)i * i + (i + 1f), 64);
        Check(wrong < 0, $"lds_order: out[{wrong}] is {(wrong < 0 ? 0 : got[wrong])}");
        var ldsKinds = RunWithoutEachWait("lds_order", lds, () => Storages(out _, outBytes), ldsDispatch);
        Check(ldsKinds.Count == 3 && ldsKinds.TrueForAll(k => k.Kind == GcnFailure.MissingWait), $"lds_order: each of its 3 waits missed: {string.Join(", ", ldsKinds.Select(k => k.Kind))}");

        // FLAT: counted on both counters, it may complete through either: its use waits both to 0.
        var (flat, flatText) = KernelOf("""
            .kernel flat_load
            .arg x, 8, global
            .arg out, 8, global
            .reqd_workgroup_size 64, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_load_dwordx4 %s[0:3], s[0:1], 0x0
              v_lshlrev_b32_e32 %v0, 2, v0
              v_mov_b32_e32 %v2, %s1
              v_add_co_u32_e32 %v1, vcc, %s0, %v0
              v_addc_co_u32_e32 %v2, vcc, 0, %v2, vcc
              flat_load_dword %v3, %v[1:2]
              v_add_f32_e32 %v4, 1.0, %v3
              global_store_dword %v0, %v4, %s[2:3]
              s_endpgm
            """);
        Check(Before(flatText, "v_add_f32") == "s_waitcnt vmcnt(0) lgkmcnt(0)", $"a FLAT load's use waits vmcnt(0) and lgkmcnt(0):\n{flatText}");
        memory = Storages(out ulong[] two, xBytes, outBytes);
        var twoDispatch = new GcnDispatch { WorkgroupSize = (64, 1, 1), Arguments = KernelArguments(16, (0, two[0], 8), (8, two[1], 8)) };
        GcnInterpreter.Run(ImageOf(flat), memory, twoDispatch);
        got = memory.Read<float>(two[1], 64);
        wrong = FirstDifferent(got, i => x[i] + 1f, 64);
        Check(wrong < 0, $"flat_load: out[{wrong}] is {(wrong < 0 ? 0 : got[wrong])}");
        var flatKinds = RunWithoutEachWait("flat_load", flat, () => Storages(out _, xBytes, outBytes), twoDispatch);
        Check(flatKinds.TrueForAll(k => k.Kind == GcnFailure.MissingWait), $"flat_load: each wait missed: {string.Join(", ", flatKinds.Select(k => k.Kind))}");

        // s_barrier: gfx9 does not wait for the wave's LDS writes itself, so lgkmcnt(0) goes first; without it, a race.
        var (barrier, barrierText) = KernelOf("""
            .kernel barrier
            .arg out, 8, global
            .lds 512
            .reqd_workgroup_size 128, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_load_dwordx2 %s[0:1], s[0:1], 0x0
              v_lshlrev_b32_e32 %v0, 2, v0
              v_cvt_f32_u32_e32 %v1, v0
              ds_write_b32 %v0, %v1
              v_xor_b32_e32 %v2, 0x100, %v0
              s_barrier
              ds_read_b32 %v3, %v2
              global_store_dword %v0, %v3, %s[0:1]
              s_endpgm
            """);
        Check(Before(barrierText, "s_barrier") == "s_waitcnt lgkmcnt(0)", $"lgkmcnt(0) before s_barrier:\n{barrierText}");
        memory = Storages(out one, outBytes);
        var barrierDispatch = new GcnDispatch { WorkgroupSize = (128, 1, 1), Arguments = KernelArguments(8, (0, one[0], 8)) };
        GcnInterpreter.Run(ImageOf(barrier), memory, barrierDispatch);
        got = memory.Read<float>(one[0], 128);
        wrong = FirstDifferent(got, i => i ^ 64, 128);
        Check(wrong < 0, $"barrier: out[{wrong}] is {(wrong < 0 ? 0 : got[wrong])}, not {wrong ^ 64}");
        var barrierKinds = RunWithoutEachWait("barrier", barrier, () => Storages(out _, outBytes), barrierDispatch);
        Check(barrierKinds is [{ Kind: GcnFailure.MissingWait } first, { Kind: GcnFailure.MissingWait }] && first.Message.Contains("before the s_barrier", StringComparison.Ordinal),
            $"barrier: without the barrier's wait, the other wave reads an LDS store no wait retired before the s_barrier; without the store's, a missing wait: {string.Join("; ", barrierKinds.Select(k => k.Message))}");

        // Across blocks: a scalar load of a block laid out later, used after the branch (the partial kernel), and a load
        // at a loop's end used at its top on the next turn: one wait at the loop's start serves both ways in.
        var (partial, partialText) = KernelOf(PartialText);
        Check(Before(partialText, "global_store_dword") == "s_waitcnt lgkmcnt(0)", $"the store waits for its predecessor's scalar load:\n{partialText}");
        var partialKinds = RunWithoutEachWait("partial", partial, () => Storages(out _, outBytes), ldsDispatch);
        Check(partialKinds is [{ Kind: GcnFailure.MissingWait }], $"partial: its wait missed: {string.Join(", ", partialKinds.Select(k => k.Kind))}");
        var (loop, loopText) = KernelOf("""
            .kernel loop_carry
            .arg x, 8, global
            .arg out, 8, global
            .arg k, 4, value
            .reqd_workgroup_size 64, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_load_dwordx4 %s[0:3], s[0:1], 0x0
              s_load_dword %s4, s[0:1], 0x10
              v_lshlrev_b32_e32 %v0, 2, v0
              v_mov_b32_e32 %v1, 0
              global_load_dword %v2, %v0, %s[0:1]
              s_mov_b32 %s5, %s4
            .Lloop:
              v_add_f32_e32 %v1, %v1, %v2
              global_load_dword %v2, %v0, %s[0:1]
              s_add_u32 %s5, %s5, -1
              s_cmp_lg_u32 %s5, 0
              s_cbranch_scc1 .Lloop
              global_store_dword %v0, %v1, %s[2:3]
              s_endpgm
            """);
        var loopLines = PrintedLines(loopText);
        Check(loopLines.Count(l => l == "s_waitcnt vmcnt(0)") == 1 && loopLines[Array.IndexOf(loopLines, ".Lloop:") + 1] == "s_waitcnt vmcnt(0)",
            $"one vmcnt(0) at the loop's top, for the load before it and the one at its end:\n{loopText}");
        memory = Storages(out two, xBytes, outBytes);
        var loopDispatch = new GcnDispatch { WorkgroupSize = (64, 1, 1), Arguments = KernelArguments(20, (0, two[0], 8), (8, two[1], 8), (16, 5, 4)) };
        GcnInterpreter.Run(ImageOf(loop), memory, loopDispatch);
        got = memory.Read<float>(two[1], 64);
        wrong = FirstDifferent(got, i => 0 + x[i] + x[i] + x[i] + x[i] + x[i], 64);
        Check(wrong < 0, $"loop_carry: out[{wrong}] is {(wrong < 0 ? 0 : got[wrong])}");
        var loopKinds = RunWithoutEachWait("loop_carry", loop, () => Storages(out _, xBytes, outBytes), loopDispatch);
        Check(loopKinds.TrueForAll(k => k.Kind == GcnFailure.MissingWait), $"loop_carry: each wait missed: {string.Join(", ", loopKinds.Select(k => k.Kind))}");

        // A wait the text writes is kept and counted: no second one is placed.
        var (_, writtenText) = KernelOf("""
            .kernel written
            .arg x, 8, global
            .reqd_workgroup_size 64, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_load_dwordx2 %s[0:1], s[0:1], 0x0
              s_waitcnt lgkmcnt(0)
              v_lshlrev_b32_e32 %v0, 2, v0
              global_load_dword %v1, %v0, %s[0:1]
              s_waitcnt vmcnt(0)
              v_add_f32_e32 %v2, 1.0, %v1
              global_store_dword %v0, %v2, %s[0:1]
              s_endpgm
            """);
        Check(PrintedLines(writtenText).Count(l => l.StartsWith("s_waitcnt", StringComparison.Ordinal)) == 2, $"the text's two waits, no more:\n{writtenText}");

        // GDS holds its data registers on expcnt: overwriting one waits expcnt(0) (the interpreter models no GDS).
        var (_, gdsText) = KernelOf("""
            .kernel gds
            .reqd_workgroup_size 64, 1, 1
              s_mov_b32 m0, -1
              v_mov_b32_e32 %v0, 0
              v_mov_b32_e32 %v1, 1
              ds_add_u32 %v0, %v1 gds
              v_mov_b32_e32 %v1, 2
              ds_add_u32 %v0, %v1 gds
              s_endpgm
            """);
        Check(Before(gdsText, "v_mov_b32_e32", ", 2") == "s_waitcnt expcnt(0)", $"expcnt(0) before a GDS operation's data register is written:\n{gdsText}");
    }

    // A kernel around `body`: block 64, the arguments' pointer in s[0:1], the work-item id in v0.
    private static string HazardKernel(string body) => $".kernel h\n.reqd_workgroup_size 64, 1, 1\n.user_sgpr kernarg_segment_ptr\n{body}\n  s_endpgm\n";

    internal static void GcnAssemblerHazards(Device device)
    {
        _ = device;
        var xnackOff = GcnIsaVersion.Parse("gfx90c:xnack-");
        (string Case, string Body, string Line, string Suffix, int Occurrence, string Expected, GcnIsaVersion? Isa)[] cases =
        [
            ("VMEM reads an SGPR a VALU wrote", "  v_readfirstlane_b32 %s4, v0\n  v_readfirstlane_b32 %s5, v0\n  global_load_dword %v1, v0, %s[4:5]", "global_load_dword", "", 0, "s_nop 4", null),
            ("VMEM after v_cmpx wrote EXEC", "  v_cmpx_gt_u32_e32 vcc, 32, v0\n  global_store_dword v0, v0, s[0:1]", "global_store_dword", "", 0, "s_nop 4", null),
            ("v_readlane's lane select a VALU wrote", "  v_mov_b32_e32 %v1, 1.0\n  v_readfirstlane_b32 %s0, v0\n  v_readlane_b32 %s1, %v1, %s0", "v_readlane_b32", "", 0, "s_nop 3", null),
            ("v_writelane's lane select a VALU wrote", "  v_mov_b32_e32 %v1, 1.0\n  v_readfirstlane_b32 %s0, v0\n  v_writelane_b32 %v1, 7, %s0", "v_writelane_b32", "", 0, "s_nop 3", null),
            ("v_div_fmas after a VALU wrote VCC", "  v_mov_b32_e32 %v1, 1.0\n  v_mov_b32_e32 %v2, 2.0\n  v_div_scale_f32 %v3, vcc, %v1, %v2, %v1\n  v_div_fmas_f32 %v4, %v1, %v2, %v3", "v_div_fmas_f32", "", 0, "s_nop 3", null),
            ("DPP reads a VGPR a VALU wrote", "  v_add_u32_e32 %v1, 1, v0\n  v_mov_b32_dpp %v2, %v1 row_shr:1 bound_ctrl:0", "v_mov_b32_dpp", "", 0, "s_nop 1", null),
            ("DPP after v_cmpx wrote EXEC", "  v_add_u32_e32 %v1, 1, v0\n  s_nop 4\n  v_cmpx_gt_u32_e32 vcc, 32, v0\n  v_mov_b32_dpp %v2, %v1 row_shr:1 bound_ctrl:0", "v_mov_b32_dpp", "", 0, "s_nop 4", null),
            ("s_getreg after s_setreg of its register", "  s_mov_b32 %s0, 0\n  s_setreg_b32 hwreg(HW_REG_MODE), %s0\n  s_getreg_b32 %s1, hwreg(HW_REG_MODE)", "s_getreg_b32", "", 0, "s_nop 1", null),
            ("s_setreg after s_setreg of its register", "  s_mov_b32 %s0, 0\n  s_setreg_b32 hwreg(HW_REG_MODE), %s0\n  s_setreg_b32 hwreg(HW_REG_MODE), %s0", "s_setreg_b32", "", 1, "s_nop 1", null),
            ("s_getreg after s_setreg_imm32", "  s_setreg_imm32_b32 hwreg(HW_REG_MODE), 0\n  s_getreg_b32 %s1, hwreg(HW_REG_MODE)", "s_getreg_b32", "", 0, "s_nop 1", null),
            ("a VALU writes a wide store's data", "  v_mov_b32_e32 %v4, 1.0\n  v_mov_b32_e32 %v5, 1.0\n  v_mov_b32_e32 %v6, 1.0\n  v_mov_b32_e32 %v7, 1.0\n  global_store_dwordx4 v0, %v[4:7], s[0:1]\n  v_mov_b32_e32 %v5, 2.0\n  global_store_dwordx4 v0, %v[4:7], s[0:1] offset:16", "v_mov_b32_e32", ", 2.0", 0, "s_nop 0", null),
            ("GDS after an SALU wrote M0", "  v_mov_b32_e32 %v0, 0\n  v_mov_b32_e32 %v1, 1\n  s_mov_b32 m0, -1\n  ds_add_u32 %v0, %v1 gds", "ds_add_u32", "", 0, "s_nop 0", null),
            ("an SMEM clause's write meets its read (xnack any)", "  s_load_dwordx2 %s[2:3], s[0:1], 0x0\n  s_load_dwordx2 %s[4:5], s[0:1], 0x8\n  v_mov_b32_e32 %v1, %s2\n  v_mov_b32_e32 %v2, %s4", "s_load_dwordx2", "", 1, "s_nop 0", null),
            ("a vector memory clause's write meets its read (xnack any)", "  v_mov_b32_e32 %v0, 0\n  v_mov_b32_e32 %v1, 0\n  global_load_dword %v2, %v[0:1], off\n  global_load_dword %v3, %v[0:1], off offset:4\n  v_add_f32_e32 %v4, %v2, %v3", "global_load_dword", "", 1, "s_nop 0", null),
            ("a store after a load in a clause (xnack any)", "  global_load_dword %v2, v0, s[0:1]\n  global_store_dword v0, v0, s[0:1] offset:4", "global_store_dword", "", 0, "s_nop 0", null),
            ("a wait is a wait state", "  global_load_dword %v2, v0, s[0:1]\n  v_readfirstlane_b32 %s4, v0\n  v_readfirstlane_b32 %s5, v0\n  global_store_dword v0, %v2, %s[4:5]", "global_store_dword", "", 0, "s_nop 3", null),
            ("the nearest write on any path", "  v_readfirstlane_b32 %s4, v0\n  v_readfirstlane_b32 %s5, v0\n  s_cmp_eq_u32 %s4, 0\n  s_cbranch_scc1 .Lskip\n  s_mov_b32 %s6, 1\n.Lskip:\n  global_load_dword %v1, v0, %s[4:5]", "global_load_dword", "", 0, "s_nop 2", null),
        ];
        foreach (var (what, body, line, suffix, occurrence, expected, isa) in cases)
        {
            var (_, printed) = KernelOf(HazardKernel(body), isa);
            Check(Before(printed, line, suffix, occurrence) == expected, $"{what}: '{expected}' before {line}, as LLVM inserts it; printed:\n{printed}");
        }

        var (_, waited) = KernelOf(HazardKernel(cases[^2].Body));
        Check(Before(waited, "s_nop 3") == "s_waitcnt vmcnt(0)", $"the wait comes first and counts as one of the 5 wait states:\n{waited}");

        // Where LLVM inserts nothing on gfx9: an SALU's SGPR read by VMEM, an SALU's lane select, an SALU's EXEC before DPP,
        // a VALU's VGPR read by v_readfirstlane, a VALU's SGPR read by an SALU, another hardware register, a 64-bit store's
        // data, and the soft clauses with xnack off.
        (string Case, string Body, string Line, int Occurrence, GcnIsaVersion? Isa)[] none =
        [
            ("VMEM reads an SGPR an SALU wrote", "  s_mov_b32 %s4, 0\n  s_mov_b32 %s5, 0\n  global_load_dword %v1, v0, %s[4:5]", "global_load_dword", 0, null),
            ("v_readlane's lane select an SALU wrote", "  v_mov_b32_e32 %v1, 1.0\n  s_mov_b32 %s0, 3\n  v_readlane_b32 %s1, %v1, %s0", "v_readlane_b32", 0, null),
            ("DPP after an SALU wrote EXEC", "  v_add_u32_e32 %v1, 1, v0\n  s_nop 4\n  s_mov_b64 exec, -1\n  v_mov_b32_dpp %v2, %v1 row_shr:1 bound_ctrl:0", "v_mov_b32_dpp", 0, null),
            ("v_readfirstlane reads a VGPR a VALU wrote", "  v_add_u32_e32 %v1, 1, v0\n  v_readfirstlane_b32 %s0, %v1", "v_readfirstlane_b32", 0, null),
            ("an SALU reads an SGPR a VALU wrote", "  v_readfirstlane_b32 %s0, v0\n  s_add_u32 %s1, %s0, 1", "s_add_u32", 0, null),
            ("s_getreg of another register", "  s_mov_b32 %s0, 0\n  s_setreg_b32 hwreg(HW_REG_MODE), %s0\n  s_getreg_b32 %s1, hwreg(HW_REG_STATUS)", "s_getreg_b32", 0, null),
            ("a VALU writes a 64-bit store's data", "  v_mov_b32_e32 %v4, 1.0\n  v_mov_b32_e32 %v5, 1.0\n  global_store_dwordx2 v0, %v[4:5], s[0:1]\n  v_mov_b32_e32 %v5, 2.0\n  global_store_dwordx2 v0, %v[4:5], s[0:1] offset:8", "v_mov_b32_e32", 2, null),
            ("an SMEM clause, xnack off", "  s_load_dwordx2 %s[2:3], s[0:1], 0x0\n  s_load_dwordx2 %s[4:5], s[0:1], 0x8\n  v_mov_b32_e32 %v1, %s2\n  v_mov_b32_e32 %v2, %s4", "s_load_dwordx2", 1, xnackOff),
            ("a vector memory clause, xnack off", "  v_mov_b32_e32 %v0, 0\n  v_mov_b32_e32 %v1, 0\n  global_load_dword %v2, %v[0:1], off\n  global_load_dword %v3, %v[0:1], off offset:4\n  v_add_f32_e32 %v4, %v2, %v3", "global_load_dword", 1, xnackOff),
        ];
        foreach (var (what, body, line, occurrence, isa) in none)
        {
            var (_, printed) = KernelOf(HazardKernel(body), isa);
            string before = Before(printed, line, "", occurrence);
            Check(!before.StartsWith("s_nop", StringComparison.Ordinal) && !before.StartsWith("(no line", StringComparison.Ordinal), $"{what}: no s_nop before {line} ({before}); printed:\n{printed}");
        }
    }

    internal static void GcnAssemblerPrinter(Device device)
    {
        _ = device;
        var golden = ReadFormsGolden();
        var counts = new List<string>();
        foreach (string cpu in GcnRegisteredCpus)
        {
            var isa = GcnIsaVersion.Parse(cpu);
            var target = GcnTargets.For(isa);
            var block = golden[cpu];
            byte[] expected = [.. block.Records.SelectMany(r => r.Bytes)];
            using var asm = new GcnAssembler(target, isa, 2 * expected.Length) { Printing = true };
            asm.Assemble(Encoding.UTF8.GetBytes(string.Join('\n', block.Records.Select(r => r.Text))));
            byte[] first = asm.ToArray();
            string printed = asm.PrintedText;
            Check(first.AsSpan().SequenceEqual(expected), $"{cpu}: the block assembles to llvm-mc's bytes");
            using var again = new GcnAssembler(target, isa, 2 * expected.Length);
            again.Assemble(Encoding.UTF8.GetBytes(printed));
            byte[] second = again.ToArray();
            int differs = first.AsSpan().CommonPrefixLength(second);
            Check(second.AsSpan().SequenceEqual(first), $"{cpu}: the printed forms read back to other bytes from byte {differs}: {PrintedNear(printed, block.Records, differs)}");
            counts.Add($"{cpu} {block.Records.Count(r => r.Bytes.Length > 0)}");
        }

        Console.WriteLine($"    forms printed and read back: {string.Join(", ", counts)}");
    }

    // The golden record whose bytes hold offset `at`, and the printed line for it.
    private static string PrintedNear(string printed, List<(string Text, byte[] Bytes)> records, int at)
    {
        int offset = 0, index = 0;
        var lines = PrintedLines(printed);
        foreach (var (text, bytes) in records)
        {
            if (offset + bytes.Length > at && bytes.Length > 0)
            {
                return $"'{text}' printed as '{(index < lines.Length ? lines[index] : "?")}'";
            }

            offset += bytes.Length;
            index++;
        }

        return "(past the end)";
    }

    // A kernel of `parts` repeated parts: a load, a compare and a narrowed add, a restore, a store.
    private static byte[] RepeatedKernel(int parts)
    {
        var text = new StringBuilder("""
            .kernel repeated
            .arg x, 8, global
            .arg out, 8, global
            .reqd_workgroup_size 64, 1, 1
            .user_sgpr kernarg_segment_ptr
              s_load_dwordx4 %s[0:3], s[0:1], 0x0
              v_lshlrev_b32_e32 %v0, 2, v0

            """);
        for (int p = 0; p < parts; p++)
        {
            int v = 10 + 3 * p, s = 10 + 2 * p;
            text.Append(CultureInfo.InvariantCulture, $"""
                  global_load_dword %v{v}, %v0, %s[0:1] offset:{4 * p}
                  v_mul_f32_e32 %v{v + 1}, 2.0, %v{v}
                  v_cmp_lt_f32_e32 vcc, 0, %v{v + 1}
                  s_and_saveexec_b64 %s[{s}:{s + 1}], vcc
                  s_cbranch_execz .Lskip{p}
                  v_add_f32_e32 %v{v + 1}, 1.0, %v{v + 1}
                .Lskip{p}:
                  s_or_b64 exec, exec, %s[{s}:{s + 1}]
                  global_store_dword %v0, %v{v + 1}, %s[2:3] offset:{4 * p}

                """);
        }

        return Encoding.UTF8.GetBytes(text.Append("  s_endpgm\n").ToString());
    }

    internal static void GcnAssemblerKernelAllocations(Device device)
    {
        _ = device;
        byte[] small = RepeatedKernel(8), large = RepeatedKernel(32);
        using var asm = new GcnAssembler(GcnTargets.For(KernelIsa), KernelIsa);
        int smallCode = 0, largeCode = 0;
        void Small() => smallCode = asm.AssembleKernel(small).Code.Length;
        void Large() => largeCode = asm.AssembleKernel(large).Code.Length;
        for (int warm = 0; warm < 40; warm++)
        {
            Large();
            Small();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        Small();
        long smallBytes = GC.GetAllocatedBytesForCurrentThread() - before;
        before = GC.GetAllocatedBytesForCurrentThread();
        Large();
        long largeBytes = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(largeBytes - smallBytes <= largeCode - smallCode + 64,
            $"four times the instructions ({largeCode} bytes of code, not {smallCode}) allocated {largeBytes} bytes, not {smallBytes}: more than the code's {largeCode - smallCode}{(largeBytes - smallBytes > largeCode - smallCode + 64 ? $" ({AllocatedTypes(Large)})" : "")}");
        Check(smallBytes <= smallCode + 2048, $"a kernel allocates its code, name, arguments and record: {smallBytes} bytes for {smallCode} bytes of code{(smallBytes > smallCode + 2048 ? $" ({AllocatedTypes(Small)})" : "")}");
    }

    private static object Value(List<KeyValuePair<string, object>> map, string key) => map.First(e => e.Key == key).Value;

    // A MessagePack reader for what the metadata holds: maps (as ordered pairs), arrays, strings, unsigned integers.
    private static object Unpack(ReadOnlySpan<byte> data, ref int at)
    {
        byte first = data[at++];
        switch (first)
        {
            case <= 0x7F:
                return (ulong)first;
            case >= 0x80 and <= 0x8F:
                return UnpackMap(data, ref at, first & 0x0F);
            case >= 0x90 and <= 0x9F:
                return UnpackArray(data, ref at, first & 0x0F);
            case >= 0xA0 and <= 0xBF:
                return UnpackString(data, ref at, first & 0x1F);
            case 0xCC:
                return (ulong)data[at++];
            case 0xCD:
                at += 2;
                return (ulong)BinaryPrimitives.ReadUInt16BigEndian(data[(at - 2)..]);
            case 0xCE:
                at += 4;
                return (ulong)BinaryPrimitives.ReadUInt32BigEndian(data[(at - 4)..]);
            case 0xD9:
                return UnpackString(data, ref at, data[at++]);
            case 0xDC:
                at += 2;
                return UnpackArray(data, ref at, BinaryPrimitives.ReadUInt16BigEndian(data[(at - 2)..]));
            case 0xDE:
                at += 2;
                return UnpackMap(data, ref at, BinaryPrimitives.ReadUInt16BigEndian(data[(at - 2)..]));
            default:
                throw new InvalidDataException($"MessagePack byte 0x{first:X2} at {at - 1} is not one the metadata uses");
        }
    }

    private static List<KeyValuePair<string, object>> UnpackMap(ReadOnlySpan<byte> data, ref int at, int count)
    {
        var map = new List<KeyValuePair<string, object>>(count);
        for (int i = 0; i < count; i++)
        {
            string key = (string)Unpack(data, ref at);
            map.Add(new(key, Unpack(data, ref at)));
        }

        return map;
    }

    private static List<object> UnpackArray(ReadOnlySpan<byte> data, ref int at, int count)
    {
        var items = new List<object>(count);
        for (int i = 0; i < count; i++)
        {
            items.Add(Unpack(data, ref at));
        }

        return items;
    }

    private static string UnpackString(ReadOnlySpan<byte> data, ref int at, int length)
    {
        string text = Encoding.UTF8.GetString(data.Slice(at, length));
        at += length;
        return text;
    }

    private sealed record ElfSection(string Name, uint Type, ulong Flags, ulong Address, ulong Offset, ulong Size, uint Link, uint Info, ulong Align, ulong EntrySize);

    private sealed record ElfSegment(uint Type, uint Flags, ulong Offset, ulong Address, ulong FileSize, ulong MemorySize, ulong Align);

    private sealed record ElfSymbol(string Name, byte Info, byte Other, ushort Section, ulong Value, ulong Size);

    // An ELF64 little-endian reader for what a code object holds: headers, sections, segments, symbols, the dynamic and
    // hash tables. Written from the System V ABI, apart from the writer.
    private sealed class ElfReader
    {
        private readonly byte[] _image;

        public ElfReader(byte[] image)
        {
            _image = image;
            int sectionHeaders = (int)U64(40), sections = U16(60), names = U16(62);
            var raw = Enumerable.Range(0, sections).Select(i => sectionHeaders + i * 64).ToArray();
            int namesAt = (int)U64(raw[names] + 24);
            Sections = [.. raw.Select(h => new ElfSection(String(namesAt + (int)U32(h)), U32(h + 4), U64(h + 8), U64(h + 16), U64(h + 24), U64(h + 32), U32(h + 40), U32(h + 44), U64(h + 48), U64(h + 56)))];
            Segments = [.. Enumerable.Range(0, U16(56)).Select(i => (int)U64(32) + i * 56)
                .Select(h => new ElfSegment(U32(h), U32(h + 4), U64(h + 8), U64(h + 16), U64(h + 32), U64(h + 40), U64(h + 48)))];
        }

        public IReadOnlyList<ElfSection> Sections { get; }

        public IReadOnlyList<ElfSegment> Segments { get; }

        public ushort U16(int at) => BinaryPrimitives.ReadUInt16LittleEndian(_image.AsSpan(at));

        public ReadOnlySpan<byte> Bytes(int at, int count) => _image.AsSpan(at, count);

        public byte[] SectionBytes(string name) => _image.AsSpan((int)Section(name).Offset, (int)Section(name).Size).ToArray();

        public uint U32(int at) => BinaryPrimitives.ReadUInt32LittleEndian(_image.AsSpan(at));

        public ulong U64(int at) => BinaryPrimitives.ReadUInt64LittleEndian(_image.AsSpan(at));

        public int IndexOf(string name) => Sections.ToList().FindIndex(s => s.Name == name);

        public ElfSection Section(string name) => Sections.Single(s => s.Name == name);

        public List<ElfSymbol> Symbols(string table)
        {
            var section = Section(table);
            int namesAt = (int)Sections[(int)section.Link].Offset;
            return [.. Enumerable.Range(0, (int)(section.Size / 24)).Select(i => (int)section.Offset + i * 24)
                .Select(s => new ElfSymbol(String(namesAt + (int)U32(s)), _image[s + 4], _image[s + 5], U16(s + 6), U64(s + 8), U64(s + 16)))];
        }

        public List<(long Tag, long Value)> Dynamic()
        {
            var section = Section(".dynamic");
            return [.. Enumerable.Range(0, (int)(section.Size / 16)).Select(i => ((long)U64((int)section.Offset + i * 16), (long)U64((int)section.Offset + i * 16 + 8)))];
        }

        // The dynamic symbol named so, through the System V hash table; -1 when there is none.
        public int Lookup(string name)
        {
            var hash = Section(".hash");
            var symbols = Symbols(".dynsym");
            int at = (int)hash.Offset;
            uint buckets = U32(at);
            uint h = 0;
            foreach (byte c in Encoding.UTF8.GetBytes(name))
            {
                h = (h << 4) + c;
                uint g = h & 0xF0000000;
                h = (h ^ (g >> 24)) & ~g;
            }

            for (uint i = U32(at + 8 + (int)(h % buckets) * 4); i != 0; i = U32(at + 8 + (int)buckets * 4 + (int)i * 4))
            {
                if (symbols[(int)i].Name == name)
                {
                    return (int)i;
                }
            }

            return -1;
        }

        private string String(int at) => Encoding.UTF8.GetString(_image, at, Array.IndexOf(_image, (byte)0, at) - at);
    }
}
