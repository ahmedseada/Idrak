// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// The GCN interpreter on the RDNA families (plan 35, G8a): the decoder against llvm-mc 18's text for every wave64 golden
// form of gfx1010, gfx1030, gfx1100 and gfx1102, and a kernel of Idrak's own gfx10 and gfx11 encoders run in the
// interpreter (its stores on vscnt, its LDS across a barrier), apart from the encoders as the gfx9 tests are.
internal abstract partial class Tests
{
    private static readonly (string Family, string Block, GcnCpu Cpu)[] RdnaInterpreterCpus =
    [
        ("gfx10", "gfx1010-w64", GcnCpu.Gfx1010), ("gfx10", "gfx1030-w64", GcnCpu.Gfx1030),
        ("gfx11", "gfx1100-w64", GcnCpu.Gfx1100), ("gfx11", "gfx1102-w64", GcnCpu.Gfx1102),
    ];

    /// <summary>
    /// The family the kernel tests assemble <paramref name="isa"/> with: the registered one, else the RDNA family made
    /// but not registered yet (plan 35, G8a: registered once all its kernels run), so the kernels are proven on it first.
    /// </summary>
    internal static IGcnTarget KernelTarget(GcnIsaVersion isa) =>
        GcnTargets.TryFor(isa, out var target) ? target
        : isa.Major == 10 && GcnTargets.Gfx10Family.Handles(isa) ? GcnTargets.Gfx10Family
        : isa.Major == 11 && GcnTargets.Gfx11Family.Handles(isa) ? GcnTargets.Gfx11Family
        : GcnTargets.For(isa);

    // Every kernel of GcnKernels.All, one text in gfx9's names, assembled by the gfx10 and gfx11 families for each of
    // their CPUs: the printed text (the family's own names, waits and hazard fixes) reads back to the same bytes, and the
    // kernels make one code object the interpreter reads back for its CPU.
    internal static void GcnKernelsRdnaAssembled(Device device)
    {
        _ = device;
        var all = GcnKernels.All;
        foreach (var (processor, cpu) in (ReadOnlySpan<(string, GcnCpu)>)[("gfx1010", GcnCpu.Gfx1010), ("gfx1030", GcnCpu.Gfx1030), ("gfx1100", GcnCpu.Gfx1100), ("gfx1102", GcnCpu.Gfx1102)])
        {
            var isa = GcnIsaVersion.Parse(processor);
            var target = KernelTarget(isa);
            var codes = new List<GcnKernelCode>(all.Count);
            var wrong = new List<string>();
            var offsets = new List<string>();
            for (int k = 0; k < all.Count; k++)
            {
                using var asm = new GcnAssembler(target, isa);
                try
                {
                    var code = asm.AssembleKernel(all[k].Text);
                    codes.Add(code);
                    byte[] again = AssembleGcn(target, isa, asm.PrintedText);
                    if (!again.AsSpan().SequenceEqual(code.Code.Span))
                    {
                        wrong.Add($"{all[k].Name}: its printed text reads back to other bytes");
                    }
                }
                catch (GcnSyntaxException e) when (target.Family == "gfx10" && e.Message.Contains("the offset is -2048 to 2047", StringComparison.Ordinal))
                {
                    offsets.Add(all[k].Name);
                }
                catch (Exception e) when (e is GcnSyntaxException or GcnAllocationException or NotSupportedException)
                {
                    wrong.Add($"{all[k].Name}: {e.Message}");
                }
            }

            // KNOWN GAP (plan 35, G8a, left): the row kernels' global offsets reach 3072 (four loads 1,024 bytes apart),
            // past ISA 10's 12-bit signed offset (±2047; gfx9 and ISA 11 take ±4095), so on gfx10 they are refused, and
            // only those, for that reason alone; the row and decoding kernel tests run on gfx1100 until they are fixed.
            Check(wrong.Count == 0, $"{processor} ({target.Family}): {wrong.Count} of {all.Count} kernels:\n  " + string.Join("\n  ", wrong.Take(20)));
            Check(target.Family == "gfx10" ? offsets.Count is > 0 and < 40 : offsets.Count == 0, $"{processor}: refused for ISA 10's offsets: {string.Join(", ", offsets)}");
            Console.WriteLine($"    {processor}: {codes.Count} of {all.Count} kernels assembled{(offsets.Count > 0 ? $"; {offsets.Count} refused for ISA 10's ±2047 global offsets: {string.Join(", ", offsets)}" : "")}");
            byte[] codeObject = GcnCodeObject.Write(target, isa, 5, [.. codes]);
            foreach (var code in codes)
            {
                var image = GcnCodeObjectReader.Read(codeObject, code.Name);
                Check(image.Cpu == cpu && image.Code.Length > 0, $"{processor}: {code.Name} read back for {image.Cpu}");
            }
        }
    }

    internal static void GcnInterpreterRdnaForms(Device device)
    {
        _ = device;
        var counts = new List<string>();
        foreach (var (family, block, cpu) in RdnaInterpreterCpus)
        {
            var records = ReadFormsGolden(family)[block].Records;
            var labels = new Dictionary<string, int>(StringComparer.Ordinal);
            int offset = 0;
            foreach (var (text, bytes) in records)
            {
                if (bytes.Length == 0)
                {
                    labels[text.TrimEnd(':')] = offset;
                }
                else
                {
                    offset += bytes.Length;
                }
            }

            var decoder = new GcnDecoder(cpu);
            var wrong = new List<string>();
            int decoded = 0, twins = 0;
            offset = 0;
            byte[]? previous = null;
            string previousText = "";
            foreach (var (text, bytes) in records)
            {
                if (bytes.Length == 0)
                {
                    continue;
                }

                string expected = Resolve(text, labels, offset);
                offset += bytes.Length;

                // forms.s follows a wave64 form with its wave32 spelling where that differs (vcc_lo for vcc, s0 for
                // s[0:1]); wave64 takes both to the same bytes, which the disassembler prints the wave64 way.
                bool twin = previous is not null && bytes.AsSpan().SequenceEqual(previous) && expected != previousText;
                string want = twin ? previousText : expected;
                try
                {
                    var instruction = decoder.Decode(bytes, 0);
                    if (instruction.Size != bytes.Length || instruction.Text != want)
                    {
                        wrong.Add($"{Convert.ToHexStringLower(bytes)}: '{instruction.Text}' ({instruction.Size} bytes) where llvm-mc printed '{want}' ({bytes.Length})");
                    }
                    else
                    {
                        decoded++;
                        twins += twin ? 1 : 0;
                    }
                }
                catch (GcnDecodeException e)
                {
                    wrong.Add($"{Convert.ToHexStringLower(bytes)} ('{want}'): refused: {e.Message}");
                }

                if (!twin)
                {
                    previous = bytes;
                    previousText = expected;
                }
            }

            int forms = records.Count(r => r.Bytes.Length > 0);
            Check(wrong.Count == 0, $"{block}: {wrong.Count} of {forms} forms differ:\n  " + string.Join("\n  ", wrong.Take(30)));
            Check(decoded == forms && forms > 380 && twins > 10, $"{block}: {decoded} of {forms} forms decoded ({twins} wave32 spellings among them)");
            counts.Add($"{block} {decoded} ({twins} wave32 spellings)");
        }

        Console.WriteLine($"    forms decoded: {string.Join(", ", counts)}");

        // Refused, naming the offset and the words: DPP and SDWA (no kernel of the families uses them), an MTBUF word, a
        // VOP3 literal cut off, and an opcode one CPU of a family lacks (v_dot2_f32_f16 is gfx1030's, not gfx1010's).
        var gfx1030 = new GcnDecoder(GcnCpu.Gfx1030);
        ThrowsNaming<GcnDecodeException>(() => gfx1030.Decode(Words([0x7E0002FA, 0xFF000000]), 0), "DPP on gfx1030", "7E0002FA", "DPP");
        ThrowsNaming<GcnDecodeException>(() => new GcnDecoder(GcnCpu.Gfx1100).Decode(Words([0x7E0002F9, 0x00060606]), 0), "SDWA on gfx1100", "SDWA");
        ThrowsNaming<GcnDecodeException>(() => gfx1030.Decode(Words([0xE8000000, 0x00000000]), 0), "MTBUF", "E8000000", "MTBUF");
        ThrowsNaming<GcnDecodeException>(() => gfx1030.Decode(Words([0xD5030000, 0x000202FF]), 0), "a VOP3 literal cut off", "literal");
        byte[] dot2 = Words([0xCC134000, 0x1C0A0501]);
        Check(GcnOpcodeTable.Find(GcnCpu.Gfx1030, Idrak.Tests.Gcn.GcnFormat.Vop3p, 0x13)?.Name == "v_dot2_f32_f16" && GcnOpcodeTable.Find(GcnCpu.Gfx1010, Idrak.Tests.Gcn.GcnFormat.Vop3p, 0x13) is null,
            "v_dot2_f32_f16 is gfx1030's and not gfx1010's");
        ThrowsNaming<GcnDecodeException>(() => new GcnDecoder(GcnCpu.Gfx1010).Decode(dot2, 0), "v_dot2_f32_f16 on gfx1010", "0x13", "gfx1010");

        // A VOP3 literal (RDNA's third word), and the renamed mnemonics read as gfx9's for their semantics.
        var literal = gfx1030.Decode(Words([0xD5030000, 0x0001FF01, 0x3E800000]), 0);
        Check(literal.Size == 12 && literal.Text == "v_add_f32_e64 v0, v1, 0x3e800000", $"a VOP3 literal: '{literal.Text}' ({literal.Size} bytes)");
        foreach (var (stem, gfx9) in (ReadOnlySpan<(string, string)>)[
                     ("v_add_nc_u32", "v_add_u32"), ("s_load_b128", "s_load_dwordx4"), ("global_load_b32", "global_load_dword"), ("global_store_b128", "global_store_dwordx4"),
                     ("global_load_u8", "global_load_ubyte"), ("global_store_b16", "global_store_short"), ("global_atomic_add_u32", "global_atomic_add"),
                     ("ds_load_b128", "ds_read_b128"), ("ds_store_2addr_b32", "ds_write2_b32"), ("ds_load_2addr_stride64_b32", "ds_read2st64_b32"),
                     ("s_and_not1_b64", "s_andn2_b64"), ("s_and_not1_saveexec_b64", "s_andn2_saveexec_b64"), ("s_mov_b32", "s_mov_b32")])
        {
            Check(GcnOpcodeTable.Gfx9Stem(stem) == gfx9, $"{stem} reads as {GcnOpcodeTable.Gfx9Stem(stem)}, not {gfx9}");
        }

        var unmodelled = RdnaInterpreterCpus.SelectMany(c => ReadFormsGolden(c.Family)[c.Block].Records.Where(r => r.Bytes.Length > 0)
                .Select(r => (c.Block, Instruction: new GcnDecoder(c.Cpu).Decode(r.Bytes, 0))))
            .Where(r => r.Instruction.Opcode.Execute is null).Select(r => $"{r.Block} {r.Instruction.Opcode.Name}").Distinct().ToList();
        Check(unmodelled.Count == 0, $"every golden form's opcode has semantics; these have none: {string.Join(", ", unmodelled)}");
    }

    // A kernel in each family's names: two loads around a store (the store on vscnt, so the first load's use waits
    // vmcnt(1)), a product and a sum, an LDS write read back across a barrier by the neighbouring lane, and a store.
    private const string RdnaRunKernel = """
        .kernel stores
        .arg x, 8, global
        .arg out, 8, global
        .arg res, 8, global
        .lds 512
        .reqd_workgroup_size 128, 1, 1
        .user_sgpr kernarg_segment_ptr
        .system_sgpr workgroup_id_x
          LOAD4 %s[0:3], s[0:1], 0x0
          LOAD2 %s[6:7], s[0:1], 0x10
          s_lshl_b32 %s4, s2, 7
          v_lshlrev_b32_e32 %v8, 2, v0
          v_add_nc_u32_e32 %v9, %s4, v0
          v_lshlrev_b32_e32 %v0, 2, %v9
          GLOAD %v1, %v0, %s[0:1]
          GSTORE %v0, %v0, %s[2:3]
          GLOAD %v2, %v0, %s[0:1] offset:512
          v_mul_f32_e32 %v3, %v1, %v1
          v_add_f32_e32 %v4, %v3, %v2
          DSTORE %v8, %v4
          s_barrier
          v_xor_b32_e32 %v6, 4, %v8
          DLOAD %v5, %v6
          v_add_f32_e32 %v7, %v5, %v4
          GSTORE %v0, %v7, %s[6:7]
          s_endpgm
        """;

    internal static void GcnInterpreterRdnaRun(Device device)
    {
        _ = device;
        const int Groups = 3, Block = 128, Count = Groups * Block;
        var random = new Random(35);
        var x = new float[Count + Block * 2];
        for (int k = 0; k < x.Length; k++)
        {
            x[k] = (float)(random.NextDouble() * 4 - 2);
        }

        foreach (var (target, cpu, names) in new[]
                 {
                     (GcnTargets.Gfx10Family, "gfx1030", (Load4: "s_load_dwordx4", Load2: "s_load_dwordx2", Load: "global_load_dword", Store: "global_store_dword", DsStore: "ds_write_b32", DsLoad: "ds_read_b32")),
                     (GcnTargets.Gfx11Family, "gfx1100", (Load4: "s_load_b128", Load2: "s_load_b64", Load: "global_load_b32", Store: "global_store_b32", DsStore: "ds_store_b32", DsLoad: "ds_load_b32")),
                 })
        {
            var isa = GcnIsaVersion.Parse(cpu);
            string text = RdnaRunKernel.Replace("LOAD4", names.Load4, StringComparison.Ordinal).Replace("LOAD2", names.Load2, StringComparison.Ordinal).Replace("GLOAD", names.Load, StringComparison.Ordinal)
                .Replace("GSTORE", names.Store, StringComparison.Ordinal).Replace("DSTORE", names.DsStore, StringComparison.Ordinal).Replace("DLOAD", names.DsLoad, StringComparison.Ordinal);
            using var asm = new GcnAssembler(target, isa);
            var code = asm.AssembleKernel(Encoding.UTF8.GetBytes(text), null);
            string printed = asm.PrintedText;
            var image = GcnCodeObjectReader.Read(GcnCodeObject.Write(target, isa, 5, [code]), "stores");
            Check(image.Cpu.ToString().Equals(cpu, StringComparison.OrdinalIgnoreCase) && image.UsableSgprs == 106 && image.Descriptor.Vgprs % 8 == 0,
                $"{cpu}: the code object reads back as {image.Cpu}, {image.UsableSgprs} SGPRs, {image.Descriptor.Vgprs} VGPRs");

            // The decoder's text of every instruction is the assembler's printed line (LLVM's syntax both).
            var lines = printed.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0 && l[0] is not ('.' or ';') && !l.EndsWith(':')).ToList();
            var decoded = new GcnDecoder(image.Cpu).DecodeAll(image.Code.AsSpan(image.Entry, code.Code.Length)).Select(i => i.Text).ToList();
            Check(decoded.SequenceEqual(lines), $"{cpu}: the decoder prints the kernel otherwise than the assembler:\n  decoded: {string.Join(" | ", decoded)}\n  printed: {string.Join(" | ", lines)}");
            Check(printed.Contains("s_waitcnt vmcnt(1)", StringComparison.Ordinal) && printed.Contains("s_waitcnt_vscnt null, ", StringComparison.Ordinal),
                $"{cpu}: the store waits on vscnt, the first load's use vmcnt(1):\n{printed}");

            var arguments = new byte[24];
            foreach (ulong seed in (ReadOnlySpan<ulong>)[1, 2])
            {
                var memory = new GcnMemory();
                ulong xAddress = memory.Map<float>("x", x);
                ulong outAddress = memory.Map("out", Count * 4);
                ulong resAddress = memory.Map("res", Count * 4);
                BinaryPrimitives.WriteUInt64LittleEndian(arguments, xAddress);
                BinaryPrimitives.WriteUInt64LittleEndian(arguments.AsSpan(8), outAddress);
                BinaryPrimitives.WriteUInt64LittleEndian(arguments.AsSpan(16), resAddress);
                GcnInterpreter.Run(image, memory, new GcnDispatch { Workgroups = (Groups, 1, 1), WorkgroupSize = (Block, 1, 1), Arguments = arguments, Seed = seed, ShuffleGroups = seed != 1 });
                var offsets = memory.Read<uint>(outAddress, Count);
                var results = memory.Read<float>(resAddress, Count);
                int bad = 0;
                string first = "";
                for (int t = 0; t < Count; t++)
                {
                    int partner = t ^ 1;
                    float own = x[t] * x[t] + x[t + Block], other = x[partner] * x[partner] + x[partner + Block];
                    float expected = other + own;
                    if (offsets[t] != (uint)(4 * t) || BitConverter.SingleToInt32Bits(results[t]) != BitConverter.SingleToInt32Bits(expected))
                    {
                        first = bad++ == 0 ? $"element {t}: out {offsets[t]}, res {results[t]} where the CPU gives {4 * t} and {expected}" : first;
                    }
                }

                Check(bad == 0, $"{cpu}, seed {seed}: {bad} of {Count} elements differ: {first}");
            }

            // wave32 code is refused (the interpreter runs waves of 64).
            var wave32 = image with { Descriptor = image.Descriptor with { Properties = (ushort)(image.Descriptor.Properties | 0x400) } };
            ThrowsNaming<GcnInterpreterException>(() => GcnInterpreter.Run(wave32, new GcnMemory(), new GcnDispatch { Workgroups = (1, 1, 1), WorkgroupSize = (Block, 1, 1), Arguments = arguments }),
                $"{cpu}: wave32", "wave32");
        }
    }
}
