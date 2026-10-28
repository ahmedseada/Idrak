// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// The GCN interpreter (plan 35, G1b): gfx9 machine code decoded and run on the CPU, apart from Idrak's encoder, so the
// GCN kernels Idrak writes are checked here, where no AMD GPU is. The decoder against LLVM 18's text for every golden
// form; fill_f32's code objects over a million floats; hand-written snippets (tests/Idrak.Tests/data/gcn/interpreter,
// assembled by llvm-mc into snippets.golden) for EXEC masking, LDS and barriers, a reduction, lane movement and float
// arithmetic; and wrong code (a wait or a barrier taken out, an access past a storage) failing loudly. Pure C#, no
// device, deterministic.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnInterpreterGroup =
    [
        ("gcn interpreter: the decoder reads every forms.golden record of gfx900, gfx906, gfx908 and gfx90c and prints it as LLVM 18 does (branches as their 16-bit dword counts); an encoding a processor lacks, one the interpreter does not decode and a cut instruction are refused with their offset and words", GcnInterpreterForms),
        ("gcn interpreter: fill_f32, the golden code objects (formats 4 and 5) and GcnKernels' own for gfx90c, fills 1,000,003 floats with 2.5 in 3,907 workgroups of 256: every value set, nothing past the count", GcnInterpreterFill),
        ("gcn interpreter: the snippets decode as llvm-objdump prints them; a bounds check and an if/else narrow EXEC (two waves, the last partial): each lane takes its branch, no lane past the count stores", GcnInterpreterDivergent),
        ("gcn interpreter: a sum in LDS across four waves (s_barrier before each step of a fixed tree) equals the same tree on the CPU bit for bit, under three seeds", GcnInterpreterReduce),
        ("gcn interpreter: lane movement: DPP (row_shr with bound_ctrl, quad_perm, row_bcast under a row mask, wave_shr, row_mirror), ds_swizzle, ds_bpermute, readlane, writelane, mbcnt, readfirstlane under a narrowed EXEC, SDWA selections", GcnInterpreterLanes),
        ("gcn interpreter: float32: LLVM's division sequence (div_scale, rcp, fma, div_fmas, div_fixup) gives a / b within an ULP, over- and underflow and the special cases included; transcendentals, fma, mad, med3, clamp, conversions and rounding as the CPU computes them", GcnInterpreterMath),
        ("gcn interpreter: a missing s_waitcnt fails loudly with both offsets: a load used before vmcnt retires it, vmcnt(2) where the second of two loads is used, two SMEM loads waited with lgkmcnt(1), LDS reads used early, a write over a pending load, LDS written without lgkmcnt(0) before the s_barrier", GcnInterpreterMissingWaits),
        ("gcn interpreter: races fail loudly: LDS read and written by two waves with no s_barrier between, two workgroups writing the same bytes", GcnInterpreterRaces),
        ("gcn interpreter: out-of-range accesses are reported, never ignored: a store and a load past a storage, LDS past the workgroup's size; and registers past the descriptor's counts", GcnInterpreterOutOfRange),
    ];

    private static readonly GcnCpu[] InterpreterCpus = [GcnCpu.Gfx900, GcnCpu.Gfx906, GcnCpu.Gfx908, GcnCpu.Gfx90c];

    internal static void GcnInterpreterForms(Device device)
    {
        _ = device;
        string[] lines = File.ReadAllLines(RepositoryPath("tests", "Idrak.Tests", "data", "gcn", "gfx9", "forms.golden"));
        var counts = new List<string>();
        foreach (var cpu in InterpreterCpus)
        {
            string name = cpu.ToString().ToLowerInvariant();
            var records = lines.Where(l => l.Length > 0 && l[0] != '#').Select(l => l.Split('\t')).Where(f => f[0] == name).ToList();
            var labels = new Dictionary<string, int>(StringComparer.Ordinal);
            int offset = 0;
            foreach (var r in records)
            {
                if (r[2].Length == 0)
                {
                    labels[r[1].TrimEnd(':')] = offset;
                }
                else
                {
                    offset += r[2].Length / 2;
                }
            }

            var decoder = new GcnDecoder(cpu);
            var wrong = new List<string>();
            int decoded = 0;
            offset = 0;
            foreach (var r in records)
            {
                if (r[2].Length == 0)
                {
                    continue;
                }

                byte[] bytes = Convert.FromHexString(r[2]);
                string expected = Resolve(r[1], labels, offset);
                offset += bytes.Length;
                try
                {
                    var instruction = decoder.Decode(bytes, 0);
                    if (instruction.Size != bytes.Length || instruction.Text != expected)
                    {
                        wrong.Add($"{r[2]}: '{instruction.Text}' ({instruction.Size} bytes) where LLVM prints '{expected}' ({bytes.Length})");
                    }
                    else
                    {
                        decoded++;
                    }
                }
                catch (GcnDecodeException e)
                {
                    wrong.Add($"{r[2]} ('{expected}'): refused: {e.Message}");
                }
            }

            Check(wrong.Count == 0, $"{name}: {wrong.Count} of {decoded + wrong.Count} forms differ:\n  " + string.Join("\n  ", wrong.Take(30)));
            int expectedForms = int.Parse(lines.First(l => l.StartsWith($"# {name}: ", StringComparison.Ordinal)).Split(' ')[2], CultureInfo.InvariantCulture);
            Check(decoded == expectedForms, $"{name}: {decoded} forms decoded, the golden's header says {expectedForms}");
            counts.Add($"{name} {decoded}");
        }

        Console.WriteLine($"    forms decoded: {string.Join(", ", counts)}");

        // Refused, naming the offset and the words: an opcode the processor lacks (v_fmac_f32 is gfx906's and later),
        // an encoding not decoded (MUBUF), a VOP3 opcode no gfx9 processor has, an 8-byte form cut after 4 bytes.
        var gfx900 = new GcnDecoder(GcnCpu.Gfx900);
        byte[] fmac = Words([0x76040301]);
        Check(new GcnDecoder(GcnCpu.Gfx906).Decode(fmac, 0).Text == "v_fmac_f32_e32 v2, v1, v1", "gfx906 has v_fmac_f32");
        ThrowsNaming<GcnDecodeException>(() => gfx900.Decode(fmac, 0), "v_fmac_f32 on gfx900", "0x0000", "76040301", "VOP2", "0x3B", "gfx900");
        ThrowsNaming<GcnDecodeException>(() => gfx900.Decode(Words([0xBF800000, 0xE0500000, 0x00000000]), 4), "MUBUF", "0x0004", "E0500000", "MUBUF");
        ThrowsNaming<GcnDecodeException>(() => gfx900.Decode(Words([0xD2B00000, 0x00000000]), 0), "VOP3 0x2B0", "D2B00000 00000000", "0x2B0");
        ThrowsNaming<GcnDecodeException>(() => gfx900.Decode(Words([0xC0020002]), 0), "an SMEM word alone", "C0020002", "8 bytes");
        ThrowsNaming<GcnDecodeException>(() => gfx900.Decode(Words([0x7E0202FF]), 0), "a literal cut off", "7E0202FF", "literal");
    }

    /// <summary>A golden branch's label replaced by the dword count LLVM's disassembler prints (16 bits, unsigned).</summary>
    private static string Resolve(string text, Dictionary<string, int> labels, int offset)
    {
        if (!text.StartsWith("s_branch ", StringComparison.Ordinal) && !text.StartsWith("s_cbranch_", StringComparison.Ordinal))
        {
            return text;
        }

        int space = text.IndexOf(' ', StringComparison.Ordinal);
        string target = text[(space + 1)..];
        if (int.TryParse(target, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int count))
        {
            // A branch written with a number: the golden holds the assembler's echo (-1), the disassembler prints 65535.
            return $"{text[..space]} {(ushort)count}";
        }

        return labels.TryGetValue(target, out int at) ? $"{text[..space]} {(ushort)((at - offset - 4) / 4)}" : text;
    }

    internal static void GcnInterpreterFill(Device device)
    {
        _ = device;
        const int Count = 1_000_003, Past = 61;
        (string Name, GcnKernelImage Kernel)[] kernels =
        [
            ("the golden, format 4", GcnCodeObjectReader.Read(File.ReadAllBytes(GoldenPath("fill_f32.gfx90c.v4.co")), "fill_f32")),
            ("the golden, format 5", GcnCodeObjectReader.Read(File.ReadAllBytes(GoldenPath("fill_f32.gfx90c.v5.co")), "fill_f32")),
            ("GcnKernels' own", GcnCodeObjectReader.Read(GcnKernels.FillCodeObject(GcnIsaVersion.Parse("gfx90c"), 5), "fill_f32")),
        ];
        foreach (var (name, kernel) in kernels)
        {
            Check(kernel is { Cpu: GcnCpu.Gfx90c, Descriptor: { KernargBytes: 16, UserSgprCount: 2, KernargSegmentPtr: true, WorkgroupIdX: true, Vgprs: 4 } },
                $"{name}: gfx90c, 16 bytes of arguments in s[0:1], the workgroup id x, 4 VGPRs: {kernel.Descriptor}");
            var memory = new GcnMemory();
            var output = new float[Count + Past];
            output.AsSpan().Fill(-7.25f);
            ulong address = memory.Map("out", MemoryMarshal.AsBytes(output.AsSpan()).ToArray());
            var arguments = new byte[16];
            BinaryPrimitives.WriteUInt64LittleEndian(arguments, address);
            BinaryPrimitives.WriteSingleLittleEndian(arguments.AsSpan(8), 2.5f);
            BinaryPrimitives.WriteUInt32LittleEndian(arguments.AsSpan(12), Count);
            long start = Stopwatch.GetTimestamp();
            var run = GcnInterpreter.Run(kernel, memory, new GcnDispatch { Workgroups = ((Count + 255) / 256, 1, 1), WorkgroupSize = (256, 1, 1), Arguments = arguments });
            var values = MemoryMarshal.Cast<byte, float>(memory.Bytes(address).AsSpan());
            int unset = values[..Count].IndexOfAnyExcept(2.5f);
            Check(unset < 0, $"{name}: out[{unset}] is {(unset < 0 ? 0 : values[unset])}, not 2.5");
            int touched = values[Count..].IndexOfAnyExcept(-7.25f);
            Check(touched < 0, $"{name}: out[{Count + touched}], past the count, was written");
            Check(run.Workgroups == 3907 && run.Waves == 4 * 3907, $"{name}: {run.Workgroups} workgroups, {run.Waves} waves");
            Console.WriteLine($"    {name}: {run.Instructions:N0} instructions in {Stopwatch.GetElapsedTime(start).TotalMilliseconds:F0} ms");
        }
    }

    // ---- Snippets ------------------------------------------------------------------------------------------------

    private sealed record SnippetRecord(int Offset, string Text, uint[] Words);

    private static Dictionary<string, List<SnippetRecord>>? snippets;

    private static Dictionary<string, List<SnippetRecord>> Snippets()
    {
        if (snippets is not null)
        {
            return snippets;
        }

        var all = new Dictionary<string, List<SnippetRecord>>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(RepositoryPath("tests", "Idrak.Tests", "data", "gcn", "interpreter", "snippets.golden")))
        {
            if (line.Length == 0 || line[0] == '#')
            {
                continue;
            }

            string[] f = line.Split('\t');
            uint[] words = f[3].Split(' ').Select(w => uint.Parse(w, NumberStyles.HexNumber, CultureInfo.InvariantCulture)).ToArray();
            (all.TryGetValue(f[0], out var list) ? list : all[f[0]] = []).Add(new SnippetRecord(int.Parse(f[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture), f[2], words));
        }

        return snippets = all;
    }

    /// <summary>
    /// A snippet as a kernel for gfx90c: its code (each record's text checked against the decoder's), with the
    /// instruction <paramref name="replace"/> names (its text, and which occurrence) replaced by one word.
    /// </summary>
    private static GcnKernelImage Snippet(string name, GcnKernelDescriptor descriptor, (string Text, int Occurrence, uint Word)? replace = null)
    {
        var records = Snippets()[name];
        var code = new List<byte>();
        var decoder = new GcnDecoder(GcnCpu.Gfx90c);
        int seen = 0;
        bool replaced = false;
        foreach (var r in records)
        {
            byte[] bytes = Words(r.Words);
            var instruction = decoder.Decode(bytes, 0);
            Check(instruction.Text == r.Text && instruction.Size == bytes.Length && code.Count == r.Offset,
                $"{name} at 0x{r.Offset:X}: decoded '{instruction.Text}' where llvm-objdump prints '{r.Text}'");
            if (replace is { } change && r.Text == change.Text && seen++ == change.Occurrence)
            {
                Check(bytes.Length == 4, $"{name}: '{r.Text}' is replaced by one word, so it must be one");
                bytes = Words([change.Word]);
                replaced = true;
            }

            code.AddRange(bytes);
        }

        Check(replace is null || replaced, $"{name}: no '{replace?.Text}' (occurrence {replace?.Occurrence}) to replace");
        return new GcnKernelImage(name, GcnCpu.Gfx90c, [.. code], 0, descriptor, Xnack: true);
    }

    private static byte[] Arguments(params ulong[] values)
    {
        var bytes = new byte[8 * values.Length];
        for (int k = 0; k < values.Length; k++)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(8 * k), values[k]);
        }

        return bytes;
    }

    private const uint SNop = 0xBF800000;

    private static readonly GcnKernelDescriptor DivergentKernel = GcnKernelDescriptor.For(vgprs: 5, sgprs: 12, kernargBytes: 16);

    private static readonly GcnKernelDescriptor ReduceKernel = GcnKernelDescriptor.For(vgprs: 11, sgprs: 13, ldsBytes: 1024, kernargBytes: 16);

    private static readonly GcnKernelDescriptor LanesKernel = GcnKernelDescriptor.For(vgprs: 21, sgprs: 12, kernargBytes: 8, workgroupIds: 0);

    private static readonly GcnKernelDescriptor CountsKernel = GcnKernelDescriptor.For(vgprs: 5, sgprs: 8, kernargBytes: 16, workgroupIds: 0);

    private static readonly GcnKernelDescriptor MathKernel = GcnKernelDescriptor.For(vgprs: 23, sgprs: 12, kernargBytes: 24, workgroupIds: 0);

    /// <summary>The divergent snippet over <paramref name="count"/> elements (an argument of <paramref name="argument"/>) into a storage of <paramref name="count"/> uints.</summary>
    private static uint[] RunDivergent(GcnKernelImage kernel, int count, int argument)
    {
        var memory = new GcnMemory();
        ulong output = memory.Map("out", count * 4);
        GcnInterpreter.Run(kernel, memory, new GcnDispatch { Workgroups = ((count + 127) / 128, 1, 1), WorkgroupSize = (128, 1, 1), Arguments = Arguments(output, (ulong)argument) });
        return memory.Read<uint>(output, count);
    }

    internal static void GcnInterpreterDivergent(Device device)
    {
        _ = device;
        foreach (string name in Snippets().Keys)
        {
            _ = Snippet(name, DivergentKernel);
        }

        // 200 elements in two workgroups of 128: the second's second wave has 8 lanes past the count, its first 64.
        uint[] output = RunDivergent(Snippet("divergent", DivergentKernel), 200, 200);
        for (int i = 0; i < 200; i++)
        {
            uint expected = (i & 1) != 0 ? (uint)(3 * i) : (uint)(100 - i);
            Check(output[i] == expected, $"out[{i}] is {output[i]}, expected {expected} ({((i & 1) != 0 ? "then" : "else")} branch)");
        }
    }

    private static (float[] Input, float[] Expected) ReduceData(int groups)
    {
        var input = new float[256 * groups];
        for (int k = 0; k < input.Length; k++)
        {
            input[k] = (float)(Math.Sin(k * 0.37) * 100 + k % 7);
        }

        var expected = new float[groups];
        var tree = new float[256];
        for (int g = 0; g < groups; g++)
        {
            input.AsSpan(256 * g, 256).CopyTo(tree);
            for (int stride = 128; stride > 0; stride >>= 1)
            {
                for (int t = 0; t < stride; t++)
                {
                    tree[t] += tree[t + stride];
                }
            }

            expected[g] = tree[0];
        }

        return (input, expected);
    }

    private static float[] RunReduce(GcnKernelImage kernel, float[] input, int groups, ulong seed)
    {
        var memory = new GcnMemory();
        ulong inAddress = memory.Map<float>("in", input);
        ulong outAddress = memory.Map("out", groups * 4);
        GcnInterpreter.Run(kernel, memory, new GcnDispatch { Workgroups = (groups, 1, 1), WorkgroupSize = (256, 1, 1), Arguments = Arguments(inAddress, outAddress), Seed = seed });
        return memory.Read<float>(outAddress, groups);
    }

    internal static void GcnInterpreterReduce(Device device)
    {
        _ = device;
        const int Groups = 5;
        var (input, expected) = ReduceData(Groups);
        var kernel = Snippet("reduce", ReduceKernel);
        foreach (ulong seed in (ulong[])[1, 2, 3])
        {
            float[] sums = RunReduce(kernel, input, Groups, seed);
            for (int g = 0; g < Groups; g++)
            {
                Check(BitConverter.SingleToInt32Bits(sums[g]) == BitConverter.SingleToInt32Bits(expected[g]), $"seed {seed}: workgroup {g}'s sum is {sums[g]:R}, the CPU's tree gives {expected[g]:R}");
            }
        }
    }

    internal static void GcnInterpreterLanes(Device device)
    {
        _ = device;
        var memory = new GcnMemory();
        ulong output = memory.Map("out", 13 * 64 * 4);
        GcnInterpreter.Run(Snippet("lanes", LanesKernel), memory, new GcnDispatch { Arguments = Arguments(output) });
        uint[] got = memory.Read<uint>(output, 13 * 64);
        static uint X(int lane) => (uint)(3 * lane + 1);
        string[] names = ["row_shr:1 bound_ctrl", "quad_perm:[1,0,3,2]", "row_bcast:15 row_mask:0xa", "ds_swizzle SWAP 16", "ds_bpermute 63 - lane", "v_readlane 5", "v_writelane 7",
            "v_mbcnt", "wave_shr:1", "row_mirror", "v_readfirstlane under EXEC 0xff00", "SDWA BYTE_0 to WORD_1", "SDWA BYTE_1 + BYTE_0"];
        for (int lane = 0; lane < 64; lane++)
        {
            int row = lane >> 4;
            uint x = X(lane);
            uint[] expected =
            [
                (lane & 15) == 0 ? 0 : X(lane - 1),
                X(lane ^ 1),
                row is 1 or 3 ? X(16 * row - 1) : 7,
                X(lane ^ 16),
                X(63 - lane),
                X(5),
                lane == 7 ? 42u : 0,
                (uint)lane,
                lane == 0 ? 9 : X(lane - 1),
                X((lane & ~15) | (15 - (lane & 15))),
                X(8),
                (x & 0xFF) << 16,
                ((x >> 8) & 0xFF) + (x & 0xFF),
            ];
            for (int k = 0; k < expected.Length; k++)
            {
                Check(got[k * 64 + lane] == expected[k], $"{names[k]}: lane {lane} holds {got[k * 64 + lane]}, expected {expected[k]}");
            }
        }
    }

    private static int UlpDistance(float a, float b)
    {
        if (float.IsNaN(a) || float.IsNaN(b))
        {
            return float.IsNaN(a) && float.IsNaN(b) ? 0 : int.MaxValue;
        }

        if (a == b)
        {
            return 0;
        }

        int x = BitConverter.SingleToInt32Bits(a), y = BitConverter.SingleToInt32Bits(b);
        if ((x < 0) != (y < 0))
        {
            return int.MaxValue;
        }

        return Math.Abs(x - y);
    }

    internal static void GcnInterpreterMath(Device device)
    {
        _ = device;
        var a = new float[64];
        var b = new float[64];
        (float, float)[] special =
        [
            (1, 3), (2, 7), (-5, 0.1f), (1e30f, 1e-10f), (1e-30f, 1e10f), (3e38f, 1e-3f), (1e-38f, 3), (0, 5), (5, 0), (float.PositiveInfinity, 2), (2, float.PositiveInfinity),
            (0, 0), (float.NaN, 1), (-0f, 4), (1.5f, -0f), (7, 7), (2.5f, 0.75f), (-2.5f, -8), (1e-40f, 1), (6e-39f, 2.5e-39f),
        ];
        var random = new Random(35);
        for (int lane = 0; lane < 64; lane++)
        {
            (a[lane], b[lane]) = lane < special.Length ? special[lane] : ((float)(random.NextDouble() * 100 - 50), (float)((random.NextDouble() * 99.9 + 0.1) * (random.Next(2) * 2 - 1)));
        }

        var memory = new GcnMemory();
        ulong aAddress = memory.Map<float>("a", a), bAddress = memory.Map<float>("b", b), outAddress = memory.Map("out", 18 * 64 * 4);
        GcnInterpreter.Run(Snippet("math", MathKernel), memory, new GcnDispatch { Arguments = [.. Arguments(aAddress, bAddress, outAddress)] });
        float[] got = memory.Read<float>(outAddress, 18 * 64);
        for (int lane = 0; lane < 64; lane++)
        {
            float x = a[lane], y = b[lane];
            float Out(int k) => got[k * 64 + lane];
            void Near(int k, string what, float expected, int ulps)
            {
                int distance = UlpDistance(Out(k), expected);
                Check(distance <= ulps, $"{what} for a = {x:R}, b = {y:R} (lane {lane}): {Out(k):R}, expected {expected:R} ({(distance == int.MaxValue ? "differs" : $"{distance} ULP")})");
            }

            Near(0, "a / b", x / y, 1);
            Near(1, "2^a", (float)Math.Pow(2, x), 1);
            Near(2, "log2 |b|", (float)Math.Log2(Math.Abs(y)), 1);
            double turn = x - Math.Round((double)x);
            float sin = (float)Math.Sin(2 * Math.PI * turn), cos = (float)Math.Cos(2 * Math.PI * turn);
            Check(!float.IsFinite(x) ? float.IsNaN(Out(3)) : Math.Abs(Out(3) - sin) <= 2e-6f, $"sin(2π a) for a = {x:R}: {Out(3):R}, expected {sin:R}");
            Check(!float.IsFinite(x) ? float.IsNaN(Out(4)) : Math.Abs(Out(4) - cos) <= 2e-6f, $"cos(2π a) for a = {x:R}: {Out(4):R}, expected {cos:R}");
            Near(5, "√|b|", MathF.Sqrt(Math.Abs(y)), 0);
            Near(6, "1/√|b|", (float)(1 / Math.Sqrt(Math.Abs(y))), 1);
            Near(7, "fma(a, b, -a)", MathF.FusedMultiplyAdd(x, y, -x), 0);
            Near(8, "mad(a, b, 0.5)", GcnFloat.Mad(x, y, 0.5f), 0);
            Near(9, "med3(a, -1, 1)", float.IsNaN(x) ? -1 : Math.Clamp(x, -1, 1), 0);
            float max = float.IsNaN(x) ? y : float.IsNaN(y) ? x : Math.Max(x, y);
            Near(10, "max(a, b) clamped", float.IsNaN(max) ? 0 : Math.Clamp(max, 0, 1), 0);
            int integer = float.IsNaN(x) ? 0 : x >= int.MaxValue ? int.MaxValue : x <= int.MinValue ? int.MinValue : (int)x;
            Check(BitConverter.SingleToInt32Bits(Out(11)) == integer, $"cvt_i32(a) for a = {x:R}: {BitConverter.SingleToInt32Bits(Out(11))}, expected {integer}");
            Near(12, "f32(f16(a))", (float)(Half)x, 0);
            Near(13, "rndne(a)", MathF.Round(x, MidpointRounding.ToEven), 0);
            Near(14, "fract(a)", float.IsNaN(x) || float.IsInfinity(x) ? float.NaN : Math.Min(x - MathF.Floor(x), BitConverter.Int32BitsToSingle(0x3F7FFFFF)), 0);
            Near(15, "a < b ? a : b", x < y ? x : y, 0);
            Near(16, "ldexp(a, 3)", x * 8, 0);
            uint packed = BitConverter.SingleToUInt32Bits(Out(17));
            Check(BitConverter.UInt16BitsToHalf((ushort)packed).Equals(GcnFloat.ToHalfTowardZero(x)) && BitConverter.UInt16BitsToHalf((ushort)(packed >> 16)).Equals(GcnFloat.ToHalfTowardZero(y)),
                $"pkrtz(a, b) for a = {x:R}, b = {y:R}: 0x{packed:X8}");
        }

        // Toward zero: 1/3 in float16 is 0x3555 to nearest, 0x3555 toward zero too; 65519 rounds to 65504 (not infinity), -1e9 to -65504.
        Check(BitConverter.HalfToUInt16Bits(GcnFloat.ToHalfTowardZero(65519f)) == 0x7BFF && BitConverter.HalfToUInt16Bits(GcnFloat.ToHalfTowardZero(-1e9f)) == 0xFBFF
              && BitConverter.HalfToUInt16Bits(GcnFloat.ToHalfTowardZero(1.0009766f + 0.0004f)) == 0x3C01, "float16 toward zero: overflow to the largest finite, a value between two halves to the lower");
    }

    internal static void GcnInterpreterMissingWaits(Device device)
    {
        _ = device;
        var input = new uint[128];
        for (int k = 0; k < input.Length; k++)
        {
            input[k] = (uint)(k * 1000 + 7);
        }

        uint[] RunCounts(GcnKernelImage kernel)
        {
            var memory = new GcnMemory();
            ulong inAddress = memory.Map<uint>("in", input), outAddress = memory.Map("out", 64 * 4);
            GcnInterpreter.Run(kernel, memory, new GcnDispatch { Arguments = Arguments(inAddress, outAddress) });
            return memory.Read<uint>(outAddress, 64);
        }

        uint[] sums = RunCounts(Snippet("counts", CountsKernel));
        for (int lane = 0; lane < 64; lane++)
        {
            Check(sums[lane] == input[lane] + input[lane + 64], $"counts: lane {lane} is {sums[lane]}");
        }

        void Missing(string what, GcnKernelImage kernel, Action<GcnKernelImage> run, params string[] naming)
        {
            try
            {
                run(kernel);
                Check(false, $"{what}: ran without a failure");
            }
            catch (GcnInterpreterException e)
            {
                Check(e.Kind == GcnFailure.MissingWait, $"{what}: failed as {e.Kind}, not a missing wait: {e.Message}");
                foreach (string text in naming)
                {
                    Check(e.Message.Contains(text, StringComparison.Ordinal), $"{what}: the message names '{text}': {e.Message}");
                }
            }
        }

        Action<GcnKernelImage> counts = k => RunCounts(k);
        Missing("the first load used with no wait", Snippet("counts", CountsKernel, ("s_waitcnt vmcnt(1)", 0, SNop)), counts,
            "0x002C 'v_mov_b32_e32 v4, v2'", "reads v2", "'global_load_dword v2, v1, s[4:5]' at 0x0018", "vmcnt");
        Missing("vmcnt(2) where the first of two loads is used", Snippet("counts", CountsKernel, ("s_waitcnt vmcnt(1)", 0, 0xBF8C0F72)), counts, "reads v2", "at 0x0018");
        Missing("the second load used with no wait", Snippet("counts", CountsKernel, ("s_waitcnt vmcnt(0)", 0, SNop)), counts,
            "0x0034 'v_add_u32_e32 v4, v4, v3'", "reads v3", "at 0x0020");
        Missing("two SMEM loads waited with lgkmcnt(1) (they complete in any order)", Snippet("counts", CountsKernel, ("s_waitcnt lgkmcnt(0)", 0, 0xBF8CC17F)), counts,
            "reads s4", "'s_load_dwordx2 s[4:5], s[0:1], 0x0' at 0x0000", "lgkmcnt");
        Missing("a write over a pending load (its late return would overwrite it)", Snippet("counts", CountsKernel, ("v_mov_b32_e32 v4, v2", 0, 0x7E060302)), counts,
            "writes v3", "at 0x0020");

        var (reduceInput, _) = ReduceData(2);
        Action<GcnKernelImage> reduce = k => RunReduce(k, reduceInput, 2, 1);
        Missing("the global load stored to LDS with no wait", Snippet("reduce", ReduceKernel, ("s_waitcnt vmcnt(0)", 0, SNop)), reduce, "ds_write_b32 v4, v3", "reads v3");
        Missing("LDS reads used with no wait", Snippet("reduce", ReduceKernel, ("s_waitcnt lgkmcnt(0)", 2, SNop)), reduce, "v_add_f32_e32 v6, v6, v7", "ds_read_b32");
        Missing("LDS written, then read by other waves after an s_barrier with no lgkmcnt(0) before it", Snippet("reduce", ReduceKernel, ("s_waitcnt lgkmcnt(0)", 1, SNop)), reduce,
            "ds_read_b32", "no s_waitcnt retired that store before the s_barrier", "'ds_write_b32 v4, v3'");
    }

    internal static void GcnInterpreterRaces(Device device)
    {
        _ = device;
        var (input, _) = ReduceData(2);
        try
        {
            RunReduce(Snippet("reduce", ReduceKernel, ("s_barrier", 0, SNop)), input, 2, 1);
            Check(false, "the LDS tree without its s_barrier ran without a failure");
        }
        catch (GcnInterpreterException e)
        {
            Check(e.Kind == GcnFailure.Race && e.Message.Contains("LDS byte", StringComparison.Ordinal) && e.Message.Contains("no s_barrier", StringComparison.Ordinal),
                $"the missing s_barrier is a race on LDS bytes: {e.Kind}: {e.Message}");
        }

        // counts writes out[lane] whatever the workgroup: two workgroups write the same bytes.
        var memory = new GcnMemory();
        ulong inAddress = memory.Map("in", 128 * 4), outAddress = memory.Map("out", 64 * 4);
        try
        {
            GcnInterpreter.Run(Snippet("counts", CountsKernel), memory, new GcnDispatch { Workgroups = (2, 1, 1), Arguments = Arguments(inAddress, outAddress) });
            Check(false, "two workgroups writing the same bytes ran without a failure");
        }
        catch (GcnInterpreterException e)
        {
            Check(e.Kind == GcnFailure.Race && e.Message.Contains("workgroup 0 also wrote", StringComparison.Ordinal) && e.Message.Contains("workgroup (1, 0, 0)", StringComparison.Ordinal),
                $"two workgroups' stores to one place are a race: {e.Kind}: {e.Message}");
        }
    }

    internal static void GcnInterpreterOutOfRange(Device device)
    {
        _ = device;
        void Fails(string what, GcnFailure kind, Action run, params string[] naming)
        {
            try
            {
                run();
                Check(false, $"{what}: ran without a failure");
            }
            catch (GcnInterpreterException e)
            {
                Check(e.Kind == kind, $"{what}: failed as {e.Kind}, not {kind}: {e.Message}");
                foreach (string text in naming)
                {
                    Check(e.Message.Contains(text, StringComparison.Ordinal), $"{what}: the message names '{text}': {e.Message}");
                }
            }
        }

        // A count one past the storage: element 200's store, lane 8 of workgroup 1's second wave.
        Fails("a store past a storage", GcnFailure.OutOfRange, () => RunDivergent(Snippet("divergent", DivergentKernel), 200, 201),
            "global_store_dword v4, v3, s[4:5]", "lane 8", "workgroup (1, 0, 0) wave 1", "0 bytes past the end of 'out' (800 bytes)");

        // counts reads in[lane + 64]: a storage of 64 holds none of them.
        Fails("a load past a storage", GcnFailure.OutOfRange, () =>
        {
            var memory = new GcnMemory();
            ulong inAddress = memory.Map("in", 64 * 4), outAddress = memory.Map("out", 64 * 4);
            GcnInterpreter.Run(Snippet("counts", CountsKernel), memory, new GcnDispatch { Arguments = Arguments(inAddress, outAddress) });
        }, "global_load_dword v3, v1, s[4:5] offset:256", "lane 0", "past the end of 'in'");

        // The tree's LDS is 1,024 bytes: with 512, the upper half's stores fall outside.
        var (input, _) = ReduceData(1);
        Fails("LDS past the workgroup's size", GcnFailure.OutOfRange, () => RunReduce(Snippet("reduce", ReduceKernel with { LdsBytes = 512 }), input, 1, 1),
            "ds_write_b32 v4, v3", "past the workgroup's 512 bytes");

        // Registers: lanes uses v20, divergent s4 to s11.
        Fails("a VGPR past the descriptor's", GcnFailure.Register, () =>
        {
            var memory = new GcnMemory();
            GcnInterpreter.Run(Snippet("lanes", GcnKernelDescriptor.For(vgprs: 16, sgprs: 12, kernargBytes: 8, workgroupIds: 0)), memory, new GcnDispatch { Arguments = Arguments(memory.Map("out", 13 * 64 * 4)) });
        }, "v20 is past the 16 VGPRs");
        Fails("an SGPR past the descriptor's", GcnFailure.Register, () => RunDivergent(Snippet("divergent", GcnKernelDescriptor.For(vgprs: 5, sgprs: 4, kernargBytes: 16)), 64, 64),
            "s4 is past the 4 SGPRs", "less VCC and the xnack mask");
    }
}
