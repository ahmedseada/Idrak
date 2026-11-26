// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;

// Proof 1: the CPU's lanes give the same bits on any CPU. The described lane over stored weights (the bits recorded by
// commit 3f1c53e2's lane-bits test for each type's description and the types made of a description alone; the written
// lanes are gone) and the operations of the new design by every width that runs here (each width forced in turn): one
// list of hashes holds on every CPU and instruction set.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] LaneBitsGroup =
    [
        ("lane bits: the CPU's described lane over stored weights (q4_k, q4_0, mxfp4, q6_k, bfloat16, float64, int2 and int64 from their types' descriptions; the types made of a description alone; 1, 3, 4, 5, 9 and 70 rows of fixed operands) gives commit 3f1c53e2's recorded bits by every instruction set that runs here, and the new design's operations (element-wise, rows, products, attention, sums) give one hash each by every width (512, 256, 128 bits, one float): the same list on any CPU", LaneBits),
    ];

    private static float[] LaneBitsValues(Random random, long count) => [.. Enumerable.Range(0, (int)count).Select(_ => random.NextSingle() * 2 - 1)];

    // An operand set's seed from its name (FNV-1a over its characters: the same on every run and every machine).
    private static int LaneBitsSeed(string name)
    {
        uint hash = 2166136261;
        foreach (char c in name)
        {
            hash = (hash ^ c) * 16777619;
        }

        return (int)(hash & int.MaxValue);
    }

    // Types made of a description alone (registered nowhere, no conversion and no kernel of their own): a 3-bit code in a
    // 2-bit and a 1-bit plane with an E8M0 scale; a table of 16 fractions with a bfloat16 scale a 64 values; a 6-bit fp E2M3
    // in a 4-bit and a 2-bit plane with an E8M0 scale; 16-bit integers with a code offset of 0.5 and without; 32-bit
    // integers. Each with the row length it is multiplied with.
    private static (string Name, TypeDescription D, int K)[] LaneBitsDescribedOnly()
    {
        var e8m0 = Idrak.Abstraction.Types.F8E8M0.Type.Description.Code;
        double[] fractions = [-1, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1];
        var made = new (string Name, TypeDescription D, int K)[]
        {
            ("3-bit codes in two planes, offset -4, an E8M0 scale a 32 values",
                new TypeDescription(32, 13, NumberType.Integer(3, false), [new CodePlane(1, 2, 4), new CodePlane(9, 1, 8)], -4, [new BlockScale(e8m0, 32, 0)]), 96),
            ("a table of 16 fractions, two a byte, a bfloat16 scale a 64 values",
                new TypeDescription(64, 34, NumberType.Entries(4, fractions), [new CodePlane(2, 4, 2)], Scales: [new BlockScale(NumberType.Floating(8, 7), 64, 0)]), 192),
            ("fp6 E2M3 in a 4-bit and a 2-bit plane, an E8M0 scale a 32 values",
                new TypeDescription(32, 25, NumberType.Floating(2, 3, bias: 1, specials: FloatSpecials.None), [new CodePlane(1, 4, 32), new CodePlane(17, 2, 32)],
                    Scales: [new BlockScale(e8m0, 32, 0)]), 320),
            ("16-bit integers plus 0.5", new TypeDescription(1, 2, NumberType.Integer(16, true), [new CodePlane(0, 16, 1)], 0.5), 100),
            ("16-bit integers", new TypeDescription(1, 2, NumberType.Integer(16, true), [new CodePlane(0, 16, 1)]), 100),
            ("32-bit integers", new TypeDescription(1, 4, NumberType.Integer(32, true), [new CodePlane(0, 32, 1)]), 100),
        };

        return made;
    }

    // n rows of k values of description d as stored, made from the description alone (never bytes an encoder wrote):
    // random bytes, then every float field whose type has codes that are not finite values (a value's code, a scale, a
    // minimum: IEEE's floats, E4M3's, E8M0's; at its planes' places, its offset or its pieces) set to a finite value of
    // either sign (an unsigned type's: positive), magnitude in [2^-4, 1): its exponent from its bias − 4 to its bias − 1, a
    // random mantissa. Every other field (integers, tables, codebook indices, floats with every code finite) stays random.
    private static byte[] LaneBitsBytes(TypeDescription d, int k, int n, Random random)
    {
        var bytes = new byte[n * (k / d.UnitValues) * d.UnitBytes];
        random.NextBytes(bytes);
        var factors = (d.Scales ?? []).Concat(d.Mins ?? []).Where(s => Bounded(s.Type)).ToArray();
        for (int at = 0; at < bytes.Length; at += d.UnitBytes)
        {
            var unit = bytes.AsSpan(at, d.UnitBytes);
            for (int v = 0; v < d.UnitValues && Bounded(d.Code); v += d.Code.Width)
            {
                ulong bits = FiniteBits(d.Code, random);
                int start = d.BlockStart(v), entry = v % d.Values / d.Code.Width;
                foreach (var plane in d.Planes)
                {
                    var (b, s) = plane.Place(entry);
                    Put(unit, 8L * (start + b) + s, plane.Bits, bits);
                    bits >>= plane.Bits;
                }
            }

            foreach (var scale in factors)
            {
                for (int v = 0; v < d.UnitValues; v += scale.Values)
                {
                    ulong bits = FiniteBits(scale.Type, random);
                    int origin = d.ScaleBase(scale, v), index = d.ScaleIndex(scale, v);
                    if (scale.Pieces is null)
                    {
                        Put(unit, 8L * (origin + scale.Offset) + (long)index * scale.Type.Bits, scale.Type.Bits, bits);
                        continue;
                    }

                    foreach (var piece in scale.Pieces[index])
                    {
                        Put(unit, 8L * (origin + piece.Offset) + piece.Shift, piece.Bits, bits);
                        bits >>= piece.Bits;
                    }
                }
            }
        }

        return bytes;

        static bool Bounded(NumberType t) => t is { Kind: NumberKind.Float, Width: 1 } && t.Specials != FloatSpecials.None;

        static ulong FiniteBits(NumberType t, Random random)
        {
            int low = Math.Max(t.Subnormals ? 1 : 0, t.ExponentBias - 4), high = Math.Max(low, t.ExponentBias - 1);
            ulong bits = (ulong)random.Next(low, high + 1) << t.Mantissa | (ulong)random.NextInt64() & ((1UL << t.Mantissa) - 1);
            return t.Signed && random.Next(2) == 1 ? bits | 1UL << (t.Exponent + t.Mantissa) : bits;
        }

        // `count` low bits of `value` at bit `bit` of `into` (little-endian), the other bits kept.
        static void Put(Span<byte> into, long bit, int count, ulong value)
        {
            for (int i = 0; i < count; i++, bit++)
            {
                int mask = 1 << (int)(bit & 7);
                into[(int)(bit >> 3)] = (byte)((value >> i & 1) != 0 ? into[(int)(bit >> 3)] | mask : into[(int)(bit >> 3)] & ~mask);
            }
        }
    }

    internal static void LaneBits(Device device)
    {
        if (device.Backend is not CpuBackend || !FirstRun(nameof(LaneBits)))
        {
            return;
        }

        var hashes = new Dictionary<string, System.Security.Cryptography.IncrementalHash>(StringComparer.Ordinal);
        var lanesOf = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var order = new List<(string Key, string Lane)>();
        void Add(string key, string lane, ReadOnlySpan<byte> bytes)
        {
            string id = key + "\0" + lane;
            if (!hashes.TryGetValue(id, out var hash))
            {
                hashes[id] = hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
                order.Add((key, lane));
            }

            hash.AppendData(bytes);
        }

        int[] rows = [1, 3, 4, 5, 9, 70];
        var formats = new (string Name, IElementType Type, int K, int N)[]
        {
            ("q4_k", Idrak.Types.Q4_K.Type, 512, 37), ("q4_0", Idrak.Types.Q4_0.Type, 320, 37), ("mxfp4", Idrak.Types.MXFP4.Type, 320, 37),
            ("q6_k", Idrak.Types.Q6_K.Type, 768, 37), ("bfloat16", Idrak.Abstraction.Types.BF16.Type, 520, 37), ("bfloat16", Idrak.Abstraction.Types.BF16.Type, 4200, 9),
            ("int64 (no row scales)", Idrak.Abstraction.Types.I64.Type, 544, 37), ("float64", Idrak.Abstraction.Types.F64.Type, 2600, 9), ("int2", Idrak.Types.Int2.Type, 320, 37),
        };
        var described = formats.Select(f => (f.Name, f.Type.Description, f.K, f.N, Seed: $"{f.Name} [{f.K}, {f.N}]"))
            .Concat(LaneBitsDescribedOnly().Select(t => (t.Name, t.D, t.K, N: 37, Seed: t.Name)));
        foreach (var (name, d, k, n, seed) in described)
        {
            var random = new Random(LaneBitsSeed(seed));
            var bytes = LaneBitsBytes(d, k, n, random);
            var words = new float[(bytes.Length + 3) / 4];
            Buffer.BlockCopy(bytes, 0, words, 0, bytes.Length);
            Add($"{name} stored", "", bytes);
            var lanes = CpuBackend.DescribedLanesOf(d, k);
            Check(lanes.Count >= 1, $"{name} [{k}]: no described lane runs here");
            foreach (int m in rows)
            {
                var x = LaneBitsValues(random, (long)m * k);
                for (int lane = 0; lane < lanes.Count; lane++)
                {
                    var y = new float[m * n];
                    Check(CpuBackend.DescribedRowLane(lane, d, words, words.Length, x, y, m, n, k), $"{name} [{k}, {n}], {m} rows: its {lanes[lane]} does not run here");
                    Add($"{name} described", lanes[lane], System.Runtime.InteropServices.MemoryMarshal.AsBytes(y.AsSpan()));
                }
            }
        }

        // The new design's operations, each by every width that runs here (forced in turn).
        var operations = LaneBitsOperations(device);
        foreach (var lanes in Enum.GetValues<CpuBackend.Lanes>().Where(CpuBackend.LanesRun))
        {
            CpuBackend.ForcedLanes = lanes;
            try
            {
                foreach (var (name, run) in operations)
                {
                    Add(name, CpuBackend.LanesName(lanes), System.Runtime.InteropServices.MemoryMarshal.AsBytes(run().AsSpan()));
                }
            }
            finally
            {
                CpuBackend.ForcedLanes = null;
            }
        }

        var made = new SortedDictionary<string, string>(StringComparer.Ordinal);
        var differ = new List<string>();
        foreach (var (key, lane) in order)
        {
            using var hash = hashes[key + "\0" + lane];
            string hex = Convert.ToHexString(hash.GetHashAndReset())[..16];
            made.TryAdd(key, hex);
            if (!LaneBitsRecorded.TryGetValue(key, out var recorded) || recorded != hex)
            {
                differ.Add($"{key}{(lane.Length > 0 ? $" by the {lane}" : "")}: the bits hash to {hex}, recorded {recorded ?? "nothing"}");
            }
        }

        differ.AddRange(LaneBitsRecorded.Keys.Where(key => !made.ContainsKey(key)).Select(key => $"{key}: recorded, not hashed here"));
        Check(differ.Count == 0, $"{differ.Count} of {order.Count} lanes differ from the recorded bits: {string.Join("; ", differ)}"
            + $"{Environment.NewLine}the hashes made here:{Environment.NewLine}{string.Join(Environment.NewLine, made.Select(e => $"        [\"{e.Key}\"] = \"{e.Value}\","))}");
        Console.WriteLine($"    {order.Count} lanes hashed (each by every instruction set and width that runs here), every one the recorded bits of its kind");
    }

    // The operations hashed by every width: fixed operands (seeded), each a function giving its output.
    private static (string Name, Func<float[]> Run)[] LaneBitsOperations(Device device)
    {
        var random = new Random(LaneBitsSeed("operations"));
        const int Rows = 3, Cols = 1037;
        float[] xs = LaneBitsValues(random, Rows * Cols), gs = LaneBitsValues(random, Rows * Cols), gain = LaneBitsValues(random, Cols), bias = LaneBitsValues(random, Cols);
        float[] a = LaneBitsValues(random, 9 * 1037), b = LaneBitsValues(random, 1037 * 37);
        const int Heads = 2, Steps = 5, Capacity = 16, Dim = 72;
        float[] q = LaneBitsValues(random, Heads * Steps * Dim), keys = LaneBitsValues(random, Heads * Capacity * Dim), values = LaneBitsValues(random, Heads * Capacity * Dim);
        var backend = device.Backend;
        float[] Run(Func<Tensor> make)
        {
            using var scope = new TensorScope();
            return make().ToArray();
        }

        Tensor X() => Tensor.From(xs, [Rows, Cols], device);
        Tensor G() => Tensor.From(gs, [Rows, Cols], device);
        float[] Unary(UnaryOp op, float[] input)
        {
            using var x = Tensor.From(input, [input.Length], device);
            using var y = Tensor.Zeros([input.Length], device);
            backend.Unary(op, x.Storage, y.Storage, input.Length);
            return y.ToArray();
        }

        float[] squares = [.. xs.Select(v => v * v + 1e-3f)];
        return
        [
            ("operation: exp", () => Unary(UnaryOp.Exp, xs)),
            ("operation: log", () => Unary(UnaryOp.Log, squares)),
            ("operation: tanh", () => Unary(UnaryOp.Tanh, xs)),
            ("operation: sigmoid", () => Unary(UnaryOp.Sigmoid, xs)),
            ("operation: silu", () => Unary(UnaryOp.Silu, xs)),
            ("operation: gelu", () => Unary(UnaryOp.Gelu, xs)),
            ("operation: exact gelu", () => Unary(UnaryOp.GeluErf, xs)),
            ("operation: sqrt", () => Unary(UnaryOp.Sqrt, squares)),
            ("operation: binary", () => Run(() => X() * G() + X())),
            ("operation: softmax", () => Run(() => X().Softmax())),
            ("operation: log-softmax", () => Run(() => X().LogSoftmax())),
            ("operation: rms norm with a gain", () => Run(() =>
            {
                var y = Tensor.Zeros([Rows, Cols], device);
                using var g = Tensor.From(gain, [Cols], device);
                backend.RmsNormAffine(X().Storage, g.Storage, y.Storage, Rows, Cols, 1e-6f, 1f);
                return y;
            })),
            ("operation: layer norm", () => Run(() =>
            {
                var y = Tensor.Zeros([Rows, Cols], device);
                using var g = Tensor.From(gain, [Cols], device);
                using var c = Tensor.From(bias, [Cols], device);
                backend.LayerNormFused(X().Storage, g.Storage, c.Storage, y.Storage, Rows, Cols, 1e-5f);
                return y;
            })),
            ("operation: gated silu", () => Run(() => Tensor.GatedActivation(X(), G(), Idrak.Abstraction.Generation.FeedForwardActivation.Silu))),
            ("operation: float32 product", () => Run(() => Tensor.From(a, [9, 1037], device).MatMul(Tensor.From(b, [1037, 37], device)))),
            ("operation: sum", () => Run(() => Tensor.From(LaneBitsValues(new Random(5), 70_001), [70_001], device).Sum())),
            ("operation: cached attention", () => Run(() =>
            {
                var y = Tensor.Zeros([Heads * Steps, Dim], device);
                using var tq = Tensor.From(q, [Heads * Steps * Dim], device);
                using var tk = Tensor.From(keys, [Heads * Capacity * Dim], device);
                using var tv = Tensor.From(values, [Heads * Capacity * Dim], device);
                using var position = Tensor.From([7f], [1], device);
                backend.AttentionDecode(tq.Storage, tk.Storage, tv.Storage, position.Storage, y.Storage, Heads, Steps, Steps, Capacity, Dim, 0.125f);
                return y;
            })),
        ];
    }

    // The hashes LaneBits makes, recorded (the CPU's lanes give these bits on any CPU): the described lane's from commit
    // 3f1c53e2 (each type's described and stored bits), the operations' from the new design.
    private static readonly Dictionary<string, string> LaneBitsRecorded = new(StringComparer.Ordinal)
    {
        ["16-bit integers described"] = "0AD092E40E7B6807",
        ["16-bit integers plus 0.5 described"] = "FA7F6C3A35719AF7",
        ["16-bit integers plus 0.5 stored"] = "DDFA6AEA60DCB501",
        ["16-bit integers stored"] = "31418B4D8B89086B",
        ["3-bit codes in two planes, offset -4, an E8M0 scale a 32 values described"] = "C5324487544045EC",
        ["3-bit codes in two planes, offset -4, an E8M0 scale a 32 values stored"] = "29A6BA49DF85CA69",
        ["32-bit integers described"] = "98069D2D957CA29D",
        ["32-bit integers stored"] = "63B5B7D922004311",
        ["a table of 16 fractions, two a byte, a bfloat16 scale a 64 values described"] = "BE2AA77B3342A1D3",
        ["a table of 16 fractions, two a byte, a bfloat16 scale a 64 values stored"] = "592ACEBFC8208E4C",
        ["bfloat16 described"] = "BC5B9EE8B5F7CE6C",
        ["bfloat16 stored"] = "12F3BE79F0791B5E",
        ["float64 described"] = "EA0FB746D0C4EF70",
        ["float64 stored"] = "93B42CD81DA75319",
        ["fp6 E2M3 in a 4-bit and a 2-bit plane, an E8M0 scale a 32 values described"] = "0FFE8FC7F31DA651",
        ["fp6 E2M3 in a 4-bit and a 2-bit plane, an E8M0 scale a 32 values stored"] = "11D168686B5532DA",
        ["int2 described"] = "C97696BC06D308F5",
        ["int2 stored"] = "B9A90C0EB7780762",
        ["mxfp4 described"] = "1595EB830B310EBE",
        ["mxfp4 stored"] = "049001E9829624B5",
        ["q4_0 described"] = "F9E0B7D12B7D7562",
        ["q4_0 stored"] = "2E9C5934D1D6F910",
        ["q4_k described"] = "F842F6266EE5BA9A",
        ["q4_k stored"] = "882CDABBE2B2BC5E",
        ["q6_k described"] = "71B0B16E2747CB5B",
        ["q6_k stored"] = "00ED6D2AADE10493",
        ["int64 (no row scales) described"] = "D7B50979F241C3B2",
        ["int64 (no row scales) stored"] = "59711973EE997480",
        ["operation: binary"] = "2ED7FE9F02A4A134",
        ["operation: cached attention"] = "4BC9EF71A6784112",
        ["operation: exact gelu"] = "AE1AEC36118F23EE",
        ["operation: exp"] = "ADEFBC94B1A9ABAE",
        ["operation: float32 product"] = "21D838DA18B4A7CF",
        ["operation: gated silu"] = "239A5EBA398FF9A6",
        ["operation: gelu"] = "1DC2E478C1A8C069",
        ["operation: layer norm"] = "E2D2BFBD9AD34656",
        ["operation: log"] = "190097B39CB5F2A2",
        ["operation: log-softmax"] = "D5FBAE77E87BA55A",
        ["operation: rms norm with a gain"] = "D6610F7CE0F4CED4",
        ["operation: sigmoid"] = "4B041F24277EA2E1",
        ["operation: silu"] = "0A07F04247DC563F",
        ["operation: softmax"] = "3F4DDA1D5F99B998",
        ["operation: sqrt"] = "B1318596A21FC3B1",
        ["operation: sum"] = "F4A6E7DCBB9D711F",
        ["operation: tanh"] = "E574758ED5F50E6D",
    };
}
