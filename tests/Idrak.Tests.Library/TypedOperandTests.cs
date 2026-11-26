// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Devices.Cpu;

// Typed operands on the CPU: an input of any type read through its type a chunk at a time (the same bits as its values as
// float32), values float32 cannot hold computed in float64 (never narrowed), the 16-bit floats' native lane (the same bits
// as the type's own read), and proof 4: no operation makes a wider copy of an operand.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TypedOperandGroup =
    [
        ("typed operands: the element-wise and row operations, the sums, Copy2D, ArgMax and the float32 product read an input of every built-in type through its type a chunk at a time (rows across its units, past two chunks): the same bits as on its values as float32", TypedInputsReadThroughTheirTypes),
        ("typed operands: values float32 cannot hold (float64, int64, int32, uint32, uint64) are computed in float64 by unary, binary, affine, the sums and the product, never narrowed (the float64 result rounded once); an operation without a float64 lane refuses them, naming itself", WideInputsInFloat64),
        ("typed operands: the 16-bit floats' native lane (bfloat16, float16 by their bits) reads every one of the 65,536 codes as the type's own read does, bit for bit, and Gather gives the same rows by either read", NativeKindLaneReads),
        ("typed operands: proof 4: no operation makes a wider copy of an operand: once warm, the managed bytes Unary, Sum, Softmax, Gather, MatMul and StoredMatMul allocate on an input of each built-in type stay below a quarter of the input's values as float32", NoWiderCopies),
    ];

    // The built-in types the encoder writes whose values float32 holds (the float32 path), and those it does not (float64).
    private static IEnumerable<IElementType> TypedWritable(bool wide) =>
        BuiltInTypes.All.Types.Where(t => t.Unwritable is null && CpuBackend.WideValues(t) == wide && t.Native != NativeKind.Float32);

    // A typed storage of `count` values written as `type` from values in [-2, 2] (their read values as float32 too).
    private static (Tensor Typed, Tensor Plain) TypedPair(IElementType type, int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = random.NextSingle() * 4f - 2f;
        }

        var typed = Tensor.Stored(type, values, [count], Device.Cpu);
        return (typed, Tensor.From(typed.ToArray(), [count], Device.Cpu));
    }

    // Runs `op` on the typed input and on its values as float32 into outputs of `outputs` floats: the same bits.
    private static void SameOnTyped(string what, Tensor typed, Tensor plain, int outputs, Action<Storage, Storage> op)
    {
        var backend = Device.Cpu.Backend;
        var a = backend.Allocate(Math.Max(1, outputs), zeroed: true);
        var b = backend.Allocate(Math.Max(1, outputs), zeroed: true);
        try
        {
            op(typed.Storage, a);
            op(plain.Storage, b);
            var (x, y) = (new float[outputs], new float[outputs]);
            backend.Download(a, x);
            backend.Download(b, y);
            int at = Enumerable.Range(0, outputs).FirstOrDefault(i => BitConverter.SingleToInt32Bits(x[i]) != BitConverter.SingleToInt32Bits(y[i]), -1);
            Check(at < 0, $"{what}: output {at} is {(at < 0 ? 0 : x[at])} from the typed input, {(at < 0 ? 0 : y[at])} from its values as float32");
        }
        finally
        {
            a.Release();
            b.Release();
        }
    }

    internal static void TypedInputsReadThroughTheirTypes(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const int Cols = 300, Rows = 134, Count = Rows * Cols;                       // rows across units; past two chunks
        var backend = Device.Cpu.Backend;
        int checkedTypes = 0;
        foreach (var type in TypedWritable(wide: false))
        {
            using var scope = new TensorScope();
            var (typed, plain) = TypedPair(type, Count, 7);
            var other = Tensor.From(RandomArray(new Random(8), Count), [Count], Device.Cpu);
            var gain = Tensor.From(RandomArray(new Random(9), Cols), [Cols], Device.Cpu);
            string name = type.Name;
            SameOnTyped($"{name}: unary exp", typed, plain, Count, (x, y) => backend.Unary(UnaryOp.Exp, x, y, Count));
            SameOnTyped($"{name}: unary SiLU", typed, plain, Count, (x, y) => backend.Unary(UnaryOp.Silu, x, y, Count));
            SameOnTyped($"{name}: binary add", typed, plain, Count, (x, y) => backend.Binary(BinaryOp.Add, x, other.Storage, y, Count));
            SameOnTyped($"{name}: binary maximum (typed second)", typed, plain, Count, (x, y) => backend.Binary(BinaryOp.Maximum, other.Storage, x, y, Count));
            SameOnTyped($"{name}: affine", typed, plain, Count, (x, y) => backend.Affine(x, y, Count, 1.5f, -0.25f));
            SameOnTyped($"{name}: sum", typed, plain, 1, (x, y) => backend.Sum(x, y, Count, 0.5f));
            SameOnTyped($"{name}: sum of squares", typed, plain, 1, (x, y) => backend.SumSquares(x, y, Count));
            SameOnTyped($"{name}: softmax", typed, plain, Count, (x, y) => backend.Softmax(x, y, Rows, Cols, false));
            SameOnTyped($"{name}: RMS norm with a gain", typed, plain, Count, (x, y) => backend.RmsNormAffine(x, gain.Storage, y, Rows, Cols, 1e-6f, 0f));
            SameOnTyped($"{name}: layer norm", typed, plain, Count, (x, y) => backend.LayerNormFused(x, gain.Storage, gain.Storage, y, Rows, Cols, 1e-5f));
            SameOnTyped($"{name}: row vector", typed, plain, Count, (x, y) => backend.AddRowVector(x, gain.Storage, y, Rows, Cols));
            SameOnTyped($"{name}: arg max", typed, plain, Rows, (x, y) => backend.ArgMax(x, y, Rows, Cols));
            SameOnTyped($"{name}: copy 2d (a narrowed block)", typed, plain, 40 * 17, (x, y) => backend.Copy2D(x, 3 * Cols + 5, Cols, y, 0, 17, 40, 17, false));
            SameOnTyped($"{name}: the product, B typed", typed, plain, 3 * Cols, (x, y) => backend.MatMul(other.Storage, x, y, 3, Cols, Rows, false, false, 0f));
            SameOnTyped($"{name}: the product, B typed and transposed", typed, plain, 3 * Rows, (x, y) => backend.MatMul(other.Storage, x, y, 3, Rows, Cols, false, true, 0f));
            SameOnTyped($"{name}: the product, A typed", typed, plain, Rows * 5, (x, y) => backend.MatMul(x, other.Storage, y, Rows, 5, Cols, false, false, 0f));
            checkedTypes++;
        }

        Check(checkedTypes >= 10, $"{checkedTypes} built-in types checked");
    }

    internal static void WideInputsInFloat64(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var backend = Device.Cpu.Backend;
        using var scope = new TensorScope();
        foreach (var type in TypedWritable(wide: true))
        {
            // Values float32 does not hold: (2^b + 1) and −2^b (their sum 1; 0 where they were narrowed first), b = 40, or 30
            // for a 32-bit integer.
            var code = type.Description.Code;
            double Big = code.Kind == NumberKind.Integer && code.Bits <= 32 ? 1073741824d : 1099511627776d;
            bool signed = code.Signed;
            double low = signed ? -Big : 0, high = Big + 1;
            var bytes = new byte[type.ByteCount(2)];
            type.Write([high, low], bytes);
            var x = Tensor.Stored(type, bytes, [2], Device.Cpu);
            var y = Tensor.Empty([2], Device.Cpu);
            var one = Tensor.Empty([1], Device.Cpu);
            backend.Sum(x.Storage, one.Storage, 2, 1f);
            float expected = (float)(high + low);
            Check(one.ToArray()[0] == expected, $"{type.Name}: the sum of (2^40 + 1) and {low} is {one.ToArray()[0]}, {expected} in float64 (never narrowed)");
            backend.Affine(x.Storage, y.Storage, 2, 1f, (float)-Big);
            Check(y.ToArray()[0] == 1f, $"{type.Name}: affine x − 2^40 of 2^40 + 1 is {y.ToArray()[0]}, 1 in float64");
            var minus = Tensor.From([(float)-Big, 0f], [2], Device.Cpu);
            backend.Binary(BinaryOp.Add, x.Storage, minus.Storage, y.Storage, 2);
            Check(y.ToArray()[0] == 1f, $"{type.Name}: binary (2^40 + 1) + (−2^40) is {y.ToArray()[0]}, 1 in float64");
            var ones = Tensor.From([1f, 1f], [1, 2], Device.Cpu);
            var product = Tensor.Empty([1], Device.Cpu);
            backend.MatMul(ones.Storage, x.Storage, product.Storage, 1, 1, 2, false, false, 0f);
            Check(product.ToArray()[0] == expected, $"{type.Name}: the product [1 1] · [2^40 + 1, {low}] is {product.ToArray()[0]}, {expected} in float64");
            backend.Unary(UnaryOp.Abs, x.Storage, y.Storage, 2);
            Check(y.ToArray()[0] == (float)high, $"{type.Name}: |2^40 + 1| rounded once");
            var refused = Refusal(() => backend.Softmax(x.Storage, y.Storage, 1, 2, false));
            Check(refused?.Contains("never narrows", StringComparison.Ordinal) == true && refused.Contains("Softmax", StringComparison.Ordinal),
                $"{type.Name}: softmax (no float64 lane) refuses, naming itself: {refused ?? "ran"}");
        }

        static string? Refusal(Action run)
        {
            try
            {
                run();
                return null;
            }
            catch (NotSupportedException e)
            {
                return e.Message;
            }
        }
    }

    internal static void NativeKindLaneReads(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        foreach (var type in BuiltInTypes.All.Types.Where(t => t.Native is NativeKind.BFloat16 or NativeKind.Float16))
        {
            var codes = new ushort[65536];
            for (int c = 0; c < codes.Length; c++)
            {
                codes[c] = (ushort)c;
            }

            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(codes.AsSpan()).ToArray();
            var expected = new float[codes.Length];
            type.Read(bytes, expected);
            var actual = new float[codes.Length];
            CpuBackend.NativeRead(type, codes, actual);

            int at = Enumerable.Range(0, codes.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]), -1);
            Check(at < 0, $"{type.Name}: code 0x{(at < 0 ? 0 : at):X4} reads 0x{(at < 0 ? 0 : BitConverter.SingleToInt32Bits(actual[at])):X8} by the kind's lane, 0x{(at < 0 ? 0 : BitConverter.SingleToInt32Bits(expected[at])):X8} by the type's read");

            // Gather over a table of the type: the same rows by either read.
            using var scope = new TensorScope();
            var table = Tensor.Stored(type, bytes, [256, 256], Device.Cpu);
            var ids = Tensor.Indices([0, 5, 127, 200, 255, 3], Device.Cpu);
            var rows = new float[2][];
            foreach (bool native in new[] { false, true })
            {
                CpuBackend.ForcedNativeRead = native;
                try
                {
                    rows[native ? 1 : 0] = table.EmbeddingLookup(ids).ToArray();
                }
                finally
                {
                    CpuBackend.ForcedNativeRead = null;
                }
            }

            Check(rows[0].Select(BitConverter.SingleToInt32Bits).SequenceEqual(rows[1].Select(BitConverter.SingleToInt32Bits)), $"{type.Name}: Gather's rows by either read");
        }
    }

    internal static void NoWiderCopies(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const int Cols = 512, Rows = 128, Count = Rows * Cols;
        var backend = Device.Cpu.Backend;
        var report = new List<string>();
        foreach (var type in TypedWritable(wide: false))
        {
            using var scope = new TensorScope();
            var (typed, _) = TypedPair(type, Count, 11);
            var y = Tensor.Empty([Count], Device.Cpu);
            var x = Tensor.From(RandomArray(new Random(12), 4 * Cols), [4, Cols], Device.Cpu);
            var ids = Tensor.Indices([0, 7, 64, 127], Device.Cpu);
            long bound = (long)Count * sizeof(float) / 4;
            foreach (var (name, run) in new (string, Action)[]
                     {
                         ("Unary", () => backend.Unary(UnaryOp.Exp, typed.Storage, y.Storage, Count)),
                         ("Sum", () => backend.Sum(typed.Storage, y.Storage, Count, 1f)),
                         ("Softmax", () => backend.Softmax(typed.Storage, y.Storage, Rows, Cols, false)),
                         ("Gather", () => backend.Gather(typed.Storage, ids.Storage, y.Storage, 4, Cols, Rows)),
                         ("MatMul", () => backend.MatMul(x.Storage, typed.Storage, y.Storage, 4, Rows, Cols, false, true, 0f)),
                         ("StoredMatMul", () => backend.StoredMatMul(x.Storage, typed.Storage, y.Storage, 4, Rows, Cols)),
                     })
            {
                run();
                run();                                                                // warm: measured, compiled, buffers made
                long least = long.MaxValue;
                for (int round = 0; round < 3; round++)
                {
                    long before = GC.GetTotalAllocatedBytes(precise: true);
                    run();
                    least = Math.Min(least, GC.GetTotalAllocatedBytes(precise: true) - before);
                }

                report.Add($"{type.Name} {name} {least} B");
                Check(least < bound, $"{type.Name}: {name} allocated {least} bytes once warm, a wider copy of the input being {(long)Count * sizeof(float)} (bound {bound})");
            }
        }

        Console.WriteLine($"  proof 4, bytes allocated once warm (input {Count} values, {(long)Count * sizeof(float)} B as float32): {string.Join("; ", report)}");
    }
}
