// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Types;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Plugins.Qwen;

// The type abstraction (IElementType): every type, the library's and any other, read and written through the interface's
// defaults from its description alone. The proofs: alien descriptions (a generator of types no file uses) read bit for bit
// as their descriptions decode; the encoder writes each of them, every code reachable, its error measured against the
// least-squares bound of the codes it chose; a type defined outside the library read, written and multiplied by the CPU.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TypeGroup =
    [
        ("types: the built-in list holds each type once, each valid and written, its native kind from its description; every description equal by value to a copy of itself and to no other type's", TypesBuiltInList),
        ("types: every built-in and alien type read through the default matches its description's own decode, bit for bit", TypesReadAsDescribed),
        ("types: the arithmetic rounding of wide codes is the table's rounding on every narrow float and integer", TypesWideRounding),
        ("types: the encoder writes every alien type: finite, every code reachable, its error against the least-squares bound", TypesEncoderWrites),
        ("types: wide scale levels' numbers ordered, stored and rounded by their own arithmetic", TypesWideLevels),
        ("types: Q8_1's and Q8_K's sums written from the values stored and read back", TypesSums),
        ("types: int8 and int16 rows carry their own float32 scale, equal by description a length, read, written and multiplied", TypesRowScale),
        ("types: 32- and 64-bit machine formats written from float64 and read back without narrowing", TypesNoNarrowing),
        ("types: a type defined outside the library is read, written and multiplied through the interface alone", TypesOutside),
        ("types: a typed tensor holds its type's bytes and reads back through its type", TypesTypedTensors),
        ("types: safetensors tensors read as their dtypes' types, their bytes as stored", TypesSafetensors),
        ("types: a Qwen3 checkpoint in bfloat16 loads its projections and table as stored, or written as any type", TypesLoadTyped),
        ("types: norm gains and biases load as their checkpoint stores them (bfloat16 kept, never written as the weights' type), the logits and greedy text a float32 twin's", TypesStoredVectors),
    ];

    // A type given by a description (the aliens below, and the type defined outside the library): the interface's two
    // members, every other one its default.
    private sealed class TestType(string name, TypeDescription description) : IElementType
    {
        public TypeDescription Description { get; } = description;

        public string Name { get; } = name;

        public override string ToString() => Name;
    }

    private static readonly NumberType HalfNumber = NumberType.Floating(5, 10);

    private static readonly NumberType E4M3Number = NumberType.Floating(4, 3, specials: FloatSpecials.AllOnesNaN);

    private static readonly NumberType F32Number = NumberType.Floating(8, 23);

    // Alien descriptions: per-element codes of 3, 5, 6, 7 and 24 bits packed tight, floats of 3, 5, 6 and 24 bits, 3- and
    // 5-bit block codes with E4M3 or 6-bit integer scales, codes over 3 planes, digits of radix 5 and 7, a 12-bit codebook,
    // three levels of scales (one shared over a group of blocks), a minimum of its own, E2M1 codes under E4M3 scales; float32
    // levels (the coarsest, a child under a float16, a minimum) and a float32 scale shared over a group of 128 blocks.
    private static IReadOnlyList<IElementType> AlienTypes()
    {
        var book = new Random(12);
        var codebook = NumberType.Codebook(12, 2, [.. Enumerable.Range(0, 2 * 4096).Select(_ => Math.Round(book.NextDouble() * 2 - 1, 3))]);
        return
        [
            new TestType("int3 (8 in 3 bytes)", new(8, 3, NumberType.Integer(3, signed: true), [new CodePlane(0, 3, 1, Stream: true)])),
            new TestType("int5 (8 in 5 bytes)", new(8, 5, NumberType.Integer(5, signed: true), [new CodePlane(0, 5, 1, Stream: true)])),
            new TestType("int6 (4 in 3 bytes)", new(4, 3, NumberType.Integer(6, signed: true), [new CodePlane(0, 6, 1, Stream: true)])),
            new TestType("uint7 (8 in 7 bytes)", new(8, 7, NumberType.Integer(7, signed: false), [new CodePlane(0, 7, 1, Stream: true)])),
            new TestType("int24", new(1, 3, NumberType.Integer(24, signed: true), [new CodePlane(0, 24, 1)])),
            new TestType("fp6 e2m3", new(4, 3, NumberType.Floating(2, 3, specials: FloatSpecials.None), [new CodePlane(0, 6, 1, Stream: true)])),
            new TestType("fp6 e3m2", new(4, 3, NumberType.Floating(3, 2, specials: FloatSpecials.None), [new CodePlane(0, 6, 1, Stream: true)])),
            new TestType("fp5 e2m2", new(8, 5, NumberType.Floating(2, 2, specials: FloatSpecials.None), [new CodePlane(0, 5, 1, Stream: true)])),
            new TestType("fp3 e1m1", new(8, 3, NumberType.Floating(1, 1, specials: FloatSpecials.None), [new CodePlane(0, 3, 1, Stream: true)])),
            new TestType("fp24 e8m15", new(1, 3, NumberType.Floating(8, 15), [new CodePlane(0, 24, 1)])),
            new TestType("int3 blocks of 32, an E4M3 scale", new(32, 13, NumberType.Integer(3, signed: true), [new CodePlane(1, 3, 1, Stream: true)], Scales: [new BlockScale(E4M3Number, 32, 0)])),
            new TestType("5-bit blocks of 64, int6 scales of 16 under a float16", new(64, 45, NumberType.Integer(5, signed: false), [new CodePlane(5, 5, 1, Stream: true)], -16,
                [new BlockScale(HalfNumber, 64, 0), new BlockScale(NumberType.Integer(6, signed: false), 16, 2)])),
            new TestType("6-bit codes over 3 planes", new(32, 26, NumberType.Integer(6, signed: false), [new CodePlane(2, 2, 32), new CodePlane(10, 2, 32), new CodePlane(18, 2, 32)], -32,
                [new BlockScale(HalfNumber, 32, 0)])),
            new TestType("radix-5 digits, 3 a byte", new(24, 10, NumberType.Integer(3, signed: false),
                [new CodePlane(0, 3, 1, Places: [.. Enumerable.Range(0, 24).Select(v => new FieldPlace(2 + v / 3, v % 3))], Radix: 5)], -2, [new BlockScale(HalfNumber, 24, 0)])),
            new TestType("radix-7 digits, 2 a byte", new(16, 9, NumberType.Integer(3, signed: false),
                [new CodePlane(0, 3, 1, Places: [.. Enumerable.Range(0, 16).Select(v => new FieldPlace(1 + v / 2, v % 2))], Radix: 7)], -3, [new BlockScale(E4M3Number, 16, 0)])),
            new TestType("12-bit codebook of pairs", new(32, 26, codebook, [new CodePlane(2, 8, 1), new CodePlane(18, 4, 2)], Scales: [new BlockScale(HalfNumber, 32, 0)])),
            new TestType("three levels of scales", new(256, 133, NumberType.Integer(4, signed: true), [new CodePlane(5, 4, 2)],
                Scales: [new BlockScale(HalfNumber, 512, 266), new BlockScale(NumberType.Integer(8, signed: false), 256, 0), new BlockScale(NumberType.Integer(4, signed: false), 32, 1)], Group: 2, GroupBytes: 268)),
            new TestType("4-bit codes, a float16 scale and a 6-bit minimum", new(32, 19, NumberType.Integer(4, signed: false), [new CodePlane(3, 4, 32)],
                Scales: [new BlockScale(HalfNumber, 32, 0)], Mins: [new BlockScale(NumberType.Integer(6, signed: false), 32, 2)])),
            new TestType("E2M1 codes, 16 a block, an E4M3 scale", new(16, 9, NumberType.Floating(2, 1, bias: 1, specials: FloatSpecials.None), [new CodePlane(1, 4, 16)],
                Scales: [new BlockScale(E4M3Number, 16, 0)])),
            new TestType("int8 blocks of 64, a float32 scale", new(64, 68, NumberType.Integer(8, signed: true), [new CodePlane(4, 8, 1)], Scales: [new BlockScale(F32Number, 64, 0)])),
            new TestType("4-bit codes, 6-bit scales of 16 under a float32", new(64, 39, NumberType.Integer(4, signed: false), [new CodePlane(7, 4, 2)], -8,
                [new BlockScale(F32Number, 64, 0), new BlockScale(NumberType.Integer(6, signed: false), 16, 4)])),
            new TestType("4-bit codes, float32 scales of 16 under a float16", new(64, 50, NumberType.Integer(4, signed: true), [new CodePlane(18, 4, 2)],
                Scales: [new BlockScale(HalfNumber, 64, 0), new BlockScale(F32Number, 16, 2)])),
            new TestType("4-bit codes, a float32 scale and a float32 minimum", new(32, 24, NumberType.Integer(4, signed: false), [new CodePlane(8, 4, 2)],
                Scales: [new BlockScale(F32Number, 32, 0)], Mins: [new BlockScale(F32Number, 32, 4)])),
            new TestType("int8 groups of 128, a float32 scale after them", new(1, 1, NumberType.Integer(8, signed: true), [new CodePlane(0, 8, 1)],
                Scales: [new BlockScale(F32Number, 128, 128)], Group: 128, GroupBytes: 132)),
        ];
    }

    private static void TypesBuiltInList(Device device)
    {
        CpuOnly(device);
        var types = BuiltInTypes.All.Types;
        Check(types.Count == types.Distinct().Count() && types.Select(t => t.Name).Distinct(StringComparer.OrdinalIgnoreCase).Count() == types.Count,
            "each type once, each its own name");
        Check(types.All(t => t.Description.Valid), $"invalid: {string.Join(", ", types.Where(t => !t.Description.Valid))}");
        // Every type written (Q8_K's float32 scale level searched by its own arithmetic).
        var unwritten = types.Where(t => t.Unwritable is not null).ToList();
        Check(unwritten.Count == 0, $"unwritten: {string.Join("; ", unwritten.Select(t => $"{t}: {t.Unwritable}"))}");
        var natives = types.ToDictionary(t => t, t => t.Native);
        Check(natives[F32.Type] == NativeKind.Float32 && natives[F64.Type] == NativeKind.Float64 && natives[F16.Type] == NativeKind.Float16 && natives[BF16.Type] == NativeKind.BFloat16
              && natives[F8E4M3.Type] == NativeKind.Float8E4M3 && natives[F8E5M2.Type] == NativeKind.Float8E5M2 && natives[I8.Type] == NativeKind.Int8 && natives[U8.Type] == NativeKind.UInt8
              && natives[I16.Type] == NativeKind.Int16 && natives[U16.Type] == NativeKind.UInt16 && natives[I32.Type] == NativeKind.Int32 && natives[U32.Type] == NativeKind.UInt32
              && natives[I64.Type] == NativeKind.Int64 && natives[U64.Type] == NativeKind.UInt64,
            "the machine formats' native kinds");
        var none = types.Where(t => t.Native == NativeKind.None).Select(t => t.Name).ToList();
        Check(none.Count == types.Count - 14 && !none.Contains("float32"), $"the rest none: {string.Join(", ", none)}");

        // A description alike to a native one is that native kind, whatever the type says of itself.
        Check(((IElementType)new TestType("anything", BF16.Type.Description with { })).Native == NativeKind.BFloat16, "a native kind from the description alone");

        // A description is its value: each equals a copy of every list it holds (its hash too), and no two types are equal.
        var all = types.Concat(AlienTypes()).ToList();
        var unlike = all.Where(t => t.Description != Copied(t.Description) || t.Description.GetHashCode() != Copied(t.Description).GetHashCode()).ToList();
        Check(unlike.Count == 0, $"not equal to a copy of itself: {string.Join(", ", unlike)}");
        var alike = all.SelectMany((t, i) => all.Skip(i + 1).Where(o => o.Description == t.Description).Select(o => $"{t} and {o}")).ToList();
        Check(alike.Count == 0, $"described alike: {string.Join(", ", alike)}");
        Console.WriteLine($"  {types.Count} built-in types, {types.Count - none.Count} native, every one written; {all.Count} descriptions equal to their copies, no two alike");
    }

    // A description with every list (and every number type's table) copied: equal to the original by value alone.
    private static TypeDescription Copied(TypeDescription d)
    {
        static NumberType Number(NumberType n) => n with { Table = n.Table is null ? null : [.. n.Table] };
        static BlockScale[]? Scales(IReadOnlyList<BlockScale>? scales) => scales is null ? null
            : [.. scales.Select(s => s with { Type = Number(s.Type), Pieces = s.Pieces is null ? null : [.. s.Pieces.Select(p => (IReadOnlyList<ScalePiece>)[.. p])] })];
        return d with
        {
            Code = Number(d.Code),
            Planes = [.. d.Planes.Select(p => p with { Places = p.Places is null ? null : [.. p.Places] })],
            Scales = Scales(d.Scales),
            Mins = Scales(d.Mins),
            Sums = d.Sums is null ? null : [.. d.Sums.Select(s => s with { Type = Number(s.Type) })],
        };
    }

    private static void TypesReadAsDescribed(Device device)
    {
        CpuOnly(device);
        var random = new Random(7);
        long values = 0;
        int count = 0;
        foreach (var type in BuiltInTypes.All.Types.Concat(AlienTypes()))
        {
            var d = type.Description;
            int units = d.UnitValues >= 256 ? 9 : 41;
            var stored = new byte[units * d.UnitBytes];
            random.NextBytes(stored);
            int n = units * d.UnitValues - (d.UnitValues > 1 ? 1 : 0);          // a last group of fewer values
            var floats = new float[n];
            var doubles = new double[n];
            type.Read(stored, floats);
            type.Read(stored, doubles);
            for (int i = 0; i < n; i++)
            {
                double reference = d.ValueOf(stored.AsSpan(i / d.UnitValues * d.UnitBytes, d.UnitBytes), i % d.UnitValues);
                bool same = double.IsNaN(reference) ? double.IsNaN(doubles[i]) && float.IsNaN(floats[i])
                    : BitConverter.DoubleToInt64Bits(reference) == BitConverter.DoubleToInt64Bits(doubles[i])
                      && BitConverter.SingleToInt32Bits((float)reference) == BitConverter.SingleToInt32Bits(floats[i]);
                Check(same, $"{type}: value {i} reads {floats[i]} and {doubles[i]}; its description gives {reference}");
            }

            values += n;
            count++;
        }

        Console.WriteLine($"  {count} types, {values} values: every one the description's own decode, bit for bit (NaN as NaN)");
    }

    private static void TypesWideRounding(Device device)
    {
        CpuOnly(device);
        var random = new Random(3);
        var numbers = new List<NumberType>
        {
            NumberType.Floating(5, 10), NumberType.Floating(8, 7), NumberType.Floating(4, 3, specials: FloatSpecials.AllOnesNaN), NumberType.Floating(5, 2),
            NumberType.Floating(2, 1, bias: 1, specials: FloatSpecials.None), NumberType.Floating(8, 0, signed: false, bias: 127, subnormals: false, specials: FloatSpecials.AllOnesNaN),
            NumberType.Floating(2, 3, specials: FloatSpecials.None), NumberType.Floating(3, 2, specials: FloatSpecials.None), NumberType.Floating(1, 1, specials: FloatSpecials.None),
            NumberType.Floating(4, 3), NumberType.Floating(5, 0), NumberType.Floating(3, 4, signed: false), NumberType.Floating(4, 2, subnormals: false, specials: FloatSpecials.None),
            NumberType.Floating(5, 10, bias: 7), NumberType.Floating(6, 9), NumberType.Floating(4, 11, specials: FloatSpecials.AllOnesNaN),
        };
        numbers.AddRange(Enumerable.Range(1, 16).SelectMany(b => new[] { NumberType.Integer(b, signed: true), NumberType.Integer(b, signed: false) }));
        long samples = 0;
        foreach (var t in numbers)
        {
            var grid = DescribedEncoder.Grid.Of(t, 0)!;
            var wide = WideNumber.Of(t)!;
            var xs = new List<double> { 0, -0.0, double.NaN, -double.NaN, double.PositiveInfinity, double.NegativeInfinity, 1e300, -1e300, 1e-300, -1e-300 };
            var v = grid.Values;
            for (int i = 0; i < v.Length; i++)
            {
                xs.Add(v[i]);
                if (i + 1 < v.Length)
                {
                    double mid = (v[i] + v[i + 1]) / 2;
                    xs.AddRange([mid, Math.BitDecrement(mid), Math.BitIncrement(mid)]);
                }
            }

            double top = v[^1], step = v.Length > 1 ? v[^1] - v[^2] : 1;
            xs.AddRange([top + step / 2, top + step / 2 * 0.999, top + step / 2 * 1.001, -top - step / 2, v[0] - step]);
            for (int i = 0; i < 2000; i++)
            {
                xs.Add((random.NextDouble() * 2 - 1) * Math.Max(Math.Abs(top), Math.Abs(v[0])) * 1.2);
                xs.Add(Math.ScaleB(random.NextDouble() * 2 - 1, random.Next(-40, 20)));
            }

            foreach (double x in xs)
            {
                ulong a = grid.Round(x), b = wide.Round(x);
                Check(a == b, $"{t}: {x.ToString("R", CultureInfo.InvariantCulture)} rounds to code {a:X} by the table, {b:X} by arithmetic");
            }

            samples += xs.Count;
        }

        Console.WriteLine($"  {numbers.Count} number types, {samples} values: the arithmetic rounding is the table's on every one");
    }

    private static void TypesEncoderWrites(Device device)
    {
        CpuOnly(device);
        var random = new Random(11);
        foreach (var type in AlienTypes())
        {
            var d = type.Description;
            Check(type.Unwritable is null, $"{type}: not written: {type.Unwritable}");
            // Values enough for every code to be some value's nearest: 8-bit codes under a scale take more (their outer
            // codes are few values' nearest).
            int values = d.Code.Bits >= 8 && d.Code.Width == 1 && (d.Scales?.Count ?? 0) > 0 ? 1 << 16 : 6144;
            int units = d.UnitValues >= 256 ? Math.Max(24, values / d.UnitValues) : Math.Max(64, values / d.UnitValues), n = units * d.UnitValues;
            var x = new float[n];
            for (int u = 0; u < units; u++)
            {
                double magnitude = Math.Pow(10, random.NextDouble() * 4 - 2);         // each group its own range
                for (int v = 0; v < d.UnitValues; v++)
                {
                    double g = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
                    x[u * d.UnitValues + v] = (float)(random.Next(40) == 0 ? 0 : magnitude * g * (random.Next(16) == 0 ? 4 : 1));
                }
            }

            var stored = new byte[type.ByteCount(n)];
            type.Write(x, stored);
            var y = new float[n];
            type.Read(stored, y);
            Check(y.All(float.IsFinite), $"{type}: a value written is not finite");

            // Every code reachable: the distinct finite numbers of the code (a codebook's entries) each written somewhere.
            var seen = new HashSet<long>();
            for (int i = 0; i < n; i++)
            {
                var unit = stored.AsSpan(i / d.UnitValues * d.UnitBytes, d.UnitBytes);
                ulong code = d.CodeBits(unit, i % d.UnitValues);
                seen.Add(d.Code.Width > 1 ? (long)code : BitConverter.DoubleToInt64Bits(d.Code.Value(code)));
            }

            var all = new HashSet<long>();
            for (ulong c = 0; c < 1UL << d.Code.Bits && d.Code.Bits <= 16; c++)
            {
                double number = d.Code.Value(c);
                if ((d.Code.Width > 1 || double.IsFinite(number)) && Holds(d, c))
                {
                    all.Add(d.Code.Width > 1 ? (long)c : BitConverter.DoubleToInt64Bits(number));
                }
            }

            int codes = all.Count;
            if ((d.Scales?.Count ?? 0) + (d.Mins?.Count ?? 0) == 0)
            {
                codes = DirectReachable(type);                                   // a per-element type: each code written by its own value
            }
            else if (d.Code.Width > 1)
            {
                // A codebook's entries by their own vectors: a block of entry e's values (times 2, a scale the type stores)
                // after the entry holding the codebook's largest first value, each slot of e written back to e's values.
                int width = d.Code.Width, entries = 1 << d.Code.Bits, anchor = 0, exact = 0;
                double largest = double.MinValue;
                for (int e = 0; e < entries; e++)
                {
                    for (int i = 0; i < width; i++)
                    {
                        (anchor, largest) = d.Code.Value((ulong)e, i) > largest ? (e, d.Code.Value((ulong)e, i)) : (anchor, largest);
                    }
                }

                var probe = new float[d.UnitValues];
                var back = new byte[d.UnitBytes];
                var read = new float[d.UnitValues];
                for (int e = 0; e < entries; e++)
                {
                    for (int v = 0; v < d.UnitValues; v++)
                    {
                        probe[v] = (float)d.Code.Value((ulong)(v < width ? anchor : e), v % width) * 2;
                    }

                    type.Write(probe, back);
                    type.Read(back, read);
                    bool same = true;
                    for (int v = width; v < d.UnitValues; v++)
                    {
                        same &= read[v] == probe[v];
                    }

                    exact += same ? 1 : 0;
                    seen.Add((long)d.CodeBits(back, width));
                }

                Console.WriteLine($"  {type}: {exact} of {entries} entries written back exactly from their own vectors");
                Check(exact == entries, $"{type}: {entries - exact} entries not written back from their own vectors");
            }
            else
            {
                var missing = all.Except(seen).Select(b => BitConverter.Int64BitsToDouble(b)).ToList();
                Check(missing.Count == 0, $"{type}: codes never written: {string.Join(", ", missing.Take(8).Select(m => m.ToString("R", CultureInfo.InvariantCulture)))}");
            }

            // The least-squares bound: in each unit (the values the finest scale or minimum covers), the error of the best
            // continuous scale (and minimum) for the codes the encoder chose; the stored, quantized ones can do no better.
            var (error2, bound, energy2) = LeastSquares(d, x, stored);
            bool blocks = (d.Scales?.Count ?? 0) + (d.Mins?.Count ?? 0) > 0;
            Console.WriteLine(blocks
                ? string.Create(CultureInfo.InvariantCulture, $"  {type}: {codes} codes all written; relative error {error2 / energy2:0.000000}, least-squares bound {bound / energy2:0.000000}, ratio {error2 / Math.Max(bound, 1e-300):0.000}")
                : string.Create(CultureInfo.InvariantCulture, $"  {type}: {codes} codes all written; each value rounded to its nearest, relative error {error2 / energy2:0.000000}"));
            Check(error2 < energy2 && error2 >= bound * 0.999999, $"{type}: error {error2} against the bound {bound} and the energy {energy2}");
        }
    }

    // Whether the planes hold code c (a plane of a radix's digits holds digits below the radix).
    private static bool Holds(TypeDescription d, ulong c)
    {
        int low = 0;
        foreach (var plane in d.Planes)
        {
            if (plane.Radix != 0 && (c >> low & ((1UL << plane.Bits) - 1)) >= (ulong)plane.Radix)
            {
                return false;
            }

            low += plane.Bits;
        }

        return true;
    }

    // A per-element type (no scale or minimum): every finite code's own value written, and that code (its number) back.
    private static int DirectReachable(IElementType type)
    {
        var d = type.Description;
        int total = 1 << d.Code.Bits, count = 0;
        var codes = new uint[total];
        var values = new float[total];
        for (uint c = 0; c < total; c++)
        {
            double v = d.CodeOffset == 0 ? d.Code.Value(c) : d.Code.Value(c) + d.CodeOffset;
            if (double.IsFinite(v))
            {
                codes[count] = c;
                values[count++] = (float)v;
            }
        }

        var stored = new byte[type.ByteCount(count)];
        type.Write(values.AsSpan(0, count), stored);
        for (int i = 0; i < count; i++)
        {
            ulong got = d.CodeBits(stored.AsSpan(i / d.UnitValues * d.UnitBytes, d.UnitBytes), i % d.UnitValues);
            Check(BitConverter.DoubleToInt64Bits(d.Code.Value(got)) == BitConverter.DoubleToInt64Bits(d.Code.Value(codes[i])), $"{type}: the value of code {codes[i]:X} written as code {got:X}");
        }

        return count;
    }

    // The squared error of the written values, the least-squares bound of the codes chosen and the values' energy.
    private static (double Error, double Bound, double Energy) LeastSquares(TypeDescription d, float[] x, byte[] stored)
    {
        var factors = (d.Scales ?? []).Where(s => s.Type.Width == 1).Concat(d.Mins ?? []).ToList();
        int unit = factors.Count == 0 ? 1 : factors.Min(s => s.Values);
        bool affine = (d.Mins?.Count ?? 0) > 0;
        double error = 0, bound = 0, energy = 0;
        for (int u0 = 0; u0 < x.Length; u0 += unit)
        {
            double sw = 0, sg = 0, sgg = 0, sy = 0, syg = 0, syy = 0;
            for (int i = u0; i < u0 + unit; i++)
            {
                var block = stored.AsSpan(i / d.UnitValues * d.UnitBytes, d.UnitBytes);
                int v = i % d.UnitValues;
                double g = d.Code.Value(d.CodeBits(block, v), v % d.Code.Width) + d.CodeOffset;
                foreach (var s in (d.Scales ?? []).Where(s => s.Type.Width > 1))
                {
                    g *= s.Value(block[d.ScaleBase(s, v)..], d.ScaleIndex(s, v), v % s.Values);
                }

                double y = x[i], r = y - d.ValueOf(block, v);
                error += r * r;
                energy += y * y;
                sw++;
                sg += g;
                sgg += g * g;
                sy += y;
                syg += y * g;
                syy += y * y;
            }

            double a = sgg > 0 ? syg / sgg : 0, c = 0;
            if (affine && sw * sgg - sg * sg > 1e-12 * sw * sgg)
            {
                a = (sw * syg - sy * sg) / (sw * sgg - sg * sg);
                c = (sy - a * sg) / sw;
            }

            bound += Math.Max(0, syy - 2 * a * syg - 2 * c * sy + a * a * sgg + 2 * a * c * sg + c * c * sw);
        }

        return (error, bound, energy);
    }

    // Wide level numbers (WideLevel): every rank's number stored by its code and rounded back to its rank, ranks in the
    // numbers' order, ±0 apart, any value in range rounded as WideNumber rounds it, past the extremes the extreme.
    private static void TypesWideLevels(Device device)
    {
        CpuOnly(device);
        var random = new Random(13);
        NumberType[] types =
        [
            F32Number, NumberType.Floating(11, 52), NumberType.Floating(8, 11), NumberType.Floating(6, 12, signed: false), NumberType.Floating(8, 15, specials: FloatSpecials.AllOnesNaN),
            NumberType.Integer(32, signed: true), NumberType.Integer(32, signed: false), NumberType.Integer(64, signed: true), NumberType.Integer(64, signed: false), NumberType.Integer(24, signed: true),
        ];
        long checks = 0;
        foreach (var t in types)
        {
            var level = DescribedEncoder.WideLevel.Of(t, 0)!;
            var wide = WideNumber.Of(t)!;
            bool integer = t.Kind == NumberKind.Integer;
            // An integer's ranks where float64 holds its numbers exactly (an unsigned 64-bit one's ranks moved by 2^63).
            long origin = t is { Kind: NumberKind.Integer, Signed: false, Bits: 64 } ? long.MinValue : 0;
            long low = !integer ? level.First : origin == long.MinValue ? origin : Math.Max(level.First, -(1L << 52)), high = integer ? Math.Min(level.Last, origin + (1L << 52)) : level.Last;
            var ranks = new List<long> { low, low + 1, high - 1, high, -2, -1, 0, 1, 2 };
            for (int i = 0; i < 4000; i++)
            {
                ranks.Add(random.NextInt64(low, high));
            }

            foreach (long r in ranks.Where(r => r >= low && r <= high).Distinct())
            {
                double v = level.NumberOf(r);
                Check(double.IsFinite(v) && BitConverter.DoubleToInt64Bits(t.Value(level.CodeOf(r))) == BitConverter.DoubleToInt64Bits(v), $"{t}: rank {r}'s code stores its number {v}");
                Check(level.RankNearest(v) == r, $"{t}: rank {r}'s number {v} rounds to rank {level.RankNearest(v)}");
                if (r < high)
                {
                    double next = level.NumberOf(r + 1);
                    Check(next > v || next == 0 && v == 0 && double.IsNegative(v) && !double.IsNegative(next), $"{t}: rank {r + 1}'s number {next} after rank {r}'s {v}");
                }

                checks++;
            }

            // Values within the finite numbers (past the largest the level keeps its largest, where the arithmetic gives the
            // type's overflow).
            double largest = level.NumberOf(level.Last);
            for (int i = 0; i < 4000; i++)
            {
                double x = integer ? (random.NextDouble() * 2 - 1) * Math.ScaleB(1, t.Bits - 1) : Math.ScaleB(random.NextDouble() * 2 - 1, random.Next(-60, 60));
                x = t.Signed ? x : Math.Abs(x);
                if (!integer && Math.Abs(x) >= largest)
                {
                    continue;
                }

                Check(level.CodeOf(level.RankNearest(x)) == wide.Round(x), $"{t}: {x.ToString("R", CultureInfo.InvariantCulture)} rounds to code {level.CodeOf(level.RankNearest(x)):X}, the arithmetic's {wide.Round(x):X}");
                checks++;
            }

            Check(level.RankNearest(double.PositiveInfinity) == level.Last && level.RankNearest(double.MaxValue) == level.Last && level.RankNearest(double.NaN) == level.RankNearest(0)
                  && level.RankNearest(double.NegativeInfinity) == level.First, $"{t}: past the extremes the extreme, NaN as 0");
        }

        Console.WriteLine($"  {types.Length} wide number types, {checks} ranks and values: each rank's code its number, ordered, and rounded as the arithmetic rounds");
    }

    // Q8_1's and Q8_K's sums: written from the values the encoder stored (Q8_1's s their sum, d · Σq, as float16; Q8_K's 16
    // sums of 16 codes, signed 16-bit), read back by the description's sums; Q8_K's scale a float32 of its own.
    private static void TypesSums(Device device)
    {
        CpuOnly(device);
        var random = new Random(14);
        foreach (var type in new[] { Idrak.Types.Q8_1.Type, Idrak.Types.Q8_K.Type })
        {
            var d = type.Description;
            int blocks = 48, n = blocks * d.Values;
            var x = new float[n];
            for (int b = 0; b < blocks; b++)
            {
                double magnitude = Math.Pow(10, random.NextDouble() * 4 - 2);
                for (int v = 0; v < d.Values; v++)
                {
                    x[b * d.Values + v] = (float)(magnitude * (random.NextDouble() * 2 - 1 + (b % 3 == 0 ? 0.4 : 0)));
                }
            }

            var stored = new byte[type.ByteCount(n)];
            type.Write(x, stored);
            var y = new float[n];
            type.Read(stored, y);
            double error = 0, energy = 0;
            for (int i = 0; i < n; i++)
            {
                error += (x[i] - y[i]) * (double)(x[i] - y[i]);
                energy += x[i] * (double)x[i];
            }

            Check(error < 1e-4 * energy, $"{type}: relative error {error / energy}");
            int sums = 0, nonzero = 0, ownScales = 0;
            for (int b = 0; b < blocks; b++)
            {
                var block = stored.AsSpan(b * d.Bytes, d.Bytes);
                double scale = d.Scales![0].Value(block, 0);
                ownScales += (double)(Half)scale != scale ? 1 : 0;
                foreach (var sum in d.Sums!)
                {
                    for (int i = 0; i < d.Values / sum.Values; i++)
                    {
                        long codes = 0;
                        for (int v = i * sum.Values; v < (i + 1) * sum.Values; v++)
                        {
                            codes += (long)d.Code.Value(d.CodeBits(block, v));
                        }

                        double expected = sum.OfValues ? (double)(Half)(scale * codes) : codes;
                        Check(sum.Value(block, i) == expected, $"{type}: block {b}'s sum {i} reads {sum.Value(block, i)}; its {(sum.OfValues ? "values'" : "codes'")} sum is {expected}");
                        sums++;
                        nonzero += expected != 0 ? 1 : 0;
                    }
                }
            }

            Check(nonzero > sums / 2, $"{type}: {nonzero} of {sums} sums nonzero");
            Check(type == Idrak.Types.Q8_1.Type || ownScales > blocks / 2, $"{type}: {ownScales} of {blocks} scales float32 of their own (not a float16's)");
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"  {type}: {sums} sums written and read back as the values stored give them, relative error {error / energy:0.0000000}"));
        }
    }

    // int8 and int16 rows: each row a block of its float32 scale and its codes (RowScale, ForRow), equal by description a
    // length (an object each ask), read, written and multiplied by the CPU through the description alone.
    private static void TypesRowScale(Device device)
    {
        CpuOnly(device);
        var random = new Random(15);
        Check(F16.Type.ForRow(96) == F16.Type && Idrak.Types.Q8_0.Type.ForRow(96) == Idrak.Types.Q8_0.Type, "a type whose description holds for any row is its own row's");
        var wrong = new List<string>();
        foreach (var (element, k) in new[] { (I8.Type, 96), (I8.Type, 64), (I8.Type, 2048), (I16.Type, 96), (I16.Type, 2048) })
        {
            int m = 3, n = 5, bytes = element.Description.Bytes, rowBytes = k * bytes + 4;
            var row = element.ForRow(k);
            var again = element.ForRow(k);
            Check(row.Description == again.Description && row.Description.GetHashCode() == again.Description.GetHashCode() && row.Description != element.ForRow(k + 32).Description
                  && row.ForRow(k) == row && row.ForRow(k + 32).Description == element.ForRow(k + 32).Description && row.Description == (row.Description with { Planes = [.. row.Description.Planes] }),
                $"{element}: rows of one length describe themselves alike, by value (whatever object each is), another length not");
            Check(row.Description.Valid && row.Native == NativeKind.None && row.Unwritable is null && row.ByteCount((long)n * k) == n * rowBytes,
                $"{row}: valid, read through its description, written, {row.ByteCount((long)n * k)} bytes for {n} rows");
            var w = new float[n * k];
            for (int r = 0; r < n; r++)
            {
                double magnitude = Math.Pow(10, random.NextDouble() * 4 - 3);
                for (int j = 0; j < k; j++)
                {
                    w[r * k + j] = (float)(magnitude * (random.NextDouble() * 2 - 1));
                }
            }

            var stored = new byte[row.ByteCount(w.Length)];
            row.Write(w, stored);
            var back = new float[w.Length];
            row.Read(stored, back);
            double error = 0, energy = 0;
            for (int r = 0; r < n; r++)
            {
                double scale = BitConverter.ToSingle(stored, r * rowBytes);
                for (int j = 0; j < k; j++)
                {
                    double code = element.Description.ValueOf(stored.AsSpan(r * rowBytes + 4 + j * bytes, bytes), 0);
                    Check(back[r * k + j] == (float)(code * scale), $"{row}: row {r}'s value {j} is its code {code} times the row's scale {scale}");
                    error += (w[r * k + j] - back[r * k + j]) * (double)(w[r * k + j] - back[r * k + j]);
                    energy += w[r * k + j] * (double)w[r * k + j];
                }
            }

            Check(error < (bytes == 1 ? 1e-4 : 1e-8) * energy, $"{row}: relative error {error / energy}");

            // int8 rows by the described lane's plan of the row; int16 rows by its plan of the codes alone, each row's scale
            // after (the scale covers the row, so it factors out of the row's sum).
            Check(CpuBackend.DescribedRoute(row.Description, k) is not null, $"{row}: the CPU's described lane reads it");
            var xs = Enumerable.Range(0, m * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            using var weight = Tensor.Stored(again, stored, [n, k], device);                 // another object of the same description
            using var x = Tensor.From(xs, [m, k], device);
            using var y = Tensor.Zeros([m, n], device);
            device.Backend.StoredMatMul(x.Storage, weight.Storage, y.Storage, m, n, k);
            var product = y.ToArray();
            double worst = 0;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double reference = 0, magnitude = 0;
                    for (int p = 0; p < k; p++)
                    {
                        reference += (double)xs[i * k + p] * back[j * k + p];
                        magnitude += Math.Abs((double)xs[i * k + p] * back[j * k + p]);
                    }

                    worst = Math.Max(worst, Math.Abs(product[i * n + j] - reference) / magnitude);
                }
            }

            // The CPU's integer lane takes x's chunks as int8 too: its products within their rounding.
            if (!(worst < 5e-3))
            {
                wrong.Add($"{row}: worst error {worst} of the magnitudes");
            }

            // The plans are each description object's own: made the first time, then held (the same object again); an equal
            // description made apart, or a copy, makes its own; none kept by the value.
            var (mine, theirs, copy) = (row.Description, again.Description, row.Description with { });
            Check(ReferenceEquals(mine.Reader, mine.Reader) && !ReferenceEquals(mine.Reader, theirs.Reader) && !ReferenceEquals(mine.Reader, copy.Reader)
                  && mine.Encoder is not null && ReferenceEquals(mine.Encoder, mine.Encoder) && !ReferenceEquals(mine.Encoder, theirs.Encoder) && !ReferenceEquals(mine.Encoder, copy.Encoder)
                  && (mine.CpuPlan is null ? theirs.CpuPlan is null : ReferenceEquals(mine.CpuPlan, mine.CpuPlan) && theirs.CpuPlan is not null && !ReferenceEquals(mine.CpuPlan, theirs.CpuPlan))
                  && ReferenceEquals(mine.CpuFingerprint, mine.CpuFingerprint) && mine.CpuFingerprint == theirs.CpuFingerprint && !ReferenceEquals(mine.CpuFingerprint, theirs.CpuFingerprint),
                $"{row}: each description object makes and holds its own reader, writer and lane plan; an equal one made apart, or a copy, its own");

            // The code's grid (no bias) is the number type's own: rows of every length hold their element type's code object,
            // so their encoders share one grid; a copy of the number type makes its own; neither takes part in its value.
            var (rowCode, copiedCode) = (mine.Code, mine.Code with { });
            Check(ReferenceEquals(rowCode, theirs.Code) && ReferenceEquals(rowCode, element.Description.Code) && rowCode.Grid is not null
                  && ReferenceEquals(DescribedEncoder.Grid.Of(rowCode, 0), DescribedEncoder.Grid.Of(element.ForRow(k + 32).Description.Code, 0)) && ReferenceEquals(rowCode.Grid, rowCode.Grid)
                  && !ReferenceEquals(rowCode.Grid, copiedCode.Grid) && rowCode == copiedCode && rowCode.GetHashCode() == copiedCode.GetHashCode(),
                $"{row}: its code's grid is held by the number type, one for its rows of every length; a copy of the type makes its own, equal by value");

            // The lanes measured for its shapes are its plan's own: the product above (by `again`'s description) kept its
            // lane in that description's plan (or its codes' plan, where the lane reads the codes alone), none in an equal
            // description's made apart.
            var (measured, unmeasured) = theirs.CpuPlan is { } plan ? (plan, mine.CpuPlan!) : (theirs.CpuRowCodes.CpuPlan!, mine.CpuRowCodes.CpuPlan!);
            Check(!ReferenceEquals(measured, unmeasured) && measured.Chosen.Count > 0 && unmeasured.Chosen.Count == 0,
                $"{row}: the product's lane is kept by its own description's plan ({measured.Chosen.Count} shapes), not by an equal one's ({unmeasured.Chosen.Count})");

            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"  {row}: {stored.Length} bytes for {n} rows, relative error {error / energy:0.0e+0}, the CPU's product within {worst:0.0e+0}"));
        }

        Check(wrong.Count == 0, $"the CPU's product of a row's scale and codes: {string.Join("; ", wrong)}");
    }

    private static void TypesNoNarrowing(Device device)
    {
        CpuOnly(device);
        var random = new Random(5);
        var doubles = new double[4096];
        for (int i = 0; i < doubles.Length; i++)
        {
            doubles[i] = BitConverter.Int64BitsToDouble(random.NextInt64());
            if (double.IsNaN(doubles[i]))
            {
                doubles[i] = Math.ScaleB(random.NextDouble(), -1070);              // a subnormal instead
            }
        }

        var stored = new byte[F64.Type.ByteCount(doubles.Length)];
        F64.Type.Write(doubles, stored);
        var back = new double[doubles.Length];
        F64.Type.Read(stored, back);
        Check(back.AsSpan().SequenceEqual(doubles), "float64 written from float64 and read back, bit for bit");

        var longs = Enumerable.Range(0, 4096).Select(_ => (double)random.NextInt64(-(1L << 53), 1L << 53)).ToArray();
        var integers = new byte[I64.Type.ByteCount(longs.Length)];
        I64.Type.Write(longs, integers);
        I64.Type.Read(integers, back);
        Check(back.AsSpan().SequenceEqual(longs) && BitConverter.ToInt64(integers, 8) == (long)longs[1], "int64 written and read back exactly, its bytes the integer");

        var ints = Enumerable.Range(0, 4096).Select(_ => (double)random.Next(int.MinValue, int.MaxValue)).Append(int.MaxValue).Append(int.MinValue).Append(1e12).ToArray();
        var words = new byte[I32.Type.ByteCount(ints.Length)];
        I32.Type.Write(ints, words);
        var read = new double[ints.Length];
        I32.Type.Read(words, read);
        Check(read.AsSpan(0, ints.Length - 1).SequenceEqual(ints.AsSpan(0, ints.Length - 1)) && read[^1] == int.MaxValue, "int32 exact over its range, past it its largest");

        var singles = new byte[F32.Type.ByteCount(doubles.Length)];
        F32.Type.Write(doubles, singles);
        var floats = new float[doubles.Length];
        F32.Type.Read(singles, floats);
        Check(Enumerable.Range(0, doubles.Length).All(i => BitConverter.SingleToInt32Bits(floats[i]) == BitConverter.SingleToInt32Bits((float)doubles[i])),
            "float32 written from float64 rounded once, as the machine rounds");
        Console.WriteLine("  float64, int64 (to 2^53), int32 and float32 written from float64 and read back as stored");
    }

    // A type the library does not know: fp5 E2M2 codes packed tight, 32 a block under a float16 scale.
    private static TestType OutsideType() => new("fp5 e2m2 blocks of 32 (outside the library)",
        new(32, 22, NumberType.Floating(2, 2, specials: FloatSpecials.None), [new CodePlane(2, 5, 1, Stream: true)], Scales: [new BlockScale(HalfNumber, 32, 0)]));

    private static void TypesOutside(Device device)
    {
        CpuOnly(device);
        IElementType type = OutsideType();
        var random = new Random(9);
        int m = 3, n = 5, k = 96;
        var w = Enumerable.Range(0, n * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var xs = Enumerable.Range(0, m * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        using var weight = Tensor.Stored(type, w, [n, k], device);
        Check(weight.Type == type && weight.Size == n * k, "the weight holds its type");
        var decoded = weight.ToArray();
        double error = 0, energy = 0;
        for (int i = 0; i < w.Length; i++)
        {
            error += (w[i] - decoded[i]) * (double)(w[i] - decoded[i]);
            energy += w[i] * (double)w[i];
        }

        Check(error < 0.01 * energy, $"written and read back: relative error {error / energy}");
        using var x = Tensor.From(xs, [m, k], device);
        using var y = Tensor.Zeros([m, n], device);
        device.Backend.StoredMatMul(x.Storage, weight.Storage, y.Storage, m, n, k);
        var product = y.ToArray();
        double worst = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double reference = 0, scale = 0;
                for (int p = 0; p < k; p++)
                {
                    reference += (double)xs[i * k + p] * decoded[j * k + p];
                    scale += Math.Abs((double)xs[i * k + p] * decoded[j * k + p]);
                }

                worst = Math.Max(worst, Math.Abs(product[i * n + j] - reference) / scale);
            }
        }

        Check(worst < 0.02, $"the CPU's product of the outside type: worst error {worst} of the products' magnitudes");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"  {type}: written (relative error {error / energy:0.0000}), read, multiplied by the CPU (worst {worst:0.00000} of the magnitudes)"));
    }

    private static void TypesTypedTensors(Device device)
    {
        var random = new Random(4);
        foreach (var type in new IElementType[] { BF16.Type, Idrak.Types.Q4_K.Type, F32.Type, I4.Type, OutsideType() })
        {
            int count = 512;
            var stored = new byte[type.ByteCount(count)];
            random.NextBytes(stored);
            if (type.Native == NativeKind.Float32)
            {
                var values = Enumerable.Range(0, count).Select(_ => (float)random.NextDouble()).ToArray();
                Buffer.BlockCopy(values, 0, stored, 0, stored.Length);
            }

            using var tensor = Tensor.Stored(type, stored, [8, 64], device);
            Check(tensor.Type == type && tensor.Size == count && tensor.Storage.Typed && tensor.Storage.ElementCount == count, $"{type}: the tensor's type and size");
            Check(tensor.ToBytes().AsSpan().SequenceEqual(stored), $"{type}: its bytes as stored");
            var expected = new float[count];
            type.Read(stored, expected);
            var values2 = tensor.ToArray();
            Check(Enumerable.Range(0, count).All(i => BitConverter.SingleToInt32Bits(values2[i]) == BitConverter.SingleToInt32Bits(expected[i]) || float.IsNaN(values2[i]) && float.IsNaN(expected[i])),
                $"{type}: read back through its type");
        }

        using var plain = Tensor.From([1f, 2f], [2], device);
        Check(plain.Type == F32.Type && !plain.Storage.Typed && plain.Storage.Description is null, "an operation's tensor is the device's own float32");
    }

    // A safetensors file of float32 tensors written again with each tensor stored as `type` (safetensors' dtype `dtype`).
    // sFloat32: the tensors (by name and shape) stored as float32 of the values 	ype gives them (its rounding, exactly).
    private static void RewriteSafetensors(string file, string dtype, IElementType type, Func<string, int, bool>? asFloat32 = null)
    {
        var bytes = File.ReadAllBytes(file);
        long length = BitConverter.ToInt64(bytes, 0);
        var header = JsonNode.Parse(bytes.AsSpan(8, (int)length))!.AsObject();
        var rewritten = new JsonObject();
        var data = new List<byte[]>();
        long offset = 0;
        foreach (var (name, node) in header)
        {
            if (name == "__metadata__")
            {
                rewritten[name] = node!.DeepClone();
                continue;
            }

            var offsets = node!["data_offsets"]!.AsArray();
            long begin = (long)offsets[0]!, end = (long)offsets[1]!;
            var values = new float[(end - begin) / 4];
            Buffer.BlockCopy(bytes, (int)(8 + length + begin), values, 0, (int)(end - begin));
            var stored = new byte[type.ByteCount(values.Length)];
            type.Write(values, stored);
            bool wide = asFloat32?.Invoke(name, node["shape"]!.AsArray().Count) == true;
            if (wide)
            {
                type.Read(stored, values);                                       // the type's values, kept as float32
                stored = new byte[4 * values.Length];
                Buffer.BlockCopy(values, 0, stored, 0, stored.Length);
            }

            data.Add(stored);
            rewritten[name] = new JsonObject { ["dtype"] = wide ? "F32" : dtype, ["shape"] = node["shape"]!.DeepClone(), ["data_offsets"] = new JsonArray(offset, offset + stored.Length) };
            offset += stored.Length;
        }

        var json = System.Text.Encoding.UTF8.GetBytes(rewritten.ToJsonString());
        using var output = File.Create(file);
        output.Write(BitConverter.GetBytes((long)json.Length));
        output.Write(json);
        foreach (var part in data)
        {
            output.Write(part);
        }
    }

    private static void TypesSafetensors(Device device)
    {
        CpuOnly(device);
        string folder = TempFolder();
        try
        {
            var random = new Random(8);
            string file = Path.Combine(folder, "model.safetensors");
            var values = Enumerable.Range(0, 64).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
            foreach (var (dtype, type) in new (string, IElementType)[] { ("BF16", BF16.Type), ("F16", F16.Type), ("F8_E4M3", F8E4M3.Type), ("I8", I8.Type), ("I64", I64.Type), ("BOOL", Bool.Type), ("F64", F64.Type) })
            {
                SafeTensorsWriter.Write(file, [("w", [8, 8], values)]);
                RewriteSafetensors(file, dtype, type);
                using var reader = SafeTensorsReader.Open(file);
                var expected = new byte[type.ByteCount(values.Length)];
                type.Write(values, expected);
                var decoded = new float[values.Length];
                type.Read(expected, decoded);
                Check(reader.TypeOf("w") == type && SafeTensorsReader.TypeOfDtype(dtype) == type, $"{dtype}: the dtype's type");
                Check(reader.ReadStored("w").AsSpan().SequenceEqual(expected), $"{dtype}: the bytes as stored");
                Check(reader.Read("w").AsSpan().SequenceEqual(decoded), $"{dtype}: the values read through the type");
                var rows = new float[16];
                Check(reader.ReadRows("w", 3, 2, rows) && rows.AsSpan().SequenceEqual(decoded.AsSpan(24, 16)), $"{dtype}: rows read through the type");
                Check(((Idrak.Abstraction.Formats.ITensorStore)reader).TypeOf("w") == type, $"{dtype}: the store presents the type");
            }

            Console.WriteLine("  BF16, F16, F8_E4M3, I8, I64, BOOL and F64 tensors read as their types: bytes as stored, values and rows through the type");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TypesLoadTyped(Device device)
    {
        CpuOnly(device);
        var spec = TinyChatSpec();
        string folder = WriteChatModel(spec);
        try
        {
            string file = Path.Combine(folder, "model.safetensors");
            RewriteSafetensors(file, "BF16", BF16.Type);
            using var stored = SafeTensorsReader.Open(file);

            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Family = QwenArchitectures.Qwen3() });
            Check(model.Tokenizer is not null && model.Tokenizer.Encode("hi").Count > 0, "the tokenizer read from tokenizer.json directly");
            var linears = Linears(model.Network).ToList();
            var projections = linears.Where(l => l.TiedTo is null).ToList();
            Check(projections.Count == spec.Layers * 7 && projections.All(l => l.Stored is { } w && w.Type == BF16.Type && w.Storage.ByteLength == 2L * w.Size),
                $"every projection kept as stored, bfloat16 (2 bytes a value): {string.Join(", ", projections.Select(l => l.ToString()))}");
            var q = projections[0];
            Check(q.Stored!.ToBytes().AsSpan().SequenceEqual(stored.ReadStored("model.layers.0.self_attn.q_proj.weight")), "the first projection's bytes are the file's");
            var head = linears.Single(l => l.TiedTo is not null);
            Check(head.TiedTo!.Weight.Type == BF16.Type, "the token table kept as stored; the tied head multiplies by it as stored");

            // The stored projection and the table run: x · Wᵀ against the weight read through its type, a row looked up.
            var random = new Random(2);
            var xs = Enumerable.Range(0, 2 * spec.Dim).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            using var x = Tensor.From(xs, [2, spec.Dim], device);
            var w = q.Stored.ToArray();
            using (var y = q.Forward(x))
            {
                var got = y.ToArray();
                int n = q.OutFeatures;
                double worst = 0;
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double reference = 0, scale = 0;
                        for (int p = 0; p < spec.Dim; p++)
                        {
                            reference += (double)xs[i * spec.Dim + p] * w[j * spec.Dim + p];
                            scale += Math.Abs((double)xs[i * spec.Dim + p] * w[j * spec.Dim + p]);
                        }

                        worst = Math.Max(worst, Math.Abs(got[i * n + j] - reference) / Math.Max(scale, 1e-30));
                    }
                }

                Check(worst < 1e-5, $"the stored bfloat16 projection's product: worst error {worst} of the magnitudes");
            }

            using (var ids = Tensor.Indices([5, 0, 259], device))
            using (var rows = head.TiedTo.Forward(ids))
            {
                var table = head.TiedTo.Weight.ToArray();
                var got = rows.ToArray();
                Check(rows.Shape.SequenceEqual([3, spec.Dim]) && got.AsSpan(0, spec.Dim).SequenceEqual(table.AsSpan(5 * spec.Dim, spec.Dim))
                      && got.AsSpan(2 * spec.Dim, spec.Dim).SequenceEqual(table.AsSpan(259 * spec.Dim, spec.Dim)), "rows looked up through the table's type");
            }

            // Written as any type: every projection (and the table) by that type's encoder.
            using var written = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Family = QwenArchitectures.Qwen3(), Weights = Idrak.Types.Q8_0.Type });
            var writtenLinears = Linears(written.Network).ToList();
            Check(writtenLinears.Where(l => l.TiedTo is null).All(l => l.Stored?.Type == Idrak.Types.Q8_0.Type) && writtenLinears.Single(l => l.TiedTo is not null).TiedTo!.Weight.Type == Idrak.Types.Q8_0.Type,
                "written as q8_0 by its encoder");

            // Written as int8: each weight's rows as int8 gives them (its codes and a float32 scale a row), and the model runs.
            // Its family the one of the built-in families that states the name config.json gives (Qwen3ForCausalLM).
            using var int8 = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Families = Idrak.Text.Plugins.BuiltInFamilies.All, Weights = I8.Type });
            Check(int8.Architecture.Names.SequenceEqual(["Qwen3ForCausalLM"]) && Idrak.Text.Plugins.BuiltInFamilies.All.Families.Contains(int8.Architecture), "the family found by the checkpoint's own name");
            var rowLinears = Linears(int8.Network).ToList();
            var rowTable = rowLinears.Single(l => l.TiedTo is not null).TiedTo!.Weight;
            Check(rowLinears.Where(l => l.TiedTo is null).All(l => l.Stored is { } s && s.Type.Description == I8.Type.ForRow(s.Shape[1]).Description && s.Storage.ElementCount == s.Size
                      && s.Type.ByteCount(s.Size) == (long)s.Shape[0] * (s.Shape[1] + 4))
                  && rowTable.Type.Description == I8.Type.ForRow(spec.Dim).Description,
                $"written as int8 rows, a float32 scale a row: {string.Join(", ", rowLinears.Select(l => l.Stored?.Type.Name ?? l.TiedTo?.Weight.Type.Name))}");
            var generated = new Idrak.Generation.TextGenerator(int8.Network, int8.Tokenizer!, 64).Generate("hi", new Idrak.Abstraction.Generation.GenerationOptions { TopK = 1, NumPredict = 4, Seed = 1 });
            Check(generated.Text is not null, "the int8 model generates");
            Console.WriteLine($"  {projections.Count} projections and the table loaded as bfloat16 as stored (2 bytes a value, the file's bytes), run, looked up; written as q8_0 and as int8 rows too, the int8 model generating");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // Norm gains and biases kept as their checkpoint stores them (never widened, never written as the weights' type): a
    // bfloat16 Qwen3 checkpoint (query and key norms) and a Qwen2 one (query, key and value biases), each against its twin
    // whose vectors are float32 of the same values: the same logits bit for bit and the same greedy text.
    private static void TypesStoredVectors(Device device)
    {
        CpuOnly(device);
        var families = Idrak.Text.Plugins.BuiltInFamilies.All;
        foreach (var (spec, architecture) in new[] { (TinyChatSpec(), "Qwen3ForCausalLM"), (TinyChatSpec() with { QkNorm = false, QkvBias = true }, "Qwen2ForCausalLM") })
        {
            string typed = WriteChatModel(spec), twin = TempFolder();
            try
            {
                string config = Path.Combine(typed, "config.json");
                var json = JsonNode.Parse(File.ReadAllText(config))!.AsObject();
                json["architectures"] = new JsonArray(architecture);
                File.WriteAllText(config, json.ToJsonString());
                foreach (string file in Directory.GetFiles(typed))
                {
                    File.Copy(file, Path.Combine(twin, Path.GetFileName(file)), overwrite: true);
                }

                RewriteSafetensors(Path.Combine(typed, "model.safetensors"), "BF16", BF16.Type);
                RewriteSafetensors(Path.Combine(twin, "model.safetensors"), "BF16", BF16.Type, asFloat32: (_, rank) => rank == 1);
                using var model = PretrainedModel.Load(typed, new PretrainedOptions { Device = device, Families = families });
                using var wide = PretrainedModel.Load(twin, new PretrainedOptions { Device = device, Families = families });
                using var written = PretrainedModel.Load(typed, new PretrainedOptions { Device = device, Families = families, Weights = Idrak.Types.Q8_0.Type });
                static List<Tensor> Vectors(Module network) => [.. network.Descendants().Distinct().SelectMany(m => m switch
                {
                    RMSNorm norm => [norm.Gain],
                    Linear { Bias: { } bias } => [bias],
                    _ => Array.Empty<Tensor>(),
                })];
                var vectors = Vectors(model.Network);
                int expected = 2 * spec.Layers + 1 + (spec.QkNorm ? 2 * spec.Layers : 0) + (spec.QkvBias ? 3 * spec.Layers : 0);
                Check(vectors.Count == expected && vectors.All(v => v.Type == BF16.Type && v.Storage.Typed),
                    $"{architecture}: {vectors.Count} of {expected} norm gains and biases kept as stored, bfloat16: {string.Join(", ", vectors.Select(v => v.Type.Name).Distinct())}");
                Check(Vectors(written.Network) is { Count: > 0 } kept && kept.All(v => v.Type == BF16.Type), $"{architecture}: kept bfloat16 when the weights are written as q8_0");
                Check(Vectors(wide.Network).All(v => !v.Storage.Typed), $"{architecture}: the twin's float32 vectors read as float32");
                using (var reader = SafeTensorsReader.Open(Path.Combine(typed, "model.safetensors")))
                {
                    var final = model.Network.Descendants().OfType<RMSNorm>().Single(n => n.Name == "norm");
                    Check(final.Gain.ToBytes().AsSpan().SequenceEqual(reader.ReadStored("model.norm.weight")), $"{architecture}: the final norm's bytes are the file's");
                }

                int[] ids = [1, 5, 9, 200, 7, 33];
                using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
                var logits = model.Network.Predict(input).ToArray();
                var reference = wide.Network.Predict(input).ToArray();
                Check(logits.Select(BitConverter.SingleToInt32Bits).SequenceEqual(reference.Select(BitConverter.SingleToInt32Bits)),
                    $"{architecture}: the logits are the float32 twin's, bit for bit");
                var options = new Idrak.Abstraction.Generation.GenerationOptions { TopK = 1, NumPredict = 12, Seed = 1 };
                string text = new Idrak.Generation.TextGenerator(model.Network, model.Tokenizer!, 64).Generate("hi", options).Text;
                string twinText = new Idrak.Generation.TextGenerator(wide.Network, wide.Tokenizer!, 64).Generate("hi", options).Text;
                Check(text == twinText, $"{architecture}: the same greedy text as the twin's ('{text}', '{twinText}')");
                Console.WriteLine($"  {architecture}: {vectors.Count} norm gains and biases kept bfloat16 as stored (also with the weights written as q8_0); the logits and greedy text the float32 twin's");
            }
            finally
            {
                Directory.Delete(typed, recursive: true);
                Directory.Delete(twin, recursive: true);
            }
        }
    }
    // Every linear layer of a network, depth first.
    private static IEnumerable<Linear> Linears(Module module) =>
        module is Linear linear ? [linear] : module.Children().SelectMany(Linears);
}
