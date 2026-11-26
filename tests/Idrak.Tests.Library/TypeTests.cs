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
        ("types: the built-in list holds each type once, each valid and written, its native kind from its description", TypesBuiltInList),
        ("types: every built-in and alien type read through the default matches its description's own decode, bit for bit", TypesReadAsDescribed),
        ("types: the arithmetic rounding of wide codes is the table's rounding on every narrow float and integer", TypesWideRounding),
        ("types: the encoder writes every alien type: finite, every code reachable, its error against the least-squares bound", TypesEncoderWrites),
        ("types: 32- and 64-bit machine formats written from float64 and read back without narrowing", TypesNoNarrowing),
        ("types: a type defined outside the library is read, written and multiplied through the interface alone", TypesOutside),
        ("types: a typed tensor holds its type's bytes and reads back through its type", TypesTypedTensors),
        ("types: safetensors tensors read as their dtypes' types, their bytes as stored", TypesSafetensors),
        ("types: a Qwen3 checkpoint in bfloat16 loads its projections and table as stored, or written as any type", TypesLoadTyped),
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

    // Alien descriptions: per-element codes of 3, 5, 6, 7 and 24 bits packed tight, floats of 3, 5, 6 and 24 bits, 3- and
    // 5-bit block codes with E4M3 or 6-bit integer scales, codes over 3 planes, digits of radix 5 and 7, a 12-bit codebook,
    // three levels of scales (one shared over a group of blocks), a minimum of its own, E2M1 codes under E4M3 scales.
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
        ];
    }

    private static void TypesBuiltInList(Device device)
    {
        CpuOnly(device);
        var types = BuiltInTypes.All.Types;
        Check(types.Count == types.Distinct().Count() && types.Select(t => t.Name).Distinct(StringComparer.OrdinalIgnoreCase).Count() == types.Count,
            "each type once, each its own name");
        Check(types.All(t => t.Description.Valid), $"invalid: {string.Join(", ", types.Where(t => !t.Description.Valid))}");
        var unwritten = types.Where(t => t.Unwritable is not null).Select(t => $"{t}: {t.Unwritable}").ToList();
        Check(unwritten.Count == 0, $"unwritten: {string.Join("; ", unwritten)}");
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
        Console.WriteLine($"  {types.Count} built-in types, {types.Count - none.Count} native, every one written");
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
            int units = d.UnitValues >= 256 ? 24 : Math.Max(64, 6144 / d.UnitValues), n = units * d.UnitValues;
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
    private static void RewriteSafetensors(string file, string dtype, IElementType type)
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
            data.Add(stored);
            rewritten[name] = new JsonObject { ["dtype"] = dtype, ["shape"] = node["shape"]!.DeepClone(), ["data_offsets"] = new JsonArray(offset, offset + stored.Length) };
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
            Console.WriteLine($"  {projections.Count} projections and the table loaded as bfloat16 as stored (2 bytes a value, the file's bytes), run, looked up; written as q8_0 too");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // Every linear layer of a network, depth first.
    private static IEnumerable<Linear> Linears(Module module) =>
        module is Linear linear ? [linear] : module.Children().SelectMany(Linears);
}
