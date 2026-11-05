// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Operations;

// No type fixed (the owner, 2026-11-05): the contracts every package builds on. The unsigned integers registered with
// exact integer conversions, the id type chosen from the data, the result types of the type promotions, an operation's
// sum type chosen per device, checked typed reads, and tensors made in a chosen type.
internal abstract partial class DesktopTests : Tests
{
    private static void UnsignedTypes(Device device)
    {
        var expected = new[] { ("uint32", 32, 0L, (long)uint.MaxValue), ("uint64", 64, 0L, long.MaxValue), ("uint2", 2, 0L, 3L) };
        foreach (var (name, bits, low, high) in expected)
        {
            var type = ElementTypes.Get(name);
            Check(type.IsInteger && !type.IsSigned && !type.IsFloat && type.BlockBits == bits && type.MinValue == low && type.MaxValue >= high && ElementTypes.Origin(name) == Overrides.Library,
                $"{name}: an unsigned integer of {bits} bits, the library's: [{type.MinValue}, {type.MaxValue}]");
        }

        Check(ElementType.Int16.Holds(-32768, 32767) && !ElementType.Int16.Holds(0, 32768) && ElementType.UInt16.Holds(0, 65535) && !ElementType.UInt16.Holds(-1, 0)
              && !ElementType.Float32.Holds(0, 1), "Holds: an integer type's range, never a float's");

        // Values round trip through the types' own conversions (saturated, rounded to nearest even), on the host and a tensor.
        float[] values = [0f, 1f, 2.5f, 3.5f, -1f, 4f, 70000f, 4294967296f, float.NaN];
        foreach (var (type, reference) in new (ElementType, Func<float, float>)[]
                 {
                     (ElementType.UInt32, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), 0f, 4294967295f)),
                     (ElementType.UInt64, x => float.IsNaN(x) ? 0f : Math.Max(MathF.Round(x), 0f)),
                     (ElementType.UInt2, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), 0f, 3f)),
                 })
        {
            var bytes = new byte[type.ByteCount(values.Length)];
            type.Encode(values, bytes);
            var back = new float[values.Length];
            type.Decode(bytes, back);
            for (int i = 0; i < values.Length; i++)
            {
                Check(back[i] == reference(values[i]), $"{type}: {values[i]} came back {back[i]}, the reference gives {reference(values[i])}");
            }

            using var t = Tensor.FromBytes(bytes, type, [values.Length], device);
            Check(t.ElementType == type && t.ToBytes().AsSpan().SequenceEqual(bytes) && t.ToArray().SequenceEqual(back), $"{type} on {device}: a tensor's bytes and values as stored");
        }

        var nibbles = new byte[1];
        ElementType.UInt2.Encode([1f, 2f, 3f, 0f], nibbles);
        Check(nibbles[0] == 0b00_11_10_01, $"uint2: four a byte, the first value in the lowest bits: {nibbles[0]:B8}");

        // Between integer types: exact, or refused past the destination's range.
        var wide = MemoryMarshal.AsBytes<long>([0, 255, 65535, 4294967295, -1]);
        var u32 = new byte[4 * 4];
        ElementConversions.Convert(ElementType.Int64, wide, ElementType.UInt32, u32, 4);
        Check(MemoryMarshal.Cast<byte, uint>(u32).SequenceEqual([0u, 255u, 65535u, 4294967295u]), "int64 to uint32: exact past 2^24");
        var u16 = new byte[2 * 3];
        ElementConversions.Convert(ElementType.UInt32, u32, ElementType.UInt16, u16, 3);
        Check(MemoryMarshal.Cast<byte, ushort>(u16).SequenceEqual(new ushort[] { 0, 255, 65535 }), "uint32 to uint16 exact");
        ThrowsNaming<ArgumentException>(() => ElementConversions.Convert(ElementType.UInt32, u32, ElementType.UInt16, new byte[8], 4), "uint32 4294967295 to uint16",
            "4294967295", "uint16");
        ThrowsNaming<ArgumentException>(() => ElementConversions.Convert(ElementType.Int64, MemoryMarshal.AsBytes<long>([7, -1]), ElementType.UInt8, new byte[2], 2), "int64 -1 to uint8", "-1", "uint8");
        var big = MemoryMarshal.AsBytes<ulong>([ulong.MaxValue]);
        var back64 = new byte[8];
        ElementConversions.Convert(ElementType.UInt64, big, ElementType.UInt64, back64, 1);
        Check(back64.AsSpan().SequenceEqual(big), "uint64 to itself: as stored");
    }

    private static void IdTypeChoice(Device device)
    {
        string? saved = Environment.GetEnvironmentVariable(ElementTypes.IdEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(ElementTypes.IdEnvironmentVariable, null);
            Check(ElementTypes.IdTypes.Take(4).Select(t => t.Name).SequenceEqual(["uint8", "int8", "uint16", "int16"]) && ElementTypes.IdTypes.All(t => t.IsInteger && t.BlockBits % 8 == 0),
                $"the id types: integers of whole bytes, fewest bits first, unsigned first: {string.Join(", ", ElementTypes.IdTypes.Select(t => t.Name))}");
            foreach (var (largest, smallest, name) in new[] { (7L, 0L, "uint8"), (255L, 0L, "uint8"), (256L, 0L, "uint16"), (151935L, 0L, "uint32"), (100L, -100L, "int8"), (5000000000L, 0L, "uint64"), (1L, -5000000000L, "int64") })
            {
                var choice = ElementTypes.ChooseIds(largest, smallest, device: device);
                Check(choice.Type.Name == name && choice.Picked && choice.Role == "id type" && choice.How.StartsWith("picked by auto: ids from", StringComparison.Ordinal)
                      && choice.ToString().StartsWith($"id type: {name} on {device}", StringComparison.Ordinal),
                    $"ids {smallest}..{largest}: {choice} (expected {name}, the smallest that holds them)");
            }

            var given = ElementTypes.ChooseIds(1000, asked: "int32", device: device);
            Check(given.Type == ElementType.Int32 && !given.Picked && given.How == "selected by the user (--id-dtype)", $"selected: {given}");
            ThrowsNaming<ArgumentException>(() => ElementTypes.ChooseIds(1000, asked: "uint8", device: device), "uint8 for ids up to 1,000", "uint8", "1,000");
            ThrowsNaming<ArgumentException>(() => ElementTypes.ChooseIds(10, asked: "float32", device: device), "a float for ids", "float32", "uint16");
            Environment.SetEnvironmentVariable(ElementTypes.IdEnvironmentVariable, "uint32");
            var variable = ElementTypes.ChooseIds(10, device: device);
            Check(variable.Type == ElementType.UInt32 && variable.Source == ElementTypes.IdEnvironmentVariable, $"the variable selects: {variable}");
            Environment.SetEnvironmentVariable(ElementTypes.IdEnvironmentVariable, null);

            // A tensor of ids in the chosen type, read back exactly and read by an operation (through a counted conversion where the kernel takes another type).
            using var ids = Tensor.Ids([3, 0, 2, 1], [4], device: device);
            Check(ids.ElementType == ElementType.UInt8 && ids.Span<byte>().SequenceEqual(new byte[] { 3, 0, 2, 1 }) && ids.ToInt32Array().SequenceEqual([3, 0, 2, 1]),
                $"Tensor.Ids: uint8 for ids under 256: {ids.ElementType}");
            using var far = Tensor.Ids([0L, 70000L], [2], device: device);
            Check(far.ElementType == ElementType.UInt32 && far.ToInt64Array().SequenceEqual([0L, 70000L]), $"Tensor.Ids: uint32 past 65,535: {far.ElementType}");
            using var table = Tensor.From([10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f], [4, 2], device);
            long before = ElementTypes.Conversions;
            var rows = table.EmbeddingLookup(ids).ToArray();
            Check(rows.SequenceEqual([22f, 23f, 10f, 11f, 20f, 21f, 12f, 13f]), $"an embedding lookup by uint8 ids gives the rows the ids name: {string.Join(", ", rows)}");
            Check(ElementTypes.Conversions - before <= 1, $"uint8 ids read through at most one counted conversion: {ElementTypes.Conversions - before}");
        }
        finally
        {
            Environment.SetEnvironmentVariable(ElementTypes.IdEnvironmentVariable, saved);
        }
    }

    private static void TypePromotionRules(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(TypePromotionRules)))
        {
            return;
        }

        var cases = new (ElementType Left, ElementType Right, ElementType Result)[]
        {
            (ElementType.Float32, ElementType.Float32, ElementType.Float32),
            (ElementType.BFloat16, ElementType.Float32, ElementType.Float32),
            (ElementType.BFloat16, ElementType.Float16, ElementType.Float32),                 // neither holds the other: the narrowest that holds both
            (ElementType.Float8E4M3, ElementType.Float8E5M2, ElementType.BFloat16),          // the first registered 16-bit float holding every code of both exactly
            (ElementType.Float8E4M3, ElementType.BFloat16, ElementType.BFloat16),
            (ElementType.Int32, ElementType.BFloat16, ElementType.BFloat16),                 // a float over an integer
            (ElementType.UInt8, ElementType.Int8, ElementType.Int16),                        // the narrowest integer holding both ranges
            (ElementType.UInt16, ElementType.Int32, ElementType.Int32),
            (ElementType.UInt8, ElementType.UInt16, ElementType.UInt16),
            (ElementType.Bool, ElementType.UInt8, ElementType.UInt8),
            (ElementTypes.Get("q4_k"), ElementType.BFloat16, ElementType.BFloat16),          // a block format takes the other input's type
            (ElementType.Int4, ElementType.Float16, ElementType.Float16),
        };
        foreach (var (left, right, result) in cases)
        {
            var a = TypePromotions.Result(left, right);
            var b = TypePromotions.Result(right, left);
            Check(a == result && b == result, $"{left} with {right}: {a} and {b}, expected {result}");
        }

        Check(TypePromotions.Result([ElementType.UInt8, ElementType.Int8, ElementType.Float16]) == ElementType.Float16, "three inputs, pairwise");
        ThrowsNaming<NotSupportedException>(() => TypePromotions.Result(ElementTypes.Get("q4_k"), ElementTypes.Get("q8_0")), "two block formats", "q4_k", "q8_0", "TypePromotions.Register");
        ThrowsNaming<NotSupportedException>(() => TypePromotions.Result(ElementType.UInt64, ElementType.Int64), "uint64 with int64", "uint64", "int64");

        // An app's rule is asked first and falls through to the library's; unregistered, the library's answers alone.
        var rule = new OwnPromotion();
        TypePromotions.Register(rule);
        try
        {
            Check(TypePromotions.Names[0] == "own-promotion" && TypePromotions.Result(ElementType.UInt64, ElementType.Int64) == ElementType.Float64
                  && TypePromotions.Result(ElementType.UInt8, ElementType.Int8) == ElementType.Int16 && TypePromotions.Origin(TypePromotions.Standard) == Overrides.Library,
                $"the app's rule first, the library's after it: {string.Join(", ", TypePromotions.Names)}");
        }
        finally
        {
            TypePromotions.Unregister("own-promotion");
        }

        Check(!TypePromotions.Names.Contains("own-promotion"), "the app's rule removed");
    }

    // uint64 with int64 as float64 (numpy's answer); anything else for the next rule.
    private sealed class OwnPromotion : ITypePromotion
    {
        public string Name => "own-promotion";

        public string Summary => "uint64 with int64 as float64";

        public ElementType? Promote(Operation? operation, ElementType left, ElementType right) =>
            (left == ElementType.UInt64 && right == ElementType.Int64) || (left == ElementType.Int64 && right == ElementType.UInt64) ? ElementType.Float64 : null;
    }

    private static void SumTypeChoice(Device device)
    {
        string? saved = Environment.GetEnvironmentVariable(SumTypes.EnvironmentVariable), savedTolerance = Environment.GetEnvironmentVariable(SumTypes.ToleranceVariable);
        try
        {
            Environment.SetEnvironmentVariable(SumTypes.EnvironmentVariable, null);
            Environment.SetEnvironmentVariable(SumTypes.ToleranceVariable, null);
            Check(SumTypes.Resolution(ElementType.Float32) == Math.Pow(2, -23) && SumTypes.Resolution(ElementType.BFloat16) == Math.Pow(2, -7)
                  && SumTypes.Resolution(ElementType.Float16) == Math.Pow(2, -10) && SumTypes.Resolution(ElementType.Float8E4M3) == Math.Pow(2, -3) && SumTypes.Resolution(ElementType.Int32) == 1,
                "each float type's resolution, found by storing 1 + 2^-k");

            var cpu = Device.Cpu.Backend.Capabilities;
            Check(cpu.SumTypesOf(Ops.BatchedMatMul).SequenceEqual([ElementType.Float32]) && cpu.ValueTypesOf(Ops.BatchedMatMul).SequenceEqual([ElementType.Float32]),
                "the CPU reports the value and sum types its kernels take");

            var measured = new[] { new SumTypeMeasurement(ElementType.Float32, 2e-3, 0), new SumTypeMeasurement(ElementType.BFloat16, 1e-3, 4e-3), new SumTypeMeasurement(ElementType.Float16, 1.5e-3, 2e-4) };
            Check(SumTypes.Pick(measured, 1e-3).Type == ElementType.Float16 && SumTypes.Pick(measured, 1e-2).Type == ElementType.BFloat16 && SumTypes.Pick(measured, 0).Type == ElementType.Float32
                  && SumTypes.Pick(measured, -1).Type == ElementType.Float32, "the fastest within the tolerance; none within it: the least error");

            var only = SumTypes.Choose(Ops.BatchedMatMul, Device.Cpu);
            Check(only.Type == ElementType.Float32 && only.Picked && only.Role == "sum type for BatchedMatMul" && only.How == "picked by auto: the only sum type cpu reports for BatchedMatMul",
                $"auto with one reported type: {only}");
            SumTypes.Record(Device.Cpu, Ops.BatchedMatMul, [new SumTypeMeasurement(ElementType.Float32, 1e-3, 0), new SumTypeMeasurement(ElementType.BFloat16, 1e-4, 1e-3)]);
            var picked = SumTypes.Choose(Ops.BatchedMatMul, Device.Cpu);
            Check(picked.Type == ElementType.Float32 && picked.How.Contains("the fastest within", StringComparison.Ordinal), $"measured: a type the device does not report is not picked: {picked}");
            SumTypes.Record(Device.Cpu, Ops.BatchedMatMul, []);
            var selected = SumTypes.Choose(Ops.BatchedMatMul, Device.Cpu, asked: "fp32");
            Check(selected.Type == ElementType.Float32 && !selected.Picked && selected.How == "selected by the user (--sum-dtype)", $"selected: {selected}");
            ThrowsNaming<ArgumentException>(() => SumTypes.Choose(Ops.BatchedMatMul, Device.Cpu, asked: "bf16"), "a sum type the CPU does not report", "bf16", "float32");
            Environment.SetEnvironmentVariable(SumTypes.ToleranceVariable, "1e-2");
            Check(SumTypes.ToleranceOn(Device.Cpu) == 1e-2, "the tolerance from its variable");
            Environment.SetEnvironmentVariable(SumTypes.ToleranceVariable, "much");
            ThrowsNaming<ArgumentException>(() => SumTypes.ToleranceOn(Device.Cpu), "a tolerance that is no number", SumTypes.ToleranceVariable, "much");
            Environment.SetEnvironmentVariable(SumTypes.ToleranceVariable, null);

            // The telemetry says the role.
            var hook = new ChosenTypes();
            using (Telemetry.Subscribe(hook))
            {
                SumTypes.Report(only, Device.Cpu);
            }

            Check(hook.Seen.Any(e => e.Role == "sum type for BatchedMatMul" && e.ElementType == "float32" && e.How == only.How), "the sum type's choice published with its role");
            _ = device;
        }
        finally
        {
            Environment.SetEnvironmentVariable(SumTypes.EnvironmentVariable, saved);
            Environment.SetEnvironmentVariable(SumTypes.ToleranceVariable, savedTolerance);
        }
    }

    private sealed class ChosenTypes : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public List<ElementTypeChosen> Seen { get; } = [];

        public void OnElementTypeChosen(in ElementTypeChosen e)
        {
            lock (Seen)
            {
                Seen.Add(e);
            }
        }
    }

    private static void TypedReads(Device device)
    {
        using var floats = Tensor.From([1f, 2f, 3f], [3], device);
        Check(floats.Span<float>().SequenceEqual([1f, 2f, 3f]), "float32 read as float");
        ThrowsNaming<InvalidOperationException>(() => floats.Span<int>(), "float32 read as int", "float32", "Int32", "int32");
        using var halves = Tensor.FromValues([0.5f, -2f, 65504f], [3], ElementType.Float16, device);
        Check(halves.ElementType == ElementType.Float16 && halves.Span<Half>().ToArray().SequenceEqual([(Half)0.5f, (Half)(-2f), (Half)65504f]), "float16 read as Half");
        using var brain = Tensor.FromValues([1.5f, 3f], [2], ElementType.BFloat16, device);
        ThrowsNaming<InvalidOperationException>(() => brain.Span<ushort>(), "bfloat16 read as ushort", "bfloat16", "uint16", "ConvertTo");
        Check(brain.ToArray().SequenceEqual([1.5f, 3f]), "FromValues: bfloat16 values as stored");
        using var shorts = Tensor.FromValues([-3f, 7f], [2], ElementType.Int16, device);
        Check(shorts.Span<short>().SequenceEqual(new short[] { -3, 7 }), "int16 read as short");
        using var nibbles = Tensor.FromBytes([0x12], ElementType.Int4, [2], device);
        ThrowsNaming<InvalidOperationException>(() => nibbles.Span<byte>(), "int4 read as byte", "int4", "ToBytes");

        // FromValues without a type: the run's choice on the device.
        string? saved = Environment.GetEnvironmentVariable(ElementTypes.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(ElementTypes.EnvironmentVariable, "bf16");
            using var chosen = Tensor.FromValues([1f, 2f], [2], device: device);
            Check(chosen.ElementType == ElementType.BFloat16, $"FromValues: the run's type when none is given: {chosen.ElementType}");
            using var kept = Tensor.FromValues([1f], [1], ElementType.Float32, device, requiresGrad: true, track: false);
            Check(kept.ElementType == ElementType.Float32 && kept.RequiresGrad, "FromValues: a given type wins");
        }
        finally
        {
            Environment.SetEnvironmentVariable(ElementTypes.EnvironmentVariable, saved);
        }
    }
}
