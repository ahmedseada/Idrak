// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Operations;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// Plan 37, T0 and T1: element and block types as registered data, their conversions, the dispatcher's counted conversion
// for an operation whose kernel takes another index type, the run's element type choice, and integer indices exact past
// 2^24 (embeddings, one-hot, scatter, top-k).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ElementTypesGroup =
    [
        ("element types: the registry holds float32, bfloat16, float16, float8_e4m3, float8_e5m2, int64, int32, int8, uint8, int4, int2, bool, int16, uint16, float64, uint32, uint64 and uint2 (any case, and the short names bf16, f16, fp32, double, fp8), ggml's block formats from Idrak, an app's type over them and back; an unknown name is \"not registered\"", ElementTypeRegistry),
        ("element types: every built-in type's values round trip against a reference (rounding to nearest even, saturation, fp8 codes both ways, packed int4 and int2, int16, uint16 and float64), on the host and through a typed tensor's bytes", ElementTypeRoundTrips),
        ("element types: conversions between float32, int32 and int64 are exact or refused (a fraction, an index past 2^24 for float32, past int32's range); the dispatcher converts an index storage of another type for its operation once, counted and published to telemetry, with the same result as the integer path", ElementTypeConversions),
        ("element types: a block format describes its bits a value, block, scales and zero points (q4_k: 4.5 bits, blocks of 256, 8 scales with minimums), takes whole blocks only, and decodes as its GGUF type does", ElementTypeBlockFormats),
        ("element types: the run's type is auto (picked from what the device reports natively: bfloat16, else float16, else float32; the CPU float32) or one the user selects (--dtype or IDRAK_DEFAULT_DTYPE, short names too, said so, and said when the device computes it in float32); another value is refused naming the accepted ones; CUDA's native types follow the PTX targets its kernels declare, as the device's reported capability reaches them; the CPU's indices are int32, a device that says nothing float32", ElementTypeChoosing),
        ("element types: indices past 2^24 are exact: an embedding lookup, its gradient (scatter-add), a one-hot and a top-k at row 16,777,217; int64 indices through the counted conversion", IndicesPast2To24),
        ("element types: decoding with learned positions (a decoder's position table, and a position encoding's) takes no counted conversion on the device: the prompt's ids, the step's positions and the sampled ids in its index type; the tokens as without the cache, and the cached lookup the float positions' converted one bit for bit", LearnedPositionDecoding),
        ("element types: uint32, uint64 and uint2 are registered unsigned integers (their ranges, round trips saturated and rounded to nearest even, uint2 four a byte); conversions between integer types are exact or refused past the destination's range", UnsignedTypes),
        ("element types: ids are stored in the smallest integer type that holds the data (uint8, int8, uint16, uint32, uint64, int64 by range), or the user's (--id-dtype, IDRAK_ID_DTYPE; one that does not hold the data is refused naming both), said with the role \"id type\"; Tensor.Ids stores them so and an embedding lookup reads uint8 ids", IdTypeChoice),
        ("element types: TypePromotions gives a result type from the inputs' types as data (a float over an integer, more bits over fewer, the narrowest type holding both exactly: bfloat16 with float16 gives float32, e4m3 with e5m2 bfloat16, uint8 with int8 int16; a block format takes the other's type); an app's rule is asked first; an undecided pair is refused", TypePromotionRules),
        ("element types: an operation's sum type is chosen per device from the types its backend reports (the CPU: float32): auto the fastest within the tolerance (IDRAK_SUM_TOLERANCE, else the values' resolution) from measurements, the only one when one is reported, or the user's (--sum-dtype, IDRAK_SUM_DTYPE); the telemetry carries its role", SumTypeChoice),
        ("element types: Span<T> reads a tensor as its own type's C# type and refuses another naming both; FromValues stores host values in a given type or the run's choice", TypedReads),
    ];

    private static void ElementTypeRegistry(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ElementTypeRegistry)))
        {
            return;
        }

        string[] builtIns = ["float32", "bfloat16", "float16", "float8_e4m3", "float8_e5m2", "int64", "int32", "int8", "uint8", "int4", "int2", "bool", "int16", "uint16", "float64", "uint32", "uint64", "uint2"];
        Check(ElementTypes.Names.Take(builtIns.Length).SequenceEqual(builtIns), $"the built-ins in order: {string.Join(", ", ElementTypes.Names)}");
        Check(ReferenceEquals(ElementTypes.Get("INT32"), ElementType.Int32) && ReferenceEquals(ElementTypes.Find("BFloat16"), ElementType.BFloat16)
              && ElementTypes.Origin("int32") == Overrides.Library, "found by name, ignoring case; the library's");
        Check(ElementTypes.Find("bf16") == ElementType.BFloat16 && ElementTypes.Get("F16") == ElementType.Float16 && ElementTypes.Find("fp32") == ElementType.Float32
              && ElementTypes.Find(" double ") == ElementType.Float64 && ElementTypes.Find("fp8") == ElementType.Float8E4M3 && ElementTypes.Find("half") == ElementType.Float16
              && ElementTypes.Find("bf17") is null, "the short names find their types");
        Check(ElementTypes.Find("q4_k") is { BlockSize: 256 } && ElementTypes.Find("iq2_xxs") is { IsBlock: true } && ElementTypes.Find("mxfp4") is { IsFloat: true },
            $"ggml's block formats registered by Idrak: {string.Join(", ", ElementTypes.Names.Skip(builtIns.Length))}");
        try
        {
            ElementTypes.Get("float7");
            Check(false, "an unknown type was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Message.Contains("not registered", StringComparison.Ordinal) && e.Message.Contains("int32", StringComparison.Ordinal) && e.Name == "float7",
                $"the refusal: {e.Message}");
        }

        var mine = new ElementType { Name = "test-int3", Summary = "a test's 3-bit integer", BlockBits = 3, IsSigned = true };
        ElementTypes.Register(mine);
        try
        {
            Check(ReferenceEquals(ElementTypes.Get("test-int3"), mine) && ElementTypes.Origin("test-int3") != Overrides.Library && mine.ByteCount(8) == 3
                  && mine.StorageWords(11) == 2 && Math.Abs(mine.BitsPerValue - 3) < 1e-12 && !mine.IsBlock, "an app's type, its sizes");
        }
        finally
        {
            ElementTypes.Unregister("test-int3");
        }

        Check(ElementTypes.Find("test-int3") is null && ElementTypes.Describe().Contains("int32: 32-bit signed integer", StringComparison.Ordinal), "unregistered; described");
    }

    private static void ElementTypeRoundTrips(Device device)
    {
        float[] values = [0f, -0f, 1f, -2.5f, 3.5f, 0.1f, 300f, -300f, 1e-3f, 65504f, 70000f, float.NaN];
        (ElementType Type, Func<float, float> Reference)[] cases =
        [
            (ElementType.Float32, x => x),
            (ElementType.BFloat16, x => float.IsNaN(x) ? float.NaN : BitConverter.UInt32BitsToSingle((BitConverter.SingleToUInt32Bits(x) + 0x7FFFu + ((BitConverter.SingleToUInt32Bits(x) >> 16) & 1u)) & 0xFFFF0000u)),
            (ElementType.Float16, x => (float)(Half)x),
            (ElementType.Int64, x => float.IsNaN(x) ? 0f : MathF.Round(x)),
            (ElementType.Int32, x => float.IsNaN(x) ? 0f : MathF.Round(x)),
            (ElementType.Int8, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), -128f, 127f)),
            (ElementType.UInt8, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), 0f, 255f)),
            (ElementType.Int4, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), -8f, 7f)),
            (ElementType.Int2, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), -2f, 1f)),
            (ElementType.Bool, x => x != 0f && !float.IsNaN(x) ? 1f : 0f),
            (ElementType.Int16, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), -32768f, 32767f)),
            (ElementType.UInt16, x => float.IsNaN(x) ? 0f : Math.Clamp(MathF.Round(x), 0f, 65535f)),
            (ElementType.Float64, x => x),
        ];
        foreach (var (type, reference) in cases)
        {
            var bytes = new byte[type.ByteCount(values.Length)];
            type.Encode(values, bytes);
            var back = new float[values.Length];
            type.Decode(bytes, back);
            for (int i = 0; i < values.Length; i++)
            {
                float expected = reference(values[i]);
                Check(float.IsNaN(expected) ? float.IsNaN(back[i]) : BitConverter.SingleToInt32Bits(back[i]) == BitConverter.SingleToInt32Bits(expected) || back[i] == expected && expected == 0f,
                    $"{type}: {values[i]} came back {back[i]}, the reference gives {expected}");
            }

            using var t = Tensor.FromBytes(bytes, type, [values.Length], device);
            var stored = t.ToBytes();
            var read = t.ToArray();
            Check(t.ElementType == type && t.Size == values.Length && stored.AsSpan().SequenceEqual(bytes)
                  && read.Select(BitConverter.SingleToInt32Bits).SequenceEqual(back.Select(BitConverter.SingleToInt32Bits)),
                $"{type} on {device}: a tensor's bytes and values as stored");
        }

        // fp8: every code widens and comes back as itself (NaN codes as the format's NaN), and known values have their codes.
        foreach (var (type, nan) in new[] { (ElementType.Float8E4M3, 0x7F), (ElementType.Float8E5M2, 0x7F) })
        {
            var codes = Enumerable.Range(0, 256).Select(c => (byte)c).ToArray();
            var widened = new float[256];
            type.Decode(codes, widened);
            var again = new byte[256];
            type.Encode(widened, again);
            for (int c = 0; c < 256; c++)
            {
                Check(float.IsNaN(widened[c]) ? (again[c] & 0x7F) == nan : again[c] == c, $"{type}: code {c:X2} ({widened[c]}) came back {again[c]:X2}");
            }
        }

        var e4m3 = new byte[6];
        ElementType.Float8E4M3.Encode([1f, -2f, 448f, 464f, 500f, 1f / 512f], e4m3);
        var e5m2 = new byte[3];
        ElementType.Float8E5M2.Encode([1f, 57344f, 61440f], e5m2);
        Check(e4m3.SequenceEqual(new byte[] { 0x38, 0xC0, 0x7E, 0x7E, 0x7F, 0x01 }) && e5m2.SequenceEqual(new byte[] { 0x3C, 0x7B, 0x7C }),
            $"fp8 codes: E4M3 {Convert.ToHexString(e4m3)} (448 the largest, 464 a tie to it, past it NaN), E5M2 {Convert.ToHexString(e5m2)} (past the largest, infinity)");
        var nibbles = new byte[2];
        ElementType.Int4.Encode([1f, -1f, 7f], nibbles);
        Check(nibbles[0] == 0xF1 && nibbles[1] == 0x07, $"int4: the first value in the low nibble: {Convert.ToHexString(nibbles)}");
    }

    private static void ElementTypeConversions(Device device)
    {
        var backend = device.Backend;
        using (var ints = Tensor.FromBytes(MemoryMarshal.AsBytes<int>([0, 5, -1, 1 << 24]), ElementType.Int32, [4], device))
        using (var floats = Tensor.Empty([4], device, ElementType.Float32))
        using (var back = Tensor.Empty([4], device, ElementType.Int32))
        {
            backend.ConvertElements(ints.Storage, floats.Storage);
            backend.ConvertElements(floats.Storage, back.Storage);
            Check(floats.ToArray().SequenceEqual([0f, 5f, -1f, 16777216f]) && back.ToInt32Array().SequenceEqual([0, 5, -1, 1 << 24]),
                $"int32 to float32 and back on {device}: {string.Join(", ", floats.ToArray())}");
        }

        if (!FirstRun(nameof(ElementTypeConversions)))
        {
            return;
        }

        // The host's conversions refuse what the other type does not hold exactly.
        Check(ThrowsWith<ArgumentException>(() => Tensor.From([2.5f], Device.Cpu).ToInt32Array(), "not a whole number"), "a fraction refused as an index");
        Check(ThrowsWith<ArgumentException>(() =>
        {
            var floats = new byte[4];
            ElementConversions.Convert(ElementType.Int32, MemoryMarshal.AsBytes<int>([(1 << 24) + 1]), ElementType.Float32, floats, 1);
        }, "past 2^24"), "an index past 2^24 refused for float32");
        Check(ThrowsWith<ArgumentException>(() => Tensor.Indices([5_000_000_000L], Device.Cpu).ToInt32Array(), "past what int32 holds"), "an int64 index past int32 refused");
        Check(Tensor.Indices([-3L, 7L], Device.Cpu).ToInt64Array().SequenceEqual([-3L, 7L]) && Tensor.Indices([4, 9], Device.Cpu).ToInt64Array().SequenceEqual([4L, 9L]),
            "int64 and int32 read back exactly");

        // The dispatcher: float32 indices to the CPU's int32 gather, converted once, counted and published.
        var table = Tensor.From([0f, 1f, 10f, 11f, 20f, 21f], [3, 2], Device.Cpu);
        var hook = new ConversionsSeen();
        long before = ElementTypes.Conversions;
        long gathers = ElementTypes.ConversionsByOperation.GetValueOrDefault("Gather");
        float[] converted;
        using (Telemetry.Subscribe(hook))
        {
            converted = table.EmbeddingLookup(Tensor.From([2f, 0f], [2], Device.Cpu)).ToArray();
        }

        long afterFloat = ElementTypes.Conversions;
        var exact = table.EmbeddingLookup(Tensor.Indices([2, 0], Device.Cpu)).ToArray();
        Check(converted.SequenceEqual([20f, 21f, 0f, 1f]) && exact.SequenceEqual(converted), $"the converted gather equals the integer one: {string.Join(", ", converted)}");
        Check(afterFloat - before == 1 && ElementTypes.Conversions == afterFloat && ElementTypes.ConversionsByOperation["Gather"] - gathers == 1,
            $"one conversion counted for the float indices, none for int32 ones ({afterFloat - before}, {ElementTypes.Conversions - afterFloat})");
        Check(hook.Events.Any(e => e is { Device: "cpu", Operation: "Gather", From: "float32", To: "int32", Elements: 2 }),
            $"published: {string.Join("; ", hook.Events)}");
        Check(Ops.Gather.IndexParameters.SequenceEqual(["indices"]) && Ops.RouteTopK.IndexParameters.SequenceEqual(["offsets", "tokenOfRow", "slotRow"])
              && Ops.Softmax.IndexParameters.Count == 0 && Ops.ConvertElements.Fallback == KernelSource.Host, "the operations name their index parameters");
    }

    private sealed class ConversionsSeen : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public List<ElementsConverted> Events { get; } = [];

        public void OnElementsConverted(in ElementsConverted e)
        {
            lock (this)
            {
                Events.Add(e);
            }
        }
    }

    // Whether `action` throws a T whose message contains `text`.
    private static bool ThrowsWith<T>(Action action, string text)
        where T : Exception
    {
        try
        {
            action();
            return false;
        }
        catch (T e)
        {
            return e.Message.Contains(text, StringComparison.Ordinal);
        }
    }

    private static void ElementTypeBlockFormats(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ElementTypeBlockFormats)))
        {
            return;
        }

        var q4k = ElementTypes.Get("q4_k");
        var q40 = ElementTypes.Get("Q4_0");
        Check(q4k.Describe() == "q4_k (4.5 bits a value, blocks of 256)" && q4k.ByteCount(512) == 288 && q4k.Scales is { PerBlock: 8, ZeroPoints: true }
              && q4k.IsBlock && q4k.FromFloat32 is null, $"q4_k: {q4k.Describe()}, {q4k.Scales}");
        Check(q40.BitsPerValue == 4.5 && q40.Scales is { PerBlock: 1, ZeroPoints: false, Type: "float16" } && ElementTypes.Get("q8_0").BitsPerValue == 8.5,
            $"q4_0 and q8_0: {q40.Describe()}");
        Check(ThrowsWith<ArgumentException>(() => q4k.ByteCount(100), "whole blocks"), "a block format takes whole blocks");

        var block = new byte[18];
        BitConverter.TryWriteBytes(block.AsSpan(0, 2), BitConverter.HalfToUInt16Bits((Half)0.5f));
        for (int i = 2; i < 18; i++)
        {
            block[i] = (byte)(i * 37);
        }

        var viaType = new float[32];
        var viaGguf = new float[32];
        q40.Decode(block, viaType);
        GgufTypes.Get(2).Dequantize(block, viaGguf);
        Check(viaType.SequenceEqual(viaGguf) && ThrowsWith<NotSupportedException>(() => q40.Encode(viaType, block), "no conversion from float32"),
            "q4_0 decodes as its GGUF type; it has no encoder");
    }

    private static void ElementTypeChoosing(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ElementTypeChoosing)))
        {
            return;
        }

        string cpuReport = Device.Cpu.Backend.Capabilities.NativeReport;
        WithEnvironment(new Dictionary<string, string?> { [ElementTypes.EnvironmentVariable] = null }, () =>
        {
            var auto = ElementTypes.Choose();
            var given = ElementTypes.Choose("FLOAT32", "--dtype");
            var half = ElementTypes.Choose("bf16", "--dtype", Device.Cpu);
            Check(auto.Picked && auto.Type == ElementType.Float32 && auto.Device == "cpu"
                  && auto.ToString() == $"element type: float32 on cpu (picked by auto: cpu computes float32 natively ({cpuReport}))"
                  && !given.Picked && given.IsNative && given.ToString() == "element type: float32 on cpu (selected by the user (--dtype))", $"{auto}; {given}");
            Check(half.Type == ElementType.BFloat16 && !half.Picked && !half.IsNative
                  && half.How == $"selected by the user (--dtype); cpu computes it in float32 (it computes float32 natively ({cpuReport}))",
                $"bfloat16 selected on the CPU, which computes it in float32: {half}");
            Check(ThrowsWith<ArgumentException>(() => ElementTypes.Choose("int8"), "auto, bfloat16, float16, float32"), "a type nothing computes in refused, naming the accepted ones");
            Check(ThrowsWith<ArgumentException>(() => ElementTypes.Choose("float7", "--dtype"), "--dtype 'float7'"), "an unknown name refused, naming the option");
            Check(ElementTypes.Choose(null, "--dtype", device).Type == ElementTypes.Pick(device.Backend.Capabilities) && device.Backend.Capabilities.NativeTypes.Contains(ElementType.Float32),
                $"auto on {device} picks from what it reports: {string.Join(", ", device.Backend.Capabilities.NativeTypes)} ({device.Backend.Capabilities.NativeReport})");
        });
        WithEnvironment(new Dictionary<string, string?> { [ElementTypes.EnvironmentVariable] = "f16" }, () =>
            Check(ElementTypes.Choose() is { Type: var t, Picked: false, Source: ElementTypes.EnvironmentVariable } && t == ElementType.Float16
                  && ElementTypes.Choose("auto").Picked && ElementTypes.Choose("float32").Source == "--dtype",
                "the variable selects (a short name), an option over it"));
        WithEnvironment(new Dictionary<string, string?> { [ElementTypes.EnvironmentVariable] = "int8" }, () =>
            Check(ThrowsWith<ArgumentException>(() => ElementTypes.Choose(), "IDRAK_DEFAULT_DTYPE 'int8'"), "the variable's value refused, naming it"));

        // Auto's pick from what a backend reports: bfloat16 first, then float16, else float32.
        BackendCapabilities Reporting(params ElementType[] types) => new() { FewRows = 1, DecodeAttentionHeadDim = 1, TiledAttentionHeadDim = 1, NativeTypes = types };
        Check(ElementTypes.Pick(Reporting(ElementType.Float16, ElementType.BFloat16, ElementType.Float32)) == ElementType.BFloat16
              && ElementTypes.Pick(Reporting(ElementType.Float16, ElementType.Float32)) == ElementType.Float16
              && ElementTypes.Pick(Reporting(ElementType.Float8E4M3, ElementType.Float32)) == ElementType.Float32
              && ElementTypes.Pick(new BackendCapabilities { FewRows = 1, DecodeAttentionHeadDim = 1, TiledAttentionHeadDim = 1 }) == ElementType.Float32,
            "auto picks bfloat16, then float16, else float32 (fp8 is never picked)");
        // CUDA's native types: what the device reaches of the PTX targets its kernels declare (read from their source), never
        // a list of cards.
        var undeclared = Idrak.Gpu.Cuda.PtxKernels.TensorCoreModules
            .Where(m => Idrak.Gpu.Cuda.PtxKernels.DeclaredTarget(m.Source).ToString()
                        != (m.Name == "fp8 products" ? Idrak.Gpu.Cuda.PtxKernels.Float8Target : Idrak.Gpu.Cuda.PtxKernels.TensorCoreTarget))
            .Select(m => $"{m.Name}: {Idrak.Gpu.Cuda.PtxKernels.DeclaredTarget(m.Source).ToString()}").ToList();
        Check(undeclared.Count == 0, $"every tensor-core module declares the target the gates read (fp8's its own): [{string.Join("; ", undeclared)}]");
        Check(Idrak.Gpu.Cuda.PtxKernels.TargetNumber("sm_80") == 80 && Idrak.Gpu.Cuda.PtxKernels.TargetNumber("sm_90a") == 90
              && Idrak.Gpu.Cuda.PtxKernels.TargetNumber("sm_") == -1 && Idrak.Gpu.Cuda.PtxKernels.TargetNumber("compute_80") == -1,
            "a PTX target read as the compute capability it names (sm_80 → 80, sm_90a → 90; anything else -1)");
        Check(Idrak.Gpu.Cuda.CudaNativeTypes.Of(false, false).SequenceEqual([ElementType.Float32])
              && Idrak.Gpu.Cuda.CudaNativeTypes.Of(true, false).SequenceEqual([ElementType.BFloat16, ElementType.Float32])
              && Idrak.Gpu.Cuda.CudaNativeTypes.Of(true, true).SequenceEqual([ElementType.BFloat16, ElementType.Float8E4M3, ElementType.Float32])
              && Idrak.Gpu.Cuda.CudaNativeTypes.Report(1, 2, true, false) is var report
              && report.Contains("compute capability 1.2", StringComparison.Ordinal) && report.Contains(Idrak.Gpu.Cuda.PtxKernels.TensorCoreTarget, StringComparison.Ordinal)
              && report.Contains(Idrak.Gpu.Cuda.PtxKernels.Float8Target, StringComparison.Ordinal),
            $"CUDA's native types follow the kernels' declared targets the device reaches: {Idrak.Gpu.Cuda.CudaNativeTypes.Report(1, 2, true, false)}");
        Check(Device.Cpu.Backend.Capabilities.IndexType == ElementType.Int32 && Device.Cpu.Backend.IndexTypeOf(Ops.Gather) == ElementType.Int32
              && new BackendCapabilities { FewRows = 1, DecodeAttentionHeadDim = 1, TiledAttentionHeadDim = 1 }.IndexType == ElementType.Float32,
            "the CPU reads int32 indices; a device that says nothing float32");
    }

    // Plan 37, T1b: a decoder that reads a position table by each step's positions (GPT-2's learned positions, and the
    // tiny model's position encoding), generating on `device`: no conversion counted, the same tokens with and without
    // the cache, and the cached lookup (the context's positions in the index type) equal to the float positions' through
    // the dispatcher's conversion, bit for bit: a prompt of several positions, then single steps.
    private static void LearnedPositionDecoding(Device device)
    {
        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .");
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 32, Layers = 2, Heads = 2, KvHeads = 2, HeadDim = 16, FfDim = 64, MaxPositions = 48,
            Norm = DecoderNorm.Layer, NormEpsilon = 1e-5f, Gated = false, Activation = FeedForwardActivation.Gelu, Rope = null,
            QkvBias = true, OutputBias = true, FeedForwardBias = true, TieEmbeddings = true, LearnedPositions = true,
        };
        using var learned = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 3, InitStd = 0.05f });
        learned.Eval();
        var (tiny, tinyTokenizer) = TinyLanguageModel(device);
        var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = 10, Seed = 1 };
        foreach (var (name, model, tokens) in new (string, Sequential, CharTokenizer)[] { ("learned positions", learned, tokenizer), ("a position encoding", tiny, tinyTokenizer) })
        {
            var hook = new ConversionsSeen();
            string cached;
            using (Telemetry.Subscribe(hook))
            {
                cached = new TextGenerator(model, tokens, 32) { KeepCache = false }.Generate("a cat sat", options).Text;
            }

            string recomputed = new TextGenerator(model, tokens, 32) { KeepCache = false }.Generate("a cat sat", options with { UseCache = false }).Text;
            Check(hook.Events.Count == 0, $"{name} on {device}: no conversion counted while decoding, {hook.Events.Count} were: {string.Join("; ", hook.Events.Take(5))}");
            Check(cached.Length > 0 && cached == recomputed, $"{name} on {device}: '{cached}' with the cache, '{recomputed}' without");
        }

        var table = learned.OfType<PositionEmbedding>().Single();
        using var context = new DecodingContext(device, 1, 16);
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            foreach (int steps in (int[])[5, 1, 1])
            {
                context.BeginStep(steps);
                var zeros = Tensor.Zeros([1, steps, spec.Dim], device);
                Check(context.PositionIndices!.ElementType == device.Backend.IndexTypeOf(Ops.Gather), $"{steps} positions: in the gather's index type ({context.PositionIndices.ElementType})");
                float[] indexed = table.ForwardCached(zeros, context).ToArray();
                float[] converted = (zeros + table.Weight.EmbeddingLookup(context.Positions!)).ToArray();
                string? differs = SameBits(converted, indexed, $"{steps} positions from {context.Length}");
                Check(differs is null, differs ?? "");
                context.EndStep(steps);
            }
        }
    }

    private static void IndicesPast2To24(Device device)
    {
        const int Rows = (1 << 24) + 2, Far = (1 << 24) + 1;           // 16,777,217: float32 holds 16,777,216 there
        var values = new float[Rows];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = i % 1000;
        }

        using var scope = new TensorScope();
        var table = Tensor.From(values, [Rows, 1], device, requiresGrad: true);
        var indices = Tensor.Indices([Far, 3, 1 << 24], device);
        Check(indices.ToInt32Array().SequenceEqual([Far, 3, 1 << 24]), $"the indices themselves on {device}: {string.Join(", ", indices.ToInt32Array())}");
        if (indices.ElementType != ElementType.Int32)
        {
            Check(false, $"{device} makes indices past 2^24 as {indices.ElementType}");
            return;
        }

        var gathered = table.EmbeddingLookup(indices);
        Check(gathered.ToArray().SequenceEqual([217f, 3f, 216f]), $"gathered rows: {string.Join(", ", gathered.ToArray())}");
        gathered.Sum().Backward();
        var gradient = table.Grad!.ToArray();
        Check(gradient[Far] == 1f && gradient[1 << 24] == 1f && gradient[3] == 1f && gradient.Sum() == 3f, "the gradient lands at row 16,777,217 (scatter-add)");

        var hot = Tensor.OneHot(Tensor.Indices([Far], device), Rows).ToArray();
        Check(hot[Far] == 1f && hot.Sum() == 1f, "one-hot at column 16,777,217");

        var logits = new float[Rows];
        logits[Far] = 5f;
        var (_, best) = Tensor.From(logits, [1, Rows], device).LogSoftmaxTopK(2);
        var bestIndices = best.ToInt32Array();
        Check(bestIndices[0] == Far && bestIndices[1] == 0, $"top-k's best index: {string.Join(", ", bestIndices)}");

        long before = ElementTypes.Conversions;
        var wide = table.EmbeddingLookup(Tensor.Indices([(long)Far], device)).ToArray();
        Check(wide.SequenceEqual([217f]) && ElementTypes.Conversions - before == 1, "an int64 index through one counted conversion");
    }
}
