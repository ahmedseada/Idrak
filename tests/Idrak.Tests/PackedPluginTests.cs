// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Generation;
using Idrak.Layers;

// Packed-weight formats defined outside the library: registered by name, multiplied through PackedWeight.MatMul (the
// expanded default or a product of their own), and never handed to the kernels for the built-in formats.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] PackedPluginGroup =
    [
        ("packed plugins: the built-in formats are registered; an unknown name lists them; a format of one's own registers, packs by name and has no built-in Format", PackedPluginRegistry),
        ("packed plugins: the PackedFormat overload makes the built-in weights from the same table the registry starts from, even with a built-in name replaced; weight file formats encode and decode through their codecs (float32 exact, float16 and bfloat16 rounding, NaN, unknown format bytes)", PackedBuiltInsAndWeightCodecs),
        ("packed plugins: a Linear layer holding a format of one's own (expanded default product, and its own product) matches the float product at 1 and 70 rows, with the input's gradient; ToFloat32, Save/Load and moving between devices", PackedPluginLinear),
        ("packed plugins: a decoder built with a format of one's own decodes token by token (and through a recorded step) like the built-in bfloat16 decoder, and trains a LoRA adapter", PackedPluginDecoder),
    ];

    private const string PluginExpanded = "test-bf16-expanded", PluginOwnProduct = "test-bf16-own-product";

    // bfloat16-rounded values held as float32: the same weights as the built-in bfloat16 format, multiplied either by
    // the base class's expanded product or by a product of its own (a plain float product, device work only).
    private sealed class RoundedWeight : PackedWeight
    {
        private RoundedWeight(Tensor values, string name, bool ownProduct)
        {
            Values = values;
            Name = name;
            OwnProduct = ownProduct;
        }

        public Tensor Values { get; private set; }

        public bool OwnProduct { get; }

        public override string Name { get; }

        public override int Rows => Values.Shape[0];

        public override int Columns => Values.Shape[1];

        public override long Bytes => 4L * Values.Size;

        public int Products { get; private set; }

        public static RoundedWeight Pack(ReadOnlySpan<float> values, int rows, int columns, Device device, string name, bool ownProduct)
        {
            var rounded = new float[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                rounded[i] = BitConverter.UInt32BitsToSingle((uint)BFloat16Bits(values[i]) << 16);
            }

            return new RoundedWeight(Tensor.Persistent(rounded, [rows, columns], device, requiresGrad: false), name, ownProduct);
        }

        private static ushort BFloat16Bits(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }

        public override Tensor Dequantize() => Values * 1f;

        public override IEnumerable<Tensor> Buffers() => [Values];

        protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move) => Values = move(Values, device);

        public override Tensor MatMul(Tensor input)
        {
            Products++;
            return OwnProduct ? input.MatMul(Values) : base.MatMul(input);
        }

        public override void Dispose() => Values.Dispose();
    }

    private static void RegisterPlugins()
    {
        PackedWeight.Register(PluginExpanded, (values, rows, columns, device) => RoundedWeight.Pack(values, rows, columns, device, PluginExpanded, ownProduct: false));
        PackedWeight.Register(PluginOwnProduct, (values, rows, columns, device) => RoundedWeight.Pack(values, rows, columns, device, PluginOwnProduct, ownProduct: true));
    }

    private static void PackedPluginRegistry(Device device)
    {
        Check(new[] { "int8", "int4", "bfloat16" }.All(PackedWeight.FormatNames.Contains), $"built-in formats registered: {string.Join(", ", PackedWeight.FormatNames)}");
        try
        {
            PackedWeight.FromValues("no-such-format", [1f], 1, 1, device);
            Check(false, "an unknown format is refused");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("int4", StringComparison.Ordinal) && ex.Message.Contains("PackedWeight.Register", StringComparison.Ordinal), $"the message lists the formats and how to add one: {ex.Message}");
        }

        float[] values = [.. Enumerable.Range(0, 12 * 8).Select(i => MathF.Sin(i) * 0.3f)];
        foreach (var (name, format) in new[] { ("INT8", PackedFormat.Int8), ("int4", PackedFormat.Int4), ("BFloat16", PackedFormat.BFloat16) })
        {
            using var builtIn = PackedWeight.FromValues(name, values, 12, 8, device);
            Check(builtIn.Format == format && builtIn.Name == name.ToLowerInvariant() && builtIn.Device == device, $"'{name}' (any case) packs the built-in {format} weights");
        }

        RegisterPlugins();
        Check(PackedWeight.FormatNames.Contains(PluginExpanded) && PackedWeight.FormatNames.Contains(PluginOwnProduct), "formats of one's own registered");
        using var plugin = PackedWeight.FromValues(PluginExpanded, values, 12, 8, device);
        Check(plugin is RoundedWeight && plugin.Format is null && plugin.Name == PluginExpanded && plugin.Rows == 12 && plugin.Columns == 8
              && plugin.Bytes == 4 * 12 * 8 && plugin.Device == device, "a format of one's own: its class, no built-in Format, its name and shape");
        try
        {
            PackedWeight.Register("test-wrong-shape", (v, r, c, d) => RoundedWeight.Pack(v, c, r, d, "test-wrong-shape", false));
            PackedWeight.FromValues("test-wrong-shape", values, 12, 8, device);
            Check(false, "a factory making the wrong shape is refused");
        }
        catch (InvalidOperationException)
        {
        }
    }

    private static void PackedPluginLinear(Device device)
    {
        RegisterPlugins();
        var r = new Random(53);
        const int Rows = 64, Columns = 96;
        var values = Enumerable.Range(0, Rows * Columns).Select(_ => (float)(r.NextDouble() * 2 - 1) * 0.1f).ToArray();
        var inputs = Enumerable.Range(0, 70 * Rows).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        var bias = Enumerable.Range(0, Columns).Select(i => 0.01f * i).ToArray();
        foreach (string name in new[] { PluginExpanded, PluginOwnProduct })
        {
            using var linear = Linear.FromPacked(PackedWeight.FromValues(name, values, Rows, Columns, device), Tensor.Persistent(bias, [Columns], device, requiresGrad: false));
            var weight = (RoundedWeight)linear.PackedWeight!;
            Check(linear.Int8 is null && linear.Int4 is null && linear.BFloat16 is null, $"{name}: none of the built-in typed views");
            Check(linear.Buffers().Single() == weight.Values && linear.ToString().Contains(name, StringComparison.Ordinal), $"{name}: its buffer and its name in {linear}");
            try
            {
                _ = linear.Weight;
                Check(false, $"{name}: no float weight");
            }
            catch (InvalidOperationException ex)
            {
                Check(ex.Message.Contains(nameof(RoundedWeight), StringComparison.Ordinal) && ex.Message.Contains("ToFloat32", StringComparison.Ordinal), $"{name}: {ex.Message}");
            }

            using var expanded = weight.Dequantize();
            foreach (int rows in new[] { 1, 70 })
            {
                using var scope = new TensorScope();
                var x = Tensor.From(inputs.AsSpan(0, rows * Rows), [rows, Rows], device);
                AssertClose((x.MatMul(expanded) + Tensor.From(bias, [Columns], device)).ToArray(), linear.Forward(x).ToArray(), 1e-4f,
                    $"{name}, {rows} rows: the product equals the expanded one");

                // The input's gradient through the format's product: g · Wᵀ with g = 1.
                var x3 = Tensor.Persistent(inputs.AsSpan(0, rows * Rows).ToArray(), [1, rows, Rows], device, requiresGrad: true);
                linear.Forward(x3).Sum().Backward();
                var reference = Tensor.Persistent(inputs.AsSpan(0, rows * Rows).ToArray(), [1, rows, Rows], device, requiresGrad: true);
                reference.MatMul(expanded).Sum().Backward();
                AssertClose(reference.Grad!.ToArray(), x3.Grad!.ToArray(), 1e-4f, $"{name}, {rows} rows: the input's gradient");
                x3.Dispose();
                reference.Dispose();
            }

            Check(weight.Products >= 4, $"{name}: the layer ran the format's MatMul ({weight.Products})");
        }

        // ToFloat32 expands a format of one's own; Save keeps its buffers exact even in a bfloat16 file.
        using (var scope = new TensorScope())
        {
            var x = Tensor.From(inputs.AsSpan(0, 3 * Rows), [3, Rows], device);
            var model = new Sequential(Linear.FromPacked(PackedWeight.FromValues(PluginExpanded, values, Rows, Columns, device)));
            var source = (RoundedWeight)model.Descendants().OfType<Linear>().Single().PackedWeight!;
            var stored = values.Select(v => v + 1e-6f).ToArray();             // not bfloat16 values: a bfloat16 file would round them
            source.Values.Load(stored);
            var expected = model.Forward(x).ToArray();
            string path = Path.GetTempFileName();
            try
            {
                model.Save(path, WeightFormat.BFloat16);
                var other = values.Select(v => v * 0.5f + 0.0123f).ToArray();
                using var reloaded = new Sequential(Linear.FromPacked(PackedWeight.FromValues(PluginExpanded, other, Rows, Columns, device)));
                reloaded.Load(path);
                var reloadedWeight = (RoundedWeight)reloaded.Descendants().OfType<Linear>().Single().PackedWeight!;
                Check(reloadedWeight.Values.ToArray().SequenceEqual(stored), "Save/Load: the format's buffers stored exactly");
                Check(reloaded.Forward(x).ToArray().SequenceEqual(expected), "Save/Load: the same outputs");
            }
            finally
            {
                File.Delete(path);
            }

            Check(model.ToFloat32(trainable: false) == 1 && model.Descendants().OfType<Linear>().Single() is { PackedWeight: null } floatLayer
                  && floatLayer.Weight.Shape.SequenceEqual([Rows, Columns]), "ToFloat32 turns a format of one's own into float weights");
            AssertClose(expected, model.Forward(x).ToArray(), 1e-5f, "ToFloat32: the same outputs");
            model.Dispose();
        }

        // Moving between devices: the format's own tensors move (skipped with the CPU alone).
        var other2 = device.Type == DeviceType.Cpu ? (Device.IsCudaAvailable ? Device.Cuda(0) : null) : Device.Cpu;
        if (other2 is not null)
        {
            using var scope = new TensorScope();
            using var model = new Sequential(Linear.FromPacked(PackedWeight.FromValues(PluginOwnProduct, values, Rows, Columns, device)));
            var x = Tensor.From(inputs.AsSpan(0, 2 * Rows), [2, Rows], device);
            var expected = model.Forward(x).ToArray();
            model.To(other2);
            var moved = model.Descendants().OfType<Linear>().Single().PackedWeight!;
            Check(moved.Device == other2 && moved.Buffers().All(b => b.Device == other2), $"moved to {other2}");
            AssertClose(expected, model.Forward(Tensor.From(inputs.AsSpan(0, 2 * Rows), [2, Rows], other2)).ToArray(), 1e-4f, $"on {other2}: the same outputs");
            model.To(device);
            AssertClose(expected, model.Forward(x).ToArray(), 1e-4f, $"back on {device}: the same outputs");
        }
    }

    private static void PackedPluginDecoder(Device device)
    {
        RegisterPlugins();
        var spec = SmallSpec;
        int[] ids = [1, 5, 3, 7, 2, 9, 4, 11, 6];
        float[][] Decode(Sequential model)
        {
            using var context = new DecodingContext(device, 1, 16, KeyValueFormat.Float32);
            var steps = new List<float[]>();
            using (Autograd.NoGrad())
            {
                foreach (int id in ids)
                {
                    using var scope = new TensorScope();
                    steps.Add(model.ForwardCached(Tensor.From([id], [1, 1], device), context).ToArray()[^spec.Vocabulary..]);
                }
            }

            return [.. steps];
        }

        float[] Prompt(Sequential model)
        {
            using var scope = new TensorScope();
            using var context = new DecodingContext(device, 1, 16, KeyValueFormat.Float32);
            using (Autograd.NoGrad())
            {
                return model.ForwardCached(Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device), context).ToArray();
            }
        }

        using var bf16 = spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device, BFloat16 = true });
        using var dense = spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device });
        var reference = Decode(bf16);
        var denseSteps = Decode(dense);
        var referencePrompt = Prompt(bf16);
        float range = reference.SelectMany(s => s).Max(MathF.Abs);
        foreach (string name in new[] { PluginExpanded, PluginOwnProduct })
        {
            using var model = spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device, PackedFormatName = name, Int4 = true });
            var linears = model.Descendants().OfType<Linear>().ToList();
            Check(linears.Count > 0 && linears.All(l => l.PackedWeight is RoundedWeight { Format: null }), $"{name}: every projection in the named format (over Int4)");
            var steps = Decode(model);
            for (int i = 0; i < ids.Length; i++)
            {
                AssertClose(reference[i], steps[i], 2e-3f * range, $"{name}, step {i}: like the built-in bfloat16 decoder");
                AssertClose(denseSteps[i], steps[i], 0.05f * range, $"{name}, step {i}: near the float32 decoder");
            }

            AssertClose(referencePrompt, Prompt(model), 2e-3f * range, $"{name}: the prompt's logits");
            Check(linears.All(l => ((RoundedWeight)l.PackedWeight!).Products > 0), $"{name}: every layer ran the format's product");

            // A recorded decoding step: on CUDA the expanded default is not recorded (the step re-runs), a product of the
            // format's own is; either way the tokens are those of direct steps.
            const int Steps = 6;
            int[] Run(bool useGraph, out ComputeGraph? recorded)
            {
                recorded = null;
                using var context = new DecodingContext(device, 1, 16, KeyValueFormat.Float32);
                using var sampler = new TokenSampler(device, 1, spec.Vocabulary, Steps + 1) { Temperature = 0.9f, Seed = 7 };
                using (Autograd.NoGrad())
                {
                    using (var scope = new TensorScope())
                    {
                        sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f, 3f], [1, 3], device), context));
                    }

                    void Step() => sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(1, 1), context));
                    var graph = useGraph ? context.CaptureStep(Step) : null;
                    for (int s = 0; s < Steps; s++)
                    {
                        if (graph is not null)
                        {
                            context.ReplayStep(graph);
                        }
                        else
                        {
                            using var scope = new TensorScope();
                            Step();
                        }
                    }

                    recorded = graph;
                }

                return [.. sampler.Read(0, Steps + 1).Select(step => step[0].Id)];
            }

            var direct = Run(false, out _);
            var replayed = Run(true, out var captured);
            using (captured)
            {
                Check(direct.SequenceEqual(replayed), $"{name}: direct [{string.Join(",", direct)}] vs recorded [{string.Join(",", replayed)}]");
                if (device.Type == DeviceType.Cuda)
                {
                    bool own = name == PluginOwnProduct;
                    Check(captured!.IsRecorded == own, $"{name}: recorded {captured.IsRecorded} ({captured.FailureReason})");
                }
            }
        }

        // LoRA over a format of one's own: the base product is the format's, the adapter trains.
        using (var model = spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device, PackedFormatName = PluginExpanded }))
        using (var scope = new TensorScope())
        {
            int added = model.AddLora(4, 8f, l => l.Name is "q" or "v", freezeBase: true, new Random(3));
            Check(added > 0, "adapters added");
            var logits = model.Forward(Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device));
            logits.Square().Mean().Backward();
            var adapters = model.Descendants().OfType<Linear>().Where(l => l.Adapter is not null).ToList();
            Check(adapters.Count == added && adapters.All(l => l.PackedWeight is RoundedWeight && l.Adapter!.B.Grad is { } g && g.ToArray().Any(v => v != 0f)),
                "the adapters' gradients through the format's product");
        }
    }

    private static void PackedBuiltInsAndWeightCodecs(Device device)
    {
        var r = new Random(3);
        float[] values = [.. Enumerable.Range(0, 64 * 48).Select(_ => r.NextSingle() * 2 - 1)];
        foreach (var (format, name) in new[] { (PackedFormat.Int8, "int8"), (PackedFormat.Int4, "int4"), (PackedFormat.BFloat16, "bfloat16") })
        {
            using var byEnum = PackedWeight.FromValues(format, values, 64, 48, device);
            using var byName = PackedWeight.FromValues(name, values, 64, 48, device);
            Check(byEnum.Format == format && byName.Format == format, $"{name}: the built-in format both ways");
            using var a = byEnum.Dequantize();
            using var b = byName.Dequantize();
            AssertClose(a.ToArray(), b.ToArray(), 0f, $"{name}: the same weights by enum and by name");
        }

        try
        {
            using var invalid = PackedWeight.FromValues((PackedFormat)99, values, 64, 48, device);
            Check(false, "an undefined PackedFormat should fail");
        }
        catch (ArgumentOutOfRangeException)
        {
        }

        // A registered "int8" replaces the name, not the enum's built-in.
        PackedWeight.Register("int8", BFloat16Weight.FromValues);
        try
        {
            using var byName = PackedWeight.FromValues("int8", values, 64, 48, device);
            using var byEnum = PackedWeight.FromValues(PackedFormat.Int8, values, 64, 48, device);
            Check(byName.Format == PackedFormat.BFloat16 && byEnum.Format == PackedFormat.Int8, "a replaced name; the enum still makes int8");
        }
        finally
        {
            PackedWeight.Register("int8", Int8Weight.Quantize);
        }

        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // the codecs are host code: once is enough
        }

        float[] samples = [0f, -0f, 1f, -1.5f, 3.14159265f, 65504f, 1e-8f, 1e30f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0.1f];
        foreach (var format in new[] { WeightFormat.Float32, WeightFormat.Float16, WeightFormat.BFloat16 })
        {
            var codec = WeightCodec.For(format);
            var bytes = new byte[samples.Length * codec.BytesPerValue];
            codec.Encode(samples, bytes);
            var back = new float[samples.Length];
            codec.Decode(bytes, back);
            for (int i = 0; i < samples.Length; i++)
            {
                float expected = format switch
                {
                    WeightFormat.Float32 => samples[i],
                    WeightFormat.Float16 => (float)(Half)samples[i],
                    _ => float.IsNaN(samples[i]) ? float.NaN
                        : BitConverter.UInt32BitsToSingle((BitConverter.SingleToUInt32Bits(samples[i]) + 0x7FFFu + ((BitConverter.SingleToUInt32Bits(samples[i]) >> 16) & 1u)) & 0xFFFF0000u),
                };
                Check(float.IsNaN(expected) ? float.IsNaN(back[i]) : BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(back[i]),
                    $"{format}: {samples[i]} → {back[i]}, expected {expected}");
            }
        }

        Check(WeightCodec.For(WeightFormat.Float32).BytesPerValue == 4 && WeightCodec.For(WeightFormat.Float16).BytesPerValue == 2
            && WeightCodec.For(WeightFormat.BFloat16).BytesPerValue == 2, "bytes per value");
        try
        {
            WeightCodec.For((WeightFormat)7);
            Check(false, "an unknown format byte should fail");
        }
        catch (InvalidDataException ex)
        {
            Check(ex.Message.Contains("Unknown weight format 7", StringComparison.Ordinal), ex.Message);
        }
    }
}
