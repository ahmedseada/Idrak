// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Formats;
using Idrak.Abstraction.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;

// Plan 37, Q0: weights kept in their file's format (StoredWeight), their products equal the float32 weights' bit for bit,
// a GGUF model loaded kept and expanded, and the setting.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] KeepPackedGroup =
    [
        ("keep packed: every registered block format this device multiplies (each of ggml's on the CPU) gives the float32 weights' products bit for bit (1, 4 and 37 rows, with a bias) and expands to their values, from random blocks", StoredBlockProducts),
        ("keep packed: int8, float8 E4M3 and E5M2 codes with one scale, a scale per row, block scales and zero points give the float32 weights' products bit for bit, their values (code − zero) · scale", StoredScaledProducts),
        ("keep packed: a Q8_0 GGUF model loads with its matrices kept as stored (about the file's bytes, the load's summary and telemetry say so) and gives the loss of the model expanded at load bit for bit; IDRAK_KEEP_PACKED=0 and KeepPacked false expand them, saying why", StoredGgufModel),
    ];

    // Blocks of random bytes of `type` for `values` values, each block whose values are not all finite zeroed (a NaN or
    // infinite scale), with its values widened.
    private static (byte[] Bytes, float[] Values) RandomBlocks(ElementType type, int values, Random random)
    {
        var bytes = new byte[type.ByteCount(values)];
        random.NextBytes(bytes);
        int blockBytes = type.BlockBits / 8;
        var block = new float[type.BlockSize];
        for (int b = 0; b < values / type.BlockSize; b++)
        {
            var span = bytes.AsSpan(b * blockBytes, blockBytes);
            type.Decode(span, block);
            if (!block.All(float.IsFinite) || block.Any(v => MathF.Abs(v) > 1e6f))
            {
                span.Clear();
            }
        }

        var widened = new float[values];
        type.Decode(bytes, widened);
        return (bytes, widened);
    }

    // [n, k] stored values as the [k, n] float32 weights.
    private static float[] Transposed(float[] stored, int n, int k)
    {
        var w = new float[k * n];
        for (int j = 0; j < n; j++)
        {
            for (int p = 0; p < k; p++)
            {
                w[p * n + j] = stored[j * k + p];
            }
        }

        return w;
    }

    // The stored layer's outputs against the float32 layer's, bit for bit, for 1, 4 and 37 rows.
    private static void SameProducts(Device device, StoredWeight stored, float[] floatWeights, string what, Random random)
    {
        int k = stored.Rows, n = stored.Columns;
        var biasValues = Enumerable.Range(0, n).Select(_ => random.NextSingle() - 0.5f).ToArray();
        using var expanded = stored.Dequantize();
        Check(expanded.ToArray().Select(BitConverter.SingleToInt32Bits).SequenceEqual(floatWeights.Select(BitConverter.SingleToInt32Bits)),
            $"{what} on {device}: the expansion equals the values");
        using var plain = Linear.FromWeights(Tensor.Persistent(floatWeights, [k, n], device), Tensor.Persistent(biasValues, [n], device));
        using var kept = Linear.FromPacked(stored, Tensor.Persistent(biasValues, [n], device));
        using (Autograd.NoGrad())
        {
            foreach (int m in new[] { 1, 4, 37 })
            {
                using var scope = new TensorScope();
                var x = Tensor.From(Enumerable.Range(0, m * k).Select(_ => random.NextSingle() * 2 - 1).ToArray(), [m, k], device);
                var a = plain.Forward(x).ToArray();
                var b = kept.Forward(x).ToArray();
                int differs = Enumerable.Range(0, a.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]), -1);
                Check(differs < 0, $"{what} on {device}, {m} rows: output {differs} is {(differs < 0 ? 0 : b[differs])}, the float32 weights give {(differs < 0 ? 0 : a[differs])}");
            }
        }
    }

    private static void StoredBlockProducts(Device device)
    {
        var random = new Random(37);
        var formats = ElementTypes.All.Where(t => t.IsBlock && t.ToFloat32 is not null && WeightStorage.Multiplies(device.Backend, t)).ToList();
        Check(!device.Backend.Capabilities.StoresAnyFormat || formats.Count >= 20, $"{device}: the block formats it multiplies: {string.Join(", ", formats)}");
        foreach (var type in formats)
        {
            int n = 13, k = 2 * type.BlockSize;
            var (bytes, values) = RandomBlocks(type, n * k, random);
            using var stored = StoredWeight.FromStored(type, bytes, k, n, null, device);
            Check(stored.Bytes == ((long)bytes.Length + 3) / 4 * 4 && stored.Type == type && stored.Name == type.Name, $"{type}: {stored.Bytes} bytes for {bytes.Length} stored");
            SameProducts(device, stored, Transposed(values, n, k), type.Name, random);
        }
    }

    private static void StoredScaledProducts(Device device)
    {
        var random = new Random(41);
        int n = 12, k = 64;
        foreach (var (type, layout, zeros) in new[]
                 {
                     (ElementType.Int8, new StoredScales(1, k, 1), true),          // a scale and a zero point per row
                     (ElementType.Float8E4M3, new StoredScales(4, 32, 2), false),  // blocks of 4 rows × 32 columns
                     (ElementType.Float8E5M2, new StoredScales(n, k, 1), false),   // one scale for the tensor
                     (ElementType.Float8E4M3, default(StoredScales), false),       // no scales
                 })
        {
            if (!WeightStorage.Multiplies(device.Backend, type))
            {
                continue;
            }

            var codes = new byte[n * k];
            random.NextBytes(codes);
            var decoded = new float[n * k];
            type.Decode(codes, decoded);
            for (int i = 0; i < codes.Length; i++)
            {
                if (!float.IsFinite(decoded[i]))
                {
                    codes[i] = 0;                                           // fp8's NaN codes: none in a weight
                    decoded[i] = 0f;
                }
            }

            StoredScaling? scaling = null;
            if (layout.HasScales)
            {
                int count = layout.IndexOf(n - 1, k - 1) + 1;
                var s = Enumerable.Range(0, count).Select(_ => 0.01f + random.NextSingle()).ToArray();
                var z = zeros ? Enumerable.Range(0, count).Select(_ => (float)random.Next(-3, 4)).ToArray() : null;
                scaling = new StoredScaling(s, z, layout);
                for (int r = 0; r < n; r++)
                {
                    for (int c = 0; c < k; c++)
                    {
                        int i = layout.IndexOf(r, c);
                        decoded[r * k + c] = z is null ? decoded[r * k + c] * s[i] : (decoded[r * k + c] - z[i]) * s[i];
                    }
                }
            }

            using var stored = StoredWeight.FromStored(type, codes, k, n, scaling, device);
            SameProducts(device, stored, Transposed(decoded, n, k), $"{type} with {(layout.HasScales ? $"scales [{layout.BlockRows}, {layout.BlockColumns}]" : "no scales")}{(zeros ? " and zero points" : "")}", random);
        }
    }

    private sealed class KeptSeen : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public List<WeightsKept> Events { get; } = [];

        public void OnWeightsKept(in WeightsKept e)
        {
            lock (this)
            {
                Events.Add(e);
            }
        }
    }

    private static void LazyGgufReadsTheMapping(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(LazyGgufReadsTheMapping)))
        {
            return;
        }

        var cpu = Device.Cpu;
        string cache = Path.Combine(Path.GetTempPath(), "ns-mapped-" + Guid.NewGuid().ToString("N"));
        try
        {
            string gguf = TestData("gguf/tiny-qwen3-q8.gguf");
            string folder = GgufModel.Prepare(gguf, cache);
            using var eager = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu, KeepPacked = true, WeightMapping = Idrak.Models.Abstractions.WeightMappings.Map });
            var start = MappedWeights.Read();
            using var lazy = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu, KeepPacked = true, LazyLayers = true, WeightMapping = Idrak.Models.Abstractions.WeightMappings.Map });
            int[] ids = [.. lazy.Tokenizer!.Encode("the answer is in the thin rain, and then another answer")];
            TrainingSequence[] probe = [new TrainingSequence(ids, [.. ids.Select(_ => true)])];
            float lossEager = FineTuner.Evaluate(eager, probe), lossLazy = FineTuner.Evaluate(lazy, probe);
            var counted = MappedWeights.Read().Since(start);
            var linears = lazy.Network.Descendants().OfType<DecoderBlock>().SelectMany(b => b.Descendants().OfType<Linear>()).ToList();
            var inPlace = linears.Where(l => l.PackedWeight is StoredWeight { Mapping: not null }).ToList();
            long inPlaceBytes = inPlace.Sum(l => ((StoredWeight)l.PackedWeight!).Type.ByteCount((long)l.InFeatures * l.OutFeatures));
            Check(eager.Network.Descendants().OfType<Linear>().All(l => l.PackedWeight is not StoredWeight { Mapping: not null }), "eager: nothing read in place on one device");
            Check(inPlace.Count > 0 && inPlace.Any(l => ((StoredWeight)l.PackedWeight!).Type.Name == "q8_0") && counted.Mapped == inPlace.Count && counted.MappedBytes == inPlaceBytes
                  && (inPlace.Count == linears.Count || counted.CopiedFor.Keys.Any(k => k.StartsWith("at load: ", StringComparison.Ordinal))),
                $"{inPlace.Count} of {linears.Count} block matrices read in place ({inPlaceBytes} bytes), the others' copies counted: {counted}");
            Check(BitConverter.SingleToInt32Bits(lossLazy) == BitConverter.SingleToInt32Bits(lossEager), $"loss {lossLazy} read in place, {lossEager} copied");
        }
        finally
        {
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }

    private static void StoredGgufModel(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(StoredGgufModel)))
        {
            return;
        }

        var cpu = Device.Cpu;
        string cache = Path.Combine(Path.GetTempPath(), "ns-kept-" + Guid.NewGuid().ToString("N"));
        try
        {
            string gguf = TestData("gguf/tiny-qwen3-q8.gguf");
            string folder = GgufModel.Prepare(gguf, cache);
            var hook = new KeptSeen();
            PretrainedModel keptModel;
            using (Telemetry.Subscribe(hook))
            {
                keptModel = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu, KeepPacked = true });
            }

            using var kept = keptModel;
            using var expanded = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu, KeepPacked = false });
            var summary = kept.KeptWeights;
            var off = expanded.KeptWeights;
            Check(summary is { Kept: > 0, Expanded: 0 } && summary.Formats.Contains("q8_0") && summary.ToString().StartsWith("weights kept packed: ", StringComparison.Ordinal),
                $"kept: {summary}");
            Check(off is { Kept: 0 } && off.Expanded == summary!.Kept && off.Why is { } why && why.Contains(WeightStorage.EnvironmentVariable, StringComparison.Ordinal)
                  && off.ToString().StartsWith("weights expanded: ", StringComparison.Ordinal), $"expanded: {off}");
            // Q8_0: 34 bytes a block of 32 values (8.5 bits a value) where float32 takes 128 (each matrix's storage whole words).
            Check(Math.Abs(summary!.KeptBytes * 128 - off!.ExpandedBytes * 34) <= 4L * 128 * summary.Kept && summary.KeptBytes < new FileInfo(gguf).Length,
                $"{summary.KeptBytes} bytes kept for {off.ExpandedBytes} float32 bytes, a file of {new FileInfo(gguf).Length}");
            Check(hook.Events.Any(e => e.Kept == summary.Kept && e.KeptBytes == summary.KeptBytes && e.Formats.Contains("q8_0", StringComparison.Ordinal)),
                $"published: {string.Join("; ", hook.Events)}");
            using (var stream = new MemoryStream())
            {
                using (var writer = new Utf8JsonWriter(stream))
                {
                    TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UnixEpoch, hook.Events[^1]));
                }

                string json = System.Text.Encoding.UTF8.GetString(stream.ToArray());
                Check(json.Contains("\"event\":\"weights_kept\"", StringComparison.Ordinal) && json.Contains("\"kept_bytes\"", StringComparison.Ordinal), $"JSON: {json}");
            }

            int[] ids = [.. kept.Tokenizer!.Encode("the answer is in the thin rain, and then another answer")];
            TrainingSequence[] probe = [new TrainingSequence(ids, [.. ids.Select(_ => true)])];
            float lossKept = FineTuner.Evaluate(kept, probe), lossExpanded = FineTuner.Evaluate(expanded, probe);
            Check(BitConverter.SingleToInt32Bits(lossKept) == BitConverter.SingleToInt32Bits(lossExpanded), $"loss {lossKept} kept, {lossExpanded} expanded at load");

            WithEnvironment(new Dictionary<string, string?> { [WeightStorage.EnvironmentVariable] = "0" }, () =>
            {
                using var byVariable = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu });
                Check(!WeightStorage.DefaultKeep && byVariable.KeptWeights is { Kept: 0 } v && v.Expanded == summary.Kept, $"IDRAK_KEEP_PACKED=0: {byVariable.KeptWeights}");
            });
            WithEnvironment(new Dictionary<string, string?> { [WeightStorage.EnvironmentVariable] = null }, () =>
                Check(WeightStorage.DefaultKeep, "on by default"));
        }
        finally
        {
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }
}
