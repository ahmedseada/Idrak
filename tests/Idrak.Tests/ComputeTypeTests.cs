// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Diagnostics;
using Idrak.Layers;
using Idrak.Models;

// Plan 37, T2: a tensor's elements converted to the 16-bit, int16, uint16 and float64 types on its device and back; the
// precision a model's run brings to its products (a caller's choice over it); the key/value cache that keeps the run's
// compute type; and a load that chooses its element type per device, says how, and applies it.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ComputeTypesGroup =
    [
        ("element types: a tensor converts to bfloat16, float16, int16, uint16 and float64 on its device (half or a quarter of the words, rounded to nearest even and saturated as the types' own conversions) and back to float32; the same type is a copy", TensorConversions),
        ("element types: a model's products run in its run's type (Sequential.Precision through MixedPrecision.Prefer), a caller's scope or IDRAK_MATMUL over it; bfloat16, float16 and fp8 map to their precisions, float16 apart from the tensor cores", ModelPrecision),
        ("element types: kv cache: auto keeps the run's compute type (bfloat16 for a bfloat16 run, float32 where no format of the type is registered), --kv or IDRAK_KV_CACHE_TYPE selects (short names too), an unknown format refused naming the source", CacheTypeChoice),
        ("element types: a load chooses its element type for its device and says how (picked by auto with what the device reports, selected by --dtype), its network multiplies in it, its cache keeps it, a load choosing its weights by memory leaves float32 out for a 16-bit run, and telemetry carries the device's report", LoadedElementType),
    ];

    private static void TensorConversions(Device device)
    {
        // Past two vectors of 16-bit values, so the widening's vector path and its tail both run.
        float[] values = [.. Enumerable.Range(0, 77).Select(i => (i - 38) * 913.37f), 0.1f, -0.5f, 2.5f, 65504f, 70000f, -70000f];
        using var x = Tensor.From(values, [values.Length], device);
        foreach (var type in new[] { ElementType.BFloat16, ElementType.Float16, ElementType.Int16, ElementType.UInt16, ElementType.Float64 })
        {
            var bytes = new byte[type.ByteCount(values.Length)];
            type.Encode(values, bytes);
            var expected = new float[values.Length];
            type.Decode(bytes, expected);
            using var narrow = x.ConvertTo(type);
            using var wide = narrow.ConvertTo(ElementType.Float32);
            Check(narrow.ElementType == type && narrow.Storage.Length == type.StorageWords(values.Length) && narrow.ToBytes().AsSpan().SequenceEqual(bytes),
                $"{type} on {device}: {narrow.Storage.Length} words, the bytes of the type's own conversion");
            Check(wide.ElementType == ElementType.Float32 && wide.ToArray().Select(BitConverter.SingleToInt32Bits).SequenceEqual(expected.Select(BitConverter.SingleToInt32Bits)),
                $"{type} on {device}: back to float32 as the type widens");
        }

        using var copy = x.ConvertTo(ElementType.Float32);
        Check(!ReferenceEquals(copy, x) && copy.ToArray().SequenceEqual(values), "the same type: a copy");
    }

    private static void ModelPrecision(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ModelPrecision)))
        {
            return;
        }

        var chosen = MixedPrecision.Chosen;
        try
        {
            MixedPrecision.Chosen = null;
            MatMulPrecision Inside(MatMulPrecision? model)
            {
                using var scope = MixedPrecision.Prefer(model);
                return MixedPrecision.Current;
            }

            Check(Inside(MatMulPrecision.BFloat16) == MatMulPrecision.BFloat16 && Inside(null) == MatMulPrecision.Float32 && MixedPrecision.Current == MatMulPrecision.Float32,
                "a model's precision applies inside its pass and ends with it");
            using (MixedPrecision.Use(MatMulPrecision.Float32))
            {
                Check(Inside(MatMulPrecision.BFloat16) == MatMulPrecision.Float32, "a caller's scope stays");
            }

            MixedPrecision.Chosen = MatMulPrecision.Float8;
            Check(Inside(MatMulPrecision.BFloat16) == MatMulPrecision.Float8 && MixedPrecision.UsesTensorCores, "the process's choice (IDRAK_MATMUL) stays");
            MixedPrecision.Chosen = null;
            Check(MixedPrecision.For(ElementType.BFloat16) == MatMulPrecision.BFloat16 && MixedPrecision.For(ElementType.Float16) == MatMulPrecision.Float16
                  && MixedPrecision.For(ElementType.Float8E5M2) == MatMulPrecision.Float8 && MixedPrecision.For(ElementType.Float32) == MatMulPrecision.Float32
                  && MixedPrecision.For(ElementType.Int8) == MatMulPrecision.Float32, "each type's precision");
            using (MixedPrecision.Use(MatMulPrecision.Float16))
            {
                Check(!MixedPrecision.UsesTensorCores, "float16 products are not the bfloat16 tensor cores'");
            }

            // A network's precision reaches its layers' products and goes with the pass.
            var seen = new List<MatMulPrecision>();
            var network = new Sequential(new Probe(seen), new Probe(seen)) { Precision = MatMulPrecision.BFloat16 };
            using (Autograd.NoGrad())
            using (var input = Tensor.Zeros([1, 2], Device.Cpu))
            using (network.Forward(input))
            {
            }

            Check(seen.Count == 2 && seen.All(p => p == MatMulPrecision.BFloat16) && MixedPrecision.Current == MatMulPrecision.Float32,
                $"the network's layers ran in bfloat16: {string.Join(", ", seen)}");
        }
        finally
        {
            MixedPrecision.Chosen = chosen;
        }
    }

    // A layer that notes the precision its forward pass runs in.
    private sealed class Probe(List<MatMulPrecision> seen) : Module
    {
        protected override Tensor ForwardCore(Tensor input)
        {
            seen.Add(MixedPrecision.Current);
            return input.Clone();
        }
    }

    private static void CacheTypeChoice(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(CacheTypeChoice)))
        {
            return;
        }

        WithEnvironment(new Dictionary<string, string?> { [KeyValueLayouts.EnvironmentVariable] = null }, () =>
        {
            var kept = KeyValueLayouts.Choose(null, ElementType.BFloat16);
            var none = KeyValueLayouts.Choose("auto", ElementType.Float16);
            var named = KeyValueLayouts.Choose("bf16", ElementType.Float32);
            Check(kept is { Picked: true } && kept.Layout.Name == "bfloat16" && kept.ToString() == "kv cache: bfloat16 (picked by auto: the run computes in bfloat16)",
                $"auto keeps the run's type: {kept}");
            Check(none.Layout.Name == "float32" && none.How.Contains("no cache format registered", StringComparison.Ordinal), $"no float16 format: {none}");
            Check(named is { Picked: false, Source: "--kv" } && named.Layout.Name == "bfloat16" && KeyValueLayouts.Choose("INT8", ElementType.BFloat16).Layout.Name == "int8",
                $"the user's, by a short name too: {named}");
            Check(ThrowsWith<NotRegisteredException>(() => KeyValueLayouts.Choose("float7", ElementType.Float32), "--kv 'float7'"), "an unknown format refused, naming --kv");
        });
        WithEnvironment(new Dictionary<string, string?> { [KeyValueLayouts.EnvironmentVariable] = "int8" }, () =>
            Check(KeyValueLayouts.Choose(null, ElementType.BFloat16) is { Picked: false, Source: KeyValueLayouts.EnvironmentVariable } chosen && chosen.Layout.Name == "int8"
                  && KeyValueLayouts.Choose("float32", ElementType.BFloat16).Layout.Name == "float32",
                "the variable selects, --kv over it"));
    }

    private static void LoadedElementType(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(LoadedElementType)))
        {
            return;
        }

        var cpu = Device.Cpu;
        string cache = Path.Combine(Path.GetTempPath(), "ns-dtype-" + Guid.NewGuid().ToString("N"));
        try
        {
            string folder = GgufModel.Prepare(TestData("gguf/tiny-qwen3-q8.gguf"), cache);
            WithEnvironment(new Dictionary<string, string?> { [ElementTypes.EnvironmentVariable] = null, [KeyValueLayouts.EnvironmentVariable] = null }, () =>
            {
                var hook = new TypesSeen();
                PretrainedModel selected;
                using (Telemetry.Subscribe(hook))
                {
                    selected = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu, ElementType = "bf16" });
                }

                using (selected)
                {
                    var type = selected.ElementType;
                    Check(type is { Picked: false, Source: "--dtype", IsNative: false } && type.Type == ElementType.BFloat16 && selected.DeviceElementTypes.SequenceEqual([type])
                          && type.ToString().StartsWith("element type: bfloat16 on cpu (selected by the user (--dtype)", StringComparison.Ordinal),
                        $"the user's type, said: {type}");
                    Check(selected.Network.Precision == MatMulPrecision.BFloat16 && selected.CacheChoice.Layout.Name == "bfloat16" && selected.CacheChoice.Picked,
                        $"the network multiplies in it, the cache keeps it: {selected.Network.Precision}, {selected.CacheChoice}");
                    Check(hook.Events.Any(e => e.Device == "cpu" && e.ElementType == "bfloat16" && e.Native.Contains(cpu.Backend.Capabilities.NativeReport, StringComparison.Ordinal)
                                               && e.How.StartsWith("selected by the user (--dtype)", StringComparison.Ordinal)),
                        $"telemetry: {string.Join("; ", hook.Events)}");
                }

                using (var auto = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu }))
                {
                    Check(auto.ElementType is { Picked: true } && auto.ElementType.Type == ElementType.Float32 && auto.Network.Precision is null
                          && auto.CacheChoice.Layout.Name == "float32" && auto.ElementType.How.StartsWith("picked by auto: cpu computes float32 natively", StringComparison.Ordinal),
                        $"auto on the CPU: float32, products and cache as before: {auto.ElementType}, {auto.CacheChoice}");
                }

                using (var chosen = PretrainedModel.Load(folder, new PretrainedOptions { Device = cpu, ElementType = "bfloat16", AutoWeights = true }))
                {
                    Check(chosen.Weights is { Format: not "float32" }, $"a 16-bit run's weights chosen by memory leave float32 out: {chosen.Weights}");
                }
            });
        }
        finally
        {
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, recursive: true);
            }
        }
    }

    private sealed class TypesSeen : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public List<ElementTypeChosen> Events { get; } = [];

        public void OnElementTypeChosen(in ElementTypeChosen e)
        {
            lock (this)
            {
                Events.Add(e);
            }
        }
    }
}
