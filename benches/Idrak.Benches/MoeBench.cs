// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Benches;

/// <summary>
/// Mixture-of-experts blocks at inference, the routing read back to the host (MixtureOfExperts.RoutesOnDevice off: the
/// path before plan 36's L4, and a GPU's until its routing kernels) against the routing where the tensors are (on: the
/// RouteTopK, ExpertGatedPair and ExpertMatMul operations): ms a token cold and warm (rule 76), host reads a layer (the
/// "route_on_host" operations telemetry counts) and launches a layer, one token (decoding) and a prompt. On the CPU the
/// host is the device: the device routing saves the read-back copy, the per-expert index uploads and gathers and the
/// concatenation, not a transfer, so it gains little there; a GPU with the routing kernels saves a synchronization a layer.
/// </summary>
internal static class MoeBench
{
    private static (int Dim, int Hidden, int Experts, int K, int Layers) Shape(BenchSize size) => size switch
    {
        BenchSize.Small => (256, 512, 8, 2, 4),
        BenchSize.Medium => (1024, 1408, 16, 4, 4),
        _ => (2048, 1408, 32, 4, 4),
    };

    public static void Run(Device device, Settings settings, Report report)
    {
        const string Section = "moe";
        var (dim, hidden, experts, k, layerCount) = Shape(settings.Size);
        int prompt = settings.Size == BenchSize.Small ? 64 : 256;
        var random = new Random(7);
        var layers = new List<MixtureOfExperts>(layerCount);
        try
        {
            for (int l = 0; l < layerCount; l++)
            {
                var layer = new MixtureOfExperts(dim, hidden, experts, k, device: device, random: random);
                layer.Eval();
                layers.Add(layer);
            }
        }
        catch (ResourceLimitExceededException e)
        {
            layers.ForEach(l => l.Dispose());
            Console.WriteLine($"   skipped: the {settings.Size.ToString().ToLowerInvariant()} blocks do not fit {device} ({e.Message}); IDRAK_BENCH_SIZE names a smaller one");
            return;
        }

        bool kernels = device.Backend.RoutesExperts;
        Console.WriteLine($"   {layerCount} blocks of {experts} experts ({k} a token), width {dim}, hidden {hidden}; the device {(kernels ? "has" : "has no")} routing kernels"
            + (kernels ? "" : " (both rows route on the host)"));
        bool before = MixtureOfExperts.RoutesOnDevice;
        try
        {
            foreach (int tokens in new[] { 1, prompt })
            {
                using var input = Bench.RandomTensor(random, [1, tokens, dim], device);
                void Step()
                {
                    var x = input;
                    foreach (var layer in layers)
                    {
                        x = layer.Forward(x);
                    }
                }

                foreach (bool onDevice in new[] { false, true })
                {
                    MixtureOfExperts.RoutesOnDevice = onDevice;
                    var (cold, warm) = Bench.Time(device, () =>
                    {
                        using (Autograd.NoGrad())
                        {
                            Step();
                        }
                    }, settings.Repeats, settings.Warmup);
                    Census census;
                    using (Autograd.NoGrad())
                    {
                        census = Bench.Count(Step);
                    }

                    long reads = census.ByCount.Where(o => o.Operation == "route_on_host").Sum(o => o.Count);
                    string label = $"{experts} experts k {k}, {tokens} token{(tokens == 1 ? "" : "s")}, {(onDevice ? "device" : "host")} routing";
                    double coldToken = cold / tokens, warmToken = warm / tokens;
                    double readsPerLayer = (double)reads / layerCount, launchesPerLayer = (double)census.Count / layerCount;
                    Console.WriteLine(Bench.Line(label, ("ms/token cold", Bench.F(coldToken, "0.000")), ("ms/token warm", Bench.F(warmToken, "0.000")),
                        ("host reads/layer", Bench.F(readsPerLayer, "0.##")), ("launches/layer", Bench.F(launchesPerLayer, "0.#"))));
                    report.Add(Section, label, device, "cold", coldToken, "ms/token");
                    report.Add(Section, label, device, "warm", warmToken, "ms/token");
                    report.Add(Section, label, device, "host reads a layer", readsPerLayer, "reads");
                    report.Add(Section, label, device, "launches a layer", launchesPerLayer, "launches");
                }
            }
        }
        finally
        {
            MixtureOfExperts.RoutesOnDevice = before;
            layers.ForEach(l => l.Dispose());
        }
    }
}
