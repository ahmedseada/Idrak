// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak profile MODEL</c>: where one decoding step's time goes: each top-level layer timed with the device drained
/// around it, the operations and inner layers from telemetry, and on devices that time kernels, each kernel's time.
/// </summary>
internal sealed class ProfileCommand : Command
{
    public override string Name => "profile";

    public override string Summary => "Time per layer, operation and kernel of one decoding step";

    public override string Usage =>
        "MODEL [options]\n\n" +
        "Reads a prompt into the key/value cache, then times one more decoding step: each layer of the network (the\n" +
        "device drained before and after it), the operations and inner layers it ran (telemetry, timed with the\n" +
        "device drained around each), and the kernels on devices that time them. Timings with draining are slower\n" +
        "than normal decoding; compare their shares, and use 'idrak bench' for speed.\n\n" +
        "Options:\n" +
        "  -w, --weights FORMAT  int8, int4, bf16 or a registered packed format\n" +
        "  -k, --kv FORMAT       the key/value cache format\n" +
        "      --context N       the context length; --adapter DIR: merge an adapter first\n" +
        "      --prompt-tokens N tokens in the cache before the step (default 128)\n" +
        "      --top N           rows per table (default 15)\n\n" +
        "Examples:\n" +
        "  idrak profile org/model -d vulkan:0 -w int8\n" +
        "  idrak profile mymodel --prompt-tokens 1000 -j";

    public override IReadOnlyCollection<string> ValueOptions => [.. ModelChoices.ValueOptions, "--prompt-tokens", "--top"];

    public override IReadOnlyDictionary<string, string> ShortForms => ModelChoices.ShortForms;

    public override int Run(CommandContext context)
    {
        string modelName = context.Argument(0, "MODEL");
        if (context.Positional.Count > 1)
        {
            throw new UsageException("profile takes one model.");
        }

        int top = context.IntOption("--top", 15);
        int promptTokens = context.IntOption("--prompt-tokens", 128);
        if (top <= 0 || promptTokens <= 0)
        {
            throw new UsageException("--top and --prompt-tokens need numbers above 0.");
        }

        var choice = ModelChoices.Choose(context, modelName);
        using var model = ModelChoices.Load(context, choice);
        var device = context.Device;
        promptTokens = Math.Max(1, Math.Min(promptTokens, model.MaxPositions - 5));
        var network = model.Network;
        var random = new Random(context.Seed ?? 1);
        float Token() => random.Next(Math.Min(model.Spec.Vocabulary, 1000));
        string kv = choice.Kv ?? "float32";
        using var decoding = new DecodingContext(device, 1, promptTokens + 5, KeyValueLayouts.Get(kv));
        using var noGrad = Autograd.NoGrad();
        using (var scope = new TensorScope())
        {
            network.ForwardCached(Tensor.From([.. Enumerable.Range(0, promptTokens).Select(_ => Token())], [1, promptTokens], device), decoding);
            network.ForwardCached(Tensor.From([Token()], [1, 1], device), decoding);          // a first step: kernels built, choices measured
            device.Synchronize();
        }

        // 1. Each top-level layer of one step, the device drained around it.
        var layers = new List<(string Name, string Type, double Ms)>();
        var stepWatch = Stopwatch.StartNew();
        using (var scope = new TensorScope())
        {
            var x = Tensor.From([Token()], [1, 1], device);
            decoding.BeginStep(1);
            for (int i = 0; i < network.Count; i++)
            {
                var module = network[i];
                device.Synchronize();
                var watch = Stopwatch.StartNew();
                x = module is ICachedModule cached ? cached.ForwardCached(x, decoding) : module.Forward(x);
                device.Synchronize();
                layers.Add((module.DisplayName, module.GetType().Name, watch.Elapsed.TotalMilliseconds));
            }

            decoding.EndStep(1);
        }

        double stepMs = stepWatch.Elapsed.TotalMilliseconds;

        // 2. Operations and inner layers of another step, from telemetry; 3. kernels where the device times them.
        var recorder = new Recorder();
        bool synchronize = Telemetry.SynchronizeForTiming;
        Telemetry.SynchronizeForTiming = true;
        IReadOnlyList<GpuProfileEntry> kernels;
        try
        {
            using (Telemetry.Subscribe(recorder))
            using (var scope = new TensorScope())
            {
                network.ForwardCached(Tensor.From([Token()], [1, 1], device), decoding);
                device.Synchronize();
            }

            // A device without kernel timing (the CPU, for one) ignores the profiler and reports no kernels.
            GpuProfiler.Start(device);
            using (var scope = new TensorScope())
            {
                network.ForwardCached(Tensor.From([Token()], [1, 1], device), decoding);
            }

            kernels = GpuProfiler.Stop(device);
        }
        finally
        {
            Telemetry.SynchronizeForTiming = synchronize;
        }

        bool kernelTiming = kernels.Count > 0;
        double layerTotal = layers.Sum(l => l.Ms);
        context.Write($"{choice.Model} on {device} ({device.Name}), weights {choice.Weights ?? "as stored"}, KV cache {kv}: one decoding step after {promptTokens} tokens");
        context.Write($"step {stepMs:F2} ms with each layer drained ({layerTotal:F2} ms in the layers)\n");
        context.Write("Layers of the network:");
        context.Table(["Layer", "Type", "ms", "Share"],
            layers.Select(l => (IReadOnlyList<string>)[l.Name, l.Type, l.Ms.ToString("F3"), Share(l.Ms, layerTotal)]));
        WriteTable(context, "\nInner layers (by type and depth):", recorder.Layers, top);
        WriteTable(context, "\nOperations (by name and shape):", recorder.Operations, top);
        if (kernelTiming)
        {
            context.Write($"\nKernels ({kernels.Sum(k => k.Calls)} launches):");
            context.Table(["Kernel", "Calls", "ms", "Share"],
                kernels.Take(top).Select(k => (IReadOnlyList<string>)[k.Name, k.Calls.ToString(), k.Milliseconds.ToString("F3"), Share(k.Milliseconds, kernels.Sum(e => e.Milliseconds))]));
        }
        else
        {
            context.Write($"\nKernel times: {device} does not time kernels; the operations above are its finest timing.");
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = choice.Model,
            ["device"] = device.ToString(),
            ["weights"] = choice.Weights,
            ["kv"] = kv,
            ["prompt_tokens"] = promptTokens,
            ["step_ms"] = stepMs,
            ["layers"] = new JsonArray([.. layers.Select(l => (JsonNode)new JsonObject { ["name"] = l.Name, ["type"] = l.Type, ["ms"] = l.Ms })]),
            ["inner_layers"] = Rows(recorder.Layers),
            ["operations"] = Rows(recorder.Operations),
            ["kernels"] = kernelTiming
                ? new JsonArray([.. kernels.Select(k => (JsonNode)new JsonObject { ["name"] = k.Name, ["calls"] = k.Calls, ["ms"] = k.Milliseconds })])
                : null,
        });
        return ExitCodes.Ok;
    }

    private static void WriteTable(CommandContext context, string title, Dictionary<string, (int Count, double Ms)> table, int top)
    {
        double total = table.Values.Sum(v => v.Ms);
        context.Write($"{title} {total:F2} ms in {table.Values.Sum(v => v.Count)} calls");
        context.Table(["Name", "Calls", "ms", "Share"], table.OrderByDescending(p => p.Value.Ms).Take(top)
            .Select(p => (IReadOnlyList<string>)[p.Key, p.Value.Count.ToString(), p.Value.Ms.ToString("F3"), Share(p.Value.Ms, total)]));
    }

    private static JsonArray Rows(Dictionary<string, (int Count, double Ms)> table) =>
        [.. table.OrderByDescending(p => p.Value.Ms).Select(p => (JsonNode)new JsonObject { ["name"] = p.Key, ["calls"] = p.Value.Count, ["ms"] = p.Value.Ms })];

    private static string Share(double part, double total) => total > 0 ? (part / total).ToString("P1") : "";

    // Sums the time of each operation (by name and shape) and each inner layer (by type and depth), as the Chat sample's profile does.
    private sealed class Recorder : ITelemetryHook
    {
        public Dictionary<string, (int Count, double Ms)> Operations { get; } = [];

        public Dictionary<string, (int Count, double Ms)> Layers { get; } = [];

        public TelemetryLevel Levels => TelemetryLevel.Operations | TelemetryLevel.Layers;

        public void OnOperation(in OperationCompleted e) => Add(Operations, $"{e.Operation} [{string.Join("x", e.Shape)}]", e.Duration);

        public void OnLayerForward(in LayerForward e) => Add(Layers, $"{e.LayerType} (depth {e.Depth})", e.Duration);

        private static void Add(Dictionary<string, (int Count, double Ms)> table, string key, TimeSpan duration)
        {
            var (count, ms) = table.GetValueOrDefault(key);
            table[key] = (count + 1, ms + duration.TotalMilliseconds);
        }
    }
}
