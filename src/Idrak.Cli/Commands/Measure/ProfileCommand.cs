// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Training;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak profile MODEL</c>: where one decoding step's time goes: each top-level layer timed with the device drained
/// around it, the operations and inner layers from telemetry, and on devices that time kernels, each kernel's time. With
/// <c>--cluster FILE</c> (plan 23, phase C8) the model runs as a pipeline over a simulated cluster of the topology, one
/// stage a rank, and a token's hops between the stages are the rows: each hop's bytes, wall time here and time on the
/// virtual clock.
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
        "      --head-weights F  the output head's format beside -w (bf16: the checkpoint's own scores, less memory, slower)\n" +
        "  -k, --kv FORMAT       the key/value cache format\n" +
        "      --context N       the context length; --adapter DIR: merge an adapter first\n" +
        "      --prompt-tokens N tokens in the cache before the step (default 128)\n" +
        "      --top N           rows per table (default 15)\n" +
        "      --cluster FILE    run the model as a pipeline over a simulated cluster of the topology (a stage a rank, or\n" +
        "                        --stages N): a token's hops between the stages as rows, their bytes, wall and simulated time\n" +
        "      --stages N        with --cluster: the pipeline's stages (one per rank)\n" +
        "      --steps N         with --cluster: the decoding steps timed (default 4)\n\n" +
        "Examples:\n" +
        "  idrak profile org/model -d vulkan:0 -w int8\n" +
        "  idrak profile mymodel --prompt-tokens 1000 -j\n" +
        "  idrak profile my-model --cluster two.json --steps 8";

    public override IReadOnlyCollection<string> ValueOptions => [.. ModelChoices.ValueOptions, "--prompt-tokens", "--top", "--cluster", "--stages", "--steps"];

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
        if (context.Option("--cluster") is { } topologyFile)
        {
            return Pipeline(context, choice, topologyFile, promptTokens);
        }

        using var model = ModelChoices.Load(context, choice);
        var device = context.Device;
        promptTokens = Math.Max(1, Math.Min(promptTokens, model.MaxPositions - 5));
        var network = model.Network;
        var random = new Random(context.Seed ?? 1);
        float Token() => random.Next(Math.Min(model.Spec.Vocabulary, 1000));
        string kv = choice.Kv ?? "float32";
        TensorOffloading.BeforeInference(device, network);                               // as generation does: room when offloaded
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
            using var step = decoding.OpenStep(1);
            for (int i = 0; i < network.Count; i++)
            {
                var module = network[i];
                device.Synchronize();
                var watch = Stopwatch.StartNew();
                x = module is ICachedModule cached ? cached.ForwardCached(x, decoding) : module.Forward(x);
                device.Synchronize();
                layers.Add((module.DisplayName, module.GetType().Name, watch.Elapsed.TotalMilliseconds));
            }

            step.Complete();
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
        context.Write(string.Create(CultureInfo.InvariantCulture, $"step {stepMs:F2} ms with each layer drained ({layerTotal:F2} ms in the layers)"));
        var memory = ComputeResources.GetMemoryUsage(device);
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{device}: {memory.InUse / 1e9:F2} GB in use{(memory.Offloaded > 0 ? $", {memory.Offloaded / 1e9:F2} GB of tensors in system memory (offloaded)" : "")}\n"));
        context.Write("Layers of the network:");
        context.Table(["Layer", "Type", "ms", "Share"],
            layers.Select(l => (IReadOnlyList<string>)[l.Name, l.Type, l.Ms.ToString("F3", CultureInfo.InvariantCulture), Share(l.Ms, layerTotal)]));
        WriteTable(context, "\nInner layers (by type and depth):", recorder.Layers, top);
        WriteTable(context, "\nOperations (by name and shape):", recorder.Operations, top);
        if (kernelTiming)
        {
            context.Write($"\nKernels ({kernels.Sum(k => k.Calls)} launches):");
            context.Table(["Kernel", "Calls", "ms", "Share"],
                kernels.Take(top).Select(k => (IReadOnlyList<string>)[k.Name, k.Calls.ToString(CultureInfo.InvariantCulture), k.Milliseconds.ToString("F3", CultureInfo.InvariantCulture), Share(k.Milliseconds, kernels.Sum(e => e.Milliseconds))]));
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

    // --cluster: the model as a pipeline over a simulated cluster, one decoding step after another on rank 0 (the first
    // stage), a token's hops between the stages recorded from the transfers each stage takes.
    private static int Pipeline(CommandContext context, ModelChoices.ModelChoice choice, string topologyFile, int promptTokens)
    {
        int steps = context.IntOption("--steps", 4);
        if (steps <= 0)
        {
            throw new UsageException("--steps needs a number above 0.");
        }

        ClusterTopology topology;
        ClusterOptions options;
        try
        {
            topology = ClusterTopology.Load(topologyFile);
            options = ClusterOptions.FromEnvironment() with { Topology = topologyFile, Placement = "pipeline" };
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or FileNotFoundException or DirectoryNotFoundException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        if (context.Option("--stages") is not null)
        {
            options = options with { Stages = context.IntOption("--stages", 1) };
        }

        int stages = options.Stages ?? topology.WorldSize;
        if (stages < 2 || topology.WorldSize % stages != 0)
        {
            throw new UsageException($"A pipeline over {topologyFile} needs 2 or more stages dividing its {topology.WorldSize} rank(s) (--stages, IDRAK_CLUSTER_STAGES).");
        }

        string kv = choice.Kv ?? "float32";
        var hops = new Hops();
        double stepMs = 0, simulatedMs = 0;
        string[] modules = [];
        var result = InProcessCluster.Run(topology, options, rank =>
        {
            using var model = ModelChoices.Load(context, choice, Device.Cpu);   // each rank keeps its own stage of it
            using var pipe = PipelineStage.Place(model.Network, rank.Collectives, options);
            if (!pipe.IsFirst)
            {
                pipe.Serve();
                return;
            }

            try
            {
                int tokens = Math.Max(1, Math.Min(promptTokens, model.MaxPositions - steps - 3));
                var random = new Random(context.Seed ?? 1);
                float Token() => random.Next(Math.Min(model.Spec.Vocabulary, 1000));
                using var decoding = new DecodingContext(rank.Device, 1, tokens + steps + 2, KeyValueLayouts.Get(kv));
                using var noGrad = Autograd.NoGrad();
                using (var scope = new TensorScope())
                {
                    pipe.Network.ForwardCached(Tensor.From([.. Enumerable.Range(0, tokens).Select(_ => Token())], [1, tokens], rank.Device), decoding);
                    pipe.Network.ForwardCached(Tensor.From([Token()], [1, 1], rank.Device), decoding);   // a first step: kernels built
                }

                var clockBefore = rank.Transport.SimulatedTime;
                var watch = System.Diagnostics.Stopwatch.StartNew();
                using (Telemetry.Subscribe(hops))
                {
                    for (int i = 0; i < steps; i++)
                    {
                        using var scope = new TensorScope();
                        pipe.Network.ForwardCached(Tensor.From([Token()], [1, 1], rank.Device), decoding);
                    }
                }

                stepMs = watch.Elapsed.TotalMilliseconds / steps;
                simulatedMs = (rank.Transport.SimulatedTime - clockBefore).TotalMilliseconds / steps;
                modules = [.. pipe.Modules.Select(m => m.Name ?? m.GetType().Name)];
            }
            finally
            {
                pipe.StopServing();
            }
        });
        if (!result.Succeeded)
        {
            foreach (var (error, rank) in result.Errors.Select((e, r) => (e, r)).Where(p => p.e is not null))
            {
                context.Error($"rank {rank}: {error!.Message}");
            }

            return ExitCodes.Failed;
        }

        var inv = CultureInfo.InvariantCulture;
        int perStage = topology.WorldSize / stages;
        var rows = hops.Rows(steps);
        context.Write($"{choice.Model} as a pipeline of {stages} stage(s) over {topologyFile} ({topology.WorldSize} rank(s), simulated): {steps} decoding step(s) after {promptTokens} prompt tokens");
        context.Write(string.Create(inv, $"a token: {stepMs:F2} ms here, {simulatedMs:F3} ms of it on the virtual clock (the links the file declares); stage 0 holds {string.Join(", ", modules)}"));
        context.Table(["hop", "stages", "messages", "bytes", "wall ms", "simulated ms"], [.. rows.Select(r => (IReadOnlyList<string>)
        [
            $"rank {r.From} -> rank {r.To}", $"{r.From / perStage} -> {r.To / perStage}", r.Messages.ToString("F1", inv), r.Bytes.ToString("F0", inv),
            r.Wall.ToString("F3", inv), r.Simulated.ToString("F3", inv),
        ])]);
        context.WriteJson(new JsonObject
        {
            ["model"] = choice.Model,
            ["topology"] = Path.GetFullPath(topologyFile),
            ["stages"] = stages,
            ["steps"] = steps,
            ["token_ms"] = stepMs,
            ["token_simulated_ms"] = simulatedMs,
            ["hops"] = new JsonArray([.. rows.Select(r => (JsonNode)new JsonObject
            {
                ["from"] = r.From, ["to"] = r.To, ["from_stage"] = r.From / perStage, ["to_stage"] = r.To / perStage,
                ["messages"] = r.Messages, ["bytes"] = r.Bytes, ["wall_ms"] = r.Wall, ["simulated_ms"] = r.Simulated,
            })]),
        });
        return ExitCodes.Ok;
    }

    // A token's hops: the transfers between ranks (per token: messages, bytes, wall and simulated time), in the order a
    // token travels (rank 0's stage out, through each stage, back to the first).
    private sealed class Hops : ITelemetryHook
    {
        private readonly Lock _gate = new();
        private readonly Dictionary<(int From, int To), (long Messages, long Bytes, double Wall, double Simulated)> _pairs = [];

        public TelemetryLevel Levels => TelemetryLevel.Network;

        public void OnTransfer(in TransferCompleted e)
        {
            lock (_gate)
            {
                var (messages, bytes, wall, simulated) = _pairs.GetValueOrDefault((e.From, e.To));
                _pairs[(e.From, e.To)] = (messages + 1, bytes + e.Bytes, wall + e.Wall.TotalMilliseconds, simulated + e.Simulated.TotalMilliseconds);
            }
        }

        public List<(int From, int To, double Messages, double Bytes, double Wall, double Simulated)> Rows(int steps)
        {
            lock (_gate)
            {
                return [.. _pairs.OrderBy(p => p.Key.To <= p.Key.From ? 1 : 0).ThenBy(p => p.Key.From).Select(p => (p.Key.From, p.Key.To, p.Value.Messages / (double)steps,
                    p.Value.Bytes / (double)steps, p.Value.Wall / steps, p.Value.Simulated / steps))];
            }
        }
    }

    private static void WriteTable(CommandContext context, string title, Dictionary<string, (int Count, double Ms)> table, int top)
    {
        double total = table.Values.Sum(v => v.Ms);
        context.Write(string.Create(CultureInfo.InvariantCulture, $"{title} {total:F2} ms in {table.Values.Sum(v => v.Count)} calls"));
        context.Table(["Name", "Calls", "ms", "Share"], table.OrderByDescending(p => p.Value.Ms).Take(top)
            .Select(p => (IReadOnlyList<string>)[p.Key, p.Value.Count.ToString(CultureInfo.InvariantCulture), p.Value.Ms.ToString("F3", CultureInfo.InvariantCulture), Share(p.Value.Ms, total)]));
    }

    private static JsonArray Rows(Dictionary<string, (int Count, double Ms)> table) =>
        [.. table.OrderByDescending(p => p.Value.Ms).Select(p => (JsonNode)new JsonObject { ["name"] = p.Key, ["calls"] = p.Value.Count, ["ms"] = p.Value.Ms })];

    private static string Share(double part, double total) => total > 0 ? (part / total).ToString("P1", CultureInfo.InvariantCulture) : "";

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
