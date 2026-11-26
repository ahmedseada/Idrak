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

    public override string Summary => "Time per layer, operation and kernel of one decoding step or one prompt pass";

    public override string Usage =>
        "MODEL [options]\n\n" +
        "Reads a prompt into the key/value cache, then times a decoding step (each layer's median over --steps steps), or,\n" +
        "with --pass prompt, one whole prompt pass of --prompt-tokens into a fresh cache, each after a garbage collection:\n" +
        "each layer of the network (the device drained before and after it; with the memory it allocated and the garbage\n" +
        "collections it caused; over the whole pass, how many cores were busy: the process's CPU time over the wall time),\n" +
        "the operations and inner layers it ran (telemetry, timed with the device drained around each), and the kernels on\n" +
        "devices that time them. Every pass runs as generation runs the network: in its precision, and a prompt's layers\n" +
        "past the last cached one over the last position only. Timings with draining are slower than normal decoding;\n" +
        "compare their shares, and use 'idrak bench' for speed.\n\n" +
        "Options:\n" +
        "  -w, --weights FORMAT  int8, int4, bf16 or a registered packed format\n" +
        "      --head-weights F  the output head's format beside -w (bf16: the checkpoint's own scores, less memory, slower)\n" +
        "  -k, --kv FORMAT       the key/value cache format\n" +
        "      --context N       the context length; --adapter DIR: merge an adapter first\n" +
        "      --prompt-tokens N tokens in the cache before the step, or the prompt pass's length (default 128)\n" +
        "      --pass KIND       step (one decoding step, the default) or prompt (one prompt pass)\n" +
        "      --top N           rows per table (default 15)\n" +
        "      --cluster FILE    run the model as a pipeline over a simulated cluster of the topology (a stage a rank, or\n" +
        "                        --stages N): a token's hops between the stages as rows, their bytes, wall and simulated time\n" +
        "      --stages N        with --cluster: the pipeline's stages (one per rank)\n" +
        "      --steps N         the decoding steps timed: a step profile's median (default 5), or with --cluster (default 4)\n\n" +
        "Examples:\n" +
        "  idrak profile org/model -d vulkan:0 -w int8\n" +
        "  idrak profile mymodel --prompt-tokens 1000 -j\n" +
        "  idrak profile org/model -d cpu -w bf16 --pass prompt --prompt-tokens 512\n" +
        "  idrak profile my-model --cluster two.json --steps 8";

    public override IReadOnlyCollection<string> ValueOptions => [.. ModelChoices.ValueOptions, "--prompt-tokens", "--top", "--cluster", "--stages", "--steps", "--pass"];

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

        bool promptPass = (context.Option("--pass") ?? "step").ToLowerInvariant() switch
        {
            "step" => false,
            "prompt" => true,
            var other => throw new UsageException($"--pass is '{other}': step or prompt."),
        };
        var choice = ModelChoices.Choose(context, modelName);
        if (context.Option("--cluster") is { } topologyFile)
        {
            return Pipeline(context, choice, topologyFile, promptTokens);
        }

        int stepCount = context.IntOption("--steps", 5);
        if (stepCount <= 0)
        {
            throw new UsageException("--steps needs a number above 0.");
        }

        using var model = ModelChoices.Load(context, choice);
        var device = context.Device;
        int timedPasses = promptPass ? 1 : stepCount;
        promptTokens = Math.Max(1, Math.Min(promptTokens, model.MaxPositions - timedPasses - 4));
        var network = model.Network;
        var random = new Random(context.Seed ?? 1);
        float Token() => random.Next(Math.Min(model.Spec.Vocabulary, 1000));
        var cache = ModelChoices.CacheLayout(choice);                    // --kv, else the cache in the run's type
        string kv = cache.Name;
        TensorOffloading.BeforeInference(device, network);                               // as generation does: room when offloaded
        using var noGrad = Autograd.NoGrad();
        Tensor Prompt() => Tensor.From([.. Enumerable.Range(0, promptTokens).Select(_ => Token())], [1, promptTokens], device);

        // Every pass below is the network's own cached pass (Sequential.ForwardCached), as generation runs it: in the network's
        // precision, each layer's intermediate results freed once the next layer has its input, and with only the last
        // position past the last cached layer (TextGenerator's LastPositionOnly: the head reads one row of a prompt, not
        // every row). The pass timed: one decoding step after the prompt (the median of --steps of them), or (--pass prompt)
        // the prompt itself into a fresh cache. The cache holds the prompt, the first step, the timed steps and the two
        // recorded after.
        int capacity = promptTokens + timedPasses + 4;
        var decoding = new DecodingContext(device, 1, capacity, cache) { LastPositionOnly = true };
        try
        {
            using (var scope = new TensorScope())
            {
                network.ForwardCached(Prompt(), decoding);
                network.ForwardCached(Tensor.From([Token()], [1, 1], device), decoding);      // a first step: kernels built, choices measured
                device.Synchronize();
            }

            // The context a timed pass runs in: the warmed one for a step, a fresh one for each prompt pass.
            DecodingContext PassContext()
            {
                if (promptPass)
                {
                    decoding.Dispose();
                    decoding = new DecodingContext(device, 1, capacity, cache) { LastPositionOnly = true };
                }

                return decoding;
            }

            Tensor PassInput() => promptPass ? Prompt() : Tensor.From([Token()], [1, 1], device);

            // 1. Each top-level layer of the timed pass, the device drained at each layer's start and at the pass's end (a
            // hook the network's pass calls before each layer): its time, the bytes it allocated and the collections it
            // caused; over the pass, how many cores were busy (the process's CPU time over the wall time: the system counts
            // CPU time in ticks too coarse for one layer). Each pass after a full collection, so the garbage of the load and
            // the warm-up is not collected inside it; a step profile reports each layer's median over its steps (and the
            // collections of all of them), so one pause does not stand for the step.
            static int Collections() => GC.CollectionCount(0) + GC.CollectionCount(1) + GC.CollectionCount(2);
            var passLayers = new List<(double Ms, double AllocatedMiB, int Collections)[]>();
            var passMs = new double[timedPasses];
            var process = Process.GetCurrentProcess();
            double cpuMs = 0, wallMs = 0;
            string products = "";
            for (int run = 0; run < timedPasses; run++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                var timedLayers = new (double Ms, double AllocatedMiB, int Collections)[network.Count];
                int current = -1, collections = 0;
                long allocated = 0;
                var watch = new Stopwatch();
                void LayerEnds()
                {
                    device.Synchronize();
                    if (current >= 0)
                    {
                        timedLayers[current] = (watch.Elapsed.TotalMilliseconds, (GC.GetTotalAllocatedBytes() - allocated) / 1048576.0, Collections() - collections);
                    }
                }

                process.Refresh();
                var passCpu = process.TotalProcessorTime;
                var stepWatch = Stopwatch.StartNew();
                using (var scope = new TensorScope())
                {
                    network.ForwardCached(PassInput(), PassContext(), network.Count, (i, x) =>
                    {
                        LayerEnds();
                        products = MixedPrecision.Current.ToString();             // what the network's pass multiplies in
                        (current, allocated, collections) = (i, GC.GetTotalAllocatedBytes(), Collections());
                        watch.Restart();
                        return x;
                    });
                    LayerEnds();
                }

                passMs[run] = stepWatch.Elapsed.TotalMilliseconds;
                process.Refresh();
                cpuMs += (process.TotalProcessorTime - passCpu).TotalMilliseconds;
                wallMs += passMs[run];
                passLayers.Add(timedLayers);
            }

            var layers = Enumerable.Range(0, network.Count).Select(i => (Name: network[i].DisplayName, Type: network[i].GetType().Name,
                Ms: Median(passLayers.Select(p => p[i].Ms)), AllocatedMiB: Median(passLayers.Select(p => p[i].AllocatedMiB)),
                Collections: passLayers.Sum(p => p[i].Collections))).ToList();
            double stepMs = Median(passMs);
            double cores = wallMs > 0 ? cpuMs / wallMs : 0;

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
                network.ForwardCached(PassInput(), PassContext());
                device.Synchronize();
            }

            // A device without kernel timing (the CPU, for one) ignores the profiler and reports no kernels.
            GpuProfiler.Start(device);
            using (var scope = new TensorScope())
            {
                network.ForwardCached(PassInput(), PassContext());
            }

            kernels = GpuProfiler.Stop(device);
        }
        finally
        {
            Telemetry.SynchronizeForTiming = synchronize;
        }

        bool kernelTiming = kernels.Count > 0;
        double layerTotal = layers.Sum(l => l.Ms);
        string pass = promptPass ? $"one prompt pass of {promptTokens} tokens" : $"one decoding step after {promptTokens} tokens";
        context.Write($"{choice.Model} on {device} ({device.Name}), KV cache {kv}: {pass}");
        string timedText = promptPass ? "one pass" : $"the median of {timedPasses} steps";
        context.Write(string.Create(CultureInfo.InvariantCulture, $"{(promptPass ? "pass" : "step")} {stepMs:F2} ms ({timedText}, each after a collection) with each layer drained "
            + $"({layerTotal:F2} ms in the layers{(promptPass ? "" : ", each layer's median")}), {layers.Sum(l => l.AllocatedMiB):F1} MiB allocated{(promptPass ? "" : " a step")}, "
            + $"{layers.Sum(l => l.Collections)} collections{(promptPass ? "" : $" in the {timedPasses} steps")}, {cores:F1} of {Environment.ProcessorCount} cores busy on average"));
        context.Write($"products in {products}, as the network runs them{(promptPass ? "; the layers past the last cached one over the last position only, as generation runs a prompt" : "")}");
        var memory = ComputeResources.GetMemoryUsage(device);
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{device}: {memory.InUse / 1e9:F2} GB in use{(memory.Offloaded > 0 ? $", {memory.Offloaded / 1e9:F2} GB of tensors in system memory (offloaded)" : "")}\n"));
        context.Write("Layers of the network:");
        context.Table(["Layer", "Type", "ms", "Share", "MiB allocated", "GCs"],
            layers.Select(l => (IReadOnlyList<string>)[l.Name, l.Type, l.Ms.ToString("F3", CultureInfo.InvariantCulture), Share(l.Ms, layerTotal),
                l.AllocatedMiB.ToString("F2", CultureInfo.InvariantCulture), l.Collections.ToString(CultureInfo.InvariantCulture)]));
        WriteTable(context, "\nInner layers (by type and depth):", recorder.Layers, top);
        WriteTable(context, "\nOperations (by name and shape):", recorder.Operations, top);
        if (kernelTiming)
        {
            context.Write($"\nKernels ({kernels.Sum(k => k.Calls)} launches):");
            context.Table(["Kernel", "Calls", "ms", "Share"],
                kernels.Take(top).Select(k => (IReadOnlyList<string>)[k.Name, k.Calls.ToString(CultureInfo.InvariantCulture), k.Milliseconds.ToString("F3", CultureInfo.InvariantCulture), Share(k.Milliseconds, kernels.Sum(e => e.Milliseconds))]));
        }
        else if (device.Backend is IKernelProfiler)
        {
            string why = (device.Backend as IKernelAvailability)?.KernelsUnavailable() is { } off ? $" (its kernels are unavailable: {off})" : "";
            context.Write($"\nKernel times: no kernel of {device}'s own ran in the step{why}; the operations above are its finest timing.");
        }
        else
        {
            context.Write($"\nKernel times: {device} does not time kernels; the operations above are its finest timing.");
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = choice.Model,
            ["device"] = device.ToString(),
            ["kv"] = kv,
            ["prompt_tokens"] = promptTokens,
            ["pass"] = promptPass ? "prompt" : "step",
            ["steps"] = timedPasses,
            ["precision"] = products,
            ["last_position_only"] = true,
            ["step_ms"] = stepMs,
            ["cores_busy"] = cores,
            ["layers"] = new JsonArray([.. layers.Select(l => (JsonNode)new JsonObject
            {
                ["name"] = l.Name, ["type"] = l.Type, ["ms"] = l.Ms, ["allocated_mib"] = l.AllocatedMiB, ["collections"] = l.Collections,
            })]),
            ["inner_layers"] = Rows(recorder.Layers),
            ["operations"] = Rows(recorder.Operations),
            ["kernels"] = kernelTiming
                ? new JsonArray([.. kernels.Select(k => (JsonNode)new JsonObject { ["name"] = k.Name, ["calls"] = k.Calls, ["ms"] = k.Milliseconds })])
                : null,
        });
        return ExitCodes.Ok;
        }
        finally
        {
            decoding.Dispose();
        }
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
                using var decoding = new DecodingContext(rank.Device, 1, tokens + steps + 2, ModelChoices.CacheLayout(choice));   // --kv, else the cache in the run's type
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

    // The median of the values (the mean of the middle two for an even count; 0 for none).
    private static double Median(IEnumerable<double> values)
    {
        var sorted = values.Order().ToArray();
        return sorted.Length == 0 ? 0 : sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
    }

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
