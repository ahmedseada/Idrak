// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak trace [options] [--] COMMAND ...</c>: runs another idrak command in this process with Idrak's telemetry
/// (Idrak.Diagnostics) subscribed: printed live, or written to a JSON Lines file with <c>--out</c>.
/// </summary>
internal sealed class TraceCommand : Command
{
    /// <summary>The telemetry kinds by the names <c>--levels</c> takes.</summary>
    public static readonly IReadOnlyDictionary<string, TelemetryLevel> Levels = new Dictionary<string, TelemetryLevel>(StringComparer.OrdinalIgnoreCase)
    {
        ["training"] = TelemetryLevel.Training,
        ["batches"] = TelemetryLevel.Batches,
        ["gradients"] = TelemetryLevel.Gradients,
        ["layers"] = TelemetryLevel.Layers,
        ["operations"] = TelemetryLevel.Operations,
        ["inference"] = TelemetryLevel.Inference,
        ["tools"] = TelemetryLevel.Tools,
        ["engine"] = TelemetryLevel.Engine,
        ["devices"] = TelemetryLevel.Devices,
        ["overrides"] = TelemetryLevel.Overrides,
        ["all"] = TelemetryLevel.All,
    };

    /// <summary>What is traced without <c>--levels</c>: everything but the per-operation events and gradient norms.</summary>
    public const TelemetryLevel DefaultLevels = TelemetryLevel.Training | TelemetryLevel.Batches | TelemetryLevel.Layers
        | TelemetryLevel.Inference | TelemetryLevel.Tools | TelemetryLevel.Engine | TelemetryLevel.Devices | TelemetryLevel.Overrides;

    public override string Name => "trace";

    public override string Summary => "Runs an idrak command with telemetry printed live or written to JSON Lines";

    public override string Usage =>
        "[-o FILE.jsonl] [--levels LIST] [--sync] [--] COMMAND [arguments] [options]\n\n" +
        "Runs COMMAND (any idrak command) with Idrak's telemetry subscribed: training epochs and batches, every layer's\n" +
        "forward pass, inference calls, tool calls, engine events and an app's overrides falling back, printed as they\n" +
        "happen (to the error output with --json), or written to FILE.jsonl with --out. Put '--' before COMMAND when it has\n" +
        "options of its own; the common options given before it (--device, --json, --quiet, --verbose, --plugin, --config,\n" +
        "--cache) are passed on to it.\n\n" +
        "Options:\n" +
        "  -o, --out FILE      write the events as JSON Lines to FILE instead of printing them\n" +
        "      --levels LIST   what to trace, separated by commas: training, batches, gradients, layers, operations\n" +
        "                      (every tensor operation and kernel, very verbose), inference, tools, engine,\n" +
        "                      devices (GPU failures and operations retried on the CPU),\n" +
        "                      overrides (an app's implementation falling back or compared in shadow), all\n" +
        "                      (default: all but operations and gradients)\n" +
        "      --sync          synchronize the device before each timing (true GPU times, slower)\n\n" +
        "Examples:\n" +
        "  idrak trace -- onnx check model.onnx -d vulkan:0\n" +
        "  idrak trace -o train.jsonl --levels training,batches -- train network.json --data houses.csv\n" +
        "  idrak trace --levels operations --sync -- bench --kernels matmul";

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--levels"];

    public override IReadOnlyCollection<string> Flags => ["--sync"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Missing the COMMAND to trace (e.g. idrak trace -- onnx check model.onnx).");
        }

        if (CommandLine.Find(CommandTable.All, context.Positional, out _) is not { } inner)
        {
            throw new UsageException($"Unknown command '{context.Positional[0]}'. Run 'idrak help' for the commands.");
        }

        if (inner is TraceCommand)
        {
            throw new UsageException("idrak trace cannot trace itself.");
        }

        var levels = ParseLevels(context.Option("--levels"));
        string? path = context.Option("--out") is { } given ? Path.GetFullPath(given) : null;

        // The common options given to trace go to the traced command.
        var args = new List<string>(context.Positional);
        foreach (string option in CommandContext.CommonValueOptions)
        {
            args.AddRange(context.Options(option).SelectMany(v => new[] { option, v }));
        }

        foreach (string flag in new[] { "--json", "--quiet", "--verbose" }.Where(context.Flag))
        {
            args.Add(flag);
        }

        var counter = new Counter(levels);
        var configuration = Telemetry.Configure().Hook(counter).SynchronizeForTiming(context.Flag("--sync"));
        if (path is not null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            configuration.JsonLines(path, levels);
        }
        else if (!context.Quiet)
        {
            configuration.Console(levels, output: context.Json ? context.ErrorOutput : context.Output);
        }

        // With --json the traced command's document is captured and wrapped in trace's own, so the output stays one document.
        var output = context.Json ? new StringWriter() : context.Output;
        int exit;
        var session = configuration.Start();
        try
        {
            exit = CommandLine.Run(args, output, context.ErrorOutput);
        }
        finally
        {
            session.Dispose();
        }

        string counts = string.Join(", ", counter.Counts.Where(c => c.Value > 0).Select(c => string.Create(CultureInfo.InvariantCulture, $"{c.Value:N0} {c.Key}")));
        context.Write($"trace: {inner.Name} exited with {exit}; {(counts.Length == 0 ? "no events" : counts)}{(path is null ? "" : $"; events in {path}")}");
        if (context.Json)
        {
            string text = output.ToString()!;
            JsonNode? result;
            try
            {
                result = text.Length == 0 ? null : JsonNode.Parse(text);
            }
            catch (System.Text.Json.JsonException)
            {
                result = text;
            }

            context.WriteJson(new JsonObject
            {
                ["command"] = string.Join(' ', context.Positional),
                ["exitCode"] = exit,
                ["levels"] = levels.ToString(),
                ["events"] = new JsonObject([.. counter.Counts.Select(c => KeyValuePair.Create(c.Key, (JsonNode?)c.Value))]),
                ["out"] = path,
                ["result"] = result,
            });
        }

        return exit;
    }

    internal static TelemetryLevel ParseLevels(string? text)
    {
        if (text is null)
        {
            return DefaultLevels;
        }

        var levels = TelemetryLevel.None;
        foreach (string name in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            levels |= Levels.TryGetValue(name, out var level) ? level
                : throw new UsageException($"Unknown telemetry level '{name}'; choose from {string.Join(", ", Levels.Keys)}.");
        }

        return levels == TelemetryLevel.None ? throw new UsageException("--levels needs at least one level.") : levels;
    }

    // Counts the events of each kind (for the summary line and the JSON): a slot per kind, counted without a lock or a
    // lookup, as every operation of the traced command reports here.
    private sealed class Counter(TelemetryLevel levels) : ITelemetryHook
    {
        // The kinds in the order the summary and the JSON list them.
        private static readonly string[] Kinds = ["training", "epochs", "batches", "layers", "operations", "inference", "tools", "engine", "devices"];

        private readonly int[] _counts = new int[Kinds.Length];

        public TelemetryLevel Levels { get; } = levels;

        public IReadOnlyDictionary<string, int> Counts
        {
            get
            {
                var counts = new Dictionary<string, int>(Kinds.Length);
                for (int i = 0; i < Kinds.Length; i++)
                {
                    counts[Kinds[i]] = Volatile.Read(ref _counts[i]);
                }

                return counts;
            }
        }

        public void OnTrainingStarted(in TrainingStarted e) => Interlocked.Increment(ref _counts[0]);

        public void OnEpochCompleted(in EpochCompleted e) => Interlocked.Increment(ref _counts[1]);

        public void OnBatchCompleted(in BatchCompleted e) => Interlocked.Increment(ref _counts[2]);

        public void OnLayerForward(in LayerForward e) => Interlocked.Increment(ref _counts[3]);

        public void OnOperation(in OperationCompleted e) => Interlocked.Increment(ref _counts[4]);

        public void OnInference(in InferenceCompleted e) => Interlocked.Increment(ref _counts[5]);

        public void OnToolCall(in ToolCallCompleted e) => Interlocked.Increment(ref _counts[6]);

        public void OnEngine(in EngineEvent e) => Interlocked.Increment(ref _counts[7]);

        public void OnDeviceFailed(in DeviceFailed e) => Interlocked.Increment(ref _counts[8]);
    }
}
