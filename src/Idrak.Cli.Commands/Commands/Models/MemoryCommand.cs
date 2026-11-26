// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Layers;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak memory MODEL</c>: the memory a model needs in each KV cache format at a context length, against each
/// device's memory. KV sizes come from each KV layout's row width.
/// </summary>
internal sealed class MemoryCommand : Command
{
    public override string Name => "memory";

    public override string Summary => "Memory per KV format at a context length, against each device's memory";

    public override string Usage => """
        MODEL [-k FORMAT...] [--context N] [--batch N] [--memory SIZE]

        Arguments:
          MODEL  a cached Hugging Face id, a folder, a .gguf file or an alias

        Options:
          -k, --kv F       KV cache formats (repeatable or comma-separated)
              --context N  the context length (default the model's longest)
              --batch N    sequences decoded together (default 1)
              --memory SIZE also compare with this much memory (e.g. 8G, 512M)

        Activations and the runtime add a little more (a few hundred MB), so a total close to a device's memory may
        still not fit.

        Examples:
          idrak memory Qwen/Qwen3-0.6B
          idrak memory Qwen/Qwen3-8B --context 32768 -d vulkan:0

        Gap: the library reports a GPU's total memory only when a limit is set (ComputeResources.GpuMemoryLimit); for
        other GPUs pass --memory with the card's memory.
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. Shared.ModelChoices.ValueOptions.Where(o => o != "--adapter"), "--batch", "--memory"];

    public override IReadOnlyDictionary<string, string> ShortForms => Shared.ModelChoices.ShortForms;

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL");
        var local = ModelCache.Locate(context, name);
        var facts = ModelFacts.Read(local);
        var spec = facts.Spec ?? throw new InvalidOperationException($"{name}: {facts.Problem}");
        var alias = Shared.ModelChoices.Choose(context, name);
        int contextLength = context.Option("--context") is null ? spec.MaxPositions : context.IntOption("--context", 0);
        int batch = context.IntOption("--batch", 1);
        if (contextLength <= 0 || batch <= 0)
        {
            throw new UsageException("--context and --batch need positive numbers.");
        }

        var kvFormats = List(context.Options("--kv"), alias.Kv, [.. KeyValueLayouts.Names]);

        var devices = Devices(context);
        long? given = context.Option("--memory") is { } size ? ParseSize(size) : null;
        var rows = new List<(string Kv, long KvBytes)>();
        foreach (string kv in kvFormats)
        {
            var layout = KeyValueLayouts.Get(kv);
            long perPosition = (long)spec.Layers * spec.KvHeads * 2 * (layout.RowWidth(spec.HeadDim) * 4L + (layout is IScaledLayout ? 4 : 0));
            rows.Add((kv, perPosition * contextLength * batch));
        }

        context.Write(string.Create(CultureInfo.InvariantCulture, $"{name}: {Units.Count(facts.Parameters)} parameters, context {contextLength:N0}")
                      + (batch > 1 ? $" × {batch} sequences" : ""));
        var memoryColumns = devices.Select(d => (Label: d.Name, d.Bytes)).ToList();
        if (given is { } g)
        {
            memoryColumns.Add(($"{Units.Bytes(g)}", g));
        }

        string Fits(long total, long? bytes) => bytes is not { } m ? "?" : total <= m ? $"fits ({Percent(total, m)})" : $"no ({Percent(total, m)})";
        context.Table(["KV", "KV cache", .. memoryColumns.Select(c => c.Label)],
            rows.Select(r => (IReadOnlyList<string>)[r.Kv, Units.Bytes(r.KvBytes), .. memoryColumns.Select(c => Fits(r.KvBytes, c.Bytes))]));
        context.Write("Memory: " + string.Join(", ", devices.Select(d => $"{d.Name} {(d.Bytes is { } b ? Units.Bytes(b) : "not reported (pass --memory SIZE)")}")));

        context.WriteJson(new JsonObject
        {
            ["model"] = name,
            ["parameters"] = facts.Parameters,
            ["context"] = contextLength,
            ["batch"] = batch,
            ["devices"] = new JsonArray([.. devices.Select(d => (JsonNode)new JsonObject { ["device"] = d.Name, ["bytes"] = d.Bytes })]),
            ["memory"] = given,
            ["formats"] = new JsonArray([.. rows.Select(r => (JsonNode)new JsonObject
            {
                ["kv"] = r.Kv,
                ["kvBytes"] = r.KvBytes,
                ["fits"] = new JsonObject([.. memoryColumns.Select(c => KeyValuePair.Create(c.Label, c.Bytes is { } m ? (JsonNode?)JsonValue.Create(r.KvBytes <= m) : null))]),
            })]),
        });
        return ExitCodes.Ok;
    }

    private static string Percent(long part, long whole)
    {
        double percent = part * 100.0 / Math.Max(1, whole);
        return percent < 0.1 ? "<0.1%" : percent < 10 ? string.Create(CultureInfo.InvariantCulture, $"{percent:0.0}%") : string.Create(CultureInfo.InvariantCulture, $"{percent:0}%");
    }

    // The formats asked for (repeated or comma-separated), else the alias's, else all.
    private static List<string> List(IReadOnlyList<string> given, string? fromAlias, IReadOnlyList<string> all)
    {
        var asked = given.SelectMany(g => g.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();
        return asked.Count > 0 ? asked : fromAlias is not null ? [fromAlias] : [.. all];
    }

    // The chosen device, or every device; the memory each has (null when the library does not report it).
    private static List<(string Name, long? Bytes)> Devices(CommandContext context)
    {
        var devices = context.Option("--device") is not null || context.Config.Get("device") is not null ? [context.Device] : Device.Available.ToList();
        return [.. devices.Select(d => (d.ToString(), Total(d)))];
    }

    private static long? Total(Device device)
    {
        if (!device.IsGpu)
        {
            return ComputeResources.CpuMemoryLimit ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        }

        try
        {
            return ComputeResources.GetMemoryUsage(device).Limit ?? ComputeResources.GpuMemoryLimit;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return null;
        }
    }

    /// <summary>"8G", "512M", "16GB", "1.5T" or a byte count.</summary>
    internal static long ParseSize(string text)
    {
        string t = text.Trim().ToUpperInvariant().TrimEnd('B');
        double scale = t.Length == 0 ? 1 : t[^1] switch { 'K' => 1L << 10, 'M' => 1L << 20, 'G' => 1L << 30, 'T' => 1L << 40, _ => 1 };
        string number = scale == 1 ? t : t[..^1];
        return double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && value > 0
            ? (long)(value * scale)
            : throw new UsageException($"'{text}' is not a size; write e.g. 8G, 512M or 1.5T.");
    }
}
