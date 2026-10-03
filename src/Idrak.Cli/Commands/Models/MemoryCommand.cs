// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Layers;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak memory MODEL</c>: the memory a model needs in each weight format and KV cache format at a context length,
/// against each device's memory. Weight sizes come from the tensors' shapes and the bytes per value each registered
/// packed format takes (measured by packing a sample), KV sizes from each registered KV layout's row width.
/// </summary>
internal sealed class MemoryCommand : Command
{
    public override string Name => "memory";

    public override string Summary => "Memory per weight and KV format at a context length, against each device's memory";

    public override string Usage => """
        MODEL [-w FORMAT...] [-k FORMAT...] [--context N] [--batch N] [--memory SIZE]

          MODEL             a cached Hugging Face id, a folder, a .gguf file or an alias
          -w, --weights F   weight formats to show (repeatable or comma-separated; default float32 and every registered
                            packed format: int8, int4, bfloat16, ...)
          -k, --kv F        KV cache formats (default every registered one: float32, int8, bfloat16, ...)
              --context N   the context length (default the model's longest)
              --batch N     sequences decoded together (default 1)
              --memory SIZE also compare with this much memory (e.g. 8G, 512M)

        Projections are counted in the weight format; embeddings and norms in float32. Activations and the runtime add
        a little more (a few hundred MB), so a total close to a device's memory may still not fit.

        Gap: the library reports a GPU's total memory only when a limit is set (ComputeResources.GpuMemoryLimit); for
        other GPUs pass --memory with the card's memory.

        Examples:
          idrak memory Qwen/Qwen3-0.6B
          idrak memory Qwen/Qwen3-8B -k int8 --context 32768 -d vulkan:0

        Environment: IDRAK_CACHE (the cache), IDRAK_DISABLE_CUDA, IDRAK_DISABLE_VULKAN, IDRAK_DISABLE_HIP (devices listed),
        DOTNET_GCHeapHardLimit (the CPU's memory)
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. Shared.Models.ValueOptions, "--batch", "--memory"];

    public override IReadOnlyDictionary<string, string> ShortForms => Shared.Models.ShortForms;

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL");
        var local = ModelCache.Locate(context, name);
        var facts = ModelFacts.Read(local);
        var spec = facts.Spec ?? throw new InvalidOperationException($"{name}: {facts.Problem}");
        var alias = Shared.Models.Choose(context, name);
        int contextLength = context.Option("--context") is null ? spec.MaxPositions : context.IntOption("--context", 0);
        int batch = context.IntOption("--batch", 1);
        if (contextLength <= 0 || batch <= 0)
        {
            throw new UsageException("--context and --batch need positive numbers.");
        }

        var weightFormats = List(context.Options("--weights"), alias.Weights, ["float32", .. PackedWeight.FormatNames.Order(StringComparer.Ordinal)]);
        var kvFormats = List(context.Options("--kv"), alias.Kv, [.. KeyValueLayouts.Names]);

        // Projections (2-D weights other than the embeddings) take the weight format; the rest stays float32.
        long packed = 0, plain = 0;
        foreach (var (tensor, shape) in facts.Tensors)
        {
            long count = shape.Aggregate(1L, (a, b) => a * b);
            bool projection = shape.Length == 2 && !tensor.Contains("embed", StringComparison.OrdinalIgnoreCase) && !tensor.Contains("wte", StringComparison.Ordinal)
                              && !tensor.Contains("wpe", StringComparison.Ordinal);
            if (projection)
            {
                packed += count;
            }
            else
            {
                plain += count;
            }
        }

        var devices = Devices(context);
        long? given = context.Option("--memory") is { } size ? ParseSize(size) : null;
        var rows = new List<(string Weights, string Kv, long WeightBytes, long KvBytes)>();
        foreach (string w in weightFormats)
        {
            double perValue = BytesPerValue(w);
            long weightBytes = (long)(packed * perValue) + plain * 4;
            foreach (string kv in kvFormats)
            {
                var layout = KeyValueLayouts.Get(Kv(kv));
                long perPosition = (long)spec.Layers * spec.KvHeads * 2 * (layout.RowWidth(spec.HeadDim) * 4L + (layout.HasScales ? 4 : 0));
                rows.Add((w, kv, weightBytes, perPosition * contextLength * batch));
            }
        }

        context.Write($"{name}: {ModelCache.Count(facts.Parameters)} parameters ({ModelCache.Count(packed)} in projections), context {contextLength:N0}"
                      + (batch > 1 ? $" × {batch} sequences" : ""));
        var memoryColumns = devices.Select(d => (Label: d.Name, d.Bytes)).ToList();
        if (given is { } g)
        {
            memoryColumns.Add(($"{ModelCache.Size(g)}", g));
        }

        string Fits(long total, long? bytes) => bytes is not { } m ? "?" : total <= m ? $"fits ({Percent(total, m)})" : $"no ({Percent(total, m)})";
        context.Table(["Weights", "KV", "Weights size", "KV cache", "Total", .. memoryColumns.Select(c => c.Label)],
            rows.Select(r => (IReadOnlyList<string>)[r.Weights, r.Kv, ModelCache.Size(r.WeightBytes), ModelCache.Size(r.KvBytes), ModelCache.Size(r.WeightBytes + r.KvBytes),
                .. memoryColumns.Select(c => Fits(r.WeightBytes + r.KvBytes, c.Bytes))]));
        context.Write("Memory: " + string.Join(", ", devices.Select(d => $"{d.Name} {(d.Bytes is { } b ? ModelCache.Size(b) : "not reported (pass --memory SIZE)")}")));

        context.WriteJson(new JsonObject
        {
            ["model"] = name,
            ["parameters"] = facts.Parameters,
            ["projectionParameters"] = packed,
            ["context"] = contextLength,
            ["batch"] = batch,
            ["devices"] = new JsonArray([.. devices.Select(d => (JsonNode)new JsonObject { ["device"] = d.Name, ["bytes"] = d.Bytes })]),
            ["memory"] = given,
            ["formats"] = new JsonArray([.. rows.Select(r => (JsonNode)new JsonObject
            {
                ["weights"] = r.Weights,
                ["kv"] = r.Kv,
                ["weightBytes"] = r.WeightBytes,
                ["kvBytes"] = r.KvBytes,
                ["totalBytes"] = r.WeightBytes + r.KvBytes,
                ["fits"] = new JsonObject([.. memoryColumns.Select(c => KeyValuePair.Create(c.Label, c.Bytes is { } m ? (JsonNode?)JsonValue.Create(r.WeightBytes + r.KvBytes <= m) : null))]),
            })]),
        });
        return ExitCodes.Ok;
    }

    private static string Percent(long part, long whole)
    {
        double percent = part * 100.0 / Math.Max(1, whole);
        return percent < 0.1 ? "<0.1%" : percent < 10 ? $"{percent:0.0}%" : $"{percent:0}%";
    }

    // The formats asked for (repeated or comma-separated), else the alias's, else all.
    private static List<string> List(IReadOnlyList<string> given, string? fromAlias, IReadOnlyList<string> all)
    {
        var asked = given.SelectMany(g => g.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList();
        return asked.Count > 0 ? asked : fromAlias is not null ? [fromAlias] : [.. all];
    }

    private static string Kv(string name) => name.ToLowerInvariant() switch
    {
        "bf16" => "bfloat16",
        "f32" or "fp32" => "float32",
        var other => other,
    };

    /// <summary>Bytes per weight value in <paramref name="format"/>, measured by packing a 256 × 256 sample on the CPU.</summary>
    internal static double BytesPerValue(string format)
    {
        string name = format.ToLowerInvariant() switch
        {
            "bf16" => "bfloat16",
            "f32" or "fp32" or "float32" => "float32",
            var other => other,
        };
        if (name == "float32")
        {
            return 4;
        }

        if (!PackedWeight.FormatNames.Contains(name, StringComparer.OrdinalIgnoreCase))
        {
            throw new UsageException($"Unknown weight format '{format}'; registered: float32, {string.Join(", ", PackedWeight.FormatNames)} (plug-ins add more with --plugin).");
        }

        const int Side = 256;
        var values = new float[Side * Side];
        var random = new Random(1);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1);
        }

        using var weight = PackedWeight.FromValues(name, values, Side, Side, Device.Cpu);
        return weight.Bytes / (double)values.Length;
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
