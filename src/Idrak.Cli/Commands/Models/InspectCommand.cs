// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Inference;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak inspect FILE</c> (<c>i</c>): tensor names, shapes, types and metadata of a GGUF file, a safetensors file or
/// checkpoint folder, or the entries of an .ikm package.
/// </summary>
internal sealed class InspectCommand : Command
{
    public override string Name => "inspect";

    public override IReadOnlyCollection<string> Aliases => ["i"];

    public override string Summary => "Tensor names, shapes, types and metadata of a GGUF, safetensors or .ikm file";

    public override string Usage => """
        FILE [--filter TEXT] [--limit N] [--no-tensors]

        Arguments:
          FILE  a .gguf file, a .safetensors file, a model folder (its safetensors, or sharded index), or an
                .ikm package

        Options:
              --filter TEXT only tensors (and metadata keys) whose name contains this text
              --limit N     at most N tensors in the text output (default all; JSON has all)
              --no-tensors  only the metadata and totals

        Long metadata arrays (a GGUF vocabulary) are shown by their length.

        Examples:
          idrak i ./model.gguf --filter blk.0.
          idrak inspect ./model/model.safetensors --json
          idrak inspect houses.ikm
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--filter", "--limit"];

    public override IReadOnlyCollection<string> Flags => ["--no-tensors"];

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "FILE (a .gguf, .safetensors or .ikm file, or a model folder)");
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            throw new UsageException($"{path} does not exist.");
        }

        string? filter = context.Option("--filter");
        bool Wanted(string name) => filter is null || name.Contains(filter, StringComparison.OrdinalIgnoreCase);
        var tensors = new List<(string Name, int[] Shape, string Type, long Bytes)>();
        var metadata = new JsonObject();
        string format;
        int? version = null;
        if (File.Exists(path) && path.EndsWith(".ikm", StringComparison.OrdinalIgnoreCase))
        {
            return Package(context, path);
        }

        if (File.Exists(path) && path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            format = "gguf";
            using var file = GgufFile.Open(path);
            version = file.Version;
            foreach (var (key, value) in file.Metadata.OrderBy(m => m.Key, StringComparer.Ordinal))
            {
                if (Wanted(key))
                {
                    metadata[key] = Value(value);
                }
            }

            foreach (var t in file.Tensors.Values)
            {
                long bytes;
                try
                {
                    var (values, blockBytes) = GgufFile.BlockSize(t.Type, t.Name);
                    bytes = t.Count / values * blockBytes;
                }
                catch (NotSupportedException)
                {
                    bytes = 0;
                }

                tensors.Add((t.Name, t.Shape, GgufFile.TypeName(t.Type), bytes));
            }
        }
        else
        {
            format = "safetensors";
            using var reader = SafeTensorsReader.Open(path);
            foreach (var (key, value) in reader.Metadata.OrderBy(m => m.Key, StringComparer.Ordinal))
            {
                if (Wanted(key))
                {
                    metadata[key] = value;
                }
            }

            tensors.AddRange(reader.Tensors.Values.Select(t => (t.Name, t.Shape, t.Type.ToString(), t.Length)));
        }

        var shown = tensors.Where(t => Wanted(t.Name)).ToList();
        long parameters = shown.Sum(t => t.Shape.Aggregate(1L, (a, b) => a * b));
        context.Write($"{path}: {format}{(version is null ? "" : $" version {version}")}, {shown.Count} tensors{(filter is null ? "" : $" matching '{filter}'")}, "
                      + $"{Units.Count(parameters)} values, {Units.Bytes(shown.Sum(t => t.Bytes))}");
        if (metadata.Count > 0)
        {
            context.Write("");
            context.Table(["Metadata", "Value"], metadata.Select(m => (IReadOnlyList<string>)[m.Key, Short(m.Value)]));
        }

        var types = shown.GroupBy(t => t.Type).OrderByDescending(g => g.Count()).Select(g => $"{g.Key} ×{g.Count()}");
        context.Write("");
        context.Write("Types: " + string.Join(", ", types));
        if (!context.Flag("--no-tensors"))
        {
            int limit = context.IntOption("--limit", int.MaxValue);
            context.Write("");
            context.Table(["Tensor", "Shape", "Type", "Size"],
                shown.Take(limit).Select(t => (IReadOnlyList<string>)[t.Name, "[" + string.Join(", ", t.Shape) + "]", t.Type, Units.Bytes(t.Bytes)]));
            if (shown.Count > limit)
            {
                context.Write($"... {shown.Count - limit} more (--limit)");
            }
        }

        context.WriteJson(new JsonObject
        {
            ["file"] = path,
            ["format"] = format,
            ["version"] = version,
            ["parameters"] = parameters,
            ["bytes"] = shown.Sum(t => t.Bytes),
            ["metadata"] = metadata,
            ["tensors"] = new JsonArray([.. shown.Select(t => (JsonNode)new JsonObject
            {
                ["name"] = t.Name,
                ["shape"] = new JsonArray([.. t.Shape.Select(d => (JsonNode)d)]),
                ["type"] = t.Type,
                ["bytes"] = t.Bytes,
            })]),
        });
        return ExitCodes.Ok;
    }

    // An .ikm package: its entries (kind, name, size) and, for architectures, the layer count.
    private static int Package(CommandContext context, string path)
    {
        using var package = ModelPackage.Open(path);
        using var zip = ZipFile.OpenRead(path);
        var rows = new List<(string Kind, string Name, long Bytes, string Detail)>();
        JsonArray files;
        using (var manifest = zip.GetEntry("manifest.json")!.Open())
        {
            files = JsonNode.Parse(manifest)?["entries"] as JsonArray ?? [];
        }

        foreach (var entry in package.Entries)
        {
            string? file = files.Where(f => (string?)f?["kind"] == entry.Kind.ToString() && (string?)f?["name"] == entry.Name).Select(f => (string?)f?["file"]).FirstOrDefault();
            long bytes = file is not null && zip.GetEntry(file) is { } z ? z.Length : 0;
            string detail = "";
            if (entry.Kind == PackageEntryKind.Architecture && package.Architecture(entry.Name)["layers"] is JsonArray layers)
            {
                detail = $"{layers.Count} layers";
            }

            rows.Add((entry.Kind.ToString(), entry.Name, bytes, detail));
        }

        context.Write($"{path}: Idrak package, {rows.Count} entries, {Units.Bytes(new FileInfo(path).Length)}");
        context.Table(["Kind", "Name", "Size", "Detail"], rows.Select(r => (IReadOnlyList<string>)[r.Kind, r.Name, r.Bytes > 0 ? Units.Bytes(r.Bytes) : "-", r.Detail]));
        context.WriteJson(new JsonObject
        {
            ["file"] = path,
            ["format"] = "ikm",
            ["entries"] = new JsonArray([.. rows.Select(r => (JsonNode)new JsonObject { ["kind"] = r.Kind, ["name"] = r.Name, ["bytes"] = r.Bytes, ["detail"] = r.Detail })]),
        });
        return ExitCodes.Ok;
    }

    // GGUF metadata as JSON; long arrays by their length.
    private static JsonNode? Value(object value) => value switch
    {
        string s => s,
        long l => l,
        ulong u => u,
        double d => d,
        bool b => b,
        Array { Length: > 16 } a => $"[{a.Length} values]",
        string[] a => new JsonArray([.. a.Select(x => (JsonNode?)JsonValue.Create(x))]),
        long[] a => new JsonArray([.. a.Select(x => (JsonNode?)x)]),
        double[] a => new JsonArray([.. a.Select(x => (JsonNode?)x)]),
        bool[] a => new JsonArray([.. a.Select(x => (JsonNode?)x)]),
        Array a => $"[{a.Length} values]",
        _ => Convert.ToString(value, CultureInfo.InvariantCulture),
    };

    private static string Short(JsonNode? node)
    {
        string text = node is JsonValue v && v.TryGetValue(out string? s) ? s : node?.ToJsonString() ?? "null";
        text = text.ReplaceLineEndings(" ");
        return text.Length > 100 ? text[..97] + "..." : text;
    }
}
