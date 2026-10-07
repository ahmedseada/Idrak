// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Abstractions;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers.Abstractions;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Onnx.Abstractions;
using Idrak.Onnx;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// Everything the libraries have registered, by category (built in, or added by a plug-in): what <c>plugins list</c>,
/// <c>formats</c> and <c>help formats</c> show. A snapshot taken before the <c>--plugin</c> assemblies load tells what
/// they added.
/// </summary>
internal static class Registries
{
    /// <summary>One category: its key, its name in listings, whether it is a format, and the names registered.</summary>
    public sealed record Category(string Key, string Title, bool Format, IReadOnlyList<string> Names);

    /// <summary>Every category, names sorted.</summary>
    public static IReadOnlyList<Category> Snapshot()
    {
        static IReadOnlyList<string> Sorted(IEnumerable<string> names) => [.. names.Order(StringComparer.OrdinalIgnoreCase)];
        return
        [
            new("weights", "Weight (packed) formats", true, Sorted(PackedWeight.FormatNames)),
            new("kv", "KV cache formats", true, Sorted(KeyValueLayouts.Names)),
            new("checkpoints", "Checkpoint formats", true, Sorted(CheckpointFormats.Names)),
            new("datafiles", "Dataset file formats", true, Sorted(DataFileFormats.Names)),
            new("datasets", "Dataset sources", true, Sorted(DatasetSources.Names)),
            new("toolcalls", "Tool-call formats", true, Sorted(ToolCallFormats.Names)),
            new("families", "Model families", false, Sorted(PretrainedArchitectures.Names)),
            new("gguf", "GGUF architectures", false, Sorted(GgufArchitectures.Names)),
            new("rope", "RoPE scalings", false, Sorted(RopeScalings.Names)),
            new("graphops", "Graph ops", false, Sorted(GraphOps.Names)),
            new("layers", "Layer types (network builder)", false, Sorted(NetworkOps.Names)),
            new("onnx", "ONNX import ops", false, Sorted(OnnxImportOps.Names)),
            new("devices", "Device kinds", false, Sorted(DeviceListing.Backends.Select(b => b.Kind).Prepend("cpu"))),
        ];
    }

    /// <summary>The names in <paramref name="after"/> that <paramref name="before"/> did not have, per category.</summary>
    public static IReadOnlyList<Category> Added(IReadOnlyList<Category> before, IReadOnlyList<Category> after) =>
    [
        .. after.Select(c => c with
        {
            Names = [.. c.Names.Except(before.FirstOrDefault(b => b.Key == c.Key)?.Names ?? [], StringComparer.OrdinalIgnoreCase)],
        }),
    ];
}
