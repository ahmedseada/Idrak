// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Reflection;
using System.Text.Json.Nodes;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Onnx;
using Module = Idrak.Abstraction.Module;

namespace Idrak.Cli.Commands.Developer;

/// <summary><c>idrak onnx import FILE.onnx</c>: rebuilds an ONNX model from Idrak layers and saves it as a model package.</summary>
internal sealed class OnnxImportCommand : Command
{
    public override string Name => "onnx import";

    public override string Summary => "Imports an ONNX model into Idrak layers and saves a model package (.ikm)";

    public override string Usage =>
        "FILE.onnx [-o OUT.ikm] [--shape D,D,...] [--force]\n\n" +
        "Prints the layers (or the graph's nodes), the approximations made and the file's metadata, and saves the model\n" +
        "with its weights as a package that Predictor.Load, idrak predict and the inference engine read.\n\n" +
        "Options:\n" +
        "  -o, --out FILE      the package to write (default FILE.ikm next to the model)\n" +
        "      --shape D,...   one input sample's shape without the batch, when the file leaves it dynamic (e.g. 3,32,32)\n" +
        "  -f, --force         overwrite the package\n\n" +
        "Examples:\n" +
        "  idrak onnx import model.onnx\n" +
        "  idrak onnx import resnet.onnx --shape 3,224,224 -o resnet.ikm -d vulkan:0";

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--shape"];

    public override IReadOnlyCollection<string> Flags => ["--force"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string file = OnnxShared.ExistingFile(context.Argument(0, "the .onnx file"));
        string output = Path.GetFullPath(context.Option("--out") ?? Path.ChangeExtension(file, ".ikm"));
        if (File.Exists(output) && !context.Flag("--force"))
        {
            context.Error($"{output} exists. Pass --force to overwrite it, or choose another file with --out.");
            return ExitCodes.Failed;
        }

        using var imported = OnnxImport.Load(file, context.Device, OnnxShared.Shape(context));
        var layers = OnnxShared.Describe(imported.Model);
        context.Write($"Imported {Path.GetFileName(file)} on {context.Device}: {(imported.IsGraph ? $"a graph of {layers.Count} nodes" : $"a chain of {layers.Count} layers")}" +
            string.Create(CultureInfo.InvariantCulture, $"{(imported.InputShape is { } shape ? $", input [{string.Join(", ", shape)}]" : "")}, {imported.Model.Parameters().Sum(p => (long)p.Size):N0} parameters"));
        foreach (string layer in layers)
        {
            context.Write($"  {layer}");
        }

        foreach (string note in imported.Notes)
        {
            context.Write($"  note: {note}");
        }

        foreach (var (key, value) in imported.Metadata)
        {
            context.Detail($"  metadata {key} = {value}");
        }

        imported.SavePackage(output);
        context.Write(string.Create(CultureInfo.InvariantCulture, $"Wrote {output} ({new FileInfo(output).Length:N0} bytes)"));
        context.WriteJson(new JsonObject
        {
            ["file"] = file,
            ["kind"] = imported.IsGraph ? "graph" : "chain",
            ["inputShape"] = imported.InputShape is { } s ? new JsonArray([.. s.Select(d => (JsonNode)d)]) : null,
            ["parameters"] = imported.Model.Parameters().Sum(p => (long)p.Size),
            ["layers"] = new JsonArray([.. layers.Select(l => (JsonNode)l)]),
            ["notes"] = new JsonArray([.. imported.Notes.Select(n => (JsonNode)n)]),
            ["metadata"] = new JsonObject([.. imported.Metadata.Select(p => KeyValuePair.Create(p.Key, (JsonNode?)p.Value))]),
            ["package"] = output,
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak onnx export MODEL</c>: writes a model package (or a network JSON) as an .onnx file.</summary>
internal sealed class OnnxExportCommand : Command
{
    public override string Name => "onnx export";

    public override string Summary => "Exports a model package (.ikm) or a network JSON to ONNX";

    public override string Usage =>
        "MODEL.ikm|network.json [-o OUT.onnx] [--shape D,D,...] [--force]\n\n" +
        "MODEL is a model package (its architecture and weights), or the network builder's JSON (as idrak suggest\n" +
        "writes it), which is exported with freshly initialized weights.\n\n" +
        "Options:\n" +
        "  -o, --out FILE      the .onnx file to write (default MODEL.onnx next to the model)\n" +
        "      --shape D,...   one input sample's shape without the batch (default: from the package or the network)\n" +
        "  -f, --force         overwrite the file\n\n" +
        "Examples:\n" +
        "  idrak onnx export houses.ikm\n" +
        "  idrak onnx export ./run/network.json -o network.onnx";

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--shape"];

    public override IReadOnlyCollection<string> Flags => ["--force"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string file = OnnxShared.ExistingFile(context.Argument(0, "the model package (.ikm) or network JSON"));
        string output = Path.GetFullPath(context.Option("--out") ?? Path.ChangeExtension(file, ".onnx"));
        if (File.Exists(output) && !context.Flag("--force"))
        {
            context.Error($"{output} exists. Pass --force to overwrite it, or choose another file with --out.");
            return ExitCodes.Failed;
        }

        bool fromJson = file.EndsWith(".json", StringComparison.OrdinalIgnoreCase);
        var (model, stored) = fromJson ? FromNetworkJson(file, context.Device) : FromPackage(file, context.Device);
        using (model)
        {
            int[] shape = OnnxShared.Shape(context) ?? stored
                ?? throw new UsageException("The model does not say its input's shape; pass --shape (one sample without the batch, e.g. 3,32,32).");
            model.Eval();
            var exporter = OnnxExport.For(model).Input(shape).Metadata("source", Path.GetFileName(file));
            exporter.Save(output);
            context.Write($"Exported {Path.GetFileName(file)} ({OnnxShared.Describe(model).Count} layers, input [{string.Join(", ", shape)}]) to {output} " +
                string.Create(CultureInfo.InvariantCulture, $"({new FileInfo(output).Length:N0} bytes)"));
            if (fromJson)
            {
                context.Write("note: a network JSON has no weights; the file has freshly initialized ones (export a trained package for real weights).");
            }

            context.WriteJson(new JsonObject
            {
                ["model"] = file,
                ["inputShape"] = new JsonArray([.. shape.Select(d => (JsonNode)d)]),
                ["weights"] = fromJson ? "initialized" : "trained",
                ["onnx"] = output,
                ["bytes"] = new FileInfo(output).Length,
            });
        }

        return ExitCodes.Ok;
    }

    private static (Module Model, int[]? Shape) FromNetworkJson(string file, Device device)
    {
        var builder = Network.FromJson(JsonNode.Parse(File.ReadAllText(file)) ?? throw new InvalidDataException($"{file} is empty."));
        int[] shape = [.. builder.InputShape];
        return (builder.OnDevice(device).Build(), shape);
    }

    private static (Module Model, int[]? Shape) FromPackage(string file, Device device)
    {
        using var package = ModelPackage.Open(file);
        int[]? shape = null;
        if (package.Contains(PackageEntryKind.Json, "predictor") && package.Json("predictor")["inputShape"] is JsonArray stored)
        {
            shape = [.. stored.Select(d => (int)d!)];
        }
        else
        {
            try
            {
                shape = [.. package.Network().InputShape];
            }
            catch (Exception e) when (e is InvalidOperationException or NotSupportedException or KeyNotFoundException or InvalidDataException or ArgumentException)
            {
                // A graph or decoder architecture: the shape comes from --shape.
            }
        }

        return (package.BuildModel(device: device), shape);
    }
}

/// <summary>
/// <c>idrak onnx check FILE.onnx</c>: imports the model, runs a batch of random samples, and compares the outputs with an
/// export-and-reimport round trip and, when Idrak.Onnx.Runtime is available, with ONNX Runtime.
/// </summary>
internal sealed class OnnxCheckCommand : Command
{
    /// <summary>The ONNX Runtime module type of Idrak.Onnx.Runtime, looked up by name so the tool needs no dependency on it.</summary>
    internal const string RuntimeType = "Idrak.Onnx.Runtime.OnnxModule";

    public override string Name => "onnx check";

    public override string Summary => "Checks an ONNX model: Idrak's import against a round trip and against ONNX Runtime";

    public override string Usage =>
        "FILE.onnx [--shape D,D,...] [--batch N] [--tolerance X] [--expected FILE]\n\n" +
        "Imports the model on the chosen device, runs N random samples (token ids for a model that starts with an\n" +
        "embedding) and compares Idrak's outputs with:\n" +
        "  - the model exported by Idrak and imported again (the round trip);\n" +
        "  - ONNX Runtime, when the Idrak.Onnx.Runtime assembly is available: next to the tool, or loaded with\n" +
        "    --plugin PATH/Idrak.Onnx.Runtime.dll (with Microsoft.ML.OnnxRuntime beside it). Without it the check says so;\n" +
        "  - the outputs another framework computed, from FILE.onnx.expected.json next to the model or --expected FILE\n" +
        "    ({\"inputs\": [[...], ...], \"outputs\": [[...], ...], \"inputShape\": [...]}, as tools/pytorch writes it).\n" +
        "Exits with 1 when a difference is larger than the tolerance.\n\n" +
        "Options:\n" +
        "      --shape D,...   one input sample's shape without the batch, when the file leaves it dynamic\n" +
        "      --batch N       samples to run (default 4)\n" +
        "      --tolerance X   largest allowed difference (default 1e-4)\n" +
        "      --expected FILE reference inputs and outputs to compare with (default FILE.onnx.expected.json when present)\n\n" +
        "Examples:\n" +
        "  idrak onnx check model.onnx\n" +
        "  idrak onnx check model.onnx -d vulkan:0 --tolerance 1e-3 -P ./ort/Idrak.Onnx.Runtime.dll";

    public override IReadOnlyCollection<string> ValueOptions => ["--shape", "--batch", "--tolerance", "--expected"];

    public override int Run(CommandContext context)
    {
        string file = OnnxShared.ExistingFile(context.Argument(0, "the .onnx file"));
        int batch = context.IntOption("--batch", 4);
        if (batch < 1)
        {
            throw new UsageException("--batch needs a positive number.");
        }

        float tolerance = context.Option("--tolerance") is not { } t ? 1e-4f
            : float.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) && parsed >= 0 ? parsed
            : throw new UsageException($"--tolerance needs a number such as 1e-4, not '{t}'.");
        var device = context.Device;
        byte[] bytes = File.ReadAllBytes(file);
        using var imported = OnnxImport.Load(bytes, device, OnnxShared.Shape(context));
        int[] shape = OnnxShared.Shape(context) ?? (imported.InputShape is { } s ? [.. s]
            : throw new UsageException("The file leaves the input's shape dynamic; pass --shape (one sample without the batch, e.g. 3,32,32)."));
        float[] input = Sample(imported.Model, shape, batch, context.Seed ?? 1);
        float[] reference = Run(imported.Model, input, shape, batch, device);
        context.Write($"Imported {Path.GetFileName(file)} on {device}: {(imported.IsGraph ? "a graph" : "a chain of layers")}, input [{string.Join(", ", shape)}], " +
            $"{batch} random samples, {reference.Length / batch} outputs each");

        var checks = new List<(string Name, string Status, float? Difference, string? Note)>();
        try
        {
            byte[] again = OnnxExport.For(imported.Model).Input(shape).ToBytes();
            using var roundTrip = OnnxImport.Load(again, device, shape);
            checks.Add(Compare("Idrak export, imported again", reference, Run(roundTrip.Model, input, shape, batch, device), tolerance));
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or KeyNotFoundException)
        {
            checks.Add(("Idrak export, imported again", "skipped", null, e.Message));
        }

        if (RuntimeLoader() is not { } load)
        {
            checks.Add(("ONNX Runtime", "unavailable", null,
                "Idrak.Onnx.Runtime is not loaded; pass --plugin PATH/Idrak.Onnx.Runtime.dll (Microsoft.ML.OnnxRuntime beside it) to compare with it"));
        }
        else
        {
            try
            {
                using var runtime = (Module)load(bytes);
                checks.Add(Compare("ONNX Runtime", reference, Run(runtime, input, shape, batch, Device.Cpu), tolerance));
            }
            catch (Exception e) when (e is TargetInvocationException or NotSupportedException or DllNotFoundException or TypeInitializationException or FileNotFoundException)
            {
                checks.Add(("ONNX Runtime", "unavailable", null, (e.InnerException ?? e).Message));
            }
        }

        string? expectedPath = context.Option("--expected") ?? (File.Exists(file + ".expected.json") ? file + ".expected.json" : null);
        if (expectedPath is not null)
        {
            var expected = JsonNode.Parse(File.ReadAllText(OnnxShared.ExistingFile(expectedPath))) ?? throw new InvalidDataException($"{expectedPath} is empty.");
            var rows = expected["inputs"]?.AsArray().Select(r => r!.AsArray().Select(v => (float)v!).ToArray()).ToList()
                ?? throw new InvalidDataException($"{expectedPath} has no \"inputs\" list.");
            var outputs = expected["outputs"]?.AsArray().SelectMany(r => r!.AsArray().Select(v => (float)v!)).ToArray()
                ?? throw new InvalidDataException($"{expectedPath} has no \"outputs\" list.");
            int[] expectedShape = expected["inputShape"] is JsonArray stored ? [.. stored.Select(d => (int)d!)] : shape;
            var actual = Run(imported.Model, [.. rows.SelectMany(r => r)], expectedShape, rows.Count, device);
            checks.Add(Compare($"reference outputs ({Path.GetFileName(expectedPath)})", outputs, actual, tolerance));
        }

        context.Table(["Check", "Largest difference", "Result"],
            checks.Select(c => (IReadOnlyList<string>)[c.Name, c.Difference is { } d ? d.ToString("E2", CultureInfo.InvariantCulture) : "-", c.Note is null ? c.Status : $"{c.Status}: {c.Note}"]));
        bool ok = checks.All(c => c.Status != "mismatch");
        context.Write(ok ? string.Create(CultureInfo.InvariantCulture, $"OK (tolerance {tolerance:E0})") : string.Create(CultureInfo.InvariantCulture, $"MISMATCH: a difference is larger than {tolerance:E0}"));
        context.WriteJson(new JsonObject
        {
            ["file"] = file,
            ["device"] = device.ToString(),
            ["inputShape"] = new JsonArray([.. shape.Select(d => (JsonNode)d)]),
            ["batch"] = batch,
            ["tolerance"] = tolerance,
            ["checks"] = new JsonArray([.. checks.Select(c => (JsonNode)new JsonObject
            {
                ["name"] = c.Name, ["status"] = c.Status, ["largestDifference"] = c.Difference, ["note"] = c.Note,
            })]),
            ["ok"] = ok,
        });
        return ok ? ExitCodes.Ok : ExitCodes.Failed;
    }

    /// <summary>
    /// A loader for ONNX Runtime modules (bytes of an .onnx file to a <see cref="Module"/>) when Idrak.Onnx.Runtime is
    /// loaded (by --plugin or a reference) or found next to the tool; otherwise null.
    /// </summary>
    internal static Func<byte[], object>? RuntimeLoader()
    {
        Type? type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(RuntimeType)).FirstOrDefault(t => t is not null);
        if (type is null)
        {
            try
            {
                type = Type.GetType($"{RuntimeType}, Idrak.Onnx.Runtime");
            }
            catch (Exception e) when (e is FileNotFoundException or FileLoadException or BadImageFormatException)
            {
                return null;
            }
        }

        var method = type?.GetMethods(BindingFlags.Public | BindingFlags.Static)
            .FirstOrDefault(m => m.Name == "Load" && m.GetParameters() is [{ ParameterType: var p }, _] && p == typeof(byte[]));
        return method is null ? null : bytes => method.Invoke(null, [bytes, null])!;
    }

    // Random samples, or token ids in the vocabulary when the model starts with an embedding.
    private static float[] Sample(Module model, int[] shape, int batch, int seed)
    {
        var random = new Random(seed);
        int vocabulary = model is Sequential { Count: > 0 } chain && chain[0] is Embedding embedding ? embedding.Vocabulary : 0;
        return [.. Enumerable.Range(0, batch * shape.Aggregate(1, (a, b) => a * b)).Select(_ => vocabulary > 0 ? random.Next(vocabulary) : random.NextSingle() * 2f - 1f)];
    }

    private static float[] Run(Module model, float[] input, int[] shape, int batch, Device device)
    {
        using var scope = new TensorScope();
        using var tensor = Tensor.From(input, [batch, .. shape], device);
        return model.Predict(tensor).ToArray();
    }

    private static (string, string, float?, string?) Compare(string name, float[] expected, float[] actual, float tolerance)
    {
        if (expected.Length != actual.Length)
        {
            return (name, "mismatch", null, $"{actual.Length} values against {expected.Length}");
        }

        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        }

        return (name, worst <= tolerance && !float.IsNaN(worst) ? "match" : "mismatch", worst, null);
    }
}

/// <summary>What the onnx commands share.</summary>
internal static class OnnxShared
{
    /// <summary>The full path of an existing file, or a failure naming it.</summary>
    public static string ExistingFile(string path) =>
        File.Exists(path) ? Path.GetFullPath(path) : throw new FileNotFoundException($"{Path.GetFullPath(path)} does not exist.");

    /// <summary>The <c>--shape</c> option (one sample, without the batch), or null.</summary>
    public static int[]? Shape(CommandContext context)
    {
        if (context.Option("--shape") is not { } text)
        {
            return null;
        }

        var parts = text.Split([',', 'x'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length > 0 && parts.All(p => int.TryParse(p, NumberStyles.Integer, CultureInfo.InvariantCulture, out int d) && d > 0)
            ? [.. parts.Select(p => int.Parse(p, CultureInfo.InvariantCulture))]
            : throw new UsageException($"--shape needs positive whole numbers separated by commas (e.g. 3,32,32), not '{text}'.");
    }

    /// <summary>One line per layer of a chain, or per node of a graph.</summary>
    public static List<string> Describe(Module model) => model switch
    {
        GraphModule graph => [.. graph.Nodes.Select(n => $"{n.Output} = {(n.Layer is { } layer ? layer.ToString() : n.Op)}({string.Join(", ", n.Inputs)})")],
        Sequential chain => [.. chain.Select(l => l.ToString() ?? l.GetType().Name)],
        _ => [model.ToString() ?? model.GetType().Name],
    };
}
