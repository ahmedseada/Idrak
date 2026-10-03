// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Layers;
using Idrak.LanguageModels;
using Idrak.Retrieval;

namespace Idrak.Cli.Commands.Run;

/// <summary><c>idrak embed MODEL --input FILE</c>: embeddings of lines or files, to JSON or .npy.</summary>
internal sealed class EmbedCommand : Command
{
    public override string Name => "embed";

    public override string Summary => "Embeddings of lines or files to JSON or .npy";

    public override string Usage => """
        MODEL ["text" ...] [options]

        Each text is embedded as the mean of the model's last hidden states over its tokens, scaled to length 1 (so a
        dot product is the cosine similarity). Texts are the arguments after MODEL and every line of the --input files
        (or each whole file with --whole). Without --out the vectors are printed as JSON.

        Options:
          -i, --input FILE       texts, one per line (repeatable)
              --whole            each --input file is one text instead of one per line
          -o, --out FILE         write the vectors: FILE.npy (float32, one row per text) or FILE.json
              --max-length N     tokens per text (default 512, at most the model's context)
              --batch-size N     texts per batch (default 32)
          -w, --weights FORMAT   int8, int4, bf16 or a registered packed format (default: as stored)
              --adapter DIR      merge a LoRA adapter into the weights as they are read

        Gap: the library has no pooling head for decoder models trained as embedders (last-token pooling, instruction
        prefixes); mean pooling of the hidden states is what this command does.

        Examples:
          idrak embed Qwen/Qwen3-0.6B "first text" "second text"
          idrak embed qwen -i sentences.txt -o vectors.npy
          idrak embed ./model.gguf -i ./docs/a.md -i ./docs/b.md --whole -o docs.json
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. Models.ValueOptions, "--input", "--out", "--max-length", "--batch-size"];

    public override IReadOnlyCollection<string> Flags => ["--whole"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>(
        Models.ShortForms.Append(KeyValuePair.Create("-i", "--input")).Append(KeyValuePair.Create("-o", "--out")));

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        var texts = context.Positional.Skip(1).ToList();
        foreach (string file in context.Options("--input"))
        {
            if (!File.Exists(file))
            {
                throw new UsageException($"Input file not found: {file}");
            }

            if (context.Flag("--whole"))
            {
                texts.Add(File.ReadAllText(file));
            }
            else
            {
                texts.AddRange(File.ReadLines(file).Where(l => l.Trim().Length > 0));
            }
        }

        if (texts.Count == 0)
        {
            throw new UsageException("Nothing to embed: give texts after MODEL or --input FILE.");
        }

        string? output = context.Option("--out");
        if (output is not null && !output.EndsWith(".npy", StringComparison.OrdinalIgnoreCase) && !output.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            throw new UsageException("--out needs a .npy or .json file name.");
        }

        int batchSize = context.IntOption("--batch-size", 32);
        var choice = Models.Choose(context, name);
        using var model = Models.Load(context, choice);
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{name} has no tokenizer.json.");
        int maxLength = Math.Min(context.IntOption("--max-length", 512), model.MaxPositions);
        if (maxLength < 1 || batchSize < 1)
        {
            throw new UsageException("--max-length and --batch-size need positive whole numbers.");
        }

        model.Network.Eval();
        var encoder = new TextEncoder(new HiddenStates(model.Network), tokenizer, maxLength, padId: 0);
        var vectors = encoder.Encode(texts, batchSize);
        int dimensions = vectors.Length == 0 ? 0 : vectors[0].Length;
        if (output is not null && output.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
        {
            Npy.Write(output, vectors, dimensions);
        }
        else
        {
            var json = new JsonObject
            {
                ["model"] = choice.Model,
                ["dimensions"] = dimensions,
                ["embeddings"] = new JsonArray([.. texts.Select((t, i) => (JsonNode)new JsonObject
                {
                    ["text"] = t,
                    ["vector"] = new JsonArray([.. vectors[i].Select(v => (JsonNode)v)]),
                })]),
            };
            if (output is not null)
            {
                File.WriteAllText(output, json.ToJsonString());
            }
            else
            {
                // The vectors are the result: printed as JSON with or without --json.
                context.Output.WriteLine(json.ToJsonString(CommandContext.JsonOutput));
                return ExitCodes.Ok;
            }
        }

        context.Write($"Wrote {texts.Count} vectors of {dimensions} dimensions to {output}.");
        context.WriteJson(new JsonObject { ["model"] = choice.Model, ["count"] = texts.Count, ["dimensions"] = dimensions, ["out"] = output });
        return ExitCodes.Ok;
    }
}

/// <summary>A decoder's hidden states: every layer of the network but the output layer (token ids [N, T] to [N, T, D]).</summary>
internal sealed class HiddenStates(Sequential network) : Module
{
    protected override Tensor ForwardCore(Tensor input) => network.ForwardFirst(input, network.Count - 1);

    public override IEnumerable<Module> Children() => [network];
}

/// <summary>Writes NumPy .npy files (version 1.0, little-endian float32), readable with numpy.load.</summary>
internal static class Npy
{
    /// <summary>Writes <paramref name="rows"/> as a [rows, columns] float32 array.</summary>
    public static void Write(string path, IReadOnlyList<float[]> rows, int columns)
    {
        string header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows.Count}, {columns}), }}";
        int total = 10 + header.Length + 1;
        header = header.PadRight(header.Length + (64 - total % 64) % 64) + "\n";   // the data starts on a 64-byte boundary
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write((byte)0x93);
        writer.Write("NUMPY"u8);
        writer.Write((byte)1);
        writer.Write((byte)0);
        writer.Write((ushort)header.Length);
        writer.Write(Encoding.ASCII.GetBytes(header));
        foreach (var row in rows)
        {
            foreach (float value in row)
            {
                writer.Write(value);
            }
        }
    }
}
