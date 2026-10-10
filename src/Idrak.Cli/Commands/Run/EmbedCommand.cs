// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Layers;
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
          -i, --input FILE      texts, one per line (repeatable)
              --whole           each --input file is one text instead of one per line
          -o, --out FILE        write the vectors: FILE.npy (float32, one row per text) or FILE.json
              --max-length N    tokens per text (default 512, at most the model's context)
              --batch-size N    texts per batch (default 32)
          -w, --weights FORMAT  int8, int4, bf16 or a registered packed format (default: as stored)
              --context N       the longest input the model is loaded for (default: the model's own)
              --adapter DIR     merge a LoRA adapter into the weights as they are read

        Examples:
          idrak embed Qwen/Qwen3-0.6B "first text" "second text"
          idrak embed qwen -i sentences.txt -o vectors.npy
          idrak embed ./model.gguf -i ./docs/a.md -i ./docs/b.md --whole -o docs.json

        Gap: the library has no pooling head for decoder models trained as embedders (last-token pooling, instruction
        prefixes); mean pooling of the hidden states is what this command does.
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. ModelChoices.ValueOptions.Where(o => o != "--kv"), "--input", "--out", "--max-length", "--batch-size"];   // no KV cache

    public override IReadOnlyCollection<string> Flags => ["--whole"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>(
        ModelChoices.ShortForms.Where(p => p.Key != "-k").Append(KeyValuePair.Create("-i", "--input")).Append(KeyValuePair.Create("-o", "--out")));

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
        var choice = ModelChoices.Choose(context, name);
        using var model = ModelChoices.Load(context, choice);
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

/// <summary>Writes NumPy .npy files (version 1.0, little-endian float32), readable with numpy.load; reads float32 and int64 ones.</summary>
internal static class Npy
{
    /// <summary>A little-endian float32 array in C order, and its shape.</summary>
    public static (float[] Values, int[] Shape) ReadFloat32(string path)
    {
        var (bytes, start, shape) = Read(path, "<f4");
        var values = new float[(bytes.Length - start) / 4];
        var source = bytes.AsSpan(start, values.Length * 4);
        if (BitConverter.IsLittleEndian)
        {
            source.CopyTo(MemoryMarshal.AsBytes(values.AsSpan()));
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = BinaryPrimitives.ReadSingleLittleEndian(source[(i * 4)..]);
            }
        }

        return (values, shape);
    }

    /// <summary>A little-endian int64 array in C order, and its shape.</summary>
    public static (long[] Values, int[] Shape) ReadInt64(string path)
    {
        var (bytes, start, shape) = Read(path, "<i8");
        var values = new long[(bytes.Length - start) / 8];
        var source = bytes.AsSpan(start, values.Length * 8);
        if (BitConverter.IsLittleEndian)
        {
            source.CopyTo(MemoryMarshal.AsBytes(values.AsSpan()));
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = BinaryPrimitives.ReadInt64LittleEndian(source[(i * 8)..]);
            }
        }

        return (values, shape);
    }

    private static (byte[] Bytes, int Start, int[] Shape) Read(string path, string type)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 12 || bytes[0] != 0x93 || Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
        {
            throw new InvalidDataException($"{path} is not a .npy file.");
        }

        int major = bytes[6];
        int length = major == 1 ? BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8)) : BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(8));
        int start = (major == 1 ? 10 : 12) + length;
        string header = Encoding.ASCII.GetString(bytes, major == 1 ? 10 : 12, length);
        if (!header.Contains($"'descr': '{type}'", StringComparison.Ordinal) || !header.Contains("'fortran_order': False", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{path}: expected a {type} array in C order, found {header.Trim()}");
        }

        var match = System.Text.RegularExpressions.Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
        int[] shape = [.. match.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(d => int.Parse(d, System.Globalization.CultureInfo.InvariantCulture))];
        return (bytes, start, shape);
    }

    /// <summary>Writes <paramref name="rows"/> as a [rows, columns] float32 array.</summary>
    public static void Write(string path, IReadOnlyList<float[]> rows, int columns)
    {
        string header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows.Count}, {columns}), }}";
        int total = 10 + header.Length + 1;
        header = header.PadRight(header.Length + (64 - total % 64) % 64) + "\n";   // the data starts on a 64-byte boundary
        using var stream = File.Create(path);
        WriteHeader(stream, header);
        foreach (var row in rows)
        {
            WriteFloat32(stream, row);
        }
    }

    /// <summary>The magic, version 1.0, the header's length (little-endian) and the header (padded, ending in a newline).</summary>
    public static void WriteHeader(Stream stream, string header)
    {
        Span<byte> start = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0, 0, 0];
        BinaryPrimitives.WriteUInt16LittleEndian(start[8..], (ushort)header.Length);
        stream.Write(start);
        stream.Write(Encoding.ASCII.GetBytes(header));
    }

    /// <summary>
    /// <paramref name="values"/> as little-endian float32 bytes: one block write on a little-endian machine, converted
    /// through a pooled buffer on a big-endian one (the file's byte order never follows the machine's).
    /// </summary>
    public static void WriteFloat32(Stream stream, ReadOnlySpan<float> values)
    {
        if (BitConverter.IsLittleEndian)
        {
            stream.Write(MemoryMarshal.AsBytes(values));
            return;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(Math.Min(values.Length, 16 * 1024) * 4);
        try
        {
            int per = buffer.Length / 4;
            for (int i = 0; i < values.Length; i += per)
            {
                int n = Math.Min(per, values.Length - i);
                for (int k = 0; k < n; k++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(k * 4), values[i + k]);
                }

                stream.Write(buffer, 0, n * 4);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
