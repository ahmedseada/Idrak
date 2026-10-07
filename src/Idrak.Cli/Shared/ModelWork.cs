// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;
using Idrak.Nlp.Abstractions;
using Idrak.Nlp;

namespace Idrak.Cli.Shared;

/// <summary>Helpers of the commands that load and write models (quantize, merge, convert): sizes, quality checks, output folders.</summary>
internal static class ModelWork
{
    /// <summary>A short English text for a quick perplexity check when none is given.</summary>
    public const string SampleText =
        "The library reads a model from its folder, builds the network on the chosen device and generates text one token at a time. "
        + "Each step looks at the tokens so far, scores every word in the vocabulary and picks the next one. "
        + "Smaller weight formats save memory and time, and a short check like this one shows how much quality they cost.";

    /// <summary>Bytes the model's parameters and buffers take on its device (packed weights at their packed size; tied tensors once).</summary>
    public static long Bytes(Module network)
    {
        var seen = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        long bytes = 0;
        foreach (var tensor in network.Parameters().Concat(network.Buffers()))
        {
            if (seen.Add(tensor))
            {
                bytes += tensor.Size * 4L;
            }
        }

        return bytes;
    }

    /// <summary>The perplexity of <paramref name="tokens"/> under the model (exp of the mean loss per predicted token).</summary>
    public static double Perplexity(PretrainedModel model, int[] tokens) =>
        Math.Exp(FineTuner.Evaluate(model, [new TrainingSequence(tokens, [.. tokens.Select(_ => true)])]));

    /// <summary>The tokens of the check text (<c>--text FILE</c> or <see cref="SampleText"/>), at most <paramref name="limit"/>.</summary>
    public static int[] CheckTokens(CommandContext context, PretrainedModel model, int limit)
    {
        string text = context.Option("--text") is { } file
            ? File.Exists(file) ? File.ReadAllText(file) : throw new UsageException($"--text: {file} does not exist.")
            : SampleText;
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{model.Folder} has no tokenizer.json, so no text can be scored.");
        int[] tokens = [.. tokenizer.Encode(text).Take(Math.Min(limit, model.MaxPositions))];
        return tokens.Length >= 2 ? tokens : throw new UsageException("The check text has fewer than two tokens; give a longer --text FILE.");
    }

    /// <summary>Makes sure <paramref name="folder"/> may be written: absent or empty, or <c>--force</c>.</summary>
    public static void CheckOutput(CommandContext context, string folder)
    {
        if (File.Exists(folder))
        {
            throw new UsageException($"{folder} is a file; the output is a folder.");
        }

        if (Directory.Exists(folder) && Directory.EnumerateFileSystemEntries(folder).Any() && !context.Flag("--force"))
        {
            throw new UsageException($"{folder} is not empty; choose another folder or pass --force to write into it.");
        }
    }

    /// <summary>The safetensors element type named by <c>--type</c> (default bf16).</summary>
    public static SafeTensorType Type(CommandContext context) => (context.Option("--type") ?? "bf16").ToLowerInvariant() switch
    {
        "bf16" or "bfloat16" => SafeTensorType.BF16,
        "f16" or "fp16" or "float16" => SafeTensorType.F16,
        "f32" or "fp32" or "float32" => SafeTensorType.F32,
        var other => throw new UsageException($"--type takes bf16, f16 or f32, not '{other}'."),
    };

    /// <summary>
    /// Writes a float32 model as a Hugging Face folder; a model read from GGUF loses the marker of its prepared folder, so
    /// the copy reads as an ordinary safetensors model.
    /// </summary>
    public static void SaveHuggingFace(PretrainedModel model, string folder, SafeTensorType type)
    {
        model.SaveHuggingFace(folder, type);
        string marker = Path.Combine(folder, "gguf.json");
        if (File.Exists(marker))
        {
            File.Delete(marker);
        }
    }
}
