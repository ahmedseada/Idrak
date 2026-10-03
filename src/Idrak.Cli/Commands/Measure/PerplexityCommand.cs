// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak perplexity MODEL FILE</c>: exp of the mean negative log-likelihood of a text's tokens, read in windows of
/// the model's context, to compare weight formats and fine-tunes on the same text.
/// </summary>
internal sealed class PerplexityCommand : Command
{
    public override string Name => "perplexity";

    public override string Summary => "Perplexity of a text under a model (compare weight formats and fine-tunes)";

    public override string Usage =>
        "MODEL FILE [options]\n\n" +
        "The text is tokenized and read in windows of --window tokens (each window starts afresh); every token after\n" +
        "the first of a window is scored. Lower is better; compare runs on the same text and window.\n\n" +
        "Options:\n" +
        "  -w, --weights FORMAT  int8, int4, bf16 or a registered packed format\n" +
        "      --context N       the model's context length; --adapter DIR: merge an adapter first\n" +
        "      --window N        tokens per window (default the context, at most 1024)\n" +
        "      --max-tokens N    score only the first N tokens of the text\n\n" +
        "Examples:\n" +
        "  idrak perplexity org/model wiki.txt\n" +
        "  idrak perplexity org/model wiki.txt -w int4 -j";

    public override IReadOnlyCollection<string> ValueOptions => [.. Models.ValueOptions, "--window", "--max-tokens"];

    public override IReadOnlyDictionary<string, string> ShortForms => Models.ShortForms;

    public override int Run(CommandContext context)
    {
        string modelName = context.Argument(0, "MODEL");
        string file = context.Argument(1, "FILE (the text to score)");
        if (context.Positional.Count > 2)
        {
            throw new UsageException("perplexity takes a model and one text file.");
        }

        if (!File.Exists(file))
        {
            throw new UsageException($"No file {file}: give a text file to score.");
        }

        if (context.Option("--kv") is not null)
        {
            throw new UsageException("--kv does not apply: perplexity reads whole windows without a key/value cache.");
        }

        var choice = Models.Choose(context, modelName);
        using var model = Models.Load(context, choice);
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{choice.Model} has no tokenizer (tokenizer.json).");
        int window = context.IntOption("--window", Math.Min(model.MaxPositions, 1024));
        if (window < 2 || window > model.MaxPositions)
        {
            throw new UsageException($"--window needs 2 to {model.MaxPositions} tokens (the model's context), not {window}.");
        }

        var ids = tokenizer.Encode(File.ReadAllText(file));
        int maxTokens = context.IntOption("--max-tokens", int.MaxValue);
        if (maxTokens < 2)
        {
            throw new UsageException("--max-tokens needs at least 2 tokens.");
        }

        if (ids.Count > maxTokens)
        {
            ids = [.. ids.Take(maxTokens)];
        }

        if (ids.Count < 2)
        {
            throw new InvalidOperationException($"{file} has {ids.Count} token(s); perplexity needs at least 2.");
        }

        var result = Measure(model, ids, window, context.Device, (done, total) => context.Detail($"  {done} of {total} tokens"));
        context.Write($"{choice.Model} on {context.Device}, weights {choice.Weights ?? "as stored"}: perplexity {result.Perplexity:F3}");
        context.Write($"{result.Scored} tokens scored in {result.Windows} window(s) of up to {window} · mean negative log-likelihood {result.MeanNll:F4} · {result.Seconds:F1} s");
        context.WriteJson(new JsonObject
        {
            ["model"] = choice.Model,
            ["device"] = context.Device.ToString(),
            ["weights"] = choice.Weights,
            ["file"] = file,
            ["perplexity"] = result.Perplexity,
            ["mean_nll"] = result.MeanNll,
            ["tokens"] = ids.Count,
            ["scored_tokens"] = result.Scored,
            ["window"] = window,
            ["windows"] = result.Windows,
            ["seconds"] = result.Seconds,
        });
        return ExitCodes.Ok;
    }

    /// <summary>The outcome of <see cref="Measure"/>.</summary>
    internal sealed record Result(double Perplexity, double MeanNll, int Scored, int Windows, double Seconds);

    /// <summary>Scores <paramref name="ids"/> in windows: every position predicts the next token from the window so far.</summary>
    internal static Result Measure(LanguageModels.PretrainedModel model, IReadOnlyList<int> ids, int window, Device device, Action<int, int>? progress = null)
    {
        var watch = Stopwatch.StartNew();
        var network = model.Network;
        var head = network[network.Count - 1];
        double sum = 0;
        int scored = 0, windows = 0;
        using var noGrad = Autograd.NoGrad();
        for (int start = 0; start + 1 < ids.Count; start += window)
        {
            int length = Math.Min(window, ids.Count - start);
            if (length < 2)
            {
                break;
            }

            using var scope = new TensorScope();
            var input = Tensor.From([.. ids.Skip(start).Take(length).Select(i => (float)i)], [1, length], device);
            var hidden = network.ForwardFirst(input, network.Count - 1);
            var flat = hidden.Reshape(length, hidden.Shape[^1]);
            int[] rows = [.. Enumerable.Range(0, length - 1)];
            int[] targets = [.. ids.Skip(start + 1).Take(length - 1)];
            var logProbabilities = Losses.TokenLogProbabilities(flat, head.Forward, rows, targets);
            sum -= logProbabilities.Sum(v => (double)v);
            scored += logProbabilities.Length;
            windows++;
            progress?.Invoke(start + length, ids.Count);
        }

        double mean = sum / Math.Max(1, scored);
        return new Result(Math.Exp(mean), mean, scored, windows, watch.Elapsed.TotalSeconds);
    }
}
