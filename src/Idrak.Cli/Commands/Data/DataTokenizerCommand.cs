// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data tokenizer FILE... --vocab N -o tokenizer.json</c>: a tokenizer learned from text (<see cref="TokenizerTrainers"/>)
/// for a model made from scratch (plan 19).
/// </summary>
internal sealed class DataTokenizerCommand : Command
{
    // The special tokens a created model's template and a vision family use, when none are given.
    internal static readonly string[] DefaultSpecials = ["<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<start_of_image>", "<image>", "<end_of_image>"];

    public override string Name => "data tokenizer";

    public override string Summary => "Learn a tokenizer (byte-pair encoding) from text files, for a model made from scratch";

    public override string Usage => """
        FILE... --vocab N -o tokenizer.json [options]

        Learns a tokenizer from UTF-8 text files (each line, with its line end, one sequence), as Hugging Face's
        BpeTrainer learns one: the same vocabulary and merges from the same files. The special tokens come first. The
        file is a tokenizer.json that idrak create, Hugging Face and every idrak command read.

        Arguments:
          FILE  text files to learn from (.txt, any UTF-8 text)

        Options:
              --vocab N            the vocabulary to reach: special tokens, the alphabet and the merged tokens (32000)
              --min-frequency N    a pair seen fewer times is never merged (2)
              --special TOKEN      a special token (repeat for several); replaces the defaults: <pad> <s> </s>
                                   <|im_start|> <|im_end|> <start_of_image> <image> <end_of_image>
              --no-special         no special tokens at all
              --pre-tokenizer P    bytelevel (bytes as characters: every text encodes, the default) or metaspace (spaces
                                   as ▁ starting a word, SentencePiece's way)
              --prefix-space       byte level: a space before each text that does not start with one
              --limit-alphabet N   keep at most N characters in the alphabet (the most frequent)
              --trainer NAME       the trainer (bpe; others from a plug-in, -P)
              --workers N          threads counting words (one per processor)
          -o, --out FILE           the tokenizer.json to write

        Examples:
          idrak data text -n 20000 -o corpus.txt
          idrak data tokenizer corpus.txt --vocab 32000 -o tokenizer.json
          idrak data tokenizer books/*.txt --vocab 64000 --special "<s>" --special "</s>" -o tokenizer.json
          idrak data tokens corpus.txt --tokenizer tokenizer.json -o tokens.bin
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--vocab", "--min-frequency", "--special", "--pre-tokenizer", "--limit-alphabet", "--trainer", "--workers", "--out"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--no-special", "--prefix-space"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the tokenizer.json to write.");
        var files = TextFiles.Expand(context.Positional);
        if (files.Count == 0)
        {
            throw new UsageException("Give the text files to learn from.");
        }

        var defaults = new TokenizerTrainingOptions();
        var specials = context.Flag("--no-special") ? [] : context.Options("--special") is { Count: > 0 } given ? given : DefaultSpecials;
        var options = defaults with
        {
            VocabularySize = context.IntOption("--vocab", defaults.VocabularySize),
            MinFrequency = context.IntOption("--min-frequency", defaults.MinFrequency),
            SpecialTokens = [.. specials],
            PreTokenizer = context.Option("--pre-tokenizer") ?? defaults.PreTokenizer,
            AddPrefixSpace = context.Flag("--prefix-space"),
            LimitAlphabet = context.Option("--limit-alphabet") is null ? null : context.IntOption("--limit-alphabet", 0),
            Workers = context.IntOption("--workers", 0),
        };
        ITokenizerTrainer trainer;
        try
        {
            trainer = TokenizerTrainers.Get(context.Option("--trainer") ?? TokenizerTrainers.Bpe);
        }
        catch (NotSupportedException e)
        {
            context.Error(Messages.T("{0} Load its plug-in with -P (--plugin).", e.Message));
            return ExitCodes.Failed;
        }

        long total = files.Sum(f => new FileInfo(f).Length);
        var watch = Stopwatch.StartNew();
        TimeSpan counted = default;
        Models.BpeTokenizer tokenizer;
        using (var line = new ProgressLine(context, "counting words", total: total, unit: ProgressUnit.Bytes))
        {
            var reports = new Reports(line, options.VocabularySize, () => counted = watch.Elapsed);
            try
            {
                tokenizer = trainer.Train(files, options, reports);
            }
            catch (Exception e) when (e is ArgumentException or FileNotFoundException or InvalidDataException)
            {
                throw new UsageException(e.Message);
            }

            line.Finish();
        }

        tokenizer.Save(output);
        int merges = tokenizer.ToJson()["model"]?["merges"]?.AsArray().Count ?? 0;
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{tokenizer.VocabularySize:N0} tokens ({specials.Count} special, {merges:N0} merges) learned from {files.Count} file{(files.Count == 1 ? "" : "s")} ({total / 1048576.0:F1} MB) in {watch.Elapsed.TotalSeconds:F1} s, written to {output}"));
        context.Detail(string.Create(CultureInfo.InvariantCulture, $"counting words {counted.TotalSeconds:F2} s, merging {(watch.Elapsed - counted).TotalSeconds:F2} s ({(options.Workers > 0 ? options.Workers : Environment.ProcessorCount)} workers)"));
        context.Write($"next: idrak data tokens {string.Join(' ', files.Select(f => f.Contains(' ') ? $"\"{f}\"" : f))} --tokenizer {output} -o tokens.bin");
        context.WriteJson(new JsonObject
        {
            ["path"] = Path.GetFullPath(output), ["vocabulary"] = tokenizer.VocabularySize, ["merges"] = merges, ["special"] = specials.Count,
            ["files"] = files.Count, ["bytes"] = total, ["seconds"] = Math.Round(watch.Elapsed.TotalSeconds, 3), ["trainer"] = trainer.Name,
        });
        return ExitCodes.Ok;
    }

    // Progress: bytes while counting words, then tokens of the vocabulary while merging.
    private sealed class Reports(ProgressLine line, int vocabulary, Action counted) : IProgress<TokenizerTrainingProgress>
    {
        private bool _merging;

        public void Report(TokenizerTrainingProgress value)
        {
            lock (this)
            {
                if (value.Stage == "merges" && !_merging)
                {
                    _merging = true;
                    counted();
                    line.Total = vocabulary;
                }

                line.Report(value.Done);
            }
        }
    }
}
