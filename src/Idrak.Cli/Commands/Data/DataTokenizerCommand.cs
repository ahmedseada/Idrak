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
        --from DATA... --vocab N -o tokenizer.json [options]

        Learns a tokenizer from UTF-8 text files (each line, with its line end, one sequence), as Hugging Face's
        BpeTrainer learns one: the same vocabulary and merges from the same files. The special tokens come first. The
        file is a tokenizer.json that idrak create, Hugging Face and every idrak command read. --from learns it from the
        texts of transcribed speech instead, read as the speech trainers read them: transcripts files (JSON lines of
        {"audio", "text"}, a manifest of data audio) or folders and zips of name.wav beside name.txt, one text a sequence.

        Arguments:
          FILE  text files to learn from (.txt, any UTF-8 text); none with --from

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
              --line-ends E        lf (\r\n read as \n, so a file trains the same on every system; the default) or keep
                                   (the bytes as they are, as Hugging Face's train_from_files reads them)
              --from DATA          transcribed speech whose texts it learns from (repeatable; with FILE too)
              --source KIND        the speech data's kind (transcripts or audio-folder; told from the paths)
              --audio DIR|ZIP      where a transcripts file's sounds are (default: beside it; rows whose sound is
                                   missing are left out, as the trainers leave them)
          -o, --out FILE           the tokenizer.json to write

        Examples:
          idrak data text -n 20000 -o corpus.txt
          idrak data tokenizer corpus.txt --vocab 32000 -o tokenizer.json
          idrak data tokenizer books/*.txt --vocab 64000 --special "<s>" --special "</s>" -o tokenizer.json
          idrak data tokens corpus.txt --tokenizer tokenizer.json -o tokens.bin
          idrak data tokenizer --from pieces/ --vocab 128 --pre-tokenizer metaspace --no-special -o speech-tokenizer.json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--vocab", "--min-frequency", "--special", "--pre-tokenizer", "--limit-alphabet", "--trainer", "--workers", "--line-ends", "--out", "--from", "--source", "--audio"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--no-special", "--prefix-space"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the tokenizer.json to write.");
        var files = context.Positional.Count > 0 ? TextFiles.Expand(context.Positional) : [];
        string? spoken = context.Options("--from") is { Count: > 0 } speech ? SpeechTexts(context, speech) : null;
        try
        {
            if (spoken is not null)
            {
                files = [.. files, spoken];
            }

            if (files.Count == 0)
            {
                throw new UsageException("Give the text files to learn from, or --from DATA (transcribed speech).");
            }

            return Train(context, output, files, spoken);
        }
        finally
        {
            if (spoken is not null)
            {
                File.Delete(spoken);
            }
        }
    }

    // The texts of transcribed speech, one a line, in a temporary file (deleted after the training): the utterances the
    // speech sources read (SpeechSources: transcripts files, folders and zips of name.wav + name.txt).
    private static string SpeechTexts(CommandContext context, IReadOnlyList<string> data)
    {
        var paths = data.Select(Path.GetFullPath).ToList();
        var source = new Idrak.Audio.Abstractions.SpeechSource(context.Option("--source") ?? Idrak.Cli.Commands.Train.SpeechPretraining.Kind(paths), paths)
        {
            Audio = context.Option("--audio") is { } audio ? Path.GetFullPath(audio) : null,
            SkipMissing = true,
        };
        string path = Path.Combine(Path.GetTempPath(), $"idrak-speech-texts-{Guid.NewGuid():N}.txt");
        try
        {
            using var reader = Idrak.Audio.Abstractions.SpeechSources.Open(source);
            int empty = 0;
            using (var writer = new StreamWriter(path, append: false, new System.Text.UTF8Encoding(false)))
            {
                for (int i = 0; i < reader.Count; i++)
                {
                    string text = reader[i].Text.Trim();
                    if (text.Length == 0)
                    {
                        empty++;
                        continue;
                    }

                    writer.Write(text.ReplaceLineEndings(" "));
                    writer.Write('\n');
                }
            }

            foreach (string note in reader.Notes)
            {
                context.Write($"note: {note}");
            }

            context.Write(string.Create(CultureInfo.InvariantCulture, $"{reader.Count - empty:N0} texts from {string.Join(", ", data)} ({source.Kind})")
                + (empty > 0 ? string.Create(CultureInfo.InvariantCulture, $"; {empty:N0} empty, left out") : ""));
            return path;
        }
        catch (Exception e) when (e is InvalidDataException or FileNotFoundException or NotSupportedException or ArgumentException)
        {
            File.Delete(path);
            throw new UsageException(e.Message);
        }
    }

    private static int Train(CommandContext context, string output, IReadOnlyList<string> files, string? spoken)
    {

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
            LineEnds = context.Option("--line-ends") ?? defaults.LineEnds,
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
        IPretrainedTokenizer tokenizer;
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

            // Merging stops early when no pair is left (a text small for the vocabulary): say so, not a bar stuck short of 100%.
            line.Finish(line.Label == "merging" && line.Total is { } wanted && line.Done < wanted
                ? string.Create(CultureInfo.InvariantCulture, $"merging  {line.Done:N0} of {wanted:N0} tokens: no pair left to merge (the text is small for --vocab {wanted})")
                : null);
        }

        tokenizer.Save(output);
        int merges = tokenizer.ToJson()["model"]?["merges"]?.AsArray().Count ?? 0;
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{tokenizer.VocabularySize:N0} tokens ({specials.Count} special, {merges:N0} merges) learned from {files.Count} file{(files.Count == 1 ? "" : "s")} ({total / 1048576.0:F1} MB) in {watch.Elapsed.TotalSeconds:F1} s, written to {output}"));
        context.Detail(string.Create(CultureInfo.InvariantCulture, $"counting words {counted.TotalSeconds:F2} s, merging {(watch.Elapsed - counted).TotalSeconds:F2} s ({(options.Workers > 0 ? options.Workers : Environment.ProcessorCount)} workers)"));
        if (spoken is null)
        {
            context.Write($"next: idrak data tokens {string.Join(' ', files.Select(f => f.Contains(' ') ? $"\"{f}\"" : f))} --tokenizer {output} -o tokens.bin");
        }

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
                    line.Label = "merging";                                   // tokens of the vocabulary from here, not bytes
                    line.Unit = ProgressUnit.Items;
                    line.Total = vocabulary;
                }

                line.Report(value.Done);
            }
        }
    }
}
