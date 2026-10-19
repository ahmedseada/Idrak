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
    // The special tokens a created model's template and a vision family use, when none are given (bpe and unigram).
    internal static readonly string[] DefaultSpecials = ["<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<start_of_image>", "<image>", "<end_of_image>"];

    // BERT's special tokens, the wordpiece trainer's when none are given.
    internal static readonly string[] WordPieceSpecials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

    // The built-in trainers --model names.
    private static readonly string[] Models = [TokenizerTrainers.Bpe, TokenizerTrainers.Unigram, TokenizerTrainers.WordPiece];

    public override string Name => "data tokenizer";

    public override string Summary => "Learn a tokenizer from text files, for a model made from scratch";

    public override string Usage => """
        FILE... --vocab N -o tokenizer.json [options]
        --from DATA... --vocab N -o tokenizer.json [options]

        Learns a tokenizer from UTF-8 text files, one of three kinds (--model):
          bpe        byte-pair encoding, as Hugging Face's BpeTrainer learns it: the same vocabulary and merges from the
                     same files (each line, with its line end, one sequence)
          unigram    SentencePiece's unigram language model, by SentencePiece's procedure (frequent substrings found
                     through a suffix array, EM steps, pruning by loss); writes tokenizer.model beside tokenizer.json
          wordpiece  BERT's WordPiece: by frequency (Hugging Face's WordPieceTrainer, its ties in a fixed order) or by
                     likelihood (--scoring)
        The special tokens come first. The file is a tokenizer.json that idrak create, Hugging Face and every idrak
        command read. --from learns it from the texts of transcribed speech instead, read as the speech trainers read
        them: transcripts files (JSON lines of {"audio", "text"}, a manifest of data audio) or folders and zips of
        name.wav beside name.txt, one text a sequence.

        Arguments:
          FILE  text files to learn from (.txt, any UTF-8 text); none with --from

        Options:
              --model M            bpe (the default), unigram or wordpiece
              --trainer NAME       a plug-in's trainer (-P), instead of --model
              --vocab N            the vocabulary to reach: special tokens, the alphabet and the learned tokens (32000)
              --min-frequency N    bpe and wordpiece: a pair seen fewer times is never merged (2)
              --special TOKEN      a special token (repeat for several); replaces the defaults: <pad> <s> </s>
                                   <|im_start|> <|im_end|> <start_of_image> <image> <end_of_image> (bpe, unigram);
                                   [PAD] [UNK] [CLS] [SEP] [MASK] (wordpiece)
              --no-special         no special tokens at all
              --unk TOKEN          unigram and wordpiece: the unknown token (<unk>, [UNK]); after the special tokens
                                   when they do not hold it
              --workers N          threads reading and learning (one per processor); one and many learn the same
              --line-ends E        lf (\r\n read as \n, so a file trains the same on every system; the default) or keep
                                   (the bytes as they are, as Hugging Face's train_from_files reads them)
          bpe:
              --pre-tokenizer P    bytelevel (bytes as characters: every text encodes, the default) or metaspace (spaces
                                   as ▁ starting a word, SentencePiece's way)
              --prefix-space       byte level: a space before each text that does not start with one
              --limit-alphabet N   keep at most N characters in the alphabet (the most frequent; wordpiece too)
          unigram:
              --character-coverage S  the share of the text's characters the alphabet covers (0.9995); the rest are
                                   unknown, or bytes with --byte-fallback
              --byte-fallback      the 256 byte pieces <0x00>..<0xFF>, so every text encodes without the unknown token
              --max-piece N        the longest piece, in characters (16)
              --seed-size N        frequent substrings that seed the vocabulary before pruning (1000000)
              --shrinking S        the share of the pieces each pruning round keeps (0.75)
              --em-steps N         EM steps before each pruning round (2)
          unigram and wordpiece:
              --split-digits       every digit a token of its own
              --arabic LIST        Arabic marks removed before learning and encoding, written into the tokenizer:
                                   tatweel, diacritics, or tatweel,diacritics
          wordpiece:
              --scoring S          frequency (the default) or likelihood
              --lowercase          lowercase the text (BERT's uncased models)
              --strip-accents      remove accents (the default with --lowercase); --keep-accents keeps them
          speech:
              --from DATA          transcribed speech whose texts it learns from (repeatable; with FILE too)
              --source KIND        the speech data's kind (transcripts or audio-folder; told from the paths)
              --audio DIR|ZIP      where a transcripts file's sounds are (default: beside it; rows whose sound is
                                   missing are left out, as the trainers leave them)
          -o, --out FILE           the tokenizer.json to write (unigram writes tokenizer.model beside it)

        Examples:
          idrak data text -n 20000 -o corpus.txt
          idrak data tokenizer corpus.txt --vocab 32000 -o tokenizer.json
          idrak data tokenizer corpus.txt --model unigram --vocab 8000 --byte-fallback -o out/tokenizer.json
          idrak data tokenizer arabic.txt --model wordpiece --vocab 30000 --arabic tatweel,diacritics -o tokenizer.json
          idrak data tokenizer books/*.txt --vocab 64000 --special "<s>" --special "</s>" -o tokenizer.json
          idrak data tokens corpus.txt --tokenizer tokenizer.json -o tokens.bin
          idrak data tokenizer --from pieces/ --vocab 128 --pre-tokenizer metaspace --no-special -o speech-tokenizer.json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--vocab", "--min-frequency", "--special", "--pre-tokenizer", "--limit-alphabet", "--trainer", "--workers", "--line-ends", "--out", "--from", "--source", "--audio",
        "--model", "--unk", "--character-coverage", "--max-piece", "--seed-size", "--shrinking", "--em-steps", "--arabic", "--scoring",
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--no-special", "--prefix-space", "--byte-fallback", "--split-digits", "--lowercase", "--strip-accents", "--keep-accents"];

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
        string? model = context.Option("--model")?.Trim().ToLowerInvariant();
        string? plugin = context.Option("--trainer");
        if (model is not null && plugin is not null)
        {
            throw new UsageException("Give --model (a built-in trainer: bpe, unigram or wordpiece) or --trainer (a plug-in's), not both.");
        }

        if (model is not null && !Models.Contains(model))
        {
            throw new UsageException($"--model is bpe, unigram or wordpiece, not '{model}'.");
        }

        if (context.Flag("--strip-accents") && context.Flag("--keep-accents"))
        {
            throw new UsageException("Give --strip-accents or --keep-accents, not both.");
        }

        string name = model ?? plugin ?? TokenizerTrainers.Bpe;
        var defaults = new TokenizerTrainingOptions();
        var specials = context.Flag("--no-special") ? []
            : context.Options("--special") is { Count: > 0 } given ? given
            : name.Equals(TokenizerTrainers.WordPiece, StringComparison.OrdinalIgnoreCase) ? WordPieceSpecials : DefaultSpecials;
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
            UnknownToken = context.Option("--unk"),
            CharacterCoverage = Number(context, "--character-coverage", defaults.CharacterCoverage),
            ByteFallback = context.Flag("--byte-fallback"),
            MaxPieceLength = context.IntOption("--max-piece", defaults.MaxPieceLength),
            SeedSize = context.IntOption("--seed-size", defaults.SeedSize),
            ShrinkingFactor = Number(context, "--shrinking", defaults.ShrinkingFactor),
            EmIterations = context.IntOption("--em-steps", defaults.EmIterations),
            SplitDigits = context.Flag("--split-digits"),
            Lowercase = context.Flag("--lowercase"),
            StripAccents = context.Flag("--strip-accents") ? true : context.Flag("--keep-accents") ? false : null,
            Scoring = context.Option("--scoring") ?? defaults.Scoring,
            Arabic = context.Option("--arabic") is { } arabic ? [.. arabic.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)] : [],
        };
        ITokenizerTrainer trainer;
        try
        {
            trainer = TokenizerTrainers.Get(name);
        }
        catch (NotSupportedException e)
        {
            context.Error(Messages.T("{0} Load its plug-in with -P (--plugin).", e.Message));
            return ExitCodes.Failed;
        }

        // Bad settings are refused before any file is read, named as the command line names them.
        try
        {
            options.Validate(trainer.Name);
        }
        catch (ArgumentException e)
        {
            throw new UsageException(OptionNames(e.Message));
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
                throw new UsageException(OptionNames(e.Message));
            }

            // Merging stops early when no pair is left (a text small for the vocabulary): say so, not a bar stuck short of 100%.
            line.Finish(line.Label == "merging" && line.Total is { } wanted && line.Done < wanted
                ? string.Create(CultureInfo.InvariantCulture, $"merging  {line.Done:N0} of {wanted:N0} tokens: no pair left to merge (the text is small for --vocab {wanted})")
                : null);
        }

        tokenizer.Save(output);
        var written = new List<string>();
        if (tokenizer is ITokenizerCompanionFiles companions)
        {
            string folder = Path.GetDirectoryName(Path.GetFullPath(output)) ?? ".";
            foreach (string companion in companions.CompanionFiles)
            {
                string path = Path.Combine(folder, companion);
                companions.SaveCompanion(companion, path);
                written.Add(path);
            }
        }

        int merges = tokenizer.ToJson()["model"]?["merges"]?.AsArray().Count ?? 0;
        string learned = merges > 0 ? string.Create(CultureInfo.InvariantCulture, $", {merges:N0} merges") : "";
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{tokenizer.VocabularySize:N0} tokens ({specials.Count} special{learned}), a {tokenizer.Model} tokenizer learned from {files.Count} file{(files.Count == 1 ? "" : "s")} ({total / 1048576.0:F1} MB) in {watch.Elapsed.TotalSeconds:F1} s, written to {output}")
            + (written.Count > 0 ? $" and {string.Join(", ", written)}" : ""));
        context.Detail(string.Create(CultureInfo.InvariantCulture, $"counting words {counted.TotalSeconds:F2} s, learning {(watch.Elapsed - counted).TotalSeconds:F2} s ({(options.Workers > 0 ? options.Workers : Environment.ProcessorCount)} workers)"));
        if (spoken is null)
        {
            context.Write($"next: idrak data tokens {string.Join(' ', files.Select(f => f.Contains(' ') ? $"\"{f}\"" : f))} --tokenizer {output} -o tokens.bin");
        }

        context.WriteJson(new JsonObject
        {
            ["path"] = Path.GetFullPath(output), ["vocabulary"] = tokenizer.VocabularySize, ["merges"] = merges, ["special"] = specials.Count,
            ["files"] = files.Count, ["bytes"] = total, ["seconds"] = Math.Round(watch.Elapsed.TotalSeconds, 3), ["trainer"] = trainer.Name,
            ["model"] = tokenizer.Model, ["companions"] = new JsonArray([.. written.Select(w => (JsonNode)JsonValue.Create(w))]),
        });
        return ExitCodes.Ok;
    }

    // A number option (invariant culture), or the fallback when absent.
    private static double Number(CommandContext context, string option, double fallback) => context.Option(option) is not { } text ? fallback
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ? value
        : throw new UsageException($"{option} needs a number, not '{text}'.");

    // The library names its settings; the command line names its options.
    private static string OptionNames(string message)
    {
        foreach (var (setting, option) in new[]
                 {
                     ("VocabularySize", "--vocab"), ("MinFrequency", "--min-frequency"), ("CharacterCoverage", "--character-coverage"),
                     ("MaxPieceLength", "--max-piece"), ("SeedSize", "--seed-size"), ("ShrinkingFactor", "--shrinking"), ("EmIterations", "--em-steps"),
                     ("Scoring", "--scoring"), ("Arabic", "--arabic"), ("UnknownToken", "--unk"), ("LineEnds", "--line-ends"), ("PreTokenizer", "--pre-tokenizer"),
                     ("LimitAlphabet", "--limit-alphabet"), ("Workers", "--workers"), ("SpecialTokens", "--special"),
                 })
        {
            message = message.Replace(setting, option, StringComparison.Ordinal);
        }

        return message.Replace(" (Parameter 'options')", "", StringComparison.Ordinal).Replace(" (Parameter 'files')", "", StringComparison.Ordinal);
    }

    // Progress, one line a stage: bytes while counting words; then tokens of the vocabulary while merging (bpe, wordpiece),
    // or suffix-tree nodes while seeding, then EM steps, the pruning rounds in the label (unigram).
    private sealed class Reports(ProgressLine line, int vocabulary, Action counted) : IProgress<TokenizerTrainingProgress>
    {
        private string _stage = "words";

        public void Report(TokenizerTrainingProgress value)
        {
            lock (this)
            {
                if (value.Stage == "prune")
                {
                    // Pruning alternates with the EM steps: the EM line says how many rounds are done.
                    line.Label = string.Create(CultureInfo.InvariantCulture, $"EM steps, {value.Done} of {value.Total} pruning rounds");
                    line.Report(line.Done);
                    return;
                }

                if (value.Stage != _stage)
                {
                    if (_stage == "words")
                    {
                        counted();
                    }

                    _stage = value.Stage;
                    var (label, total) = value.Stage switch
                    {
                        "merges" => ("merging", (long?)vocabulary),
                        "seeds" => ("seeding (suffix-tree nodes)", value.Total),
                        "em" => ("EM steps", value.Total),
                        _ => (value.Stage, value.Total),
                    };
                    line.NextStage(label, total, ProgressUnit.Items);
                }

                if (value.Stage != "merges" && value.Total > 0)
                {
                    line.Total = value.Total;
                }

                line.Report(value.Done);
            }
        }
    }
}
