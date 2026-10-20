// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Text;
using Idrak.Text.Abstractions;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data text -n N --unit pages|lines|words -l ar -o text.txt</c>: random text in a language (<see cref="RandomText"/>)
/// for drawing training pages when there is no text to draw, written as the plain text <c>idrak data synth</c> reads.
/// </summary>
internal sealed class DataTextCommand : Command
{
    public override string Name => "data text";

    public override string Summary => "Random text in a language (pages, lines or words) to draw training pages from";

    public override string Usage => """
        -n N [--unit pages|lines|words] [-l CODE] -o OUT.txt [options]

        Writes N documents of random text, each drawn later as one page: pages of several lines, single lines, or single
        words. The text has the words, names, numbers, dates and punctuation of official letters, not meaningful
        sentences. The file is plain text with the documents between lines of ---, which idrak data synth reads.

        Options:
          -n, --count N           how many pages, lines or words (100)
              --unit U            pages (the default), lines or words: what N counts and what one page shows
          -l, --language CODE     the text's language: ar (the default) and en are built in; any other with --pools or
                                  --words (--lang is the language of idrak's own messages)
              --pools FILE        word pools for --language (JSON: "pools", "patterns", "punctuation", "end", "digits", "marks", "years")
              --words FILE        a word list or any text in --language: its words are drawn at random
              --page-lines A..B   the lines of a page (8..20)
              --line-words A..B   the words of a line (6..14)
              --native-digits P   the share of numbers in the language's own digits (0.3; Arabic: ٠١٢٣٤٥٦٧٨٩)
              --diacritics P      the share of words with marks (0; Arabic: harakat)
              --punctuation P     how often a phrase ends with punctuation (0.3)
          -o, --out FILE          the text file to write (.txt)
              --seed N            the same seed and options give the same text (0)

        Examples:
          idrak data text -n 500 --unit pages -o pages.txt
          idrak data text -n 5000 --unit lines --line-words 4..10 -o lines.txt
          idrak data text -n 2000 --unit words -l en -o words.txt
          idrak data text -n 300 -l fr --words french-book.txt -o fr.txt
          idrak data synth pages.txt -o synthetic/pages.jsonl --fonts DIR --answer-key full_text_extract
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--count", "--unit", "--language", "--pools", "--words", "--page-lines", "--line-words", "--native-digits", "--diacritics", "--punctuation", "--out"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-n"] = "--count", ["-o"] = "--out", ["-l"] = "--language" };

    public override int Run(CommandContext context)
    {
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the text to write (.txt).");
        string language = context.Option("--language") ?? "ar";
        IRandomTextSource source;
        RandomTextOptions options;
        try
        {
            source = context.Option("--pools") is { } pools ? Rename(PooledTextSource.Read(pools), language)
                : context.Option("--words") is { } words ? PooledTextSource.FromWords(language, File.ReadLines(words, Encoding.UTF8))
                : RandomTextSources.Get(language);
            var defaults = new RandomTextOptions();
            options = defaults with
            {
                Count = context.IntOption("--count", defaults.Count),
                Unit = context.Option("--unit") is not { } unit ? defaults.Unit
                    : Enum.TryParse<RandomTextUnit>(unit, ignoreCase: true, out var u) && !int.TryParse(unit, out _) ? u
                    : throw new UsageException($"--unit is pages, lines or words, not '{unit}'."),
                PageLines = Range(context, "--page-lines", defaults.PageLines),
                LineWords = Range(context, "--line-words", defaults.LineWords),
                NativeDigits = Share(context, "--native-digits", defaults.NativeDigits),
                Diacritics = Share(context, "--diacritics", defaults.Diacritics),
                Punctuation = Share(context, "--punctuation", defaults.Punctuation),
                Seed = context.Seed ?? 0,
            };
            _ = RandomText.Documents(source, options);     // checks the options now
        }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException or FileNotFoundException or DirectoryNotFoundException)
        {
            throw new UsageException(e.Message);
        }

        RandomTextResult result;
        using (var line = new ProgressLine(context, $"writing {options.Unit.ToString().ToLowerInvariant()} of {source.Name} text", total: options.Count, unit: ProgressUnit.Items))
        {
            result = RandomText.Write(output, source, options, new Reports(line));
            line.Report(result.Documents);
            line.Finish();
        }

        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{result.Documents:N0} {options.Unit.ToString().ToLowerInvariant()} of {source.Name} text ({result.Lines:N0} lines, {result.Words:N0} words) in {result.Path}"));
        context.Write($"next: idrak data synth \"{result.Path}\" -o synthetic/pages.jsonl --fonts DIR --answer-key full_text_extract");
        context.WriteJson(new JsonObject
        {
            ["path"] = result.Path, ["language"] = source.Name, ["unit"] = options.Unit.ToString().ToLowerInvariant(),
            ["documents"] = result.Documents, ["lines"] = result.Lines, ["words"] = result.Words, ["characters"] = result.Characters,
        });
        return ExitCodes.Ok;
    }

    private sealed class Reports(ProgressLine line) : IProgress<int>
    {
        public void Report(int value) => line.Report(value);
    }

    // A pools file read for --language: its words, under the language asked for.
    private static IRandomTextSource Rename(PooledTextSource source, string language) =>
        source.Name.Equals(language, StringComparison.OrdinalIgnoreCase) ? source : new Named(language, source);

    private sealed class Named(string name, IRandomTextSource inner) : IRandomTextSource
    {
        public string Name => name;

        public string Summary => inner.Summary;

        public string Line(Random random, int words, RandomTextOptions options) => inner.Line(random, words, options);
    }

    private static (int, int) Range(CommandContext context, string option, (int, int) otherwise)
    {
        if (context.Option(option) is not { } text)
        {
            return otherwise;
        }

        int dots = text.IndexOf("..", StringComparison.Ordinal);
        string a = dots < 0 ? text : text[..dots], b = dots < 0 ? text : text[(dots + 2)..];
        return int.TryParse(a, NumberStyles.Integer, CultureInfo.InvariantCulture, out int min) && int.TryParse(b, NumberStyles.Integer, CultureInfo.InvariantCulture, out int max)
            ? (min, max)
            : throw new UsageException($"{option} is a number or a range MIN..MAX, not '{text}'.");
    }

    private static double Share(CommandContext context, string option, double otherwise) => context.Option(option) is not { } text ? otherwise
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v
        : throw new UsageException($"{option} is a share from 0 to 1, not '{text}'.");
}
