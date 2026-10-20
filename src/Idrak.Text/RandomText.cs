// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Text.Abstractions;

namespace Idrak.Text;

/// <summary>What a random text is counted in: each one a document (a page image when drawn).</summary>
public enum RandomTextUnit
{
    /// <summary>Pages of <see cref="RandomTextOptions.PageLines"/> lines.</summary>
    Pages,

    /// <summary>Single lines of <see cref="RandomTextOptions.LineWords"/> words.</summary>
    Lines,

    /// <summary>Single words.</summary>
    Words,
}

/// <summary>How much random text, and how it reads (see <see cref="RandomText"/>).</summary>
public sealed record RandomTextOptions
{
    /// <summary>What <see cref="Count"/> counts.</summary>
    public RandomTextUnit Unit { get; init; } = RandomTextUnit.Pages;

    /// <summary>How many pages, lines or words (each a document).</summary>
    public int Count { get; init; } = 100;

    /// <summary>The seed: the same seed and options give the same text.</summary>
    public int Seed { get; init; }

    /// <summary>The lines of a page, a range.</summary>
    public (int Min, int Max) PageLines { get; init; } = (8, 20);

    /// <summary>The words of a line, a range.</summary>
    public (int Min, int Max) LineWords { get; init; } = (6, 14);

    /// <summary>The share of numbers written in the language's own digits (its pools' "digits"), 0 to 1.</summary>
    public double NativeDigits { get; init; } = 0.3;

    /// <summary>The share of words given marks (its pools' "marks", Arabic's harakat), 0 to 1.</summary>
    public double Diacritics { get; init; }

    /// <summary>How often a phrase ends with punctuation, 0 to 1.</summary>
    public double Punctuation { get; init; } = 0.3;

    internal void Check()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(Count, 1, nameof(Count));
        if (PageLines.Min < 1 || PageLines.Min > PageLines.Max || LineWords.Min < 1 || LineWords.Min > LineWords.Max)
        {
            throw new ArgumentException($"Page lines ({PageLines.Min}..{PageLines.Max}) and line words ({LineWords.Min}..{LineWords.Max}) are ranges MIN..MAX from 1.");
        }

        foreach (var (name, value) in new[] { (nameof(NativeDigits), NativeDigits), (nameof(Diacritics), Diacritics), (nameof(Punctuation), Punctuation) })
        {
            if (value is < 0 or > 1)
            {
                throw new ArgumentOutOfRangeException(name, value, $"{name} is a share from 0 to 1.");
            }
        }
    }
}

/// <summary>What <see cref="RandomText.Write"/> wrote.</summary>
/// <param name="Documents">Documents (pages, lines or words).</param>
/// <param name="Lines">Lines in all.</param>
/// <param name="Words">Words in all.</param>
/// <param name="Characters">Characters in all.</param>
/// <param name="Path">The file.</param>
public sealed record RandomTextResult(int Documents, int Lines, int Words, int Characters, string Path);

/// <summary>
/// Random text for training pages when there is no text to draw (plan 18): documents of random lines from a source
/// (<see cref="RandomTextSources"/>), written as plain text that <see cref="SyntheticPages"/> reads ("text": documents
/// separated by lines of <c>---</c>). Each document is drawn from the seed and its place, so the same options give the
/// same file.
/// </summary>
public static class RandomText
{
    /// <summary>The documents, one string each (lines separated by '\n').</summary>
    public static IEnumerable<string> Documents(IRandomTextSource source, RandomTextOptions options)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        return Iterate(source, options);
    }

    private static IEnumerable<string> Iterate(IRandomTextSource source, RandomTextOptions options)
    {
        for (int i = 0; i < options.Count; i++)
        {
            var random = new Random(SyntheticPages.PageSeed(options.Seed, i, 0x7e47));
            yield return options.Unit switch
            {
                RandomTextUnit.Words => source.Line(random, 1, options),
                RandomTextUnit.Lines => source.Line(random, Between(random, options.LineWords), options),
                _ => string.Join("\n", Enumerable.Range(0, Between(random, options.PageLines)).Select(_ => source.Line(random, Between(random, options.LineWords), options))),
            };
        }
    }

    private static int Between(Random random, (int Min, int Max) range) => random.Next(range.Min, range.Max + 1);

    /// <summary>Writes the documents to <paramref name="path"/> (UTF-8, '\n', documents between lines of <c>---</c>).</summary>
    public static RandomTextResult Write(string path, IRandomTextSource source, RandomTextOptions options, IProgress<int>? progress = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string full = Path.GetFullPath(path);
        if (Path.GetDirectoryName(full) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        int documents = 0, lines = 0, words = 0, characters = 0;
        using (var writer = new StreamWriter(full, append: false, new UTF8Encoding(false)) { NewLine = "\n" })
        {
            foreach (string document in Documents(source, options))
            {
                if (documents > 0)
                {
                    writer.WriteLine("---");
                }

                writer.WriteLine(document);
                documents++;
                foreach (string line in document.Split('\n'))
                {
                    lines++;
                    words += line.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
                    characters += line.Length;
                }

                if (documents % 100 == 0)
                {
                    progress?.Report(documents);
                }
            }
        }

        progress?.Report(documents);
        return new RandomTextResult(documents, lines, words, characters, full);
    }
}

/// <summary>
/// A random text source from word pools: lines made of patterns of pools (a pattern such as <c>["verb", "noun",
/// "adjective"]</c> draws one word of each), with numbers, years, dates and amounts in them, punctuation after some
/// phrases, the language's digits and marks on some. Written as JSON (<see cref="Read"/>): <c>{"name", "summary",
/// "pools": {"noun": [...], ...}, "patterns": [[...], ...], "punctuation": [...], "end": [...], "digits": "٠١٢٣٤٥٦٧٨٩",
/// "marks": [...], "unmarked": "اآى", "years": [1380, 1450]}</c> ("unmarked": letters no mark follows). A pattern's item is a pool's name, a number (<c>#number</c>,
/// <c>#year</c>, <c>#date</c>, <c>#amount</c>) or a word as it is (<c>=هـ</c>).
/// </summary>
public sealed class PooledTextSource : IRandomTextSource
{
    private static readonly string[] Generators = ["#number", "#year", "#date", "#amount"];
    private static readonly string[] Keys = ["name", "summary", "pools", "patterns", "punctuation", "end", "digits", "marks", "unmarked", "years"];

    private readonly Dictionary<string, string[]> _pools;
    private readonly string[][] _patterns;
    private readonly string[] _punctuation, _end, _marks;
    private readonly string _unmarked;
    private readonly string[]? _digits;
    private readonly (int Min, int Max) _years;

    private PooledTextSource(string name, string summary, Dictionary<string, string[]> pools, string[][] patterns, string[] punctuation, string[] end,
        string? digits, string[] marks, string unmarked, (int, int) years)
    {
        Name = name;
        Summary = summary;
        _pools = pools;
        _patterns = patterns;
        _punctuation = punctuation;
        _end = end;
        _digits = digits?.EnumerateRunes().Select(r => r.ToString()).ToArray();
        _marks = marks;
        _unmarked = unmarked;
        _years = years;
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Summary { get; }

    /// <summary>The pools' names.</summary>
    public IReadOnlyCollection<string> Pools => _pools.Keys;

    /// <summary>The source in the JSON file at <paramref name="path"/> (see the class).</summary>
    /// <exception cref="InvalidDataException">A key, pool or pattern item that is not known, or a value of the wrong kind; the message names it.</exception>
    public static PooledTextSource Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var json = JsonNode.Parse(File.ReadAllBytes(path), documentOptions: new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true }) as JsonObject
            ?? throw new InvalidDataException($"{path}: text pools are a JSON object.");
        return FromJson(json, path, Path.GetFileNameWithoutExtension(path));
    }

    /// <summary>A source of single words drawn from <paramref name="words"/> (blank ones left out), any language.</summary>
    public static PooledTextSource FromWords(string name, IEnumerable<string> words, string? summary = null)
    {
        ArgumentNullException.ThrowIfNull(words);
        var list = words.SelectMany(w => w.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToArray();
        if (list.Length == 0)
        {
            throw new InvalidDataException($"{name}: the word list has no words.");
        }

        return new PooledTextSource(name, summary ?? $"words drawn from a list of {list.Length:N0}", new Dictionary<string, string[]> { ["word"] = list }, [["word"]],
            [], [], null, [], "", (1990, 2030));
    }

    internal static PooledTextSource Embedded(string resource)
    {
        using var stream = typeof(PooledTextSource).Assembly.GetManifestResourceStream("Idrak.Text." + resource)
            ?? throw new InvalidOperationException($"The library's text pools {resource} are missing from Idrak.Text.");
        return FromJson((JsonObject)JsonNode.Parse(stream)!, resource, resource);
    }

    /// <summary>The source written in <paramref name="json"/> (see the class).</summary>
    public static PooledTextSource FromJson(JsonObject json, string where = "text pools", string? name = null)
    {
        ArgumentNullException.ThrowIfNull(json);
        foreach (var (key, _) in json)
        {
            if (!Keys.Contains(key, StringComparer.Ordinal))
            {
                throw new InvalidDataException($"{where}: unknown key '{key}' (known: {string.Join(", ", Keys)}).");
            }
        }

        string[] Texts(JsonNode? node, string what) => node switch
        {
            null => [],
            JsonArray a => [.. a.Select(x => x is JsonValue v && v.GetValueKind() == JsonValueKind.String && v.GetValue<string>().Length > 0 ? v.GetValue<string>()
                : throw new InvalidDataException($"{where}: {what} holds text, not {x?.ToJsonString()}."))],
            _ => throw new InvalidDataException($"{where}: {what} is a list of text, not {node.ToJsonString()}."),
        };

        var pools = new Dictionary<string, string[]>(StringComparer.Ordinal);
        if (json["pools"] is not JsonObject poolObject || poolObject.Count == 0)
        {
            throw new InvalidDataException($"{where}: \"pools\" is an object of named word lists.");
        }

        foreach (var (pool, words) in poolObject)
        {
            pools[pool] = Texts(words, $"pool '{pool}'");
            if (pools[pool].Length == 0)
            {
                throw new InvalidDataException($"{where}: pool '{pool}' has no words.");
            }
        }

        var patterns = json["patterns"] is JsonArray p ? p.Select((x, i) => Texts(x, $"pattern {i + 1}")).ToArray() : [.. pools.Keys.Select(k => new[] { k })];
        foreach (string item in patterns.SelectMany(x => x))
        {
            if (!item.StartsWith('=') && !Generators.Contains(item) && !pools.ContainsKey(item))
            {
                throw new InvalidDataException($"{where}: a pattern names '{item}', which is no pool ({string.Join(", ", pools.Keys)}), number ({string.Join(", ", Generators)}) or =word.");
            }
        }

        if (patterns.Length == 0 || patterns.Any(x => x.Length == 0))
        {
            throw new InvalidDataException($"{where}: every pattern has at least one item.");
        }

        string? digits = (string?)json["digits"];
        if (digits is not null && digits.EnumerateRunes().Count() != 10)
        {
            throw new InvalidDataException($"{where}: digits are the ten digits 0 to 9 of the language, not '{digits}'.");
        }

        var years = (1990, 2030);
        if (json["years"] is JsonArray y)
        {
            years = y.Count == 2 && y[0] is JsonValue a && a.TryGetValue(out int from) && y[1] is JsonValue b && b.TryGetValue(out int to) && from <= to && from > 0
                ? (from, to)
                : throw new InvalidDataException($"{where}: years are [FIRST, LAST], not {y.ToJsonString()}.");
        }

        return new PooledTextSource((string?)json["name"] ?? name ?? where, (string?)json["summary"] ?? $"words from the pools of {where}", pools, patterns,
            Texts(json["punctuation"], "punctuation"), Texts(json["end"], "end"), digits, Texts(json["marks"], "marks"), (string?)json["unmarked"] ?? "", years);
    }

    /// <inheritdoc/>
    public string Line(Random random, int words, RandomTextOptions options)
    {
        ArgumentNullException.ThrowIfNull(random);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(words, 1);
        var tokens = new List<string>(words + 8);
        while (tokens.Count < words)
        {
            foreach (string item in _patterns[random.Next(_patterns.Length)])
            {
                tokens.Add(Token(item, random, options));
            }

            if (_punctuation.Length > 0 && random.NextDouble() < options.Punctuation)
            {
                tokens[^1] += _punctuation[random.Next(_punctuation.Length)];
            }
        }

        tokens.RemoveRange(words, tokens.Count - words);
        string last = tokens[^1];
        if (_punctuation.Concat(_end).FirstOrDefault(p => last.Length > p.Length && last.EndsWith(p, StringComparison.Ordinal)) is { } mark)
        {
            last = last[..^mark.Length];
        }

        tokens[^1] = words > 1 && _end.Length > 0 && random.NextDouble() < options.Punctuation ? last + _end[random.Next(_end.Length)] : last;
        return string.Join(' ', tokens);
    }

    private string Token(string item, Random random, RandomTextOptions options)
    {
        if (item.StartsWith('='))
        {
            return item[1..];
        }

        string token = item switch
        {
            "#number" => random.Next(1, (int)Math.Pow(10, random.Next(1, 6))).ToString(CultureInfo.InvariantCulture),
            "#year" => random.Next(_years.Min, _years.Max + 1).ToString(CultureInfo.InvariantCulture),
            "#date" => string.Create(CultureInfo.InvariantCulture, $"{random.Next(1, 31)}/{random.Next(1, 13)}/{random.Next(_years.Min, _years.Max + 1)}"),
            "#amount" => (random.Next(1, 10000) * (int)Math.Pow(10, random.Next(0, 3))).ToString(CultureInfo.InvariantCulture),
            _ => _pools[item][random.Next(_pools[item].Length)],
        };

        if (item.StartsWith('#'))
        {
            return _digits is not null && random.NextDouble() < options.NativeDigits ? Native(token) : token;
        }

        return _marks.Length > 0 && random.NextDouble() < options.Diacritics ? Marked(token, random) : token;
    }

    private string Native(string number)
    {
        var sb = new StringBuilder(number.Length);
        foreach (char c in number)
        {
            sb.Append(c is >= '0' and <= '9' ? _digits![c - '0'] : c.ToString());
        }

        return sb.ToString();
    }

    // Marks after some of the word's letters (none on "ال", none after the pools' "unmarked" letters, none twice in a row).
    private string Marked(string word, Random random)
    {
        var sb = new StringBuilder(word.Length * 2);
        int start = word.StartsWith("ال", StringComparison.Ordinal) ? 2 : 0;
        for (int i = 0; i < word.Length; i++)
        {
            sb.Append(word[i]);
            if (i >= start && char.IsLetter(word[i]) && !_unmarked.Contains(word[i]) && random.NextDouble() < 0.5)
            {
                sb.Append(_marks[random.Next(_marks.Length)]);
            }
        }

        return sb.ToString();
    }
}
