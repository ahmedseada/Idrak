// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace Idrak.Abstraction.Text;

/// <summary>
/// Habits of a hand writing text (a hamza left off, a final letter written as a similar one): a text varied as a person
/// writes it, for pages drawn from text (the label then says what is drawn, or keeps the source's spelling). Registered
/// in <see cref="TextHabits"/> under its <see cref="IKind.Name"/>; thread-safe (the randomness is the caller's).
/// </summary>
public interface ITextHabits : IKind
{
    /// <summary>The habits' names, each with its own chance (0 to 1).</summary>
    IReadOnlyDictionary<string, double> Chances { get; }

    /// <summary>
    /// <paramref name="text"/> with each habit applied where it holds, each place at the habit's chance:
    /// <paramref name="chance"/>'s answer for the habit's name when it gives one, else the habit's own. Numbers are drawn
    /// from <paramref name="random"/> only where a habit could apply, in text order, so the same text and generator give
    /// the same result, and a text no habit holds for draws nothing.
    /// </summary>
    string Apply(string text, Random random, Func<string, double?>? chance = null);
}

/// <summary>
/// The text habits, by name (ignoring case). The library's engine is <see cref="RuleTextHabits"/> (rules as data: what is
/// written for what, where in a word, at what chance); rule sets ship as data with a language's package (Idrak.Text:
/// "arabic-handwriting"). Register another with <see cref="Register"/>; one under a library name shadows the library's,
/// guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class TextHabits
{
    private static readonly KindTable<ITextHabits> Table = new(nameof(TextHabits), "Text habits",
        "add it with TextHabits.Register (RuleTextHabits.Read reads a rules file), or reference Idrak.Text", (slot, app, library) => new GuardedHabits(slot, app, library),
        StringComparer.OrdinalIgnoreCase);

    static TextHabits() => LibraryDefaults.Ensure(typeof(TextHabits));

    /// <summary>Registers <paramref name="habits"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITextHabits habits) => Table.Register(habits, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's habits <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The habits registered as <paramref name="name"/>, or null.</summary>
    public static ITextHabits? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The habits registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ITextHabits Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's habits <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITextHabits? Default(string name) => Table.Default(name);

    /// <summary>Who registered the habits <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's habits <paramref name="name"/> fail.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered set as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    // An app's habits over the library's: a failed call is answered by the library's (from the same generator state), or
    // both are compared on a sample.
    private sealed class GuardedHabits(Slot slot, ITextHabits app, ITextHabits library) : ITextHabits
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlyDictionary<string, double> Chances => app.Chances;

        public string Apply(string text, Random random, Func<string, double?>? chance = null)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                // Both from generators of one seed drawn from the caller's (a compared call draws one number more).
                var run = slot.Shadow();
                int seed = random.Next();
                var answer = library.Apply(text, new Random(seed), chance);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        run.Done(Comparisons.Exact(answer, app.Apply(text, new Random(seed), chance)));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Apply(text, random, chance);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Apply(text, random, chance);
            }
        }
    }
}

/// <summary>Where in a word a habit holds: at its start, inside it, at its end, or as the whole word.</summary>
[Flags]
public enum TextHabitPlace
{
    /// <summary>At a word's start (more of the word follows).</summary>
    Initial = 1,

    /// <summary>Inside a word.</summary>
    Medial = 2,

    /// <summary>At a word's end (more of the word comes before).</summary>
    Final = 4,

    /// <summary>The whole word.</summary>
    Isolated = 8,

    /// <summary>Anywhere.</summary>
    Any = Initial | Medial | Final | Isolated,
}

/// <summary>One habit of a <see cref="RuleTextHabits"/>: texts written as another where it holds, at a chance.</summary>
/// <param name="Name">The habit's name (the key of its chance in a page's settings).</param>
/// <param name="From">The texts it replaces (any of them).</param>
/// <param name="To">What is written instead (empty: left out).</param>
/// <param name="Where">Where in a word it holds.</param>
/// <param name="Chance">Its chance at each place it holds (0 to 1).</param>
public sealed record TextHabitRule(string Name, IReadOnlyList<string> From, string To, TextHabitPlace Where, double Chance);

/// <summary>
/// The library's text habits: rules as data, read from JSON ({"name", "summary", "rules": [{"name", "from" (a text or a
/// list), "to", "where" ("any", "initial", "medial", "final", "isolated" or a list), "chance" (0 to 1; 0 when not
/// given)}]}). A word is a run of letters and their marks (Unicode categories, no language's own): a place is initial at
/// a word's start, final at its end (only marks after it), isolated when it is the whole word, medial otherwise. At each
/// position the first rule whose text is there and whose place holds is tried at its chance; the text moves past what it
/// replaced, or past one character.
/// </summary>
public sealed class RuleTextHabits : ITextHabits
{
    private static readonly string[] TopKeys = ["name", "summary", "rules"];
    private static readonly string[] RuleKeys = ["name", "from", "to", "where", "chance"];
    private static readonly string[] Places = ["any", "initial", "medial", "final", "isolated"];

    /// <summary>Habits of <paramref name="rules"/>, in the order they are tried.</summary>
    public RuleTextHabits(string name, string summary, IReadOnlyList<TextHabitRule> rules)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(rules);
        Name = name;
        Summary = summary ?? name;
        Rules = rules;
        Chances = rules.GroupBy(r => r.Name, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.First().Chance, StringComparer.Ordinal);
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Summary { get; }

    /// <summary>The rules, in the order they are tried.</summary>
    public IReadOnlyList<TextHabitRule> Rules { get; }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, double> Chances { get; }

    /// <summary>Reads a rules file (JSON).</summary>
    /// <exception cref="InvalidDataException">The file is not a valid rules file: the message names it, the key and what is valid.</exception>
    public static RuleTextHabits Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return Parse(File.ReadAllBytes(path), path);
    }

    /// <summary>Reads rules from their JSON bytes (UTF-8, comments allowed); <paramref name="source"/> names them in messages.</summary>
    /// <exception cref="InvalidDataException">Not a valid rules file: the message names the source, the key and what is valid.</exception>
    public static RuleTextHabits Parse(ReadOnlyMemory<byte> utf8, string source)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(source);
        InvalidDataException Bad(string key, string what) => new($"{source}: '{key}' {what}.");
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(utf8, new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{source}: not valid JSON ({e.Message}).", e);
        }

        using (document)
        {
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw Bad("", "must be an object (name, summary, rules)");
            }

            foreach (var p in root.EnumerateObject())
            {
                if (Array.IndexOf(TopKeys, p.Name) < 0)
                {
                    throw Bad(p.Name, $"is not a key of text habits (valid: {string.Join(", ", TopKeys)})");
                }
            }

            string name = root.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String && n.GetString()!.Length > 0 ? n.GetString()! : throw Bad("name", "is missing");
            string summary = root.TryGetProperty("summary", out var s) && s.ValueKind == JsonValueKind.String ? s.GetString()! : name;
            if (!root.TryGetProperty("rules", out var list) || list.ValueKind != JsonValueKind.Array)
            {
                throw Bad("rules", "is missing: a list of {name, from, to, where, chance}");
            }

            var rules = new List<TextHabitRule>();
            int index = 0;
            foreach (var rule in list.EnumerateArray())
            {
                string key = $"rules[{index++}]";
                if (rule.ValueKind != JsonValueKind.Object)
                {
                    throw Bad(key, "must be an object");
                }

                foreach (var p in rule.EnumerateObject())
                {
                    if (Array.IndexOf(RuleKeys, p.Name) < 0)
                    {
                        throw Bad($"{key}.{p.Name}", $"is not a key of a habit (valid: {string.Join(", ", RuleKeys)})");
                    }
                }

                string ruleName = rule.TryGetProperty("name", out var rn) && rn.ValueKind == JsonValueKind.String && rn.GetString()!.Length > 0 ? rn.GetString()! : throw Bad($"{key}.name", "is missing");
                var from = rule.TryGetProperty("from", out var f) ? Texts(f) : null;
                if (from is null || from.Count == 0 || from.Any(t => t.Length == 0))
                {
                    throw Bad($"{key}.from", "must be a text, or a list of texts, none empty");
                }

                string to = rule.TryGetProperty("to", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString()! : throw Bad($"{key}.to", "must be a text (empty: left out)");
                var where = TextHabitPlace.Any;
                if (rule.TryGetProperty("where", out var w))
                {
                    where = 0;
                    foreach (string place in Texts(w) ?? throw Bad($"{key}.where", "must be a place or a list of places"))
                    {
                        where |= place switch
                        {
                            "any" => TextHabitPlace.Any,
                            "initial" => TextHabitPlace.Initial,
                            "medial" => TextHabitPlace.Medial,
                            "final" => TextHabitPlace.Final,
                            "isolated" => TextHabitPlace.Isolated,
                            _ => throw Bad($"{key}.where", $"is '{place}' (valid: {string.Join(", ", Places)})"),
                        };
                    }
                }

                double chance = rule.TryGetProperty("chance", out var c)
                    ? c.ValueKind == JsonValueKind.Number && c.GetDouble() is >= 0 and <= 1 ? c.GetDouble() : throw Bad($"{key}.chance", "must be a number from 0 to 1")
                    : 0;
                rules.Add(new TextHabitRule(ruleName, from, to, where, chance));
            }

            return new RuleTextHabits(name, summary, rules);
        }
    }

    private static List<string>? Texts(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.String => [e.GetString()!],
        JsonValueKind.Array when e.EnumerateArray().All(x => x.ValueKind == JsonValueKind.String) => [.. e.EnumerateArray().Select(x => x.GetString()!)],
        _ => null,
    };

    /// <inheritdoc/>
    public string Apply(string text, Random random, Func<string, double?>? chance = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(random);
        StringBuilder? result = null;
        int copied = 0;
        for (int i = 0; i < text.Length;)
        {
            int replaced = 0;
            foreach (var rule in Rules)
            {
                foreach (string from in rule.From)
                {
                    if (!text.AsSpan(i).StartsWith(from, StringComparison.Ordinal) || (Place(text, i, from.Length) & rule.Where) == 0)
                    {
                        continue;
                    }

                    // A place the habit holds at: a number drawn whatever its chance, so one habit's chance does not move another's.
                    double draw = random.NextDouble();
                    if (draw < (chance?.Invoke(rule.Name) ?? rule.Chance))
                    {
                        result ??= new StringBuilder(text.Length);
                        result.Append(text, copied, i - copied).Append(rule.To);
                        replaced = from.Length;
                        copied = i + replaced;
                    }

                    goto tried;
                }
            }

        tried:
            i += replaced > 0 ? replaced : 1;
        }

        return result is null ? text : result.Append(text, copied, text.Length - copied).ToString();
    }

    // Where text[start, start + length) sits in its word.
    private static TextHabitPlace Place(string text, int start, int length)
    {
        bool atStart = start == 0 || !IsWord(text[start - 1]);
        int end = start + length;
        while (end < text.Length && IsMark(text[end]))
        {
            end++;
        }

        bool atEnd = end >= text.Length || !IsWord(text[end]);
        return (atStart, atEnd) switch
        {
            (true, true) => TextHabitPlace.Isolated,
            (true, false) => TextHabitPlace.Initial,
            (false, true) => TextHabitPlace.Final,
            _ => TextHabitPlace.Medial,
        };
    }

    private static bool IsWord(char c) => char.IsLetter(c) || IsMark(c);

    private static bool IsMark(char c) => CharUnicodeInfo.GetUnicodeCategory(c) is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark;
}
