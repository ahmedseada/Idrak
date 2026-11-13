// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A mix of shapes for pages: from a seed, the shape pipeline one page gets, each step taken at its chance with its
/// values drawn from its ranges (the same seed, the same steps). Register one with <see cref="ShapeProfiles.Register"/>;
/// <see cref="ShapeProfile"/> is one written as text.
/// </summary>
public interface IShapeProfile : IKind
{
    /// <summary>The shapes for the page of <paramref name="seed"/> (empty when none was taken).</summary>
    ShapePipeline Sample(int seed);
}

/// <summary>
/// A shape profile written as text: steps separated by commas, each <c>NAME[=VALUE][@CHANCE]</c> where NAME is a
/// registered shape (<see cref="Shapes"/>), VALUE a number, a range <c>MIN..MAX</c> or a word in which every range is
/// drawn (<c>grid=auto x 3..6</c>), and CHANCE how often the step is taken (0 to 1; 1 when not given). An item
/// <c>KEY=VALUE</c> after a step is one of its options (its own or a common one), its value written the same way. A range
/// is drawn as a whole number for the keys the shape declares whole (<see cref="IShape.WholeKeys"/>), with fractions
/// otherwise. For example <c>box@0.5,grid=auto x auto,in=last,row-height=1.4..2,heavy-width=2..3</c>.
/// </summary>
public sealed class ShapeProfile : IShapeProfile
{
    private readonly Step[] _steps;

    /// <summary>The profile <paramref name="name"/> written as <paramref name="text"/> (see the class).</summary>
    /// <exception cref="ArgumentException">An unknown shape or option, a bad range or chance, or values the shape refuses.</exception>
    public ShapeProfile(string name, string text, string? summary = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(text);
        Name = name;
        _steps = Parse(text);
        Text = string.Join(",", _steps.Select(s => s.ToString()));
        Summary = summary ?? (Text.Length == 0 ? "no shapes" : Text);
        foreach (var step in _steps)
        {
            // Both ends of every range must be values the shape takes.
            var shape = Shapes.Get(step.Shape);
            ShapePipeline.Check(shape, step.Make(0.0, shape, null));
            ShapePipeline.Check(shape, step.Make(1.0, shape, null));
        }
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Summary { get; }

    /// <summary>The profile as text, as it parses back.</summary>
    public string Text { get; }

    /// <summary>The steps' shape names, in order.</summary>
    public IReadOnlyList<string> ShapeNames => [.. _steps.Select(s => s.Shape)];

    /// <inheritdoc/>
    public ShapePipeline Sample(int seed)
    {
        var random = new Random(seed);
        var steps = new List<ShapeStep>(_steps.Length);
        foreach (var step in _steps)
        {
            // Every step draws the same numbers whether taken or not, so one step's chance does not move the others.
            double take = random.NextDouble();
            int stepSeed = random.Next();
            if (take < step.Chance)
            {
                steps.Add(step.Make(0, Shapes.Get(step.Shape), new Random(stepSeed)));
            }
        }

        return steps.Count == 0 ? ShapePipeline.Empty : new ShapePipeline(steps);
    }

    /// <inheritdoc/>
    public override string ToString() => Text;

    private static Step[] Parse(string text)
    {
        var steps = new List<Step>();
        foreach (string raw in text.Split(','))
        {
            string item = raw.Trim();
            if (item.Length == 0)
            {
                continue;
            }

            double chance = 1;
            int atChance = item.LastIndexOf('@');
            if (atChance >= 0)
            {
                string c = item[(atChance + 1)..].Trim();
                if (!double.TryParse(c, NumberStyles.Float, CultureInfo.InvariantCulture, out chance) || chance is < 0 or > 1)
                {
                    throw new ArgumentException($"A shape step's chance is from 0 to 1, not '{c}' (in '{item}').");
                }

                item = item[..atChance].Trim();
            }

            int eq = item.IndexOf('=', StringComparison.Ordinal);
            string name = (eq < 0 ? item : item[..eq]).Trim().ToLowerInvariant();
            Value? value = eq < 0 ? null : Value.Parse(item[(eq + 1)..].Trim(), item);
            if (name.Length == 0)
            {
                throw new ArgumentException($"A shape step is NAME[=VALUE][@CHANCE] (such as grid=auto x 3..6@0.8), not '{raw.Trim()}'.");
            }

            if (Shapes.Find(name) is not null)
            {
                steps.Add(new Step(name, value, [], chance));
                continue;
            }

            if (steps.Count > 0 && value is not null && atChance < 0 && ShapePipeline.Takes(Shapes.Get(steps[^1].Shape), name))
            {
                steps[^1].Options.Add((name, value.Value));
                continue;
            }

            string after = steps.Count > 0 && Shapes.Get(steps[^1].Shape) is { } owner
                ? $"; after {steps[^1].Shape} it can also be one of its options ({(owner.Keys.Count > 0 ? string.Join(", ", owner.Keys) + ", " : "")}{Shapes.CommonKeys}), with no chance of its own" : "";
            throw new ArgumentException($"Unknown shape '{name}' in a shape profile (registered: {string.Join(", ", Shapes.Names)}{after}).");
        }

        return [.. steps];
    }

    // A value as written: words and numbers kept, every range MIN..MAX in it drawn ("auto x 3..6", "2..3", "1:4:1*16").
    private readonly record struct Value(string[] Parts, (double Min, double Max)?[] Ranges)
    {
        // A range's ends are numbers; the characters around them split a value into parts.
        private static readonly System.Buffers.SearchValues<char> Separators = System.Buffers.SearchValues.Create(": x*X×/");

        public static Value Parse(string text, string item)
        {
            var parts = new List<string>();
            var ranges = new List<(double, double)?>();
            int start = 0;
            while (start <= text.Length)
            {
                int length = text.AsSpan(start).IndexOfAny(Separators);
                int end = length < 0 ? text.Length : start + length;
                string part = text[start..end];
                int dots = part.IndexOf("..", StringComparison.Ordinal);
                if (dots >= 0)
                {
                    if (!double.TryParse(part.AsSpan(0, dots), NumberStyles.Float, CultureInfo.InvariantCulture, out double min)
                        || !double.TryParse(part.AsSpan(dots + 2), NumberStyles.Float, CultureInfo.InvariantCulture, out double max) || !(min <= max)
                        || !double.IsFinite(min) || !double.IsFinite(max))
                    {
                        throw new ArgumentException($"A range in a shape profile is MIN..MAX with MIN not above MAX, not '{part}' (in '{item}').");
                    }

                    parts.Add(part);
                    ranges.Add((min, max));
                }
                else
                {
                    parts.Add(part);
                    ranges.Add(null);
                }

                if (end >= text.Length)
                {
                    break;
                }

                parts.Add(text[end].ToString());
                ranges.Add(null);
                start = end + 1;
            }

            return new Value([.. parts], [.. ranges]);
        }

        // The value with each range at `u` along it (the same `u` for every range), or each drawn from `random`.
        public string Draw(double u, Random? random, bool whole)
        {
            if (Ranges.All(r => r is null))
            {
                return string.Concat(Parts);
            }

            var text = new StringBuilder();
            for (int i = 0; i < Parts.Length; i++)
            {
                if (Ranges[i] is not { } range)
                {
                    text.Append(Parts[i]);
                    continue;
                }

                double v = range.Min + (random?.NextDouble() ?? u) * (range.Max - range.Min);
                text.Append(whole ? ((long)Math.Round(v)).ToString(CultureInfo.InvariantCulture) : Math.Round(v, 3).ToString("0.###", CultureInfo.InvariantCulture));
            }

            return text.ToString();
        }

        public override string ToString() => string.Concat(Parts);
    }

    private sealed record Step(string Shape, Value? Value, List<(string Key, Value Value)> Options, double Chance)
    {
        // The step with every range at `u` along it, or drawn from `random` when one is given.
        public ShapeStep Make(double u, IShape shape, Random? random)
        {
            var whole = shape.WholeKeys;
            string? value = Value?.Draw(u, random, whole.Contains(""));
            var options = new List<KeyValuePair<string, string>>(Options.Count);
            foreach (var (key, option) in Options)
            {
                options.Add(KeyValuePair.Create(key, option.Draw(u, random, whole.Contains(key))));
            }

            return new ShapeStep(Shape, value, options);
        }

        public override string ToString() => Shape + (Value is { } v ? "=" + v : "")
            + (Chance < 1 ? "@" + Chance.ToString("0.###", CultureInfo.InvariantCulture) : "")
            + string.Concat(Options.Select(o => $",{o.Key}={o.Value}"));
    }
}

/// <summary>
/// The shape profiles, by name. Built in (Idrak.Imaging): "none" (no shapes), "grid" (a grid fitted to the text: a row a
/// line, a column a unit), "ruled" (a line under every text line: a region a line) and "boxed" (a box around the text: a
/// region a line). Register another, or one of those names over the library's, with <see cref="Register"/>; or write one
/// as text (<see cref="ShapeProfile"/>, through <see cref="Resolve"/>).
/// </summary>
public static class ShapeProfiles
{
    private static readonly KindTable<IShapeProfile> Table = new(nameof(ShapeProfiles), "Shape profile",
        "add it with ShapeProfiles.Register, or write one as text (ShapeProfile)",
        (slot, app, library) => new GuardedProfile(slot, app, library), StringComparer.OrdinalIgnoreCase);

    // Profiles written as text, parsed once each (a page never parses its profile again); bounded so that texts made
    // on the fly cannot grow it without end.
    private static readonly ConcurrentDictionary<string, ShapeProfile> Written = new(StringComparer.Ordinal);
    private const int MostWritten = 1024;

    static ShapeProfiles() => LibraryDefaults.Ensure(typeof(ShapeProfiles));

    /// <summary>Registers <paramref name="profile"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IShapeProfile profile) => Table.Register(profile, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's profile of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The profile registered as <paramref name="name"/>, or null.</summary>
    public static IShapeProfile? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The profile registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IShapeProfile Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>
    /// The profile <paramref name="nameOrText"/> names; or, when it names none and has a step in it (an '=', a ',', an '@'
    /// or a registered shape's name), a profile written as that text (see <see cref="ShapeProfile"/>), parsed once and kept.
    /// </summary>
    /// <exception cref="NotSupportedException">A name that is not registered.</exception>
    /// <exception cref="ArgumentException">A profile written as text that does not parse.</exception>
    public static IShapeProfile Resolve(string nameOrText)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(nameOrText);
        string text = nameOrText.Trim();
        if (Table.Find(text) is { } named)
        {
            return named;
        }

        if (Written.TryGetValue(text, out var written))
        {
            return written;
        }

        if (!(text.Contains('=') || text.Contains(',') || text.Contains('@') || Shapes.Find(text) is not null))
        {
            return Get(text);
        }

        var parsed = new ShapeProfile("custom", text);
        if (Written.Count >= MostWritten)
        {
            Written.Clear();
        }

        return Written.GetOrAdd(text, parsed);
    }

    /// <summary>The library's profile of that name, or null.</summary>
    public static IShapeProfile? Default(string name) => Table.Default(name);

    /// <summary>Who registered the profile of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's profile of that name fails.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each profile's name and summary, one a line.</summary>
    public static string Describe() => Table.Describe();

    private sealed class GuardedProfile(Slot slot, IShapeProfile app, IShapeProfile library) : IShapeProfile
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public ShapePipeline Sample(int seed) => slot.Call(() => app.Sample(seed), () => library.Sample(seed), (a, b) => Comparisons.Exact(a.ToString(), b.ToString()));
    }
}
