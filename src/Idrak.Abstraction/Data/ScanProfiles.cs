// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A mix of scan degradations (rotation, blur, noise, faded ink, JPEG, ...): from a seed, the image transforms one page
/// gets, each step taken at its chance with its values drawn from its ranges (the same seed, the same steps). Register
/// one with <see cref="ScanProfiles.Register"/>; <see cref="ScanProfile"/> is one written as text.
/// </summary>
public interface IScanProfile
{
    /// <summary>The profile's name ("office").</summary>
    string Name { get; }

    /// <summary>One line on the scans it makes.</summary>
    string Summary { get; }

    /// <summary>The steps for the page of <paramref name="seed"/> (empty when none was taken).</summary>
    ImageTransformPipeline Sample(int seed);
}

/// <summary>
/// A scan profile written as text: steps separated by commas, each <c>NAME[=VALUE][@CHANCE]</c> where NAME is a
/// registered image transform (<see cref="ImageTransforms"/>), VALUE a number, a range <c>MIN..MAX</c> or a word (a
/// range gives whole numbers when the transform declares that input whole, <see cref="ImageTransformInput.Integer"/>,
/// fractions when it declares a number, and otherwise whole numbers only when both ends are written whole), and CHANCE how often the step is taken (0 to 1; 1 when not given). An
/// item <c>KEY=VALUE</c> after a step is one of its options (a value or a range). A step whose transform takes a
/// <c>seed</c> option and is given none gets one drawn from the page's seed. For example
/// <c>rotate=-1.5..1.5@0.8,blur=0.3..1@0.4,noise=2..8@0.6,jpeg=40..90@0.7</c>.
/// </summary>
public sealed class ScanProfile : IScanProfile
{
    private readonly Step[] _steps;

    /// <summary>The profile <paramref name="name"/> written as <paramref name="text"/> (see the class).</summary>
    /// <exception cref="ArgumentException">An unknown transform or option, a bad range or chance, or values the transform refuses.</exception>
    public ScanProfile(string name, string text, string? summary = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(text);
        Name = name;
        _steps = Parse(text);
        Text = string.Join(",", _steps.Select(s => s.ToString()));
        Summary = summary ?? Text;
        foreach (var step in _steps)
        {
            // Both ends of every range must be values the transform takes.
            var transform = ImageTransforms.Get(step.Transform);
            foreach (bool high in new[] { false, true })
            {
                var made = step.Make(high ? 1.0 : 0.0, 0, transform);
                made.ThrowIfUnknownOptions(transform.Keys);
                transform.Check(made);
            }
        }
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Summary { get; }

    /// <summary>The profile as text, as it parses back.</summary>
    public string Text { get; }

    /// <summary>The steps' transform names, in order.</summary>
    public IReadOnlyList<string> Transforms => [.. _steps.Select(s => s.Transform)];

    /// <inheritdoc/>
    public ImageTransformPipeline Sample(int seed)
    {
        var random = new Random(seed);
        var steps = new List<ImageTransformStep>(_steps.Length);
        foreach (var step in _steps)
        {
            // Every step draws the same numbers whether taken or not, so one step's chance does not move the others.
            double take = random.NextDouble(), draw = random.NextDouble();
            int stepSeed = random.Next();
            if (take < step.Chance)
            {
                steps.Add(step.Make(draw, stepSeed, ImageTransforms.Get(step.Transform), new Random(stepSeed)));
            }
        }

        return new ImageTransformPipeline(steps);
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
                    throw new ArgumentException($"A scan step's chance is from 0 to 1, not '{c}' (in '{item}').");
                }

                item = item[..atChance].Trim();
            }

            int eq = item.IndexOf('=', StringComparison.Ordinal);
            string name = (eq < 0 ? item : item[..eq]).Trim().ToLowerInvariant();
            Value? value = eq < 0 ? null : Value.Parse(item[(eq + 1)..].Trim(), item);
            if (name.Length == 0)
            {
                throw new ArgumentException($"A scan step is NAME[=VALUE][@CHANCE] (such as rotate=-1..1@0.8), not '{raw.Trim()}'.");
            }

            if (ImageTransforms.Find(name) is { } transform)
            {
                steps.Add(new Step(name, value is { } v ? Kind(v, transform, "") : null, [], chance));
                continue;
            }

            if (steps.Count > 0 && value is not null && atChance < 0)
            {
                var owner = ImageTransforms.Get(steps[^1].Transform);
                if (owner.Keys.Contains(name, StringComparer.OrdinalIgnoreCase))
                {
                    steps[^1].Options.Add((name, Kind(value.Value, owner, name)));
                    continue;
                }
            }

            string after = steps.Count > 0 && ImageTransforms.Get(steps[^1].Transform).Keys is { Count: > 0 } k
                ? $"; after {steps[^1].Transform} it can also be one of its options ({string.Join(", ", k)}), with no chance of its own" : "";
            throw new ArgumentException($"Unknown image transform '{name}' in a scan profile (registered: {string.Join(", ", ImageTransforms.Names)}{after}).");
        }

        return [.. steps];
    }

    // Whole or fractional as the transform declares the input (key "" for its value), else as written.
    private static Value Kind(Value value, IImageTransform transform, string key) =>
        value.Word is null && transform.Parameters.FirstOrDefault(p => p.Key.Equals(key, StringComparison.OrdinalIgnoreCase)) is { Input: ImageTransformInput.Integer or ImageTransformInput.Number } p
            ? value with { Whole = p.Input == ImageTransformInput.Integer }
            : value;

    // A value: a word, a number, or a range drawn from (whole numbers when both ends are written whole).
    private readonly record struct Value(string? Word, double Min, double Max, bool Whole)
    {
        public static Value Parse(string text, string item)
        {
            int dots = text.IndexOf("..", StringComparison.Ordinal);
            if (dots > 0)
            {
                string a = text[..dots].Trim(), b = text[(dots + 2)..].Trim();
                if (!double.TryParse(a, NumberStyles.Float, CultureInfo.InvariantCulture, out double min)
                    || !double.TryParse(b, NumberStyles.Float, CultureInfo.InvariantCulture, out double max) || !(min <= max))
                {
                    throw new ArgumentException($"A scan range is MIN..MAX with MIN not above MAX, not '{text}' (in '{item}').");
                }

                return new Value(null, min, max, IsWhole(a) && IsWhole(b));
            }

            return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v)
                ? new Value(null, v, v, IsWhole(text))
                : new Value(text, 0, 0, false);
        }

        private static bool IsWhole(string s) => !s.Contains('.') && !s.Contains('e', StringComparison.OrdinalIgnoreCase);

        public string Draw(double u)
        {
            if (Word is not null)
            {
                return Word;
            }

            double v = Min + u * (Max - Min);
            return Whole ? ((long)Math.Round(v)).ToString(CultureInfo.InvariantCulture) : Math.Round(v, 3).ToString("0.###", CultureInfo.InvariantCulture);
        }

        public override string ToString() => Word
            ?? (Min == Max ? Format(Min) : $"{Format(Min)}..{Format(Max)}");

        private string Format(double v) => Whole ? ((long)v).ToString(CultureInfo.InvariantCulture) : v.ToString("0.0##", CultureInfo.InvariantCulture);
    }

    private sealed record Step(string Transform, Value? Value, List<(string Key, Value Value)> Options, double Chance)
    {
        // The step with its value at `u` along its range, its options drawn from `random` (their low ends when none),
        // and a seed when the transform takes one and none is given.
        public ImageTransformStep Make(double u, int seed, IImageTransform transform, Random? random = null)
        {
            var options = Options.Select(o => KeyValuePair.Create(o.Key, o.Value.Draw(random?.NextDouble() ?? u))).ToList();
            if (transform.Keys.Contains("seed", StringComparer.OrdinalIgnoreCase) && !options.Any(o => o.Key.Equals("seed", StringComparison.OrdinalIgnoreCase)))
            {
                options.Add(KeyValuePair.Create("seed", seed.ToString(CultureInfo.InvariantCulture)));
            }

            return new ImageTransformStep(Transform, Value?.Draw(u), options);
        }

        public override string ToString() => Transform + (Value is { } v ? "=" + v : "")
            + (Chance < 1 ? "@" + Chance.ToString("0.###", CultureInfo.InvariantCulture) : "")
            + string.Concat(Options.Select(o => $",{o.Key}={o.Value}"));
    }
}

/// <summary>
/// The scan profiles, by name. Built in (Idrak.Imaging): "clean" (a careful scan: rarely a slight turn, light
/// JPEG), "office" (an office scanner or a phone: sometimes a turn, blur, noise, JPEG, some lower resolution), "old" (an old letter:
/// faded, uneven and worn ink, tinted and stained paper, specks, bleed-through, low resolution), "pale" (faint ink over
/// the whole page), "worn" (broken strokes and thinned letters), "fax" (black and white at low resolution, specks),
/// "handled" (wrinkles, folds, a dog-ear, staples, clips, smudges, ink blots, fingerprints, correction fluid, a sticky
/// note), "marked" (struck through, underlined, crossed out, highlighted, notes in the margins), "damaged" (torn edges
/// and pieces, punch, worm and burn holes, scorched edges, scratches, ripped and taped back, water), "scanner" (an edge
/// shadow, streaks, uneven light, a curled edge, the page cut off), "aged" (yellowed, sun-faded, foxed, tape glue),
/// "copied" (photocopied, carbon, low toner, banding, a ghost image), "photo" (a phone photo on a desk: glare, a hand's
/// shadow, darker corners), "wrecked" (many kinds of damage at once) and "mixed" (every effect at a small chance of
/// its own); several are
/// taken together on one page when joined with '+' ("damaged+marked", see <see cref="Resolve"/>). Register another, or
/// one of those names over the library's, with <see cref="Register"/>.
/// </summary>
public static class ScanProfiles
{
    private static readonly SlotTable<string, IScanProfile> Registry =
        new(nameof(ScanProfiles), (slot, app, library) => new GuardedProfile(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static ScanProfiles() => LibraryDefaults.Ensure(typeof(ScanProfiles));

    /// <summary>Registers <paramref name="profile"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IScanProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrWhiteSpace(profile.Name);
        Registry.Register(profile.Name, profile, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's profile of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The profile of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IScanProfile Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(ScanProfiles), name, Names,
            $"No scan profile '{name}' is registered ({string.Join(", ", Names)}); add it with ScanProfiles.Register, or write one as text (ScanProfile).");

    /// <summary>
    /// The profile <paramref name="nameOrText"/> names; profiles joined with '+' ("damaged+marked"), each drawing its
    /// own steps for the page, in that order, a step two of them took (a turn, the JPEG) taken once, where the later
    /// one puts it; or, when it names none and has a step in it (an '=' or a registered transform's name), a profile
    /// written as that text (see <see cref="ScanProfile"/>).
    /// </summary>
    /// <exception cref="NotSupportedException">A name that is not registered.</exception>
    public static IScanProfile Resolve(string nameOrText)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(nameOrText);
        string text = nameOrText.Trim();
        if (Registry.Find(text) is { } named)
        {
            return named;
        }

        if (text.Contains('+') && !text.Contains('=') && !text.Contains(','))
        {
            var looks = text.Split('+', StringSplitOptions.TrimEntries);
            return looks.Any(l => l.Length == 0)
                ? throw new ArgumentException($"'{text}' joins scan profiles with '+', a name on each side.")
                : new JoinedProfile(string.Join("+", looks), [.. looks.Select(Get)]);
        }

        return text.Contains('=') || text.Contains(',') || ImageTransforms.Find(text.Split('@')[0]) is not null
            ? new ScanProfile("custom", text)
            : Get(text);
    }

    /// <summary>The library's profile of that name, or null.</summary>
    public static IScanProfile? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the profile of that name.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>What happens when the app's profile of that name fails.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Each profile's name and summary, one a line.</summary>
    public static string Describe() => string.Join("\n", Names.Select(n => $"{n}: {Get(n).Summary}"));

    // Profiles taken together: each samples the page from its own seed. A transform a later one names too is left to
    // the later one, at its chance and place (so the capture steps, a turn or the JPEG, stay last and are not done twice);
    // for a profile that is not written as text, a transform two of them took runs once, where the later one puts it.
    private sealed class JoinedProfile : IScanProfile
    {
        private readonly IScanProfile[] _looks;
        private readonly HashSet<string>[] _later;

        public JoinedProfile(string name, IScanProfile[] looks)
        {
            Name = name;
            _looks = looks;
            _later = new HashSet<string>[looks.Length];
            var named = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = looks.Length - 1; i >= 0; i--)
            {
                _later[i] = [.. named];
                if (looks[i] is ScanProfile written)
                {
                    named.UnionWith(written.Transforms);
                }
            }
        }

        public string Name { get; }

        public string Summary => string.Join("; then ", _looks.Select(l => l.Summary));

        public ImageTransformPipeline Sample(int seed)
        {
            var random = new Random(seed);
            var steps = _looks.SelectMany((look, index) => look.Sample(random.Next()).Steps.Where(step => !_later[index].Contains(step.Name)).Select(step => (index, step))).ToList();
            return new ImageTransformPipeline(steps.Where((s, at) => !steps.Skip(at + 1)
                .Any(t => t.index != s.index && t.step.Name.Equals(s.step.Name, StringComparison.OrdinalIgnoreCase))).Select(s => s.step));
        }
    }

    private sealed class GuardedProfile(Slot slot, IScanProfile app, IScanProfile library) : IScanProfile
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public ImageTransformPipeline Sample(int seed) => slot.Call(() => app.Sample(seed), () => library.Sample(seed), (a, b) => Comparisons.Exact(a.ToString(), b.ToString()));
    }
}
