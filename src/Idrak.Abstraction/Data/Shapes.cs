// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Data;

/// <summary>
/// One drawing step of a page's shapes (a line, a box, a grid, a circle, ...): pure geometry, it never touches pixels.
/// Given its step (<see cref="ShapeStep"/>: its value and options) and the box it is placed in
/// (<see cref="ShapeContext"/>), it plans what to draw as data (<see cref="ShapePlan"/>): ideal paths a pen strokes,
/// filled areas, and the regions it makes for text. Register one with <see cref="Shapes.Register"/>; a pipeline
/// (<see cref="ShapePipeline"/>) names shapes in the order they are drawn.
/// </summary>
/// <remarks>
/// Its <see cref="IKind.Name"/> is what a pipeline uses (<c>grid</c>; names compare ignoring case), its
/// <see cref="IKind.Summary"/> one line on what it draws and what its value is, and its <see cref="IConfigurableKind.Keys"/>
/// the option keys it takes besides the ones every shape takes (<see cref="Shapes.CommonKeys"/>). The same step and
/// context give the same plan.
/// </remarks>
public interface IShape : IConfigurableKind
{
    /// <summary>Throws <see cref="ArgumentException"/> when <paramref name="step"/>'s value or own options are not ones it takes (checked when a pipeline is parsed).</summary>
    void Check(ShapeStep step);

    /// <summary>What to draw for <paramref name="step"/> in <paramref name="context"/>'s box (already placed by the step's <c>at</c>, <c>size</c> and <c>margin</c>).</summary>
    ShapePlan Plan(ShapeStep step, ShapeContext context);

    /// <summary>
    /// The keys ("" for the value) whose ranges a shape profile draws as whole numbers; the others are drawn with
    /// fractions. A plain set (an <see cref="OptionKeys"/> takes no empty key); none by default.
    /// </summary>
    IReadOnlySet<string> WholeKeys => OptionKeys.None;
}

/// <summary>
/// Where a shape is planned: its box (pixels), the reading direction ("start" is the right side of a right-to-left
/// page), the text it is drawn for (its size and line height, its lines and the most units, whitespace-separated
/// tokens, on one line, for sizes set to <c>auto</c>), the stroke width and a seed.
/// </summary>
public sealed record ShapeContext
{
    /// <summary>The box's left edge.</summary>
    public float X { get; init; }

    /// <summary>The box's top edge.</summary>
    public float Y { get; init; }

    /// <summary>The box's width.</summary>
    public float Width { get; init; }

    /// <summary>The box's height.</summary>
    public float Height { get; init; }

    /// <summary>Whether the page reads right to left (its start side is the right).</summary>
    public bool RightToLeft { get; init; }

    /// <summary>
    /// Whether a shape may grow below the box when the text needs more height than it has (rows of text on a page that
    /// grows); otherwise every region stays inside the box.
    /// </summary>
    public bool GrowsDown { get; init; }

    /// <summary>The text's size, in pixels per em.</summary>
    public float TextSize { get; init; } = 24;

    /// <summary>The distance between the text's baselines, in pixels.</summary>
    public float LineHeight { get; init; } = 30;

    /// <summary>The text's lines.</summary>
    public int Lines { get; init; } = 1;

    /// <summary>The most units (whitespace-separated tokens) on one line of the text.</summary>
    public int MostUnits { get; init; } = 1;

    /// <summary>The stroke's width, in pixels (a path's own <see cref="ShapePath.Width"/> is a factor of it).</summary>
    public float StrokeWidth { get; init; } = 2;

    /// <summary>The seed of the shape's own choices (a range drawn per shape).</summary>
    public int Seed { get; init; }

    /// <summary>The most a curve flattened to a polyline may stray from the true curve, in pixels.</summary>
    public float Flatness { get; init; } = 0.25f;

    /// <summary>The box's right edge.</summary>
    public float Right => X + Width;

    /// <summary>The box's bottom edge.</summary>
    public float Bottom => Y + Height;

    /// <summary>The x of a share of the box's width counted from its start side (the right of a right-to-left page).</summary>
    public float XAt(float share) => RightToLeft ? X + Width - share * Width : X + share * Width;

    /// <summary>The y of a share of the box's height counted from its top.</summary>
    public float YAt(float share) => Y + share * Height;

    /// <summary>
    /// This context with the box <paramref name="step"/>'s common options ask for inside this one: <c>at=X:Y</c> and
    /// <c>size=W:H</c> (shares of this box, X from its start side), then <c>margin</c> (a share of the smaller side) kept
    /// inside on every side.
    /// </summary>
    /// <exception cref="ArgumentException">A bad <c>at</c>, <c>size</c> or <c>margin</c>.</exception>
    public ShapeContext Place(ShapeStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        var (ax, ay) = step.At();
        var (sw, sh) = step.Size();
        float w = Width * sw, h = Height * sh;
        float x = RightToLeft ? X + Width - (ax + sw) * Width : X + ax * Width;
        float y = Y + ay * Height;
        float m = step.Margin() * Math.Min(w, h);
        return this with { X = x + m, Y = y + m, Width = Math.Max(0, w - 2 * m), Height = Math.Max(0, h - 2 * m) };
    }
}

/// <summary>A region a shape makes for text: its group (a row: one line of text fills a group), its place in the group in reading order, and its box (pixels).</summary>
/// <param name="Group">The group: regions one line of text fills (a grid's row; a box's or a circle's own).</param>
/// <param name="Index">Its place in the group, in reading order, from 0 (a grid's column from the start side).</param>
/// <param name="X">Its left edge.</param>
/// <param name="Y">Its top edge.</param>
/// <param name="Width">Its width.</param>
/// <param name="Height">Its height.</param>
public readonly record struct ShapeRegion(int Group, int Index, float X, float Y, float Width, float Height)
{
    /// <summary>
    /// Whether it holds wrapped paragraphs (a step's <c>holds=paragraphs</c>, a <c>text</c> step) rather than one line of
    /// units: its text is a step's lines, each a paragraph, wrapped to its width; each drawn paragraph is one line of the
    /// page's text.
    /// </summary>
    public bool Paragraphs { get; init; }

    /// <summary>Whether it is the ellipse inscribed in its box (a circle's or an ellipse's): paragraph lines are shortened to fit it at each line's height.</summary>
    public bool Round { get; init; }
}

/// <summary>
/// An ideal path a pen strokes: a polyline of points (pixels), open or closed, its width (a factor of the stroke
/// width), its dash pattern and its tone.
/// </summary>
public sealed class ShapePath
{
    /// <summary>The path through <paramref name="points"/> (x, y pairs in pixels; used directly, not copied).</summary>
    /// <param name="points">The points, x then y, at least one point.</param>
    /// <param name="closed">Whether the last point joins the first.</param>
    /// <param name="width">Its width, a factor of the stroke width (above 0).</param>
    /// <param name="dash">Its dash pattern in stroke widths: on, off, on, off, ...; an "on" of 0 is a dot; empty or null for a solid line.</param>
    /// <param name="tone">How dark it is drawn, a share of the ink (0 to 1).</param>
    /// <exception cref="ArgumentException">No point, an odd count, a width not above 0, a tone outside 0 to 1, or a dash pattern of an odd count or with a negative length.</exception>
    public ShapePath(float[] points, bool closed = false, float width = 1, float[]? dash = null, float tone = 1)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (points.Length < 2 || points.Length % 2 != 0)
        {
            throw new ArgumentException($"A path is x, y pairs, at least one point; {points.Length} values are given.", nameof(points));
        }

        if (!(width > 0) || !(tone is >= 0 and <= 1))
        {
            throw new ArgumentException($"A path's width is above 0 ({width}) and its tone from 0 to 1 ({tone}).", nameof(width));
        }

        dash ??= [];
        if (dash.Length % 2 != 0 || dash.Any(d => !(d >= 0)) || (dash.Length > 0 && dash.Sum() <= 0))
        {
            throw new ArgumentException("A dash pattern is pairs of lengths (on, off), none negative, not all zero.", nameof(dash));
        }

        Points = points;
        Closed = closed;
        Width = width;
        Dash = dash;
        Tone = tone;
    }

    /// <summary>The points, x then y, in pixels.</summary>
    public ReadOnlyMemory<float> Points { get; }

    /// <summary>How many points there are.</summary>
    public int Count => Points.Length / 2;

    /// <summary>Whether the last point joins the first.</summary>
    public bool Closed { get; }

    /// <summary>Its width, a factor of the stroke width.</summary>
    public float Width { get; }

    /// <summary>Its dash pattern in stroke widths (on, off, ...; an "on" of 0 is a dot); empty for a solid line.</summary>
    public ReadOnlyMemory<float> Dash { get; }

    /// <summary>How dark it is drawn, a share of the ink.</summary>
    public float Tone { get; }

    /// <summary>The smallest box around its points.</summary>
    public (float MinX, float MinY, float MaxX, float MaxY) Bounds() => ShapePlan.Bounds(Points.Span);
}

/// <summary>A filled area: a closed polygon (pixels) and its tone (a tint over a box, the head of an arrow).</summary>
public sealed class ShapeArea
{
    /// <summary>The area inside <paramref name="points"/> (x, y pairs in pixels, at least three points; used directly, not copied), drawn at <paramref name="tone"/> (0 to 1).</summary>
    /// <exception cref="ArgumentException">Fewer than three points, an odd count, or a tone outside 0 to 1.</exception>
    public ShapeArea(float[] points, float tone)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (points.Length < 6 || points.Length % 2 != 0 || !(tone is >= 0 and <= 1))
        {
            throw new ArgumentException($"An area is at least three x, y pairs ({points.Length} values) with a tone from 0 to 1 ({tone}).", nameof(points));
        }

        Points = points;
        Tone = tone;
    }

    /// <summary>The polygon's points, x then y, in pixels.</summary>
    public ReadOnlyMemory<float> Points { get; }

    /// <summary>How dark it is filled, a share of the ink.</summary>
    public float Tone { get; }

    /// <summary>The smallest box around its points.</summary>
    public (float MinX, float MinY, float MaxX, float MaxY) Bounds() => ShapePlan.Bounds(Points.Span);
}

/// <summary>What a shape draws, as data: the ideal paths a pen strokes, the filled areas and the regions it makes for text.</summary>
public sealed class ShapePlan
{
    /// <summary>A plan of <paramref name="paths"/>, <paramref name="areas"/> and <paramref name="regions"/> (each list kept as given).</summary>
    public ShapePlan(IReadOnlyList<ShapePath> paths, IReadOnlyList<ShapeArea>? areas = null, IReadOnlyList<ShapeRegion>? regions = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        Paths = paths;
        Areas = areas ?? [];
        Regions = regions ?? [];
    }

    /// <summary>Nothing to draw.</summary>
    public static ShapePlan Empty { get; } = new([]);

    /// <summary>The paths, in the order they are drawn.</summary>
    public IReadOnlyList<ShapePath> Paths { get; }

    /// <summary>The filled areas, drawn before the paths.</summary>
    public IReadOnlyList<ShapeArea> Areas { get; }

    /// <summary>The regions for text, in reading order.</summary>
    public IReadOnlyList<ShapeRegion> Regions { get; }

    /// <summary>Whether it draws nothing and makes no region.</summary>
    public bool IsEmpty => Paths.Count == 0 && Areas.Count == 0 && Regions.Count == 0;

    /// <summary>A line on what it holds (counts and the box around it), for comparisons and descriptions.</summary>
    public override string ToString()
    {
        var text = new StringBuilder();
        text.Append(System.Globalization.CultureInfo.InvariantCulture, $"{Paths.Count} paths, {Areas.Count} areas, {Regions.Count} regions");
        foreach (var path in Paths)
        {
            var (x0, y0, x1, y1) = path.Bounds();
            text.Append(System.Globalization.CultureInfo.InvariantCulture, $"; path {path.Count}{(path.Closed ? " closed" : "")} {x0:F2},{y0:F2},{x1:F2},{y1:F2}");
        }

        foreach (var region in Regions)
        {
            text.Append(System.Globalization.CultureInfo.InvariantCulture, $"; region {region.Group}.{region.Index} {region.X:F2},{region.Y:F2} {region.Width:F2}x{region.Height:F2}");
        }

        return text.ToString();
    }

    // The smallest box around x, y pairs.
    internal static (float MinX, float MinY, float MaxX, float MaxY) Bounds(ReadOnlySpan<float> p)
    {
        if (p.Length < 2)
        {
            return default;
        }

        float minX = p[0], maxX = p[0], minY = p[1], maxY = p[1];
        for (int i = 2; i < p.Length; i += 2)
        {
            minX = MathF.Min(minX, p[i]);
            maxX = MathF.Max(maxX, p[i]);
            minY = MathF.Min(minY, p[i + 1]);
            maxY = MathF.Max(maxY, p[i + 1]);
        }

        return (minX, minY, maxX, maxY);
    }
}

/// <summary>
/// One step of a shape pipeline: a shape's name, its value (<c>3x4</c> in <c>grid=3x4</c>; null when none is given) and
/// its options (<c>widths=1:2:1</c>), with typed readers for both (key "" reads the value) and for the options every
/// shape takes (<see cref="Shapes.CommonKeys"/>). Immutable.
/// </summary>
public sealed class ShapeStep : IEquatable<ShapeStep>
{
    private readonly SortedDictionary<string, string> _options;

    /// <summary>The step <paramref name="name"/> with <paramref name="value"/> and <paramref name="options"/> (keys ignore case).</summary>
    public ShapeStep(string name, string? value = null, IEnumerable<KeyValuePair<string, string>>? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name.Trim().ToLowerInvariant();
        Value = value?.Trim();
        _options = new SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, text) in options ?? [])
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(key);
            _options[key.Trim().ToLowerInvariant()] = text?.Trim() ?? throw new ArgumentException($"The option '{key}' of {Name} has no value.", nameof(options));
        }
    }

    /// <summary>The shape's name (lower case).</summary>
    public string Name { get; }

    /// <summary>The value, as written; null when the step has none.</summary>
    public string? Value { get; }

    /// <summary>The options, by key (lower case).</summary>
    public IReadOnlyDictionary<string, string> Options => _options;

    /// <summary>This step with the option <paramref name="key"/> set to <paramref name="value"/>.</summary>
    public ShapeStep With(string key, string value) => new(Name, Value, _options.Where(o => !o.Key.Equals(key, StringComparison.OrdinalIgnoreCase)).Append(KeyValuePair.Create(key, value)));

    /// <summary>The option <paramref name="key"/> ("" for the value), or <paramref name="otherwise"/> when not given.</summary>
    public string? Option(string key, string? otherwise = null) =>
        key.Length == 0 ? (string.IsNullOrEmpty(Value) ? otherwise : Value) : _options.TryGetValue(key, out var v) ? v : otherwise;

    /// <summary>Whether the option <paramref name="key"/> ("" for the value) is given.</summary>
    public bool Has(string key) => Option(key) is not null;

    /// <summary>The option <paramref name="key"/> ("" for the value) as a number from <paramref name="least"/> to <paramref name="most"/>; <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">Not such a number.</exception>
    public float Number(string key, float otherwise, float least, float most) =>
        Option(key) is { } text ? PageOptions.Number(What(key), text, least, most) : otherwise;

    /// <summary>The option <paramref name="key"/> ("" for the value) as a whole number from <paramref name="least"/> to <paramref name="most"/>; <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">Not such a number.</exception>
    public int Integer(string key, int otherwise, int least, int most) =>
        Option(key) is { } text ? PageOptions.Integer(What(key), text, least, most) : otherwise;

    /// <summary>The option <paramref name="key"/> ("" for the value) as a value or a range <c>MIN..MAX</c> within <paramref name="least"/> to <paramref name="most"/>; <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">Not such a range.</exception>
    public (float Min, float Max) Range(string key, (float Min, float Max) otherwise, float least, float most) =>
        Option(key) is { } text ? PageOptions.Range(What(key), text, least, most) : otherwise;

    /// <summary>The option <paramref name="key"/> ("" for the value) as a list of numbers (<c>1:4:1*16</c>, see <see cref="PageOptions.List"/>); null when not given.</summary>
    /// <exception cref="ArgumentException">A bad list or number.</exception>
    public float[]? Numbers(string key, float least, float most) =>
        Option(key) is { } text ? PageOptions.Numbers(What(key), text, least, most) : null;

    /// <summary>The option <paramref name="key"/> ("" for the value) as a list of words (see <see cref="PageOptions.List"/>); null when not given.</summary>
    /// <exception cref="ArgumentException">A bad list.</exception>
    public IReadOnlyList<string>? List(string key) => Option(key) is { } text ? PageOptions.List(What(key), text) : null;

    /// <summary>The option <paramref name="key"/> ("" for the value) as on or off; <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">Neither.</exception>
    public bool Switch(string key, bool otherwise) => Option(key) is { } text ? PageOptions.Switch(What(key), text) : otherwise;

    /// <summary>The option <paramref name="key"/> ("" for the value) as one of <paramref name="choices"/>; <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">None of them.</exception>
    public string Choice(string key, string otherwise, params ReadOnlySpan<string> choices) =>
        Option(key) is { } text ? PageOptions.Choice(What(key), text, choices) : otherwise;

    /// <summary>The option <paramref name="key"/> ("" for the value) as a pair <c>A:B</c>, each from <paramref name="least"/> to <paramref name="most"/>; <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">Not two numbers in range.</exception>
    public (float A, float B) Pair(string key, (float A, float B) otherwise, float least, float most)
    {
        if (Option(key) is not { } text)
        {
            return otherwise;
        }

        var numbers = PageOptions.Numbers(What(key), text, least, most);
        return numbers.Length == 2 ? (numbers[0], numbers[1]) : throw new ArgumentException($"{What(key)} is two numbers A:B, not '{text}'.");
    }

    /// <summary>The option <paramref name="key"/> ("" for the value) as points <c>X:Y:X:Y:...</c> (at least <paramref name="least"/>), each number from 0 to 1; null when not given.</summary>
    /// <exception cref="ArgumentException">An odd count, too few points, or a number out of range.</exception>
    public float[]? Points(string key, int least)
    {
        if (Option(key) is not { } text)
        {
            return null;
        }

        var numbers = PageOptions.Numbers(What(key), text, 0, 1);
        return numbers.Length % 2 == 0 && numbers.Length / 2 >= least ? numbers
            : throw new ArgumentException($"{What(key)} is at least {least} points X:Y:X:Y:... (shares of the box), not '{text}'.");
    }

    /// <summary>Where the shape's box starts in the box it is placed in, <c>at=X:Y</c> (shares, X from the start side; 0:0).</summary>
    public (float X, float Y) At() => Pair("at", (0, 0), 0, 1);

    /// <summary>The share of the box it is placed in that the shape's box takes, <c>size=W:H</c> (1:1).</summary>
    public (float Width, float Height) Size() => Pair("size", (1, 1), 0, 1);

    /// <summary>Whether the shape is drawn once inside each region the step before it made (<c>in=last</c>) rather than in the page (<c>in=page</c>, the default).</summary>
    public bool InLast() => Choice("in", "page", "page", "last") == "last";

    /// <summary>The space kept inside the shape's box on every side, a share of its smaller side (<c>margin</c>; 0).</summary>
    public float Margin() => Number("margin", 0, 0, 0.45f);

    /// <summary>The stroke's width, a factor of the page's stroke width (<c>width</c>; 1).</summary>
    public float WidthFactor() => Number("width", 1, 0.05f, 50);

    /// <summary>The dash pattern in stroke widths (<c>dash</c>: <c>solid</c>, the default, empty; <c>dashed</c>; <c>dotted</c>).</summary>
    public float[] Dash() => Choice("dash", "solid", "solid", "dashed", "dotted") switch
    {
        "dashed" => [5f, 3.5f],
        "dotted" => [0f, 2.5f],
        _ => [],
    };

    /// <summary>How dark the shape is drawn, a share of the ink (<c>tone</c>); <paramref name="otherwise"/> when not given.</summary>
    public float Tone(float otherwise = 1) => Number("tone", otherwise, 0, 1);

    /// <summary>
    /// Whether the step is stacked below the parts before it down the page (<c>flow=down</c>), its height from what it
    /// holds, rather than placed in the page by <c>at</c> and <c>size</c> (<c>flow=page</c>, the default).
    /// </summary>
    public bool FlowsDown() => Choice("flow", "page", "page", "down") == "down";

    /// <summary>The space left below a stacked step, in text line heights (<c>gap</c>; 0.6).</summary>
    public float Gap() => Number("gap", 0.6f, 0, 50);

    /// <summary>A stacked step's height, in text line heights (<c>height</c>); null: from what it holds.</summary>
    public float? FlowHeight() => Has("height") ? Number("height", 1, 0.1f, 1000) : null;

    /// <summary>
    /// How many of the page's lines, in order, the step takes (<c>lines=N</c>, or <c>lines=rest</c>: what the other steps
    /// leave); null when not given (a step making a fixed number of groups takes one line a group, one whose groups follow
    /// the text, or that holds paragraphs, the rest).
    /// </summary>
    public (int Count, bool Remainder)? Lines() => Option("lines") switch
    {
        null => null,
        var text when text.Equals("rest", StringComparison.OrdinalIgnoreCase) => (0, true),
        var text => (PageOptions.Integer(What("lines"), text, 0, 1_000_000), false),
    };

    /// <summary>Whether the step's regions hold wrapped paragraphs (<c>holds=paragraphs</c>) rather than one line of units each (<c>holds=lines</c>, the default).</summary>
    public bool HoldsParagraphs() => Choice("holds", "lines", "lines", "paragraphs") == "paragraphs";

    /// <summary>Reads every common option (<see cref="Shapes.CommonKeys"/>), throwing on a bad one, and that the shape's box stays inside the box it is placed in.</summary>
    /// <exception cref="ArgumentException">A bad common option.</exception>
    public void CheckCommon()
    {
        var (ax, ay) = At();
        var (w, h) = Size();
        if (ax + w > 1.0001f || ay + h > 1.0001f)
        {
            throw new ArgumentException($"{Name}'s at and size leave the box it is placed in (at {ax}:{ay}, size {w}:{h}).");
        }

        _ = InLast();
        _ = Margin();
        _ = WidthFactor();
        _ = Dash();
        _ = Tone();
        _ = FlowsDown();
        _ = Gap();
        _ = FlowHeight();
        _ = Lines();
        _ = HoldsParagraphs();
    }

    /// <summary>Throws unless every option is one the shape takes: one of <paramref name="keys"/> or a common one (<see cref="Shapes.CommonKeys"/>), naming them.</summary>
    /// <exception cref="ArgumentException">An option it does not take.</exception>
    public void ThrowIfUnknownOptions(IReadOnlySet<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        foreach (string key in _options.Keys)
        {
            if (!keys.Contains(key) && !Shapes.CommonKeys.Contains(key))
            {
                throw new ArgumentException($"The shape {Name} does not take '{key}'; it takes: {(keys.Count == 0 ? "" : string.Join(", ", keys) + ", and ")}every shape's {Shapes.CommonKeys}.");
            }
        }
    }

    /// <summary>As a pipeline writes it: <c>name</c>, <c>name=value</c>, then <c>,key=value</c> per option.</summary>
    public override string ToString()
    {
        var text = new StringBuilder(Name);
        if (!string.IsNullOrEmpty(Value))
        {
            text.Append('=').Append(Value);
        }

        foreach (var (key, value) in _options)
        {
            text.Append(',').Append(key).Append('=').Append(value);
        }

        return text.ToString();
    }

    /// <inheritdoc />
    public bool Equals(ShapeStep? other) => other is not null && other.ToString() == ToString();

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as ShapeStep);

    /// <inheritdoc />
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(ToString());

    private string What(string key) => key.Length == 0 ? $"{Name}'s value" : $"{Name}'s {key}";
}

/// <summary>
/// Shapes in the order they are drawn, each a registered <see cref="IShape"/>. Written as text, comma-separated:
/// <c>box,grid=3x4,in=last,widths=1:2:1:1</c>; an item whose name is an option of the step before it (its own or a
/// common one, <see cref="Shapes.CommonKeys"/>) belongs to that step. As JSON: that text, or an array whose items are such
/// texts or objects <c>{"name": "grid", "value": "3x4", "widths": "1:2:1:1"}</c>. An unknown name or option is an
/// <see cref="ArgumentException"/> naming what is registered. Immutable; equal when their text is.
/// </summary>
public sealed class ShapePipeline : IEquatable<ShapePipeline>
{
    /// <summary>The steps <paramref name="steps"/>, checked against the registry.</summary>
    /// <exception cref="ArgumentException">A step names no registered shape, or one that refuses its value or options.</exception>
    public ShapePipeline(IEnumerable<ShapeStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);
        Steps = [.. steps];
        foreach (var step in Steps)
        {
            Check(Shapes.Find(step.Name) ?? throw Unknown(step.Name), step);
        }
    }

    /// <summary>No steps: nothing drawn.</summary>
    public static ShapePipeline Empty { get; } = new([]);

    /// <summary>The steps, in order.</summary>
    public IReadOnlyList<ShapeStep> Steps { get; }

    /// <summary>Whether there are no steps.</summary>
    public bool IsEmpty => Steps.Count == 0;

    /// <summary>Checks <paramref name="step"/> as <paramref name="shape"/> takes it: its options known, the common ones good, its value and own options as the shape checks them.</summary>
    /// <exception cref="ArgumentException">An option it does not take, or a bad value.</exception>
    public static void Check(IShape shape, ShapeStep step)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(step);
        step.ThrowIfUnknownOptions(shape.Keys);
        step.CheckCommon();
        shape.Check(step);
    }

    /// <summary>The pipeline written as text (see the class); null, empty or "none" give <see cref="Empty"/>.</summary>
    /// <exception cref="ArgumentException">An unknown shape or option, a bad value, or an item that is not NAME or NAME=VALUE.</exception>
    public static ShapePipeline Parse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return Empty;
        }

        var steps = new List<(string Name, string? Value, List<KeyValuePair<string, string>> Options)>();
        foreach (string raw in text.Split(','))
        {
            string item = raw.Trim();
            if (item.Length == 0)
            {
                continue;
            }

            int at = item.IndexOf('=', StringComparison.Ordinal);
            string name = (at < 0 ? item : item[..at]).Trim().ToLowerInvariant();
            string? value = at < 0 ? null : item[(at + 1)..].Trim();
            if (name.Length == 0)
            {
                throw new ArgumentException($"A shape is NAME or NAME=VALUE (such as grid=3x4), not '{item}'.");
            }

            if (Shapes.Find(name) is not null)
            {
                steps.Add((name, value, []));
                continue;
            }

            if (steps.Count > 0 && value is not null && Takes(Shapes.Get(steps[^1].Name), name))
            {
                steps[^1].Options.Add(KeyValuePair.Create(name, value));
                continue;
            }

            throw Unknown(name, steps.Count > 0 ? steps[^1].Name : null);
        }

        return new ShapePipeline(steps.Select(s => new ShapeStep(s.Name, s.Value, s.Options)));
    }

    /// <summary>The pipeline in JSON: null, a string (as <see cref="Parse"/>), or an array of strings and objects (<c>{"name", "value", ...options}</c>).</summary>
    /// <exception cref="ArgumentException">An unknown shape or option, a bad value, or JSON of another shape.</exception>
    public static ShapePipeline FromJson(JsonNode? json)
    {
        switch (json)
        {
            case null:
                return Empty;
            case JsonValue value when value.GetValueKind() == JsonValueKind.String:
                return Parse(value.GetValue<string>());
            case JsonArray array:
            {
                var steps = new List<ShapeStep>();
                foreach (var item in array)
                {
                    if (item is JsonValue text && text.GetValueKind() == JsonValueKind.String)
                    {
                        steps.AddRange(Parse(text.GetValue<string>()).Steps);
                    }
                    else if (item is JsonObject step)
                    {
                        string name = step["name"] is JsonValue n && n.GetValueKind() == JsonValueKind.String ? n.GetValue<string>()
                            : throw new ArgumentException($"A shape object needs a \"name\": {step.ToJsonString()}.");
                        if (Shapes.Find(name) is null)
                        {
                            throw Unknown(name);
                        }

                        var options = step.Where(p => p.Key is not "name" and not "value" && p.Value is not null)
                            .Select(p => KeyValuePair.Create(p.Key, Text(p.Value!, p.Key))).ToList();
                        steps.Add(new ShapeStep(name, step["value"] is { } v ? Text(v, "value") : null, options));
                    }
                    else
                    {
                        throw new ArgumentException($"A shape is a string or an object, not {item?.ToJsonString() ?? "null"}.");
                    }
                }

                return new ShapePipeline(steps);
            }

            default:
                throw new ArgumentException($"Shapes are a string or an array, not {json.ToJsonString()}.");
        }

        static string Text(JsonNode node, string key) => node.GetValueKind() switch
        {
            JsonValueKind.String => node.GetValue<string>(),
            JsonValueKind.Number => node.ToJsonString(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            var kind => throw new ArgumentException($"The shape's '{key}' is {kind.ToString().ToLowerInvariant()}: give a string, a number or true/false."),
        };
    }

    /// <summary>This pipeline's steps, then <paramref name="next"/>'s.</summary>
    public ShapePipeline Then(ShapePipeline? next) =>
        next is null || next.IsEmpty ? this : IsEmpty ? next : new ShapePipeline(Steps.Concat(next.Steps));

    /// <summary>The pipeline as JSON text (a string; null when empty).</summary>
    public JsonNode? ToJson() => IsEmpty ? null : JsonValue.Create(ToString());

    /// <summary>The steps written as text, comma-separated; "" when empty.</summary>
    public override string ToString() => string.Join(",", Steps);

    /// <inheritdoc />
    public bool Equals(ShapePipeline? other) => other is not null && other.ToString() == ToString();

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as ShapePipeline);

    /// <inheritdoc />
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(ToString());

    // Whether `key` is an option `shape` takes: its own or a common one.
    internal static bool Takes(IShape shape, string key) => shape.Keys.Contains(key) || Shapes.CommonKeys.Contains(key);

    private static ArgumentException Unknown(string name, string? after = null)
    {
        string also = after is not null && Shapes.Find(after) is { } owner
            ? $"; after {after} it can also be one of its options: {(owner.Keys.Count > 0 ? string.Join(", ", owner.Keys) + ", " : "")}{Shapes.CommonKeys}" : "";
        return new ArgumentException($"Unknown shape '{name}' (registered: {string.Join(", ", Shapes.Names)}{also}).");
    }
}

/// <summary>
/// The shapes a pipeline can name, by name (ignoring case). Built in (Idrak.Imaging): <c>line</c>, <c>polyline</c>,
/// <c>polygon</c>, <c>box</c>, <c>grid</c>, <c>circle</c>, <c>ellipse</c>, <c>arc</c>, <c>arrow</c>, <c>fill</c> and
/// <c>hatch</c>. Every shape also takes the common options (<see cref="CommonKeys"/>): <c>at=X:Y</c> and <c>size=W:H</c>
/// (shares of the box it is placed in, X from the start side), <c>in=page|last</c> (<c>last</c>: drawn once inside each
/// region the step before made), <c>margin</c>, <c>width</c> (a factor of the stroke width), <c>dash</c>
/// (solid|dashed|dotted), <c>tone</c>, <c>flow</c> (page|down: stacked below the parts before it, its height from what it
/// holds), <c>gap</c> (below a stacked step, line heights), <c>height</c> (a stacked step's own, line heights),
/// <c>lines</c> (N|rest: how many of the page's lines it takes) and <c>holds</c> (lines|paragraphs). Built in also:
/// <c>text</c> (no lines drawn, one region of wrapped paragraphs). Register another with <see cref="Register"/>; one under a library name shadows
/// the library's, which <see cref="Unregister"/> brings back, guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class Shapes
{
    private static readonly KindTable<IShape> Table = new(nameof(Shapes), "Shape", "add it with Shapes.Register",
        (slot, app, library) => new GuardedShape(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static Shapes() => LibraryDefaults.Ensure(typeof(Shapes));

    /// <summary>The options every shape takes besides its own: at, size, in, margin, width, dash, tone, flow, gap, height, lines, holds.</summary>
    public static OptionKeys CommonKeys { get; } = new("at", "size", "in", "margin", "width", "dash", "tone", "flow", "gap", "height", "lines", "holds");

    /// <summary>Registers <paramref name="shape"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    /// <exception cref="ArgumentException">A name with ',' or '=', or one that is a common option's key.</exception>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IShape shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentException.ThrowIfNullOrWhiteSpace(shape.Name);
        if (shape.Name.Contains(',') || shape.Name.Contains('=') || shape.Name.Contains('@') || CommonKeys.Contains(shape.Name.ToLowerInvariant()))
        {
            throw new ArgumentException($"A shape's name has no ',', '=' or '@' and is not a common option ({CommonKeys}): '{shape.Name}'.", nameof(shape));
        }

        Table.Register(shape, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's shape <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The shape registered as <paramref name="name"/>, or null.</summary>
    public static IShape? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The shape registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IShape Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's shape <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IShape? Default(string name) => Table.Default(name);

    /// <summary>Who registered the shape <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's shape <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> plans with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's plan
    /// and compares the app's on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered shape as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    private sealed class GuardedShape(Slot slot, IShape app, IShape library) : IShape
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public IReadOnlySet<string> WholeKeys => app.WholeKeys;

        public void Check(ShapeStep step) => app.Check(step);

        public ShapePlan Plan(ShapeStep step, ShapeContext context) =>
            slot.Call(() => app.Plan(step, context), () => library.Plan(step, context), (a, b) => Comparisons.Exact(a.ToString(), b.ToString()));
    }
}
