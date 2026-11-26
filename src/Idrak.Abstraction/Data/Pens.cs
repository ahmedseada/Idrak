// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>A point of a stroke: where it is (pixels), how wide the stroke is there, and how much ink it leaves (0 to 1).</summary>
/// <param name="X">Its x, in pixels.</param>
/// <param name="Y">Its y, in pixels (down).</param>
/// <param name="Width">The stroke's width there, in pixels (above 0).</param>
/// <param name="Ink">How much ink the pen leaves there, 0 to 1 (its pressure).</param>
public readonly record struct PenPoint(float X, float Y, float Width, float Ink = 1f);

/// <summary>
/// A pen's settings, read once per page from its options (<see cref="IPen.Read"/>): the options as given, and whatever a
/// pen parses them into (a pen's own settings derive from this class).
/// </summary>
public class PenSettings
{
    /// <summary>Settings of <paramref name="options"/> (none when null).</summary>
    public PenSettings(IReadOnlyDictionary<string, string>? options = null) => Options = options ?? PageOptions.None;

    /// <summary>No options: a pen's defaults.</summary>
    public static PenSettings None { get; } = new();

    /// <summary>The options as given, by key.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }
}

/// <summary>
/// The points a pen draws for one path, in a pooled buffer that grows as needed and is reused stroke after stroke: the
/// caller makes one per page and disposes it (the buffer goes back to the pool).
/// </summary>
public sealed class PenStroke : IDisposable
{
    private PenPoint[] _points;

    /// <summary>An empty stroke with room for <paramref name="capacity"/> points.</summary>
    public PenStroke(int capacity = 64) => _points = ArrayPool<PenPoint>.Shared.Rent(Math.Max(4, capacity));

    /// <summary>How many points there are.</summary>
    public int Count { get; private set; }

    /// <summary>Whether the last point joins the first.</summary>
    public bool Closed { get; set; }

    /// <summary>The points, in drawing order (valid until the next change).</summary>
    public ReadOnlySpan<PenPoint> Points => _points.AsSpan(0, Count);

    /// <summary>Empties the stroke (its buffer kept).</summary>
    public void Clear()
    {
        Count = 0;
        Closed = false;
    }

    /// <summary>Adds <paramref name="point"/>.</summary>
    public void Add(in PenPoint point)
    {
        ObjectDisposedException.ThrowIf(_points.Length == 0, this);
        if (Count == _points.Length)
        {
            var bigger = ArrayPool<PenPoint>.Shared.Rent(_points.Length * 2);
            _points.AsSpan(0, Count).CopyTo(bigger);
            ArrayPool<PenPoint>.Shared.Return(_points);
            _points = bigger;
        }

        _points[Count++] = point;
    }

    /// <summary>The ink of the stroke's points, on average (1 when it has none).</summary>
    public float MeanInk()
    {
        if (Count == 0)
        {
            return 1;
        }

        float sum = 0;
        foreach (var p in Points)
        {
            sum += p.Ink;
        }

        return sum / Count;
    }

    /// <summary>Gives the buffer back to the pool.</summary>
    public void Dispose()
    {
        if (_points.Length > 0)
        {
            ArrayPool<PenPoint>.Shared.Return(_points);
            _points = [];
            Count = 0;
        }
    }
}

/// <summary>
/// The stage from geometry to strokes: a pen turns an ideal path (a polyline with a width per point) into the stroke it
/// draws, points with a width and an ink per point, as a hand with a ballpoint would (a slow wobble, a little jitter,
/// pressure along the stroke, an overshoot at its ends) or exactly (<c>ideal</c>). Its settings are read once per page
/// (<see cref="Read"/>), then every stroke of the page is drawn with them. The same path, settings and seed give the
/// same stroke, and no point strays from the path further than <see cref="Stray"/> says.
/// </summary>
public interface IPen
{
    /// <summary>The pen's settings from <paramref name="options"/> (its own keys, <c>Keys</c>;
    /// defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad
    /// value.</exception>
    PenSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>The most a point of a stroke drawn with <paramref name="settings"/> strays from its ideal path, in pixels, for a stroke <paramref name="width"/> pixels wide.</summary>
    float Stray(PenSettings settings, float width);

    /// <summary>
    /// Draws <paramref name="path"/> (open, or <paramref name="closed"/>) into <paramref name="stroke"/>, emptied first
    /// (its <see cref="PenStroke.Closed"/> set): the same path, settings and <paramref name="seed"/>, the same points.
    /// </summary>
    void Stroke(ReadOnlySpan<PenPoint> path, bool closed, PenSettings settings, int seed, PenStroke stroke);
}

/// <summary>
/// The pens. The library's (Idrak.Imaging): <c>ideal</c> (the path exactly, at its width) and
/// <c>ballpoint</c> (a hand's ballpoint: a slow wobble, a little jitter, pressure along the stroke, an overshoot or a
/// short stop at its ends, a slight drift; its keys pen-wobble, pen-jitter, pen-pressure, pen-overshoot and pen-tilt).
/// Picking one by its name is not on the new design yet.
/// </summary>
public static class Pens
{
    /// <summary>The pen that draws a path exactly.</summary>
    public const string Ideal = "ideal";
}
