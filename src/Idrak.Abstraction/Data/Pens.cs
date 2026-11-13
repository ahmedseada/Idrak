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
/// same stroke, and no point strays from the path further than <see cref="Stray"/> says. Register one with
/// <see cref="Pens.Register"/>.
/// </summary>
public interface IPen : IConfigurableKind
{
    /// <summary>The pen's settings from <paramref name="options"/> (its own keys, <see cref="IConfigurableKind.Keys"/>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad value.</exception>
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
/// The pens, by name (ignoring case). Built in (Idrak.Imaging): <c>ideal</c> (the path exactly, at its width) and
/// <c>ballpoint</c> (a hand's ballpoint: a slow wobble, a little jitter, pressure along the stroke, an overshoot or a
/// short stop at its ends, a slight drift; its keys pen-wobble, pen-jitter, pen-pressure, pen-overshoot and pen-tilt).
/// Register another with <see cref="Register"/>; one under a library name shadows the library's, which
/// <see cref="Unregister"/> brings back, guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class Pens
{
    /// <summary>The pen that draws a path exactly.</summary>
    public const string Ideal = "ideal";

    private static readonly KindTable<IPen> Table = new(nameof(Pens), "Pen", "add it with Pens.Register",
        (slot, app, library) => new GuardedPen(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static Pens() => LibraryDefaults.Ensure(typeof(Pens));

    /// <summary>Registers <paramref name="pen"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPen pen) => Table.Register(pen, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's pen <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The pen registered as <paramref name="name"/>, or null.</summary>
    public static IPen? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The pen registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IPen Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's pen <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IPen? Default(string name) => Table.Default(name);

    /// <summary>Who registered the pen <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's pen <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> draws with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's draw and
    /// compares the app's on a sample of strokes).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered pen as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The keys of every registered pen, each once, in registration order.</summary>
    public static IReadOnlyList<string> AllKeys() => [.. Table.Kinds.SelectMany(p => p.Keys).Distinct(StringComparer.OrdinalIgnoreCase)];

    // The app's settings and the library's, read from the same options (the library's null when it refuses them).
    private sealed class GuardedSettings(PenSettings app, PenSettings? library, IReadOnlyDictionary<string, string> options) : PenSettings(options)
    {
        public PenSettings App => app;

        public PenSettings? Library => library;
    }

    // An app's pen over the library's: each stroke falls back on its own, or is compared with the library's.
    private sealed class GuardedPen(Slot slot, IPen app, IPen library) : IPen
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public PenSettings Read(IReadOnlyDictionary<string, string> options)
        {
            var own = app.Read(options);
            PenSettings? theirs;
            try
            {
                theirs = library.Read(PageOptions.Only(options, library.Keys));
            }
            catch (ArgumentException)
            {
                theirs = null;
            }

            return new GuardedSettings(own, theirs, options);
        }

        public float Stray(PenSettings settings, float width) => app.Stray(settings is GuardedSettings g ? g.App : settings, width);

        public void Stroke(ReadOnlySpan<PenPoint> path, bool closed, PenSettings settings, int seed, PenStroke stroke)
        {
            var (own, theirs) = settings is GuardedSettings g ? (g.App, g.Library) : (settings, settings);
            if (slot.Policy == SlotPolicy.Shadow && theirs is not null)
            {
                var run = slot.Shadow();
                library.Stroke(path, closed, theirs, seed, stroke);
                if (run is not null)
                {
                    run.Answered();
                    using var other = new PenStroke(stroke.Count);
                    try
                    {
                        app.Stroke(path, closed, own, seed, other);
                        run.Done(Compare(stroke, other));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return;
            }

            try
            {
                app.Stroke(path, closed, own, seed, stroke);
            }
            catch (Exception e) when (theirs is not null && slot.Failed(e))
            {
                library.Stroke(path, closed, theirs, seed, stroke);
            }
        }

        private static string? Compare(PenStroke a, PenStroke b)
        {
            if (a.Count != b.Count)
            {
                return string.Create(CultureInfo.InvariantCulture, $"{a.Count} points, {b.Count} from the override");
            }

            for (int i = 0; i < a.Count; i++)
            {
                var (p, q) = (a.Points[i], b.Points[i]);
                if (Comparisons.Close(p.X, q.X, 1e-3f) && Comparisons.Close(p.Y, q.Y, 1e-3f) && Comparisons.Close(p.Width, q.Width, 1e-3f))
                {
                    continue;
                }

                return string.Create(CultureInfo.InvariantCulture, $"point {i}: ({p.X:F3}, {p.Y:F3}) {p.Width:F3} wide, the override's ({q.X:F3}, {q.Y:F3}) {q.Width:F3}");
            }

            return null;
        }
    }
}
