// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;

namespace Idrak.Imaging;

/// <summary>"none": every glyph as set (the identity).</summary>
internal sealed class NonePose : IGlyphPose
{
    public string Name => GlyphPoses.None;

    public string Summary => "every glyph as set (no drift, slant or noise)";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public GlyphPoseSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        foreach (string key in options.Keys)
        {
            throw new ArgumentException($"The glyph pose none takes no options (given '{key}').");
        }

        return GlyphPoseSettings.None;
    }

    public GlyphPoseBounds Bounds(GlyphPoseSettings settings) => GlyphPoseBounds.None;

    public GlyphPose Pose(in GlyphPoseInput glyph, GlyphPoseSettings settings) => GlyphPose.Identity;
}

/// <summary>
/// "drift": a hand's writing along a line. The baseline, the slant, the turn and the size drift slowly with the distance
/// along the line (smooth noise, each line its own), the slant about a page's own (pose-slant: a range of slants, x per
/// unit of height), the baseline by up to pose-baseline (ems), the turn by up to pose-rotate (degrees), the size by up to
/// pose-size (a ratio) and the ink by up to pose-pressure (a share); each cluster adds a little noise of its own to its
/// size, its ink and (within a quarter of pose-baseline) its place along the line, never to its height, slant or turn,
/// so joined letters stay joined. The amplitudes are drawn per page from their ranges.
/// </summary>
internal sealed class DriftPose : IGlyphPose
{
    private static readonly OptionKeys Own = new("pose-slant", "pose-size", "pose-baseline", "pose-rotate", "pose-pressure");

    private static readonly Settings Defaults = new(PageOptions.None, (-0.05f, 0.05f), (0f, 0.04f), (0f, 0.04f), (0f, 1.2f), (0f, 0.2f));

    // The channels of the smooth noise and of a cluster's own.
    private const int Baseline = 0, Slant = 1, Turn = 2, Size = 3, Ink = 4, Along = 5;

    public string Name => "drift";

    public string Summary => "a hand's writing: the baseline, slant, turn and size drift along each line, a little noise per cluster in size, ink and place (pose-slant, pose-size, pose-baseline, pose-rotate, pose-pressure: ranges)";

    public IReadOnlySet<string> Keys => Own;

    public GlyphPoseSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var (slant, size, baseline, rotate, pressure) = (Defaults.Slant, Defaults.Size, Defaults.Baseline, Defaults.Rotate, Defaults.Pressure);
        foreach (var (raw, value) in options)
        {
            string key = raw.ToLowerInvariant();
            string what = $"Page setting {key}";
            switch (key)
            {
                case "pose-slant":
                    slant = PageOptions.Range(what, value, -1, 1);
                    break;
                case "pose-size":
                    size = PageOptions.Range(what, value, 0, 0.5f);
                    break;
                case "pose-baseline":
                    baseline = PageOptions.Range(what, value, 0, 0.5f);
                    break;
                case "pose-rotate":
                    rotate = PageOptions.Range(what, value, 0, 20);
                    break;
                case "pose-pressure":
                    pressure = PageOptions.Range(what, value, 0, 0.9f);
                    break;
                default:
                    throw new ArgumentException($"The glyph pose drift does not take '{raw}'; it takes: {Own}.");
            }
        }

        return new Settings(options, slant, size, baseline, rotate, pressure);
    }

    public GlyphPoseBounds Bounds(GlyphPoseSettings settings)
    {
        var s = settings as Settings ?? Defaults;
        return new GlyphPoseBounds(s.Baseline.Max, s.Size.Max, Math.Max(Math.Abs(s.Slant.Min), Math.Abs(s.Slant.Max)), (float)(s.Rotate.Max * Math.PI / 180), 1 - s.Pressure.Max);
    }

    public GlyphPose Pose(in GlyphPoseInput glyph, GlyphPoseSettings settings)
    {
        var s = settings as Settings ?? Defaults;
        int seed = glyph.Seed, line = glyph.Line;
        double size = Math.Max(1e-3, glyph.Size), along = glyph.X / size;

        // The page's own amplitudes, drawn from their ranges.
        double slant = Draw(s.Slant, Unit(seed, -1, 0, Slant)), sizeAmp = Draw(s.Size, Unit(seed, -1, 0, Size));
        double baseAmp = Draw(s.Baseline, Unit(seed, -1, 0, Baseline)), turnAmp = Draw(s.Rotate, Unit(seed, -1, 0, Turn)) * Math.PI / 180;
        double press = Draw(s.Pressure, Unit(seed, -1, 0, Ink));
        double bound = Math.Max(Math.Abs(s.Slant.Min), Math.Abs(s.Slant.Max)), span = s.Slant.Max - s.Slant.Min;

        double dy = baseAmp * size * Smooth(seed, line, along, Baseline);
        double dx = 0.25 * baseAmp * size * Noise(seed, line, glyph.Cluster, Along);
        double shear = Math.Clamp(slant + 0.25 * span * Smooth(seed, line, along, Slant), -bound, bound);
        double turn = turnAmp * Smooth(seed, line, along, Turn);
        double scale = 1 + sizeAmp * (0.6 * Smooth(seed, line, along, Size) + 0.4 * Noise(seed, line, glyph.Cluster, Size));
        double ink = 1 - press * Math.Clamp(0.5 + 0.35 * Smooth(seed, line, along, Ink) + 0.15 * Noise(seed, line, glyph.Cluster, Ink), 0, 1);
        return new GlyphPose((float)dx, (float)dy, (float)scale, (float)shear, (float)turn, (float)ink);
    }

    private static double Draw((float Min, float Max) range, double u) => range.Min + u * (range.Max - range.Min);

    // Smooth noise along a line, from -1 to 1: two waves 6 to 14 and 4 to 7 ems long, each line and channel its own.
    private static double Smooth(int seed, int line, double along, int channel)
    {
        double first = 6 + 8 * Unit(seed, line, 0, 16 + channel), second = 4 + 3 * Unit(seed, line, 1, 16 + channel);
        double a = 2 * Math.PI * Unit(seed, line, 2, 16 + channel), b = 2 * Math.PI * Unit(seed, line, 3, 16 + channel);
        return 0.6 * PortableMath.Sin(2 * Math.PI * along / first + a) + 0.4 * PortableMath.Sin(2 * Math.PI * along / second + b);
    }

    // A cluster's own noise, from -1 to 1.
    private static double Noise(int seed, int line, int cluster, int channel) => 2 * Unit(seed, line, cluster, 32 + channel) - 1;

    // A number from 0 to 1 hashed from its four keys (SplitMix64; the same in every process).
    private static double Unit(int seed, int line, int key, int channel)
    {
        ulong z = ((ulong)(uint)seed << 32 | (uint)line) * 0x9E3779B97F4A7C15UL + ((ulong)(uint)key << 8 | (uint)channel);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 11) * (1.0 / (1UL << 53));
    }

    // The ranges read from a page's options.
    private sealed class Settings(IReadOnlyDictionary<string, string> given, (float Min, float Max) slant, (float Min, float Max) size, (float Min, float Max) baseline,
        (float Min, float Max) rotate, (float Min, float Max) pressure) : GlyphPoseSettings(given)
    {
        public (float Min, float Max) Slant { get; } = slant;

        public (float Min, float Max) Size { get; } = size;

        public (float Min, float Max) Baseline { get; } = baseline;

        public (float Min, float Max) Rotate { get; } = rotate;

        public (float Min, float Max) Pressure { get; } = pressure;
    }
}

/// <summary>
/// Draws laid-out text with a page's glyph pose, the one way every layout draws text: read once per page
/// (<see cref="Read"/>), each line a running number, each cluster one pose (a mark moves with its base) about its pen
/// origin on the baseline, the outline's points transformed and filled by the outline rasterizer, below full ink through
/// a pooled layer of its own. With "none" it is exactly <see cref="TextRenderer.Draw"/>.
/// </summary>
internal sealed class GlyphPainter
{
    private readonly IGlyphPose _pose;
    private readonly GlyphPoseSettings _settings;
    private readonly int _seed;
    private readonly Dictionary<int, float[]> _points = [];       // transformed points, one array per length, reused
    private int _line;

    public GlyphPainter(IGlyphPose pose, GlyphPoseSettings settings, int seed)
    {
        _pose = pose;
        _settings = settings;
        _seed = seed;
        Bounds = pose.Bounds(settings);
        IsIdentity = pose is NonePose;
    }

    /// <summary>The bounds of the page's pose.</summary>
    public GlyphPoseBounds Bounds { get; }

    /// <summary>Whether glyphs are drawn as set (the "none" pose).</summary>
    public bool IsIdentity { get; }

    /// <summary>The pose's name.</summary>
    public string Name => _pose.Name;

    /// <summary>The pose a page's options choose (<c>pose</c>, "none" by default) with its own keys, read once.</summary>
    /// <exception cref="ArgumentException">A bad pose setting.</exception>
    /// <exception cref="NotSupportedException">A pose that is not registered.</exception>
    public static GlyphPainter Read(IReadOnlyDictionary<string, string> options, int seed)
    {
        var pose = GlyphPoses.Get(options.TryGetValue("pose", out var name) && name.Length > 0 ? name : GlyphPoses.None);
        return new GlyphPainter(pose, pose.Read(PageOptions.Only(options, pose.Keys)), seed);
    }

    /// <summary>The most a posed glyph's ink moves, in pixels, for text of <paramref name="size"/> whose outlines reach <paramref name="reach"/> pixels from their origins (0 for none).</summary>
    public float Reach(float size, float reach) => IsIdentity ? 0 : Bounds.Reach(size, reach);

    /// <summary>Adds <paramref name="block"/>'s glyphs, its top-left at (<paramref name="x"/>, <paramref name="y"/>), to <paramref name="layer"/>; each of its lines takes the page's next line number.</summary>
    public void Draw(TextBlock block, float x, float y, Span<float> layer, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(block);
        if (IsIdentity)
        {
            TextRenderer.Draw(block, x, y, layer, width, height);
            _line += block.Lines.Count;
            return;
        }

        var fill = OutlineRasterizers.Get();
        float cos = 1, sin = 0;
        var last = GlyphPose.Identity;
        foreach (var (g, pose) in Posed(block, _line))
        {
            if (pose != last)
            {
                (last, cos, sin) = (pose, (float)PortableMath.Cos(pose.Rotation), (float)PortableMath.Sin(pose.Rotation));
            }

            var outline = g.Font.Outline(g.Glyph);
            if (!outline.IsEmpty)
            {
                Fill(fill, Transform(outline, g.Scale, pose, cos, sin), x + g.X, y + g.Y, layer, width, height, pose.Ink);
            }
        }

        _line += block.Lines.Count;
    }

    /// <summary>
    /// Each glyph of <paramref name="block"/> in drawing order with its pose, its lines numbered from
    /// <paramref name="firstLine"/>: one pose a cluster, asked with the place along the line of the cluster's first glyph.
    /// </summary>
    public IEnumerable<(PlacedGlyph Glyph, GlyphPose Pose)> Posed(TextBlock block, int firstLine)
    {
        int number = firstLine;
        foreach (var line in block.Lines)
        {
            int cluster = int.MinValue;
            var pose = GlyphPose.Identity;
            foreach (var g in line.Glyphs)
            {
                if (g.Cluster != cluster)
                {
                    cluster = g.Cluster;
                    pose = IsIdentity ? GlyphPose.Identity : _pose.Pose(new GlyphPoseInput(number, g.X, g.Cluster, block.Style.Size, _seed), _settings);
                }

                yield return (g, pose);
            }

            number++;
        }
    }

    /// <summary>The outline in pixels about its pen origin (y up, as outlines are), scaled, slanted, turned and moved by <paramref name="pose"/>.</summary>
    public GlyphOutline Transform(GlyphOutline outline, float scale, GlyphPose pose, float cos, float sin)
    {
        var source = outline.Points.Span;
        if (!_points.TryGetValue(source.Length, out var points))
        {
            _points[source.Length] = points = new float[source.Length];
        }

        float k = pose.Scale * scale;
        for (int i = 0; i < source.Length; i += 2)
        {
            float u = source[i] * k, v = source[i + 1] * k;
            u += pose.Shear * v;
            points[i] = u * cos - v * sin + pose.Dx;
            points[i + 1] = u * sin + v * cos - pose.Dy;
        }

        var verbs = MemoryMarshal.TryGetArray(outline.Verbs, out var segment) && segment.Offset == 0 && segment.Count == segment.Array!.Length
            ? segment.Array : outline.Verbs.ToArray();
        return new GlyphOutline(verbs, points);
    }

    // The outline (pixels about its origin) at (x, y): straight onto the layer at full ink, else through a pooled layer of
    // its own laid on at the ink's share, row by row.
    private static void Fill(IOutlineRasterizer fill, GlyphOutline outline, float x, float y, Span<float> layer, int width, int height, float ink)
    {
        if (ink >= 1f)
        {
            fill.Fill(outline, 1f, x, y, layer, width, height);
            return;
        }

        if (!(ink > 0))
        {
            return;
        }

        var (minX, minY, maxX, maxY) = outline.Bounds();
        int left = Math.Max(0, (int)MathF.Floor(x + minX) - 1), right = Math.Min(width, (int)MathF.Ceiling(x + maxX) + 2);
        int top = Math.Max(0, (int)MathF.Floor(y - maxY) - 1), bottom = Math.Min(height, (int)MathF.Ceiling(y - minY) + 2);
        if (right <= left || bottom <= top)
        {
            return;
        }

        int w = right - left, h = bottom - top;
        float[] own = ArrayPool<float>.Shared.Rent(w * h);
        try
        {
            var coverage = own.AsSpan(0, w * h);
            coverage.Clear();
            fill.Fill(outline, 1f, x - left, y - top, coverage, w, h);
            for (int row = 0; row < h; row++)
            {
                var source = coverage.Slice(row * w, w);
                var target = layer.Slice((top + row) * width + left, w);
                for (int i = 0; i < w; i++)
                {
                    float c = source[i] * ink, t = target[i];
                    target[i] = t + c - t * c;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(own);
        }
    }
}
