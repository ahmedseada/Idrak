// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// A shading's colours on the surface (ISO 32000-2, 8.7.4.5): an axial or radial one through a table of its colour along t
/// (1,024 steps, read between them), extended past its ends when asked; a function-based one asked at each pixel's centre
/// within its rectangle; a mesh's triangles filled at each pixel centre they hold, their corners' colours blended.
/// Outside its extent: nothing, or its background when it is a path's paint.
/// </summary>
internal sealed class ShadingPaint
{
    private const int Steps = 1024;

    private readonly DocumentShading _shading;
    private readonly Affine _toShading;          // surface pixels to the shading's space
    private readonly float _alpha;
    private readonly (float R, float G, float B)? _background;
    private readonly float[]? _table;            // axial and radial: red, green, blue at each step of t
    private readonly float[]? _triangles;        // a mesh: x, y (pixels), r, g, b a corner
    private readonly float _x0, _y0, _x1, _y1, _r0, _r1;

    private ShadingPaint(DocumentShading shading, Affine toPixels, float alpha, bool background)
    {
        _shading = shading;
        _alpha = alpha;
        _toShading = toPixels.Invertible ? toPixels.Inverse() : new Affine(0, 0, 0, 0, float.NaN, float.NaN);
        _background = background && shading.Background is { } b ? (b.Red, b.Green, b.Blue) : null;
        var c = shading.Coords.Span;
        switch (shading.Kind)
        {
            case DocumentShadingKind.Axial when c.Length >= 4:
                (_x0, _y0, _x1, _y1) = (c[0], c[1], c[2], c[3]);
                _table = Table(shading);
                break;
            case DocumentShadingKind.Radial when c.Length >= 6:
                (_x0, _y0, _r0, _x1, _y1, _r1) = (c[0], c[1], c[2], c[3], c[4], c[5]);
                _table = Table(shading);
                break;
            case DocumentShadingKind.Function when c.Length >= 4:
                (_x0, _x1, _y0, _y1) = (c[0], c[1], c[2], c[3]);
                break;
            case DocumentShadingKind.Mesh:
                _triangles = shading.Triangles.ToArray();
                break;
        }
    }

    /// <summary><paramref name="shading"/> on a surface placed by <paramref name="map"/> (page points to pixels), at <paramref name="alpha"/>; its background used when <paramref name="background"/>.</summary>
    public static ShadingPaint Of(DocumentShading shading, Affine map, float alpha, bool background)
    {
        var paint = new ShadingPaint(shading, Affine.Of(shading.Matrix).Then(map), alpha, background);
        if (paint._triangles is { } t)
        {
            // A mesh's corners are on the page: into the surface's pixels.
            for (int i = 0; i + 4 < t.Length; i += 5)
            {
                var (x, y) = map.Apply(t[i], t[i + 1]);
                (t[i], t[i + 1]) = ((float)x, (float)y);
            }
        }

        return paint;
    }

    /// <summary>The pixels it can paint: a mesh's triangles' box, else the whole surface.</summary>
    public (int Left, int Top, int Right, int Bottom) Bounds(int width, int height)
    {
        if (_triangles is not { } t)
        {
            return (0, 0, width, height);
        }

        float minX = float.PositiveInfinity, minY = float.PositiveInfinity, maxX = float.NegativeInfinity, maxY = float.NegativeInfinity;
        for (int i = 0; i + 4 < t.Length; i += 5)
        {
            (minX, maxX, minY, maxY) = (MathF.Min(minX, t[i]), MathF.Max(maxX, t[i]), MathF.Min(minY, t[i + 1]), MathF.Max(maxY, t[i + 1]));
        }

        return float.IsFinite(minX) ? ((int)MathF.Floor(minX), (int)MathF.Floor(minY), (int)MathF.Ceiling(maxX) + 1, (int)MathF.Ceiling(maxY) + 1) : (0, 0, 0, 0);
    }

    public void Colours(SceneWork work, int x0, int x1, int y0, int y1)
    {
        int w = work.Width;
        var colour = work.Colour;
        if (_triangles is { } triangles)
        {
            for (int y = y0; y < y1; y++)
            {
                int row = (y - work.Top) * w;
                for (int x = x0; x < x1; x++)
                {
                    Outside(colour, 4 * (row + x));
                }
            }

            Mesh(work, triangles, x0, x1, y0, y1);
            return;
        }

        for (int y = y0; y < y1; y++)
        {
            int row = (y - work.Top) * w;
            double py = y + 0.5;
            for (int x = x0; x < x1; x++)
            {
                int k = 4 * (row + x);
                if (work.Coverage[row + x] <= 0f)
                {
                    continue;
                }

                var (sx, sy) = _toShading.Apply(x + 0.5, py);
                if (!Colour(sx, sy, colour, k))
                {
                    Outside(colour, k);
                }
            }
        }
    }

    // The colour at (sx, sy) of the shading's space into colour[k..]; false outside its extent.
    private bool Colour(double sx, double sy, float[] colour, int k)
    {
        switch (_shading.Kind)
        {
            case DocumentShadingKind.Axial when _table is not null:
            {
                double dx = _x1 - _x0, dy = _y1 - _y0, length = dx * dx + dy * dy;
                double s = length > 0 ? ((sx - _x0) * dx + (sy - _y0) * dy) / length : 0;
                if (!Extended(ref s))
                {
                    return false;
                }

                Lookup(s, colour, k);
                return true;
            }

            case DocumentShadingKind.Radial when _table is not null:
            {
                if (!Radial(sx, sy, out double s))
                {
                    return false;
                }

                Lookup(s, colour, k);
                return true;
            }

            case DocumentShadingKind.Function when _shading.Colour is { } f:
            {
                if (!(sx >= Math.Min(_x0, _x1) && sx <= Math.Max(_x0, _x1) && sy >= Math.Min(_y0, _y1) && sy <= Math.Max(_y0, _y1)))
                {
                    return false;
                }

                var c = f((float)sx, (float)sy);
                (colour[k], colour[k + 1], colour[k + 2], colour[k + 3]) = (c.Red, c.Green, c.Blue, _alpha * c.Alpha);
                return true;
            }

            default:
                return false;
        }
    }

    // s within [0, 1], or past an end that is extended (then held at that end); false otherwise.
    private bool Extended(ref double s)
    {
        if (s < 0)
        {
            if (!_shading.ExtendStart)
            {
                return false;
            }

            s = 0;
        }
        else if (s > 1)
        {
            if (!_shading.ExtendEnd)
            {
                return false;
            }

            s = 1;
        }

        return double.IsFinite(s);
    }

    // The largest s whose circle passes through the point with a radius not below 0, within [0, 1] or an extended end
    // (8.7.4.5.4).
    private bool Radial(double px, double py, out double s)
    {
        double cdx = _x1 - _x0, cdy = _y1 - _y0, dr = _r1 - _r0, pdx = px - _x0, pdy = py - _y0;
        double a = cdx * cdx + cdy * cdy - dr * dr, b = pdx * cdx + pdy * cdy + _r0 * dr, c = pdx * pdx + pdy * pdy - _r0 * _r0;
        Span<double> candidates = stackalloc double[2];
        int n = 0;
        if (Math.Abs(a) < 1e-9)
        {
            if (Math.Abs(b) > 1e-12)
            {
                candidates[n++] = c / (2 * b);
            }
        }
        else
        {
            double disc = b * b - a * c;
            if (disc >= 0)
            {
                double root = Math.Sqrt(disc);
                double s1 = (b + root) / a, s2 = (b - root) / a;
                candidates[n++] = Math.Max(s1, s2);
                candidates[n++] = Math.Min(s1, s2);
            }
        }

        for (int i = 0; i < n; i++)
        {
            double t = candidates[i];
            if (_r0 + t * dr < 0)
            {
                continue;
            }

            if ((t >= 0 && t <= 1) || (t < 0 && _shading.ExtendStart) || (t > 1 && _shading.ExtendEnd))
            {
                s = Math.Clamp(t, 0, 1);
                return true;
            }
        }

        s = 0;
        return false;
    }

    private void Lookup(double s, float[] colour, int k)
    {
        var table = _table!;
        double at = s * (Steps - 1);
        int i = Math.Clamp((int)at, 0, Steps - 2);
        float f = (float)(at - i);
        int a = 3 * i, b = a + 3;
        colour[k] = table[a] + (table[b] - table[a]) * f;
        colour[k + 1] = table[a + 1] + (table[b + 1] - table[a + 1]) * f;
        colour[k + 2] = table[a + 2] + (table[b + 2] - table[a + 2]) * f;
        colour[k + 3] = _alpha;
    }

    private void Outside(float[] colour, int k)
    {
        if (_background is { } b)
        {
            (colour[k], colour[k + 1], colour[k + 2], colour[k + 3]) = (b.R, b.G, b.B, _alpha);
        }
        else
        {
            colour[k + 3] = 0f;
        }
    }

    // An axial or radial shading's colour at each of the table's steps of t, from its domain's start to its end.
    private static float[] Table(DocumentShading shading)
    {
        var table = new float[3 * Steps];
        var f = shading.Colour!;
        for (int i = 0; i < Steps; i++)
        {
            float t = shading.DomainStart + (shading.DomainEnd - shading.DomainStart) * i / (Steps - 1);
            var c = f(t, 0f);
            (table[3 * i], table[3 * i + 1], table[3 * i + 2]) = (c.Red, c.Green, c.Blue);
        }

        return table;
    }

    // Each triangle's pixels (their centres inside it, edges included), coloured by their corners blended.
    private void Mesh(SceneWork work, float[] t, int x0, int x1, int y0, int y1)
    {
        int w = work.Width;
        var colour = work.Colour;
        for (int i = 0; i + 14 < t.Length; i += 15)
        {
            float ax = t[i], ay = t[i + 1], bx = t[i + 5], by = t[i + 6], cx = t[i + 10], cy = t[i + 11];
            float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if (!(MathF.Abs(area) > 1e-9f))
            {
                continue;
            }

            int left = Math.Max(x0, (int)MathF.Floor(MathF.Min(ax, MathF.Min(bx, cx)))), right = Math.Min(x1, (int)MathF.Ceiling(MathF.Max(ax, MathF.Max(bx, cx))) + 1);
            int top = Math.Max(y0, (int)MathF.Floor(MathF.Min(ay, MathF.Min(by, cy)))), bottom = Math.Min(y1, (int)MathF.Ceiling(MathF.Max(ay, MathF.Max(by, cy))) + 1);
            float inv = 1f / area;
            for (int y = top; y < bottom; y++)
            {
                float py = y + 0.5f;
                int row = (y - work.Top) * w;
                for (int x = left; x < right; x++)
                {
                    float px = x + 0.5f;
                    float wa = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv;
                    float wb = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv;
                    float wc = 1f - wa - wb;
                    if (wa < -1e-5f || wb < -1e-5f || wc < -1e-5f)
                    {
                        continue;
                    }

                    int k = 4 * (row + x);
                    colour[k] = wa * t[i + 2] + wb * t[i + 7] + wc * t[i + 12];
                    colour[k + 1] = wa * t[i + 3] + wb * t[i + 8] + wc * t[i + 13];
                    colour[k + 2] = wa * t[i + 4] + wb * t[i + 9] + wc * t[i + 14];
                    colour[k + 3] = _alpha;
                }
            }
        }
    }
}

/// <summary>
/// A tiling's colours. An upright tiling (its matrix neither turned nor slanted) is snapped to whole pixels as readers draw
/// tilings, so its cells meet with no seam: its cell drawn once into a tile of the cell's size on the surface rounded up,
/// the tiles repeated every that many pixels from where the pattern's origin falls, rounded. Any other tiling is drawn
/// into a tile at the surface's resolution (the cells around it too when the cell reaches past its steps), each surface
/// pixel's centre taken back into the pattern's space, wrapped by the steps and read from the tile.
/// </summary>
internal sealed class TilePaint
{
    private const int MaxTilePixels = 1 << 20;

    private readonly DocumentPattern _pattern;
    private readonly Affine _toPattern;
    private readonly float _alpha;
    private readonly bool _upright;
    private readonly int _startX, _startY;      // an upright tiling: where one tile starts on the surface
    private readonly Lazy<(float[] Pixels, int Width, int Height, float Scale)?> _tile;

    public TilePaint(DocumentPattern pattern, Affine map, float alpha, DrawContext context)
    {
        _pattern = pattern;
        _alpha = alpha;
        var toPixels = Affine.Of(pattern.Matrix).Then(map);
        _toPattern = toPixels.Invertible ? toPixels.Inverse() : new Affine(0, 0, 0, 0, double.NaN, double.NaN);
        double xs = Math.Abs(pattern.XStep), ys = Math.Abs(pattern.YStep);
        _upright = Math.Abs(toPixels.B) < 1e-9 && Math.Abs(toPixels.C) < 1e-9 && toPixels.Invertible && Math.Abs(toPixels.A * xs) * Math.Abs(toPixels.D * ys) <= MaxTilePixels;
        if (_upright)
        {
            // The cell (one step by the other from the bounds' corner) on the surface, its size rounded up.
            var (ax, ay) = toPixels.Apply(pattern.Bounds.X, pattern.Bounds.Y);
            var (bx, by) = toPixels.Apply(pattern.Bounds.X + xs, pattern.Bounds.Y + ys);
            double left = Math.Min(ax, bx), top = Math.Min(ay, by);
            int width = Math.Max(1, (int)Math.Ceiling(Math.Abs(bx - ax) - 1e-9)), height = Math.Max(1, (int)Math.Ceiling(Math.Abs(by - ay) - 1e-9));
            _startX = (int)Math.Floor(Math.Round(toPixels.E) + (left - toPixels.E));
            _startY = (int)Math.Floor(Math.Round(toPixels.F) + (top - toPixels.F));
            var cellMap = toPixels.Then(new Affine(1, 0, 0, 1, -left, -top));
            _tile = new Lazy<(float[], int, int, float)?>(() => DrawUpright(pattern, cellMap, width, height, context), LazyThreadSafetyMode.ExecutionAndPublication);
            return;
        }

        double scale = toPixels.Scale;
        _tile = new Lazy<(float[], int, int, float)?>(() => Draw(pattern, scale, context), LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public void Colours(SceneWork work, int x0, int x1, int y0, int y1)
    {
        int w = work.Width;
        var colour = work.Colour;
        var tile = _tile.Value;
        float xs = MathF.Abs(_pattern.XStep), ys = MathF.Abs(_pattern.YStep), bx = _pattern.Bounds.X, by = _pattern.Bounds.Y;
        for (int y = y0; y < y1; y++)
        {
            int row = (y - work.Top) * w;
            for (int x = x0; x < x1; x++)
            {
                int k = 4 * (row + x);
                if (tile is not { } t || work.Coverage[row + x] <= 0f)
                {
                    colour[k + 3] = 0f;
                    continue;
                }

                int tx, ty;
                if (_upright)
                {
                    tx = ((x - _startX) % t.Width + t.Width) % t.Width;
                    ty = ((y - _startY) % t.Height + t.Height) % t.Height;
                }
                else
                {
                    var (u, v) = _toPattern.Apply(x + 0.5, y + 0.5);
                    if (!double.IsFinite(u) || !double.IsFinite(v))
                    {
                        colour[k + 3] = 0f;
                        continue;
                    }

                    double a = (u - bx) % xs, b = (v - by) % ys;
                    a = a < 0 ? a + xs : a;
                    b = b < 0 ? b + ys : b;
                    tx = Math.Clamp((int)(a * t.Scale), 0, t.Width - 1);
                    ty = Math.Clamp((int)((ys - b) * t.Scale), 0, t.Height - 1);
                }

                int p = 4 * (ty * t.Width + tx);
                float alpha = t.Pixels[p + 3];
                if (alpha <= 0f)
                {
                    colour[k + 3] = 0f;
                    continue;
                }

                (colour[k], colour[k + 1], colour[k + 2], colour[k + 3]) = (t.Pixels[p] / alpha, t.Pixels[p + 1] / alpha, t.Pixels[p + 2] / alpha, alpha * _alpha);
            }
        }
    }

    // An upright cell drawn at its place's scale into a tile of `width` x `height` pixels.
    private static (float[] Pixels, int Width, int Height, float Scale)? DrawUpright(DocumentPattern pattern, Affine cellMap, int width, int height, DrawContext context)
    {
        var surface = new float[4 * width * height];
        var scene = DocumentScene.Build(pattern.Cell!, cellMap, width, height, context, DocumentParts.All, hidden: false);
        scene.Render(surface, 0, height);
        return (surface, width, height, 1f);
    }

    // The cell drawn into a tile of one step by the other at the surface's resolution (smaller for a huge tile).
    private static (float[] Pixels, int Width, int Height, float Scale)? Draw(DocumentPattern pattern, double scale, DrawContext context)
    {
        float xs = MathF.Abs(pattern.XStep), ys = MathF.Abs(pattern.YStep);
        double s = scale;
        if (!(s > 0) || !double.IsFinite(s))
        {
            return null;
        }

        double pixels = xs * s * ys * s;
        if (pixels > MaxTilePixels)
        {
            s *= Math.Sqrt(MaxTilePixels / pixels);
        }

        int width = Math.Max(1, (int)Math.Ceiling(xs * s)), height = Math.Max(1, (int)Math.Ceiling(ys * s));
        var surface = new float[4 * width * height];
        float bx = pattern.Bounds.X, by = pattern.Bounds.Y;
        bool spills = pattern.Bounds.Width > xs + 1e-3f || pattern.Bounds.Height > ys + 1e-3f || pattern.Bounds.Width < 0 || pattern.Bounds.Height < 0;
        int reach = spills ? 1 : 0;
        for (int j = -reach; j <= reach; j++)
        {
            for (int i = -reach; i <= reach; i++)
            {
                // Pattern space to the tile: x from the cell's left, y down from its top, a copy shifted by whole steps.
                var map = new Affine(s, 0, 0, -s, -(bx - i * xs) * s, (by - j * ys + ys) * s);
                var scene = DocumentScene.Build(pattern.Cell!, map, width, height, context, DocumentParts.All, hidden: false);
                scene.Render(surface, 0, height);
            }
        }

        return (surface, width, height, (float)s);
    }
}
