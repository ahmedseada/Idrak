// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// A page's parts made ready to draw once (outlines in the surface's pixels, strokes outlined, glyph runs as outlines,
/// images placed, shadings and tilings prepared, clips, groups and soft masks), then drawn into any band of rows:
/// <see cref="Render"/> is safe from several threads at once, each band with buffers of its own. A band works a clip's
/// coverage once, over the clip's own box (<see cref="ClipCover"/>), and gives it back after the last part that needs it,
/// so the memory a band holds follows the clips in use, not the clips on the page.
/// </summary>
internal sealed class DocumentScene
{
    private readonly List<SceneItem> _items;
    private readonly SceneClip?[] _clips;
    private readonly int[] _releaseStart;   // item k's clips no later item needs: _releases[_releaseStart[k] .. _releaseStart[k + 1])
    private readonly int[] _releases;
    private readonly bool _bounded;         // the library's rasterizer: a clip worked over its own box (ClipCover.Bounded)
    private readonly IReadOnlyList<DocumentGroup> _groups;
    private readonly SceneMask[] _masks;
    private readonly int _width;
    private readonly DrawContext _context;
    private readonly float _flatness;   // pixels a fill's flattened curve may stray (DocumentDrawOptions.FillFlatness)

    private DocumentScene(List<SceneItem> items, SceneClip?[] clips, IReadOnlyList<DocumentGroup> groups, SceneMask[] masks, int width, DrawContext context)
    {
        (_items, _clips, _groups, _masks, _width, _context, _flatness) = (items, clips, groups, masks, width, context, context.Options.FillFlatness);
        (_releaseStart, _releases) = Lifetimes(items, clips);
        // The library's rasterizer works each row and column in the outline's own frame, so a clip filled over its own box
        // is that box of the band-wide fill, value for value; any other rasterizer fills the band, as the drawer always did.
        _bounded = context.Rasterizer is CoverageRasterizer;
    }

    /// <summary>
    /// <paramref name="page"/>'s parts (those of <paramref name="parts"/>; hidden text only when <paramref name="hidden"/>)
    /// on a surface <paramref name="width"/> x <paramref name="height"/> pixels, placed by <paramref name="map"/> (the
    /// parts' points to the surface's pixels).
    /// </summary>
    public static DocumentScene Build(DocumentPage page, Affine map, int width, int height, DrawContext context, DocumentParts parts, bool hidden)
    {
        var builder = new Builder(page, map, width, height, context);
        var items = new List<SceneItem>();
        if ((parts & DocumentParts.Paths) != 0)
        {
            foreach (var path in page.Paths)
            {
                builder.Path(path, items);
            }

            foreach (var shading in page.Shadings)
            {
                builder.Shading(shading, items);
            }
        }

        if ((parts & DocumentParts.Images) != 0)
        {
            foreach (var image in page.Images)
            {
                builder.Image(image, items);
            }
        }

        if ((parts & DocumentParts.Text) != 0)
        {
            foreach (var run in page.Runs)
            {
                if (!run.Hidden || hidden)
                {
                    builder.Run(run, items);
                }
            }
        }

        for (int i = 0; i < items.Count; i++)
        {
            items[i].Sequence = i;
        }

        items.Sort((a, b) => a.Order != b.Order ? a.Order.CompareTo(b.Order) : a.Sequence.CompareTo(b.Sequence));
        var clips = new SceneClip?[page.Clips.Count];
        for (int c = 0; c < clips.Length; c++)
        {
            clips[c] = builder.Clip(c);
        }

        Unread(items, clips);
        builder.ReportMissing();

        var masks = new SceneMask[page.SoftMasks.Count];
        for (int m = 0; m < masks.Length; m++)
        {
            var mask = page.SoftMasks[m];
            masks[m] = new SceneMask(mask, new Lazy<DocumentScene>(() => Build(mask.Content, map, width, height, context, DocumentParts.All, hidden: false)));
        }

        return new DocumentScene(items, clips, page.Groups, masks, width, context);
    }

    // A clip of text whose runs were not read (their font not there, or the runs left out) clips nothing of its own: every
    // part and every clip made under it is put under the clip it was made under (the nearest one that clips, or none), once
    // a scene, and the clip itself dropped. So no band asks for it, and every part under it is cut by the same clips in
    // every band, whatever the band asked before. A page with no such clip is left as it is.
    private static void Unread(List<SceneItem> items, SceneClip?[] clips)
    {
        int first = 0;
        while (first < clips.Length && clips[first] is not { Ignored: true })
        {
            first++;
        }

        if (first == clips.Length)
        {
            return;
        }

        // Each clip's stand-in: itself, or for one dropped the stand-in of the clip it was made under (a parent comes first).
        var under = new int[clips.Length];
        for (int c = 0; c < clips.Length; c++)
        {
            if (clips[c] is not { } clip)
            {
                under[c] = -1;
                continue;
            }

            int parent = clip.Parent >= 0 && clip.Parent < c ? under[clip.Parent] : -1;
            if (clip.Ignored)
            {
                under[c] = parent;
                clips[c] = null;
            }
            else
            {
                under[c] = c;
                if (parent != clip.Parent)
                {
                    clips[c] = clip with { Parent = parent };
                }
            }
        }

        foreach (var item in items)
        {
            if (item.Clip >= 0)
            {
                item.Clip = under[item.Clip];
            }
        }
    }

    // For each item (in drawing order), the clips no item after it needs, directly or as the clip a later item's clip was
    // made under: a band gives their coverage back there, so it holds the clips in use, never every clip it met.
    private static (int[] Start, int[] Clips) Lifetimes(List<SceneItem> items, SceneClip?[] clips)
    {
        var last = new int[clips.Length];
        Array.Fill(last, -1);
        for (int k = 0; k < items.Count; k++)
        {
            if (items[k].Clip >= 0)
            {
                last[items[k].Clip] = k;
            }
        }

        // A clip's parent comes before it, so walking back carries each clip's last use to its parents.
        for (int c = clips.Length - 1; c > 0; c--)
        {
            if (clips[c] is { Parent: >= 0 } clip && last[c] > last[clip.Parent])
            {
                last[clip.Parent] = last[c];
            }
        }

        var start = new int[items.Count + 1];
        foreach (int k in last)
        {
            if (k >= 0)
            {
                start[k + 1]++;
            }
        }

        for (int k = 0; k < items.Count; k++)
        {
            start[k + 1] += start[k];
        }

        var released = new int[start[items.Count]];
        var next = start[..^1];
        for (int c = 0; c < last.Length; c++)
        {
            if (last[c] >= 0)
            {
                released[next[last[c]]++] = c;
            }
        }

        return (start, released);
    }

    /// <summary>Draws every part over rows [<paramref name="top"/>, top + <paramref name="rows"/>) of the surface into <paramref name="target"/> (premultiplied red, green, blue and alpha, a band of the surface's width).</summary>
    public void Render(Span<float> target, int top, int rows)
    {
        int w = _width, size = w * rows;
        float[] coverage = ArrayPool<float>.Shared.Rent(size);
        float[] colour = ArrayPool<float>.Shared.Rent(4 * size);
        var work = new SceneWork(w, top, rows, coverage, colour, _context.Rasterizer, _flatness);
        var covers = new Dictionary<int, ClipCover>();
        var softMasks = new Dictionary<int, float[]>();
        var open = new List<Layer>();
        try
        {
            coverage.AsSpan(0, size).Clear();
            Span<int> chain = stackalloc int[64];
            for (int index = 0; index < _items.Count; index++)
            {
                var item = _items[index];
                int y0 = Math.Max(item.Top, top), y1 = Math.Min(item.Bottom, top + rows);
                int x0 = Math.Max(item.Left, 0), x1 = Math.Min(item.Right, w);
                if (y0 < y1 && x0 < x1)
                {
                    // The groups the item sits in, outermost first: the layers not in its chain are closed, the missing opened.
                    int depth = Chain(item.Group, chain);
                    while (open.Count > depth || (open.Count > 0 && open[^1].Group != chain[open.Count - 1]))
                    {
                        Close(open, target, softMasks, work);
                    }

                    for (int k = open.Count; k < depth; k++)
                    {
                        open.Add(Open(chain[k], size));
                    }

                    var clip = item.Clip >= 0 ? Cover(item.Clip, covers, work) : default;
                    float[]? soft = item.SoftMask >= 0 && item.SoftMask < _masks.Length ? SoftMask(item.SoftMask, softMasks, work) : null;
                    // Nothing of the item shows outside its clip's box: its colours are made and composed within it alone. An
                    // item wholly clipped away still renders, on an empty box, so what it reads and reports (an image
                    // decoded, a tiling's cell drawn) is as it was.
                    var (cx0, cx1, cy0, cy1) = clip.Clips ? clip.Within(x0, x1, y0, y1) : (x0, x1, y0, y1);
                    bool solid = item.Render(work, cx0, cx1, cy0, cy1, out var paint);
                    var layer = TargetOf(open, target, out bool knockout);
                    if (cx0 < cx1 && cy0 < cy1)
                    {
                        Composite(work, layer, cx0, cx1, cy0, cy1, solid, paint, clip, soft, item.Blend, knockout);
                    }

                    // The item's whole box: its fill wrote its coverage there, whatever its clip.
                    for (int y = y0; y < y1; y++)
                    {
                        coverage.AsSpan((y - top) * w + x0, x1 - x0).Clear();
                    }
                }

                for (int r = _releaseStart[index]; r < _releaseStart[index + 1]; r++)
                {
                    if (covers.Remove(_releases[r], out var done))
                    {
                        done.Return();
                    }
                }
            }

            while (open.Count > 0)
            {
                Close(open, target, softMasks, work);
            }
        }
        finally
        {
            foreach (var layer in open)
            {
                if (layer.Pixels is { } pixels)
                {
                    ArrayPool<float>.Shared.Return(pixels);
                }
            }

            foreach (var cover in covers.Values)
            {
                cover.Return();
            }

            foreach (var mask in softMasks.Values)
            {
                ArrayPool<float>.Shared.Return(mask);
            }

            ArrayPool<float>.Shared.Return(coverage);
            ArrayPool<float>.Shared.Return(colour);
        }
    }

    // ---- groups ------------------------------------------------------------------------------------------------------

    // A group's chain of groups from the outermost; 0 for none (a chain that loops or runs too deep stops there).
    private int Chain(int group, Span<int> chain)
    {
        int depth = 0;
        for (int g = group; g >= 0 && g < _groups.Count && depth < chain.Length; g = _groups[g].Parent < g ? _groups[g].Parent : -1)
        {
            chain[depth++] = g;
        }

        chain[..depth].Reverse();
        return depth;
    }

    // A group drawn straight onto what is under it (not isolated, not knockout, opaque, normal, no mask: the same
    // pixels), or into a transparent layer of its own (an isolated group, a knockout group, or one composed with an
    // opacity, a blend mode or a soft mask; a group that is not isolated is drawn as one that is, here).
    private Layer Open(int group, int size)
    {
        var g = _groups[group];
        if (!g.Isolated && !g.Knockout && g.Alpha >= 1f && g.Blend == DocumentBlend.Normal && g.SoftMask < 0)
        {
            return new Layer(group, null, false);
        }

        var pixels = ArrayPool<float>.Shared.Rent(4 * size);
        pixels.AsSpan(0, 4 * size).Clear();
        return new Layer(group, pixels, g.Knockout);
    }

    private static Span<float> TargetOf(List<Layer> open, Span<float> target, out bool knockout)
    {
        for (int k = open.Count - 1; k >= 0; k--)
        {
            if (open[k].Pixels is { } pixels)
            {
                knockout = open[k].Knockout;
                return pixels.AsSpan(0, target.Length);
            }
        }

        knockout = false;
        return target;
    }

    // The innermost open layer composed onto the one under it with its group's opacity, blend mode and soft mask.
    private void Close(List<Layer> open, Span<float> target, Dictionary<int, float[]> softMasks, SceneWork work)
    {
        var layer = open[^1];
        open.RemoveAt(open.Count - 1);
        if (layer.Pixels is not { } pixels)
        {
            return;
        }

        try
        {
            var g = _groups[layer.Group];
            float[]? soft = g.SoftMask >= 0 && g.SoftMask < _masks.Length ? SoftMask(g.SoftMask, softMasks, work) : null;
            var under = TargetOf(open, target, out bool knockout);
            int n = target.Length / 4;
            for (int i = 0; i < n; i++)
            {
                float a = pixels[4 * i + 3];
                if (a <= 0f)
                {
                    continue;
                }

                float f = g.Alpha * (soft is null ? 1f : soft[i]);
                if (f <= 0f)
                {
                    continue;
                }

                float r = pixels[4 * i] / a, gr = pixels[4 * i + 1] / a, b = pixels[4 * i + 2] / a;
                Blend(under, 4 * i, r, gr, b, a * f, 1f, g.Blend, knockout);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(pixels);
        }
    }

    // ---- clips and soft masks ----------------------------------------------------------------------------------------

    // A clip's coverage over the band, intersected with the clips it was made under; none for one that clips nothing. A
    // clip of text whose runs were not read is never asked: the scene put what was made under it under its parent (Unread).
    private ClipCover Cover(int index, Dictionary<int, ClipCover> cache, SceneWork work)
    {
        if (cache.TryGetValue(index, out var known))
        {
            return known;
        }

        ClipCover cover = default;
        if (index < _clips.Length && _clips[index] is { } clip)
        {
            var parent = clip.Parent >= 0 && clip.Parent < index ? Cover(clip.Parent, cache, work) : default;
            cover = _bounded ? ClipCover.Bounded(clip, parent, work) : ClipCover.Banded(clip, parent, work);
        }

        cache[index] = cover;
        return cover;
    }

    // A soft mask's values over the band: its content drawn on its backdrop (luminosity) or on nothing (opacity).
    private float[] SoftMask(int index, Dictionary<int, float[]> cache, SceneWork work)
    {
        if (cache.TryGetValue(index, out var known))
        {
            return known;
        }

        var mask = _masks[index];
        int size = work.Width * work.Rows;
        float[] surface = ArrayPool<float>.Shared.Rent(4 * size);
        float[] values = ArrayPool<float>.Shared.Rent(size);
        try
        {
            var target = surface.AsSpan(0, 4 * size);
            if (mask.Mask.Luminosity)
            {
                var b = mask.Mask.Backdrop;
                CoverageDocumentDrawer.Fill(target, b.Red, b.Green, b.Blue, 1f);
            }
            else
            {
                target.Clear();
            }

            mask.Scene.Value.Render(target, work.Top, work.Rows);
            for (int i = 0; i < size; i++)
            {
                values[i] = mask.Mask.Luminosity
                    ? Math.Clamp(0.3f * target[4 * i] + 0.59f * target[4 * i + 1] + 0.11f * target[4 * i + 2], 0f, 1f)
                    : Math.Clamp(target[4 * i + 3], 0f, 1f);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(surface);
        }

        cache[index] = values;
        return values;
    }

    // ---- composing ---------------------------------------------------------------------------------------------------

    // The item's coverage (times the clip and the soft mask) and colour composed onto the layer, row by row in order; a
    // clip's box holds [x0, x1) x [y0, y1).
    private static void Composite(SceneWork work, Span<float> layer, int x0, int x1, int y0, int y1, bool solid, (float R, float G, float B, float A) paint,
        in ClipCover clip, float[]? soft, DocumentBlend blend, bool knockout)
    {
        int w = work.Width;
        var coverage = work.Coverage;
        var colour = work.Colour;
        bool clipped = clip.Clips;
        int origin = clip.Origin;
        for (int y = y0; y < y1; y++)
        {
            int row = (y - work.Top) * w;
            ReadOnlySpan<float> cut = clipped ? clip.Row(y) : default;
            for (int x = x0; x < x1; x++)
            {
                int i = row + x;
                float shape = coverage[i];
                if (shape <= 0f)
                {
                    continue;
                }

                if (clipped)
                {
                    shape *= cut[x - origin];
                    if (shape <= 0f)
                    {
                        continue;
                    }
                }

                float r, g, b, a;
                if (solid)
                {
                    (r, g, b, a) = paint;
                }
                else
                {
                    (r, g, b, a) = (colour[4 * i], colour[4 * i + 1], colour[4 * i + 2], colour[4 * i + 3]);
                }

                if (soft is not null)
                {
                    a *= soft[i];
                }

                if (a <= 0f)
                {
                    continue;
                }

                Blend(layer, 4 * i, r, g, b, knockout ? a : a * MathF.Min(1f, shape), MathF.Min(1f, shape), blend, knockout);
            }
        }
    }

    // One pixel: the colour (r, g, b; straight) at opacity `a` over the premultiplied pixel at `at`, by the blend mode.
    // In a knockout group the pixel is the colour over the group's empty backdrop, mixed with what was there by `shape`.
    internal static void Blend(Span<float> pixels, int at, float r, float g, float b, float a, float shape, DocumentBlend blend, bool knockout)
    {
        if (knockout)
        {
            float s = shape;
            pixels[at] = pixels[at] * (1 - s) + r * a * s;
            pixels[at + 1] = pixels[at + 1] * (1 - s) + g * a * s;
            pixels[at + 2] = pixels[at + 2] * (1 - s) + b * a * s;
            pixels[at + 3] = pixels[at + 3] * (1 - s) + a * s;
            return;
        }

        float pr = pixels[at], pg = pixels[at + 1], pb = pixels[at + 2], ab = pixels[at + 3];
        if (blend == DocumentBlend.Normal || ab <= 0f)
        {
            pixels[at] = pr * (1 - a) + r * a;
            pixels[at + 1] = pg * (1 - a) + g * a;
            pixels[at + 2] = pb * (1 - a) + b * a;
            pixels[at + 3] = ab + a - ab * a;
            return;
        }

        float cr = pr / ab, cg = pg / ab, cb = pb / ab;
        var (br, bg, bb) = BlendModes.Mix(blend, cr, cg, cb, r, g, b);
        pixels[at] = (1 - a) * pr + a * ((1 - ab) * r + ab * br);
        pixels[at + 1] = (1 - a) * pg + a * ((1 - ab) * g + ab * bg);
        pixels[at + 2] = (1 - a) * pb + a * ((1 - ab) * b + ab * bb);
        pixels[at + 3] = ab + a - ab * a;
    }

    private readonly record struct Layer(int Group, float[]? Pixels, bool Knockout);

    // ---- building ----------------------------------------------------------------------------------------------------

    private sealed class Builder(DocumentPage page, Affine map, int width, int height, DrawContext context)
    {
        private readonly Dictionary<string, int> _missing = new(StringComparer.Ordinal);
        private readonly float _scale = (float)map.Scale;

        // The options' numbers, read once a scene: a stroke's flatness (pixels), a shading's steps, the largest tile, and a
        // face found for a font that is not embedded: its stretch bounds, slant and stroke (DocumentDrawOptions).
        private readonly float _strokeFlatness = context.Options.StrokeFlatness;
        private readonly int _shadingSteps = context.Options.ShadingSteps;
        private readonly int _maxTilePixels = context.Options.MaxTilePixels;
        private readonly float _minStretch = context.Options.MinSubstituteStretch;
        private readonly float _maxStretch = context.Options.MaxSubstituteStretch;
        private readonly float _slant = context.Options.SubstituteSlant;
        private readonly float _embolden = context.Options.SubstituteStroke;

        public void ReportMissing()
        {
            foreach (var (font, count) in _missing)
            {
                context.Issue(DocumentIssue.MissingFont,
                    $"page {page.Index + 1}: the font {font} is not embedded and no font in {(context.Options.Fonts is { } f ? f : "DocumentDrawOptions.Fonts (none given)")} has its characters: its runs are left out", count);
            }
        }

        public void Path(DocumentPath path, List<SceneItem> items)
        {
            if (path.IsEmpty)
            {
                return;
            }

            if (path.Fill != DocumentFill.None)
            {
                var outline = Outline(path.Verbs.Span, path.Points.Span);
                if (!outline.IsEmpty)
                {
                    var paint = Paint(path.FillPattern, path.FillColour);
                    items.Add(Place(new SceneItem(new OutlineItem(outline.Build(), path.Fill == DocumentFill.EvenOdd ? OutlineFill.EvenOdd : OutlineFill.NonZero, paint).Render), outline, path.Order,
                        path.Clip, path.Group, path.SoftMask, path.Blend));
                }
            }

            if (path.Stroke is { } stroke)
            {
                var outline = new PixelOutline();
                float width = stroke > 0 ? stroke * _scale : 1f;   // 0: the thinnest line, a pixel
                float[] dash = path.Dash.Length == 0 ? [] : [.. path.Dash.ToArray().Select(d => d * _scale)];
                Stroke(outline, path.Verbs.Span, path.Points.Span, width, path.Cap, path.Join, path.MiterLimit, dash, path.DashPhase * _scale, path.Order);
                if (!outline.IsEmpty)
                {
                    var paint = Paint(path.StrokePattern, path.StrokeColour);
                    items.Add(Place(new SceneItem(new OutlineItem(outline.Build(), OutlineFill.NonZero, paint).Render), outline, path.Order, path.Clip, path.Group, path.SoftMask, path.Blend));
                }
            }
        }

        public void Shading(DocumentShading shading, List<SceneItem> items)
        {
            var paint = ShadingPaint.Of(shading, map, shading.Alpha, background: false, _shadingSteps);
            var item = new SceneItem(new ShadingItem(paint).Render);
            var (left, top, right, bottom) = paint.Bounds(width, height);
            items.Add(Set(item, left, top, right, bottom, shading.Order, shading.Clip, shading.Group, shading.SoftMask, shading.Blend));
        }

        public void Image(DocumentImage image, List<SceneItem> items)
        {
            var placed = Affine.Of(image.Placement).Then(map);
            if (!placed.Invertible)
            {
                return;
            }

            var outline = new PixelOutline();
            Span<double> corners = stackalloc double[8];
            ReadOnlySpan<(double U, double V)> unit = [(0, 0), (1, 0), (1, 1), (0, 1)];
            for (int k = 0; k < 4; k++)
            {
                (corners[2 * k], corners[2 * k + 1]) = placed.Apply(unit[k].U, unit[k].V);
            }

            outline.Polygon([(float)corners[0], (float)corners[1], (float)corners[2], (float)corners[3], (float)corners[4], (float)corners[5], (float)corners[6], (float)corners[7]]);
            var item = new SceneItem(new ImageItem(outline.Build(), image, placed.Inverse(), context, page.Index).Render);
            items.Add(Place(item, outline, image.Order, image.Clip, image.Group, image.SoftMask, image.Blend));
        }

        public void Run(DocumentTextRun run, List<SceneItem> items)
        {
            var outline = RunOutline(run, null, out bool embolden);
            if (outline is null || outline.IsEmpty)
            {
                return;
            }

            bool fill = run.Mode is DocumentTextMode.Fill or DocumentTextMode.FillStroke or DocumentTextMode.Hidden;
            bool stroke = run.Mode is DocumentTextMode.Stroke or DocumentTextMode.FillStroke;
            if (fill)
            {
                items.Add(Place(new SceneItem(new OutlineItem(outline.Build(), OutlineFill.NonZero, new ScenePaint(run.Colour)).Render), outline, run.Order, run.Clip, run.Group, run.SoftMask, run.Blend));
            }

            // A face found for a bold font that is not bold itself: its glyphs stroked a little in their own colour.
            if (stroke || (fill && embolden))
            {
                var glyphs = outline.Build();
                var stroked = new PixelOutline();
                float w = stroke ? (run.StrokeWidth > 0 ? run.StrokeWidth * _scale : 1f) : _embolden * run.FontSize * _scale;
                // The glyphs' outline is held with y turned up: turned back before it is stroked.
                var points = glyphs.Points.ToArray();
                for (int i = 1; i < points.Length; i += 2)
                {
                    points[i] = -points[i];
                }

                Stroke(stroked, glyphs.Verbs.Span, points, w, stroke ? DocumentLineCap.Butt : DocumentLineCap.Round, stroke ? DocumentLineJoin.Miter : DocumentLineJoin.Round, 10f, [], 0f, run.Order, pixels: true);
                if (!stroked.IsEmpty)
                {
                    items.Add(Place(new SceneItem(new OutlineItem(stroked.Build(), OutlineFill.NonZero, new ScenePaint(stroke ? run.StrokeColour : run.Colour)).Render), stroked, run.Order, run.Clip, run.Group, run.SoftMask, run.Blend));
                }
            }
        }

        /// <summary>Clip <paramref name="index"/> of the page made ready: its outline and rule (a clip of text: its runs' glyphs).</summary>
        public SceneClip? Clip(int index)
        {
            var clip = page.Clips[index];
            int parent = clip.Clip >= 0 && clip.Clip < index ? clip.Clip : -1;
            if (clip.ClipRun >= 0)
            {
                var outline = new PixelOutline();
                int found = 0;
                foreach (var run in page.Runs)
                {
                    if (run.Order >= clip.ClipRun && run.Order < clip.ClipRun + Math.Max(1, clip.ClipRunCount) && RunOutline(run, outline, out _) is not null)
                    {
                        found++;
                    }
                }

                return found == 0 ? new SceneClip(Abstraction.Data.GlyphOutline.Empty, OutlineFill.NonZero, parent, Ignored: true, default) : Made(outline.Build(), OutlineFill.NonZero, parent);
            }

            var path = Outline(clip.Verbs.Span, clip.Points.Span);
            return Made(path.Build(), clip.Fill == DocumentFill.EvenOdd ? OutlineFill.EvenOdd : OutlineFill.NonZero, parent);
        }

        private SceneClip Made(GlyphOutline outline, OutlineFill rule, int parent) => new(outline, rule, parent, false, ClipBox.Of(outline, width, height));

        // The paint of a fill or a stroke: its pattern (a shading, a tiling), else its colour.
        private ScenePaint Paint(DocumentPattern? pattern, DocumentColour colour)
        {
            if (pattern?.Shading is { } shading)
            {
                return new ScenePaint(ShadingPaint.Of(shading, map, colour.Alpha, background: true, _shadingSteps).Colours);
            }

            if (pattern?.Cell is not null)
            {
                return new ScenePaint(new TilePaint(pattern, map, colour.Alpha, context, _maxTilePixels).Colours);
            }

            return new ScenePaint(colour);
        }

        // A path's points through the map, into surface pixels.
        private PixelOutline Outline(ReadOnlySpan<OutlineVerb> verbs, ReadOnlySpan<float> points)
        {
            var outline = new PixelOutline();
            int at = 0;
            foreach (var verb in verbs)
            {
                switch (verb)
                {
                    case OutlineVerb.Move:
                    {
                        var (x, y) = map.Apply(points[at], points[at + 1]);
                        outline.Move((float)x, (float)y);
                        at += 2;
                        break;
                    }

                    case OutlineVerb.Line:
                    {
                        var (x, y) = map.Apply(points[at], points[at + 1]);
                        outline.Line((float)x, (float)y);
                        at += 2;
                        break;
                    }

                    case OutlineVerb.Quadratic:
                    {
                        var (x1, y1) = map.Apply(points[at], points[at + 1]);
                        var (x, y) = map.Apply(points[at + 2], points[at + 3]);
                        outline.Quadratic((float)x1, (float)y1, (float)x, (float)y);
                        at += 4;
                        break;
                    }

                    case OutlineVerb.Cubic:
                    {
                        var (x1, y1) = map.Apply(points[at], points[at + 1]);
                        var (x2, y2) = map.Apply(points[at + 2], points[at + 3]);
                        var (x, y) = map.Apply(points[at + 4], points[at + 5]);
                        outline.Cubic((float)x1, (float)y1, (float)x2, (float)y2, (float)x, (float)y);
                        at += 6;
                        break;
                    }

                    case OutlineVerb.Close:
                        outline.Close();
                        break;
                }
            }

            return outline;
        }

        // A path stroked: flattened into polylines in pixels (each subpath closed by its close), cut by the dash, through
        // the options' pen (its points) and outlined by the stroker with the path's caps, joins and miter limit.
        private void Stroke(PixelOutline outline, ReadOnlySpan<OutlineVerb> verbs, ReadOnlySpan<float> points, float width, DocumentLineCap cap, DocumentLineJoin join,
            float miterLimit, float[] dash, float phase, int seed, bool pixels = false)
        {
            var polyline = new List<float>();
            var pen = context.Pen;
            var settings = context.PenSettings;
            using var stroke = new PenStroke();
            var path = new List<PenPoint>();
            void Emit(bool closed)
            {
                if (polyline.Count < 2)
                {
                    polyline.Clear();
                    return;
                }

                var xy = polyline.ToArray();
                polyline.Clear();
                DocumentStroker.Dash(xy, closed, dash, phase, (piece, pieceClosed, dx, dy) =>
                {
                    path.Clear();
                    for (int i = 0; i + 1 < piece.Length; i += 2)
                    {
                        path.Add(new PenPoint(piece[i], piece[i + 1], width));
                    }

                    pen.Stroke(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(path), pieceClosed, settings, seed, stroke);
                    DocumentStroker.Stroke(outline, stroke.Points, stroke.Closed, cap, join, miterLimit, _strokeFlatness, dx, dy);
                });
            }

            int k = 0;
            double cx = 0, cy = 0, sx = 0, sy = 0;
            foreach (var verb in verbs)
            {
                switch (verb)
                {
                    case OutlineVerb.Move:
                        Emit(false);
                        (cx, cy) = At(points, k, pixels);
                        (sx, sy) = (cx, cy);
                        polyline.Add((float)cx);
                        polyline.Add((float)cy);
                        k += 2;
                        break;
                    case OutlineVerb.Line:
                        (cx, cy) = At(points, k, pixels);
                        polyline.Add((float)cx);
                        polyline.Add((float)cy);
                        k += 2;
                        break;
                    case OutlineVerb.Quadratic:
                    {
                        var (qx, qy) = At(points, k, pixels);
                        var (ex, ey) = At(points, k + 2, pixels);
                        double dd = Math.Sqrt((cx - 2 * qx + ex) * (cx - 2 * qx + ex) + (cy - 2 * qy + ey) * (cy - 2 * qy + ey));
                        int n = Math.Clamp((int)Math.Ceiling(Math.Sqrt(dd / (8 * _strokeFlatness))), 1, 4096);
                        for (int i = 1; i <= n; i++)
                        {
                            double t = (double)i / n, u = 1 - t;
                            polyline.Add((float)(u * u * cx + 2 * u * t * qx + t * t * ex));
                            polyline.Add((float)(u * u * cy + 2 * u * t * qy + t * t * ey));
                        }

                        (cx, cy) = (ex, ey);
                        k += 4;
                        break;
                    }

                    case OutlineVerb.Cubic:
                    {
                        var (ax, ay) = At(points, k, pixels);
                        var (bx, by) = At(points, k + 2, pixels);
                        var (ex, ey) = At(points, k + 4, pixels);
                        double dd = Math.Max(Math.Sqrt((cx - 2 * ax + bx) * (cx - 2 * ax + bx) + (cy - 2 * ay + by) * (cy - 2 * ay + by)),
                            Math.Sqrt((ax - 2 * bx + ex) * (ax - 2 * bx + ex) + (ay - 2 * by + ey) * (ay - 2 * by + ey)));
                        int n = Math.Clamp((int)Math.Ceiling(Math.Sqrt(3 * dd / (4 * _strokeFlatness))), 1, 4096);
                        for (int i = 1; i <= n; i++)
                        {
                            double t = (double)i / n, u = 1 - t;
                            double b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
                            polyline.Add((float)(b0 * cx + b1 * ax + b2 * bx + b3 * ex));
                            polyline.Add((float)(b0 * cy + b1 * ay + b2 * by + b3 * ey));
                        }

                        (cx, cy) = (ex, ey);
                        k += 6;
                        break;
                    }

                    case OutlineVerb.Close:
                        Emit(true);
                        (cx, cy) = (sx, sy);
                        polyline.Add((float)cx);
                        polyline.Add((float)cy);
                        break;
                }
            }

            Emit(false);
        }

        // A point of a path: through the map, or already in pixels.
        private (double X, double Y) At(ReadOnlySpan<float> points, int at, bool pixels) => pixels ? (points[at], points[at + 1]) : map.Apply(points[at], points[at + 1]);

        // A run's glyphs as one outline in surface pixels (a new one, or added to `into`), or null when its font is not there
        // (counted: the run is left out). A face found for a font the document does not embed draws each glyph stretched to
        // the advance the document gives it (within the options' stretch bounds), slanted by the options' slant when the
        // font's name says it is italic or oblique and the face is not, and says when the name says bold and the face is not
        // and the options stroke such faces (`embolden`).
        private PixelOutline? RunOutline(DocumentTextRun run, PixelOutline? into, out bool embolden)
        {
            var face = run.Face;
            bool substitute = false;
            embolden = false;
            if (face is null)
            {
                face = context.Substitute(run);
                substitute = true;
                if (face is null)
                {
                    string name = run.Font ?? "(no name)";
                    _missing[name] = _missing.TryGetValue(name, out int seen) ? seen + 1 : 1;
                    return null;
                }
            }

            int upem = face.Metrics.UnitsPerEm > 0 ? face.Metrics.UnitsPerEm : 1000;
            var m = run.GlyphMatrix ?? new DocumentMatrix(run.FontSize, 0f, 0f, -run.FontSize, 0f, 0f);
            if (substitute)
            {
                string font = run.Font ?? "";
                bool slanted = face.Style.Contains("Italic", StringComparison.OrdinalIgnoreCase) || face.Style.Contains("Oblique", StringComparison.OrdinalIgnoreCase);
                if (_slant > 0f && !slanted && (font.Contains("Italic", StringComparison.OrdinalIgnoreCase) || font.Contains("Oblique", StringComparison.OrdinalIgnoreCase)))
                {
                    m = m with { C = m.C + _slant * m.A, D = m.D + _slant * m.B };
                }

                embolden = _embolden > 0f && font.Contains("Bold", StringComparison.OrdinalIgnoreCase) && !face.Style.Contains("Bold", StringComparison.OrdinalIgnoreCase);
            }

            var outline = into ?? new PixelOutline();
            var glyphs = run.Glyphs.Span;
            if (glyphs.IsEmpty)
            {
                // A run with no glyphs given (a format that gives text alone): its characters by the face's map, along its baseline.
                float x = run.Box.X, y = run.Baseline ?? run.Box.Bottom;
                foreach (var rune in run.Text.EnumerateRunes())
                {
                    int g = face.Glyph(rune.Value);
                    AddGlyph(outline, face, g, x, y, m, upem);
                    float advance = face.Advance(g) / (float)upem;
                    x += m.A * advance;
                    y += m.B * advance;
                }

                return outline;
            }

            // A face found for a run not embedded: each glyph by the character it stands for (one a glyph when they pair up).
            int[]? mapped = null;
            if (substitute)
            {
                var runes = run.Text.EnumerateRunes().ToArray();
                mapped = new int[glyphs.Length];
                for (int i = 0; i < glyphs.Length; i++)
                {
                    mapped[i] = i < runes.Length ? face.Glyph(runes[i].Value) : 0;
                }
            }

            float across = MathF.Sqrt(m.A * m.A + m.B * m.B);
            for (int i = 0; i < glyphs.Length; i++)
            {
                var g = glyphs[i];
                int id = mapped is null ? g.Glyph : mapped[i];
                var placed = m;
                if (mapped is not null && (uint)id < (uint)face.GlyphCount && face.Advance(id) > 0 && g.Advance > 0 && across > 0)
                {
                    float stretch = Math.Clamp(g.Advance / (face.Advance(id) / (float)upem * across), _minStretch, _maxStretch);
                    placed = m with { A = m.A * stretch, B = m.B * stretch };
                }

                AddGlyph(outline, face, id, g.X, g.Y, placed, upem);
            }

            return outline;
        }

        private void AddGlyph(PixelOutline outline, IFontFace face, int glyph, float ox, float oy, DocumentMatrix m, int upem)
        {
            if (glyph < 0 || glyph >= face.GlyphCount)
            {
                return;
            }

            GlyphOutline shape;
            try
            {
                shape = face.Outline(glyph);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or IndexOutOfRangeException or ArgumentException)
            {
                context.Issue(DocumentIssue.NotDrawn, $"page {page.Index + 1}: a glyph of {face.Family} could not be read ({e.Message})");
                return;
            }

            var verbs = shape.Verbs.Span;
            var p = shape.Points.Span;
            float k = 1f / upem;
            var glyphMap = new Affine(m.A * k, m.B * k, m.C * k, m.D * k, ox, oy).Then(map);

            int at = 0;
            foreach (var verb in verbs)
            {
                switch (verb)
                {
                    case OutlineVerb.Move:
                    {
                        var (x, y) = glyphMap.Apply(p[at], p[at + 1]);
                        outline.Move((float)x, (float)y);
                        at += 2;
                        break;
                    }

                    case OutlineVerb.Line:
                    {
                        var (x, y) = glyphMap.Apply(p[at], p[at + 1]);
                        outline.Line((float)x, (float)y);
                        at += 2;
                        break;
                    }

                    case OutlineVerb.Quadratic:
                    {
                        var (x1, y1) = glyphMap.Apply(p[at], p[at + 1]);
                        var (x, y) = glyphMap.Apply(p[at + 2], p[at + 3]);
                        outline.Quadratic((float)x1, (float)y1, (float)x, (float)y);
                        at += 4;
                        break;
                    }

                    case OutlineVerb.Cubic:
                    {
                        var (x1, y1) = glyphMap.Apply(p[at], p[at + 1]);
                        var (x2, y2) = glyphMap.Apply(p[at + 2], p[at + 3]);
                        var (x, y) = glyphMap.Apply(p[at + 4], p[at + 5]);
                        outline.Cubic((float)x1, (float)y1, (float)x2, (float)y2, (float)x, (float)y);
                        at += 6;
                        break;
                    }

                    case OutlineVerb.Close:
                        outline.Close();
                        break;
                }
            }

            outline.Close();
        }

        private SceneItem Place(SceneItem item, PixelOutline outline, int order, int clip, int group, int softMask, DocumentBlend blend) =>
            Set(item, (int)MathF.Floor(outline.MinX) - 1, (int)MathF.Floor(outline.MinY) - 1, (int)MathF.Ceiling(outline.MaxX) + 2, (int)MathF.Ceiling(outline.MaxY) + 2,
                order, clip, group, softMask, blend);

        private SceneItem Set(SceneItem item, int left, int top, int right, int bottom, int order, int clip, int group, int softMask, DocumentBlend blend)
        {
            item.Left = Math.Clamp(left, 0, width);
            item.Right = Math.Clamp(right, 0, width);
            item.Top = Math.Clamp(top, 0, height);
            item.Bottom = Math.Clamp(bottom, 0, height);
            item.Order = order;
            item.Clip = clip >= 0 && clip < page.Clips.Count ? clip : -1;
            item.Group = group >= 0 && group < page.Groups.Count ? group : -1;
            item.SoftMask = softMask >= 0 && softMask < page.SoftMasks.Count ? softMask : -1;
            item.Blend = blend;
            return item;
        }
    }
}

/// <summary>A clip made ready: its outline in the surface's pixels and its rule, the clip it was made under, whether it clips nothing of its own (text whose runs were not read: the scene puts what is under it under its parent), and where its fill reaches.</summary>
internal sealed record SceneClip(GlyphOutline Outline, OutlineFill Rule, int Parent, bool Ignored, ClipBox Box);

/// <summary>
/// Where a clip's fill reaches on the surface, as the library's rasterizer finds it (<see cref="CoverageRasterizer"/>:
/// the outline's box in whole pixels, within the surface; empty when it fills nothing), and, for an upright rectangle (one
/// contour of four lines, two exactly level and two exactly upright: a <c>re</c>, a form's box at a right angle), the
/// first and last rows its sides reach: every row strictly between them gets the same values from the rasterizer (each
/// upright side lays one whole row's winding in the same cells, the level ones lay none).
/// </summary>
internal readonly record struct ClipBox(int Left, int Top, int Right, int Bottom, bool Rectangle, int FirstRow, int LastRow)
{
    public bool IsEmpty => Left >= Right || Top >= Bottom;

    /// <summary>A rectangle's kind of row: 0 its first, 2 its last, 1 any row between (the same values).</summary>
    public int Kind(int y) => y == FirstRow ? 0 : y == LastRow ? 2 : 1;

    /// <summary><paramref name="outline"/>'s box on a surface <paramref name="width"/> x <paramref name="height"/> pixels, by the rasterizer's own arithmetic for a fill at the surface's origin.</summary>
    public static ClipBox Of(GlyphOutline outline, int width, int height)
    {
        if (outline.IsEmpty)
        {
            return default;
        }

        // The outline is held with y up: its rows run from -maxY down to -minY.
        var (minX, minY, maxX, maxY) = outline.Bounds();
        float leftF = MathF.Floor(minX), rightF = MathF.Ceiling(maxX), topF = MathF.Floor(-maxY), bottomF = MathF.Ceiling(-minY);
        if (!(rightF > 0f) || !(bottomF > 0f) || !(leftF < width) || !(topF < height))
        {
            return default;                                                     // nothing of it on the surface (or not finite)
        }

        int left = (int)MathF.Max(0f, leftF), right = (int)MathF.Min(width, rightF), top = (int)MathF.Max(0f, topF), bottom = (int)MathF.Min(height, bottomF);
        if (right <= left || bottom <= top)
        {
            return default;
        }

        bool rectangle = Upright(outline);
        return new ClipBox(left, top, right, bottom, rectangle, rectangle ? Row(topF, height) : -1, rectangle ? Row(bottomF - 1f, height) : -1);
    }

    // A row of the surface, or -1 above it and `height` below it (never a row a band holds).
    private static int Row(float y, int height) => y < 0f ? -1 : y >= height ? height : (int)y;

    // One contour of four lines (closed, or with a fifth back to its start) whose sides are exactly level and upright.
    private static bool Upright(GlyphOutline outline)
    {
        var verbs = outline.Verbs.Span;
        var p = outline.Points.Span;
        bool four = verbs is [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close] && p.Length == 8;
        bool five = verbs is [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close] && p.Length == 10
                    && p[8] == p[0] && p[9] == p[1];
        return (four || five)
               && ((p[1] == p[3] && p[2] == p[4] && p[5] == p[7] && p[6] == p[0]) || (p[0] == p[2] && p[3] == p[5] && p[4] == p[6] && p[7] == p[1]));
    }
}

/// <summary>
/// A clip's coverage over a band, worked once and shared by every part under it: nothing clipped (<see cref="Clips"/>
/// false), or values over a box of the surface, [<see cref="Left"/>, <see cref="Right"/>) x [<see cref="Top"/>,
/// <see cref="Bottom"/>), and nothing outside it. Row y's values are a row of <see cref="Values"/> (the one
/// <see cref="Rows"/> names, or row y - <see cref="Top"/> when every row is its own) from column <see cref="Origin"/>; the
/// rows an upright rectangle fills alike share one. Its arrays are rented from the shared pools and given back by
/// <see cref="Return"/>.
/// </summary>
internal readonly struct ClipCover
{
    private ClipCover(float[]? values, int[]? rows, int left, int top, int right, int bottom, int origin, int stride)
    {
        (Values, Rows, Left, Top, Right, Bottom, Origin, Stride) = (values, rows, left, top, right, bottom, origin, stride);
        Clips = true;
    }

    /// <summary>Whether it clips (false: the default, nothing clipped).</summary>
    public bool Clips { get; }

    /// <summary>The stored rows, <see cref="Stride"/> values each.</summary>
    public float[]? Values { get; }

    /// <summary>Each row's stored row from <see cref="Top"/>, or null when row y is stored row y - <see cref="Top"/>.</summary>
    public int[]? Rows { get; }

    public int Left { get; }

    public int Top { get; }

    public int Right { get; }

    public int Bottom { get; }

    /// <summary>The surface column of a stored row's first value.</summary>
    public int Origin { get; }

    public int Stride { get; }

    // A clip that lets nothing through: its fill covers none of the band, or misses the clip it was made under.
    private static ClipCover Nothing => new(null, null, 0, 0, 0, 0, 0, 0);

    /// <summary>Row <paramref name="y"/>'s values (from column <see cref="Origin"/>); <paramref name="y"/> within [<see cref="Top"/>, <see cref="Bottom"/>).</summary>
    public Span<float> Row(int y) => Values.AsSpan(StoredRow(y) * Stride, Stride);

    private int StoredRow(int y) => Rows is { } rows ? rows[y - Top] : y - Top;

    /// <summary>The box [x0, x1) x [y0, y1) within this one's, or an empty box at (x0, y0) when they do not meet.</summary>
    public (int X0, int X1, int Y0, int Y1) Within(int x0, int x1, int y0, int y1)
    {
        int a = Math.Max(x0, Left), b = Math.Min(x1, Right), c = Math.Max(y0, Top), d = Math.Min(y1, Bottom);
        return a < b && c < d ? (a, b, c, d) : (x0, x0, y0, y0);
    }

    /// <summary>Gives its arrays back to the shared pools.</summary>
    public void Return()
    {
        if (Values is { } values)
        {
            ArrayPool<float>.Shared.Return(values);
        }

        if (Rows is { } rows)
        {
            ArrayPool<int>.Shared.Return(rows);
        }
    }

    /// <summary>
    /// The clip filled over the whole band by any rasterizer, times <paramref name="parent"/> (a band-wide coverage too):
    /// the drawer's way for a rasterizer that is not the library's.
    /// </summary>
    public static ClipCover Banded(SceneClip clip, in ClipCover parent, SceneWork work)
    {
        int size = work.Width * work.Rows;
        float[] mask = ArrayPool<float>.Shared.Rent(size);
        bool kept = false;
        try
        {
            var span = mask.AsSpan(0, size);
            span.Clear();
            if (!clip.Outline.IsEmpty)
            {
                work.Rasterizer.Fill(clip.Outline, 1f, 0f, -work.Top, span, work.Width, work.Rows, clip.Rule, work.Flatness);
            }

            if (parent.Clips)
            {
                Product(span, parent.Values.AsSpan(0, size), span);
            }

            kept = true;
            return new ClipCover(mask, null, 0, work.Top, work.Width, work.Top + work.Rows, 0, work.Width);
        }
        finally
        {
            if (!kept)
            {
                ArrayPool<float>.Shared.Return(mask);
            }
        }
    }

    /// <summary>
    /// The clip by the library's rasterizer over its own box within the band and <paramref name="parent"/>'s box, times
    /// <paramref name="parent"/>: each value the band-wide fill's at that pixel (the fill's columns are the clip's own, so
    /// its arithmetic is the band-wide fill's; its rows are worked alone, as bands are). An upright rectangle fills only
    /// the rows that differ (its first, its last, one between), each under its parent's row.
    /// </summary>
    public static ClipCover Bounded(SceneClip clip, in ClipCover parent, SceneWork work)
    {
        var box = clip.Box;
        int top = Math.Max(box.Top, work.Top), bottom = Math.Min(box.Bottom, work.Top + work.Rows), left = box.Left, right = box.Right;
        if (parent.Clips)
        {
            (top, bottom, left, right) = (Math.Max(top, parent.Top), Math.Min(bottom, parent.Bottom), Math.Max(left, parent.Left), Math.Min(right, parent.Right));
        }

        if (box.IsEmpty || top >= bottom || left >= right)
        {
            return Nothing;
        }

        int stride = box.Right - box.Left, rows = bottom - top;
        int stored = rows;
        if (box.Rectangle)
        {
            // The rows that differ: a change of the rectangle's kind of row, or of the parent's row.
            stored = 0;
            for (int y = top, kind = -1, under = -1; y < bottom; y++)
            {
                int k = box.Kind(y), u = parent.Clips ? parent.StoredRow(y) : -1;
                if (k != kind || u != under)
                {
                    (stored, kind, under) = (stored + 1, k, u);
                }
            }
        }

        float[] values = ArrayPool<float>.Shared.Rent(stored * stride);
        int[]? map = stored < rows ? ArrayPool<int>.Shared.Rent(rows) : null;
        bool kept = false;
        try
        {
            if (!box.Rectangle)
            {
                var span = values.AsSpan(0, rows * stride);
                span.Clear();
                work.Rasterizer.Fill(clip.Outline, 1f, -box.Left, -top, span, stride, rows, clip.Rule, work.Flatness);
                if (parent.Clips)
                {
                    for (int y = top; y < bottom; y++)
                    {
                        var row = values.AsSpan((y - top) * stride + left - box.Left, right - left);
                        Product(row, parent.Row(y).Slice(left - parent.Origin, right - left), row);
                    }
                }
            }
            else
            {
                for (int y = top, at = -1, kind = -1, under = -1; y < bottom; y++)
                {
                    int k = box.Kind(y), u = parent.Clips ? parent.StoredRow(y) : -1;
                    if (k != kind || u != under)
                    {
                        (at, kind, under) = (at + 1, k, u);
                        var row = values.AsSpan(at * stride, stride);
                        row.Clear();
                        work.Rasterizer.Fill(clip.Outline, 1f, -box.Left, -y, row, stride, 1, clip.Rule, work.Flatness);
                        if (parent.Clips)
                        {
                            var inside = row.Slice(left - box.Left, right - left);
                            Product(inside, parent.Row(y).Slice(left - parent.Origin, right - left), inside);
                        }
                    }

                    if (map is not null)
                    {
                        map[y - top] = at;
                    }
                }
            }

            kept = true;
            return new ClipCover(values, map, left, top, right, bottom, box.Left, stride);
        }
        finally
        {
            if (!kept)
            {
                ArrayPool<float>.Shared.Return(values);
                if (map is not null)
                {
                    ArrayPool<int>.Shared.Return(map);
                }
            }
        }
    }

    // into[i] = a[i] * b[i] (into may be a): a vector at a time, then the rest one by one; each product rounded once, so the
    // same values either way.
    private static void Product(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> into)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (new Vector<float>(a[i..]) * new Vector<float>(b[i..])).CopyTo(into[i..]);
            }
        }

        for (; i < a.Length; i++)
        {
            into[i] = a[i] * b[i];
        }
    }
}

/// <summary>A soft mask and its content made ready when a band first needs it.</summary>
internal sealed record SceneMask(DocumentSoftMask Mask, Lazy<DocumentScene> Scene);

/// <summary>The buffers one band is drawn with: coverage (a value a pixel) and colour (straight red, green, blue, alpha), the band's rows of the surface; the rasterizer and the flatness it fills curves within.</summary>
internal sealed class SceneWork(int width, int top, int rows, float[] coverage, float[] colour, IOutlineRasterizer rasterizer, float flatness)
{
    public int Width { get; } = width;

    public int Top { get; } = top;

    public int Rows { get; } = rows;

    public float[] Coverage { get; } = coverage;

    public float[] Colour { get; } = colour;

    public IOutlineRasterizer Rasterizer { get; } = rasterizer;

    /// <summary>How far, in pixels, a fill's flattened curve may stray (<see cref="DocumentDrawOptions.FillFlatness"/>).</summary>
    public float Flatness { get; } = flatness;

    public Span<float> CoverageSpan => Coverage.AsSpan(0, Width * Rows);
}

/// <summary>A part made ready: its place in the content, clip, group, soft mask and blend mode, its pixel box on the surface, and how it renders.</summary>
internal sealed class SceneItem(SceneItem.Renderer render)
{
    /// <summary>
    /// Writes a part's coverage over [x0, x1) x [y0, y1) (surface rows within the band) into the work's coverage, and its
    /// colours into the work's colours unless they are one: then true, with that colour.
    /// </summary>
    public delegate bool Renderer(SceneWork work, int x0, int x1, int y0, int y1, out (float R, float G, float B, float A) solid);

    public int Order { get; set; }

    public int Sequence { get; set; }

    public int Clip { get; set; } = -1;

    public int Group { get; set; } = -1;

    public int SoftMask { get; set; } = -1;

    public DocumentBlend Blend { get; set; }

    public int Left { get; set; }

    public int Top { get; set; }

    public int Right { get; set; }

    public int Bottom { get; set; }

    /// <summary>The part's coverage and colours over a box of the band (see <see cref="Renderer"/>).</summary>
    public bool Render(SceneWork work, int x0, int x1, int y0, int y1, out (float R, float G, float B, float A) solid) => render(work, x0, x1, y0, y1, out solid);
}

/// <summary>An outline filled by its rule with a paint (a path's fill or stroke, a glyph run).</summary>
internal sealed class OutlineItem(GlyphOutline outline, OutlineFill rule, ScenePaint paint)
{
    public bool Render(SceneWork work, int x0, int x1, int y0, int y1, out (float R, float G, float B, float A) solid)
    {
        work.Rasterizer.Fill(outline, 1f, 0f, -work.Top, work.CoverageSpan, work.Width, work.Rows, rule, work.Flatness);
        if (paint.Solid(out solid))
        {
            return true;
        }

        paint.Colours(work, x0, x1, y0, y1);
        return false;
    }
}

/// <summary>A shading painted on its own: everywhere it reaches (under its clip), its colours.</summary>
internal sealed class ShadingItem(ShadingPaint paint)
{
    public bool Render(SceneWork work, int x0, int x1, int y0, int y1, out (float R, float G, float B, float A) solid)
    {
        solid = default;
        int w = work.Width;
        for (int y = y0; y < y1; y++)
        {
            work.Coverage.AsSpan((y - work.Top) * w + x0, x1 - x0).Fill(1f);
        }

        paint.Colours(work, x0, x1, y0, y1);
        return false;
    }
}

/// <summary>An image placed by its matrix: its quadrilateral's coverage, its pixels sampled through the inverse map (blocks when magnified, means when shrunk), its mask, opacity and stencil colour.</summary>
internal sealed class ImageItem
{
    private readonly GlyphOutline _quad;
    private readonly DocumentImage _image;
    private readonly double[] _toImage;   // surface pixel to the image's unit square (a homography, 9 numbers)
    private readonly Lazy<Source?> _source;

    public ImageItem(GlyphOutline quad, DocumentImage image, Affine toUnit, DrawContext context, int pageIndex)
    {
        (_quad, _image) = (quad, image);
        // The map from surface pixels to the unit square, as the shared homography of four corners.
        var unit = toUnit.Inverse();
        var (x0, y0) = unit.Apply(0, 0);
        var (x1, y1) = unit.Apply(1, 0);
        var (x2, y2) = unit.Apply(1, 1);
        var (x3, y3) = unit.Apply(0, 1);
        _toImage = PerspectiveWarp.Homography([x0, y0, x1, y1, x2, y2, x3, y3], [0, 0, 1, 0, 1, 1, 0, 1]);
        _source = new Lazy<Source?>(() =>
        {
            try
            {
                var pixels = image.Stencil is null ? image.Decode() : null;
                var mask = image.Mask();
                return new Source(pixels, mask);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException)
            {
                context.Issue(DocumentIssue.NotDrawn, $"page {pageIndex + 1}: an image ({image.Encoding}, {image.PixelWidth} x {image.PixelHeight}) is not drawn: {e.Message}");
                return null;
            }
        }, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public bool Render(SceneWork work, int x0, int x1, int y0, int y1, out (float R, float G, float B, float A) solid)
    {
        solid = default;
        if (_source.Value is not { } source || (source.Pixels is null && _image.Stencil is null))
        {
            return true;   // nothing drawn: no coverage was written
        }

        work.Rasterizer.Fill(_quad, 1f, 0f, -work.Top, work.CoverageSpan, work.Width, work.Rows, OutlineFill.NonZero, work.Flatness);
        var h = _toImage;
        var pixels = source.Pixels;
        var mask = source.Mask;
        int iw = pixels?.Width ?? mask!.Width, ih = pixels?.Height ?? mask!.Height;
        // The footprint of a surface pixel in the image's pixels (the map is affine here: its derivatives are constant).
        double fx = (Math.Abs(h[0]) + Math.Abs(h[1])) * iw, fy = (Math.Abs(h[3]) + Math.Abs(h[4])) * ih;
        var stencil = _image.Stencil;
        float alpha = _image.Alpha;
        int w = work.Width;
        for (int y = y0; y < y1; y++)
        {
            int row = (y - work.Top) * w;
            double py = y + 0.5;
            for (int x = x0; x < x1; x++)
            {
                if (work.Coverage[row + x] <= 0f)
                {
                    continue;
                }

                double px = x + 0.5, d = h[6] * px + h[7] * py + h[8];
                double u = (h[0] * px + h[1] * py + h[2]) / d, v = (h[3] * px + h[4] * py + h[5]) / d;
                int k = 4 * (row + x);
                float a = alpha;
                if (mask is not null)
                {
                    a *= source.Sample(mask, 0, u, v, fx * mask.Width / iw, fy * mask.Height / ih);
                }

                if (stencil is { } s)
                {
                    (work.Colour[k], work.Colour[k + 1], work.Colour[k + 2], work.Colour[k + 3]) = (s.Red, s.Green, s.Blue, a * s.Alpha);
                    continue;
                }

                float r = source.Sample(pixels!, 0, u, v, fx, fy);
                float g = pixels!.Channels == 3 ? source.Sample(pixels, 1, u, v, fx, fy) : r;
                float b = pixels.Channels == 3 ? source.Sample(pixels, 2, u, v, fx, fy) : r;
                (work.Colour[k], work.Colour[k + 1], work.Colour[k + 2], work.Colour[k + 3]) = (r, g, b, a);
            }
        }

        return false;
    }

    // The decoded pixels and mask (8-bit planes read as they are, else their values).
    private sealed class Source
    {
        private readonly ImagePlanes? _pixels;
        private readonly ImagePlanes? _mask;

        public Source(ImageData? pixels, ImageData? mask)
        {
            (Pixels, Mask) = (pixels, mask);
            _pixels = pixels is null ? null : new ImagePlanes(pixels);
            _mask = mask is null ? null : new ImagePlanes(mask);
        }

        public ImageData? Pixels { get; }

        public ImageData? Mask { get; }

        // A channel at (u, v) of the unit square: the pixel there when a surface pixel covers about one image pixel or
        // less (a magnified image's blocks), else the mean of the pixels its footprint (fx x fy image pixels) covers.
        public float Sample(ImageData image, int channel, double u, double v, double fx, double fy)
        {
            var planes = ReferenceEquals(image, Pixels) ? _pixels! : _mask!;
            int w = image.Width, h = image.Height;
            double ix = u * w, iy = v * h;
            if (fx <= 1.5 && fy <= 1.5)
            {
                return planes.At(channel, Math.Clamp((int)Math.Floor(ix), 0, w - 1), Math.Clamp((int)Math.Floor(iy), 0, h - 1));
            }

            int x0 = Math.Clamp((int)Math.Floor(ix - fx / 2), 0, w - 1), x1 = Math.Clamp((int)Math.Ceiling(ix + fx / 2), x0 + 1, w);
            int y0 = Math.Clamp((int)Math.Floor(iy - fy / 2), 0, h - 1), y1 = Math.Clamp((int)Math.Ceiling(iy + fy / 2), y0 + 1, h);
            int sx = Math.Max(1, (x1 - x0) / 8), sy = Math.Max(1, (y1 - y0) / 8);
            float sum = 0;
            int n = 0;
            for (int yy = y0; yy < y1; yy += sy)
            {
                for (int xx = x0; xx < x1; xx += sx)
                {
                    sum += planes.At(channel, xx, yy);
                    n++;
                }
            }

            return sum / n;
        }
    }

    // An image's channels read where they are: its 8-bit planes through its levels, or its float values.
    private sealed class ImagePlanes
    {
        private readonly ReadOnlyMemory<byte>[]? _planes;
        private readonly float[]? _levels;
        private readonly float[]? _values;
        private readonly int _width, _size;

        public ImagePlanes(ImageData image)
        {
            (_width, _size) = (image.Width, image.Width * image.Height);
            if (image.TryGetPlanes(out var planes, out var levels))
            {
                (_planes, _levels) = (planes, levels.ToArray());
            }
            else
            {
                _values = image.Pixels;
            }
        }

        public float At(int channel, int x, int y) => _planes is { } planes
            ? _levels![planes[channel].Span[y * _width + x]]
            : _values![channel * _size + y * _width + x];
    }
}

/// <summary>Where an item's colours come from: one colour, or a paint that gives each pixel's (a shading, a tiling).</summary>
internal sealed class ScenePaint
{
    private readonly (float R, float G, float B, float A)? _solid;
    private readonly Action<SceneWork, int, int, int, int>? _colours;

    /// <summary>One colour, with its opacity.</summary>
    public ScenePaint(DocumentColour colour) => _solid = (colour.Red, colour.Green, colour.Blue, colour.Alpha);

    /// <summary>Each pixel's colour from <paramref name="colours"/> (see <see cref="Colours"/>).</summary>
    public ScenePaint(Action<SceneWork, int, int, int, int> colours) => _colours = colours;

    /// <summary>One colour for every pixel (straight red, green, blue and alpha): true with it; false when <see cref="Colours"/> gives them.</summary>
    public bool Solid(out (float R, float G, float B, float A) colour)
    {
        colour = _solid ?? default;
        return _solid is not null;
    }

    /// <summary>The colours of the pixels [x0, x1) x [y0, y1) of the band (straight, with their alpha) into the work's colours.</summary>
    public void Colours(SceneWork work, int x0, int x1, int y0, int y1) => _colours?.Invoke(work, x0, x1, y0, y1);
}

/// <summary>The sixteen blend modes (ISO 32000-2, 11.3.5): a mix of the colour under a part and the part's.</summary>
internal static class BlendModes
{
    public static (float R, float G, float B) Mix(DocumentBlend mode, float br, float bg, float bb, float sr, float sg, float sb)
    {
        switch (mode)
        {
            case DocumentBlend.Hue:
                return SetLum(SetSat(sr, sg, sb, Sat(br, bg, bb)), Lum(br, bg, bb));
            case DocumentBlend.Saturation:
                return SetLum(SetSat(br, bg, bb, Sat(sr, sg, sb)), Lum(br, bg, bb));
            case DocumentBlend.Color:
                return SetLum((sr, sg, sb), Lum(br, bg, bb));
            case DocumentBlend.Luminosity:
                return SetLum((br, bg, bb), Lum(sr, sg, sb));
            default:
                return (Separable(mode, br, sr), Separable(mode, bg, sg), Separable(mode, bb, sb));
        }
    }

    private static float Separable(DocumentBlend mode, float b, float s) => mode switch
    {
        DocumentBlend.Multiply => b * s,
        DocumentBlend.Screen => b + s - b * s,
        DocumentBlend.Overlay => HardLight(s, b),
        DocumentBlend.Darken => MathF.Min(b, s),
        DocumentBlend.Lighten => MathF.Max(b, s),
        DocumentBlend.ColorDodge => b <= 0f ? 0f : s >= 1f ? 1f : MathF.Min(1f, b / (1f - s)),
        DocumentBlend.ColorBurn => b >= 1f ? 1f : s <= 0f ? 0f : 1f - MathF.Min(1f, (1f - b) / s),
        DocumentBlend.HardLight => HardLight(b, s),
        DocumentBlend.SoftLight => s <= 0.5f ? b - (1f - 2f * s) * b * (1f - b) : b + (2f * s - 1f) * (SoftD(b) - b),
        DocumentBlend.Difference => MathF.Abs(b - s),
        DocumentBlend.Exclusion => b + s - 2f * b * s,
        _ => s,
    };

    private static float HardLight(float b, float s) => s <= 0.5f ? b * 2f * s : b + (2f * s - 1f) - b * (2f * s - 1f);

    private static float SoftD(float x) => x <= 0.25f ? ((16f * x - 12f) * x + 4f) * x : MathF.Sqrt(x);

    private static float Lum(float r, float g, float b) => 0.3f * r + 0.59f * g + 0.11f * b;

    private static (float R, float G, float B) SetLum((float R, float G, float B) c, float l)
    {
        float d = l - Lum(c.R, c.G, c.B);
        return Clip(c.R + d, c.G + d, c.B + d);
    }

    private static (float R, float G, float B) Clip(float r, float g, float b)
    {
        float l = Lum(r, g, b), n = MathF.Min(r, MathF.Min(g, b)), x = MathF.Max(r, MathF.Max(g, b));
        if (n < 0f && l - n > 0f)
        {
            (r, g, b) = (l + (r - l) * l / (l - n), l + (g - l) * l / (l - n), l + (b - l) * l / (l - n));
        }

        if (x > 1f && x - l > 0f)
        {
            (r, g, b) = (l + (r - l) * (1f - l) / (x - l), l + (g - l) * (1f - l) / (x - l), l + (b - l) * (1f - l) / (x - l));
        }

        return (r, g, b);
    }

    private static float Sat(float r, float g, float b) => MathF.Max(r, MathF.Max(g, b)) - MathF.Min(r, MathF.Min(g, b));

    private static (float R, float G, float B) SetSat(float r, float g, float b, float s)
    {
        Span<float> c = [r, g, b];
        int max = c[0] >= c[1] ? (c[0] >= c[2] ? 0 : 2) : (c[1] >= c[2] ? 1 : 2);
        int min = c[0] < c[1] ? (c[0] < c[2] ? 0 : 2) : (c[1] < c[2] ? 1 : 2);
        if (max == min)
        {
            return (0f, 0f, 0f);
        }

        int mid = 3 - max - min;
        if (c[max] > c[min])
        {
            c[mid] = (c[mid] - c[min]) * s / (c[max] - c[min]);
            c[max] = s;
        }
        else
        {
            c[mid] = c[max] = 0f;
        }

        c[min] = 0f;
        return (c[0], c[1], c[2]);
    }
}
