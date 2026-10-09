// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of an augmentation of detection and segmentation samples (<see cref="IAugmentation"/>) on synthetic samples:
/// objects drawn as rectangles of a value their class decides, apart from each other, each with its box, a mask and its class in a pixel-class
/// map; other samples to draw are made the same way.
/// <list type="bullet">
/// <item>the result is well formed: a class, and with masks a mask of the image's size, per box; boxes inside the image
/// and not empty; pixel classes of the image's size; classes among those given;</item>
/// <item>the same sample and random state give the same result, and the sample given is not changed;</item>
/// <item>every box holds its mask (the mask's pixels lie within the box, to a pixel);</item>
/// <item>with <see cref="PixelClassesFollowMasks"/>, every mask's pixels have its object's class in the pixel classes;</item>
/// <item>with <see cref="ColorsKept"/>, the pixels inside each mask (off its edge) are closer to its object's value than to the background's:
/// boxes and masks moved with the pixels.</item>
/// </list>
/// </summary>
public sealed class AugmentationSuite : ContractSuite<IAugmentation>
{
    /// <inheritdoc />
    public override string Name => "augmentation";

    /// <summary>Whether the augmentation keeps the objects' colours (false for colour changes, mixing and erasing), so pixels inside masks are checked.</summary>
    public bool ColorsKept { get; init; } = true;

    /// <summary>Whether the pixel classes keep following the masks (false for mixing, which keeps one sample's pixel classes).</summary>
    public bool PixelClassesFollowMasks { get; init; } = true;

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("no objects", seed: 1, width: 24, height: 16, channels: 3, objects: 0, masks: false, classes: false);
        yield return Case("one object, boxes only", seed: 2, width: 32, height: 32, channels: 3, objects: 1, masks: false, classes: false);
        yield return Case("three objects with masks and pixel classes", seed: 3, width: 48, height: 40, channels: 3, objects: 3, masks: true, classes: true);
        yield return Case("grey, five objects with masks", seed: 4, width: 40, height: 56, channels: 1, objects: 5, masks: true, classes: false);
        yield return Case("pixel classes only", seed: 5, width: 33, height: 21, channels: 3, objects: 2, masks: false, classes: true);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int max = large ? 160 : 64;
        return Case("random", random.Next(), random.Next(12, max), random.Next(12, max), random.Next(2) == 0 ? 1 : 3, random.Next(0, 6), random.Next(2) == 0, random.Next(2) == 0);
    }

    /// <inheritdoc />
    public override void Run(IAugmentation implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int seed = (int)d["seed"]!, width = (int)d["width"]!, height = (int)d["height"]!, channels = (int)d["channels"]!, objects = (int)d["objects"]!;
        bool masks = (bool)d["masks"]!, classes = (bool)d["classes"]!;
        var sample = Synthetic(new Random(seed), width, height, channels, objects, masks, classes);
        var pixelsBefore = (float[])sample.Image.Pixels.Clone();
        var boxesBefore = sample.Boxes.ToArray();
        AugmentationContext Context() => new(new Random(seed ^ 0x5bd1e995), r => Synthetic(new Random(r.Next()), r.Next(12, 48), r.Next(12, 48), channels, r.Next(0, 4), masks, classes));

        var result = implementation.Apply(sample, Context());
        var again = implementation.Apply(sample, Context());
        checks.Check("the sample given is not changed", sample.Image.Pixels.AsSpan().SequenceEqual(pixelsBefore) && sample.Boxes.SequenceEqual(boxesBefore),
            "the augmentation changed its input");
        checks.Compare("the same random state gives the same sample", Same(result, again));

        int w = result.Width, h = result.Height;
        string? shape = result.Labels.Count != result.Boxes.Count ? $"{result.Boxes.Count} boxes and {result.Labels.Count} classes"
            : result.Masks is { } m && (m.Count != result.Count || m.Any(x => x.Length != w * h)) ? $"{m.Count} masks for {result.Count} boxes, or one not {w} x {h}"
            : result.PixelClasses is { } p && p.Length != w * h ? $"{p.Length} pixel classes for {w} x {h}"
            : result.Labels.Any(l => l is < 1 or > 6) ? $"classes {string.Join(", ", result.Labels)} outside those given (1 to 6)" : null;
        checks.Compare("well formed: a class (and mask) per box, sizes that agree", shape);
        checks.Compare("boxes inside the image and not empty", result.Boxes.Select((b, i) => (b, i))
            .Where(x => !(x.b.Width > 0 && x.b.Height > 0 && x.b.X >= -1e-3f && x.b.Y >= -1e-3f && x.b.Right <= w + 1e-3f && x.b.Bottom <= h + 1e-3f))
            .Select(x => $"box {x.i} is {x.b} in a {w} x {h} image").FirstOrDefault());
        if (result.Masks is not { } outMasks || shape is not null)
        {
            checks.Skip("every box holds its mask", "no masks");
            checks.Skip("pixel classes follow the masks", "no masks");
            checks.Skip("pixels move with their masks", "no masks");
            return;
        }

        checks.Compare("every box holds its mask", Enumerable.Range(0, result.Count).Select(i =>
        {
            var bounds = ObjectMask.PixelBounds(outMasks[i], w, h);
            var box = result.Boxes[i];
            return bounds.Width <= 0 ? $"object {i}'s mask is empty"
                : bounds.X < box.X - 1 || bounds.Y < box.Y - 1 || bounds.Right > box.Right + 1 || bounds.Bottom > box.Bottom + 1 ? $"object {i}: mask {bounds}, box {box}" : null;
        }).FirstOrDefault(x => x is not null));
        if (PixelClassesFollowMasks && result.PixelClasses is { } map)
        {
            checks.Compare("pixel classes follow the masks", Enumerable.Range(0, result.Count).Select(i =>
                Enumerable.Range(0, w * h).Where(k => outMasks[i][k] != 0 && map[k] != result.Labels[i]).Select(k => (int?)k).FirstOrDefault() is { } k
                    ? $"object {i} (class {result.Labels[i]}): pixel ({k % w}, {k / w}) has class {map[k]}" : null).FirstOrDefault(x => x is not null));
        }
        else
        {
            checks.Skip("pixel classes follow the masks", PixelClassesFollowMasks ? "no pixel classes" : "not asked");
        }

        if (!ColorsKept)
        {
            checks.Skip("pixels move with their masks", "the augmentation changes colours");
            return;
        }

        checks.Compare("pixels move with their masks", Enumerable.Range(0, result.Count).Select(i =>
        {
            // Every mask pixel off its edge (its four neighbours in the mask too) must be closer to the object's value than to
            // the background's: an enlarged or shrunk image blends the object's edge over half a source pixel, which the
            // nearest-pixel mask does not, so a pixel there is at least half the object.
            var mask = outMasks[i];
            float expected = Value(result.Labels[i]);
            int inside = 0, wrong = 0;
            for (int y = 1; y < h - 1; y++)
            {
                for (int x = 1; x < w - 1; x++)
                {
                    int k = y * w + x;
                    if (mask[k] == 0 || mask[k - 1] == 0 || mask[k + 1] == 0 || mask[k - w] == 0 || mask[k + w] == 0)
                    {
                        continue;
                    }

                    inside++;
                    float v = result.Image.Pixels[k];
                    wrong += Math.Abs(v - expected) < Math.Abs(v - Background) ? 0 : 1;
                }
            }

            return wrong > inside / 10 ? $"object {i} (class {result.Labels[i]}): {wrong} of {inside} inner pixels are not its value {expected}" : null;
        }).FirstOrDefault(x => x is not null));
    }

    // The value an object of class c is drawn with, and the background's.
    private static float Value(int c) => 0.2f + 0.13f * c;

    private const float Background = 0.05f;

    /// <summary>
    /// A synthetic sample: objects of classes 1 to 6 as rectangles of their class's value on a dark background, laid out so
    /// they do not overlap, with their boxes and, as asked, masks and pixel classes (0 for the background).
    /// </summary>
    public static AnnotatedImage Synthetic(Random random, int width, int height, int channels, int objects, bool masks, bool pixelClasses)
    {
        ArgumentNullException.ThrowIfNull(random);
        var pixels = new float[channels * width * height];
        Array.Fill(pixels, Background);
        int[]? map = pixelClasses ? new int[width * height] : null;
        var boxes = new List<BoundingBox>();
        var labels = new List<int>();
        var rasters = new List<byte[]>();
        int columns = Math.Max(1, (int)Math.Ceiling(Math.Sqrt(objects))), rows = Math.Max(1, (objects + columns - 1) / columns);
        int cellW = width / columns, cellH = height / rows;
        for (int i = 0; i < objects && cellW >= 3 && cellH >= 3; i++)
        {
            int cx = i % columns * cellW, cy = i / columns * cellH;
            // A pixel of background at least between objects: no two touch.
            int bw = random.Next(Math.Max(2, cellW / 2), cellW), bh = random.Next(Math.Max(2, cellH / 2), cellH);
            int x = cx + random.Next(0, cellW - bw), y = cy + random.Next(0, cellH - bh);
            int label = random.Next(1, 7);
            var box = new BoundingBox(x, y, bw, bh);
            var mask = AnnotatedImage.Fill(box, width, height);
            for (int k = 0; k < mask.Length; k++)
            {
                if (mask[k] == 0)
                {
                    continue;
                }

                for (int c = 0; c < channels; c++)
                {
                    pixels[c * width * height + k] = Value(label);
                }

                if (map is not null)
                {
                    map[k] = label;
                }
            }

            boxes.Add(box);
            labels.Add(label);
            rasters.Add(mask);
        }

        return new AnnotatedImage(new ImageData(pixels, channels, height, width), [.. boxes], [.. labels], masks ? [.. rasters] : null, map);
    }

    private static string? Same(AnnotatedImage a, AnnotatedImage b) =>
        Comparisons.Exact((a.Width, a.Height, a.Image.Channels, a.Count), (b.Width, b.Height, b.Image.Channels, b.Count))
        ?? Comparisons.Difference(a.Image.Pixels, b.Image.Pixels, 0f)
        ?? Comparisons.Difference(a.Labels, b.Labels)
        ?? (a.Boxes.SequenceEqual(b.Boxes) ? null : "the boxes differ")
        ?? ((a.Masks is null) == (b.Masks is null) && (a.Masks is null || a.Masks.Zip(b.Masks!).All(p => p.First.AsSpan().SequenceEqual(p.Second))) ? null : "the masks differ")
        ?? ((a.PixelClasses is null) == (b.PixelClasses is null) && (a.PixelClasses is null || a.PixelClasses.AsSpan().SequenceEqual(b.PixelClasses)) ? null : "the pixel classes differ");

    private static ContractCase Case(string name, int seed, int width, int height, int channels, int objects, bool masks, bool classes) =>
        new($"{name} ({width} x {height} x {channels}, {objects} objects{(masks ? ", masks" : "")}{(classes ? ", pixel classes" : "")})", new JsonObject
        {
            ["seed"] = seed,
            ["width"] = width,
            ["height"] = height,
            ["channels"] = channels,
            ["objects"] = objects,
            ["masks"] = masks,
            ["classes"] = classes,
        });
}
