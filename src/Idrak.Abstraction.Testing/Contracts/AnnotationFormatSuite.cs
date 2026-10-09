// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>What an annotation format keeps of a dataset, so <see cref="AnnotationFormatSuite"/> checks those parts and no others.</summary>
public sealed record AnnotationFormatTraits
{
    /// <summary>The file the format writes in the dataset's folder ("annotations.json"), or null when it writes the folder itself.</summary>
    public string? FileName { get; init; }

    /// <summary>The class names (in order) survive a round trip.</summary>
    public bool ClassNames { get; init; } = true;

    /// <summary>COCO's crowd flag survives.</summary>
    public bool Crowd { get; init; }

    /// <summary>VOC's difficult flag survives.</summary>
    public bool Difficult { get; init; }

    /// <summary>VOC's truncated flag survives.</summary>
    public bool Truncated { get; init; }

    /// <summary>VOC's pose survives.</summary>
    public bool Pose { get; init; }

    /// <summary>An object's area survives (where it had one).</summary>
    public bool Area { get; init; }

    /// <summary>Image and object ids survive (where they had them).</summary>
    public bool Ids { get; init; }

    /// <summary>A mask of one polygon survives (its points within <see cref="BoxTolerance"/>).</summary>
    public bool Polygons { get; init; }

    /// <summary>A run-length mask survives exactly.</summary>
    public bool RunLength { get; init; }

    /// <summary>How far a box edge or polygon point may move in a round trip, in pixels (formats that normalize lose a little).</summary>
    public double BoxTolerance { get; init; } = 1e-3;
}

/// <summary>
/// An annotation format as the suite drives it: its reading (a path in, a dataset out) and writing. The format contract
/// lives with the package that uses it (<c>IAnnotationFormat</c> in Idrak.Data) and this kit depends on Idrak.Abstraction
/// alone, so a test wraps the format in these delegates.
/// </summary>
/// <param name="read">Reads the dataset at a path.</param>
/// <param name="write">Writes a dataset at a path.</param>
/// <param name="traits">What the format keeps.</param>
public sealed class AnnotationFormatUnderTest(Func<string, AnnotatedDataset> read, Action<AnnotatedDataset, string> write, AnnotationFormatTraits traits)
{
    /// <summary>Reads the dataset at <paramref name="path"/>.</summary>
    public AnnotatedDataset Read(string path) => read(path);

    /// <summary>Writes <paramref name="dataset"/> at <paramref name="path"/>.</summary>
    public void Write(AnnotatedDataset dataset, string path) => write(dataset, path);

    /// <summary>What the format keeps.</summary>
    public AnnotationFormatTraits Traits { get; } = traits ?? throw new ArgumentNullException(nameof(traits));
}

/// <summary>
/// The checks of a detection or segmentation dataset format (<see cref="AnnotationFormatUnderTest"/>) on generated datasets,
/// written with their images (PPM files in <c>images/</c> under a temporary folder) and read back:
/// <list type="bullet">
/// <item>the same images (by file), sizes and classes, and each image's objects in order with the same classes and boxes
/// (within <see cref="AnnotationFormatTraits.BoxTolerance"/>) and what else the format keeps (flags, areas, ids, masks);</item>
/// <item>reading again gives the same dataset, and so does writing what was read and reading that;</item>
/// <item>images without objects and classes without objects are kept;</item>
/// <item>a path that does not exist is refused with an <see cref="IOException"/> (file or folder not found).</item>
/// </list>
/// </summary>
public sealed class AnnotationFormatSuite : ContractSuite<AnnotationFormatUnderTest>
{
    /// <inheritdoc />
    public override string Name => "annotation-format";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one image, one object", seed: 1, images: 1, classes: 1, objects: 1, fractional: false);
        yield return Case("three images, whole-pixel boxes, an image without objects", seed: 2, images: 3, classes: 3, objects: 4, fractional: false);
        yield return Case("sub-pixel boxes, classes without objects", seed: 3, images: 4, classes: 6, objects: 3, fractional: true);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("random", random.Next(), random.Next(1, large ? 40 : 6), random.Next(1, 8), random.Next(0, large ? 30 : 6), random.Next(2) == 0);
    }

    /// <inheritdoc />
    public override void Run(AnnotationFormatUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        var traits = implementation.Traits;
        string root = Path.Combine(Path.GetTempPath(), "idrak-annotations-" + Guid.NewGuid().ToString("N"));
        try
        {
            var dataset = Generate(new Random((int)d["seed"]!), (int)d["images"]!, (int)d["classes"]!, (int)d["objects"]!, (bool)d["fractional"]!, traits, root);
            string path = traits.FileName is null ? root : Path.Combine(root, traits.FileName);
            implementation.Write(dataset, path);
            var back = implementation.Read(path);
            checks.Compare("written and read back, the same dataset", Difference(dataset, back, traits));
            checks.Compare("read again, the same dataset", Difference(back, implementation.Read(path), traits));
            implementation.Write(back, path);
            checks.Compare("what was read, written and read again, the same dataset", Difference(back, implementation.Read(path), traits));
            try
            {
                implementation.Read(Path.Combine(root, "missing", traits.FileName ?? "dataset"));
                checks.Fail("a missing path is refused", "reading a path that does not exist did not fail");
            }
            catch (IOException)
            {
                checks.Pass("a missing path is refused");
            }
        }
        finally
        {
            try
            {
                Directory.Delete(root, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }

    // A dataset of `images` PPM images (written under root/images) with objects of `classes` classes.
    private static AnnotatedDataset Generate(Random random, int images, int classes, int objects, bool fractional, AnnotationFormatTraits traits, string root)
    {
        string folder = Path.Combine(root, "images");
        Directory.CreateDirectory(folder);
        var list = new List<ImageAnnotations>();
        long id = 1;
        for (int i = 0; i < images; i++)
        {
            int w = random.Next(16, 80), h = random.Next(16, 80);
            string file = $"image{i:D3}.ppm";
            WritePpm(Path.Combine(folder, file), w, h, random);
            int count = i == 1 ? 0 : random.Next(0, objects + 1);
            var found = new List<ObjectAnnotation>();
            for (int k = 0; k < count; k++)
            {
                float bw = Coordinate(random, 2, w - 1, fractional), bh = Coordinate(random, 2, h - 1, fractional);
                float x = Coordinate(random, 0, w - bw, fractional), y = Coordinate(random, 0, h - bh, fractional);
                var box = new BoundingBox(x, y, bw, bh);
                int kind = random.Next(3);
                ObjectMask? mask = kind == 1 && traits.Polygons ? ObjectMask.FromPolygons([[box.X + box.Width / 2, box.Y, box.Right, box.Y + box.Height / 2, box.X + box.Width / 2, box.Bottom, box.X, box.Y + box.Height / 2]])
                    : kind == 2 && traits.RunLength ? ObjectMask.FromPixels(AnnotatedImage.Fill(box, w, h), w, h) : null;
                found.Add(new ObjectAnnotation(box, random.Next(classes))
                {
                    Crowd = traits.Crowd && random.Next(5) == 0,
                    Difficult = traits.Difficult && random.Next(4) == 0,
                    Truncated = traits.Truncated && random.Next(4) == 0,
                    Pose = traits.Pose && random.Next(2) == 0 ? "Left" : null,
                    Area = traits.Area && random.Next(2) == 0 ? bw * bh * 0.75f : null,
                    Id = traits.Ids ? id++ : null,
                    Mask = mask,
                });
            }

            list.Add(new ImageAnnotations(file, w, h, found) { Id = traits.Ids ? 100 + i : null });
        }

        return new AnnotatedDataset([.. Enumerable.Range(0, classes).Select(c => $"class {c}")], list) { ImagesRoot = folder };
    }

    private static float Coordinate(Random random, float low, float high, bool fractional)
    {
        float v = low + random.NextSingle() * Math.Max(0, high - low);
        return fractional ? MathF.Round(v * 8) / 8 : MathF.Floor(v);
    }

    private static void WritePpm(string path, int width, int height, Random random)
    {
        var header = Encoding.ASCII.GetBytes($"P6\n{width} {height}\n255\n");
        var bytes = new byte[header.Length + width * height * 3];
        header.CopyTo(bytes, 0);
        random.NextBytes(bytes.AsSpan(header.Length));
        File.WriteAllBytes(path, bytes);
    }

    // What differs between two datasets in the parts the format keeps; null when nothing does.
    private static string? Difference(AnnotatedDataset expected, AnnotatedDataset actual, AnnotationFormatTraits traits)
    {
        if (traits.ClassNames ? !expected.Classes.SequenceEqual(actual.Classes) : actual.Classes.Count < expected.Classes.Count)
        {
            return $"classes [{string.Join(", ", actual.Classes)}], expected [{string.Join(", ", expected.Classes)}]";
        }

        var byFile = actual.Images.GroupBy(i => i.File.Replace('\\', '/')).ToDictionary(g => g.Key, g => g.First());
        if (byFile.Count != expected.Images.Count)
        {
            return $"{actual.Images.Count} images, expected {expected.Images.Count}";
        }

        double tolerance = traits.BoxTolerance;
        foreach (var image in expected.Images)
        {
            if (!byFile.TryGetValue(image.File.Replace('\\', '/'), out var other))
            {
                return $"{image.File} is missing";
            }

            if (other.Width != image.Width || other.Height != image.Height || other.Objects.Count != image.Objects.Count)
            {
                return $"{image.File}: {other.Width} x {other.Height} with {other.Objects.Count} objects, expected {image.Width} x {image.Height} with {image.Objects.Count}";
            }

            if (traits.Ids && image.Id is { } imageId && other.Id != imageId)
            {
                return $"{image.File}: id {other.Id}, expected {imageId}";
            }

            for (int k = 0; k < image.Objects.Count; k++)
            {
                ObjectAnnotation e = image.Objects[k], a = other.Objects[k];
                string where = $"{image.File}, object {k}";
                if (a.Class != e.Class)
                {
                    return $"{where}: class {a.Class}, expected {e.Class}";
                }

                if (Math.Abs(a.Box.X - e.Box.X) > tolerance || Math.Abs(a.Box.Y - e.Box.Y) > tolerance || Math.Abs(a.Box.Right - e.Box.Right) > tolerance
                    || Math.Abs(a.Box.Bottom - e.Box.Bottom) > tolerance)
                {
                    return $"{where}: box {a.Box}, expected {e.Box} (within {tolerance.ToString(CultureInfo.InvariantCulture)} pixels)";
                }

                string? flags = traits.Crowd && a.Crowd != e.Crowd ? "crowd" : traits.Difficult && a.Difficult != e.Difficult ? "difficult"
                    : traits.Truncated && a.Truncated != e.Truncated ? "truncated" : traits.Pose && a.Pose != e.Pose ? "pose"
                    : traits.Area && e.Area is { } area && !(a.Area is { } read && Math.Abs(read - area) <= 1e-3 * Math.Max(1, area)) ? "area"
                    : traits.Ids && e.Id is { } objectId && a.Id != objectId ? "id" : null;
                if (flags is not null)
                {
                    return $"{where}: its {flags} differs";
                }

                if (e.Mask is { IsPolygons: true } polygons && traits.Polygons)
                {
                    var got = a.Mask?.Polygons;
                    if (got is null || got.Count != polygons.Polygons!.Count || got.Zip(polygons.Polygons).Any(p => p.First.Length != p.Second.Length
                        || p.First.Zip(p.Second).Any(v => Math.Abs(v.First - v.Second) > tolerance)))
                    {
                        return $"{where}: its polygons differ";
                    }
                }
                else if (e.Mask is { IsPolygons: false } runs && traits.RunLength)
                {
                    if (a.Mask?.Counts is not { } counts || !counts.SequenceEqual(runs.Counts!) || a.Mask.Width != runs.Width || a.Mask.Height != runs.Height)
                    {
                        return $"{where}: its run-length mask differs";
                    }
                }
            }
        }

        return null;
    }

    private static ContractCase Case(string name, int seed, int images, int classes, int objects, bool fractional) =>
        new($"{name} ({images} images, {classes} classes, up to {objects} objects{(fractional ? ", sub-pixel" : "")})", new JsonObject
        {
            ["seed"] = seed,
            ["images"] = images,
            ["classes"] = classes,
            ["objects"] = objects,
            ["fractional"] = fractional,
        });
}
