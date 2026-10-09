// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The pixels of one object: polygons (as COCO and YOLO segmentation write them) or a run-length encoding (COCO's RLE:
/// counts of alternating background and object runs, column by column, starting with background). Kept in the form it
/// was read in, so a dataset written again keeps it; <see cref="Rasterize"/> gives the pixels.
/// </summary>
public sealed class ObjectMask
{
    private readonly float[][]? _polygons;
    private readonly int[]? _counts;

    private ObjectMask(float[][]? polygons, int[]? counts, int width, int height)
    {
        _polygons = polygons;
        _counts = counts;
        Width = width;
        Height = height;
    }

    /// <summary>Polygons, each x0, y0, x1, y1, ... in pixels (at least three points); a pixel is inside when its centre is (even-odd rule).</summary>
    /// <exception cref="ArgumentException">A polygon with fewer than three points or an odd number of values.</exception>
    public static ObjectMask FromPolygons(IEnumerable<float[]> polygons)
    {
        ArgumentNullException.ThrowIfNull(polygons);
        float[][] list = [.. polygons.Select(p => (float[])(p ?? throw new ArgumentException("A polygon is null.", nameof(polygons))).Clone())];
        if (list.Length == 0 || list.Any(p => p.Length < 6 || p.Length % 2 != 0))
        {
            throw new ArgumentException("A mask's polygons each have at least three points (x, y pairs).", nameof(polygons));
        }

        return new ObjectMask(list, null, 0, 0);
    }

    /// <summary>A run-length encoding of a <paramref name="width"/> x <paramref name="height"/> mask: run lengths, column by column, background first.</summary>
    /// <exception cref="ArgumentException">The runs do not add up to the mask's pixels, or one is negative.</exception>
    public static ObjectMask FromRunLength(IReadOnlyList<int> counts, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(counts);
        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(height);
        long total = 0;
        foreach (int count in counts)
        {
            total += count >= 0 ? count : throw new ArgumentException("A run length is negative.", nameof(counts));
        }

        if (total != (long)width * height)
        {
            throw new ArgumentException($"The runs cover {total} pixels, not {width} x {height}.", nameof(counts));
        }

        return new ObjectMask(null, [.. counts], width, height);
    }

    /// <summary>The run-length encoding of <paramref name="pixels"/> (row by row, non-zero inside).</summary>
    public static ObjectMask FromPixels(ReadOnlySpan<byte> pixels, int width, int height)
    {
        if (pixels.Length != width * height)
        {
            throw new ArgumentException($"{pixels.Length} pixels for a {width} x {height} mask.", nameof(pixels));
        }

        var counts = new List<int>();
        bool inside = false;
        int run = 0;
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (pixels[y * width + x] != 0 != inside)
                {
                    counts.Add(run);
                    run = 0;
                    inside = !inside;
                }

                run++;
            }
        }

        counts.Add(run);
        return new ObjectMask(null, [.. counts], width, height);
    }

    /// <summary>Whether the mask is polygons (else a run-length encoding).</summary>
    public bool IsPolygons => _polygons is not null;

    /// <summary>The polygons, each x0, y0, x1, y1, ...; null for a run-length mask.</summary>
    public IReadOnlyList<float[]>? Polygons => _polygons;

    /// <summary>The run lengths (column by column, background first); null for polygons.</summary>
    public IReadOnlyList<int>? Counts => _counts;

    /// <summary>The width of a run-length mask (0 for polygons, which have no size of their own).</summary>
    public int Width { get; }

    /// <summary>The height of a run-length mask (0 for polygons).</summary>
    public int Height { get; }

    /// <summary>
    /// The mask on a <paramref name="width"/> x <paramref name="height"/> image, row by row: 1 inside, 0 outside. A
    /// run-length mask must have that size.
    /// </summary>
    public byte[] Rasterize(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(height);
        var pixels = new byte[width * height];
        if (_counts is not null)
        {
            if (width != Width || height != Height)
            {
                throw new ArgumentException($"A {Width} x {Height} run-length mask on a {width} x {height} image.");
            }

            int at = 0;
            for (int i = 0; i < _counts.Length; i++)
            {
                for (int k = 0; k < _counts[i]; k++, at++)
                {
                    if ((i & 1) == 1)
                    {
                        pixels[at % height * width + at / height] = 1;
                    }
                }
            }

            return pixels;
        }

        var crossings = new List<float>();
        for (int y = 0; y < height; y++)
        {
            float cy = y + 0.5f;
            crossings.Clear();
            foreach (var polygon in _polygons!)
            {
                int n = polygon.Length / 2;
                for (int i = 0, j = n - 1; i < n; j = i++)
                {
                    float xi = polygon[2 * i], yi = polygon[2 * i + 1], xj = polygon[2 * j], yj = polygon[2 * j + 1];
                    if (yi > cy != yj > cy)
                    {
                        crossings.Add(xi + (cy - yi) / (yj - yi) * (xj - xi));
                    }
                }
            }

            crossings.Sort();
            for (int k = 0; k + 1 < crossings.Count; k += 2)
            {
                // Pixels whose centre lies in [from, to).
                int from = Math.Max(0, (int)MathF.Ceiling(crossings[k] - 0.5f)), to = Math.Min(width, (int)MathF.Ceiling(crossings[k + 1] - 0.5f));
                for (int x = from; x < to; x++)
                {
                    pixels[y * width + x] = 1;
                }
            }
        }

        return pixels;
    }

    /// <summary>The smallest box around the polygons' points or the run-length mask's pixels; empty (all zero) for an empty mask.</summary>
    public BoundingBox Bounds()
    {
        if (_polygons is not null)
        {
            float x1 = float.MaxValue, y1 = float.MaxValue, x2 = float.MinValue, y2 = float.MinValue;
            foreach (var polygon in _polygons)
            {
                for (int i = 0; i < polygon.Length; i += 2)
                {
                    (x1, x2) = (Math.Min(x1, polygon[i]), Math.Max(x2, polygon[i]));
                    (y1, y2) = (Math.Min(y1, polygon[i + 1]), Math.Max(y2, polygon[i + 1]));
                }
            }

            return BoundingBox.FromCorners(x1, y1, x2, y2);
        }

        return PixelBounds(Rasterize(Width, Height), Width, Height);
    }

    /// <summary>The smallest box around the non-zero pixels of a row-by-row mask; all zero when there are none.</summary>
    public static BoundingBox PixelBounds(ReadOnlySpan<byte> pixels, int width, int height)
    {
        int x1 = width, y1 = height, x2 = -1, y2 = -1;
        for (int y = 0; y < height; y++)
        {
            var row = pixels.Slice(y * width, width);
            int first = row.IndexOfAnyExcept((byte)0);
            if (first < 0)
            {
                continue;
            }

            int last = row.LastIndexOfAnyExcept((byte)0);
            (x1, x2) = (Math.Min(x1, first), Math.Max(x2, last));
            (y1, y2) = (Math.Min(y1, y), y);
        }

        return x2 < 0 ? default : new BoundingBox(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    }

    /// <summary>
    /// COCO's compressed form of run lengths (pycocotools' <c>rleToString</c>): each count, after the third less the count
    /// two before it, in 5-bit groups as the characters 48 to 111.
    /// </summary>
    public static string EncodeCounts(IReadOnlyList<int> counts)
    {
        ArgumentNullException.ThrowIfNull(counts);
        var text = new StringBuilder();
        for (int i = 0; i < counts.Count; i++)
        {
            long x = counts[i];
            if (i > 2)
            {
                x -= counts[i - 2];
            }

            for (bool more = true; more;)
            {
                long c = x & 0x1f;
                x >>= 5;
                more = (c & 0x10) != 0 ? x != -1 : x != 0;
                if (more)
                {
                    c |= 0x20;
                }

                text.Append((char)(c + 48));
            }
        }

        return text.ToString();
    }

    /// <summary>The run lengths of COCO's compressed form (pycocotools' <c>rleFrString</c>).</summary>
    /// <exception cref="FormatException">A character outside 48 to 111, or a count cut short.</exception>
    public static int[] DecodeCounts(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var counts = new List<int>();
        for (int p = 0; p < text.Length;)
        {
            long x = 0;
            int k = 0;
            for (bool more = true; more; k++)
            {
                if (p >= text.Length)
                {
                    throw new FormatException("A compressed run-length count is cut short.");
                }

                int c = text[p++] - 48;
                if ((uint)c > 63)
                {
                    throw new FormatException($"'{text[p - 1]}' is not a character of a compressed run-length encoding.");
                }

                x |= (long)(c & 0x1f) << (5 * k);
                more = (c & 0x20) != 0;
                if (!more && (c & 0x10) != 0)
                {
                    x |= -1L << (5 * (k + 1));
                }
            }

            if (counts.Count > 2)
            {
                x += counts[^2];
            }

            counts.Add(checked((int)x));
        }

        return [.. counts];
    }
}

/// <summary>
/// One object of a dataset image: its box (pixels), its class (an index into <see cref="AnnotatedDataset.Classes"/>) and
/// what some formats add.
/// </summary>
/// <param name="Box">The object's box, in the image's pixels.</param>
/// <param name="Class">Its class: an index into the dataset's classes.</param>
public sealed record ObjectAnnotation(BoundingBox Box, int Class)
{
    /// <summary>COCO's <c>iscrowd</c>: the box covers a crowd of objects; detections inside it count neither way.</summary>
    public bool Crowd { get; init; }

    /// <summary>Pascal VOC's <c>difficult</c>: left out of the evaluation (detections of it count neither way).</summary>
    public bool Difficult { get; init; }

    /// <summary>Pascal VOC's <c>truncated</c>: the object extends beyond the image.</summary>
    public bool Truncated { get; init; }

    /// <summary>Pascal VOC's <c>pose</c> ("Left", "Frontal", ...), or null.</summary>
    public string? Pose { get; init; }

    /// <summary>COCO's <c>area</c> (the mask's area), or null when the format does not give one.</summary>
    public float? Area { get; init; }

    /// <summary>COCO's annotation id, or null.</summary>
    public long? Id { get; init; }

    /// <summary>The object's pixels, or null when only its box is known.</summary>
    public ObjectMask? Mask { get; init; }
}

/// <summary>One image of an annotated dataset: its file (relative to the dataset's images), size and objects.</summary>
/// <param name="File">The image file, relative to <see cref="AnnotatedDataset.ImagesRoot"/> (or absolute).</param>
/// <param name="Width">The image's width in pixels.</param>
/// <param name="Height">The image's height in pixels.</param>
/// <param name="Objects">Its objects.</param>
public sealed record ImageAnnotations(string File, int Width, int Height, IReadOnlyList<ObjectAnnotation> Objects)
{
    /// <summary>COCO's image id, or null.</summary>
    public long? Id { get; init; }

    /// <summary>
    /// The training sample of this image with its decoded <paramref name="pixels"/>: the boxes and classes of its objects
    /// and, when any object has a mask and <paramref name="masks"/> is set, one mask per object (an object without a mask
    /// gets its box filled). Crowd and difficult objects are left out unless <paramref name="includeIgnored"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The pixels are not this image's size.</exception>
    public AnnotatedImage ToSample(ImageData pixels, bool masks = true, bool includeIgnored = false)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        if (pixels.Width != Width || pixels.Height != Height)
        {
            throw new ArgumentException($"{File} is annotated as {Width} x {Height}, decoded as {pixels.Width} x {pixels.Height}.", nameof(pixels));
        }

        var kept = Objects.Where(o => includeIgnored || !o.Crowd && !o.Difficult).ToList();
        byte[][]? rasters = null;
        if (masks && kept.Any(o => o.Mask is not null))
        {
            rasters = [.. kept.Select(o => o.Mask?.Rasterize(Width, Height) ?? AnnotatedImage.Fill(o.Box, Width, Height))];
        }

        return new AnnotatedImage(pixels, [.. kept.Select(o => o.Box)], [.. kept.Select(o => o.Class)], rasters);
    }
}

/// <summary>
/// A detection or segmentation dataset as its annotation files describe it (COCO, YOLO, Pascal VOC, ...): the class names
/// and each image's objects. Images are named, not decoded, so a large dataset costs only its annotations in memory.
/// </summary>
/// <param name="Classes">The class names; an object's class is an index into them.</param>
/// <param name="Images">The images and their objects.</param>
public sealed record AnnotatedDataset(IReadOnlyList<string> Classes, IReadOnlyList<ImageAnnotations> Images)
{
    /// <summary>The folder image files are relative to (null: the current folder, or absolute paths).</summary>
    public string? ImagesRoot { get; init; }

    /// <summary>The id each class has in the file (COCO's category ids, which need not run from 0), or null when they are the indices.</summary>
    public IReadOnlyList<long>? ClassIds { get; init; }

    /// <summary>The full path of image <paramref name="index"/>'s file.</summary>
    public string PathOf(int index)
    {
        string file = Images[index].File;
        return ImagesRoot is null || Path.IsPathRooted(file) ? file : Path.Combine(ImagesRoot, file);
    }
}

/// <summary>
/// A training sample for detection and segmentation: an image with its objects' boxes and classes and, when known, one
/// mask per object and a class per pixel. Augmentations (<see cref="IAugmentation"/>) take one and give a new one with
/// the boxes and masks moved with the pixels. Immutable: the arrays are not copied, and are not to be changed.
/// </summary>
public sealed class AnnotatedImage
{
    private readonly BoundingBox[] _boxes;
    private readonly int[] _labels;
    private readonly byte[][]? _masks;

    /// <summary>The image with its objects (none by default).</summary>
    /// <param name="image">The pixels.</param>
    /// <param name="boxes">Each object's box, in the image's pixels.</param>
    /// <param name="labels">Each object's class.</param>
    /// <param name="masks">Each object's mask (width x height, row by row, non-zero inside), or null.</param>
    /// <param name="pixelClasses">A class for every pixel (semantic segmentation, row by row), or null.</param>
    /// <exception cref="ArgumentException">The counts or sizes do not agree.</exception>
    public AnnotatedImage(ImageData image, BoundingBox[]? boxes = null, int[]? labels = null, byte[][]? masks = null, int[]? pixelClasses = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        Image = image;
        _boxes = boxes ?? [];
        _labels = labels ?? (_boxes.Length == 0 ? [] : throw new ArgumentException("Boxes need their classes.", nameof(labels)));
        if (_labels.Length != _boxes.Length)
        {
            throw new ArgumentException($"{_boxes.Length} boxes and {_labels.Length} classes.", nameof(labels));
        }

        int plane = image.Width * image.Height;
        if (masks is not null && (masks.Length != _boxes.Length || masks.Any(m => m is null || m.Length != plane)))
        {
            throw new ArgumentException($"A mask per box ({_boxes.Length}), each {image.Width} x {image.Height}.", nameof(masks));
        }

        if (pixelClasses is not null && pixelClasses.Length != plane)
        {
            throw new ArgumentException($"{pixelClasses.Length} pixel classes for a {image.Width} x {image.Height} image.", nameof(pixelClasses));
        }

        _masks = masks;
        PixelClasses = pixelClasses;
    }

    /// <summary>The pixels.</summary>
    public ImageData Image { get; }

    /// <summary>The image's width.</summary>
    public int Width => Image.Width;

    /// <summary>The image's height.</summary>
    public int Height => Image.Height;

    /// <summary>The number of objects.</summary>
    public int Count => _boxes.Length;

    /// <summary>Each object's box, in the image's pixels.</summary>
    public IReadOnlyList<BoundingBox> Boxes => _boxes;

    /// <summary>Each object's class.</summary>
    public IReadOnlyList<int> Labels => _labels;

    /// <summary>Each object's mask (row by row, non-zero inside), or null.</summary>
    public IReadOnlyList<byte[]>? Masks => _masks;

    /// <summary>A class for every pixel, row by row, or null.</summary>
    public int[]? PixelClasses { get; }

    /// <summary>The same objects on other pixels of the same size (a colour change).</summary>
    public AnnotatedImage WithImage(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Width != Width || image.Height != Height)
        {
            throw new ArgumentException($"A {image.Width} x {image.Height} image for a {Width} x {Height} sample; move the boxes and masks too.", nameof(image));
        }

        return new AnnotatedImage(image, _boxes, _labels, _masks, PixelClasses);
    }

    /// <summary>Only the objects at <paramref name="indices"/>, in that order.</summary>
    public AnnotatedImage Select(IReadOnlyList<int> indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        return new AnnotatedImage(Image, [.. indices.Select(i => _boxes[i])], [.. indices.Select(i => _labels[i])],
            _masks is null ? null : [.. indices.Select(i => _masks[i])], PixelClasses);
    }

    /// <summary>A row-by-row mask of <paramref name="width"/> x <paramref name="height"/> with <paramref name="box"/>'s pixels (those whose centre it holds) set.</summary>
    public static byte[] Fill(BoundingBox box, int width, int height)
    {
        var pixels = new byte[width * height];
        int x1 = Math.Max(0, (int)MathF.Ceiling(box.X - 0.5f)), x2 = Math.Min(width, (int)MathF.Ceiling(box.Right - 0.5f));
        int y1 = Math.Max(0, (int)MathF.Ceiling(box.Y - 0.5f)), y2 = Math.Min(height, (int)MathF.Ceiling(box.Bottom - 0.5f));
        for (int y = y1; y < y2; y++)
        {
            pixels.AsSpan(y * width + x1, Math.Max(0, x2 - x1)).Fill(1);
        }

        return pixels;
    }
}
