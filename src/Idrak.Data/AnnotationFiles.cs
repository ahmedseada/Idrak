// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Xml;
using System.Xml.Linq;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>The library's annotation formats (<see cref="AnnotationFormats"/>), registered on the registry's first use.</summary>
internal static class AnnotationFiles
{
    internal static void RegisterAll()
    {
        AnnotationFormats.Register(new CocoFormat());
        AnnotationFormats.Register(new YoloFormat());
        AnnotationFormats.Register(new VocFormat());
    }

    internal static string Text(float value) => value.ToString(CultureInfo.InvariantCulture);

    internal static float Number(string text, string path, string what) =>
        float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) ? value
            : throw new InvalidDataException($"{path}: {what} is not a number ('{text}').");

    // The image files under `folder` (by the registered codecs' extensions), as paths relative to it with '/', in ordinal order.
    internal static List<string> Images(string folder)
    {
        var extensions = new HashSet<string>(ImageCodecs.Extensions, StringComparer.OrdinalIgnoreCase);
        return [.. Directory.EnumerateFiles(folder, "*", SearchOption.AllDirectories).Where(f => extensions.Contains(Path.GetExtension(f)))
            .Select(f => Path.GetRelativePath(folder, f).Replace('\\', '/')).Order(StringComparer.Ordinal)];
    }

    internal static (int Width, int Height) SizeOf(string path) =>
        ImageCodecs.ReadInfo(path) is { Width: > 0, Height: > 0 } info ? (info.Width, info.Height)
            : throw new InvalidDataException($"{path}: no registered image codec reads its size.");

    // classes.txt: one name a line (empty lines dropped at the end only).
    internal static List<string>? ClassesFile(string folder)
    {
        string file = Path.Combine(folder, "classes.txt");
        if (!File.Exists(file))
        {
            return null;
        }

        var lines = File.ReadAllLines(file).Select(l => l.TrimEnd('\r')).ToList();
        while (lines.Count > 0 && lines[^1].Trim().Length == 0)
        {
            lines.RemoveAt(lines.Count - 1);
        }

        return lines;
    }

    internal static void WriteClassesFile(string folder, IReadOnlyList<string> classes)
    {
        var text = new StringBuilder();
        foreach (string name in classes)
        {
            if (name.Contains('\n') || name.Contains('\r'))
            {
                throw new ArgumentException($"A class name in classes.txt is one line ('{name}').");
            }

            text.Append(name).Append('\n');
        }

        File.WriteAllText(Path.Combine(folder, "classes.txt"), text.ToString());
    }
}

/// <summary>
/// COCO's instances format ("coco" in <see cref="AnnotationFormats"/>): one JSON file with <c>images</c> (id, file_name,
/// width, height), <c>categories</c> (id, name) and <c>annotations</c> (bbox as x, y, width, height in pixels, area,
/// iscrowd, and segmentation as polygons or a run-length encoding, plain or compressed). Classes are the categories in
/// file order, their ids kept (<see cref="AnnotatedDataset.ClassIds"/>). Images are relative to
/// <see cref="AnnotationReadOptions.ImagesRoot"/> or else the JSON file's folder. The file is parsed into a compact
/// document (not a tree of objects), so large files cost about their size.
/// </summary>
public sealed class CocoFormat : IAnnotationFormat
{
    /// <inheritdoc />
    public string Name => "coco";

    /// <inheritdoc />
    public string Summary => "COCO instances JSON (a .json file with images, categories and annotations: boxes, polygon or RLE masks, crowds)";

    /// <inheritdoc />
    public bool CanRead(string path)
    {
        if (!File.Exists(path) || !path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        using var stream = File.OpenRead(path);
        var head = new byte[(int)Math.Min(stream.Length, 1 << 16)];
        stream.ReadExactly(head);
        string text = Encoding.UTF8.GetString(head);
        return text.Contains("\"images\"", StringComparison.Ordinal) && (text.Contains("\"annotations\"", StringComparison.Ordinal) || text.Contains("\"categories\"", StringComparison.Ordinal));
    }

    /// <inheritdoc />
    public AnnotatedDataset Read(string path, AnnotationReadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = File.OpenRead(path);
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(stream);
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{path}: not JSON ({e.Message}).", e);
        }

        using (document)
        {
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("images", out var images) || images.ValueKind != JsonValueKind.Array)
            {
                throw new InvalidDataException($"{path}: a COCO file is an object with an \"images\" array.");
            }

            var names = new List<string>();
            var ids = new List<long>();
            var classOf = new Dictionary<long, int>();
            if (root.TryGetProperty("categories", out var categories))
            {
                foreach (var category in categories.EnumerateArray())
                {
                    long id = Long(category, "id", path, "a category");
                    classOf[id] = names.Count;
                    ids.Add(id);
                    names.Add(category.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String ? n.GetString()! : id.ToString(CultureInfo.InvariantCulture));
                }
            }

            if (options?.Classes is { } given)
            {
                classOf = names.Select((n, i) => (n, i)).ToDictionary(x => ids[x.i], x => given.ToList().IndexOf(x.n) is var at and >= 0 ? at
                    : throw new InvalidDataException($"{path}: the category '{x.n}' is not among the classes given."));
                (names, ids) = ([.. given], [.. given.Select(n => names.IndexOf(n) is var at and >= 0 ? ids[at] : -1L)]);
            }

            var byImage = new Dictionary<long, List<ObjectAnnotation>>();
            if (root.TryGetProperty("annotations", out var annotations))
            {
                foreach (var a in annotations.EnumerateArray())
                {
                    long image = Long(a, "image_id", path, "an annotation"), category = Long(a, "category_id", path, "an annotation");
                    if (!classOf.TryGetValue(category, out int c))
                    {
                        throw new InvalidDataException($"{path}: annotation {(a.TryGetProperty("id", out var i) ? i.ToString() : "")} has category {category}, which is not in \"categories\".");
                    }

                    if (!a.TryGetProperty("bbox", out var bbox) || bbox.GetArrayLength() != 4)
                    {
                        throw new InvalidDataException($"{path}: an annotation of image {image} has no bbox [x, y, width, height].");
                    }

                    var box = new BoundingBox(bbox[0].GetSingle(), bbox[1].GetSingle(), bbox[2].GetSingle(), bbox[3].GetSingle());
                    var annotation = new ObjectAnnotation(box, c)
                    {
                        Id = a.TryGetProperty("id", out var id) ? id.GetInt64() : null,
                        Area = a.TryGetProperty("area", out var area) && area.ValueKind == JsonValueKind.Number ? area.GetSingle() : null,
                        Crowd = a.TryGetProperty("iscrowd", out var crowd) && (crowd.ValueKind == JsonValueKind.True || crowd.ValueKind == JsonValueKind.Number && crowd.GetInt32() != 0),
                        Mask = a.TryGetProperty("segmentation", out var segmentation) ? Mask(segmentation, path) : null,
                    };
                    (byImage.TryGetValue(image, out var list) ? list : byImage[image] = []).Add(annotation);
                }
            }

            var result = new List<ImageAnnotations>();
            foreach (var image in images.EnumerateArray())
            {
                long id = Long(image, "id", path, "an image");
                string file = image.TryGetProperty("file_name", out var f) && f.ValueKind == JsonValueKind.String ? f.GetString()!
                    : throw new InvalidDataException($"{path}: image {id} has no file_name.");
                int width = image.TryGetProperty("width", out var w) ? w.GetInt32() : 0, height = image.TryGetProperty("height", out var h) ? h.GetInt32() : 0;
                result.Add(new ImageAnnotations(file, width, height, byImage.Remove(id, out var objects) ? objects : []) { Id = id });
            }

            if (byImage.Count > 0)
            {
                throw new InvalidDataException($"{path}: annotations of image {byImage.Keys.First()}, which is not in \"images\".");
            }

            return new AnnotatedDataset(names, result) { ImagesRoot = options?.ImagesRoot ?? Path.GetDirectoryName(Path.GetFullPath(path)), ClassIds = ids };
        }
    }

    private static long Long(JsonElement element, string property, string path, string what) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt64()
            : throw new InvalidDataException($"{path}: {what} has no \"{property}\".");

    private static ObjectMask? Mask(JsonElement segmentation, string path)
    {
        switch (segmentation.ValueKind)
        {
            case JsonValueKind.Array when segmentation.GetArrayLength() == 0:
                return null;
            case JsonValueKind.Array:
                return ObjectMask.FromPolygons(segmentation.EnumerateArray().Select(p => p.EnumerateArray().Select(v => v.GetSingle()).ToArray()));
            case JsonValueKind.Object:
            {
                if (!segmentation.TryGetProperty("size", out var size) || size.GetArrayLength() != 2 || !segmentation.TryGetProperty("counts", out var counts))
                {
                    throw new InvalidDataException($"{path}: a run-length segmentation has \"size\" [height, width] and \"counts\".");
                }

                int height = size[0].GetInt32(), width = size[1].GetInt32();
                int[] runs = counts.ValueKind == JsonValueKind.String ? ObjectMask.DecodeCounts(counts.GetString()!) : [.. counts.EnumerateArray().Select(v => v.GetInt32())];
                return ObjectMask.FromRunLength(runs, width, height);
            }

            default:
                return null;
        }
    }

    /// <inheritdoc />
    /// <remarks>Run-length masks are written compressed; an object's area is its own, else its box's; missing ids are numbered.</remarks>
    public void Write(AnnotatedDataset dataset, string path)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = File.Create(path);
        using var json = new Utf8JsonWriter(stream);
        json.WriteStartObject();
        json.WriteStartArray("images");
        var imageIds = new long[dataset.Images.Count];
        long nextImage = dataset.Images.Select(i => i.Id ?? 0).DefaultIfEmpty(0).Max() + 1;
        for (int i = 0; i < dataset.Images.Count; i++)
        {
            var image = dataset.Images[i];
            imageIds[i] = image.Id ?? nextImage++;
            json.WriteStartObject();
            json.WriteNumber("id", imageIds[i]);
            json.WriteString("file_name", image.File.Replace('\\', '/'));
            json.WriteNumber("width", image.Width);
            json.WriteNumber("height", image.Height);
            json.WriteEndObject();
        }

        json.WriteEndArray();
        json.WriteStartArray("categories");
        for (int c = 0; c < dataset.Classes.Count; c++)
        {
            json.WriteStartObject();
            json.WriteNumber("id", CategoryId(dataset, c));
            json.WriteString("name", dataset.Classes[c]);
            json.WriteEndObject();
        }

        json.WriteEndArray();
        json.WriteStartArray("annotations");
        long nextAnnotation = dataset.Images.SelectMany(i => i.Objects).Select(o => o.Id ?? 0).DefaultIfEmpty(0).Max() + 1;
        for (int i = 0; i < dataset.Images.Count; i++)
        {
            foreach (var o in dataset.Images[i].Objects)
            {
                if ((uint)o.Class >= (uint)dataset.Classes.Count)
                {
                    throw new ArgumentException($"{dataset.Images[i].File}: class {o.Class} is outside the {dataset.Classes.Count} classes.");
                }

                json.WriteStartObject();
                json.WriteNumber("id", o.Id ?? nextAnnotation++);
                json.WriteNumber("image_id", imageIds[i]);
                json.WriteNumber("category_id", CategoryId(dataset, o.Class));
                json.WriteStartArray("bbox");
                json.WriteNumberValue(o.Box.X);
                json.WriteNumberValue(o.Box.Y);
                json.WriteNumberValue(o.Box.Width);
                json.WriteNumberValue(o.Box.Height);
                json.WriteEndArray();
                json.WriteNumber("area", o.Area ?? o.Box.Width * o.Box.Height);
                json.WriteNumber("iscrowd", o.Crowd ? 1 : 0);
                if (o.Mask is { IsPolygons: true } polygons)
                {
                    json.WriteStartArray("segmentation");
                    foreach (var polygon in polygons.Polygons!)
                    {
                        json.WriteStartArray();
                        foreach (float v in polygon)
                        {
                            json.WriteNumberValue(v);
                        }

                        json.WriteEndArray();
                    }

                    json.WriteEndArray();
                }
                else if (o.Mask is { } rle)
                {
                    json.WriteStartObject("segmentation");
                    json.WriteStartArray("size");
                    json.WriteNumberValue(rle.Height);
                    json.WriteNumberValue(rle.Width);
                    json.WriteEndArray();
                    json.WriteString("counts", ObjectMask.EncodeCounts(rle.Counts!));
                    json.WriteEndObject();
                }
                else
                {
                    json.WriteStartArray("segmentation");
                    json.WriteEndArray();
                }

                json.WriteEndObject();
            }
        }

        json.WriteEndArray();
        json.WriteEndObject();
    }

    private static long CategoryId(AnnotatedDataset dataset, int c) => dataset.ClassIds is { } ids && c < ids.Count && ids[c] >= 0 ? ids[c] : c + 1;
}

/// <summary>
/// YOLO's text format ("yolo" in <see cref="AnnotationFormats"/>): a folder with <c>images/</c> and <c>labels/</c> (the
/// same sub-folders), or images and their .txt side by side; each image's .txt holds a line per object, <c>class cx cy w h</c>
/// normalized to the image's size, or <c>class x1 y1 x2 y2 ...</c> for a polygon (YOLO segmentation). An image without a
/// .txt has no objects. Class names from <c>classes.txt</c>, else <c>data.yaml</c>'s <c>names</c>, else the class numbers.
/// Image sizes are read from the files' headers.
/// </summary>
public sealed class YoloFormat : IAnnotationFormat
{
    /// <inheritdoc />
    public string Name => "yolo";

    /// <inheritdoc />
    public string Summary => "YOLO text (a folder of images/ and labels/: a .txt per image, normalized class cx cy w h or polygons; classes.txt or data.yaml names)";

    /// <inheritdoc />
    public bool CanRead(string path) =>
        Directory.Exists(path) && (Directory.Exists(Path.Combine(path, "labels")) || File.Exists(Path.Combine(path, "classes.txt")) && !Directory.Exists(Path.Combine(path, "Annotations")));

    /// <inheritdoc />
    public AnnotatedDataset Read(string path, AnnotationReadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!Directory.Exists(path))
        {
            throw new DirectoryNotFoundException($"{path}: no YOLO dataset folder there.");
        }

        string images = options?.ImagesRoot ?? (Directory.Exists(Path.Combine(path, "images")) ? Path.Combine(path, "images") : path);
        string labels = Directory.Exists(Path.Combine(path, "labels")) ? Path.Combine(path, "labels") : images;
        var classes = options?.Classes?.ToList() ?? AnnotationFiles.ClassesFile(path) ?? YamlNames(path);
        var result = new List<ImageAnnotations>();
        int largest = -1;
        foreach (string file in AnnotationFiles.Images(images))
        {
            var (width, height) = AnnotationFiles.SizeOf(Path.Combine(images, file));
            string label = Path.Combine(labels, Path.ChangeExtension(file, ".txt"));
            var objects = new List<ObjectAnnotation>();
            if (File.Exists(label))
            {
                int number = 0;
                foreach (string raw in File.ReadLines(label))
                {
                    number++;
                    string[] parts = raw.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 0)
                    {
                        continue;
                    }

                    string where = $"{label}:{number}";
                    if (!int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int c) || c < 0 || classes is not null && c >= classes.Count)
                    {
                        throw new InvalidDataException($"{where}: '{parts[0]}' is not a class{(classes is null ? "" : $" (0 to {classes.Count - 1})")}.");
                    }

                    largest = Math.Max(largest, c);
                    var values = parts.Skip(1).Select((v, i) => AnnotationFiles.Number(v, where, $"value {i + 1}") * (i % 2 == 0 ? width : height)).ToArray();
                    if (values.Length == 4)
                    {
                        objects.Add(new ObjectAnnotation(BoundingBox.FromCenter(values[0], values[1], values[2], values[3]), c));
                    }
                    else if (values.Length >= 6 && values.Length % 2 == 0)
                    {
                        var mask = ObjectMask.FromPolygons([values]);
                        objects.Add(new ObjectAnnotation(mask.Bounds(), c) { Mask = mask });
                    }
                    else
                    {
                        throw new InvalidDataException($"{where}: a line is 'class cx cy w h' or 'class x1 y1 x2 y2 x3 y3 ...', not {parts.Length} values.");
                    }
                }
            }

            result.Add(new ImageAnnotations(file, width, height, objects));
        }

        classes ??= [.. Enumerable.Range(0, largest + 1).Select(i => i.ToString(CultureInfo.InvariantCulture))];
        return new AnnotatedDataset(classes, result) { ImagesRoot = images };
    }

    // data.yaml's names: a flow list [a, b], a block list (- a), or a map (0: a); quotes removed.
    private static List<string>? YamlNames(string folder)
    {
        string? file = new[] { "data.yaml", "data.yml", "dataset.yaml" }.Select(n => Path.Combine(folder, n)).FirstOrDefault(File.Exists);
        if (file is null)
        {
            return null;
        }

        var lines = File.ReadAllLines(file);
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i].Trim();
            if (!line.StartsWith("names:", StringComparison.Ordinal))
            {
                continue;
            }

            string rest = line["names:".Length..].Trim();
            if (rest.StartsWith('['))
            {
                return [.. rest.Trim('[', ']').Split(',').Select(Unquote).Where(n => n.Length > 0)];
            }

            var names = new SortedDictionary<int, string>();
            var listed = new List<string>();
            for (int j = i + 1; j < lines.Length && (lines[j].StartsWith(' ') || lines[j].StartsWith('\t') || lines[j].TrimStart().StartsWith('-')); j++)
            {
                string item = lines[j].Trim();
                if (item.StartsWith('-'))
                {
                    listed.Add(Unquote(item[1..]));
                }
                else if (item.IndexOf(':', StringComparison.Ordinal) is var colon and > 0
                         && int.TryParse(item[..colon], NumberStyles.Integer, CultureInfo.InvariantCulture, out int index))
                {
                    names[index] = Unquote(item[(colon + 1)..]);
                }
            }

            return listed.Count > 0 ? listed : names.Count > 0 ? [.. Enumerable.Range(0, names.Keys.Max() + 1).Select(k => names.GetValueOrDefault(k, k.ToString(CultureInfo.InvariantCulture)))] : null;
        }

        return null;

        static string Unquote(string text) => text.Trim().Trim('"', '\'');
    }

    /// <inheritdoc />
    /// <remarks>
    /// Writes <c>labels/</c> (a .txt per image, its path the image's with .txt) and <c>classes.txt</c> under
    /// <paramref name="path"/>; the images stay where they are (read them back with <see cref="AnnotationReadOptions.ImagesRoot"/>
    /// unless they are in <c>images/</c> there). A polygon mask is written as its first polygon; a run-length mask as its box.
    /// </remarks>
    public void Write(AnnotatedDataset dataset, string path)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string labels = Path.Combine(path, "labels");
        Directory.CreateDirectory(labels);
        AnnotationFiles.WriteClassesFile(path, dataset.Classes);
        foreach (var image in dataset.Images)
        {
            if (image.Width <= 0 || image.Height <= 0)
            {
                throw new ArgumentException($"{image.File}: YOLO normalizes by the image's size, which is not known ({image.Width} x {image.Height}).");
            }

            string file = Path.Combine(labels, Path.ChangeExtension(image.File, ".txt"));
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            var text = new StringBuilder();
            foreach (var o in image.Objects)
            {
                if ((uint)o.Class >= (uint)dataset.Classes.Count)
                {
                    throw new ArgumentException($"{image.File}: class {o.Class} is outside the {dataset.Classes.Count} classes.");
                }

                text.Append(o.Class.ToString(CultureInfo.InvariantCulture));
                if (o.Mask is { IsPolygons: true } mask)
                {
                    var polygon = mask.Polygons![0];
                    for (int i = 0; i < polygon.Length; i++)
                    {
                        text.Append(' ').Append(AnnotationFiles.Text(polygon[i] / (i % 2 == 0 ? image.Width : image.Height)));
                    }
                }
                else
                {
                    text.Append(' ').Append(AnnotationFiles.Text(o.Box.CenterX / image.Width)).Append(' ').Append(AnnotationFiles.Text(o.Box.CenterY / image.Height))
                        .Append(' ').Append(AnnotationFiles.Text(o.Box.Width / image.Width)).Append(' ').Append(AnnotationFiles.Text(o.Box.Height / image.Height));
                }

                text.Append('\n');
            }

            File.WriteAllText(file, text.ToString());
        }
    }
}

/// <summary>
/// Pascal VOC's XML format ("voc" in <see cref="AnnotationFormats"/>): a folder with <c>Annotations/*.xml</c> (or the XML
/// files themselves), each an image's <c>filename</c>, <c>size</c> and <c>object</c>s (name, pose, truncated, difficult,
/// bndbox xmin ymin xmax ymax), read with System.Xml. VOC's boxes count pixels from 1 with both ends inside, so a box is
/// (xmin - 1, ymin - 1) to (xmax, ymax) in the library's pixel coordinates. Classes from <c>classes.txt</c> when there is
/// one, else the names found, in ordinal order. Images are in <c>JPEGImages/</c> (or <c>images/</c>, or the folder itself).
/// </summary>
public sealed class VocFormat : IAnnotationFormat
{
    /// <inheritdoc />
    public string Name => "voc";

    /// <inheritdoc />
    public string Summary => "Pascal VOC XML (a folder with Annotations/*.xml: filename, size, objects with name, pose, truncated, difficult, bndbox)";

    /// <inheritdoc />
    public bool CanRead(string path) => Directory.Exists(Path.Combine(path, "Annotations"))
        || Directory.Exists(path) && Directory.EnumerateFiles(path, "*.xml").Any();

    /// <inheritdoc />
    public AnnotatedDataset Read(string path, AnnotationReadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!Directory.Exists(path))
        {
            throw new DirectoryNotFoundException($"{path}: no Pascal VOC dataset folder there.");
        }

        string annotations = Directory.Exists(Path.Combine(path, "Annotations")) ? Path.Combine(path, "Annotations") : path;
        string images = options?.ImagesRoot ?? new[] { "JPEGImages", "images" }.Select(n => Path.Combine(path, n)).FirstOrDefault(Directory.Exists) ?? path;
        var files = Directory.EnumerateFiles(annotations, "*.xml", SearchOption.AllDirectories).Order(StringComparer.Ordinal).ToList();
        var parsed = new List<(string File, int Width, int Height, List<(string Name, ObjectAnnotation Object)> Objects)>();
        foreach (string file in files)
        {
            XElement root;
            try
            {
                root = XDocument.Load(file).Root ?? throw new InvalidDataException($"{file}: empty.");
            }
            catch (XmlException e)
            {
                throw new InvalidDataException($"{file}: not XML ({e.Message}).", e);
            }

            string image = root.Element("filename")?.Value.Trim() is { Length: > 0 } fileName ? fileName : throw new InvalidDataException($"{file}: no <filename>.");
            int width = (int)Value(root.Element("size")?.Element("width"), file, 0), height = (int)Value(root.Element("size")?.Element("height"), file, 0);
            if (width <= 0 || height <= 0)
            {
                (width, height) = AnnotationFiles.SizeOf(Path.Combine(images, image));
            }

            var objects = new List<(string, ObjectAnnotation)>();
            foreach (var element in root.Elements("object"))
            {
                string name = element.Element("name")?.Value.Trim() is { Length: > 0 } n ? n : throw new InvalidDataException($"{file}: an <object> has no <name>.");
                var box = element.Element("bndbox") ?? throw new InvalidDataException($"{file}: the object '{name}' has no <bndbox>.");
                float xmin = Value(box.Element("xmin"), file), ymin = Value(box.Element("ymin"), file), xmax = Value(box.Element("xmax"), file), ymax = Value(box.Element("ymax"), file);
                objects.Add((name, new ObjectAnnotation(new BoundingBox(xmin - 1, ymin - 1, xmax - xmin + 1, ymax - ymin + 1), 0)
                {
                    Pose = element.Element("pose")?.Value.Trim() is { Length: > 0 } pose && pose != "Unspecified" ? pose : null,
                    Truncated = Flag(element.Element("truncated")),
                    Difficult = Flag(element.Element("difficult")),
                }));
            }

            parsed.Add((image, width, height, objects));
        }

        var classes = options?.Classes?.ToList() ?? AnnotationFiles.ClassesFile(path)
            ?? [.. parsed.SelectMany(p => p.Objects.Select(o => o.Name)).Distinct().Order(StringComparer.Ordinal)];
        var index = classes.Select((c, i) => (c, i)).GroupBy(x => x.c).ToDictionary(g => g.Key, g => g.First().i, StringComparer.Ordinal);
        var result = parsed.Select(p => new ImageAnnotations(p.File, p.Width, p.Height, [.. p.Objects.Select(o => o.Object with
        {
            Class = index.TryGetValue(o.Name, out int c) ? c : throw new InvalidDataException($"{p.File}: the class '{o.Name}' is not among the classes ({string.Join(", ", classes)})."),
        })])).ToList();
        return new AnnotatedDataset(classes, result) { ImagesRoot = images };

        static float Value(XElement? element, string file, float? otherwise = null) => element is null
            ? otherwise ?? throw new InvalidDataException($"{file}: a box needs xmin, ymin, xmax and ymax.")
            : AnnotationFiles.Number(element.Value.Trim(), file, $"<{element.Name}>");

        static bool Flag(XElement? element) => element?.Value.Trim() is { } text && (text == "1" || text.Equals("true", StringComparison.OrdinalIgnoreCase));
    }

    /// <inheritdoc />
    /// <remarks>Writes <c>Annotations/</c> (an XML per image, its path the image's with .xml) and <c>classes.txt</c> under <paramref name="path"/>; the images stay where they are.</remarks>
    public void Write(AnnotatedDataset dataset, string path)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string annotations = Path.Combine(path, "Annotations");
        Directory.CreateDirectory(annotations);
        AnnotationFiles.WriteClassesFile(path, dataset.Classes);
        var settings = new XmlWriterSettings { Indent = true, NewLineChars = "\n", OmitXmlDeclaration = true, Encoding = new UTF8Encoding(false) };
        foreach (var image in dataset.Images)
        {
            string file = Path.Combine(annotations, Path.ChangeExtension(image.File, ".xml"));
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            var root = new XElement("annotation",
                new XElement("filename", image.File.Replace('\\', '/')),
                new XElement("size", new XElement("width", image.Width), new XElement("height", image.Height), new XElement("depth", 3)),
                new XElement("segmented", 0),
                image.Objects.Select(o => new XElement("object",
                    new XElement("name", (uint)o.Class < (uint)dataset.Classes.Count ? dataset.Classes[o.Class]
                        : throw new ArgumentException($"{image.File}: class {o.Class} is outside the {dataset.Classes.Count} classes.")),
                    new XElement("pose", o.Pose ?? "Unspecified"),
                    new XElement("truncated", o.Truncated ? 1 : 0),
                    new XElement("difficult", o.Difficult ? 1 : 0),
                    new XElement("bndbox",
                        new XElement("xmin", AnnotationFiles.Text(o.Box.X + 1)), new XElement("ymin", AnnotationFiles.Text(o.Box.Y + 1)),
                        new XElement("xmax", AnnotationFiles.Text(o.Box.Right)), new XElement("ymax", AnnotationFiles.Text(o.Box.Bottom))))));
            using var writer = XmlWriter.Create(file, settings);
            root.Save(writer);
        }
    }
}
