// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Data.Pdf;
using Idrak.Imaging;

// Plan 28, phase D4: PDF images (XObjects and inline images, every colour space and bit depth, /Decode, masks, JPEG
// through the jpeg codec, placement by the CTM) and embedded files, checked on the samples tools/documents/make_fixtures.py
// writes into the ignored test-output/documents/samples (IDRAK_DOCUMENTS_SAMPLES moves them), and on hand-made bytes.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfImagesGroup =
    [
        ("documents pdf: images: decoding every D4 sample image, timed cold and warm apart (time and managed allocation per megapixel), and the scan-like page's image alone (the same pixels on every pass)", PdfImagesTimed),
        ("documents pdf: images: every image of every sample that lists them (D4's colour spaces, bit depths, /Decode arrays, masks, JPEGs, inline images and placements; D1's predictors, D6's scans, D7's and D10's images) in content order with its encoding, size, box and matrix (turned, slanted, flipped, mirrored, on a turned page), its bytes as stored (a raw image's samples, a JPEG's file) and its pixels exactly (raw against the expected hashes, Lab within its tolerance, a JPEG as the jpeg codec decodes the same bytes), its mask as coverage (a soft mask's levels, an explicit mask and a stencil 255 where they paint, a colour key 0 inside its ranges)", PdfImagesSamples),
        ("documents pdf: images: the scan-like pages give their one JPEG over the whole page and their hidden OCR text, read together; every image page read in parallel gives the same images (boxes, matrices, pixels, masks) as read in order", PdfImagesScanAndParallel),
        ("documents pdf: images: embedded files (an EmbeddedFiles name tree with kids and limits, file-attachment annotations) with their names, media types, sizes and descriptions, their bytes read back exactly and again, none in the other samples, refused once the document is closed", PdfImagesAttachments),
        ("documents pdf: images: hand-made files: an encrypted file's XObject and inline image decode; an image drawn by a form takes the form's matrix; JPEG 2000, CCITT and JBIG2 are handed over as stored and named (an issue, a NotSupportedException); the pixel limit is named; a damaged image is left out or refused as damaged with its place kept; a Separation image decodes through its tint transform (D7); a JPEG whose size is not its dictionary's is refused naming both", PdfImagesHandMade),
    ];

    private static readonly float[] PdfImageLevels = [.. Enumerable.Range(0, 256).Select(k => k * (1f / 255f))];

    // The readable samples whose JSON lists images (any phase).
    private static List<PdfSampleEntry> PdfImageSamples()
    {
        var samples = PdfSamples().Where(s => s.Readable && s.Expected["pages"]!.AsArray().Any(p => p!["images"] is JsonArray { Count: > 0 })).ToList();
        Check(samples.Count(s => s.Phases.Contains("D4")) >= 9, $"{samples.Count} samples with images; the generator writes 9 in d4 (make them again: tools/documents/make_fixtures.py)");
        return samples;
    }

    private static void PdfImagesTimed(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples().Where(s => s.Phases.Contains("D4") && s.Readable).ToList();
        var documents = samples.Select(s => PdfOpened(File.ReadAllBytes(s.Path).AsMemory(), Path.GetFileName(s.Path))).ToList();
        try
        {
            var images = documents.SelectMany(d => Enumerable.Range(0, d.PageCount).SelectMany(i => d.ReadPage(i, DocumentParts.Images).Images)).ToList();
            var scanDocument = documents[samples.FindIndex(s => s.Name == "scan-ocr")];
            var scan = scanDocument.ReadPage(0, DocumentParts.Images).Images.Single();
            double megapixels = images.Sum(i => (double)i.PixelWidth * i.PixelHeight) / 1e6, scanMegapixels = (double)scan.PixelWidth * scan.PixelHeight / 1e6;
            Check(images.Count >= 45, $"{images.Count} D4 images");

            // Cold: the scan's JPEG first (the first decode of the process's JPEG path), then every image once.
            long start = Stopwatch.GetTimestamp();
            string scanFirst = PdfImageHash(scan.Decode());
            double scanCold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            start = Stopwatch.GetTimestamp();
            string first = string.Join(",", images.Select(i => PdfImageHash(i.Decode())));
            double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;

            var warm = new List<double>();
            long allocated = 0;
            for (int pass = 0; pass < 20; pass++)
            {
                long before = GC.GetTotalAllocatedBytes(precise: true);
                start = Stopwatch.GetTimestamp();
                var decoded = images.Select(i => i.Decode()).ToList();
                warm.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
                if (pass == 19)
                {
                    Check(string.Join(",", decoded.Select(PdfImageHash)) == first, "the last pass decoded other pixels than the first");
                }
            }

            var scanWarm = new List<double>();
            long scanAllocated = 0;
            for (int pass = 0; pass < 20; pass++)
            {
                long before = GC.GetTotalAllocatedBytes(precise: true);
                start = Stopwatch.GetTimestamp();
                var pixels = scan.Decode();
                scanWarm.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                scanAllocated = GC.GetTotalAllocatedBytes(precise: true) - before;
                if (pass == 19)
                {
                    Check(PdfImageHash(pixels) == scanFirst, "the scan's image decoded to other pixels");
                }
            }

            // The scan-like page read with its image listed (nothing decoded).
            var pageWarm = new List<double>();
            for (int pass = 0; pass < 20; pass++)
            {
                start = Stopwatch.GetTimestamp();
                _ = scanDocument.ReadPage(0, DocumentParts.Text | DocumentParts.Images);
                pageWarm.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            }

            warm.Sort();
            scanWarm.Sort();
            pageWarm.Sort();
            double median = warm[warm.Count / 2], scanMedian = scanWarm[scanWarm.Count / 2];
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {images.Count} D4 images, {megapixels:F3} megapixels: cold {cold:F1} ms, warm median {median:F2} ms a pass ({median / megapixels:F1} ms a megapixel), {allocated / megapixels / 1048576:F2} MiB allocated a megapixel"));
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    the scan-like page's image ({scan.PixelWidth} x {scan.PixelHeight} grey JPEG, {scanMegapixels:F3} megapixels): cold {scanCold:F1} ms, warm median {scanMedian:F2} ms ({scanMedian / scanMegapixels:F1} ms a megapixel), {scanAllocated / scanMegapixels / 1048576:F2} MiB allocated a megapixel; the page read with its text and image listed: warm median {pageWarm[pageWarm.Count / 2]:F2} ms"));
        }
        finally
        {
            documents.ForEach(d => d.Dispose());
        }
    }

    private static void PdfImagesSamples(Device device)
    {
        CpuOnly(device);
        int checkedImages = 0, masks = 0, inline = 0, jpegs = 0;
        foreach (var sample in PdfImageSamples())
        {
            string folder = Path.GetDirectoryName(sample.Path)!;
            using var document = PdfOpened(sample.Path);
            foreach (var page in sample.Expected["pages"]!.AsArray())
            {
                var expected = (page!["images"] as JsonArray)?.Select(i => i!.AsObject()).OrderBy(i => (int)i["order"]!).ToList() ?? [];
                if (expected.Count == 0)
                {
                    continue;
                }

                int index = (int)page["index"]!;
                string what = $"{sample.Name}: page {index}";
                var read = document.ReadPage(index, DocumentParts.Images);
                Check(read.Images.Count == expected.Count, $"{what}: {read.Images.Count} images, expected {expected.Count}");
                var wrong = read.Issues.Where(i => i.Kind is DocumentIssue.ImageEncoding or DocumentIssue.NotSupported or DocumentIssue.DamagedStream).ToList();
                Check(wrong.Count == 0, $"{what}: issues {string.Join("; ", wrong.Select(i => $"{i.Kind}: {i.Detail}"))}");
                for (int k = 0; k < expected.Count; k++)
                {
                    var e = expected[k];
                    PdfCheckImage($"{what}: image {k + 1} ({e["label"]})", read.Images[k], e, page.AsObject(), folder);
                    checkedImages++;
                    masks += e["softMask"] is not null || e["mask"] is not null || e["colourKeyMask"] is not null || (bool)e["imageMask"]! ? 1 : 0;
                    inline += (bool)e["inline"]! ? 1 : 0;
                    jpegs += (string?)e["encoding"] == DocumentImage.Jpeg ? 1 : 0;
                }
            }
        }

        Check(checkedImages >= 70 && masks >= 9 && inline >= 6 && jpegs >= 10, $"{checkedImages} images checked ({masks} with masks, {inline} inline, {jpegs} JPEGs)");
    }

    // One image against its JSON: place, size, encoding, bytes, pixels, mask.
    private static void PdfCheckImage(string what, DocumentImage image, JsonObject e, JsonObject page, string folder)
    {
        Check(image.Order == (int)e["order"]! && image.PixelWidth == (int)e["pixelWidth"]! && image.PixelHeight == (int)e["pixelHeight"]! && image.Encoding == (string?)e["encoding"],
            $"{what}: order {image.Order}, {image.PixelWidth} x {image.PixelHeight}, {image.Encoding}; expected order {e["order"]}, {e["pixelWidth"]} x {e["pixelHeight"]}, {e["encoding"]}");
        var box = e["box"]!.AsArray().Select(v => (float)v!).ToArray();
        Check(MathF.Abs(image.Box.X - box[0]) < 0.01f && MathF.Abs(image.Box.Y - box[1]) < 0.01f && MathF.Abs(image.Box.Width - box[2]) < 0.01f && MathF.Abs(image.Box.Height - box[3]) < 0.01f,
            $"{what}: the box {image.Box}, expected ({string.Join(", ", box)})");

        // The matrix: the unit square from its top-left corner, (u, 1 - v) through the CTM, then onto the page as shown.
        var ctm = e["ctm"]!.AsArray().Select(v => (double)v!).ToArray();
        double[] flipped = [ctm[0], ctm[1], -ctm[2], -ctm[3], ctm[2] + ctm[4], ctm[3] + ctm[5]];
        double[] placed = PdfThen(flipped, PdfDisplay(page));
        var m = image.Matrix ?? default;
        double[] got = [m.A, m.B, m.C, m.D, m.E, m.F];
        Check(image.Matrix is not null && got.Zip(placed).All(p => Math.Abs(p.First - p.Second) < 0.01),
            $"{what}: the matrix ({string.Join(", ", got)}), expected ({string.Join(", ", placed.Select(v => v.ToString("0.###", CultureInfo.InvariantCulture)))})");

        // The bytes as stored: a raw image's samples, a JPEG's file.
        string stored = Convert.ToHexStringLower(SHA256.HashData(image.Bytes().Span));
        Check(stored == (string?)(image.Encoding == DocumentImage.Jpeg ? e["jpegSha256"] : e["samplesSha256"]), $"{what}: its bytes' SHA-256 {stored}");

        var pixels = e["pixels"]!.AsObject();
        var decoded = image.Decode();
        Check(decoded.Channels == (int)pixels["channels"]! && decoded.Width == image.PixelWidth && decoded.Height == image.PixelHeight,
            $"{what}: decoded {decoded.Channels} x {decoded.Width} x {decoded.Height}, expected {pixels["channels"]} channels");
        if (image.Encoding == DocumentImage.Jpeg)
        {
            // A JPEG's pixels are the codec's own decode of the same bytes.
            var codec = new JpegCodec().Decode(image.Bytes().Span);
            Check(PdfInterleaved(decoded).AsSpan().SequenceEqual(PdfInterleaved(codec)), $"{what}: not the jpeg codec's decode of its bytes");
        }

        PdfCheckPixels(what, decoded, pixels, folder, invert: false);
        PdfCheckMask(what, image, e, folder);
    }

    private static void PdfCheckMask(string what, DocumentImage image, JsonObject e, string folder)
    {
        var mask = image.Mask();
        if (e["softMask"] is JsonObject soft)
        {
            Check(mask is not null && mask.Width == (int)soft["pixelWidth"]! && mask.Height == (int)soft["pixelHeight"]!, $"{what}: its soft mask is {mask?.Width} x {mask?.Height}");
            PdfCheckPixels($"{what}: its soft mask", mask!, soft["pixels"]!.AsObject(), folder, invert: false);
        }
        else if (e["mask"] is JsonObject stencil)
        {
            // The explicit mask's pixels as the samples give them (0 where it paints); the mask is its coverage (255 there).
            Check(mask is not null && mask.Width == (int)stencil["pixelWidth"]! && mask.Height == (int)stencil["pixelHeight"]!, $"{what}: its mask is {mask?.Width} x {mask?.Height}");
            PdfCheckPixels($"{what}: its mask", mask!, stencil["pixels"]!.AsObject(), folder, invert: true);
        }
        else if (e["colourKeyMask"] is JsonArray key)
        {
            // 0 where every component's sample lies in its range (8-bit samples here).
            int[] ranges = [.. key.Select(v => (int)v!)];
            int components = (int)e["components"]!;
            var samples = image.Bytes().Span;
            byte[] expected = new byte[image.PixelWidth * image.PixelHeight];
            for (int i = 0; i < expected.Length; i++)
            {
                bool inside = true;
                for (int c = 0; c < components; c++)
                {
                    int v = samples[i * components + c];
                    inside &= v >= ranges[2 * c] && v <= ranges[2 * c + 1];
                }

                expected[i] = inside ? (byte)0 : (byte)255;
            }

            Check(mask is not null && mask.Channels == 1 && PdfInterleaved(mask).AsSpan().SequenceEqual(expected) && expected.Contains((byte)0) && expected.Contains((byte)255),
                $"{what}: its colour key's mask");
        }
        else if ((bool)e["imageMask"]!)
        {
            // A stencil's mask is its coverage: its grey turned over.
            byte[] grey = PdfInterleaved(image.Decode());
            Check(mask is not null && PdfInterleaved(mask).AsSpan().SequenceEqual(grey.Select(b => (byte)(255 - b)).ToArray()), $"{what}: a stencil's mask is not its grey turned over");
        }
        else
        {
            Check(mask is null, $"{what}: a mask where the image has none");
        }
    }

    // Decoded pixels against the JSON: exactly by the SHA-256 of the interleaved samples (16 bits big-endian where the
    // depth is 16), or within the tolerance against the PNG beside the sample; `invert` turns the expected levels over.
    private static void PdfCheckPixels(string what, ImageData image, JsonObject pixels, string folder, bool invert)
    {
        int tolerance = (int?)pixels["tolerance"] ?? 0;
        if ((int)pixels["depth"]! == 16)
        {
            Check(Convert.ToHexStringLower(SHA256.HashData(PdfSixteenBits(image))) == (string?)pixels["sha256"], $"{what}: its 16-bit pixels are not the expected ones");
            return;
        }

        byte[] got = PdfInterleaved(image);
        if (tolerance == 0 && !invert)
        {
            Check(Convert.ToHexStringLower(SHA256.HashData(got)) == (string?)pixels["sha256"], $"{what}: its pixels are not the expected ones (SHA-256)");
            return;
        }

        byte[] expected = PdfInterleaved(new PngCodec().Decode(File.ReadAllBytes(Path.Combine(folder, (string)pixels["file"]!))));
        Check(Convert.ToHexStringLower(SHA256.HashData(expected)) == (string?)pixels["sha256"], $"{what}: {pixels["file"]} is not the expected pixels");
        Check(expected.Length == got.Length, $"{what}: {got.Length} samples, expected {expected.Length}");
        int worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - (invert ? 255 - expected[i] : expected[i])));
        }

        Check(worst <= tolerance, $"{what}: a sample {worst} levels from the expected one, more than its tolerance {tolerance}");
    }

    // An image's samples interleaved, 8 bits (its planes when it holds them, else its values rounded).
    private static byte[] PdfInterleaved(ImageData image)
    {
        int size = image.Width * image.Height, channels = image.Channels;
        byte[] interleaved = new byte[size * channels];
        if (image.TryGetPlanes(out var planes, out _))
        {
            for (int c = 0; c < channels; c++)
            {
                var plane = planes[c].Span;
                for (int i = 0; i < size; i++)
                {
                    interleaved[i * channels + c] = plane[i];
                }
            }

            return interleaved;
        }

        var values = image.Pixels;
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < size; i++)
            {
                interleaved[i * channels + c] = (byte)Math.Clamp((int)MathF.Round(values[c * size + i] * 255f), 0, 255);
            }
        }

        return interleaved;
    }

    // An image's values as 16-bit samples, interleaved, big-endian.
    private static byte[] PdfSixteenBits(ImageData image)
    {
        int size = image.Width * image.Height, channels = image.Channels;
        var values = image.Pixels;
        byte[] bytes = new byte[size * channels * 2];
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < size; i++)
            {
                int level = Math.Clamp((int)MathF.Round(values[c * size + i] * 65535f), 0, 65535);
                bytes[2 * (i * channels + c)] = (byte)(level >> 8);
                bytes[2 * (i * channels + c) + 1] = (byte)level;
            }
        }

        return bytes;
    }

    private static string PdfImageHash(ImageData image) =>
        $"{image.Channels}x{image.Width}x{image.Height}:{Convert.ToHexStringLower(SHA256.HashData(PdfInterleaved(image)))[..16]}";

    // [a b c d e f] then [a b c d e f]: a point goes through the first map, then the second.
    private static double[] PdfThen(double[] m, double[] n) =>
    [
        m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2], m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5],
    ];

    // The map from a page's user space onto the page as shown (the crop box within the media box, turned clockwise, y down).
    private static double[] PdfDisplay(JsonObject page)
    {
        var media = page["mediaBox"]!.AsArray().Select(v => (double)v!).ToArray();
        var crop = (page["cropBox"] as JsonArray)?.Select(v => (double)v!).ToArray() ?? media;
        double left = Math.Max(Math.Min(crop[0], crop[2]), Math.Min(media[0], media[2])), bottom = Math.Max(Math.Min(crop[1], crop[3]), Math.Min(media[1], media[3]));
        double right = Math.Min(Math.Max(crop[0], crop[2]), Math.Max(media[0], media[2])), top = Math.Min(Math.Max(crop[1], crop[3]), Math.Max(media[1], media[3]));
        double w = right - left, h = top - bottom;
        return (int)page["rotation"]! switch
        {
            90 => [0, 1, 1, 0, -bottom, -left],
            180 => [-1, 0, 0, 1, w + left, -bottom],
            270 => [0, -1, -1, 0, h + bottom, w + left],
            _ => [1, 0, 0, -1, -left, h + bottom],
        };
    }

    private static void PdfImagesScanAndParallel(Device device)
    {
        CpuOnly(device);
        var all = PdfImageSamples();

        // The scan-like pages: one JPEG over the whole page (order 0) and the hidden OCR text, read together.
        int scans = 0;
        foreach (var sample in all.Where(s => s.Kind == "scan"))
        {
            using var document = PdfOpened(sample.Path);
            foreach (var expected in sample.Expected["pages"]!.AsArray())
            {
                int index = (int)expected!["index"]!;
                string what = $"{sample.Name}: page {index}";
                var page = document.ReadPage(index, DocumentParts.Text | DocumentParts.Images);
                var image = page.Images.Single();
                Check(image.Encoding == DocumentImage.Jpeg && image.Order == 0 && image.Box.Within(page.Box, 0.01f) && image.Box.Width > page.Width - 0.01f && image.Box.Height > page.Height - 0.01f,
                    $"{what}: the image {image.Encoding} at {image.Box}, order {image.Order}, on a page of {page.Width} x {page.Height}");
                var pixels = image.Decode();
                Check(pixels.Channels == 1 && pixels.Width == image.PixelWidth && PdfInterleaved(pixels).AsSpan().SequenceEqual(PdfInterleaved(new JpegCodec().Decode(image.Bytes().Span))),
                    $"{what}: the scan's pixels are not the jpeg codec's decode");
                Check(page.Runs.Count > 0 && page.Runs.All(r => r.Hidden), $"{what}: {page.Runs.Count} runs, hidden: {string.Join(" ", page.Runs.Select(r => r.Hidden))}");
                if (expected["runs"] is JsonArray { Count: > 0 } && expected["lines"] is JsonArray)
                {
                    PdfCheckTextPage(what, page, expected.AsObject());
                }

                scans++;
            }
        }

        Check(scans >= 3, $"{scans} scan-like pages read");

        // Every image page in order, then all at once from fresh documents.
        var documents = all.Select(s => PdfOpened(s.Path)).ToList();
        try
        {
            var jobs = documents.SelectMany(d => Enumerable.Range(0, d.PageCount).Select(i => (Document: d, Index: i))).ToList();
            var inOrder = jobs.Select(j => PdfImagesSignature(j.Document.ReadPage(j.Index, DocumentParts.Images))).ToList();
            using var fresh = new PdfDocumentList(all.Select(s => PdfOpened(s.Path)));
            var freshJobs = fresh.Items.SelectMany(d => Enumerable.Range(0, d.PageCount).Select(i => (Document: d, Index: i))).ToList();
            var parallel = new string[freshJobs.Count];
            Parallel.For(0, freshJobs.Count, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount) },
                k => parallel[k] = PdfImagesSignature(freshJobs[k].Document.ReadPage(freshJobs[k].Index, DocumentParts.Images)));
            for (int k = 0; k < inOrder.Count; k++)
            {
                Check(parallel[k] == inOrder[k], $"page {freshJobs[k].Index} read in parallel:\n{parallel[k]}\nin order:\n{inOrder[k]}");
            }

            Check(inOrder.Sum(s => s.Split('\n').Count(l => l.StartsWith("image", StringComparison.Ordinal))) >= 70, "fewer than 70 images read both ways");
        }
        finally
        {
            documents.ForEach(d => d.Dispose());
        }
    }

    // A page's images as text: encoding, size, box, matrix, order, the decoded pixels' and the mask's hashes.
    private static string PdfImagesSignature(DocumentPage page)
    {
        var text = new StringBuilder();
        foreach (var image in page.Images)
        {
            string pixels, mask;
            try
            {
                pixels = PdfImageHash(image.Decode());
                mask = image.Mask() is { } m ? PdfImageHash(m) : "none";
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException)
            {
                (pixels, mask) = (e.GetType().Name, "-");
            }

            text.Append(CultureInfo.InvariantCulture, $"image {image.Encoding} {image.PixelWidth} x {image.PixelHeight} {image.Box} {image.Matrix} {image.Order} {pixels} mask {mask}\n");
        }

        foreach (var issue in page.Issues)
        {
            text.Append(CultureInfo.InvariantCulture, $"issue {issue.Kind} {issue.Count} {issue.Detail}\n");
        }

        return text.ToString();
    }

    private static void PdfImagesAttachments(Device device)
    {
        CpuOnly(device);
        int read = 0;
        foreach (var sample in PdfSamples().Where(s => s.Readable))
        {
            // What the sample carries: its JSON's attachments, then its file-attachment annotations' files (D10).
            var expected = (sample.Expected["attachments"] as JsonArray)?.Select(a => a!.AsObject()).ToList() ?? [];
            var names = expected.Select(a => (string)a["fileName"]!).ToHashSet(StringComparer.Ordinal);
            foreach (var annotation in (sample.Expected["annotations"] as JsonArray)?.Select(a => a!.AsObject()) ?? [])
            {
                if (annotation["attachment"] is JsonObject file && names.Add((string)file["fileName"]!))
                {
                    expected.Add(new JsonObject { ["fileName"] = (string)file["fileName"]!, ["size"] = (long)file["size"]!, ["sha256"] = (string)file["sha256"]! });
                }
            }

            using var document = PdfOpened(sample.Path);
            var attachments = document.Attachments;
            Check(attachments.Count == expected.Count, $"{sample.Name}: {attachments.Count} attachments ({string.Join(", ", attachments.Select(a => a.Name))}), expected {expected.Count}");
            Check(ReferenceEquals(attachments, document.Attachments), $"{sample.Name}: the attachments are listed again on each call");
            string folder = Path.GetDirectoryName(sample.Path)!;
            for (int i = 0; i < expected.Count; i++)
            {
                var a = attachments[i];
                var e = expected[i];
                string what = $"{sample.Name}: attachment {i} ({a.Name})";
                Check(a.Name == (string?)e["fileName"] && a.Size == (long)e["size"]!, $"{what}: {a.Name}, {a.Size} bytes; expected {e["fileName"]}, {e["size"]} bytes");
                if (e["mediaType"] is not null)
                {
                    Check(a.MediaType == (string?)e["mediaType"] && a.Description == (string?)e["description"], $"{what}: {a.MediaType}, \"{a.Description}\"; expected {e["mediaType"]}, \"{e["description"]}\"");
                }

                byte[] bytes = PdfReadAll(a);
                Check(Convert.ToHexStringLower(SHA256.HashData(bytes)) == (string?)e["sha256"] && bytes.Length == a.Size, $"{what}: its {bytes.Length} bytes are not the file's");
                if (e["file"] is not null)
                {
                    Check(bytes.AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(folder, (string)e["file"]!))), $"{what}: not the bytes of {e["file"]}");
                }

                Check(PdfReadAll(a).AsSpan().SequenceEqual(bytes), $"{what}: read again, other bytes");
                using (var stream = a.Read())
                {
                    Check(stream.CanRead && stream.CanSeek && !stream.CanWrite && stream.Length == bytes.Length, $"{what}: its stream reads {stream.Length} bytes, seekable {stream.CanSeek}");
                    stream.Seek(-3, SeekOrigin.End);
                    byte[] tail = new byte[8];
                    int n = stream.Read(tail);
                    Check(n == 3 && tail.AsSpan(0, 3).SequenceEqual(bytes.AsSpan(bytes.Length - 3)) && stream.Read(tail) == 0, $"{what}: seeking to its end");
                }

                read++;
            }

            if (attachments.Count > 0)
            {
                var kept = attachments[0];
                document.Dispose();
                try
                {
                    using var late = kept.Read();
                    Check(false, $"{sample.Name}: an attachment read after its document was closed");
                }
                catch (ObjectDisposedException)
                {
                    // Expected: the document's file is closed.
                }
            }
        }

        Check(read >= 5, $"{read} attachments read");
    }

    private static byte[] PdfReadAll(DocumentAttachment attachment)
    {
        using var stream = attachment.Read();
        using var copy = new MemoryStream();
        stream.CopyTo(copy);
        return copy.ToArray();
    }

    private static void PdfImagesHandMade(Device device)
    {
        CpuOnly(device);
        PdfImagesEncrypted();
        PdfImagesForm();
        PdfImagesEncodings();
        PdfImagesLimitsAndDamage();
    }

    // A file encrypted by the standard handler (revision 2, RC4 of 40 bits, empty passwords): its image XObject's stream
    // and its content (holding an inline image) encrypted, both decoded exactly.
    private static void PdfImagesEncrypted()
    {
        byte[] id = [.. Enumerable.Range(0, 16).Select(i => (byte)(i * 17 + 3))];
        byte[] owner = new byte[32], user = new byte[32];
        PdfCrypt.Rc4(MD5.HashData(PdfCrypt.Padding)[..5], PdfCrypt.Padding, owner);
        byte[] key = MD5.HashData([.. PdfCrypt.Padding, .. owner, 0xFC, 0xFF, 0xFF, 0xFF, .. id])[..5];   // /P -4, four bytes low first
        PdfCrypt.Rc4(key, PdfCrypt.Padding, user);
        byte[] Encrypted(int number, byte[] data)
        {
            Span<byte> objectKey = stackalloc byte[16];
            int n = PdfCrypt.ObjectKey(key, number, 0, aes: false, objectKey);
            byte[] output = new byte[data.Length];
            PdfCrypt.Rc4(objectKey[..n], data, output);
            return output;
        }

        byte[] grey = [.. Enumerable.Range(0, 24).Select(i => (byte)(i * 10 + 5))];
        byte[] inlineData = [9, 80, 160, 250, 3, 77, 140, 201];
        byte[] content = [.. Encoding.Latin1.GetBytes("q 60 0 0 40 10 10 cm /Im1 Do Q q 30 0 0 20 100 10 cm BI /W 4 /H 2 /CS /G /BPC 8 ID "), .. inlineData, .. "\nEI Q\n"u8];
        byte[] file = PdfImagesFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>",
            PdfHandMadeStream("", Encrypted(4, content)),
            PdfHandMadeStream("/Type /XObject /Subtype /Image /Width 6 /Height 4 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode", Encrypted(5, PdfDeflate(grey, header: true))),
            $"<< /Filter /Standard /V 1 /R 2 /Length 40 /O <{Convert.ToHexString(owner)}> /U <{Convert.ToHexString(user)}> /P -4 >>",
        ], $"/Encrypt 6 0 R /ID [<{Convert.ToHexString(id)}> <{Convert.ToHexString(id)}>]");
        using var document = PdfOpened(file.AsMemory(), "encrypted-images.pdf");
        Check(document.Info.Encrypted, "the hand-made file is not read as encrypted");
        var page = document.ReadPage(0, DocumentParts.Images);
        Check(page.Images.Count == 2 && page.Issues.Count == 0, $"the encrypted page: {page.Images.Count} images, issues {string.Join("; ", page.Issues.Select(i => i.Detail))}");
        Check(PdfInterleaved(page.Images[0].Decode()).AsSpan().SequenceEqual(grey) && page.Images[0].Bytes().Span.SequenceEqual(grey), "an encrypted image XObject's pixels or bytes");
        Check(PdfInterleaved(page.Images[1].Decode()).AsSpan().SequenceEqual(inlineData) && page.Images[1].Order == 1, "an inline image in an encrypted content stream");
    }

    // An image drawn by a form: the page's cm, the form's /Matrix and the image's cm make its matrix.
    private static void PdfImagesForm()
    {
        byte[] grey = [0, 64, 128, 255];
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R /Resources << /XObject << /Fm1 5 0 R >> >> >>",
            PdfHandMadeStream("", "q 2 0 0 2 0 0 cm /Fm1 Do Q\n"u8.ToArray()),
            PdfHandMadeStream("/Type /XObject /Subtype /Form /BBox [0 0 100 100] /Matrix [1 0 0 1 10 5] /Resources << /XObject << /Im1 6 0 R >> >>", "q 20 0 0 10 0 0 cm /Im1 Do Q\n"u8.ToArray()),
            PdfHandMadeStream("/Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 8", grey),
        ]);
        using var document = PdfOpened(file.AsMemory(), "form-image.pdf");
        var image = document.ReadPage(0, DocumentParts.Images).Images.Single();
        Check(image.Matrix == new DocumentMatrix(40, 0, 0, 20, 20, 70) && image.Box == new DocumentBox(20, 70, 40, 20) && PdfInterleaved(image.Decode()).AsSpan().SequenceEqual(grey),
            $"an image in a form: matrix {image.Matrix}, box {image.Box}");
    }

    // JPEG 2000, CCITT (inline, abbreviated) and JBIG2 (after ASCIIHex): handed over as stored, named in an issue and by
    // Decode (the library has no decoder for them).
    private static void PdfImagesEncodings()
    {
        byte[] fax = [0x00, 0x11, 0x22, 0x33];
        byte[] content = [.. "q 10 0 0 10 0 0 cm /Im1 Do Q q 10 0 0 10 20 0 cm /Im2 Do Q q 8 0 0 2 40 0 cm BI /W 8 /H 2 /BPC 1 /CS /G /F /CCF /DP << /K -1 /Columns 8 >> ID "u8, .. fax, .. "\nEI Q\n"u8];
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R /Im2 6 0 R >> >> >>",
            PdfHandMadeStream("", content),
            PdfHandMadeStream("/Type /XObject /Subtype /Image /Width 10 /Height 10 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /JPXDecode", "jpx-stand-in"u8.ToArray()),
            PdfHandMadeStream("/Type /XObject /Subtype /Image /Width 16 /Height 4 /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter [/ASCIIHexDecode /JBIG2Decode]", "4A42320A>"u8.ToArray()),
        ]);
        using var document = PdfOpened(file.AsMemory(), "encodings.pdf");
        var page = document.ReadPage(0, DocumentParts.Images);
        Check(page.Images.Select(i => i.Encoding).SequenceEqual([DocumentImage.Jpeg2000, DocumentImage.Jbig2, DocumentImage.Ccitt]) && page.Images.Select(i => i.Order).SequenceEqual([0, 1, 2]),
            $"encodings {string.Join(", ", page.Images.Select(i => $"{i.Order} {i.Encoding}"))}");
        Check(page.Images[0].Bytes().Span.SequenceEqual("jpx-stand-in"u8) && page.Images[1].Bytes().Span.SequenceEqual("JB2\n"u8) && page.Images[2].Bytes().Span.SequenceEqual(fax),
            "the encoded images' bytes as stored (after the filters before their encoding)");
        var named = page.Issues.Where(i => i.Kind == DocumentIssue.ImageEncoding).Select(i => i.Detail).ToList();
        Check(named.Count == 3 && new[] { "jpeg2000", "jbig2", "ccitt" }.All(n => named.Any(d => d.Contains(n, StringComparison.Ordinal))), $"the encoding issues: {string.Join("; ", named)}");
        foreach (var image in page.Images)
        {
            try
            {
                image.Decode();
                Check(false, $"a {image.Encoding} image decoded with no codec registered");
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains(image.Encoding, StringComparison.Ordinal), $"\"{e.Message}\" does not name {image.Encoding}");
            }
        }
    }

    // The pixel limit; damaged images (no size: left out, its place kept; an unknown colour space; data cut short); a
    // Separation image through its tint transform; a JPEG whose size is not its dictionary's.
    private static void PdfImagesLimitsAndDamage()
    {
        byte[] jpeg;
        var dct = PdfSamples().Single(s => s.Name == "dct");
        using (var sample = PdfOpened(dct.Path))
        {
            jpeg = sample.ReadPage(0, DocumentParts.Images).Images[0].Bytes().ToArray();   // a 64 x 48 grey JPEG
        }

        string Image(string entries) => $"/Type /XObject /Subtype /Image {entries}";
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R /Im2 6 0 R /Im3 7 0 R /Im4 8 0 R /Im5 9 0 R /Im6 10 0 R >> >> >>",
            PdfHandMadeStream("", "/Im1 Do /Im2 Do /Im3 Do /Im4 Do /Im5 Do /Im6 Do\n"u8.ToArray()),
            PdfHandMadeStream(Image("/Width 400 /Height 400 /ImageMask true /Filter /FlateDecode"), PdfDeflate(new byte[50 * 400], header: true)),
            PdfHandMadeStream(Image("/Height 4 /ColorSpace /DeviceGray /BitsPerComponent 8"), new byte[16]),
            PdfHandMadeStream(Image("/Width 4 /Height 4 /ColorSpace /NoSuchSpace /BitsPerComponent 8"), new byte[16]),
            PdfHandMadeStream(Image("/Width 4 /Height 4 /ColorSpace [/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 0 0] /N 1 >>] /BitsPerComponent 8"), new byte[16]),
            PdfHandMadeStream(Image("/Width 4 /Height 4 /ColorSpace /DeviceGray /BitsPerComponent 8"), [1, 2, 3, 4, 5, 6, 7, 8]),
            PdfHandMadeStream(Image("/Width 32 /Height 48 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode"), jpeg),
        ]);
        using var document = PdfOpened(file.AsMemory(), "damaged-images.pdf", new DocumentOptions { MaxStreamBytes = 50_000 });
        var page = document.ReadPage(0, DocumentParts.Images);
        Check(page.Images.Select(i => i.Order).SequenceEqual([0, 2, 3, 4, 5]), $"the images' orders {string.Join(" ", page.Images.Select(i => i.Order))} (the second has no size: left out, its place kept)");
        string issues = string.Join("; ", page.Issues.Select(i => $"{i.Kind}: {i.Detail}"));
        Check(page.Issues.Any(i => i.Kind == DocumentIssue.DamagedStream && i.Detail.Contains("/Im2", StringComparison.Ordinal) && i.Detail.Contains("left out", StringComparison.Ordinal))
              && page.Issues.Any(i => i.Kind == DocumentIssue.DamagedStream && i.Detail.Contains("/Im3", StringComparison.Ordinal) && i.Detail.Contains("NoSuchSpace", StringComparison.Ordinal))
              && !page.Issues.Any(i => i.Kind == DocumentIssue.NotSupported), $"the issues: {issues}");

        // The stencil's pixels (160,000 bytes) pass the limit its samples (20,000) do not: named when decoded.
        foreach (var read in new Func<ImageData?>[] { page.Images[0].Decode, page.Images[0].Mask })
        {
            try
            {
                read();
                Check(false, "an image past MaxStreamBytes decoded");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains("MaxStreamBytes", StringComparison.Ordinal) && e.Message.Contains("DocumentOptions.MaxStreamBytes", StringComparison.Ordinal) && e.Message.Contains("/Im1", StringComparison.Ordinal),
                    $"\"{e.Message}\" does not name the limit, its option and the image");
            }
        }

        // An unknown colour space is damage; a Separation space decodes through its tint transform (tint 0: no ink, white).
        PdfImagesRefused<InvalidDataException>(page.Images[1], "NoSuchSpace");
        var spot = page.Images[2].Decode();
        Check(spot.Channels == 3 && spot.Width == 4 && spot.Height == 4 && PdfInterleaved(spot).All(v => v == 255), $"a Separation image of no tint: {spot.Channels} channels, {string.Join(" ", PdfInterleaved(spot).Take(6))}");

        // Data cut short: what it holds, then zeros.
        Check(PdfInterleaved(page.Images[3].Decode()).AsSpan().SequenceEqual((byte[])[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]), "an image whose data is cut short");

        // A JPEG of 64 x 48 pixels whose dictionary says 32 x 48.
        PdfImagesRefused<InvalidDataException>(page.Images[4], "64 x 48");
        PdfImagesRefused<InvalidDataException>(page.Images[4], "32 x 48");
    }

    private static void PdfImagesRefused<TException>(DocumentImage image, string named) where TException : Exception
    {
        try
        {
            image.Decode();
            Check(false, $"an image {named} decoded");
        }
        catch (TException e)
        {
            Check(e.GetType() == typeof(TException) || typeof(TException) == typeof(NotSupportedException), $"{e.GetType().Name}: {e.Message}");
            Check(e.Message.Contains(named, StringComparison.Ordinal), $"\"{e.Message}\" does not name {named}");
        }
    }

    // A file of the given objects (numbered from 1), a classic table and a trailer naming object 1 its catalog, with more
    // trailer entries.
    private static byte[] PdfImagesFile(IReadOnlyList<string> objects, string trailer)
    {
        var file = new MemoryStream();
        file.Write("%PDF-1.4\n%âã\n"u8);
        var offsets = new long[objects.Count];
        for (int i = 0; i < objects.Count; i++)
        {
            offsets[i] = file.Position;
            file.Write(Encoding.Latin1.GetBytes($"{i + 1} 0 obj\n{objects[i]}\nendobj\n"));
        }

        long table = file.Position;
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $"xref\n0 {objects.Count + 1}\n0000000000 65535 f\r\n");
        foreach (long offset in offsets)
        {
            text.Append(CultureInfo.InvariantCulture, $"{offset:D10} 00000 n\r\n");
        }

        text.Append(CultureInfo.InvariantCulture, $"trailer\n<< /Size {objects.Count + 1} /Root 1 0 R {trailer} >>\nstartxref\n{table}\n%%EOF\n");
        file.Write(Encoding.Latin1.GetBytes(text.ToString()));
        return file.ToArray();
    }
}
