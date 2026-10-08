// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Data.Abstractions;

// The JPEG codec against Pillow's (libjpeg-turbo's) pixels, and the image preprocessing of vision-language models against
// transformers' Gemma3ImageProcessorPil; the files come from tools/jpeg/make_fixtures.py.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ImageDecodingGroup =
    [
        ("jpeg: baseline, progressive, grey, RGB, CMYK, YCCK, restart intervals and every sampling decode to Pillow's RGB exactly", JpegMatchesPillow),
        ("jpeg: 12-bit, arithmetic, lossless and hierarchical files are refused by name; damaged and cut files", JpegRefusals),
        ("image preprocessing: every colour format of the codecs to RGB, Pillow's resize (bilinear, bicubic, Lanczos; up and down), rescale and normalize as transformers, to 1e-5", PreprocessingMatchesTransformers),
        ("image preprocessing: preprocessor_config.json keys, pan and scan refused, grey to three channels, tensors", PreprocessingConfig),
    ];

    private static string JpegFixtures => Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "jpeg");

    private static void JpegMatchesPillow(Device device)
    {
        Check(ImageCodecs.Names.SequenceEqual(["png", "jpeg", "bmp", "netpbm"]) && ImageCodecs.Origin("jpeg") == Overrides.Library,
            $"the built-in codecs: {string.Join(", ", ImageCodecs.Names)}");
        Check(ImageCodecs.CanDecode("scan.JPG") && ImageCodecs.CanDecode("a.jpeg"), "JPEG extensions");
        var files = Directory.GetFiles(JpegFixtures, "*.jpg").Order().ToList();
        Check(files.Count == 21, $"{files.Count} JPEG fixtures");
        foreach (string file in files)
        {
            string name = Path.GetFileNameWithoutExtension(file);
            string reference = Path.ChangeExtension(file, File.Exists(Path.ChangeExtension(file, ".pgm")) ? ".pgm" : ".ppm");
            var expected = ImageCodecs.Decode(reference);
            var info = ImageCodecs.ReadInfo(file);
            Check(info == new ImageInfo(expected.Width, expected.Height, expected.Channels, "jpeg"), $"{name}: header {info}");
            var actual = ImageCodecs.Decode(file);
            Check(actual.Channels == expected.Channels && actual.Height == expected.Height && actual.Width == expected.Width,
                $"{name}: {actual.Channels} x {actual.Height} x {actual.Width}, Pillow {expected.Channels} x {expected.Height} x {expected.Width}");
            int worst = 0;
            for (int i = 0; i < actual.Pixels.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs((int)MathF.Round(actual.Pixels[i] * 255) - (int)MathF.Round(expected.Pixels[i] * 255)));
            }

            Check(worst == 0, $"{name}: a pixel differs from Pillow's by {worst}");
        }
    }

    private static void JpegRefusals(Device device)
    {
        var jpeg = ImageCodecs.Get("jpeg");
        byte[] baseline = File.ReadAllBytes(Path.Combine(JpegFixtures, "baseline-420.jpg"));
        int sof = baseline.AsSpan().IndexOf([(byte)0xFF, (byte)0xC0]);
        byte[] With(int at, byte value)
        {
            var copy = (byte[])baseline.Clone();
            copy[at] = value;
            return copy;
        }

        void Refused(byte[] file, string words, string what)
        {
            try
            {
                jpeg.Decode(file);
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains(words, StringComparison.Ordinal), $"{what}: {e.Message}");
                return;
            }

            throw new InvalidOperationException($"{what}: decoded");
        }

        Refused(With(sof + 4, 12), "12-bit", "12-bit samples");
        Refused(With(sof + 1, 0xC9), "arithmetic", "arithmetic coding");
        Refused(With(sof + 1, 0xC3), "lossless", "a lossless frame");
        Refused(With(sof + 1, 0xC5), "hierarchical", "a hierarchical frame");

        string twelve = Path.Combine(Path.GetTempPath(), $"idrak-jpeg-{Guid.NewGuid():N}.jpg");
        File.WriteAllBytes(twelve, With(sof + 4, 12));
        try
        {
            ImageCodecs.Decode(twelve);
            Check(false, "the registry decoded a 12-bit file");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.StartsWith(twelve, StringComparison.Ordinal), $"the file is named: {e.Message}");
        }
        finally
        {
            File.Delete(twelve);
        }

        // Extended sequential (SOF1) at 8 bits is baseline with more tables allowed: it decodes the same.
        var extended = jpeg.Decode(With(sof + 1, 0xC1));
        Check(extended.Pixels.SequenceEqual(jpeg.Decode(baseline).Pixels), "SOF1 at 8 bits");

        // A file cut inside its image data decodes (the missing data as zeros, as libjpeg does); one cut before it does not.
        var cut = jpeg.Decode(baseline.AsSpan(0, baseline.Length * 3 / 4));
        Check(cut.Width == 45 && cut.Height == 37, "a cut file keeps its size");
        Throws<InvalidDataException>(() => jpeg.Decode(baseline.AsSpan(0, sof + 4)), "a file cut in its frame header");
        Throws<InvalidDataException>(() => jpeg.Decode([0xFF, 0xD8, 0xFF, 0xD9]), "no image data");
        Check(jpeg.ReadInfo("\x89PNG"u8) is null && jpeg.ReadInfo(baseline) == new ImageInfo(45, 37, 3, "jpeg"), "sniffing");

        // A frame header after the first 64 KiB (long APPn segments, as EXIF thumbnails and ICC profiles make): the
        // registry reads the whole file for its size.
        string padded = Path.Combine(Path.GetTempPath(), $"idrak-jpeg-{Guid.NewGuid():N}.jpg");
        try
        {
            var app = new byte[60_000];
            using (var stream = File.Create(padded))
            {
                stream.Write(baseline.AsSpan(0, 2));
                for (int i = 0; i < 2; i++)
                {
                    stream.Write([0xFF, 0xE2, (byte)((app.Length + 2) >> 8), (byte)(app.Length + 2)]);
                    stream.Write(app);
                }

                stream.Write(baseline.AsSpan(2));
            }

            Check(ImageCodecs.ReadInfo(padded) == new ImageInfo(45, 37, 3, "jpeg"), "a frame header past 64 KiB");
            Check(ImageCodecs.Decode(padded).Pixels.SequenceEqual(jpeg.Decode(baseline).Pixels), "the same pixels after long segments");
        }
        finally
        {
            File.Delete(padded);
        }
    }

    private static void PreprocessingMatchesTransformers(Device device)
    {
        var cases = JsonNode.Parse(File.ReadAllText(Path.Combine(JpegFixtures, "preprocess.json")))!.AsArray();
        Check(cases.Count == 54, $"{cases.Count} cases");
        var failures = new List<string>();
        foreach (var item in cases)
        {
            string name = (string)item!["name"]!;
            var config = item["config"]!;
            var processor = ImagePreprocessor.Parse(config.ToJsonString(), grayscale: (bool)item["grayscale"]!);
            int h = (int)config["size"]!["height"]!, w = (int)config["size"]!["width"]!;
            Check(processor.Height == h && processor.Width == w && (int)processor.Resampling == (int)config["resample"]! && processor.Resize == (bool)config["do_resize"]!, $"{name}: the config");
            var actual = processor.Pixels(Path.Combine(JpegFixtures, (string)item["image"]!));
            var bytes = File.ReadAllBytes(Path.Combine(JpegFixtures, name + ".f32"));
            var expected = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, expected, 0, bytes.Length);
            Check((!processor.Resize || actual.Length == 3 * h * w) && expected.Length == actual.Length, $"{name}: {actual.Length} values, expected {expected.Length}");
            float worst = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                worst = MathF.Max(worst, MathF.Abs(actual[i] - expected[i]));
            }

            if (worst > 1e-5f)
            {
                failures.Add($"{name} by {worst}");
            }
        }

        Check(failures.Count == 0, $"differ from transformers: {string.Join("; ", failures)}");
    }

    private static void PreprocessingConfig(Device device)
    {
        string gemma = """
            {"do_convert_rgb": null, "do_normalize": true, "do_pan_and_scan": null, "do_rescale": true, "do_resize": true,
             "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Gemma3ImageProcessor", "image_seq_length": 256,
             "image_std": [0.5, 0.5, 0.5], "pan_and_scan_max_num_crops": null, "processor_class": "Gemma3Processor",
             "resample": 2, "rescale_factor": 0.00392156862745098, "size": {"height": 896, "width": 896}}
            """;
        var processor = ImagePreprocessor.Parse(gemma);
        Check(processor is { Resize: true, Height: 896, Width: 896, Resampling: ImageResampling.Bilinear, Rescale: true, Normalize: true, ConvertRgb: true, Grayscale: false }
              && Math.Abs(processor.RescaleFactor - 1 / 255.0) < 1e-15 && processor.Mean.SequenceEqual([0.5f, 0.5f, 0.5f]), "Gemma 3's config");

        void Refused(string json, string words)
        {
            try
            {
                ImagePreprocessor.Parse(json);
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains(words, StringComparison.Ordinal), e.Message);
                return;
            }

            throw new InvalidOperationException($"accepted: {json}");
        }

        Refused(gemma.Replace("\"do_pan_and_scan\": null", "\"do_pan_and_scan\": true", StringComparison.Ordinal), "do_pan_and_scan");
        Refused("""{"size": {"shortest_edge": 224}}""", "shortest");
        Refused("""{"size": {"height": 8, "width": 8}, "resample": 0}""", "resample 0");
        Refused("""{"size": {"height": 8, "width": 8}, "do_center_crop": true}""", "do_center_crop");

        // A model folder holding the file; grey through the steps; a tensor.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-preprocess-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            File.WriteAllText(Path.Combine(folder, "preprocessor_config.json"),
                """{"size": {"height": 4, "width": 6}, "resample": 3, "image_mean": 0.25, "image_std": [1, 2, 4]}""");
            var small = ImagePreprocessor.FromConfig(folder);
            Check(small is { Height: 4, Width: 6, Resampling: ImageResampling.Bicubic } && small.Mean.SequenceEqual([0.25f]) && small.Std.SequenceEqual([1f, 2f, 4f]), "a folder's config");

            var grey = new ImageData([.. Enumerable.Range(0, 12).Select(i => i * 20 / 255f)], 1, 3, 4);
            var values = small.Pixels(grey);
            Check(values.Length == 3 * 4 * 6, "a grey image becomes three channels");
            for (int i = 0; i < 24; i++)
            {
                Check(Math.Abs(values[i] - values[24 + i] * 2) < 1e-6 && Math.Abs(values[i] - values[48 + i] * 4) < 1e-6, "the same grey in every channel, each normalized by its std");
            }

            var kept = new ImagePreprocessor { Resize = false, ConvertRgb = false, Normalize = false }.Pixels(grey);
            Check(kept.Length == 12 && Math.Abs(kept[11] - 220 / 255f) < 1e-6, "no resize, rescale only, grey kept grey");

            // Grayscale: Pillow's L of the 8-bit colours, in three channels.
            var colour = new ImageData([10 / 255f, 200 / 255f, 30 / 255f, 255 / 255f, 7 / 255f, 128 / 255f], 3, 1, 2);
            var flat = new ImagePreprocessor { Resize = false, Rescale = false, Normalize = false, Grayscale = true }.Pixels(colour);
            int L(int r, int g, int b) => (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
            Check(flat.SequenceEqual([L(10, 30, 7), L(200, 255, 128), L(10, 30, 7), L(200, 255, 128), L(10, 30, 7), L(200, 255, 128)]), $"grayscale: {string.Join(", ", flat)}");

            using var tensor = small.Process(colour, device);
            Check(tensor.Shape.SequenceEqual([3, 4, 6]), "a [3, height, width] tensor");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
