// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Vision;

// Plan 13, step 5: augmentations that move boxes and masks with the pixels (registered by name, deterministic with a
// seed, run off the training thread by AugmentedImageLoader), and the COCO, YOLO and Pascal VOC dataset formats
// (registered in AnnotationFormats, each round-tripping through the testing kit's suite).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionAugmentationGroup =
    [
        ("vision augment: the Augmentations registry has core's flip and shift and Idrak.Vision's resized-crop, rotation, affine, color-jitter, cutout, mosaic, mixup and resize; pipelines parse from text; unknown names and options are refused", AugmentationRegistry),
        ("vision augment: every library augmentation passes the testing kit's suite (well formed, deterministic, boxes hold their masks, pixel classes and pixels move with the masks)", AugmentationsConformant),
        ("vision augment: flip and shift move boxes, masks and pixel classes exactly with the pixels, drawing what the sample-transform path draws; crops and letterboxes scale boxes; a wrong augmentation fails the kit", AugmentationsAligned),
        ("vision augment: AugmentedImageLoader batches augmented samples off the training thread (workers from the machine's cores, batches ahead from its free memory); the same batches whatever the workers; letterbox, channels, pixel classes, leaving early", AugmentedLoaderBatches),
        ("vision data: COCO, YOLO and VOC are registered in AnnotationFormats and pass the testing kit's round-trip suite; hand-written files read as expected (RLE and polygon masks, crowds, data.yaml names, difficult objects); errors name the file", AnnotationFormatsRoundTrip),
        ("vision data: a COCO dataset with masks loads into augmented batches (decoded, letterboxed, boxes and masks moved)", AnnotatedDatasetIntoLoader),
    ];

    private static void AugmentationRegistry(Device device)
    {
        _ = device;
        string[] expected = ["flip", "shift", "resized-crop", "rotation", "affine", "color-jitter", "cutout", "mosaic", "mixup", "resize"];
        Check(expected.All(Augmentations.Names.Contains) && expected.All(n => Augmentations.Origin(n) == Overrides.Library), $"augmentations: {string.Join(", ", Augmentations.Names)}");
        var pipeline = Augmentations.Parse("flip(p=1, vertical=true), rotation(degrees=15), resized-crop(width=32, height=24, scale=0.5:1), color-jitter(brightness=0.3, hue=0.1), mosaic");
        Check(pipeline.Select(a => a.Name).SequenceEqual(["flip", "rotation", "resized-crop", "color-jitter", "mosaic"]), $"parsed: {string.Join(", ", pipeline.Select(a => a.Name))}");
        Check(pipeline[0] is RandomFlip { Vertical: true, Probability: 1 } && pipeline[1] is RandomAffine { Degrees: (-15, 15) } && pipeline[2] is RandomResizedCrop { Width: 32, Scale: (0.5, 1) },
            "parsed options");
        Check(Augmentations.Parse(" none ").Count == 0 && Augmentations.Parse(null).Count == 0, "none");
        foreach (string bad in new[] { "blur", "flip(sigma=2)", "rotation(degrees=a)", "resized-crop(scale=1:0.5)", "flip(p=1", "resize" })
        {
            try
            {
                Augmentations.Parse(bad);
                throw new Exception($"'{bad}' parsed");
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Length > 0, bad);
            }
        }

        try
        {
            Augmentations.Parse("blur");
        }
        catch (ArgumentException e)
        {
            Check(e.Message.Contains("registered:", StringComparison.Ordinal) && e.Message.Contains("mosaic", StringComparison.Ordinal), $"unknown name: {e.Message}");
        }

        // An app's augmentation shadows the library's and falls back to it with the same random state.
        var sample = AugmentationSuite.Synthetic(new Random(1), 20, 20, 3, 2, masks: false, pixelClasses: false);
        var libraryResult = Augmentations.Create("flip", [new("p", "1")]).Apply(sample, new AugmentationContext(new Random(3)));
        Augmentations.Register("flip", _ => throw new InvalidOperationException("no flips here"));
        try
        {
            Augmentations.SetPolicy("flip", SlotPolicy.FallBack);
            var fallen = Augmentations.Create("flip", [new("p", "1")]).Apply(sample, new AugmentationContext(new Random(3)));
            Check(fallen.Boxes.SequenceEqual(libraryResult.Boxes), "FallBack flips as the library does");
        }
        finally
        {
            Augmentations.SetPolicy("flip", SlotPolicy.Throw);
            Augmentations.Unregister("flip");
        }

        Check(Augmentations.Origin("flip") == Overrides.Library, "Unregister brings the library's flip back");
    }

    private static void AugmentationsConformant(Device device)
    {
        _ = device;
        var options = new ContractCheckOptions { RandomCases = 12 };
        foreach (string text in new[]
                 {
                     "flip(p=1, vertical=true)", "flip", "shift(pixels=6)", "resized-crop(width=40, height=30)", "resized-crop(scale=0.3:0.6, min_visibility=0.3)",
                     "rotation(degrees=30)", "affine(degrees=20, translate=0.2, scale=0.7:1.3, shear=10)", "mosaic(width=48, height=40)", "mosaic",
                     "resize(width=50, height=20)", "resize(width=37, height=41, keep_ratio=true)",
                 })
        {
            var augmentation = Augmentations.Parse(text).Single();
            var report = Conformance.Check(augmentation, new AugmentationSuite(), options);
            Check(report.Passed, $"{text}:\n{report}");
        }

        // Colour changes, erasing and mixing change pixels inside the objects (and mixing keeps one sample's pixel classes).
        foreach (string text in new[] { "color-jitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)", "cutout(p=1, count=3)" })
        {
            Conformance.Check(Augmentations.Parse(text).Single(), new AugmentationSuite { ColorsKept = false }, options).ThrowIfFailed();
        }

        Conformance.Check(Augmentations.Parse("mixup").Single(), new AugmentationSuite { ColorsKept = false, PixelClassesFollowMasks = false }, options).ThrowIfFailed();
    }

    private sealed class FlipsPixelsOnly : IAugmentation
    {
        public string Name => "broken-flip";

        public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
        {
            var flipped = new RandomFlip(probability: 1).Apply(sample, context);
            return new AnnotatedImage(flipped.Image, [.. sample.Boxes], [.. sample.Labels], sample.Masks?.ToArray(), sample.PixelClasses);
        }
    }

    private static void AugmentationsAligned(Device device)
    {
        _ = device;
        var sample = AugmentationSuite.Synthetic(new Random(11), 30, 20, 3, 3, masks: true, pixelClasses: true);
        int w = sample.Width, h = sample.Height;

        // Flip: the boxes mirror; masks and pixel classes mirror as the pixels do.
        var flipped = new RandomFlip(horizontal: true, vertical: true, probability: 1).Apply(sample, new AugmentationContext(new Random(1)));
        for (int i = 0; i < sample.Count; i++)
        {
            var b = sample.Boxes[i];
            Check(flipped.Boxes[i] == new BoundingBox(w - b.Right, h - b.Bottom, b.Width, b.Height), $"flipped box {flipped.Boxes[i]} of {b}");
            Check(ObjectMask.PixelBounds(flipped.Masks![i], w, h) == flipped.Boxes[i], $"flipped mask bounds of object {i}");
        }

        Check(flipped.PixelClasses![0] == sample.PixelClasses![w * h - 1] && flipped.Image.Pixels[0] == sample.Image.Pixels[w * h - 1], "pixel classes and pixels flip together");

        // The augmentation draws what the sample transform draws: the same seed flips the same way.
        foreach (int seed in Enumerable.Range(0, 8))
        {
            var flip = new RandomFlip(horizontal: true, vertical: true);
            var features = (float[])sample.Image.Pixels.Clone();
            flip.Apply(features, Span<float>.Empty, [3, h, w], new Random(seed));
            var augmented = flip.Apply(sample, new AugmentationContext(new Random(seed)));
            Check(augmented.Image.Pixels.SequenceEqual(features), $"seed {seed}: the two paths of RandomFlip agree");
            var shift = new RandomShift(4);
            var shifted = (float[])sample.Image.Pixels.Clone();
            shift.Apply(shifted, Span<float>.Empty, [3, h, w], new Random(seed));
            var moved = shift.Apply(sample, new AugmentationContext(new Random(seed)));
            Check(moved.Image.Pixels.SequenceEqual(shifted), $"seed {seed}: the two paths of RandomShift agree");
            for (int i = 0; i < moved.Count; i++)
            {
                Check(ObjectMask.PixelBounds(moved.Masks![i], w, h) == moved.Boxes[i], $"seed {seed}: shifted mask and box of object {i}");
            }
        }

        // A letterbox scales and centres boxes; a stretch scales them per axis.
        var box = new BoundingBox(3, 2, 6, 4);
        var single = new AnnotatedImage(new ImageData(new float[3 * 20 * 40], 3, 20, 40), [box], [1]);
        var letterboxed = new SampleResize(80, 80, keepRatio: true).Apply(single, new AugmentationContext(new Random(0)));
        Check(letterboxed.Boxes[0] == new BoundingBox(6, 4 + 20, 12, 8), $"letterboxed {letterboxed.Boxes[0]}");
        var stretched = new SampleResize(20, 40).Apply(single, new AugmentationContext(new Random(0)));
        Check(stretched.Boxes[0] == new BoundingBox(1.5f, 4, 3, 8), $"stretched {stretched.Boxes[0]}");

        // An augmentation that moves the pixels but not the boxes and masks fails the kit's suite.
        var report = Conformance.Check<IAugmentation>(new FlipsPixelsOnly(), new AugmentationSuite());
        Check(!report.Passed && report.Failures.Any(f => f.Check.Contains("pixels move with their masks", StringComparison.Ordinal)), $"a flip of the pixels alone passes:\n{report}");
    }

    private static void AugmentedLoaderBatches(Device device)
    {
        int testThread = Environment.CurrentManagedThreadId, readsElsewhere = 0;
        AnnotatedImage Read(int i)
        {
            if (Environment.CurrentManagedThreadId != testThread)
            {
                Interlocked.Increment(ref readsElsewhere);
            }

            return AugmentationSuite.Synthetic(new Random(100 + i), 40 + i % 3 * 7, 30, i % 4 == 0 ? 1 : 3, 3, masks: true, pixelClasses: true);
        }

        var augmentations = Augmentations.Parse("flip, affine(degrees=10, scale=0.8:1.2), mosaic(p=0.5), color-jitter(brightness=0.2)");
        List<(float[] Images, BoundingBox[][] Boxes, int[][] Labels, float[] Classes)> Epoch(int workers)
        {
            var loader = new AugmentedImageLoader(10, Read, 32, 24, batchSize: 4, shuffle: true, seed: 7, device: device) { Augmentations = augmentations, Workers = workers };
            var batches = new List<(float[], BoundingBox[][], int[][], float[])>();
            foreach (var batch in loader)
            {
                using (batch)
                {
                    Check(batch.Images.Shape.SequenceEqual([batch.Count, 3, 24, 32]) && batch.PixelClasses!.Shape.SequenceEqual([batch.Count, 24, 32]),
                        $"batch {batch.Index}: {Tensor.FormatShape(batch.Images.Shape)}");
                    Check(Enumerable.Range(0, batch.Count).All(i => batch.Masks[i] is { } m && m.Count == batch.Boxes[i].Count && batch.Labels[i].Count == m.Count
                                                                 && batch.Boxes[i].All(b => b.X >= 0 && b.Right <= 32.001f && b.Bottom <= 24.001f)),
                        $"batch {batch.Index}: masks and boxes inside the batch's size");
                    using var corners = batch.Corners(0);
                    Check(corners.Shape.SequenceEqual([batch.Boxes[0].Count, 4]), "corners");
                    batches.Add((batch.Images.ToArray(), [.. batch.Boxes.Select(b => b.ToArray())], [.. batch.Labels.Select(l => l.ToArray())],
                        batch.PixelClasses!.ToArray()));
                }
            }

            return batches;
        }

        var one = Epoch(1);
        var three = Epoch(3);
        Check(one.Count == 3 && one.Select(b => b.Boxes.Length).SequenceEqual([4, 4, 2]), $"{one.Count} batches");
        for (int b = 0; b < one.Count; b++)
        {
            Check(one[b].Images.SequenceEqual(three[b].Images) && one[b].Classes.SequenceEqual(three[b].Classes)
                  && one[b].Boxes.Zip(three[b].Boxes).All(p => p.First.SequenceEqual(p.Second)) && one[b].Labels.Zip(three[b].Labels).All(p => p.First.SequenceEqual(p.Second)),
                $"batch {b}: one worker and three give the same batch");
        }

        Check(readsElsewhere > 0, "samples are read and augmented off the training thread");
        var defaults = new AugmentedImageLoader(10, Read, 32, 24, batchSize: 4);
        Check(defaults.Workers == Math.Max(1, ComputeResources.MaxCpuThreads - 1) && defaults.Prefetch >= 1 && defaults.Prefetch <= defaults.Workers,
            $"measured: {defaults.Workers} workers, {defaults.Prefetch} batches ahead");

        // Leaving an epoch early stops the workers; the next epoch differs (new random numbers).
        var loader = new AugmentedImageLoader(10, Read, 32, 24, batchSize: 2, seed: 3, device: device) { Augmentations = augmentations };
        float[] first;
        using (var e = loader.GetEnumerator())
        {
            Check(e.MoveNext(), "a first batch");
            first = e.Current.Images.ToArray();
            e.Current.Dispose();
        }

        using (var e = loader.GetEnumerator())
        {
            Check(e.MoveNext() && !e.Current.Images.ToArray().SequenceEqual(first), "the second epoch augments differently");
            e.Current.Dispose();
        }

        Check(loader.Prepare(0, epoch: 0).Image.Pixels.SequenceEqual(first[..(3 * 24 * 32)]), "Prepare(index, epoch) is the batch's sample");
    }

    // A P6 image of `width` x `height` (values from the seed).
    private static void WritePpm(string path, int width, int height, int seed)
    {
        var header = Encoding.ASCII.GetBytes($"P6\n{width} {height}\n255\n");
        var bytes = new byte[header.Length + width * height * 3];
        header.CopyTo(bytes, 0);
        new Random(seed).NextBytes(bytes.AsSpan(header.Length));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, bytes);
    }

    private static void AnnotationFormatsRoundTrip(Device device)
    {
        _ = device;
        Check(new[] { "coco", "yolo", "voc" }.All(AnnotationFormats.Names.Contains), $"formats: {string.Join(", ", AnnotationFormats.Names)}");
        var options = new ContractCheckOptions { RandomCases = 6 };
        foreach (var (name, traits) in new[]
                 {
                     ("coco", new AnnotationFormatTraits { FileName = "instances.json", Crowd = true, Area = true, Ids = true, Polygons = true, RunLength = true }),
                     ("yolo", new AnnotationFormatTraits { Polygons = true }),
                     ("voc", new AnnotationFormatTraits { Difficult = true, Truncated = true, Pose = true }),
                 })
        {
            var format = AnnotationFormats.Get(name);
            Conformance.Check(new AnnotationFormatUnderTest(path => format.Read(path), format.Write, traits), new AnnotationFormatSuite(), options).ThrowIfFailed();
        }

        string root = Path.Combine(Path.GetTempPath(), "idrak-formats-" + Guid.NewGuid().ToString("N"));
        try
        {
            // COCO by hand: category ids 3 and 7, a polygon, a crowd with uncompressed runs, a compressed mask.
            string coco = Path.Combine(root, "coco", "annotations.json");
            Directory.CreateDirectory(Path.GetDirectoryName(coco)!);
            string compressed = ObjectMask.EncodeCounts([3, 2, 5, 2, 8]);
            File.WriteAllText(coco, $$$"""
                {"images": [{"id": 10, "file_name": "a.ppm", "width": 4, "height": 5}, {"id": 11, "file_name": "b.ppm", "width": 8, "height": 6}],
                 "categories": [{"id": 3, "name": "cat"}, {"id": 7, "name": "dog"}],
                 "annotations": [
                  {"id": 1, "image_id": 10, "category_id": 7, "bbox": [0.5, 1, 2, 3], "area": 4.5, "iscrowd": 0, "segmentation": [[0.5, 1, 2.5, 1, 2.5, 4]]},
                  {"id": 2, "image_id": 10, "category_id": 3, "bbox": [1, 0, 2, 4], "area": 4, "iscrowd": 1, "segmentation": {"size": [5, 4], "counts": [3, 2, 5, 2, 8]}},
                  {"id": 5, "image_id": 11, "category_id": 3, "bbox": [0, 0, 8, 6], "area": 48, "iscrowd": 0, "segmentation": {"size": [5, 4], "counts": "{{{compressed}}}"}}]}
                """);
            Check(AnnotationFormats.Find(coco)?.Name == "coco", "a COCO file is found as COCO");
            var d = AnnotationFormats.Read(coco);
            Check(d.Classes.SequenceEqual(["cat", "dog"]) && d.ClassIds!.SequenceEqual([3L, 7L]) && d.Images.Count == 2 && d.Images[0].Id == 10, "COCO classes and images");
            var o = d.Images[0].Objects;
            Check(o[0] is { Class: 1, Area: 4.5f, Crowd: false, Id: 1 } && o[0].Box == new BoundingBox(0.5f, 1, 2, 3) && o[0].Mask!.IsPolygons, "COCO polygon object");
            Check(o[1] is { Class: 0, Crowd: true } && o[1].Mask!.Counts!.SequenceEqual([3, 2, 5, 2, 8]) && d.Images[1].Objects[0].Mask!.Counts!.SequenceEqual([3, 2, 5, 2, 8]),
                "COCO crowd with plain runs; compressed runs");
            var sample = d.Images[0].ToSample(new ImageData(new float[3 * 20], 3, 5, 4), includeIgnored: true);
            Check(sample.Count == 2 && sample.Masks![1].Count(v => v != 0) == 4 && d.Images[0].ToSample(new ImageData(new float[3 * 20], 3, 5, 4)).Count == 1,
                "a sample's masks; the crowd left out of training");
            File.WriteAllText(Path.Combine(root, "coco", "bad.json"), """{"images": [{"id": 1, "file_name": "a.ppm"}], "categories": [], "annotations": [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [0, 0, 1, 1]}]}""");
            ExpectInvalid(() => AnnotationFormats.Read(Path.Combine(root, "coco", "bad.json"), "coco"), "bad.json", "category 4");

            // YOLO by hand: data.yaml names as a map, a box line and a polygon line, an image without a label file.
            string yolo = Path.Combine(root, "yolo");
            WritePpm(Path.Combine(yolo, "images", "train", "x.ppm"), 20, 10, 1);
            WritePpm(Path.Combine(yolo, "images", "train", "empty.ppm"), 8, 8, 2);
            Directory.CreateDirectory(Path.Combine(yolo, "labels", "train"));
            File.WriteAllText(Path.Combine(yolo, "labels", "train", "x.txt"), "1 0.5 0.5 0.5 0.4\n0 0.1 0.2 0.3 0.2 0.3 0.6\n");
            File.WriteAllText(Path.Combine(yolo, "data.yaml"), "path: .\ntrain: images/train\nnames:\n  0: person\n  1: 'bicycle'\n");
            Check(AnnotationFormats.Find(yolo)?.Name == "yolo", "a YOLO folder is found as YOLO");
            var y = AnnotationFormats.Read(yolo, "yolo");
            Check(y.Classes.SequenceEqual(["person", "bicycle"]) && y.Images.Select(i => i.File).SequenceEqual(["train/empty.ppm", "train/x.ppm"]) && y.Images[0].Objects.Count == 0,
                $"YOLO images {string.Join(", ", y.Images.Select(i => i.File))}");
            var x = y.Images[1];
            Check(x is { Width: 20, Height: 10 } && x.Objects[0].Class == 1 && Near(x.Objects[0].Box, new BoundingBox(5, 3, 10, 4)) && x.Objects[1].Mask!.IsPolygons
                  && Near(x.Objects[1].Box, BoundingBox.FromCorners(2, 2, 6, 6)), $"YOLO objects {x.Objects[0].Box}, {x.Objects[1].Box}");
            File.WriteAllText(Path.Combine(yolo, "labels", "train", "x.txt"), "1 0.5 0.5\n");
            ExpectInvalid(() => AnnotationFormats.Read(yolo, "yolo"), "x.txt:1", "class cx cy w h");

            // VOC by hand: 1-based pixels, a difficult object, sizes in the XML.
            string voc = Path.Combine(root, "voc");
            Directory.CreateDirectory(Path.Combine(voc, "Annotations"));
            File.WriteAllText(Path.Combine(voc, "Annotations", "000001.xml"), """
                <annotation><folder>VOC2007</folder><filename>000001.jpg</filename><size><width>353</width><height>500</height><depth>3</depth></size>
                <object><name>dog</name><pose>Left</pose><truncated>1</truncated><difficult>0</difficult><bndbox><xmin>48</xmin><ymin>240</ymin><xmax>195</xmax><ymax>371</ymax></bndbox></object>
                <object><name>person</name><pose>Unspecified</pose><truncated>0</truncated><difficult>1</difficult><bndbox><xmin>8</xmin><ymin>12</ymin><xmax>352</xmax><ymax>498</ymax></bndbox></object>
                </annotation>
                """);
            Check(AnnotationFormats.Find(voc)?.Name == "voc", "a VOC folder is found as VOC");
            var v = AnnotationFormats.Read(voc);
            var dog = v.Images[0].Objects[0];
            Check(v.Classes.SequenceEqual(["dog", "person"]) && v.Images[0] is { File: "000001.jpg", Width: 353, Height: 500 }
                  && dog.Box == new BoundingBox(47, 239, 148, 132) && dog is { Pose: "Left", Truncated: true, Difficult: false } && v.Images[0].Objects[1] is { Difficult: true, Pose: null },
                $"VOC objects {dog.Box}");
            File.WriteAllText(Path.Combine(voc, "Annotations", "000002.xml"), "<annotation><filename>2.jpg</filename><size><width>5</width><height>5</height></size><object><name>cat</name></object></annotation>");
            ExpectInvalid(() => AnnotationFormats.Read(voc, "voc"), "000002.xml", "<bndbox>");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }

        static bool Near(BoundingBox a, BoundingBox b) =>
            MathF.Abs(a.X - b.X) < 1e-4f && MathF.Abs(a.Y - b.Y) < 1e-4f && MathF.Abs(a.Width - b.Width) < 1e-4f && MathF.Abs(a.Height - b.Height) < 1e-4f;

        static void ExpectInvalid(Action read, string file, string text)
        {
            try
            {
                read();
                throw new Exception($"{file} read without an error");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains(file, StringComparison.Ordinal) && e.Message.Contains(text, StringComparison.Ordinal), $"the error names {file}: {e.Message}");
            }
        }
    }

    private static void AnnotatedDatasetIntoLoader(Device device)
    {
        string root = Path.Combine(Path.GetTempPath(), "idrak-dataset-" + Guid.NewGuid().ToString("N"));
        try
        {
            var images = new List<ImageAnnotations>();
            for (int i = 0; i < 5; i++)
            {
                int w = 30 + 4 * i, h = 20 + 2 * i;
                WritePpm(Path.Combine(root, "images", $"{i}.ppm"), w, h, i);
                var box = new BoundingBox(2 + i, 3, 10, 8);
                images.Add(new ImageAnnotations($"{i}.ppm", w, h,
                    [new ObjectAnnotation(box, i % 2) { Mask = ObjectMask.FromPolygons([[box.X, box.Y, box.Right, box.Y, box.Right, box.Bottom, box.X, box.Bottom]]) }]));
            }

            AnnotationFormats.Write(new AnnotatedDataset(["a", "b"], images), Path.Combine(root, "instances.json"), "coco");
            var dataset = AnnotationFormats.Read(Path.Combine(root, "instances.json"), options: new AnnotationReadOptions { ImagesRoot = Path.Combine(root, "images") });
            var loader = AugmentedImageLoader.FromDataset(dataset, 48, 48, batchSize: 2, seed: 1, device: device, masks: true);
            int seen = 0;
            foreach (var batch in loader)
            {
                using (batch)
                {
                    for (int i = 0; i < batch.Count; i++)
                    {
                        Check(batch.Boxes[i].Count == 1 && batch.Masks[i] is { Count: 1 } && batch.Images.Shape[3] == 48, "one object with its mask, letterboxed");
                        var bounds = ObjectMask.PixelBounds(batch.Masks[i]![0], 48, 48);
                        var box = batch.Boxes[i][0];
                        Check(MathF.Abs(bounds.X - box.X) <= 1 && MathF.Abs(bounds.Right - box.Right) <= 1, $"mask {bounds}, box {box}");
                        seen++;
                    }
                }
            }

            Check(seen == 5, $"{seen} samples");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
