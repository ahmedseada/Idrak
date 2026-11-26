// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Imaging;
using Idrak.Vision;

// Plan 13, step 5: augmentations that move boxes and masks with the pixels (deterministic with a seed, run off the
// training thread by AugmentedImageLoader), and the COCO, YOLO and Pascal VOC dataset formats (each round-tripping through
// the testing kit's suite).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] VisionAugmentationGroup =
    [
        ("vision augment: every library augmentation passes the testing kit's suite (well formed, deterministic, boxes hold their masks, pixel classes and pixels move with the masks)", AugmentationsConformant),
        ("vision augment: flip and shift move boxes, masks and pixel classes exactly with the pixels, drawing what the sample-transform path draws; crops and letterboxes scale boxes; a wrong augmentation fails the kit", AugmentationsAligned),
        ("vision augment: AugmentedImageLoader batches augmented samples off the training thread (workers from the machine's cores, batches ahead from its free memory); the same batches whatever the workers; letterbox, channels, pixel classes, leaving early", AugmentedLoaderBatches),
    ];

    internal static void AugmentationsConformant(Device device)
    {
        _ = device;
        var options = new ContractCheckOptions { RandomCases = 12 };
        foreach (IAugmentation augmentation in new IAugmentation[]
                 {
                     new RandomFlip(vertical: true, probability: 1), new RandomFlip(), new RandomShift(6), new RandomResizedCrop(40, 30), new RandomResizedCrop(scale: (0.3, 0.6), minVisibility: 0.3),
                     new RandomAffine((-30, 30)), new RandomAffine((-20, 20), (0.2, 0.2), (0.7, 1.3), (-10, 10)), new Mosaic(48, 40), new Mosaic(),
                     new SampleResize(50, 20), new SampleResize(37, 41, keepRatio: true),
                 })
        {
            var report = Conformance.Check(augmentation, new AugmentationSuite(), options);
            Check(report.Passed, $"{augmentation.Name}:\n{report}");
        }

        // Colour changes, erasing and mixing change pixels inside the objects (and mixing keeps one sample's pixel classes).
        foreach (IAugmentation augmentation in new IAugmentation[] { new ColorJitter(0.4, 0.4, 0.4, 0.2), new Cutout(probability: 1, count: 3) })
        {
            Conformance.Check(augmentation, new AugmentationSuite { ColorsKept = false }, options).ThrowIfFailed();
        }

        Conformance.Check(new MixUp(), new AugmentationSuite { ColorsKept = false, PixelClassesFollowMasks = false }, options).ThrowIfFailed();
    }

    internal sealed class FlipsPixelsOnly : IAugmentation
    {
        public string Name => "broken-flip";

        public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
        {
            var flipped = new RandomFlip(probability: 1).Apply(sample, context);
            return new AnnotatedImage(flipped.Image, [.. sample.Boxes], [.. sample.Labels], sample.Masks?.ToArray(), sample.PixelClasses);
        }
    }

    internal static void AugmentationsAligned(Device device)
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

    internal static void AugmentedLoaderBatches(Device device)
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

        IAugmentation[] augmentations = [new RandomFlip(), new RandomAffine((-10, 10), scale: (0.8, 1.2)), new Mosaic(probability: 0.5), new ColorJitter(brightness: 0.2)];
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
    internal static void WritePpm(string path, int width, int height, int seed)
    {
        var header = Encoding.ASCII.GetBytes($"P6\n{width} {height}\n255\n");
        var bytes = new byte[header.Length + width * height * 3];
        header.CopyTo(bytes, 0);
        new Random(seed).NextBytes(bytes.AsSpan(header.Length));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, bytes);
    }
}
