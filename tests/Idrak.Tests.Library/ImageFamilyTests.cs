// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Data;
using Idrak.Imaging;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Onnx;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

// Plan 13, step 6: image model families as plug-ins. What runs without a family or a decoder picked by name: an ONNX file
// naming no architecture refused, the segmenter and the predictors' checks, the conformance kit's family suite.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ImageFamilyGroup =
    [
        ("image families: an ONNX file naming no architecture is refused", ImageFamiliesRefuse),
        ("image families: ModelSegmenter with a preprocessor (a crop refused); predictors refuse the wrong task", ImageFamiliesPredictors),
        ("conformance kit: the image model family suite passes a consistent family and fails one whose outputs change between loads, one whose images leak into each other in a batch, and one that loads an invalid checkpoint", ImageFamilySuiteCatches),
    ];

    internal static void ExpectRefusal<TException>(Action action, string text, string what) where TException : Exception
    {
        try
        {
            action();
        }
        catch (TException e)
        {
            Check(e.Message.Contains(text, StringComparison.Ordinal), $"{what}: \"{e.Message}\" does not say \"{text}\"");
            return;
        }

        Check(false, $"{what}: no {typeof(TException).Name}");
    }

    internal static string TempFolder(string what)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-{what}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    internal static void ImageFamiliesRefuse(Device device)
    {
        string folder = TempFolder("image-refuse");
        try
        {
            using (var net = new Sequential(new Flatten(), new Linear(12, 2, device: Device.Cpu, random: new Random(1))))
            {
                OnnxExport.For(net).Input(3, 2, 2).Save(Path.Combine(folder, "plain.onnx"));
            }

            ExpectRefusal<InvalidDataException>(() => ImageModels.Load(Path.Combine(folder, "plain.onnx")), "names no architecture", "an ONNX file naming none");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A network that answers every image with the same values (a [N, ...] tensor of them).
    internal sealed class ConstantOutputs(float[] values, int[] shape) : Module
    {
        protected override Tensor ForwardCore(Tensor input) =>
            Tensor.From([.. Enumerable.Range(0, input.Shape[0]).SelectMany(_ => values)], [input.Shape[0], .. shape], input.Device);
    }

    internal static void ImageFamiliesPredictors(Device device)
    {
        // A network whose one box is (0, 0)-(32, 32) in its 32 x 32 input, score 0.9.
        using var boxes = new ConstantOutputs([0, 0, 32, 32, 0.9f], [1, 5]);
        var crop = new ImagePreprocessor { ShortestEdge = 40, CenterCrop = true, CropHeight = 32, CropWidth = 32, Normalize = false };
        var resize = new ImagePreprocessor { Height = 32, Width = 32, Normalize = false };
        ImageData Blank(int width, int height) => new(new float[3 * width * height], 3, height, width);

        // A segmenter: class 1 on the left half of a 4 x 4 map, resized back to the image.
        var logits = new float[2 * 16];
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 2; x++)
            {
                logits[16 + y * 4 + x] = 1f;
            }
        }

        using var map = new ConstantOutputs(logits, [2, 4, 4]);
        var mask = new ModelSegmenter(map, resize, device).Segment(Blank(8, 6));
        Check(mask.Width == 8 && mask.Height == 6 && mask.LabelAt(0, 0) == 1 && mask.LabelAt(3, 5) == 1 && mask.LabelAt(4, 0) == 0, "the mask, back to the image");
        ExpectRefusal<NotSupportedException>(() => _ = new ModelSegmenter(map, crop), "cannot center-crop", "a segmenter's crop");

        // The predictors check the task, the labels and that every image becomes one size.
        var classifier = new ImageModel { Architecture = "t", Task = ImageTask.Classification, Network = boxes, Preprocessor = resize };
        ExpectRefusal<InvalidOperationException>(() => classifier.Detector(), "not a detection one", "a classifier as a detector");
        ExpectRefusal<InvalidOperationException>(() => classifier.Classifier(), "names no classes", "a classifier without labels");
        ExpectRefusal<NotSupportedException>(() => (classifier with { Preprocessor = new ImagePreprocessor { ShortestEdge = 20 } }).Outputs(), "different sizes", "sizes that follow the image");
        ExpectRefusal<InvalidOperationException>(() => RegionClassifier.For(classifier), "with labels", "a region classifier without labels");
        Check(classifier.Classifier(["a", "b", "c", "d", "e"]) is not null && (classifier with { Task = ImageTask.Segmentation }).Segmenter() is not null, "labels given; a segmenter");
    }

    internal static void ImageFamilySuiteCatches(Device device)
    {
        var image = new ImageData([.. Enumerable.Range(0, 3 * 5 * 7).Select(i => i % 11 / 10f)], 3, 5, 7);
        var samples = new[] { new ImageFamilySample("good", [image, new ImageData(new float[16], 1, 4, 4)]) };
        static float[] Mean(ImageData image) => [image.Pixels.Average(), image.Pixels.Max()];
        ImageModelUnderTest Wrap(Func<IReadOnlyList<ImageData>, IReadOnlyList<float[]>> predict) => new("features", null, predict);
        var options = new ContractCheckOptions { RandomCases = 3 };

        var good = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(Mean)]), samples, options);
        Check(good.Passed, $"a consistent family passes: {good}");

        int loads = 0;
        var changing = Conformance.CheckImageModelFamily(_ =>
        {
            float offset = Interlocked.Increment(ref loads);
            return Wrap(images => [.. images.Select(i => Mean(i).Select(v => v + offset).ToArray())]);
        }, samples, options);
        Check(changing.Failures.Any(f => f.Check == "the same outputs after loading again"), $"outputs that change between loads: {changing}");

        var leaking = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(i => Mean(i).Select(v => v + images.Count).ToArray())]), samples, options);
        Check(leaking.Failures.Any(f => f.Check == "each image's outputs alone as in the batch"), $"images leaking into each other: {leaking}");

        var lenient = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(Mean)]), [new ImageFamilySample("broken", Error: "not registered")], options);
        Check(lenient.Failures.Any(f => f.Check == "refuses an invalid checkpoint"), $"an invalid checkpoint loaded: {lenient}");

        var reference = Conformance.CheckImageModelFamily(_ => Wrap(images => [.. images.Select(Mean)]),
            [new ImageFamilySample("good", [image], [[0f, 0f]])], options with { RandomCases = 0 });
        Check(reference.Failures.Any(f => f.Check == "the reference's outputs"), $"outputs that are not the reference's: {reference}");
    }
}
