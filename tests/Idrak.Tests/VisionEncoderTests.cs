// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Data;
using Idrak.Gemma3Vision;
using Idrak.Models;

// Gemma 3's image side (plan 11, phase 3b): the SigLIP encoder and the projector of the tiny reference in
// tests/Idrak.Tests/data/vlm give transformers' embeddings, patch outputs and projected features, from the reference's
// pixel values and end to end from the PNG.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionEncoderGroup =
    [
        ("SigLIP encoder and Gemma 3 projector: the tiny model's patch embeddings, encoder outputs and projected features match transformers in every layout", SiglipMatchesReference),
        ("SigLIP encoder and Gemma 3 projector: image.png through the preprocessor, as a file, bytes and chat images, gives transformers' projected features", SiglipEndToEnd),
    ];

    private static void SiglipMatchesReference(Device device)
    {
        RegisterGemma3Vision();
        float[] pixels = ReadNpyFloat32(TestData("vlm/reference/pixel_values-png.npy"));
        float[] embeddings = ReadNpyFloat32(TestData("vlm/reference/vision_embeddings.npy"));
        float[] hidden = ReadNpyFloat32(TestData("vlm/reference/vision_last_hidden_state.npy"));
        float[] features = ReadNpyFloat32(TestData("vlm/reference/image_features.npy"));
        foreach (string folder in new[] { "tiny-gemma3-pre452", "tiny-gemma3", "tiny-gemma3-v5" })
        {
            using var model = PretrainedModel.Load(TestData($"vlm/{folder}"), new PretrainedOptions { Device = device });
            using var encoder = ((Gemma3Vision)model.Vision!).CreateEncoder(device);
            Check(encoder.Encoder.Config.Patches == 16 && encoder.Projector is { PoolSize: 2, VisionDim: 8, TextDim: 24 } && encoder.WeightsDevice == device,
                $"{folder}: {encoder}");
            using var input = Tensor.From(pixels, [1, 3, 56, 56], device);
            float[] embedded, outputs, projected;
            using (Autograd.NoGrad())
            using (new TensorScope())
            {
                embedded = encoder.Encoder.Embed(input).ToArray();
                outputs = encoder.Encoder.Forward(input).ToArray();
            }

            using (var result = encoder.Encode(input))
            {
                Check(result.Shape.SequenceEqual([1, 4, 24]), $"{folder}: features {Tensor.FormatShape(result.Shape)}");
                projected = result.ToArray();
            }

            Console.WriteLine($"    {folder} on {device}: largest differences {MaxDifference(embeddings, embedded):G3} (embeddings), "
                + $"{MaxDifference(hidden, outputs):G3} (encoder), {MaxDifference(features, projected):G3} (projected)");
            AssertClose(embeddings, embedded, 1e-5f, $"{folder}: patch and position embeddings");
            AssertClose(hidden, outputs, 1e-5f, $"{folder}: encoder outputs after post_layernorm");
            AssertClose(features, projected, 1e-5f, $"{folder}: projected features");

            // One image given as [channels, size, size] is a batch of one.
            using var single = Tensor.From(pixels, [3, 56, 56], device);
            using var again = encoder.Encode(single);
            AssertClose(features, again.ToArray(), 1e-5f, $"{folder}: one image without a batch dimension");
        }
    }

    private static void SiglipEndToEnd(Device device)
    {
        RegisterGemma3Vision();
        float[] features = ReadNpyFloat32(TestData("vlm/reference/image_features.npy"));
        string path = TestData("vlm/image.png");
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3-v5"), new PretrainedOptions { Device = device });
        var vision = (Gemma3Vision)model.Vision!;
        var preprocessor = vision.Preprocessor();
        Check(preprocessor is { Height: 56, Width: 56, Resampling: ImageResampling.Bilinear, Grayscale: false }, "the model folder's preprocessor_config.json");
        using var encoder = vision.CreateEncoder(device);

        using (var fromFile = encoder.Encode(path))
        {
            float worst = MaxDifference(features, fromFile.ToArray());
            Console.WriteLine($"    image.png on {device}: largest difference of the projected features {worst:G3}");
            AssertClose(features, fromFile.ToArray(), 1e-5f, "image.png as a file");
        }

        using (var pixels = preprocessor.Process(path, device))
        using (var fromPixels = encoder.Encode(pixels))
        {
            AssertClose(features, fromPixels.ToArray(), 1e-5f, "image.png's pixel values from the preprocessor");
        }

        // Chat images (encoded bytes), several in one batch: each image's rows are its own features.
        var image = ChatImage.FromFile(path);
        var gray = ChatImage.FromFile(TestData("vlm/image-gray.jpg"));
        using (var batch = encoder.Encode([image, gray, image]))
        {
            Check(batch.Shape.SequenceEqual([3, 4, 24]), $"three images: {Tensor.FormatShape(batch.Shape)}");
            var values = batch.ToArray();
            AssertClose(features, values[..96], 1e-5f, "the first chat image");
            AssertClose(features, values[192..], 1e-5f, "the third chat image");
            Check(MaxDifference(features, values[96..192]) > 1e-3f, "the grey image differs");
            using var grayAlone = encoder.Encode(gray);
            AssertClose(grayAlone.ToArray(), values[96..192], 1e-5f, "an image encodes the same alone and in a batch");
        }

        // A preprocessor of the wrong size is refused, naming both sizes.
        using var wrong = vision.CreateEncoder(device, new ImagePreprocessor { Height = 28, Width = 28 });
        try
        {
            using var _ = wrong.Encode(path);
            Check(false, "pixel values of the wrong size are refused");
        }
        catch (InvalidOperationException ex)
        {
            Check(ex.Message.Contains("56 x 56", StringComparison.Ordinal), ex.Message);
        }
    }

    private static float MaxDifference(float[] expected, float[] actual) =>
        expected.Length != actual.Length ? float.PositiveInfinity : expected.Zip(actual, (a, b) => MathF.Abs(a - b)).Max();
}
