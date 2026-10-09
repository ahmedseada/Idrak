// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Onnx;

// Plan 13, step 2 (the rest): the layers image backbones and decoders use beyond the recognizer's: transposed convolution,
// upsampling (nearest, bilinear), adaptive pooling, group norm, ceil-mode pooling and padding below and right. Checked
// against PyTorch (tests/Idrak.Tests/data/vision-layers, written by tools/pytorch/vision_layers_reference.py), onnx's
// reference evaluator, ONNX Runtime and finite differences.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionLayersGroup =
    [
        ("vision layers: ConvTranspose2d (rectangular, strided, padded, output padding, dilation, groups; a square decoder step) matches PyTorch's output and gradients (input, weight, bias)", TransposedConvMatchesPyTorch),
        ("vision layers: gradient: ConvTranspose2d against finite differences (input and weights; grouped, dilated, without bias)", TransposedConvGradients),
        ("vision layers: Upsample nearest and bilinear (corners aligned or not; scale factors 2, 1.5, (1.7, 2.3); sizes up and down) matches PyTorch's output and input gradient", UpsampleMatchesPyTorch),
        ("vision layers: gradient: nearest and bilinear interpolation against finite differences", UpsampleGradients),
        ("vision layers: AdaptiveAvgPool2d and AdaptiveMaxPool2d (uneven, overlapping windows) match PyTorch with gradients; divisible sizes give ordinary pooling's results; gradient: finite differences", AdaptivePoolsMatchPyTorch),
        ("vision layers: GroupNorm (affine over [N, C, H, W], plain over [N, C, L]) matches PyTorch's output and gradients; gradient: finite differences", GroupNormMatchesPyTorch),
        ("vision layers: ceil-mode MaxPool2d and AvgPool2d (padding counted and not) match PyTorch with gradients; padding below and right; gradient: finite differences", CeilPoolsMatchPyTorch),
        ("vision layers: builder steps replay from JSON, layer types describe and create, weights save and load; pooling without the new options is described as before", VisionLayersDescribedAgain),
        ("vision layers: ONNX: a decoder exports (ONNX Runtime matches) and imports again; PyTorch's decoder.onnx and windows.onnx (padding below and right, GroupNormalization) import and match", VisionLayersOnnxImport),
    ];

    private static readonly Lazy<JsonObject> LayersReference = new(() => JsonNode.Parse(File.ReadAllText(
        Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vision-layers", "reference.json")))!.AsObject());

    private static JsonObject LayerCase(string name) => LayersReference.Value[name]!.AsObject();

    // Runs `module` on the case's input, checks the output, then the gradients of sum(output · upstream) on the input and on
    // the named parameters against PyTorch's.
    private static void CheckAgainstPyTorch(JsonObject c, Module module, Device device, string what, float tolerance = 1e-4f, params (string Name, Tensor Parameter)[] parameters)
    {
        foreach (var (_, p) in parameters)
        {
            p.ZeroGrad();
        }

        var x = FromCase(c["input"]!, device, requiresGrad: true);
        var y = module.Forward(x);
        Check(y.Shape.SequenceEqual(Shape(c["output"]!)), $"{what}: shape {Tensor.FormatShape(y.Shape)}, PyTorch {Tensor.FormatShape(Shape(c["output"]!))}");
        AssertClose(Floats(c["output"]!), y.ToArray(), tolerance, what);
        (y * FromCase(c["upstream"]!, device)).Sum().Backward();
        AssertClose(Floats(c["grad_input"]!), x.Grad!.ToArray(), tolerance, $"{what}: gradient of the input");
        foreach (var (name, p) in parameters)
        {
            AssertClose(Floats(c["grad_" + name]!), p.Grad!.ToArray(), tolerance, $"{what}: gradient of the {name}");
        }
    }

    private static void TransposedConvMatchesPyTorch(Device device)
    {
        using var scope = new TensorScope();
        foreach (string name in new[] { "convt_rect", "convt_square" })
        {
            var c = LayerCase(name);
            using var conv = ConvTranspose2d.FromWeights(FromCase(c["weight"]!, device, requiresGrad: true), FromCase(c["bias"]!, device, requiresGrad: true),
                PairOf(c["kernel"]!), PairOf(c["stride"]!), PairOf(c["padding"]!), PairOf(c["output_padding"]!), PairOf(c["dilation"]!), (int)c["groups"]!);
            CheckAgainstPyTorch(c, conv, device, name, 1e-4f, ("weight", conv.Weight), ("bias", conv.Bias!));
        }

        // The square constructor is the pair constructor with equal pairs; the text names what differs from the defaults.
        using var square = new ConvTranspose2d(2, 3, 4, stride: 2, padding: 1, device: device, random: new Random(5));
        using var pairs = new ConvTranspose2d(2, 3, (4, 4), (2, 2), (1, 1), device: device, random: new Random(5));
        var x = Tensor.From(RandomArray(new Random(6), 2 * 2 * 3 * 5), [2, 2, 3, 5], device);
        AssertClose(square.Forward(x).ToArray(), pairs.Forward(x).ToArray(), 0f, "square and pair constructors");
        Check(square.Forward(x).Shape.SequenceEqual([2, 3, 6, 10]) && square.ToString() == "ConvTranspose2d(2 -> 3, 4x4, stride 2, padding 1)", square.ToString());
        Check(new ConvTranspose2d(4, 6, (3, 2), (2, 1), (1, 0), (1, 0), (1, 2), groups: 2).ToString()
              == "ConvTranspose2d(4 -> 6, 3x2, stride 2x1, padding 1x0, output padding 1x0, dilation 1x2, groups 2)", "ToString");
        Expect<ArgumentOutOfRangeException>(() => new ConvTranspose2d(2, 2, 3, stride: 2, outputPadding: 2), "an output padding not below the stride");
        Expect<ArgumentException>(() => new ConvTranspose2d(4, 6, (3, 3), groups: 4), "groups that do not divide the outputs");
        Expect<ArgumentException>(() => ConvTranspose2d.FromWeights(Tensor.Zeros([2, 3, 3, 3], device), Tensor.Zeros([2], device), (3, 3)), "a bias of the wrong size");
    }

    private static void TransposedConvGradients(Device device)
    {
        var grouped = new ConvTranspose2d(4, 6, (3, 2), (2, 1), (1, 0), (1, 0), (1, 2), groups: 2, device: device, random: new Random(91));
        GradCheck(device, [2, 4, 3, 4], x => grouped.Forward(x).Tanh().Sum());
        ParameterGradCheck(grouped, Tensor.From(RandomArray(new Random(92), 2 * 4 * 3 * 4), [2, 4, 3, 4], device), y => y.Tanh().Sum());
        var plain = new ConvTranspose2d(3, 2, 4, stride: 2, padding: 1, outputPadding: 1, bias: false, device: device, random: new Random(93));
        GradCheck(device, [1, 3, 3, 2], x => plain.Forward(x).Tanh().Sum());
        ParameterGradCheck(plain, Tensor.From(RandomArray(new Random(94), 3 * 3 * 2), [1, 3, 3, 2], device), y => y.Tanh().Sum());
    }

    // The case's layer: a scale factor (one number or a pair) or a size, nearest or bilinear, corners aligned or not.
    private static Upsample UpsampleFromCase(JsonObject c)
    {
        var mode = (string)c["mode"]! == "bilinear" ? InterpolationMode.Bilinear : InterpolationMode.Nearest;
        bool align = c["align_corners"] is { } a && (bool)a;
        return c["size"] is JsonArray size ? Upsample.ToSize(((int)size[0]!, (int)size[1]!), mode, align)
            : c["scale_factor"] is JsonArray pair ? new Upsample(((float)pair[0]!, (float)pair[1]!), mode, align)
            : new Upsample((float)c["scale_factor"]!, mode, align);
    }

    private static void UpsampleMatchesPyTorch(Device device)
    {
        using var scope = new TensorScope();
        foreach (string name in new[] { "interp_nearest_2", "interp_nearest_1_5", "interp_nearest_size", "interp_nearest_down", "interp_bilinear_2", "interp_bilinear_pair",
            "interp_bilinear_aligned", "interp_bilinear_down" })
        {
            var c = LayerCase(name);
            CheckAgainstPyTorch(c, UpsampleFromCase(c), device, name, 1e-5f);
        }

        // Tensor.Interpolate is the operation underneath: the layer's output for a size, any rank with [..., H, W].
        var x = Tensor.From(RandomArray(new Random(95), 2 * 3 * 4), [2, 3, 4], device);
        AssertClose(Upsample.ToSize((5, 7), InterpolationMode.Bilinear).Forward(x.Reshape(1, 2, 3, 4)).ToArray(),
            x.Interpolate((5, 7), InterpolationMode.Bilinear).ToArray(), 0f, "Interpolate on [C, H, W]");
        Check(new Upsample(2).ToString() == "Upsample(x2, nearest)" && Upsample.ToSize((8, 6), InterpolationMode.Bilinear, true).ToString() == "Upsample(to 8x6, bilinear, corners aligned)",
            new Upsample(2).ToString());
        Expect<ArgumentException>(() => new Upsample(2, InterpolationMode.Nearest, alignCorners: true), "aligned corners with nearest");
        Expect<ArgumentOutOfRangeException>(() => new Upsample(0f), "a zero scale factor");
    }

    private static void UpsampleGradients(Device device)
    {
        using var weights = Tensor.From(RandomArray(new Random(96), 2 * 2 * 7 * 9), [2, 2, 7, 9], device);
        foreach (var layer in new[] { Upsample.ToSize((7, 9), InterpolationMode.Bilinear), Upsample.ToSize((7, 9), InterpolationMode.Bilinear, true), Upsample.ToSize((7, 9)) })
        {
            GradCheck(device, [2, 2, 4, 5], x => (layer.Forward(x) * weights).Sum());
        }

        using var down = Tensor.From(RandomArray(new Random(97), 3 * 2), [1, 1, 3, 2], device);
        GradCheck(device, [1, 1, 7, 5], x => (Upsample.ToSize((3, 2), InterpolationMode.Bilinear).Forward(x) * down).Sum());
    }

    private static void AdaptivePoolsMatchPyTorch(Device device)
    {
        using var scope = new TensorScope();
        foreach (string name in new[] { "adaptive_avg", "adaptive_avg_small", "adaptive_max", "adaptive_max_small" })
        {
            var c = LayerCase(name);
            var size = PairOf(c["size"]!);
            Module pool = name.StartsWith("adaptive_avg", StringComparison.Ordinal) ? new AdaptiveAvgPool2d(size) : new AdaptiveMaxPool2d(size);
            CheckAgainstPyTorch(c, pool, device, name, 1e-5f);
        }

        // When the output divides the input the windows are ordinary ones: the same results as k = s pooling.
        var x = Tensor.From(RandomArray(new Random(98), 2 * 3 * 8 * 6), [2, 3, 8, 6], device);
        AssertClose(new AvgPool2d((4, 2)).Forward(x).ToArray(), new AdaptiveAvgPool2d((2, 3)).Forward(x).ToArray(), 1e-6f, "adaptive average pooling with even windows");
        AssertClose(new MaxPool2d((4, 2)).Forward(x).ToArray(), new AdaptiveMaxPool2d((2, 3)).Forward(x).ToArray(), 0f, "adaptive max pooling with even windows");
        AssertClose(new GlobalAveragePool2d().Forward(x).ToArray(), new AdaptiveAvgPool2d(1).Forward(x).ToArray(), 1e-6f, "adaptive average pooling to one position");
        using var weights = Tensor.From(RandomArray(new Random(99), 2 * 2 * 3 * 4), [2, 2, 3, 4], device);
        GradCheck(device, [2, 2, 7, 5], x => (new AdaptiveAvgPool2d((3, 4)).Forward(x) * weights).Sum());
        Check(new AdaptiveAvgPool2d((3, 4)).ToString() == "AdaptiveAvgPool2d(3x4)" && new AdaptiveMaxPool2d(1).ToString() == "AdaptiveMaxPool2d(1x1)", "ToString");
        Expect<ArgumentOutOfRangeException>(() => new AdaptiveMaxPool2d((0, 2)), "an empty output");
    }

    private static void GroupNormMatchesPyTorch(Device device)
    {
        using var scope = new TensorScope();
        var c = LayerCase("groupnorm");
        using var norm = new GroupNorm((int)c["groups"]!, (int)c["channels"]!, (float)c["eps"]!, device: device);
        norm.Gamma!.Load(Floats(c["weight"]!));
        norm.Beta!.Load(Floats(c["bias"]!));
        CheckAgainstPyTorch(c, norm, device, "group norm", 1e-4f, ("weight", norm.Gamma), ("bias", norm.Beta));
        norm.Eval();
        AssertClose(Floats(c["output"]!), norm.Forward(FromCase(c["input"]!, device)).ToArray(), 1e-4f, "group norm in evaluation (no running statistics)");

        var p = LayerCase("groupnorm_plain");
        using var plain = new GroupNorm((int)p["groups"]!, (int)p["channels"]!, (float)p["eps"]!, affine: false, device: device);
        Check(!plain.Parameters().Any() && plain.ToString() == "GroupNorm(2, 4, no affine)", plain.ToString());
        CheckAgainstPyTorch(p, plain, device, "group norm without affine over [N, C, L]", 1e-4f);

        var gn = new GroupNorm(2, 4, device: device);
        gn.Gamma!.Load(RandomArray(new Random(100), 4, 1f));
        gn.Beta!.Load(RandomArray(new Random(101), 4, 1f));
        using var weights = Tensor.From(RandomArray(new Random(102), 2 * 4 * 3 * 3), [2, 4, 3, 3], device);
        GradCheck(device, [2, 4, 3, 3], x => (gn.Forward(x) * weights).Sum(), scale: 2f);
        ParameterGradCheck(gn, Tensor.From(RandomArray(new Random(103), 2 * 4 * 3 * 3, 2f), [2, 4, 3, 3], device), y => (y * weights).Sum());
        Expect<ArgumentException>(() => new GroupNorm(3, 4), "groups that do not divide the channels");
    }

    private static void CeilPoolsMatchPyTorch(Device device)
    {
        using var scope = new TensorScope();
        var m = LayerCase("maxpool_ceil");
        CheckAgainstPyTorch(m, new MaxPool2d(PairOf(m["kernel"]!), PairOf(m["stride"]!), PairOf(m["padding"]!), ceilMode: true), device, "ceil-mode max pooling", 0f);
        foreach (string name in new[] { "avgpool_ceil", "avgpool_ceil_nopad" })
        {
            var a = LayerCase(name);
            CheckAgainstPyTorch(a, new AvgPool2d(PairOf(a["kernel"]!), PairOf(a["stride"]!), PairOf(a["padding"]!), (bool)a["count_include_pad"]!, ceilMode: true),
                device, name, 1e-5f);
        }

        // Padding below and right only: the windows a symmetric padding of the same total would give, shifted up and left.
        var x = Tensor.From(RandomArray(new Random(104), 2 * 7 * 6), [1, 2, 7, 6], device);
        var uneven = new MaxPool2d((2, 2), (2, 2), (0, 0), paddingEnd: (1, 0));
        var y = uneven.Forward(x);
        Check(y.Shape.SequenceEqual([1, 2, 4, 3]) && uneven.ToString() == "MaxPool2d(2x2, stride 2x2, padding 0x0 (1x0 below and right))", $"{uneven}: {Tensor.FormatShape(y.Shape)}");
        var values = x.ToArray();
        AssertClose([MathF.Max(values[36], values[37])], [y.ToArray()[9]], 0f, "the last window reads the last row only");
        using var weights = Tensor.From(RandomArray(new Random(105), 2 * 4 * 4), [1, 2, 4, 4], device);
        GradCheck(device, [1, 2, 7, 6], x => (new AvgPool2d((3, 2), (2, 2), (1, 1), countIncludePad: false, ceilMode: true, paddingEnd: (1, 0)).Forward(x)
            * weights).Sum());
        Expect<ArgumentOutOfRangeException>(() => new MaxPool2d((2, 2), paddingEnd: (2, 0)), "padding below of more than half the window");
    }

    // A small decoder built with every new step.
    private static NetworkBuilder SmallDecoder(Device device) => Network.Image(2, 9, 10).OnDevice(device).Seed(111)
        .Conv2d(4, 3, padding: 1).GroupNorm(2).ReLU().MaxPool2d((3, 3), (2, 2), (1, 1), ceilMode: true)
        .ConvTranspose2d(6, (3, 2), (2, 2), (1, 0), (1, 1), groups: 2).Upsample(2).UpsampleToSize((15, 13), InterpolationMode.Bilinear, alignCorners: true)
        .AvgPool2d((2, 2), countIncludePad: false, ceilMode: true).Upsample(1.5f, InterpolationMode.Bilinear).AdaptiveMaxPool2d((4, 5)).GroupNorm(3, affine: false)
        .AdaptiveAvgPool2d(1).Flatten().Linear(3);

    private static void VisionLayersDescribedAgain(Device device)
    {
        using var scope = new TensorScope();
        var builder = SmallDecoder(device);
        using var model = builder.Build();
        var json = builder.ToJson();
        using var replayed = Network.FromJson(json).OnDevice(device).Build();
        var x = Tensor.From(RandomArray(new Random(112), 2 * 2 * 9 * 10), [2, 2, 9, 10], device);
        AssertClose(model.Forward(x).ToArray(), replayed.Forward(x).ToArray(), 0f, "replayed from JSON");
        Check(builder.CurrentShape.SequenceEqual([3]), $"shape {Tensor.FormatShape([.. builder.CurrentShape])}");

        // Pooling without the new options is written and described as before.
        var steps = Network.Image(1, 8, 8).MaxPool2d((2, 2)).AvgPool2d((2, 2)).ToJson()["steps"]!.AsArray();
        Check(steps[0]!.ToJsonString() == """{"op":"maxpool2d","kernel":2,"padding":0}""" && steps[1]!.ToJsonString() == """{"op":"avgpool2d","kernel":2,"padding":0,"countIncludePad":true}""",
            steps.ToJsonString());
        Check(LayerTypes.Describe(new AvgPool2d(2)).ToJsonString() == """{"type":"avgpool2d","kernel":2,"stride":2,"padding":0,"countIncludePad":true}""", "average pooling described as before");

        Module[] layers =
        [
            new ConvTranspose2d(4, 6, (3, 2), (2, 1), (1, 0), (1, 0), (1, 2), groups: 2, device: device, random: new Random(113)),
            new ConvTranspose2d(4, 2, 3, bias: false, device: device, random: new Random(114)), new Upsample((1.5f, 2f), InterpolationMode.Bilinear),
            Upsample.ToSize((11, 7), InterpolationMode.Bilinear, alignCorners: true), new Upsample(2), new AdaptiveAvgPool2d((3, 2)), new AdaptiveMaxPool2d(2),
            new GroupNorm(2, 4, 1e-3f, device: device), new GroupNorm(4, 4, affine: false, device: device),
            new MaxPool2d((3, 3), (2, 2), (1, 1), ceilMode: true, paddingEnd: (0, 1)), new AvgPool2d((3, 2), (2, 2), (1, 1), false, ceilMode: true, paddingEnd: (1, 0)),
        ];
        foreach (var layer in layers)
        {
            foreach (var p in layer.Parameters())
            {
                p.Load(RandomArray(new Random(p.Size), p.Size));
            }

            var description = LayerTypes.Describe(layer);
            using var created = LayerTypes.Create(description, device);
            Check(LayerTypes.Describe(created).ToJsonString() == description.ToJsonString() && created.ToString() == layer.ToString(), $"{layer}: {description.ToJsonString()}");
            using var stream = new MemoryStream();
            layer.Save(stream);
            stream.Position = 0;
            created.Load(stream);
            var input = Tensor.From(RandomArray(new Random(115), 2 * 4 * 9 * 8), [2, 4, 9, 8], device);
            AssertClose(layer.Forward(input).ToArray(), created.Forward(input).ToArray(), 0f, $"{layer} saved and loaded");
        }
    }

    private static void VisionLayersOnnxImport(Device device)
    {
        using var scope = new TensorScope();
        using var model = SmallDecoder(device).Build();
        var x = Tensor.From(RandomArray(new Random(121), 2 * 2 * 9 * 10), [2, 2, 9, 10], device);
        CheckOnnx(model, [2, 9, 10], x, "a decoder on ONNX Runtime");
        CheckImport(model, [2, 9, 10], x, "a decoder imported again");

        // PyTorch's export and onnx's reference evaluator.
        string folder = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vision-layers");
        foreach (var (file, name) in new[] { ("decoder.onnx", "onnx_decoder"), ("windows.onnx", "onnx_windows") })
        {
            var c = LayerCase(name);
            using var imported = OnnxImport.Load(Path.Combine(folder, file), device);
            var y = imported.Model.Predict(FromCase(c["input"]!, device));
            Check(y.Shape.SequenceEqual(Shape(c["output"]!)), $"{file}: shape {Tensor.FormatShape(y.Shape)}");
            AssertClose(Floats(c["output"]!), y.ToArray(), 1e-4f, $"{file} imported");
        }
    }
}
