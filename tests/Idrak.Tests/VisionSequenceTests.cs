// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Inference.Abstractions;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Onnx;

// Plan 13, steps 3 and 2 (the parts a text-line recognizer uses): CTC loss and decoding, convolutions with rectangular
// kernels, strides, padding, dilation and groups, rectangular pooling, bidirectional and stacked recurrent layers. Checked
// against PyTorch (tests/Idrak.Tests/data/vision-sequence, written by tools/pytorch/vision_sequence_reference.py), plain
// loops and finite differences.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionSequenceGroup =
    [
        ("vision sequence: CTC loss matches PyTorch (none, sum, mean, zero infinity, an impossible sequence, batch first, blank 4) and so does its gradient on the logits", CtcMatchesPyTorch),
        ("vision sequence: gradient: CTC loss against finite differences (time-major and batch-first, padded and concatenated targets, sum and mean)", CtcGradient),
        ("vision sequence: CTC decoding: greedy best path, prefix beam search against every path listed, pruning, batches; the CtcDecoders registry", CtcDecoding),
        ("vision sequence: Conv2d with rectangular kernels, strides, padding, dilation and groups (depthwise) matches PyTorch; the square constructor is the same layer", ConvMatchesPyTorch),
        ("vision sequence: gradient: Conv2d rectangular, dilated, grouped and depthwise (input and weights)", ConvGradients),
        ("vision sequence: MaxPool2d and AvgPool2d with rectangular windows match PyTorch (padding counted and not); gradient: their gradients", PoolsMatchPyTorch),
        ("vision sequence: bidirectional two-layer LSTM and GRU (with the candidate bias) match PyTorch's outputs and last states through LoadGateWeights", RecurrentMatchesPyTorch),
        ("vision sequence: gradient: bidirectional stacked LSTM and GRU (inputs and weights)", RecurrentGradients),
        ("vision sequence: builder steps replay from JSON, layer types describe and create, weights save and load; square and one-direction layers are described as before", VisionLayersDescribed),
        ("vision sequence: ONNX: a convolutional recurrent network exports (ONNX Runtime matches) and imports again; PyTorch's exports of the new layers import and match", VisionLayersOnnx),
    ];

    private static readonly Lazy<JsonObject> SequenceReference = new(() => JsonNode.Parse(File.ReadAllText(
        Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vision-sequence", "reference.json")))!.AsObject());

    private static JsonObject Case(string name) => SequenceReference.Value[name]!.AsObject();

    private static float[] Floats(JsonNode node) => [.. node["values"]!.AsArray().Select(v => (float)v!)];

    private static float[] FloatList(JsonNode node) => [.. node.AsArray().Select(v => v!.GetValueKind() == System.Text.Json.JsonValueKind.String ? float.Parse((string)v!, System.Globalization.CultureInfo.InvariantCulture) : (float)v!)];

    private static int[] Shape(JsonNode node) => [.. node["shape"]!.AsArray().Select(v => (int)v!)];

    private static int[] IntList(JsonNode node) => [.. node.AsArray().Select(v => (int)v!)];

    private static (int, int) PairOf(JsonNode node) => ((int)node[0]!, (int)node[1]!);

    private static Tensor FromCase(JsonNode node, Device device, bool requiresGrad = false) => Tensor.From(Floats(node), Shape(node), device, requiresGrad);

    // PyTorch's losses are finite or +inf; json.dumps writes inf as Infinity, which System.Text.Json reads as a string.
    private static void AssertLosses(float[] expected, float[] actual, float tolerance, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: {actual.Length} losses, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            bool ok = float.IsPositiveInfinity(expected[i]) ? float.IsPositiveInfinity(actual[i]) : MathF.Abs(expected[i] - actual[i]) <= tolerance * (1 + MathF.Abs(expected[i]));
            Check(ok, $"{what}: loss {i} is {actual[i]}, PyTorch {expected[i]}");
        }
    }

    private static void CtcMatchesPyTorch(Device device)
    {
        using var scope = new TensorScope();
        var c = Case("ctc");
        var logits = FromCase(c["logits"]!, device);
        var targets = FromCase(c["targets"]!, device);
        int[] inputs = IntList(c["input_lengths"]!), lengths = IntList(c["target_lengths"]!);
        var logProbs = logits.LogSoftmax();
        foreach (bool zero in new[] { false, true })
        {
            var expected = c[zero ? "zero_infinity" : "plain"]!;
            foreach (var (name, reduction) in new[] { ("none", LossReduction.None), ("sum", LossReduction.Sum), ("mean", LossReduction.Mean) })
            {
                var loss = Losses.Ctc(logProbs, targets, inputs, lengths, blank: 0, reduction, zeroInfinity: zero);
                AssertLosses(FloatList(expected[name]!), loss.ToArray(), 1e-5f, $"CTC {name}, zero infinity {zero}");
            }
        }

        // Labels given as arrays (concatenated targets) give the same losses.
        int[][] labels = [.. Enumerable.Range(0, lengths.Length).Select(n => targets.ToArray().Skip(n * targets.Shape[1]).Take(lengths[n]).Select(v => (int)v).ToArray())];
        AssertLosses(FloatList(c["zero_infinity"]!["none"]!), Losses.Ctc(logProbs, labels, inputs, reduction: LossReduction.None, zeroInfinity: true).ToArray(), 1e-5f,
            "CTC with labels as arrays");

        var x = FromCase(c["logits"]!, device, requiresGrad: true);
        Losses.Ctc(x.LogSoftmax(), targets, inputs, lengths, zeroInfinity: true).Backward();
        AssertClose(FloatList(c["zero_infinity"]!["grad_logits_mean"]!), x.Grad!.ToArray(), 1e-5f, "CTC gradient on the logits (mean, zero infinity)");

        var b = Case("ctc_batch_first");
        var bf = FromCase(b["logits"]!, device).LogSoftmax();
        var none = Losses.Ctc(bf, FromCase(b["targets"]!, device), IntList(b["input_lengths"]!), IntList(b["target_lengths"]!), blank: (int)b["blank"]!,
            reduction: LossReduction.None, batchFirst: true);
        AssertLosses(FloatList(b["none"]!), none.ToArray(), 1e-5f, "CTC batch first, blank 4");

        // Errors name what is wrong.
        Expect<ArgumentException>(() => Losses.Ctc(logProbs, targets, [12, 9, 4], lengths), "a missing input length");
        Expect<ArgumentException>(() => Losses.Ctc(logProbs, targets, [13, 9, 4, 5], lengths), "an input length past the steps");
        Expect<ArgumentException>(() => Losses.Ctc(logProbs, Tensor.From(new float[,] { { 0, 1 }, { 2, 3 }, { 1, 1 }, { 1, 1 } }, device), inputs, [1, 1, 1, 1]), "a blank label");
    }

    private static void Expect<TException>(Action action, string what) where TException : Exception
    {
        try
        {
            action();
        }
        catch (TException)
        {
            return;
        }

        Check(false, $"{what}: no {typeof(TException).Name}");
    }

    private static void CtcGradient(Device device)
    {
        using var padded = Tensor.From(new float[,] { { 1, 2, 2 }, { 3, 1, 0 } }, device);
        int[] inputs = [6, 5], lengths = [3, 2];
        GradCheck(device, [6, 2, 4], x => Losses.Ctc(x, padded, inputs, lengths), tolerance: 1e-2f);
        GradCheck(device, [6, 2, 4], x => Losses.Ctc(x.LogSoftmax(), padded, inputs, lengths, reduction: LossReduction.Sum), tolerance: 1e-2f);
        using var concatenated = Tensor.From([1f, 3f, 3f, 2f, 1f], [5], device);
        GradCheck(device, [2, 7, 4], x => Losses.Ctc(x.LogSoftmax(), concatenated, [7, 4], [3, 2], blank: 0, batchFirst: true), tolerance: 1e-2f);
        GradCheck(device, [2, 5, 3], x => Losses.Ctc(x, Tensor.From([0f, 1f, 1f], [3], device), [5, 3], [2, 1], blank: 2, batchFirst: true, reduction: LossReduction.Sum),
            tolerance: 1e-2f);
    }

    private static void CtcDecoding(Device device)
    {
        // Greedy: the best class per step, repeats collapsed and blanks dropped.
        int[] path = [1, 1, 0, 2, 2, 0, 0, 1, 1, 2];
        var rows = new float[path.Length * 3];
        for (int t = 0; t < path.Length; t++)
        {
            for (int k = 0; k < 3; k++)
            {
                rows[t * 3 + k] = MathF.Log(k == path[t] ? 0.8f : 0.1f);
            }
        }

        var greedy = CtcDecoders.Decode(rows, path.Length, 3);
        Check(greedy is [{ Labels: [1, 2, 1, 2] } best] && MathF.Abs(best.LogProbability - path.Length * MathF.Log(0.8f)) < 1e-4f,
            $"greedy: [{string.Join(",", greedy[0].Labels)}] {greedy[0].LogProbability}");

        // Beam search against every path listed: with a wide beam it finds the most likely label sequences and their probabilities.
        var r = new Random(41);
        int steps = 6, classes = 3;
        var logProbs = new float[steps * classes];
        for (int t = 0; t < steps; t++)
        {
            var e = Enumerable.Range(0, classes).Select(_ => Math.Exp(r.NextDouble() * 3)).ToArray();
            for (int k = 0; k < classes; k++)
            {
                logProbs[t * classes + k] = (float)Math.Log(e[k] / e.Sum());
            }
        }

        var exact = new Dictionary<string, double>();
        for (int code = 0; code < (int)Math.Pow(classes, steps); code++)
        {
            var labels = new List<int>();
            double p = 0;
            int previous = -1;
            for (int t = 0, rest = code; t < steps; t++, rest /= classes)
            {
                int k = rest % classes;
                p += logProbs[t * classes + k];
                if (k != 0 && k != previous)
                {
                    labels.Add(k);
                }

                previous = k;
            }

            string key = string.Join(",", labels);
            exact[key] = exact.TryGetValue(key, out var sum) ? sum + Math.Exp(p) : Math.Exp(p);
        }

        var ranked = exact.OrderByDescending(p => p.Value).Take(3).ToList();
        var beam = CtcDecoders.Decode(logProbs, steps, classes, CtcDecoders.Beam, new CtcDecodeOptions { BeamWidth = 1000, Results = 3 });
        for (int i = 0; i < 3; i++)
        {
            Check(string.Join(",", beam[i].Labels) == ranked[i].Key && Math.Abs(beam[i].LogProbability - Math.Log(ranked[i].Value)) < 1e-4,
                $"beam result {i}: [{string.Join(",", beam[i].Labels)}] {beam[i].LogProbability}, exact [{ranked[i].Key}] {Math.Log(ranked[i].Value)}");
        }

        // Pruning that keeps every class changes nothing; a narrow beam still returns readings, the best first.
        var pruned = CtcDecoders.Decode(logProbs, steps, classes, CtcDecoders.Beam, new CtcDecodeOptions { BeamWidth = 1000, Results = 3, TopClasses = 3, MinLogProbability = -100f });
        Check(pruned.Select(h => string.Join(",", h.Labels)).SequenceEqual(beam.Select(h => string.Join(",", h.Labels))), "pruning that keeps every class");
        var narrow = CtcDecoders.Decode(logProbs, steps, classes, CtcDecoders.Beam, new CtcDecodeOptions { BeamWidth = 2, Results = 2, TopClasses = 2 });
        Check(narrow.Count == 2 && narrow[0].LogProbability >= narrow[1].LogProbability, "a narrow beam");

        // Batches from a tensor, time-major or batch-first, with lengths.
        using var batch = Tensor.From([.. logProbs, .. logProbs], [2, steps, classes], device);
        var decoded = CtcDecoders.Decode(batch, [steps, 3], CtcDecoders.Beam, new CtcDecodeOptions { BeamWidth = 1000 }, batchFirst: true);
        var shorter = CtcDecoders.Decode(logProbs.AsMemory(0, 3 * classes), 3, classes, CtcDecoders.Beam, new CtcDecodeOptions { BeamWidth = 1000 });
        Check(decoded[0][0].Labels.SequenceEqual(beam[0].Labels) && decoded[1][0].Labels.SequenceEqual(shorter[0].Labels), "batch decoding with lengths");
        using var timeMajor = batch.Permute(1, 0, 2);
        var greedyBatch = CtcDecoders.Decode(timeMajor);
        Check(greedyBatch.Length == 2 && greedyBatch[0][0].Labels.SequenceEqual(greedyBatch[1][0].Labels), "time-major batch");

        // The registry: built-ins, an app's decoder, a replaced built-in and its fallback, errors that name the registered.
        Check(CtcDecoders.Names.Contains(CtcDecoders.Greedy) && CtcDecoders.Names.Contains(CtcDecoders.Beam), "built-in decoders");
        try
        {
            CtcDecoders.Register("first-class", (values, n, k, _) => [new CtcHypothesis([1], 0f)]);
            Check(CtcDecoders.Decode(logProbs, steps, classes, "first-class")[0].Labels is [1], "an app's decoder");
            CtcDecoders.Register(CtcDecoders.Greedy, (_, _, _, _) => throw new InvalidOperationException("broken"));
            Check(CtcDecoders.Origin(CtcDecoders.Greedy) != Overrides.Library && CtcDecoders.Default(CtcDecoders.Greedy) is not null, "replaced greedy");
            CtcDecoders.SetPolicy(CtcDecoders.Greedy, SlotPolicy.FallBack);
            Check(CtcDecoders.Decode(rows, path.Length, 3)[0].Labels.SequenceEqual([1, 2, 1, 2]), "falls back to the library's greedy");
        }
        finally
        {
            CtcDecoders.SetPolicy(CtcDecoders.Greedy, SlotPolicy.Throw);
            CtcDecoders.Unregister("first-class");
            CtcDecoders.Unregister(CtcDecoders.Greedy);
        }

        Check(CtcDecoders.Origin(CtcDecoders.Greedy) == Overrides.Library, "greedy back");
        try
        {
            CtcDecoders.Get("lexicon");
            Check(false, "an unknown decoder");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("greedy") && e.Message.Contains("beam"), e.Message);
        }
    }

    private static Conv2d ConvFromCase(JsonObject c, Device device)
    {
        int groups = (int)c["groups"]!;
        return Conv2d.FromWeights(FromCase(c["weight"]!, device), FromCase(c["bias"]!, device), PairOf(c["kernel"]!), PairOf(c["stride"]!), PairOf(c["padding"]!),
            PairOf(c["dilation"]!), groups);
    }

    private static void ConvMatchesPyTorch(Device device)
    {
        using var scope = new TensorScope();
        foreach (string name in new[] { "conv_rect", "conv_dilated_grouped", "conv_depthwise" })
        {
            var c = Case(name);
            using var conv = ConvFromCase(c, device);
            var y = conv.Forward(FromCase(c["input"]!, device));
            Check(y.Shape.SequenceEqual(Shape(c["output"]!)), $"{name}: shape {Tensor.FormatShape(y.Shape)}");
            AssertClose(Floats(c["output"]!), y.ToArray(), 1e-4f, name);
        }

        // The square constructor is the pair constructor with equal pairs: the same weights, the same outputs.
        using var square = new Conv2d(2, 3, 3, stride: 2, padding: 1, device: device, random: new Random(5));
        using var pairs = new Conv2d(2, 3, (3, 3), (2, 2), (1, 1), device: device, random: new Random(5));
        Check(square.IsSquare && pairs.IsSquare && square.ToString() == pairs.ToString() && square.ToString() == "Conv2d(2 -> 3, 3x3, stride 2, padding 1)", square.ToString());
        var x = Tensor.From(RandomArray(new Random(6), 2 * 2 * 7 * 6), [2, 2, 7, 6], device);
        AssertClose(square.Forward(x).ToArray(), pairs.Forward(x).ToArray(), 0f, "square and pair constructors");
        Check(new Conv2d(4, 6, (3, 2), (1, 2), (2, 1), (2, 3), groups: 2).ToString() == "Conv2d(4 -> 6, 3x2, stride 1x2, padding 2x1, dilation 2x3, groups 2)", "ToString");
        Expect<ArgumentException>(() => new Conv2d(4, 6, (3, 3), groups: 4), "groups that do not divide the outputs");
        Expect<ArgumentOutOfRangeException>(() => new Conv2d(4, 6, (3, 0)), "an empty kernel");
    }

    private static void ConvGradients(Device device)
    {
        var rect = new Conv2d(3, 4, (3, 2), (2, 1), (1, 1), (2, 1), groups: 1, device: device, random: new Random(51));
        GradCheck(device, [2, 3, 7, 6], x => rect.Forward(x).Tanh().Sum());
        ParameterGradCheck(rect, Tensor.From(RandomArray(new Random(52), 2 * 3 * 7 * 6), [2, 3, 7, 6], device), y => y.Tanh().Sum());
        var grouped = new Conv2d(4, 6, (2, 3), (1, 2), (1, 2), (1, 2), groups: 2, device: device, random: new Random(53));
        GradCheck(device, [1, 4, 5, 9], x => grouped.Forward(x).Tanh().Sum());
        ParameterGradCheck(grouped, Tensor.From(RandomArray(new Random(54), 4 * 5 * 9), [1, 4, 5, 9], device), y => y.Tanh().Sum());
        var depthwise = new Conv2d(3, 3, (3, 3), padding: (1, 1), groups: 3, bias: false, device: device, random: new Random(55));
        GradCheck(device, [2, 3, 4, 5], x => depthwise.Forward(x).Tanh().Sum());
        ParameterGradCheck(depthwise, Tensor.From(RandomArray(new Random(56), 2 * 3 * 4 * 5), [2, 3, 4, 5], device), y => y.Tanh().Sum());
    }

    private static void PoolsMatchPyTorch(Device device)
    {
        using var scope = new TensorScope();
        var m = Case("maxpool");
        var max = new MaxPool2d(PairOf(m["kernel"]!), PairOf(m["stride"]!), PairOf(m["padding"]!));
        AssertClose(Floats(m["output"]!), max.Forward(FromCase(m["input"]!, device)).ToArray(), 0f, "max pooling");
        foreach (string name in new[] { "avgpool", "avgpool_nopad" })
        {
            var a = Case(name);
            var average = new AvgPool2d(PairOf(a["kernel"]!), PairOf(a["stride"]!), PairOf(a["padding"]!), (bool)a["count_include_pad"]!);
            var y = average.Forward(FromCase(a["input"]!, device));
            Check(y.Shape.SequenceEqual(Shape(a["output"]!)), $"{name}: shape {Tensor.FormatShape(y.Shape)}");
            AssertClose(Floats(a["output"]!), y.ToArray(), 1e-5f, name);
        }

        using var weights = Tensor.From(RandomArray(new Random(57), 2 * 2 * 4 * 6), [2, 2, 4, 6], device);
        GradCheck(device, [2, 2, 7, 6], x => (new AvgPool2d((2, 3), (2, 1), (1, 1), countIncludePad: false).Forward(x) * weights).Sum());
        Tensor Spread(Tensor t)
        {
            var offsets = Enumerable.Range(0, t.Size).Select(i => 0.5f * i).ToArray();
            new Random(58).Shuffle(offsets);
            return t + Tensor.From(offsets, t.Shape, device);
        }

        GradCheck(device, [1, 2, 7, 6], x => new MaxPool2d((3, 2), (2, 1), (1, 0)).Forward(Spread(x)).Square().Sum() * 0.01f, scale: 0.1f);
    }

    // A layer made like the reference, loaded with PyTorch's parameters through LoadGateWeights.
    private static RecurrentModule RecurrentFromCase(JsonObject c, bool gru, bool sequences, Device device)
    {
        int inputs = (int)c["input_size"]!, hidden = (int)c["hidden_size"]!, layers = (int)c["layers"]!;
        RecurrentModule module = gru
            ? new GRU(inputs, hidden, sequences, bidirectional: true, layers, candidateBias: true, device)
            : new LSTM(inputs, hidden, sequences, bidirectional: true, layers, device);
        var p = c["parameters"]!;
        for (int layer = 0; layer < layers; layer++)
        {
            foreach (string suffix in new[] { "", "_reverse" })
            {
                string at = $"_l{layer}{suffix}";
                module.LoadGateWeights(layer, suffix.Length > 0, Floats(p["weight_ih" + at]!), Floats(p["weight_hh" + at]!), Floats(p["bias_ih" + at]!),
                    Floats(p["bias_hh" + at]!));
            }
        }

        return module;
    }

    private static void RecurrentMatchesPyTorch(Device device)
    {
        using var scope = new TensorScope();
        foreach (string name in new[] { "lstm", "gru" })
        {
            var c = Case(name);
            var input = FromCase(c["input"]!, device);
            using var sequences = RecurrentFromCase(c, name == "gru", true, device);
            var y = sequences.Forward(input);
            Check(y.Shape.SequenceEqual(Shape(c["output"]!)), $"{name}: shape {Tensor.FormatShape(y.Shape)}");
            AssertClose(Floats(c["output"]!), y.ToArray(), 1e-5f, $"{name} every step");

            // The last states: the forward direction's after the last step and the backward one's after the first, h_n[-2:].
            using var last = RecurrentFromCase(c, name == "gru", false, device);
            var hn = Floats(c["h_n"]!);
            int batch = input.Shape[0], hidden = sequences.HiddenSize;
            var expected = new float[batch * 2 * hidden];
            for (int n = 0; n < batch; n++)
            {
                for (int d = 0; d < 2; d++)
                {
                    Array.Copy(hn, ((2 + d) * batch + n) * hidden, expected, (n * 2 + d) * hidden, hidden);
                }
            }

            AssertClose(expected, last.Forward(input).ToArray(), 1e-5f, $"{name} last states");
        }

        // A GRU without the candidate bias takes PyTorch's weights only when that bias is zero, and says so otherwise.
        var g = Case("gru");
        var plain = new GRU(3, 4, returnSequences: true, bidirectional: true, layers: 2, device: device);
        var parameters = g["parameters"]!;
        Expect<NotSupportedException>(() => plain.LoadGateWeights(0, false, Floats(parameters["weight_ih_l0"]!), Floats(parameters["weight_hh_l0"]!),
            Floats(parameters["bias_ih_l0"]!), Floats(parameters["bias_hh_l0"]!)), "a GRU without a candidate bias");
        Expect<ArgumentOutOfRangeException>(() => new LSTM(3, 4).LoadGateWeights(0, true, new float[48], new float[64], [], []), "a reverse direction of a one-direction layer");
    }

    private static void RecurrentGradients(Device device)
    {
        var lstm = new LSTM(3, 4, returnSequences: true, bidirectional: true, layers: 2, device: device, random: new Random(61));
        GradCheck(device, [2, 3, 3], x => lstm.Forward(x).Sum());
        ParameterGradCheck(lstm, Tensor.From(RandomArray(new Random(62), 18), [2, 3, 3], device), y => y.Square().Sum());
        var gru = new GRU(3, 4, returnSequences: false, bidirectional: true, layers: 2, candidateBias: true, device: device, random: new Random(63));
        foreach (var w in gru.Weights)
        {
            w.HiddenBias!.Load(RandomArray(new Random(64), 4));
        }

        GradCheck(device, [2, 3, 3], x => gru.Forward(x).Sum());
        ParameterGradCheck(gru, Tensor.From(RandomArray(new Random(65), 18), [2, 3, 3], device), y => y.Square().Sum());
    }

    // A small convolutional recurrent network over [1, 16, 24] images: features, columns as steps, a bidirectional LSTM, scores.
    private static NetworkBuilder SequenceReader(Device device) => Network.Image(1, 16, 24).OnDevice(device).Seed(71)
        .Conv2d(6, (3, 5), padding: (1, 2)).ReLU().MaxPool2d((2, 2))
        .Conv2d(6, (3, 3), padding: (2, 1), dilation: (2, 1), groups: 2).ReLU().AvgPool2d((2, 1))
        .Conv2d(4, (4, 1)).ReLU().ColumnsToSequence()
        .LSTM(5, returnSequences: true, bidirectional: true, layers: 2).GRU(3, returnSequences: true, bidirectional: true, candidateBias: true).Linear(7);

    private static void VisionLayersDescribed(Device device)
    {
        using var scope = new TensorScope();
        var builder = SequenceReader(device);
        Check(builder.CurrentShape.SequenceEqual([12, 7]), $"shape {Tensor.FormatShape([.. builder.CurrentShape])}");
        using var model = builder.Build();
        var json = builder.ToJson();
        using var replayed = Network.FromJson(json).OnDevice(device).Build();
        var x = Tensor.From(RandomArray(new Random(72), 2 * 16 * 24), [2, 1, 16, 24], device);
        AssertClose(model.Forward(x).ToArray(), replayed.Forward(x).ToArray(), 0f, "replayed from JSON");

        // Square and one-direction steps are written as before.
        var square = Network.Image(1, 8, 8).Conv2d(4, 3, padding: 1).MaxPool2d(2).ToJson()["steps"]!.AsArray();
        Check(square[0]!.ToJsonString() == """{"op":"conv2d","out":4,"kernel":3,"stride":1,"padding":1,"bias":true}""" && square[1]!.ToJsonString() == """{"op":"maxpool2d","kernel":2,"padding":0}""",
            $"square steps: {square.ToJsonString()}");

        // Layer types: rectangular, dilated, grouped, pooled and recurrent layers describe themselves and are created again.
        Module[] layers =
        [
            new Conv2d(4, 6, (3, 2), (1, 2), (2, 1), (2, 3), groups: 2, device: device, random: new Random(73)), new MaxPool2d((3, 2), (2, 1), (1, 0)),
            new AvgPool2d((2, 3), (2, 1), (1, 1), countIncludePad: false), new LSTM(3, 4, true, true, 2, device, new Random(74)),
            new GRU(3, 4, false, true, 2, candidateBias: true, device: device, random: new Random(75)),
        ];
        foreach (var layer in layers)
        {
            var description = LayerTypes.Describe(layer);
            using var created = LayerTypes.Create(description, device);
            Check(LayerTypes.Describe(created).ToJsonString() == description.ToJsonString() && created.ToString() == layer.ToString(), $"{layer}: {description.ToJsonString()}");
            using var stream = new MemoryStream();
            layer.Save(stream);
            stream.Position = 0;
            created.Load(stream);
            var input = layer is RecurrentModule ? Tensor.From(RandomArray(new Random(76), 2 * 5 * 3), [2, 5, 3], device) : Tensor.From(RandomArray(new Random(76), 2 * 4 * 9 * 8), [2, 4, 9, 8], device);
            AssertClose(layer.Forward(input).ToArray(), created.Forward(input).ToArray(), 0f, $"{layer} saved and loaded");
        }

        Check(LayerTypes.Describe(new Conv2d(1, 2, 3)).ToJsonString() == """{"type":"conv2d","in":1,"out":2,"kernel":3,"stride":1,"padding":0,"bias":true}"""
              && LayerTypes.Describe(new LSTM(3, 4)).ToJsonString() == """{"type":"lstm","in":3,"hidden":4,"sequences":false}"""
              && LayerTypes.Describe(new MaxPool2d(2)).ToJsonString() == """{"type":"maxpool2d","kernel":2,"stride":2,"padding":0}""", "square and one-direction descriptions");
    }

    private static void VisionLayersOnnx(Device device)
    {
        using var scope = new TensorScope();
        using var model = SequenceReader(device).Build();
        var x = Tensor.From(RandomArray(new Random(81), 2 * 16 * 24), [2, 1, 16, 24], device);
        CheckOnnx(model, [1, 16, 24], x, "convolutional recurrent network on ONNX Runtime");
        CheckImport(model, [1, 16, 24], x, "convolutional recurrent network imported again");

        // One-direction recurrent layers export and import as before; a stacked GRU with a candidate bias too.
        using var stacked = Network.Sequence(6, 3).OnDevice(device).Seed(82).GRU(4, true, false, 2, candidateBias: true).LSTM(3, false, true).Build();
        var s = Tensor.From(RandomArray(new Random(83), 2 * 6 * 3), [2, 6, 3], device);
        CheckOnnx(stacked, [6, 3], s, "stacked GRU and a bidirectional LSTM's last states on ONNX Runtime");
        CheckImport(stacked, [6, 3], s, "stacked GRU and a bidirectional LSTM's last states imported again");

        // PyTorch's own exports.
        string folder = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vision-sequence");
        foreach (var (file, name) in new[] { ("conv.onnx", "onnx_conv"), ("recurrent.onnx", "onnx_recurrent") })
        {
            var c = Case(name);
            using var imported = OnnxImport.Load(Path.Combine(folder, file), device);
            var y = imported.Model.Predict(FromCase(c["input"]!, device));
            Check(y.Shape.SequenceEqual(Shape(c["output"]!)), $"{file}: shape {Tensor.FormatShape(y.Shape)}");
            AssertClose(Floats(c["output"]!), y.ToArray(), 1e-4f, $"{file} imported");
        }
    }
}
