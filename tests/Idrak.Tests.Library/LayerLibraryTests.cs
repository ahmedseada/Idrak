// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Layers;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Vision;

// The generic layers and tensor operations that came out of the speech plug-ins: padding and flips, convolutions over
// time with a weight-norm parametrization, padded batches, the in-memory tensor store, weight-norm folding.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] LayerLibraryGroup =
    [
        ("layers: Tensor.Pad adds zeros, mirrored or repeated edges along any axis of a [2, 3, 5] (against plain arrays), refuses a reflection wider than the axis allows, and its gradient reaches the positions each padded value copies; Flip reverses an axis and its gradient", PaddingAndFlip),
        ("layers: Conv1d and ConvTranspose1d over [C, T] and [B, C, T] equal Conv2d and ConvTranspose2d over [B, C, 1, T]; a weight-normed Conv1d's weight is g · v / ‖v‖ and its bound step gives g and v the gradients of the per-pass norm within 1e-5", ConvolutionsOverTime),
        ("layers: PaddedSequences gives masks per channel count, rows and key bias that outlive the scope they are first asked in and go with Dispose; TensorDictionary round-trips tensors by name and refuses a mismatched shape; StoredWeights folds a weight norm and finds both naming schemes", PaddedBatchesAndStores),
        ("layers: StoredLayers reads a store's convolutions, transposed convolutions, linears, norms and 1 × 1 projections under their stored names (trainable: every tensor a parameter listed in Trained, a weight norm kept as gain and direction); WaveNet, ResidualCouplingBlock (forward then reverse the identity within 1e-5, with and without a speaker) and PosteriorEncoder (means at noise 0) built from it; a batch with masks equals the sequences alone", AudioLayersFromStored),
        ("layers: HifiGanGenerator decodes long latents in pieces with context equal to one pass within 1e-5 and halves a piece the device refuses; HifiGanDiscriminators created from a seed give a score and a map per layer per discriminator, save and reload bit for bit under VITS's names, and refuse sound that is not [B, L]; the GAN losses equal plain loops", HifiGanLayers),
        ("layers: VisionTransformerEncoder built from a store of every tensor Expected names gives [images, patches, dim], its Tensors round-trip the store bit for bit (q, k, v split from the fused projection), Load takes the names given and sets the values, the checkpointed pass equals the plain one, a wrong shape and an unknown activation are refused; PatchGrid.AveragePool equals a loop", VisionTransformerFromStored),
        ("layers: HifiGanVocoder reads transformers' root-level HiFi-GAN names (conv_pre, upsampler, resblocks, conv_post, mean, scale) from a store and from a saved folder, standardizes the log-mel and gives frames · hop samples equal to the generator's; registered under a name Vocoders.Create gives it back; another mel count and disagreeing lists are refused", HifiGanVocoderFromStored),
        ("layers: ConvolutionalFeedForward (odd and even kernels, the activations by name) equals the convolutions written out and gives each sequence of a padded batch what it gets alone; ElementwiseAffineFlow forward then Reverse is the identity with the log-determinant Σ log scale · positions; Tensor.Indices holds integers exactly", FeedForwardAndAffine),
        ("layers: RelativePositionAttention with window 0 equals scaled-dot-product attention from the same projections within 1e-5, zero relative embeddings change nothing, the trained path (embeddings as parameters) equals the speaking path and gives them gradients; MonotonicAlignment finds the best monotonic path and its durations; DifferentiableLogMel equals the host STFT's log-mel within 1e-4", AttentionAlignmentAndMel),
    ];

    internal static void PaddingAndFlip(Device device)
    {
        float[] values = [.. Enumerable.Range(0, 30).Select(i => (float)i)];
        using var x = Tensor.From(values, [2, 3, 5], device);
        float At(int b, int c, int t) => values[(b * 3 + c) * 5 + t];

        // Zeros, reflection and repetition along the last axis.
        using (var zeros = x.Pad(-1, 2, 1))
        {
            Check(zeros.Shape.SequenceEqual([2, 3, 8]), $"zeros: {Tensor.FormatShape(zeros.Shape)}");
            var z = zeros.ToArray();
            Check(z[0] == 0 && z[1] == 0 && z[2] == At(0, 0, 0) && z[6] == At(0, 0, 4) && z[7] == 0, "zeros before and after, the values between");
        }

        using (var reflected = x.Pad(2, 2, 3, PaddingMode.Reflect))
        {
            var r = reflected.ToArray();
            float[] expected = [At(1, 2, 2), At(1, 2, 1), At(1, 2, 0), At(1, 2, 1), At(1, 2, 2), At(1, 2, 3), At(1, 2, 4), At(1, 2, 3), At(1, 2, 2), At(1, 2, 1)];
            AssertClose(expected, r[^10..], 0f, "reflection of the last row");
        }

        using (var repeated = x.Pad(2, 1, 2, PaddingMode.Replicate))
        {
            var r = repeated.ToArray();
            AssertClose([At(0, 0, 0), At(0, 0, 0), At(0, 0, 1), At(0, 0, 2), At(0, 0, 3), At(0, 0, 4), At(0, 0, 4), At(0, 0, 4)], r[..8], 0f, "repetition of the first row");
        }

        // Along the middle axis, and the first.
        using (var middle = x.Pad(1, 1, 0, PaddingMode.Reflect))
        {
            Check(middle.Shape.SequenceEqual([2, 4, 5]), $"middle axis: {Tensor.FormatShape(middle.Shape)}");
            var m = middle.ToArray();
            AssertClose([At(0, 1, 0), At(0, 1, 1), At(0, 1, 2), At(0, 1, 3), At(0, 1, 4)], m[..5], 0f, "the row before the first is the second (mirrored)");
        }

        using (var first = x.Pad(0, 0, 1, PaddingMode.Replicate))
        {
            Check(first.Shape.SequenceEqual([3, 3, 5]) && first.ToArray()[^1] == At(1, 2, 4), "the first axis repeats the last batch");
        }

        ThrowsNaming<ArgumentException>(() => x.Pad(2, 5, 0, PaddingMode.Reflect).Dispose(), "at most 4");

        // The gradient of a padded tensor's sum reaches each source as many times as it was copied.
        using (var tracked = Tensor.From(values, [2, 3, 5], device, requiresGrad: true))
        {
            using var padded = tracked.Pad(2, 2, 2, PaddingMode.Reflect);
            using var loss = (padded * 2f).Sum();
            loss.Backward();
            var g = tracked.Grad!.ToArray();
            // The left padding copies positions 2 and 1, the right one 3 and 2: position 2 is read three times, 1 and 3 twice, 0 and 4 once.
            AssertClose([2f, 4f, 6f, 4f, 2f], g[..5], 0f, "reflection's gradient counts the copies");
        }

        using (var flipped = x.Flip(2))
        {
            var f = flipped.ToArray();
            AssertClose([At(0, 0, 4), At(0, 0, 3), At(0, 0, 2), At(0, 0, 1), At(0, 0, 0)], f[..5], 0f, "flip over time");
        }

        using (var tracked = Tensor.From(values, [2, 3, 5], device, requiresGrad: true))
        using (var weights = Tensor.From([.. Enumerable.Range(0, 30).Select(i => (float)(i % 5))], [2, 3, 5], device))
        {
            using var loss = (tracked.Flip(1) * weights).Sum();
            loss.Backward();
            var g = tracked.Grad!.ToArray();
            Check(g[0] == 0f && g[1] == 1f && g[4] == 4f && g[5] == 0f, "flip's gradient lands on the mirrored row with the same column weights");
        }
    }

    internal static void ConvolutionsOverTime(Device device)
    {
        var random = new Random(5);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (random.NextSingle() * 2 - 1))];
        using var conv2 = new Conv2d(3, 4, (1, 5), (1, 1), (0, 2), (1, 2), 1, true, device, new Random(9));
        using var conv1 = new Conv1d(new Conv2d(3, 4, (1, 5), (1, 1), (0, 2), (1, 2), 1, true, device, new Random(9)));
        using var x = Tensor.From(Values(2 * 3 * 16), [2, 3, 16], device);
        using (Autograd.NoGrad())
        {
            using var y1 = conv1.Forward(x);
            using var y2 = conv2.Forward(x.Reshape(2, 3, 1, 16));
            Check(y1.Shape.SequenceEqual([2, 4, 12]), $"[B, C, T] out (a dilated filter of 5 spans 9): {Tensor.FormatShape(y1.Shape)}");
            AssertClose(y2.ToArray(), y1.ToArray(), 1e-6f, "Conv1d over a batch is Conv2d with a (1, k) filter");
            using var y0 = conv1.Forward(x.Narrow(0, 0, 1).Reshape(3, 16));
            AssertClose(y2.ToArray()[..(4 * 12)], y0.ToArray(), 1e-6f, "Conv1d over [C, T]");
            Check(conv1.Kernel == 5 && conv1.Padding == 2 && conv1.Dilation == 2 && conv1.Parameters().Count() == 2 && conv1.ToString().Contains("k5"), conv1.ToString());

            using var up2 = new ConvTranspose2d(3, 2, (1, 4), (1, 2), (0, 1), null, null, 1, true, device, new Random(11));
            using var up1 = new ConvTranspose1d(new ConvTranspose2d(3, 2, (1, 4), (1, 2), (0, 1), null, null, 1, true, device, new Random(11)));
            using var u1 = up1.Forward(x);
            using var u2 = up2.Forward(x.Reshape(2, 3, 1, 16));
            Check(u1.Shape.SequenceEqual([2, 2, 32]), $"upsampled: {Tensor.FormatShape(u1.Shape)}");
            AssertClose(u2.ToArray(), u1.ToArray(), 1e-6f, "ConvTranspose1d is ConvTranspose2d with a (1, k) filter");
            Check(up1.Stride == 2 && up1.Kernel == 4, up1.ToString());
        }

        using (var square = new Conv1d(new Conv2d(3, 4, 3, 1, 1, device: device)))
        using (Autograd.NoGrad())
        {
            ThrowsNaming<ArgumentException>(() => square.Forward(x).Dispose(), "(1, k)");
            using var image = square.Forward(x.Reshape(2, 3, 4, 4));
            Check(image.Shape.SequenceEqual([2, 4, 4, 4]), "any filter over [B, C, H, W]");
        }

        // Weight norm: the weight is g · v / ‖v‖; the bound step's gradients match the per-pass graph's.
        using var direction = Tensor.From(Values(4 * 3 * 5), [4, 3, 5], device, requiresGrad: true);
        using var gain = Tensor.From(Values(4), [4, 1], device, requiresGrad: true);
        using var perPass = Tensor.From(direction.ToArray(), [4, 3, 5], device, requiresGrad: true);
        using var perPassGain = Tensor.From(gain.ToArray(), [4, 1], device, requiresGrad: true);
        using var normed = new WeightNorm(gain, direction);
        using var folded = Tensor.From(StoredWeights.FoldWeightNorm(gain.ToArray(), direction.ToArray(), [4, 3, 5]), [4, 3, 5], device);
        using (Autograd.NoGrad())
        {
            AssertClose(folded.ToArray(), normed.Weight.ToArray(), 1e-6f, "the normed weight is the folded one");
        }

        using var input = Tensor.From(Values(2 * 3 * 16), [2, 3, 16], device);
        using var scale = Tensor.From(Values(2 * 4 * 16), [2, 4, 16], device);
        float referenceLoss;
        {
            using var scope = new TensorScope();
            var y = input.Reshape(2, 3, 1, 16).Convolution(Tensor.WeightNorm(perPassGain, perPass), null, new ConvGeometry(2, 3, 1, 16, 1, 5, 1, 1, 0, 2), 1);
            var loss = (y.Reshape(2, 4, 16) * scale).Sum();
            referenceLoss = loss.Item();
            loss.Backward();
        }

        normed.Bind();
        Check(normed.Bound && normed.Weight.RequiresGrad, "bound: a leaf taking gradients");
        float boundLoss;
        {
            using var scope = new TensorScope();
            var y = input.Reshape(2, 3, 1, 16).Convolution(normed.Weight, null, new ConvGeometry(2, 3, 1, 16, 1, 5, 1, 1, 0, 2), 1);
            var loss = (y.Reshape(2, 4, 16) * scale).Sum();
            boundLoss = loss.Item();
            loss.Backward();
        }

        normed.Flush();
        normed.Unbind();
        Check(!normed.Bound && MathF.Abs(boundLoss - referenceLoss) <= 1e-5f * MathF.Abs(referenceLoss), $"bound loss {boundLoss} against {referenceLoss}");
        AssertClose(perPassGain.Grad!.ToArray(), gain.Grad!.ToArray(), 1e-4f, "the gains' gradient through the bound weight");
        AssertClose(perPass.Grad!.ToArray(), direction.Grad!.ToArray(), 1e-4f, "the directions' gradient through the bound weight");
    }

    internal static void PaddedBatchesAndStores(Device device)
    {
        Tensor? mask;
        using (var batch = new PaddedSequences([3, 5, 2], 5, device))
        {
            Check(batch.Batch == 3 && batch.Length == 5 && batch.Positions == 10, "sizes");
            using (new TensorScope())
            {
                mask = batch.Of(2);                                              // first asked for inside a scope
                Check(ReferenceEquals(mask, batch.Of(2)), "one mask per channel count");
            }

            AssertClose([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0], mask.ToArray(), 0f, "the mask survives the scope and marks each sequence's positions");
            AssertClose([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], batch.Rows.ToArray(), 0f, "rows");
            var keys = batch.Keys(2).ToArray();
            Check(keys.Length == 3 * 2 * 25 && keys[0] == 0f && keys[3] == -1e4f && keys[4] == -1e4f && keys[2 * 25 + 4] == 0f, "key bias: padded keys of the first sequence, none of the full one");
            using var x = Tensor.Ones([3, 2, 5], device);
            Check(batch.Apply(x).ToArray().Sum() == 20f, "apply zeroes the padding");
            using var rows = Tensor.Ones([15, 4], device);
            Check(batch.ApplyRows(rows).ToArray().Sum() == 40f, "apply rows zeroes the padded rows");
            ThrowsNaming<ArgumentOutOfRangeException>(() => new PaddedSequences([6], 5, device).Dispose(), "between 0 and");
        }

        bool disposed = false;
        try
        {
            mask.ToArray();
        }
        catch (ObjectDisposedException)
        {
            disposed = true;
        }

        Check(disposed, "Dispose releases the masks");

        using var store = new TensorDictionary();
        store.Add("a.weight", [2, 3], [1, 2, 3, 4, 5, 6]);
        using (var t = Tensor.From([7f, 8f], [2], device))
        {
            store.Add("b.weight_g", t);
        }

        store.Add("b.weight_v", [2, 2], [3, 4, 0, 5]);
        Check(store.Count == 3 && store.Contains("a.weight") && !store.Contains("c") && store.ShapeOf("a.weight").SequenceEqual([2, 3]) && store.Read("b.weight_g").SequenceEqual([7f, 8f]), "round trip");
        Check(store.Names.Order().SequenceEqual(["a.weight", "b.weight_g", "b.weight_v"]), string.Join(", ", store.Names));
        ThrowsNaming<ArgumentException>(() => store.Add("bad", [2, 2], [1f]), "1 values");
        ThrowsNaming<KeyNotFoundException>(() => store.Read("c"), "not held");
        Check(store.Remove("a.weight") && !store.Contains("a.weight"), "remove");

        Check(StoredWeights.WeightNormNames(store, "b") is ("b.weight_g", "b.weight_v") && StoredWeights.WeightNormNames(store, "a") is null, "the older scheme found, a plain weight not");
        store.Add("c.parametrizations.weight.original0", [1], [2f]);
        store.Add("c.parametrizations.weight.original1", [1, 2], [3f, 4f]);
        Check(StoredWeights.WeightNormNames(store, "c") is ("c.parametrizations.weight.original0", "c.parametrizations.weight.original1"), "the parametrizations scheme found");
        AssertClose([7f * 0.6f, 7f * 0.8f, 0f, 8f], StoredWeights.FoldWeightNorm(store.Read("b.weight_g"), store.Read("b.weight_v"), store.ShapeOf("b.weight_v")), 1e-6f, "folded: each row scaled to its gain");
        ThrowsNaming<InvalidDataException>(() => StoredWeights.FoldWeightNorm([1f], [1f, 2f, 3f, 4f], [2, 2]), "1 values for 2");
    }

    // A store of uniform random tensors under the names the layers read: {prefix}.weight [shape] and {prefix}.bias [shape[0]].
    internal static void DrawLayer(TensorDictionary store, Random random, string prefix, int[] shape, bool bias = true, bool normed = false, int biasRows = -1)
    {
        int size = shape.Aggregate(1, (a, b) => a * b), rows = shape[0];
        if (biasRows < 0)
        {
            biasRows = rows;                                                     // a transposed convolution's bias is its output channels, shape[1]
        }

        float bound = 1f / MathF.Sqrt(size / rows);
        float[] values = [.. Enumerable.Range(0, size).Select(_ => (random.NextSingle() * 2 - 1) * bound)];
        if (normed)
        {
            store.Add(prefix + ".weight_g", [rows, .. Enumerable.Repeat(1, shape.Length - 1)], [.. Enumerable.Range(0, rows).Select(_ => 0.5f + random.NextSingle())]);
            store.Add(prefix + ".weight_v", shape, values);
        }
        else
        {
            store.Add(prefix + ".weight", shape, values);
        }

        if (bias)
        {
            store.Add(prefix + ".bias", [biasRows], [.. Enumerable.Range(0, biasRows).Select(_ => (random.NextSingle() * 2 - 1) * bound)]);
        }
    }

    internal static void DrawWaveNet(TensorDictionary store, Random random, string prefix, int channels, int layers, int kernel, int conditioning)
    {
        for (int i = 0; i < layers; i++)
        {
            DrawLayer(store, random, $"{prefix}.in_layers.{i}", [2 * channels, channels, kernel], normed: true);
            DrawLayer(store, random, $"{prefix}.res_skip_layers.{i}", [i < layers - 1 ? 2 * channels : channels, channels, 1], normed: true);
        }

        if (conditioning > 0)
        {
            DrawLayer(store, random, $"{prefix}.cond_layer", [2 * channels * layers, conditioning, 1], normed: true);
        }
    }

    internal static void AudioLayersFromStored(Device device)
    {
        var random = new Random(3);
        using var store = new TensorDictionary();
        const int C = 4, Hidden = 6, Flow = 4, Bins = 9, E = 3, T = 7, Layers = 2, Kernel = 3;
        DrawWaveNet(store, random, "wn", C, Layers, Kernel, 0);
        for (int i = 0; i < 2; i++)
        {
            DrawLayer(store, random, $"flow.flows.{i}.conv_pre", [Hidden, Flow / 2, 1]);
            DrawWaveNet(store, random, $"flow.flows.{i}.wavenet", Hidden, Layers, Kernel, E);
            DrawLayer(store, random, $"flow.flows.{i}.conv_post", [Flow / 2, Hidden, 1]);
        }

        DrawLayer(store, random, "posterior_encoder.conv_pre", [Hidden, Bins, 1]);
        DrawWaveNet(store, random, "posterior_encoder.wavenet", Hidden, Layers, Kernel, E);
        DrawLayer(store, random, "posterior_encoder.conv_proj", [2 * Flow, Hidden, 1]);
        DrawLayer(store, random, "lin", [5, C]);
        DrawLayer(store, random, "proj", [5, C, 1]);
        store.Add("norm.weight", [C], [1f, 1f, 1f, 1f]);
        store.Add("norm.bias", [C], [0f, 0f, 0f, 0f]);
        DrawLayer(store, random, "up", [C, 2, 4], normed: true, biasRows: 2);

        // Trainable: every tensor read is a parameter under its stored name; a weight norm stays its gain and direction.
        var trainable = new StoredLayers(store, device, trainable: true);
        using var linear = trainable.Linear("lin", C, 5);
        using var projection = trainable.ProjectionLinear("proj", C, 5);
        using var norm = trainable.Norm("norm", C, 1e-5f);
        using var up = trainable.ConvTranspose("up", C, 2, 4, 2, 1);
        using var waveNet = WaveNet.FromStored(trainable, "wn", C, Layers, Kernel, 1, 0);
        Check(trainable.Trained.Count == 2 + 2 + 2 + 3 + 3 * 2 * Layers && trainable.Normed.Count == 1 + 2 * Layers, $"{trainable.Trained.Count} parameters, {trainable.Normed.Count} weight norms");
        Check(trainable.Trained.Any(p => p.Name == "up.weight_g" && p.Stored.SequenceEqual([C, 1, 1])) && trainable.Trained.Single(p => p.Name == "lin.weight").Transposed, "stored names and shapes kept");
        Check(trainable.Trained.All(p => p.Tensor.RequiresGrad), "every parameter takes gradients");
        ThrowsNaming<InvalidDataException>(() => trainable.Linear("lin", 5, 5), "the configuration says");
        ThrowsNaming<InvalidDataException>(() => trainable.Tensor("nothing"), "no tensor 'nothing'");

        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var x = Tensor.From([.. Enumerable.Range(0, 2 * C * T).Select(_ => random.NextSingle() * 2 - 1)], [2, C, T], device);
            var y = waveNet.Forward(x, null, null);
            Check(y.Shape.SequenceEqual([2, C, T]), $"WaveNet out: {Tensor.FormatShape(y.Shape)}");
            var first = waveNet.Forward(x.Narrow(0, 0, 1).Reshape(C, T), null);
            AssertClose(y.ToArray()[..(C * T)], first.ToArray(), 1e-5f, "one sequence of the batch alone");
            Check(up.Forward(x).Shape.SequenceEqual([2, 2, 2 * T]) && linear.Forward(x.Reshape(2 * T, C)).Shape[1] == 5 && projection.Forward(x.Reshape(2 * T, C)).Shape[1] == 5, "the other layers run");
            Check(waveNet.ToString().Contains("2 layers") && waveNet.Channels == C, waveNet.ToString());
        }

        // Inference: weight norms folded once; the flow forward then reverse is the identity, with and without a speaker.
        var stored = new StoredLayers(store, device);
        using var flow = ResidualCouplingBlock.FromStored(stored, "flow", 2, Flow, Hidden, Layers, Kernel, 1, E);
        using var posterior = PosteriorEncoder.FromStored(stored, "posterior_encoder", Bins, Hidden, Flow, Layers, Kernel, 1, E);
        Check(stored.Trained.Count == 0 && stored.Normed.Count == 0, "nothing trained when read to speak");
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            float[] values = [.. Enumerable.Range(0, Flow * T).Select(_ => random.NextSingle() * 2 - 1)];
            var z = Tensor.From(values, [Flow, T], device);
            var speaker = Tensor.From([.. Enumerable.Range(0, E).Select(_ => random.NextSingle())], [E, 1], device);
            AssertClose(values, flow.Reverse(flow.Forward(z, speaker), speaker).ToArray(), 1e-5f, "the flow forward then reverse with a speaker");
            AssertClose(values, flow.Reverse(flow.Forward(z, null), null).ToArray(), 1e-5f, "the flow forward then reverse without one");
            Check(!flow.Forward(z, speaker).ToArray().SequenceEqual(values), "the flow moves its second half");

            // A padded batch of two sequences (7 and 5 frames) with masks: each one's rows are what it gets alone.
            var shorter = Tensor.From(values, [Flow, T], device).Narrow(1, 0, 5);
            var padded = Tensor.Concat([z.Reshape(1, Flow, T), shorter.Pad(1, 0, 2).Reshape(1, Flow, T)], 0);
            using var mask = new PaddedSequences([T, 5], T, device);
            var batched = flow.Forward(padded, null, mask).ToArray();
            AssertClose(flow.Forward(z, null).ToArray(), batched[..(Flow * T)], 1e-5f, "the long sequence in the batch");
            var alone = flow.Forward(shorter, null).ToArray();
            for (int c = 0; c < Flow; c++)
            {
                AssertClose(alone[(c * 5)..((c + 1) * 5)], batched[(Flow * T + c * T)..(Flow * T + c * T + 5)], 1e-5f, $"the short sequence's channel {c} in the batch");
                Check(batched[(Flow * T + c * T + 5)..(Flow * T + (c + 1) * T)].All(v => v == 0f), "zero over the padding");
            }

            var spectrogram = Tensor.From([.. Enumerable.Range(0, Bins * T).Select(_ => random.NextSingle())], [Bins, T], device);
            var (means, logs) = posterior.Stats(spectrogram, speaker);
            Check(means.Shape.SequenceEqual([Flow, T]) && logs.Shape.SequenceEqual([Flow, T]) && posterior.FlowSize == Flow, "the posterior's stats");
            AssertClose(means.ToArray(), posterior.Forward(spectrogram, speaker, 0f, new Random(1)).ToArray(), 0f, "noise scale 0 gives the means");
            Check(!posterior.Forward(spectrogram, speaker, 1f, new Random(1)).ToArray().SequenceEqual(means.ToArray()), "noise scale 1 draws");
        }
    }

    internal static void HifiGanLayers(Device device)
    {
        var random = new Random(4);
        using var store = new TensorDictionary();
        const int Flow = 3, Initial = 8;
        int[] rates = [2, 2], kernels = [4, 4], resKernels = [3, 5];
        int[][] dilations = [[1, 3], [1, 3]];
        DrawLayer(store, random, "decoder.conv_pre", [Initial, Flow, 7]);
        for (int i = 0; i < rates.Length; i++)
        {
            DrawLayer(store, random, $"decoder.upsampler.{i}", [Initial >> i, Initial >> (i + 1), kernels[i]], normed: true, biasRows: Initial >> (i + 1));
            for (int j = 0; j < resKernels.Length; j++)
            {
                for (int d = 0; d < dilations[j].Length; d++)
                {
                    DrawLayer(store, random, $"decoder.resblocks.{i * resKernels.Length + j}.convs1.{d}", [Initial >> (i + 1), Initial >> (i + 1), resKernels[j]], normed: true);
                    DrawLayer(store, random, $"decoder.resblocks.{i * resKernels.Length + j}.convs2.{d}", [Initial >> (i + 1), Initial >> (i + 1), resKernels[j]], normed: true);
                }
            }
        }

        DrawLayer(store, random, "decoder.conv_post", [1, Initial >> rates.Length, 7]);
        using var generator = HifiGanGenerator.FromStored(new StoredLayers(store, device), "decoder", Flow, Initial, rates, kernels, resKernels, dilations, 0.1f, 0.01f, 0);
        Check(generator.Hop == 4 && generator.ContextFrames == HifiGanGenerator.ContextFramesFor(rates, kernels, resKernels, dilations) && generator.ContextFrames > 3, $"hop {generator.Hop}, context {generator.ContextFrames} frames");
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            const int T = 60;
            var latents = Tensor.From([.. Enumerable.Range(0, Flow * T).Select(_ => random.NextSingle() * 2 - 1)], [Flow, T], device);
            var whole = generator.Forward(latents, null);
            Check(whole.Shape.SequenceEqual([1, T * 4]), $"one pass: {Tensor.FormatShape(whole.Shape)}");
            var pieces = generator.Samples(latents, null, 16);
            AssertClose(whole.ToArray(), pieces, 1e-5f, "decoded in pieces of 16 frames with context");
            var batch = generator.Forward(Tensor.Concat([latents.Reshape(1, Flow, T), latents.Reshape(1, Flow, T)], 0), null);
            Check(batch.Shape.SequenceEqual([2, T * 4]), $"a batch: {Tensor.FormatShape(batch.Shape)}");
            AssertClose(whole.ToArray(), batch.ToArray()[(T * 4)..], 1e-5f, "the batch's second row is the single pass");
        }

        // Discriminators drawn from a seed: a score and a map per layer per discriminator; saved and reloaded bit for bit.
        var config = new HifiGanDiscriminatorConfig { Periods = [2, 3], PeriodChannels = [4, 8], Scales = 2, ScaleChannels = [4, 8], ScaleKernels = [15, 5], ScaleStrides = [1, 2], ScaleGroups = [1, 2] };
        using var discriminators = HifiGanDiscriminators.Create(config, device, 11);
        Check(discriminators.Module.Named.Count == 3 * (3 * 4) && discriminators.Normed.Count == 3 * 4, $"{discriminators.Module.Named.Count} tensors, {discriminators.Normed.Count} weight norms");
        Check(discriminators.Module.Named.Any(p => p.Name == "discriminators.2.convs.0.parametrizations.weight.original1" && p.Stored.SequenceEqual([4, 1, 5, 1])), "the period discriminators follow the scales, under VITS's names");
        float[] sound = [.. Enumerable.Range(0, 2 * 200).Select(_ => random.NextSingle() * 2 - 1)];
        using (new TensorScope())
        {
            var x = Tensor.From(sound, [2, 200], device);
            var (scores, maps) = discriminators.Forward(x);
            Check(scores.Count == 4 && maps.Count == 4 && maps.All(m => m.Count == 3) && scores.All(s => s.Shape[0] == 1), $"{scores.Count} scores, maps of {string.Join(",", maps.Select(m => m.Count))}");
            var (made, madeMaps) = discriminators.Forward(Tensor.From(sound, [2, 200], device) * 0.5f);
            float plainD = 0, plainG = 0, plainF = 0;
            for (int d = 0; d < 4; d++)
            {
                plainD += scores[d].ToArray().Select(v => (1 - v) * (1 - v)).Average() + made[d].ToArray().Select(v => v * v).Average();
                plainG += made[d].ToArray().Select(v => (1 - v) * (1 - v)).Average();
                for (int l = 0; l < 3; l++)
                {
                    plainF += maps[d][l].ToArray().Zip(madeMaps[d][l].ToArray(), (a, b) => MathF.Abs(a - b)).Average();
                }
            }

            Check(MathF.Abs(AdversarialLosses.LeastSquaresDiscriminator(scores, made).Item() - plainD) <= 1e-5f * MathF.Abs(plainD), "the discriminator loss");
            Check(MathF.Abs(AdversarialLosses.LeastSquaresGenerator(made).Item() - plainG) <= 1e-5f * MathF.Abs(plainG), "the generator's adversarial loss");
            Check(MathF.Abs(AdversarialLosses.FeatureMatching(maps, madeMaps).Item() - plainF) <= 1e-5f * MathF.Abs(plainF), "the feature-matching loss");
            ThrowsNaming<ArgumentException>(() => discriminators.Forward(x.Reshape(2, 1, 200)), "[B, L]");
            ThrowsNaming<ArgumentException>(() => AdversarialLosses.LeastSquaresDiscriminator(scores, [made[0]]), "pair up");
        }

        string folder = TempFolder("hifigan-discriminators");
        try
        {
            discriminators.Save(folder);
            using var reader = SafeTensorsReader.Open(Path.Combine(folder, HifiGanDiscriminators.WeightsFile));
            var again = HifiGanDiscriminatorConfig.FromJson(System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(folder, HifiGanDiscriminators.ConfigFile)))!.AsObject());
            Check(again.ToJson().ToJsonString() == config.ToJson().ToJsonString(), "discriminator.json round-trips");
            using var reloaded = new HifiGanDiscriminators(again, reader, device);
            foreach (var (a, b) in discriminators.Module.Named.Zip(reloaded.Module.Named))
            {
                Check(a.Name == b.Name && a.Values().SequenceEqual(b.Values()), $"{a.Name} reloaded bit for bit");
            }

            ThrowsNaming<InvalidDataException>(() => HifiGanDiscriminatorConfig.FromJson(new System.Text.Json.Nodes.JsonObject { ["periods"] = new System.Text.Json.Nodes.JsonArray(0) }), "not positive");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    internal static void AttentionAlignmentAndMel(Device device)
    {
        var random = new Random(6);
        using var store = new TensorDictionary();
        const int C = 8, Heads = 2, D = C / Heads, T = 5, Window = 2;
        foreach (string name in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
        {
            DrawLayer(store, random, "attn." + name, [C, C]);
        }

        store.Add("attn.emb_rel_k", [1, 2 * Window + 1, D], new float[(2 * Window + 1) * D]);
        store.Add("attn.emb_rel_v", [1, 2 * Window + 1, D], new float[(2 * Window + 1) * D]);
        float[] xValues = [.. Enumerable.Range(0, T * C).Select(_ => random.NextSingle() * 2 - 1)];
        float[] plain, windowed;
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var stored = new StoredLayers(store, device);
            using var none = RelativePositionAttention.FromStored(stored, "attn", C, Heads, 0);
            using var zeros = RelativePositionAttention.FromStored(stored, "attn", C, Heads, Window);
            var x = Tensor.From(xValues, [T, C], device);
            plain = none.Forward(x, 1, T, null).ToArray();
            windowed = zeros.Forward(x, 1, T, null).ToArray();
            AssertClose(plain, windowed, 1e-6f, "zero relative embeddings change nothing");

            // Scaled dot-product attention from the same projections, head by head.
            var q = stored.Linear("attn.q_proj", C, C).Forward(x).ToArray();
            var k = stored.Linear("attn.k_proj", C, C).Forward(x).ToArray();
            var v = stored.Linear("attn.v_proj", C, C).Forward(x).ToArray();
            var merged = new float[T * C];
            for (int h = 0; h < Heads; h++)
            {
                for (int i = 0; i < T; i++)
                {
                    var logits = new double[T];
                    for (int j = 0; j < T; j++)
                    {
                        for (int d = 0; d < D; d++)
                        {
                            logits[j] += (double)q[i * C + h * D + d] * k[j * C + h * D + d];
                        }

                        logits[j] /= Math.Sqrt(D);
                    }

                    double max = logits.Max(), sum = logits.Sum(s => Math.Exp(s - max));
                    for (int j = 0; j < T; j++)
                    {
                        double p = Math.Exp(logits[j] - max) / sum;
                        for (int d = 0; d < D; d++)
                        {
                            merged[i * C + h * D + d] += (float)(p * v[j * C + h * D + d]);
                        }
                    }
                }
            }

            var reference = stored.Linear("attn.out_proj", C, C).Forward(Tensor.From(merged, [T, C], device)).ToArray();
            AssertClose(reference, plain, 1e-5f, "window 0 is scaled dot-product attention");
            Check(none.ToString().Contains("2 heads") && zeros.Window == Window && none.Hidden == C, none.ToString());
        }

        // Non-zero embeddings: the speaking path (host slices) equals the trained path (parameters gathered), which gives them gradients.
        store.Remove("attn.emb_rel_k");
        store.Remove("attn.emb_rel_v");
        store.Add("attn.emb_rel_k", [1, 2 * Window + 1, D], [.. Enumerable.Range(0, (2 * Window + 1) * D).Select(_ => random.NextSingle() - 0.5f)]);
        store.Add("attn.emb_rel_v", [1, 2 * Window + 1, D], [.. Enumerable.Range(0, (2 * Window + 1) * D).Select(_ => random.NextSingle() - 0.5f)]);
        using (new TensorScope())
        {
            using var speaking = RelativePositionAttention.FromStored(new StoredLayers(store, device), "attn", C, Heads, Window);
            var trainable = new StoredLayers(store, device, trainable: true);
            using var training = RelativePositionAttention.FromStored(trainable, "attn", C, Heads, Window);
            float[] spoken;
            using (Autograd.NoGrad())
            {
                spoken = speaking.Forward(Tensor.From(xValues, [T, C], device), 1, T, null).ToArray();
            }

            Check(!spoken.SequenceEqual(windowed), "the relative embeddings change the output");
            var trained = training.Forward(Tensor.From(xValues, [T, C], device), 1, T, null);
            AssertClose(spoken, trained.ToArray(), 1e-5f, "the trained path equals the speaking path");
            trained.Sum().Backward();
            var relK = trainable.Trained.Single(p => p.Name == "attn.emb_rel_k");
            Check(relK.Tensor.Grad is { } g && g.ToArray().Any(v => v != 0f), "the relative keys take a gradient");
            Check(training.Parameters().Count() == 10, $"{training.Parameters().Count()} parameters (4 weights, 4 biases, 2 relative tables)");

            // A padded batch: the second sequence has 3 positions; its rows are what it gets alone.
            using var mask = new PaddedSequences([T, 3], T, device);
            using (Autograd.NoGrad())
            {
                var shorter = Tensor.From(xValues, [T, C], device).Narrow(0, 0, 3);
                var batchRows = Tensor.Concat([Tensor.From(xValues, [T, C], device), shorter.Pad(0, 0, 2)], 0);
                var rows = speaking.Forward(batchRows, 2, T, mask).ToArray();
                AssertClose(spoken, rows[..(T * C)], 1e-5f, "the long sequence in the batch");
                AssertClose(speaking.Forward(shorter, 1, 3, null).ToArray(), rows[(T * C)..((T + 3) * C)], 1e-5f, "the short sequence in the batch");
            }
        }

        // The alignment: 4 frames over 2 tokens whose scores favour frames 0-2 on token 0 and frame 3 on token 1.
        float[] scores = [1f, 0f, 1f, 0f, 1f, 0f, 0f, 1f];
        var path = MonotonicAlignment.Search(scores, 4, 2);
        Check(path.SequenceEqual([0, 0, 0, 1]) && MonotonicAlignment.Durations(path, 2).SequenceEqual([3, 1]), $"path {string.Join(" ", path)}");
        Check(MonotonicAlignment.Search([0f, 1f, 0f, 1f], 2, 2).SequenceEqual([0, 1]), "every token gets a frame even against the scores");
        ThrowsNaming<ArgumentException>(() => MonotonicAlignment.Search(new float[6], 2, 3), "at least as many frames as tokens");

        // The differentiable log-mel against the host STFT of the same (reflect-padded) samples.
        const int Size = 64, Hop = 16, Mels = 8, Rate = 1600;
        var options = new AudioFeatureOptions { SampleRate = Rate, WindowSize = Size, HopSize = Hop, Window = "hann", Center = false, Mels = Mels, PadSeconds = null, Normalize = null, DynamicRange = null, Path = "host" };
        float[] samples = [.. Enumerable.Range(0, 20 * Hop).Select(i => MathF.Sin(i * 0.3f) * 0.5f + (random.NextSingle() - 0.5f) * 0.1f)];
        int pad = (Size - Hop) / 2;
        var padded = new float[samples.Length + 2 * pad];
        samples.CopyTo(padded.AsSpan(pad));
        for (int i = 1; i <= pad; i++)
        {
            padded[pad - i] = samples[i];
            padded[pad + samples.Length - 1 + i] = samples[samples.Length - 1 - i];
        }

        int bins = Size / 2 + 1, frames = Stft.FramesFor(padded.Length, options);
        var power = new float[frames * bins];
        Stft.PowerOnHost(padded, options, power);
        var filters = MelFilterbank.Build(options);
        var expected = new float[frames * Mels];
        for (int f = 0; f < frames; f++)
        {
            for (int m = 0; m < Mels; m++)
            {
                double sum = 0;
                for (int b = 0; b < bins; b++)
                {
                    sum += Math.Sqrt(power[f * bins + b] + 1e-6) * filters[b * Mels + m];
                }

                expected[f * Mels + m] = MathF.Log(MathF.Max((float)sum, 1e-5f));
            }
        }

        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var basis = Tensor.From(DifferentiableLogMel.Basis(Size, Stft.Window("hann", Size)), [2 * bins, Size], device);
            var mel = Tensor.From(filters, [bins, Mels], device);
            var logMel = DifferentiableLogMel.Compute(Tensor.From(samples, [1, samples.Length], device), basis, mel, Size, Hop);
            Check(logMel.Shape.SequenceEqual([frames, Mels]) && frames == 20, $"{Tensor.FormatShape(logMel.Shape)} for {frames} frames");
            AssertClose(expected, logMel.ToArray(), 1e-4f, "the differentiable log-mel against the host STFT's");
            var two = DifferentiableLogMel.Compute(Tensor.From([.. samples, .. samples], [2, samples.Length], device), basis, mel, Size, Hop);
            AssertClose(expected, two.ToArray()[(frames * Mels)..], 1e-4f, "a batch's second row");
            ThrowsNaming<ArgumentException>(() => DifferentiableLogMel.Basis(Size, new float[3]), "values for frames");
        }
    }

    internal static void VisionTransformerFromStored(Device device)
    {
        var random = new Random(8);
        var config = new VisionTransformerConfig { Dim = 16, FfDim = 32, Layers = 2, Heads = 2, ImageSize = 12, PatchSize = 4, Channels = 3, Activation = "gelu" };
        using var store = new TensorDictionary();
        var expected = VisionTransformerEncoder.Expected(config, "tower.").ToList();
        foreach (var (name, shape) in expected)
        {
            int size = shape.Aggregate(1, (a, b) => a * b);
            float bound = name.EndsWith("norm1.weight", StringComparison.Ordinal) || name.EndsWith("norm2.weight", StringComparison.Ordinal) || name.EndsWith("post_layernorm.weight", StringComparison.Ordinal) ? 0.1f : 0.3f;
            store.Add(name, shape, [.. Enumerable.Range(0, size).Select(_ => (random.NextSingle() * 2 - 1) * bound)]);
        }

        Check(expected.Count == 3 + 2 * 16 + 2, $"{expected.Count} tensors expected");
        using var encoder = VisionTransformerEncoder.FromTensors(config, store, "tower.", device, EncoderWeights.Float32);
        Check(encoder.Config.Patches == 9 && encoder.Layers.Count == 2 && encoder.LayerWeights.Count == 2 && encoder.ToString().Contains("9 patches"), encoder.ToString());

        // The tensors come back as stored, name for name and bit for bit.
        var back = encoder.Tensors("tower.").ToList();
        Check(back.Select(t => t.Name).SequenceEqual(expected.Select(t => t.Name)), "Tensors gives Expected's names in order");
        foreach (var (name, shape, values) in back)
        {
            Check(shape.SequenceEqual(store.ShapeOf(name)) && values.SequenceEqual(store.Read(name)), $"{name} round-trips");
        }

        // Load sets what it is given, under the stored layout, and names what it took.
        var replaced = new Dictionary<string, float[]>(StringComparer.Ordinal)
        {
            ["tower.encoder.layers.1.self_attn.k_proj.weight"] = [.. Enumerable.Range(0, 16 * 16).Select(i => i * 0.001f)],
            ["tower.encoder.layers.0.mlp.fc2.bias"] = [.. Enumerable.Range(0, 16).Select(i => (float)i)],
            ["tower.post_layernorm.bias"] = new float[16],
            ["elsewhere.weight"] = [1f],
        };
        var taken = encoder.Load("tower.", replaced);
        Check(taken.Count == 3 && !taken.Contains("elsewhere.weight"), $"took {string.Join(", ", taken)}");
        var after = encoder.Tensors("tower.").ToDictionary(t => t.Name, t => t.Values, StringComparer.Ordinal);
        Check(after["tower.encoder.layers.1.self_attn.k_proj.weight"].SequenceEqual(replaced["tower.encoder.layers.1.self_attn.k_proj.weight"]), "k_proj set inside the fused projection");
        Check(after["tower.encoder.layers.1.self_attn.q_proj.weight"].SequenceEqual(store.Read("tower.encoder.layers.1.self_attn.q_proj.weight")), "q_proj untouched beside it");
        Check(after["tower.encoder.layers.0.mlp.fc2.bias"].SequenceEqual(replaced["tower.encoder.layers.0.mlp.fc2.bias"]) && after["tower.post_layernorm.bias"].All(v => v == 0f), "the bias and the norm set");

        // The pass: [images, patches, dim]; checkpointed layers give the same output and gradients.
        float[] pixels = [.. Enumerable.Range(0, 2 * 3 * 12 * 12).Select(_ => random.NextSingle() * 2 - 1)];
        using var x = Tensor.From(pixels, [2, 3, 12, 12], device);
        float[] plain, checkpointed, plainGrad, checkpointedGrad;
        using (new TensorScope())
        {
            var y = encoder.ForwardTraining(x, checkpointed: false);
            Check(y.Shape.SequenceEqual([2, 9, 16]), $"out {Tensor.FormatShape(y.Shape)}");
            plain = y.ToArray();
            y.Sum().Backward();
            plainGrad = encoder.PostNorm.Gamma.Grad!.ToArray();
            encoder.PostNorm.Gamma.ZeroGrad();
            var z = encoder.ForwardTraining(x, checkpointed: true);
            checkpointed = z.ToArray();
            z.Sum().Backward();
            checkpointedGrad = encoder.PostNorm.Gamma.Grad!.ToArray();
            encoder.PostNorm.Gamma.ZeroGrad();
            using (Autograd.NoGrad())
            {
                AssertClose(plain, encoder.Forward(x).ToArray(), 1e-6f, "Forward equals the training pass");
                AssertClose(plain[..(9 * 16)], encoder.Forward(x.Narrow(0, 0, 1).Reshape(3, 12, 12)).ToArray(), 1e-5f, "one image [channels, size, size]");
            }
        }

        AssertClose(plain, checkpointed, 1e-6f, "the checkpointed pass's output");
        AssertClose(plainGrad, checkpointedGrad, 1e-5f, "the checkpointed pass's gradient");

        // Refusals: a tensor of the wrong shape, an unknown activation, pixels of another size.
        using (var wrong = new TensorDictionary())
        {
            foreach (var (name, shape) in expected)
            {
                wrong.Add(name, name.EndsWith("position_embedding.weight", StringComparison.Ordinal) ? [8, 16] : shape, new float[name.EndsWith("position_embedding.weight", StringComparison.Ordinal) ? 8 * 16 : shape.Aggregate(1, (a, b) => a * b)]);
            }

            ThrowsNaming<InvalidDataException>(() => VisionTransformerEncoder.FromTensors(config, wrong, "tower.", device).Dispose(), "expected [9, 16]");
        }

        ThrowsNaming<InvalidDataException>(() => (config with { Activation = "swish" }).Checked(), "not one a vision transformer takes");
        ThrowsNaming<ArgumentException>(() => encoder.Forward(Tensor.From(new float[3 * 8 * 8], [1, 3, 8, 8], device)).Dispose(), "[images, 3, 12, 12]");

        // The grid pooled 3 x 3 over a 6 x 6 grid against a loop.
        float[] grid = [.. Enumerable.Range(0, 36 * 4).Select(_ => random.NextSingle())];
        using var g = Tensor.From(grid, [1, 36, 4], device);
        var pooled = new float[4 * 4];
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int d = 0; d < 4; d++)
                {
                    float sum = 0;
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            sum += grid[((r * 3 + i) * 6 + c * 3 + j) * 4 + d];
                        }
                    }

                    pooled[(r * 2 + c) * 4 + d] = sum / 9;
                }
            }
        }

        using (new TensorScope())
        {
            var p = PatchGrid.AveragePool(g, 3);
            Check(p.Shape.SequenceEqual([1, 4, 4]), $"pooled {Tensor.FormatShape(p.Shape)}");
            AssertClose(pooled, p.ToArray(), 1e-6f, "3 x 3 average pooling of the grid");
            Check(ReferenceEquals(PatchGrid.AveragePool(g, 1), g), "pool size 1 is the input");
            ThrowsNaming<ArgumentException>(() => PatchGrid.AveragePool(g, 4).Dispose(), "divisible by 4");
        }
    }

    internal static void HifiGanVocoderFromStored(Device device)
    {
        var random = new Random(9);
        var config = new HifiGanVocoderConfig { Mels = 4, SamplingRate = 800, UpsampleInitialChannel = 8, UpsampleRates = [2, 2], UpsampleKernelSizes = [4, 4], ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]] };
        using var store = new TensorDictionary();
        DrawLayer(store, random, "conv_pre", [8, 4, 7]);
        for (int i = 0; i < 2; i++)
        {
            DrawLayer(store, random, $"upsampler.{i}", [8 >> i, 8 >> (i + 1), 4], biasRows: 8 >> (i + 1));
            for (int d = 0; d < 2; d++)
            {
                DrawLayer(store, random, $"resblocks.{i}.convs1.{d}", [8 >> (i + 1), 8 >> (i + 1), 3]);
                DrawLayer(store, random, $"resblocks.{i}.convs2.{d}", [8 >> (i + 1), 8 >> (i + 1), 3]);
            }
        }

        DrawLayer(store, random, "conv_post", [1, 2, 7]);
        float[] mean = [0.1f, -0.2f, 0.3f, 0.5f], scale = [1.5f, 0.5f, 2f, 1f];
        store.Add("mean", [4], mean);
        store.Add("scale", [4], scale);
        using var vocoder = HifiGanVocoder.FromStored(new StoredLayers(store, device), "", config);
        Check(vocoder.Mels == 4 && vocoder.SampleRate == 800 && vocoder.Config.Hop == 4 && vocoder.Normalizes && vocoder.Name == "hifigan" && vocoder.ToString().Contains("standardized"), vocoder.ToString());

        const int Frames = 30;
        float[] mel = [.. Enumerable.Range(0, Frames * 4).Select(_ => random.NextSingle() * 4 - 6)];
        using var logMel = Tensor.From(mel, [Frames, 4], Device.Cpu);
        var sound = vocoder.Synthesize(logMel, device);
        Check(sound.Length == Frames * 4 && sound.SampleRate == 800 && sound.Channels == 1 && sound.Samples.All(float.IsFinite), $"{sound.Length} samples at {sound.SampleRate} Hz");
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var standardized = new float[mel.Length];
            for (int i = 0; i < mel.Length; i++)
            {
                standardized[i] = (mel[i] - mean[i % 4]) / scale[i % 4];
            }

            var direct = vocoder.Generator.Samples(Tensor.From(standardized, [Frames, 4], device).Transpose(), null, 0);
            AssertClose(direct, sound.Samples, 1e-5f, "the vocoder is the generator over the standardized log-mel");
            vocoder.DecoderChunkSeconds = 10.0 * 4 / 800;                                   // pieces of 10 frames
            AssertClose(direct, vocoder.Synthesize(logMel, device).Samples, 1e-5f, "decoded in pieces of 10 frames");
        }

        // The same checkpoint saved as a folder and read back through the checkpoint formats, then registered by name.
        string folder = TempFolder("hifigan-vocoder");
        try
        {
            File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJson().ToJsonString());
            SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), store.Names.Select(n => (n, store.ShapeOf(n), store.Read(n))), SafeTensorType.F32);
            using var loaded = HifiGanVocoder.Load(folder, device);
            Check(loaded.Normalizes && loaded.Config.Hop == 4, loaded.ToString());
            AssertClose(sound.Samples, loaded.Synthesize(logMel, device).Samples, 1e-6f, "the folder's vocoder gives the store's samples");
            HifiGanVocoder.RegisterFolder(folder, "hifigan-test", device);
            try
            {
                var created = Vocoders.Create("hifigan-test");
                Check(created.Mels == 4 && created.SampleRate == 800 && ReferenceEquals(created, Vocoders.Create("hifigan-test")), "registered under a name, read once");
                Check(Vocoders.Names.Contains("hifigan-test"), "listed");
            }
            finally
            {
                Vocoders.Unregister("hifigan-test");
            }

            ThrowsNaming<ArgumentException>(() => loaded.Synthesize(Tensor.From(new float[Frames * 5], [Frames, 5], Device.Cpu), device), "[frames, 4]");
            ThrowsNaming<InvalidDataException>(() => (config with { UpsampleKernelSizes = [4] }).Checked(), "pair up");
            ThrowsNaming<InvalidDataException>(() => HifiGanVocoderConfig.FromJson(new System.Text.Json.Nodes.JsonObject { ["upsample_rates"] = 3 }), "not a list");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    internal static void FeedForwardAndAffine(Device device)
    {
        var random = new Random(10);
        using var store = new TensorDictionary();
        const int C = 4, F = 6, T = 5;
        DrawLayer(store, random, "ff.conv_1", [F, C, 3]);
        DrawLayer(store, random, "ff.conv_2", [C, F, 3]);
        DrawLayer(store, random, "even.conv_1", [F, C, 4]);
        DrawLayer(store, random, "even.conv_2", [C, F, 4]);
        store.Add("flow.log_scale", [2, 1], [0.3f, -0.7f]);
        store.Add("flow.translate", [2, 1], [1.5f, -0.25f]);
        var stored = new StoredLayers(store, device);
        float[] xs = [.. Enumerable.Range(0, 2 * T * C).Select(_ => random.NextSingle() * 2 - 1)];
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            using var odd = ConvolutionalFeedForward.FromStored(stored, "ff", C, F, 3, "gelu_new");
            using var even = ConvolutionalFeedForward.FromStored(stored, "even", C, F, 4, "relu");
            var x = Tensor.From(xs, [2 * T, C], device);
            var y = odd.Forward(x, 2, T, null);
            Check(y.Shape.SequenceEqual([2 * T, C]) && odd.Kernel == 3 && odd.ToString() == "ConvolutionalFeedForward(k3)", $"{Tensor.FormatShape(y.Shape)}");

            // Written out: the convolutions over [B, C, T] with symmetric padding and the tanh GELU between.
            var conv1 = stored.Conv("ff.conv_1", C, F, 3, 1);
            var conv2 = stored.Conv("ff.conv_2", F, C, 3, 1);
            var reference = conv2.Forward(conv1.Forward(x.Reshape(2, T, C).Permute(0, 2, 1)).Gelu()).Permute(0, 2, 1).Reshape(2 * T, C);
            AssertClose(reference.ToArray(), y.ToArray(), 1e-5f, "the odd kernel's pass");

            // The even kernel pads 2 before and 1 after, as transformers does: the output keeps the length.
            var e = even.Forward(x, 2, T, null);
            var c1 = stored.Conv("even.conv_1", C, F, 4, 0);
            var c2 = stored.Conv("even.conv_2", F, C, 4, 0);
            var expected = c2.Forward(c1.Forward(x.Reshape(2, T, C).Permute(0, 2, 1).Pad(-1, 2, 1)).Relu().Pad(-1, 2, 1)).Permute(0, 2, 1).Reshape(2 * T, C);
            AssertClose(expected.ToArray(), e.ToArray(), 1e-5f, "the even kernel's pass");

            // A padded batch (5 and 3 positions): the short sequence's rows are what it gets alone, the padding zero after conv_2's reach.
            using var mask = new PaddedSequences([T, 3], T, device);
            var shorter = Tensor.From(xs, [2 * T, C], device).Narrow(0, T, 3);
            var padded = Tensor.Concat([Tensor.From(xs, [2 * T, C], device).Narrow(0, 0, T), shorter.Pad(0, 0, 2)], 0);
            var rows = odd.Forward(padded, 2, T, mask).ToArray();
            AssertClose(y.ToArray()[..(T * C)], rows[..(T * C)], 1e-5f, "the long sequence in the batch");
            AssertClose(odd.Forward(shorter, 1, 3, null).ToArray(), rows[(T * C)..((T + 3) * C)], 1e-5f, "the short sequence in the batch");
            ThrowsNaming<NotSupportedException>(() => ConvolutionalFeedForward.Activation("swishy"), "not supported");
            Check(ConvolutionalFeedForward.Activation("quick_gelu")(Tensor.From([1f], [1], device)).ToArray()[0] is > 0.84f and < 0.86f, "quick_gelu(1) ≈ 0.846");
        }

        // The affine flow: forward (the batch's log-determinant) and the host inverse.
        using var flow = ElementwiseAffineFlow.FromStored(stored, "flow", 2);
        Check(flow.Channels == 2 && flow.ToString() == "ElementwiseAffineFlow(2 channels)", flow.ToString());
        float[] latents = [.. Enumerable.Range(0, 2 * T).Select(_ => random.NextSingle() * 4 - 2)];
        using (new TensorScope())
        {
            var z = Tensor.From(latents, [1, 2, T], device);
            var moved = flow.Forward(z, null, T, out var logDet);
            var expected = new float[latents.Length];
            for (int c = 0; c < 2; c++)
            {
                for (int n = 0; n < T; n++)
                {
                    expected[c * T + n] = latents[c * T + n] * MathF.Exp(c == 0 ? 0.3f : -0.7f) + (c == 0 ? 1.5f : -0.25f);
                }
            }

            AssertClose(expected, moved.ToArray(), 1e-6f, "translate + e^(log scale) · x per channel");
            Check(MathF.Abs(logDet.Item() - (0.3f - 0.7f) * T) < 1e-5f, $"log-determinant {logDet.Item()}");
            var back = moved.ToArray();
            flow.Reverse(back, T);
            AssertClose(latents, back, 1e-5f, "Reverse gives the latents back");
            ThrowsNaming<ArgumentException>(() => flow.Forward(Tensor.From(new float[3 * T], [1, 3, T], device), null, T, out _), "[B, 2, T]");
        }

        ThrowsNaming<InvalidDataException>(() => ElementwiseAffineFlow.FromStored(stored, "flow", 3).Dispose(), "the configuration says 3");

        // Index tensors hold their integers exactly.
        int[] indices = [0, 1, 7, 65535, 16777216, 3];
        using var ix = Tensor.Indices(indices, device);
        Check(ix.Shape.SequenceEqual([6]) && ix.ToArray().Select(v => (int)v).SequenceEqual(indices), string.Join(" ", ix.ToArray()));
    }
}
