// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

// The generic layers and tensor operations that came out of the speech plug-ins: padding and flips, convolutions over
// time with a weight-norm parametrization, padded batches, the in-memory tensor store, weight-norm folding.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] LayerLibraryGroup =
    [
        ("layers: Tensor.Pad adds zeros, mirrored or repeated edges along any axis of a [2, 3, 5] (against plain arrays), refuses a reflection wider than the axis allows, and its gradient reaches the positions each padded value copies; Flip reverses an axis and its gradient", PaddingAndFlip),
        ("layers: Conv1d and ConvTranspose1d over [C, T] and [B, C, T] equal Conv2d and ConvTranspose2d over [B, C, 1, T]; a weight-normed Conv1d's weight is g · v / ‖v‖ and its bound step gives g and v the gradients of the per-pass norm within 1e-5", ConvolutionsOverTime),
        ("layers: PaddedSequences gives masks per channel count, rows and key bias that outlive the scope they are first asked in and go with Dispose; TensorDictionary round-trips tensors by name and refuses a mismatched shape; StoredWeights folds a weight norm and finds both naming schemes", PaddedBatchesAndStores),
    ];

    private static void PaddingAndFlip(Device device)
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

    private static void ConvolutionsOverTime(Device device)
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

    private static void PaddedBatchesAndStores(Device device)
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
}
