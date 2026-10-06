// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends;
using Idrak.Diagnostics;

namespace Idrak;

// Operations of decoder-only language model layers.
public sealed partial class Tensor
{
    /// <summary>x / sqrt(mean(x²) + eps) over the last dimension (RMS normalization without a gain).</summary>
    internal Tensor RmsNormalize(float eps)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = _shape[^1], rows = Size / cols;
        var y = Empty(_shape, Device);
        var inv = Empty([rows], Device, track: false);
        Backend.RmsNorm(Storage, y.Storage, inv.Storage, rows, cols, eps);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("rms_norm", g =>
            {
                x.Backend.RmsNormBackward(g.Storage, y.Storage, inv.Storage, x.GradStorage(), rows, cols);
                inv.Dispose();
            }, x);
        }
        else
        {
            inv.Dispose();
        }

        return Traced("rms_norm", y, start);
    }

    /// <summary>
    /// input · weights[j] (+ biases[j]) for every j in one pass over the input (inference: not recorded). The input is
    /// [..., k]; each weight is [k, n_j]; results are [..., n_j].
    /// </summary>
    internal static Tensor[] MatMulMany(Tensor input, IReadOnlyList<Tensor> weights, IReadOnlyList<Tensor?> biases)
    {
        input.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int k = input._shape[^1], m = input.Size / k;
        var outputs = new Tensor[weights.Count];
        var products = new (Abstraction.Devices.Storage, Abstraction.Devices.Storage?, Abstraction.Devices.Storage, int)[weights.Count];
        for (int j = 0; j < weights.Count; j++)
        {
            int n = weights[j]._shape[1];
            outputs[j] = Empty([.. input._shape[..^1], n], input.Device);
            products[j] = (weights[j].Storage, biases[j]?.Storage, outputs[j].Storage, n);
        }

        input.Backend.MatMulMany(input.Storage, m, k, products);
        foreach (var output in outputs)
        {
            Traced("matmul_many", output, start);
        }

        return outputs;
    }

    /// <summary>
    /// <paramref name="forward"/>(input) without storing its intermediate results (activation checkpointing): the
    /// forward pass runs without recording, and the backward pass runs it again with recording, back-propagates through
    /// it and frees it at once. Parameters used inside receive their gradients then. Memory drops to the checkpointed
    /// outputs at the cost of a second forward pass. The function must be deterministic (no dropout).
    /// </summary>
    internal static Tensor Checkpoint(Func<Tensor, Tensor> forward, Tensor input)
    {
        if (!Autograd.IsEnabled)
        {
            return forward(input);
        }

        // Only the output survives the first pass: the module's intermediate results are freed at once.
        Tensor output;
        var seeds = new List<uint>();                                   // dropout masks are drawn again identically
        using (Layers.DropoutSeeds.Record(seeds))
        using (Autograd.NoGrad())
        using (var pass = new TensorScope())
        {
            output = pass.Keep(forward(input));
        }

        if (ReferenceEquals(output, input))
        {
            return output;
        }

        output.Record("checkpoint", g =>
        {
            using var scope = new TensorScope();
            var replay = input.Detach();
            replay.RequiresGrad = input.RequiresGrad;
            Tensor recomputed;
            using (Layers.DropoutSeeds.Replay(seeds))
            {
                recomputed = forward(replay);
            }

            if (recomputed.RequiresGrad)
            {
                recomputed.Backward(g);
            }

            if (input.RequiresGrad && replay.Grad is { } grad)
            {
                input.AddGradient(grad, adopt: true);             // the replay dies with this scope
            }
        }, input);
        return output;
    }

    /// <summary>
    /// Weighted token cross-entropy of a language model's output head applied to <paramref name="hidden"/> [rows, dim]:
    /// Σ_r w_r · (logsumexp(head(h_r)) - head(h_r)[t_r]) / <paramref name="normalizer"/>. The head runs on
    /// <paramref name="chunkRows"/> rows at a time and each chunk's gradient is back-propagated through the head at once,
    /// so the [rows, vocabulary] logits never exist together (a 2048-token sequence over a 152k vocabulary would need
    /// 1.2 GB for them and as much for their gradient). The result's backward pass adds the collected gradient to
    /// <paramref name="hidden"/>; parameters inside the head (adapters) receive theirs during this call, so back-propagate
    /// the loss unscaled (scale through <paramref name="normalizer"/> instead).
    /// </summary>
    internal static Tensor TokenCrossEntropy(Tensor hidden, Func<Tensor, Tensor> head, Tensor targets, Tensor weights, float normalizer, int chunkRows)
    {
        hidden.ThrowIfDisposed();
        if (hidden.Rank != 2 || targets.Size != hidden._shape[0] || weights.Size != hidden._shape[0])
        {
            throw new ArgumentException($"TokenCrossEntropy needs hidden [rows, dim] with rows targets and weights, got {FormatShape(hidden._shape)}, {targets.Size} and {weights.Size}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int rows = hidden._shape[0], dim = hidden._shape[1];
        chunkRows = Math.Max(1, chunkRows);
        var device = hidden.Device;
        var backend = hidden.Backend;
        bool record = WillRecord(hidden);
        var gradient = record ? Empty([rows, dim], device, zeroed: true) : null;
        var losses = Empty([rows], device);
        for (int r0 = 0; r0 < rows; r0 += chunkRows)
        {
            int n = Math.Min(chunkRows, rows - r0);
            using var scope = new TensorScope();
            var chunk = Empty([n, dim], device);
            backend.Copy2D(hidden.Storage, r0 * dim, n * dim, chunk.Storage, 0, n * dim, 1, n * dim, accumulate: false);
            chunk.RequiresGrad = record;
            var logits = head(chunk);
            int vocabulary = logits._shape[^1];
            if (logits.Size != n * vocabulary)
            {
                throw new ArgumentException($"The head must map [{n}, {dim}] to [{n}, vocabulary], got {FormatShape(logits._shape)}.");
            }

            var chunkTargets = Empty([n], device);
            var chunkWeights = Empty([n], device);
            var chunkLosses = Empty([n], device);
            backend.Copy2D(targets.Storage, r0, n, chunkTargets.Storage, 0, n, 1, n, accumulate: false);
            backend.Copy2D(weights.Storage, r0, n, chunkWeights.Storage, 0, n, 1, n, accumulate: false);
            backend.SoftmaxCrossEntropyRows(logits.Storage, chunkTargets.Storage, chunkWeights.Storage, chunkLosses.Storage, n, vocabulary,
                1f / normalizer);
            backend.Copy2D(chunkLosses.Storage, 0, n, losses.Storage, r0, n, 1, n, accumulate: false);
            if (record && logits.RequiresGrad)
            {
                logits.Backward(logits);                                         // the logits now hold their gradient
                backend.Copy2D(chunk.GradStorage(), 0, n * dim, gradient!.Storage, r0 * dim, n * dim, 1, n * dim, accumulate: false);
            }
        }

        var loss = Empty([1], device);
        backend.Sum(losses.Storage, loss.Storage, rows, 1f / normalizer);
        if (record)
        {
            loss.Record("token_cross_entropy", g => backend.GroupScaleShift(gradient!.Storage, g.Storage, null, hidden.GradStorage(), rows * dim, 1, rows * dim, true), hidden);
        }

        return Traced("token_cross_entropy", loss, start);
    }

    /// <summary>
    /// <paramref name="product"/> + scale · (x·A)·B (a LoRA adapter's term), accumulated into <paramref name="product"/>'s
    /// buffer: only the rank-wide x·A is stored for the backward pass, not the full-width (x·A)·B and its scaled copy, and
    /// no separate scale or addition pass runs. <paramref name="product"/> must be a fresh result nothing else reads (the
    /// base projection of the same input). Gradients as for <c>product + x.MatMul(a).MatMul(b) * scale</c>.
    /// </summary>
    internal static Tensor AddLowRank(Tensor product, Tensor x, Tensor a, Tensor b, float scale)
    {
        int inputs = a._shape[0], rank = a._shape[1], outputs = b._shape[1];
        int m = x.Size / inputs;
        if (product.Size != m * outputs || b._shape[0] != rank)
        {
            throw new ArgumentException($"AddLowRank: product {FormatShape(product._shape)}, input {FormatShape(x._shape)}, A {FormatShape(a._shape)}, B {FormatShape(b._shape)}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var backend = x.Backend;
        var u = Empty([m, rank], x.Device, zeroed: true);           // scale · x·A
        using (var t = Empty([m, rank], x.Device, track: false))
        {
            backend.BatchedMatMul(x.Storage, a.Storage, t.Storage, 1, m, rank, inputs, false, false, 0f);
            backend.Axpy(t.Storage, u.Storage, m * rank, scale);
        }

        backend.BatchedMatMul(u.Storage, b.Storage, product.Storage, 1, m, outputs, rank, false, false, 1f);   // product += u·B
        product.Storage.AddRef();
        var y = new Tensor([.. product._shape], product.Storage, product.Device, track: true);
        if (Autograd.IsEnabled && (product.RequiresGrad || x.RequiresGrad || a.RequiresGrad || b.RequiresGrad))
        {
            y.Record("lora", g =>
            {
                if (b.RequiresGrad)
                {
                    backend.BatchedMatMul(u.Storage, g.Storage, b.GradStorage(), 1, rank, outputs, m, true, false, 1f);       // dB += uᵀ·g
                }

                if (a.RequiresGrad || x.RequiresGrad)
                {
                    using var du = Empty([m, rank], x.Device, track: false);
                    using var dt = Empty([m, rank], x.Device, zeroed: true, track: false);
                    backend.BatchedMatMul(g.Storage, b.Storage, du.Storage, 1, m, rank, outputs, false, true, 0f);        // du = g·Bᵀ
                    backend.Axpy(du.Storage, dt.Storage, m * rank, scale);                                                // dt = scale · du
                    if (a.RequiresGrad)
                    {
                        backend.BatchedMatMul(x.Storage, dt.Storage, a.GradStorage(), 1, inputs, rank, m, true, false, 1f);  // dA += xᵀ·dt
                    }

                    if (x.RequiresGrad)
                    {
                        backend.BatchedMatMul(dt.Storage, a.Storage, x.GradStorage(), 1, m, inputs, rank, false, true, 1f);  // dx += dt·Aᵀ
                    }
                }

                if (product.RequiresGrad)
                {
                    product.AddGradient(g, adopt: true);                  // last: the base product's backward may reuse g's buffer
                }
            }, product, x, a, b);
        }

        return Traced("lora", y, start);
    }

    /// <summary>
    /// x · W_j + scale_j · (x·A_j)·B_j for 1-3 layers with frozen weights (float32, bfloat16 or 4-bit) and LoRA adapters of
    /// one rank ≤ 32, with each low-rank term computed inside its base product (one more k step of the tensor-core kernel)
    /// instead of in separate passes over the full-width output. The backward pass does the same for the input's
    /// gradient: dx = g·Wᵀ + (scale · g·Bᵀ)·Aᵀ in one product. Null (nothing computed) when the device has no fused
    /// version; callers then run the base products and <see cref="AddLowRank"/>. A frozen bias is added to each output in
    /// place (a bias that trains is not: null then).
    /// </summary>
    internal static Tensor[]? LoraProducts(Tensor input, IReadOnlyList<Layers.Linear> layers, bool withBias = true)
    {
        input.ThrowIfDisposed();
        int k = input._shape[^1], m = input.Size / Math.Max(1, k);
        if (!input.Backend.Capabilities.MatrixUnits || layers.Count is < 1 or > 3 || m < 64 || k < 32 || !MixedPrecision.UsesTensorCores
            || layers[0].Lora is not { } first || first.Rank > 32)
        {
            return null;
        }

        // One weight form for all: float32 (one frozen, untied layer), or one built-in packed format the low-rank products
        // read (formats of one's own have no Format: their layers run their own products).
        var packedFormat = layers[0].PackedWeight?.Format;
        bool dense = layers[0].PackedWeight is null;
        foreach (var layer in layers)
        {
            bool fits = dense
                ? layer.PackedWeight is null && layer.TiedTo is null && layers.Count == 1 && !layer.Weight.RequiresGrad
                : layer.PackedWeight is { LowRankProducts: true, Format: { } own } && own == packedFormat;
            if (!fits || layer.InFeatures != k || layer.Lora is not { } adapter || adapter.Rank != first.Rank
                || adapter.A.Device != input.Device || withBias && layer.Bias is { RequiresGrad: true })
            {
                return null;
            }
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var backend = input.Backend;
        int rank = first.Rank;
        var flat = input.Rank == 2 ? input : input.Reshape(-1, k);
        var us = new Tensor[layers.Count];
        var outputs = new Tensor[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var adapter = layers[j].Lora!;
            us[j] = Empty([m, rank], input.Device);                                      // scale · x·A (kept for dB)
            backend.BatchedMatMul(flat.Storage, adapter.A.Storage, us[j].Storage, 1, m, rank, k, false, false, 0f);
            backend.Affine(us[j].Storage, us[j].Storage, m * rank, adapter.Scale, 0f);
            outputs[j] = Empty([m, layers[j].OutFeatures], input.Device);
        }

        // FP8 copies of the frozen weights (Linear.AttachFloat8): y = u·B first, then x·W on FP8 tensor cores added to it.
        bool done = false;
        if (layers.All(l => l.Float8 is not null))
        {
            done = true;
            for (int j = 0; j < layers.Count && done; j++)
            {
                var f8 = layers[j].Float8!;
                backend.BatchedMatMul(us[j].Storage, layers[j].Lora!.B.Storage, outputs[j].Storage, 1, m, layers[j].OutFeatures, rank, false, false, 0f);
                done = backend.Float8MatMul(flat.Storage, m, k, f8.Values.Storage, f8.Scales.Storage, f8.Columns, outputs[j].Storage, 1f);
            }
        }

        if (!done && dense)
        {
            done = backend.MatMulLowRank(flat.Storage, layers[0].Weight.Storage, outputs[0].Storage, m, layers[0].OutFeatures, k, false, 0f,
                us[0].Storage, layers[0].Lora!.B.Storage, rank);
        }
        else if (!done)
        {
            done = backend.PackedMatMulLowRank(packedFormat!.Value, flat.Storage, m, k,
                [.. layers.Select((l, j) => (l.PackedWeight!.PackedValues.Storage, l.PackedWeight.ScaleValues?.Storage,
                    outputs[j].Storage, l.OutFeatures, us[j].Storage, l.Lora!.B.Storage))], rank);
        }

        if (!done)
        {
            foreach (var unused in us.Concat(outputs))
            {
                unused.Dispose();
            }

            if (!ReferenceEquals(flat, input))
            {
                flat.Dispose();
            }

            return null;
        }

        var results = new Tensor[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var layer = layers[j];
            if (withBias && layer.Bias is { } bias)
            {
                backend.AddRowVector(outputs[j].Storage, bias.Storage, outputs[j].Storage, m, layer.OutFeatures);   // frozen: no gradient
            }

            var (a, b, scale) = (layer.Lora!.A, layer.Lora.B, layer.Lora.Scale);
            var u = us[j];
            int n = layer.OutFeatures;
            if (Autograd.IsEnabled && (flat.RequiresGrad || a.RequiresGrad || b.RequiresGrad))
            {
                outputs[j].Record("lora_fused", g =>
                {
                    if (b.RequiresGrad)
                    {
                        backend.BatchedMatMul(u.Storage, g.Storage, b.GradStorage(), 1, rank, n, m, true, false, 1f);         // dB += uᵀ·g
                    }

                    using var dt = Empty([m, rank], flat.Device, track: false);
                    backend.BatchedMatMul(g.Storage, b.Storage, dt.Storage, 1, m, rank, n, false, true, 0f);                 // g·Bᵀ
                    backend.Affine(dt.Storage, dt.Storage, m * rank, scale, 0f);                                            // dt = scale · g·Bᵀ
                    if (a.RequiresGrad)
                    {
                        backend.BatchedMatMul(flat.Storage, dt.Storage, a.GradStorage(), 1, k, rank, m, true, false, 1f);   // dA += xᵀ·dt
                    }

                    float beta = 1f;
                    var dx = flat.RequiresGrad ? flat.GradientTarget(out beta) : null;         // the first gradient written, not added
                    if (dx is not null && layer.PackedWeight is { } stored && stored.TransposedProduct(g, dx, m, k, n, beta, dt.Storage, a.Storage, rank))
                    {
                        // dx (+)= g·Wᵀ + dt·Aᵀ with W read as the words it is stored in (bfloat16).
                    }
                    else if (dx is not null)
                    {
                        // dx += g·Wᵀ + dt·Aᵀ (W [k, n] as float32; packed weights expanded first).
                        Tensor? expanded = null;
                        if (layer.PackedWeight is { } weight)
                        {
                            expanded = Empty([k, n], flat.Device, track: false);
                            weight.DequantizeInto(expanded.Storage);
                        }

                        using (expanded)
                        {
                            var w = (expanded ?? layer.Weight).Storage;
                            if (!backend.MatMulLowRank(g.Storage, w, dx, m, k, n, true, beta, dt.Storage, a.Storage, rank))
                            {
                                backend.BatchedMatMul(g.Storage, w, dx, 1, m, k, n, false, true, beta);
                                backend.BatchedMatMul(dt.Storage, a.Storage, dx, 1, m, k, rank, false, true, 1f);
                            }
                        }
                    }
                }, flat, a, b);
            }
            else
            {
                u.Dispose();
            }

            results[j] = input.Rank == 2 ? outputs[j] : outputs[j].Reshape([.. input._shape[..^1], n]);
            Traced("lora_fused", results[j], start);
        }

        return results;
    }

    /// <summary>
    /// The frozen packed layers' products of one input in one device pass (<see cref="MatMulPackedMany"/>), recorded for
    /// training: each output's gradient flows to the input (dx += g · Wᵀ with the weight expanded to float32). Null when
    /// the device has no such pass.
    /// </summary>
    internal static Tensor[]? MatMulPackedManyRecorded(Tensor input, PackedFormat format, IReadOnlyList<Layers.Linear> layers)
    {
        Tensor[]? outputs;
        using (Autograd.NoGrad())
        {
            outputs = MatMulPackedMany(input, format, layers);
        }

        if (outputs is null || !WillRecord(input))
        {
            return outputs;
        }

        int k = input._shape[^1], m = input.Size / k;
        for (int j = 0; j < outputs.Length; j++)
        {
            var layer = layers[j];
            int n = layer.OutFeatures;
            outputs[j].Record("matmul_packed", g =>
            {
                var weight = layer.PackedWeight!;
                if (weight.TransposedProduct(g, input.GradStorage(), m, k, n, 1f, null, null, 0))
                {
                    return;                                                                        // dx += g · Wᵀ, W as stored
                }

                using var w = weight.Dequantize();
                input.Backend.BatchedMatMul(g.Storage, w.Storage, input.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += g · wᵀ
            }, input);
        }

        return outputs;
    }

    /// <summary>
    /// x · Wᵀ for a frozen <paramref name="weight"/> W [outputs, inputs] also kept as <paramref name="transposed"/> Wᵀ
    /// [inputs, outputs]: the product reads Wᵀ as stored and the input's gradient (g · W) reads W as stored, so neither
    /// direction makes a transposed copy of W (the tensor-core kernels copy transposed operands into place first).
    /// W receives no gradient.
    /// </summary>
    internal static Tensor MatMulFrozenTransposed(Tensor input, Tensor weight, Layers.BFloat16Weight transposed)
    {
        Tensor output;
        using (Autograd.NoGrad())
        {
            output = input.MatMulBFloat16(transposed);
        }

        if (WillRecord(input))
        {
            int outputs = weight._shape[0], inputs = weight._shape[1];
            output.Record("matmul_frozen", g =>
            {
                // dx += g · W, accumulated by the product itself (no temporary).
                input.Backend.BatchedMatMul(g.Storage, weight.Storage, input.GradStorage(), 1, g.Size / outputs, inputs, outputs, false, false, 1f);
            }, input);
        }

        return output;
    }

    /// <summary>A transposed copy of a 2-D tensor that no <see cref="TensorScope"/> releases; the caller owns it.</summary>
    internal static Tensor TransposedCopy(Tensor weight)
    {
        var scope = new TensorScope();
        Tensor copy;
        using (Autograd.NoGrad())
        {
            copy = weight.Transpose();
        }

        foreach (var other in scope.Detach().Where(t => !ReferenceEquals(t, copy)))
        {
            other.Dispose();
        }

        return copy;
    }

    /// <summary>
    /// <see cref="TokenCrossEntropy"/> over only the rows listed in <paramref name="rows"/> (the positions with a non-zero
    /// weight): the head and the softmax run on those rows alone, gathered from <paramref name="hidden"/> chunk by chunk,
    /// and their gradients are scattered back. <paramref name="targets"/> and <paramref name="weights"/> hold one value
    /// per listed row. Rows left out have zero loss and zero gradient, as they would with weight 0.
    /// </summary>
    internal static Tensor TokenCrossEntropyRows(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, float[] targets, float[] weights, float normalizer, int chunkRows)
    {
        if (targets.Length != rows.Length || weights.Length != rows.Length)
        {
            throw new ArgumentException("TokenCrossEntropyRows needs one target and weight per listed row.");
        }

        using var rowTensor = From(rows.Length == 0 ? [0f] : [.. rows.Select(r => (float)r)], [Math.Max(1, rows.Length)], hidden.Device);
        using var targetTensor = From(targets.Length == 0 ? [0f] : targets, [Math.Max(1, rows.Length)], hidden.Device);
        using var weightTensor = From(weights.Length == 0 ? [0f] : weights, [Math.Max(1, rows.Length)], hidden.Device);
        return TokenCrossEntropyRows(hidden, head, rowTensor, targetTensor, weightTensor, normalizer, chunkRows);
    }

    /// <summary>
    /// log p(targets[i] | position rows[i]) under <paramref name="head"/>(hidden) for each listed row of hidden [positions,
    /// dim], computed on the device (the head and a fused log-softmax over <paramref name="chunkRows"/> rows at a time):
    /// only the listed values come back, not the [rows, vocabulary] logits. Not recorded.
    /// </summary>
    internal static float[] TokenLogProbabilities(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, int[] targets, int chunkRows)
    {
        hidden.ThrowIfDisposed();
        if (hidden.Rank != 2 || targets.Length != rows.Length)
        {
            throw new ArgumentException("TokenLogProbabilities needs hidden [positions, dim] and one target per listed row.");
        }

        int total = hidden._shape[0], dim = hidden._shape[1], count = rows.Length;
        var result = new float[count];
        if (count == 0)
        {
            return result;
        }

        foreach (int row in rows)
        {
            if ((uint)row >= (uint)total)
            {
                throw new ArgumentOutOfRangeException(nameof(rows), $"Row {row} is outside the {total} positions.");
            }
        }

        chunkRows = Math.Max(1, chunkRows);
        var (device, backend) = (hidden.Device, hidden.Backend);
        using var noGrad = Autograd.NoGrad();
        for (int r0 = 0; r0 < count; r0 += chunkRows)
        {
            int n = Math.Min(chunkRows, count - r0);
            using var scope = new TensorScope();
            var index = From([.. rows.AsSpan(r0, n).ToArray().Select(r => (float)r)], [n], device);
            var chunkTargets = From([.. targets.AsSpan(r0, n).ToArray().Select(t => (float)t)], [n], device);
            var ones = Full([n], 1f, device);
            var chunk = Empty([n, dim], device);
            backend.Gather(hidden.Storage, index.Storage, chunk.Storage, n, dim, total);
            var logits = head(chunk);
            int vocabulary = logits._shape[^1];
            if (logits.Size != n * vocabulary)
            {
                throw new ArgumentException($"The head must map [{n}, {dim}] to [{n}, vocabulary], got {FormatShape(logits._shape)}.");
            }

            foreach (int t in targets.AsSpan(r0, n))
            {
                if ((uint)t >= (uint)vocabulary)
                {
                    throw new ArgumentOutOfRangeException(nameof(targets), $"Token {t} is outside the vocabulary of {vocabulary}.");
                }
            }

            var losses = Empty([n], device);
            backend.SoftmaxCrossEntropyRows(logits.Storage, chunkTargets.Storage, ones.Storage, losses.Storage, n, vocabulary, 1f);   // -log p
            var values = losses.ToArray();
            for (int i = 0; i < n; i++)
            {
                result[r0 + i] = -values[i];
            }
        }

        return result;
    }

    /// <summary>
    /// <see cref="TokenCrossEntropyRows(Tensor, Func{Tensor, Tensor}, int[], float[], float[], float, int)"/> with the rows,
    /// targets and weights already on the device ([count] floats each; a row with weight 0 adds nothing): no host data, so
    /// the pass can be recorded as a graph and replayed with new values in those tensors.
    /// </summary>
    internal static Tensor TokenCrossEntropyRows(Tensor hidden, Func<Tensor, Tensor> head, Tensor rows, Tensor targets, Tensor weights, float normalizer, int chunkRows)
    {
        hidden.ThrowIfDisposed();
        if (hidden.Rank != 2 || targets.Size != rows.Size || weights.Size != rows.Size)
        {
            throw new ArgumentException("TokenCrossEntropyRows needs hidden [rows, dim] and one target and weight per listed row.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int total = hidden._shape[0], dim = hidden._shape[1], count = rows.Size;
        chunkRows = Math.Max(1, chunkRows);
        var device = hidden.Device;
        var backend = hidden.Backend;
        bool record = WillRecord(hidden);
        // The gradient of the listed rows only ([count, dim], not [total, dim]): scattered into the hidden gradient in the
        // backward pass, with the positions they came from.
        var gradient = record ? Empty([count, dim], device, zeroed: true) : null;
        var gradientRows = record ? Empty([count], device) : null;
        if (gradientRows is not null)
        {
            backend.Copy2D(rows.Storage, 0, count, gradientRows.Storage, 0, count, 1, count, accumulate: false);
        }

        var losses = Empty([count], device, zeroed: true);
        for (int r0 = 0; r0 < count; r0 += chunkRows)
        {
            int n = Math.Min(chunkRows, count - r0);
            using var scope = new TensorScope();
            var index = Empty([n], device);
            var chunkTargets = Empty([n], device);
            var chunkWeights = Empty([n], device);
            backend.Copy2D(rows.Storage, r0, n, index.Storage, 0, n, 1, n, accumulate: false);
            backend.Copy2D(targets.Storage, r0, n, chunkTargets.Storage, 0, n, 1, n, accumulate: false);
            backend.Copy2D(weights.Storage, r0, n, chunkWeights.Storage, 0, n, 1, n, accumulate: false);
            var chunk = Empty([n, dim], device);
            backend.Gather(hidden.Storage, index.Storage, chunk.Storage, n, dim, total);
            chunk.RequiresGrad = record;
            var logits = head(chunk);
            int vocabulary = logits._shape[^1];
            if (logits.Size != n * vocabulary)
            {
                throw new ArgumentException($"The head must map [{n}, {dim}] to [{n}, vocabulary], got {FormatShape(logits._shape)}.");
            }

            var chunkLosses = Empty([n], device);
            backend.SoftmaxCrossEntropyRows(logits.Storage, chunkTargets.Storage, chunkWeights.Storage, chunkLosses.Storage, n, vocabulary, 1f / normalizer);
            backend.Copy2D(chunkLosses.Storage, 0, n, losses.Storage, r0, n, 1, n, accumulate: false);
            if (record && logits.RequiresGrad)
            {
                logits.Backward(logits);                                         // the logits now hold their gradient
                backend.Copy2D(chunk.GradStorage(), 0, dim, gradient!.Storage, r0 * dim, dim, n, dim, accumulate: false);
            }
        }

        var loss = Empty([1], device);
        backend.Sum(losses.Storage, loss.Storage, count, 1f / normalizer);
        if (record)
        {
            // dhidden[rows] += gradient · g, with g read on the device (no host read: recordable as a graph).
            loss.Record("token_cross_entropy", g =>
            {
                backend.GroupScaleShift(gradient!.Storage, g.Storage, null, gradient.Storage, count * dim, 1, count * dim, false);
                backend.ScatterAdd(gradient.Storage, gradientRows!.Storage, hidden.GradStorage(), count, dim, total);
            }, hidden);
        }

        return Traced("token_cross_entropy", loss, start);
    }

    /// <summary>
    /// (act(gate) · up) · W for a packed layer W with few rows, the activation applied as the input is read (not recorded;
    /// no bias), or null when the layer is not packed or the device has no fused version.
    /// </summary>
    internal static Tensor? MatMulPackedGated(Tensor gate, Tensor up, int activation, Layers.Linear layer)
    {
        using var offload = Offloading.Enter(layer, gate);              // offloaded weights staged (else nothing)
        gate.ThrowIfDisposed();
        up.ThrowIfDisposed();
        int k = gate._shape[^1], m = gate.Size / Math.Max(1, k);
        if (layer.PackedWeight is not { Format: { } format } weight || k != layer.InFeatures || up.Size != gate.Size || m > gate.Backend.Capabilities.FewRows)
        {
            return null;                                                // not packed, or a format the kernels do not read
        }

        var (packed, scales) = (weight.PackedValues, weight.ScaleValues);
        if (packed.Device != gate.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([.. gate._shape[..^1], layer.OutFeatures], gate.Device);
        if (!gate.Backend.PackedMatMulGated(format, activation, gate.Storage, up.Storage, packed.Storage, scales?.Storage, y.Storage, m, layer.OutFeatures, k))
        {
            y.Dispose();
            return null;
        }

        return Traced("matmul_gated_packed", y, start);
    }

    /// <summary>
    /// residual + x · W for a packed layer W with few rows (no bias, no adapter) and the RMS normalization of that sum
    /// with gain, in one device pass (inference: not recorded): the output projection of a block, its residual addition
    /// and the next normalization. Null when the layer or the shapes do not fit or the device has no fused version.
    /// </summary>
    internal static (Tensor Sum, Tensor Normalized)? MatMulPackedAddRmsNorm(Tensor x, Layers.Linear layer, Tensor residual, Layers.RMSNorm norm)
    {
        using var offload = Offloading.EnterMany([layer, norm], x);
        x.ThrowIfDisposed();
        residual.ThrowIfDisposed();
        int k = x._shape[^1], m = x.Size / Math.Max(1, k), n = layer.OutFeatures;
        if (layer.PackedWeight is not { Format: { } format } weight || layer.Bias is not null || layer.Adapter is not null || k != layer.InFeatures || m > x.Backend.Capabilities.FewRows
            || residual.Size != m * n || residual._shape[^1] != n || norm.Features != n || !x.Backend.Capabilities.FusedKernels)
        {
            return null;
        }

        var (packed, scales) = (weight.PackedValues, weight.ScaleValues);
        if (packed.Device != x.Device || residual.Device != x.Device || norm.Gain.Device != x.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(residual._shape, x.Device, track: false);
        var sum = Empty(residual._shape, x.Device);
        var normalized = Empty(residual._shape, x.Device);
        bool done = x.Backend.PackedMatMulAddRmsNorm(format, x.Storage, packed.Storage, scales?.Storage, y.Storage, m, n, k, residual.Storage, sum.Storage,
            norm.Gain.Storage, normalized.Storage, norm.Epsilon, norm.Offset);
        y.Dispose();                                                 // stream-ordered: freed after the kernel read it
        if (!done)
        {
            sum.Dispose();
            normalized.Dispose();
            return null;
        }

        return (sum, Traced("matmul_add_rms_norm_packed", normalized, start));
    }

    /// <summary>
    /// act(input · gate) · (input · up) for packed gate and up layers of one kind with few rows (no biases or adapters),
    /// the activation applied by the product's own last blocks (inference: not recorded), or null when the layers do
    /// not fit or the device has no fused version.
    /// </summary>
    internal static Tensor? MatMulPackedGatedPair(Tensor input, Layers.Linear gate, Layers.Linear up, int activation)
    {
        using var offload = Offloading.EnterMany([gate, up], input);
        input.ThrowIfDisposed();
        int k = input._shape[^1], m = input.Size / Math.Max(1, k), n = gate.OutFeatures;
        if (gate.PackedWeight is not { Format: { } format } gateWeight || up.PackedWeight is not { } upWeight || upWeight.Format != format
            || m > input.Backend.Capabilities.FewRows || !input.Backend.Capabilities.FusedKernels || up.OutFeatures != n
            || gate.InFeatures != k || up.InFeatures != k || gate.Bias is not null || up.Bias is not null || gate.Adapter is not null
            || up.Adapter is not null)
        {
            return null;
        }

        var (gatePacked, gateScales) = (gateWeight.PackedValues, gateWeight.ScaleValues);
        var (upPacked, upScales) = (upWeight.PackedValues, upWeight.ScaleValues);
        if (gatePacked.Device != input.Device || upPacked.Device != input.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var gateOut = Empty([m * n], input.Device, track: false);
        var upOut = Empty([m * n], input.Device, track: false);
        var hidden = Empty([.. input._shape[..^1], n], input.Device);
        bool done = input.Backend.PackedMatMulGatedPair(format, activation, input.Storage, m, k,
            [(gatePacked.Storage, gateScales?.Storage, null, gateOut.Storage, n), (upPacked.Storage, upScales?.Storage, null, upOut.Storage, n)],
            hidden.Storage);
        gateOut.Dispose();                                           // stream-ordered: freed after the kernel read them
        upOut.Dispose();
        if (!done)
        {
            hidden.Dispose();
            return null;
        }

        return Traced("matmul_gated_pair_packed", hidden, start);
    }

    /// <summary>
    /// The layers' packed products of one input in one device pass (few rows, not recorded), or null when the device has
    /// no single-pass version.
    /// </summary>
    internal static Tensor[]? MatMulPackedMany(Tensor input, PackedFormat format, IReadOnlyList<Layers.Linear> layers)
    {
        input.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int k = input._shape[^1], m = input.Size / k;
        var outputs = new Tensor[layers.Count];
        var products = new (Abstraction.Devices.Storage, Abstraction.Devices.Storage?, Abstraction.Devices.Storage?, Abstraction.Devices.Storage, int)[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var layer = layers[j];
            var weight = layer.PackedWeight!;                                  // the caller checked: one format for all
            if (weight.Format != format)
            {
                return null;                                                   // never reached: a format the kernels do not read
            }

            var (packed, scales) = (weight.PackedValues, weight.ScaleValues);
            if (packed.Device != input.Device)
            {
                return null;
            }

            outputs[j] = Empty([.. input._shape[..^1], layer.OutFeatures], input.Device);
            products[j] = (packed.Storage, scales?.Storage, layer.Bias?.Storage, outputs[j].Storage, layer.OutFeatures);
        }

        if (!input.Backend.PackedMatMulMany(format, input.Storage, m, k, products))
        {
            foreach (var output in outputs)
            {
                output.Dispose();
            }

            return null;
        }

        foreach (var output in outputs)
        {
            Traced("matmul_many_packed", output, start);
        }

        return outputs;
    }

    /// <summary>x / sqrt(mean(x²) + eps) · (gain + offset) over the last dimension in one pass (inference: not recorded).</summary>
    internal Tensor RmsNormAffine(Tensor gain, float eps, float offset)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = _shape[^1], rows = Size / cols;
        var y = Empty(_shape, Device);
        Backend.RmsNormAffine(Storage, gain.Storage, y.Storage, rows, cols, eps, offset);
        return Traced("rms_norm_affine", y, start);
    }

    /// <summary>a + b (a residual addition) and its RMS normalization with gain, in one pass (inference: not recorded).</summary>
    internal static (Tensor Sum, Tensor Normalized) AddRmsNormAffine(Tensor a, Tensor b, Tensor gain, float eps, float offset)
    {
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        CheckSameDevice(a, b);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = a._shape[^1], rows = a.Size / cols;
        var sum = Empty(a._shape, a.Device);
        var y = Empty(a._shape, a.Device);
        a.Backend.AddRmsNormAffine(a.Storage, b.Storage, sum.Storage, gain.Storage, y.Storage, rows, cols, eps, offset);
        return (sum, Traced("add_rms_norm", y, start));
    }

    /// <summary>
    /// RMS normalization with gain of each head's vector of this [batch, steps, heads, dim] tensor, then the rotary
    /// embedding (see <see cref="Rope"/>), in one pass (inference: not recorded).
    /// </summary>
    internal Tensor RmsNormRope(Tensor gain, float eps, float offset, Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = _shape[1], heads = _shape[2], dim = _shape[3], rows = Size / dim;
        var y = Empty(_shape, Device);
        Backend.RmsNormRope(Storage, gain.Storage, cos.Storage, sin.Storage, positions.Storage, y.Storage, rows, dim, eps, offset, heads, steps,
            half, interleaved);
        return Traced("rms_norm_rope", y, start);
    }

    /// <summary><see cref="RmsNormRope"/> of the queries and the keys (same positions and tables) in one pass where the device can.</summary>
    internal static (Tensor Q, Tensor K) RmsNormRopePair(Tensor q, Tensor gainQ, float epsQ, float offsetQ, Tensor k, Tensor gainK, float epsK,
        float offsetK, Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        q.ThrowIfDisposed();
        k.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = q._shape[1], dim = q._shape[3];
        var yq = Empty(q._shape, q.Device);
        var yk = Empty(k._shape, k.Device);
        q.Backend.RmsNormRopePair(q.Storage, gainQ.Storage, yq.Storage, q.Size / dim, epsQ, offsetQ, q._shape[2],
            k.Storage, gainK.Storage, yk.Storage, k.Size / dim, epsK, offsetK, k._shape[2], cos.Storage, sin.Storage, positions.Storage,
            dim, steps, half, interleaved);
        Traced("rms_norm_rope", yq, start);
        return (yq, Traced("rms_norm_rope", yk, start));
    }

    /// <summary>
    /// An attention layer's projections q [batch, steps, heads·dim], k and v [batch, steps, kvHeads·dim] in the layouts
    /// attention reads, in one pass where the device can (inference: not recorded): each query and key head
    /// RMS-normalized with gain (when the norms are given) and rotated (when the tables are given), queries as
    /// [batch·kvHeads, group·steps, dim] (the heads sharing a key/value head stacked), keys and values as
    /// [batch·kvHeads, steps, dim], or, with a cache, written into it at <paramref name="position"/> (then K and V are
    /// null). Null when the device has no fused version.
    /// </summary>
    internal static (Tensor Q, Tensor? K, Tensor? V)? NormRopeHeads(Tensor q, Tensor k, Tensor v, int heads, int kvHeads, int dim,
        Layers.RMSNorm? queryNorm, Layers.RMSNorm? keyNorm, Tensor? cos, Tensor? sin, Tensor positions, bool interleaved,
        Layers.KeyValueCache? cache, Tensor? position)
    {
        q.ThrowIfDisposed();
        k.ThrowIfDisposed();
        v.ThrowIfDisposed();
        if ((queryNorm is null) != (keyNorm is null) || (cos is null) != (sin is null) || (cache is null) != (position is null)
            || cache is { Layout.FusedWrite: false } || !q.Backend.Capabilities.FusedKernels)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int n = q._shape[0], t = q._shape[1];
        var yq = Empty([n * kvHeads, heads / kvHeads * t, dim], q.Device);
        Tensor? yk = null, yv = null;
        int capacity = t, stride = dim;
        bool bfloat16 = cache is { Layout.HalfWords: true };
        if (cache is null)
        {
            yk = Empty([n * kvHeads, t, dim], q.Device);
            yv = Empty([n * kvHeads, t, dim], q.Device);
        }
        else
        {
            capacity = cache.Keys._shape[1];
            stride = bfloat16 ? 2 * cache.Keys._shape[2] : cache.Keys._shape[2];
        }

        if (!q.Backend.NormRopeHeads(q.Storage, k.Storage, v.Storage, n, t, heads, kvHeads, dim, queryNorm?.Gain.Storage, queryNorm?.Epsilon ?? 0f,
            queryNorm?.Offset ?? 0f, keyNorm?.Gain.Storage, keyNorm?.Epsilon ?? 0f, keyNorm?.Offset ?? 0f, cos?.Storage, sin?.Storage, positions.Storage,
            cos?._shape[1] ?? 0, interleaved, yq.Storage, (yk ?? cache!.Keys).Storage, (yv ?? cache!.Values).Storage, position?.Storage, capacity, stride,
            bfloat16))
        {
            yq.Dispose();
            yk?.Dispose();
            yv?.Dispose();
            return null;
        }

        Traced("norm_rope_heads", yq, start);
        return (yq, yk, yv);
    }

    /// <summary>act(gate) · up element-wise (kind 0 = SiLU, 1 = GELU, 2 = ReLU), with its gradient: one pass either way.</summary>
    internal static Tensor GatedActivation(Tensor gate, Tensor up, int kind)
    {
        gate.ThrowIfDisposed();
        up.ThrowIfDisposed();
        CheckSameDevice(gate, up);
        if (!gate._shape.AsSpan().SequenceEqual(up._shape))
        {
            throw new ArgumentException($"Gate {FormatShape(gate._shape)} and up {FormatShape(up._shape)} differ.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(gate._shape, gate.Device);
        gate.Backend.GatedActivation(gate.Storage, up.Storage, y.Storage, gate.Size, kind);
        RecordGatedActivation(gate, up, y, kind);
        return Traced("gated_activation", y, start);
    }

    /// <summary>
    /// <see cref="GatedActivation(Tensor, Tensor, int)"/> for training under <see cref="ActivationMemory.CompressToBFloat16"/>:
    /// the one kernel also writes gate and up as bfloat16 words, and evicts them to those (the backward kernel reads the
    /// words as they are, unpacking nothing), and with <paramref name="packOutput"/> writes the output's words too, for the
    /// caller to evict the output to (<see cref="EvictToPacked"/>) once its forward uses are done. Null when gate and up
    /// share memory (views of one product).
    /// </summary>
    internal static Tensor? GatedActivationCompressed(Tensor gate, Tensor up, int kind, bool packOutput, out Storage? packedOutput)
    {
        packedOutput = null;
        gate.ThrowIfDisposed();
        up.ThrowIfDisposed();
        int n = gate.Size;
        if (!gate._shape.AsSpan().SequenceEqual(up._shape) || ReferenceEquals(gate.Storage, up.Storage) || gate.Device != up.Device
            || gate.Storage.Length != n || up.Storage.Length != n || gate.Storage.Evicted || up.Storage.Evicted || n < 2)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var backend = gate.Backend;
        var y = Empty(gate._shape, gate.Device);
        var (packedGate, packedUp) = (backend.Allocate((n + 1) / 2, zeroed: false), backend.Allocate((n + 1) / 2, zeroed: false));
        packedOutput = packOutput ? backend.Allocate((n + 1) / 2, zeroed: false) : null;
        backend.GatedActivationPacked(gate.Storage, up.Storage, packedGate, packedUp, y.Storage, packedOutput ?? y.Storage, n, kind, 2 | 4 | (packOutput ? 8 : 0));
        RecordGatedActivation(gate, up, y, kind);
        gate.EvictToPacked(packedGate);                                  // read again only by the backward kernel, as words
        up.EvictToPacked(packedUp);
        return Traced("gated_activation", y, start);
    }

    /// <summary>
    /// Recomputes act(gate) · up into <paramref name="y"/> for <see cref="Evict"/>: from the bfloat16 words when gate and up
    /// were evicted to them, else from their values.
    /// </summary>
    internal static void RecomputeGatedActivation(Tensor gate, Tensor up, Tensor y, int kind)
    {
        var (gs, us) = (gate.Storage, up.Storage);
        if (gs is { Evicted: true, Packed: { } packedGate } && us is { Evicted: true, Packed: { } packedUp })
        {
            y.Backend.GatedActivationPacked(y.Storage, y.Storage, packedGate, packedUp, y.Storage, y.Storage, y.Size, kind, 1 | 4);
            return;
        }

        WithValues([gate, up], () => y.Backend.GatedActivation(gate.Storage, up.Storage, y.Storage, y.Size, kind));
    }

    private static void RecordGatedActivation(Tensor gate, Tensor up, Tensor y, int kind)
    {
        if (!WillRecord(gate, up))
        {
            return;
        }

        y.Record("gated_activation", g =>
        {
            // Bits 2 and 3: the first gradient of gate / up, written rather than added.
            int flags = (gate.RequiresGrad ? 1 : 0) | (up.RequiresGrad ? 2 : 0);
            float betaGate = 1f, betaUp = 1f;
            var dgate = gate.RequiresGrad ? gate.GradientTarget(out betaGate) : g.Storage;
            var dup = up.RequiresGrad ? up.GradientTarget(out betaUp) : g.Storage;
            flags |= (betaGate == 0f ? 4 : 0) | (betaUp == 0f ? 8 : 0);
            if (gate.Storage is { Evicted: true, Packed: { } packedGate } && up.Storage is { Evicted: true, Packed: { } packedUp })
            {
                gate.Backend.GatedActivationBackwardPacked(packedGate, packedUp, g.Storage, dgate, dup, gate.Size, kind, flags);   // read as words
                return;
            }

            WithValues([gate, up], () => gate.Backend.GatedActivationBackward(gate.Storage, up.Storage, g.Storage, dgate, dup, gate.Size, kind, flags));
        }, gate, up);
    }

    /// <summary>
    /// Rotary position embedding of this [batch, steps, heads, dim] tensor: pair p of each head's vector at step t
    /// rotates by the angle with cos/sin[positions[t], p] ([positions, half] tables). Dimensions beyond 2·half pass through.
    /// </summary>
    internal Tensor Rope(Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        ThrowIfDisposed();
        if (Rank != 4)
        {
            throw new ArgumentException($"Rotary embedding expects [batch, steps, heads, dim], got {FormatShape(_shape)}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = _shape[1], heads = _shape[2], dim = _shape[3], rows = Size / dim;
        var y = Empty(_shape, Device);
        if (2 * half < dim)
        {
            Backend.Copy(Storage, y.Storage, Size);                        // dimensions beyond the rotated pairs pass through
        }

        Backend.Rope(Storage, y.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, 1f);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("rope", g =>
            {
                using var back = Empty(x._shape, x.Device, track: false);
                if (2 * half < dim)
                {
                    x.Backend.Copy(g.Storage, back.Storage, x.Size);
                }

                x.Backend.Rope(g.Storage, back.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, -1f);
                x.AddGradient(back, adopt: true);
            }, x);
        }

        return Traced("rope", y, start);
    }
}
