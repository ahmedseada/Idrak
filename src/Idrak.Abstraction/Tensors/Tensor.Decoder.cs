// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

// Operations of decoder-only language model layers.
public sealed partial class Tensor
{    /// <summary>x / sqrt(mean(x²) + eps) over the last dimension (RMS normalization without a gain).</summary>
    internal Tensor RmsNormalize(float eps)
    {
        ThrowIfDisposed();
        long start = OperationTelemetry.Start();
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
        long start = OperationTelemetry.Start();
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

        long start = OperationTelemetry.Start();
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

        long start = OperationTelemetry.Start();
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

        long start = OperationTelemetry.Start();
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

    /// <summary>x / sqrt(mean(x²) + eps) · (gain + offset) over the last dimension in one pass (inference: not recorded).</summary>
    internal Tensor RmsNormAffine(Tensor gain, float eps, float offset)
    {
        ThrowIfDisposed();
        long start = OperationTelemetry.Start();
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
        long start = OperationTelemetry.Start();
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
        long start = OperationTelemetry.Start();
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
        long start = OperationTelemetry.Start();
        int steps = q._shape[1], dim = q._shape[3];
        var yq = Empty(q._shape, q.Device);
        var yk = Empty(k._shape, k.Device);
        q.Backend.RmsNormRopePair(q.Storage, gainQ.Storage, yq.Storage, q.Size / dim, epsQ, offsetQ, q._shape[2],
            k.Storage, gainK.Storage, yk.Storage, k.Size / dim, epsK, offsetK, k._shape[2], cos.Storage, sin.Storage, positions.Storage,
            dim, steps, half, interleaved);
        Traced("rms_norm_rope", yq, start);
        return (yq, Traced("rms_norm_rope", yk, start));
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

        long start = OperationTelemetry.Start();
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

        long start = OperationTelemetry.Start();
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

        long start = OperationTelemetry.Start();
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
