using Idrak.Diagnostics;

namespace Idrak;

// Fused inference-only operations and incremental-decoding primitives. None of these record gradients:
// callers use them only when autograd is off (inference), and fall back to the differentiable ops otherwise.
public sealed partial class Tensor
{
    /// <summary>Copies <paramref name="destination"/>.Length elements starting at element <paramref name="offset"/> to the host.</summary>
    public void CopyTo(Span<float> destination, int offset = 0)
    {
        ThrowIfDisposed();
        if (offset < 0 || offset + destination.Length > Size)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"[{offset}, {offset + destination.Length}) is outside a tensor of {Size} elements.");
        }

        Backend.DownloadRange(Storage, offset, destination);
    }

    /// <summary>softmax(scale · x + mask) over the last dimension in one kernel; the mask's rows repeat over the rows of x.</summary>
    internal Tensor ScaleMaskSoftmax(float scale, Tensor? mask)
    {
        ThrowIfDisposed();
        int cols = _shape[^1], rows = Size / cols;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.ScaleMaskSoftmax(Storage, mask?.Storage, y.Storage, rows, cols, mask is null ? 1 : mask.Size / cols, scale);
        return Traced("scale_mask_softmax", y, start);
    }

    /// <summary>LayerNorm over the last dimension with gamma/beta, in one kernel.</summary>
    internal Tensor LayerNormFused(Tensor gamma, Tensor beta, float eps)
    {
        ThrowIfDisposed();
        int cols = _shape[^1];
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.LayerNormFused(Storage, gamma.Storage, beta.Storage, y.Storage, Size / cols, cols, eps);
        return Traced("layernorm_fused", y, start);
    }

    /// <summary>LayerNorm over the last dimension for training: one forward kernel (keeping each row's mean and 1 / std) and one backward pass.</summary>
    internal Tensor LayerNormTrain(Tensor gamma, Tensor beta, float eps)
    {
        ThrowIfDisposed();
        int cols = _shape[^1], rows = Size / cols;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        var stats = Empty([2 * rows], Device, track: false);                 // kept for the backward pass, which frees it
        Backend.LayerNormTrain(Storage, gamma.Storage, beta.Storage, y.Storage, stats.Storage, rows, cols, eps);
        if ((RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad) && Autograd.IsEnabled)
        {
            var x = this;
            y.Record("layernorm", g =>
            {
                x.Backend.LayerNormBackward(x.Storage, gamma.Storage, g.Storage, stats.Storage,
                    x.RequiresGrad ? x.GradStorage() : null, gamma.RequiresGrad ? gamma.GradStorage() : null, beta.RequiresGrad ? beta.GradStorage() : null,
                    rows, cols);
                stats.Dispose();
            }, x, gamma, beta);
        }
        else
        {
            stats.Dispose();
        }

        return Traced("layernorm", y, start);
    }

    /// <summary>gelu(x + bias) over the last dimension, in one kernel.</summary>
    internal Tensor BiasGelu(Tensor bias)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.BiasGelu(Storage, bias.Storage, y.Storage, Size, bias.Size);
        return Traced("bias_gelu", y, start);
    }

    /// <summary>[rows, capacity] causal mask for queries at positions position..position+rows-1 (position read on the device).</summary>
    internal static Tensor DecoderMask(Tensor position, int rows, int capacity, bool track = true)
    {
        var mask = Empty([rows, capacity], position.Device, track: track);
        position.Backend.DecoderMask(position.Storage, mask.Storage, rows, capacity);
        return mask;
    }

    /// <summary>Writes [heads, steps, dim] keys or values into a [heads, capacity, dim] cache at the device-side position.</summary>
    internal static void WriteKeyValues(Tensor source, Tensor cache, Tensor position)
    {
        int heads = source._shape[0], steps = source._shape[1], dim = source._shape[2];
        source.Backend.KeyValueWrite(source.Storage, cache.Storage, position.Storage, heads, steps, cache._shape[1], dim);
    }

    /// <summary>Quantizes <paramref name="source"/> [rows, steps, dim] into an int8 cache at the device position.</summary>
    internal static void WriteKeyValuesInt8(Tensor source, Tensor cache, Tensor scales, Tensor position)
    {
        int heads = source._shape[0], steps = source._shape[1], dim = source._shape[2];
        source.Backend.KeyValueWriteInt8(source.Storage, cache.Storage, scales.Storage, position.Storage, heads, steps, cache._shape[1], dim);
    }

    /// <summary>
    /// Attention of q [heads, rowsPerHead, dim] over a float cache filled up to <paramref name="position"/> (see
    /// Backend.AttentionDecode): the softmax and the weighted values in one pass, reading only filled positions.
    /// </summary>
    internal static Tensor AttentionDecode(Tensor q, Layers.KeyValueCache cache, Tensor position, int steps, float scale)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Empty([heads, rowsPerHead, dim], q.Device);
        q.Backend.AttentionDecode(q.Storage, cache.Keys.Storage, cache.Values.Storage, position.Storage, y.Storage, heads, rowsPerHead, steps,
            cache.Keys._shape[1], dim, scale);
        return Traced("attention_decode", y, start);
    }

    /// <summary>
    /// Attention of q [heads, rowsPerHead, dim] over keys and values [heads, capacity, dim] up to the causal limit
    /// position[0] + (row % steps), tiled for many query rows (a prompt or a whole sequence; see Backend.AttentionTiled).
    /// </summary>
    internal static Tensor AttentionTiled(Tensor q, Tensor keys, Tensor values, Tensor position, int steps, float scale)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Empty([heads, rowsPerHead, dim], q.Device);
        q.Backend.AttentionTiled(q.Storage, keys.Storage, values.Storage, position.Storage, y.Storage, null, heads, rowsPerHead, steps,
            keys._shape[1], dim, scale);
        return Traced("attention_tiled", y, start);
    }

    /// <summary>
    /// Causal attention of q [heads, rowsPerHead, dim] over keys and values [heads, steps, dim] (row i sees positions
    /// ≤ i % steps) with its gradient, without storing attention weights: the backward pass recomputes them from each
    /// row's log-sum-exp (FlashAttention-style). Memory grows with the sequence length, not its square.
    /// </summary>
    internal static Tensor CausalAttention(Tensor q, Tensor keys, Tensor values, Tensor zero, int steps, float scale)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2], capacity = keys._shape[1];
        var y = Empty([heads, rowsPerHead, dim], q.Device);
        bool record = Autograd.IsEnabled && (q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad);
        var lse = record ? Empty([heads, rowsPerHead], q.Device, track: false) : null;
        q.Backend.AttentionTiled(q.Storage, keys.Storage, values.Storage, zero.Storage, y.Storage, lse?.Storage, heads, rowsPerHead, steps,
            capacity, dim, scale);
        if (record)
        {
            y.Record("attention", g =>
            {
                // Gradients go to scratch buffers for inputs that do not need them.
                using var dq = q.RequiresGrad ? null : Empty(q._shape, q.Device, zeroed: true, track: false);
                using var dk = keys.RequiresGrad ? null : Empty(keys._shape, q.Device, zeroed: true, track: false);
                using var dv = values.RequiresGrad ? null : Empty(values._shape, q.Device, zeroed: true, track: false);
                q.Backend.AttentionTiledBackward(q.Storage, keys.Storage, values.Storage, y.Storage, lse!.Storage, g.Storage,
                    dq?.Storage ?? q.GradStorage(), dk?.Storage ?? keys.GradStorage(), dv?.Storage ?? values.GradStorage(),
                    heads, rowsPerHead, steps, capacity, dim, scale);
                lse.Dispose();
            }, q, keys, values);
        }

        return Traced("attention", y, start);
    }

    /// <summary>
    /// <see cref="AttentionTiled"/> for rows of different lengths: head h's row i sees cached positions c with
    /// starts[(h / headsPerRow)·steps + i % steps] ≤ c ≤ position[0] + i % steps. Null when the device has no such pass.
    /// </summary>
    internal static Tensor? AttentionRows(Tensor q, Tensor keys, Tensor values, Tensor position, int steps, float scale, Tensor starts, int headsPerRow)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Empty([heads, rowsPerHead, dim], q.Device);
        if (!q.Backend.AttentionRows(q.Storage, keys.Storage, values.Storage, position.Storage, y.Storage, starts.Storage, heads, headsPerRow, rowsPerHead,
                steps, keys._shape[1], dim, scale))
        {
            y.Dispose();
            return null;
        }

        return Traced("attention_rows", y, start);
    }

    /// <summary>
    /// <see cref="CausalAttention"/> over packed sequences: keys and values [heads, steps, dim], and each position sees
    /// only its own sequence (<paramref name="packing"/>'s starts; head h belongs to packed row h / <paramref name="headsPerRow"/>).
    /// Null when the device has no such pass.
    /// </summary>
    internal static Tensor? CausalAttentionSegmented(Tensor q, Tensor keys, Tensor values, Layers.PackedSequences packing, int headsPerRow, float scale)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2], steps = keys._shape[1];
        var y = Empty([heads, rowsPerHead, dim], q.Device);
        bool record = Autograd.IsEnabled && (q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad);
        var lse = record ? Empty([heads, rowsPerHead], q.Device, track: false) : null;
        if (!q.Backend.AttentionSegmented(q.Storage, keys.Storage, values.Storage, y.Storage, lse?.Storage, packing.Starts.Storage, packing.Ends.Storage,
                heads, headsPerRow, rowsPerHead, steps, dim, scale))
        {
            y.Dispose();
            lse?.Dispose();
            return null;
        }

        if (record)
        {
            y.Record("attention_packed", g =>
            {
                using var dq = q.RequiresGrad ? null : Empty(q._shape, q.Device, zeroed: true, track: false);
                using var dk = keys.RequiresGrad ? null : Empty(keys._shape, q.Device, zeroed: true, track: false);
                using var dv = values.RequiresGrad ? null : Empty(values._shape, q.Device, zeroed: true, track: false);
                if (!q.Backend.AttentionSegmentedBackward(q.Storage, keys.Storage, values.Storage, y.Storage, lse!.Storage, g.Storage,
                        dq?.Storage ?? q.GradStorage(), dk?.Storage ?? keys.GradStorage(), dv?.Storage ?? values.GradStorage(),
                        packing.Starts.Storage, packing.Ends.Storage, heads, headsPerRow, rowsPerHead, steps, dim, scale))
                {
                    throw new NotSupportedException("Packed attention's backward pass is not available on this device.");
                }

                lse.Dispose();
            }, q, keys, values);
        }

        return Traced("attention_packed", y, start);
    }

    /// <summary>Attention of q [heads, rowsPerHead, dim] over an int8 cache filled up to <paramref name="position"/>.</summary>
    internal static Tensor AttentionInt8(Tensor q, Layers.KeyValueCache cache, Tensor position, int steps, float scale, bool tiled)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Empty([heads, rowsPerHead, dim], q.Device);
        q.Backend.AttentionInt8(q.Storage, cache.Keys.Storage, cache.Values.Storage, cache.KeyScales!.Storage, cache.ValueScales!.Storage,
            position.Storage, y.Storage, heads, rowsPerHead, steps, cache.Keys._shape[1], dim, scale, tiled);
        return Traced("attention_int8", y, start);
    }

    /// <summary>Writes [heads, steps, dim] keys or values into a bfloat16 cache at the device-side position.</summary>
    internal static void WriteKeyValuesBFloat16(Tensor source, Tensor cache, Tensor position, int dim)
    {
        int heads = source._shape[0], steps = source._shape[1];
        source.Backend.KeyValueWriteBFloat16(source.Storage, cache.Storage, position.Storage, heads, steps, cache._shape[1], dim);
    }

    /// <summary>Attention of q [heads, rows, dim] over the filled part of a bfloat16 cache.</summary>
    internal static Tensor AttentionBFloat16(Tensor q, Layers.KeyValueCache cache, Tensor position, int steps, float scale, bool tiled)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Empty([heads, rowsPerHead, dim], q.Device);
        q.Backend.AttentionBFloat16(q.Storage, cache.Keys.Storage, cache.Values.Storage, position.Storage, y.Storage, heads, rowsPerHead, steps,
            cache.Keys._shape[1], dim, scale, tiled);
        return Traced("attention_bf16", y, start);
    }

    /// <summary>q [rows, steps, dim] · int8 keysᵀ → [rows, steps, capacity].</summary>
    internal static Tensor AttentionScoresInt8(Tensor q, Layers.KeyValueCache cache)
    {
        int rows = q._shape[0], steps = q._shape[1], capacity = cache.Keys._shape[1];
        var y = Empty([rows, steps, capacity], q.Device);
        q.Backend.AttentionScoresInt8(q.Storage, cache.Keys.Storage, cache.KeyScales!.Storage, y.Storage, rows, steps, capacity, cache.HeadDim);
        return y;
    }

    /// <summary>weights [rows, steps, capacity] · int8 values → [rows, steps, dim].</summary>
    internal static Tensor AttentionContextInt8(Tensor weights, Layers.KeyValueCache cache)
    {
        int rows = weights._shape[0], steps = weights._shape[1], capacity = weights._shape[2];
        var y = Empty([rows, steps, cache.HeadDim], weights.Device);
        weights.Backend.AttentionContextInt8(weights.Storage, cache.Values.Storage, cache.ValueScales!.Storage, y.Storage, rows, steps, capacity, cache.HeadDim);
        return y;
    }

    /// <summary>Adds <paramref name="value"/> to every element in place (not recorded by autograd).</summary>
    internal void AddInPlace(float value)
    {
        ThrowIfDisposed();
        Backend.Affine(Storage, Storage, Size, 1f, value);
    }

    /// <summary>Sets every element in place (not recorded by autograd).</summary>
    internal void FillInPlace(float value)
    {
        ThrowIfDisposed();
        Backend.Fill(Storage, Size, value);
    }
}
