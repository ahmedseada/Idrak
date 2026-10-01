// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// Multi-head scaled dot-product self-attention over [batch, time, dim]: every position attends to every
/// other (or only to earlier ones when <see cref="Causal"/>). Heads run as one batched matrix product.
/// </summary>
public sealed class MultiHeadAttention : Module, ICachedModule
{
    private readonly Linear _qkv;
    private readonly Linear _output;
    private readonly Dropout? _dropout;
    private Tensor? _mask;

    /// <summary>Creates the layer.</summary>
    /// <param name="dim">Model width; must be divisible by <paramref name="heads"/>.</param>
    /// <param name="heads">Number of attention heads.</param>
    /// <param name="causal">Mask future positions (for autoregressive models).</param>
    /// <param name="dropout">Dropout on the attention weights.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for weights and dropout.</param>
    public MultiHeadAttention(int dim, int heads, bool causal = false, float dropout = 0f, Device? device = null, Random? random = null)
    {
        if (dim % heads != 0)
        {
            throw new ArgumentException($"dim ({dim}) must be divisible by heads ({heads}).");
        }

        Dim = dim;
        Heads = heads;
        Causal = causal;
        _qkv = new Linear(dim, 3 * dim, device: device, random: random);
        _output = new Linear(dim, dim, device: device, random: random);
        _dropout = dropout > 0f ? new Dropout(dropout, random) : null;
    }

    /// <summary>Model width.</summary>
    public int Dim { get; }

    /// <summary>Number of heads.</summary>
    public int Heads { get; }

    /// <summary>Whether future positions are masked.</summary>
    public bool Causal { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 3 || input.Shape[2] != Dim)
        {
            throw new ArgumentException($"MultiHeadAttention expects [batch, time, {Dim}], got {Tensor.FormatShape(input.Shape)}.");
        }

        int n = input.Shape[0], t = input.Shape[1], dh = Dim / Heads;
        var qkv = _qkv.Forward(input);                                    // [N, T, 3D]
        Tensor SplitHeads(int part) => qkv.Narrow(2, part * Dim, Dim)
            .Reshape(n, t, Heads, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n * Heads, t, dh);                                   // [N·H, T, dh]

        var q = SplitHeads(0);
        var k = SplitHeads(1);
        var v = SplitHeads(2);
        var raw = q.MatMul(k, transposeB: true);                           // [N·H, T, T]
        Tensor weights;
        if (!Autograd.IsEnabled)
        {
            // Inference: scale, mask and softmax in one kernel.
            weights = raw.ScaleMaskSoftmax(1f / MathF.Sqrt(dh), Causal ? CausalMask(t, input.Device) : null);
        }
        else
        {
            var scores = raw * (1f / MathF.Sqrt(dh));
            if (Causal)
            {
                scores = scores + CausalMask(t, input.Device);
            }

            weights = scores.Softmax();
        }

        if (_dropout is not null)
        {
            weights = _dropout.Forward(weights);
        }

        var context = weights.MatMul(v)                                   // [N·H, T, dh]
            .Reshape(n, Heads, t, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n, t, Dim);
        return _output.Forward(context);
    }

    /// <summary>
    /// Cached attention for the new positions of [batch, newSteps, dim]: their keys and values are appended to this
    /// layer's <see cref="KeyValueCache"/>, and each new query attends to every cached position up to its own
    /// (the causal mask comes from the context), so a decoding step costs O(capacity) instead of O(steps²).
    /// </summary>
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        int n = input.Shape[0], t = input.Shape[1], dh = Dim / Heads;
        var cache = context.CacheFor(this, n * Heads, dh);
        var qkv = _qkv.Forward(input);
        Tensor SplitHeads(int part) => qkv.Narrow(2, part * Dim, Dim)
            .Reshape(n, t, Heads, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n * Heads, t, dh);

        var q = SplitHeads(0);
        Tensor output;
        if (cache.Format == KeyValueFormat.BFloat16)
        {
            throw new NotSupportedException("A bfloat16 KV cache is supported by decoder models (DecoderSpec); use Float32 or Int8 here.");
        }

        if (cache.Format == KeyValueFormat.Int8)
        {
            Tensor.WriteKeyValuesInt8(SplitHeads(1), cache.Keys, cache.KeyScales!, context.Position);
            Tensor.WriteKeyValuesInt8(SplitHeads(2), cache.Values, cache.ValueScales!, context.Position);
            if (dh <= Backends.Cuda.PtxKernels.DecodeMaxDim)
            {
                // Only the filled positions (as decoder models do), not every slot of the cache behind a mask.
                output = Tensor.AttentionInt8(q, cache, context.Position, t, 1f / MathF.Sqrt(dh), tiled: t >= 8);
            }
            else
            {
                var weights8 = Tensor.AttentionScoresInt8(q, cache).ScaleMaskSoftmax(1f / MathF.Sqrt(dh), context.Mask);
                output = Tensor.AttentionContextInt8(weights8, cache);
            }
        }
        else
        {
            Tensor.WriteKeyValues(SplitHeads(1), cache.Keys, context.Position);
            Tensor.WriteKeyValues(SplitHeads(2), cache.Values, context.Position);
            var weights = q.MatMul(cache.Keys, transposeB: true)         // [N·H, t, capacity]
                .ScaleMaskSoftmax(1f / MathF.Sqrt(dh), context.Mask);   // unwritten positions are masked out
            output = weights.MatMul(cache.Values);                        // [N·H, t, dh]
        }

        output = output
            .Reshape(n, Heads, t, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n, t, Dim);
        return _output.Forward(output);
    }

    /// <summary>[T, T] with 0 on and below the diagonal and -1e9 above, cached per length and device.</summary>
    private Tensor CausalMask(int t, Device device)
    {
        if (_mask is not null && _mask.Shape[0] == t && _mask.Device == device)
        {
            return _mask;
        }

        _mask?.Dispose();
        var values = new float[t * t];
        for (int i = 0; i < t; i++)
        {
            for (int j = i + 1; j < t; j++)
            {
                values[i * t + j] = -1e9f;
            }
        }

        _mask = CreateBuffer(values, [t, t], device);
        return _mask;
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _dropout is null ? [_qkv, _output] : [_qkv, _output, _dropout];

    /// <inheritdoc />
    public override void Dispose()
    {
        _mask?.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"MultiHeadAttention(dim {Dim}, {Heads} heads{(Causal ? ", causal" : "")})";
}

/// <summary>
/// One pre-norm transformer encoder block over [batch, time, dim]:
/// x + Attention(LayerNorm(x)), then x + FeedForward(LayerNorm(x)) with a GELU feed-forward of width ffDim.
/// </summary>
public sealed class TransformerEncoderLayer : Module, ICachedModule
{
    private readonly LayerNorm _norm1;
    private readonly MultiHeadAttention _attention;
    private readonly LayerNorm _norm2;
    private readonly Linear _feedForward1;
    private readonly Linear _feedForward2;
    private readonly Dropout? _dropout;

    /// <summary>Creates the block.</summary>
    /// <param name="dim">Model width.</param>
    /// <param name="heads">Attention heads.</param>
    /// <param name="ffDim">Hidden width of the feed-forward part (typically 4 × dim).</param>
    /// <param name="dropout">Dropout after attention and feed-forward.</param>
    /// <param name="causal">Mask future positions.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source.</param>
    public TransformerEncoderLayer(int dim, int heads, int? ffDim = null, float dropout = 0.1f, bool causal = false, Device? device = null, Random? random = null)
    {
        _norm1 = new LayerNorm(dim, device: device);
        _attention = new MultiHeadAttention(dim, heads, causal, dropout, device, random);
        _norm2 = new LayerNorm(dim, device: device);
        _feedForward1 = new Linear(dim, ffDim ?? 4 * dim, device: device, random: random);
        _feedForward2 = new Linear(ffDim ?? 4 * dim, dim, device: device, random: random);
        _dropout = dropout > 0f ? new Dropout(dropout, random) : null;
        Dim = dim;
    }

    /// <summary>Model width.</summary>
    public int Dim { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        var attended = _attention.Forward(_norm1.Forward(input));
        var x = input + (_dropout?.Forward(attended) ?? attended);
        var hidden = _feedForward2.Forward(FeedForwardHidden(_norm2.Forward(x)));
        return x + (_dropout?.Forward(hidden) ?? hidden);
    }

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        var x = input + _attention.ForwardCached(_norm1.Forward(input), context);
        return x + _feedForward2.Forward(FeedForwardHidden(_norm2.Forward(x)));
    }

    /// <summary>GELU(x·W1 + b1); fused into one kernel (plus the product) during inference.</summary>
    private Tensor FeedForwardHidden(Tensor x) =>
        !Autograd.IsEnabled && _feedForward1.Bias is { } bias
            ? _feedForward1.ProjectWithoutBias(x).BiasGelu(bias)
            : _feedForward1.Forward(x).Gelu();

    /// <inheritdoc />
    public override IEnumerable<Module> Children() =>
        _dropout is null
            ? [_norm1, _attention, _norm2, _feedForward1, _feedForward2]
            : [_norm1, _attention, _norm2, _feedForward1, _feedForward2, _dropout];

    /// <inheritdoc />
    public override string ToString() => $"TransformerEncoderLayer(dim {Dim})";
}

/// <summary>
/// Adds fixed sinusoidal position information to [batch, time, dim] embeddings, so attention can tell
/// positions apart. Supports sequences up to <see cref="MaxLength"/>.
/// </summary>
public sealed class PositionalEncoding : Module, ICachedModule
{
    private Tensor _table;

    /// <summary>Precomputes the encodings.</summary>
    public PositionalEncoding(int maxLength, int dim, Device? device = null)
    {
        MaxLength = maxLength;
        Dim = dim;
        var values = new float[maxLength * dim];
        for (int pos = 0; pos < maxLength; pos++)
        {
            for (int i = 0; i < dim; i++)
            {
                double angle = pos / Math.Pow(10000, 2 * (i / 2) / (double)dim);
                values[pos * dim + i] = (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        }

        _table = CreateBuffer(values, [maxLength, dim], device ?? Device.Default);
    }

    /// <summary>Longest supported sequence.</summary>
    public int MaxLength { get; }

    /// <summary>Embedding width.</summary>
    public int Dim { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        int t = input.Shape[^2];
        if (t > MaxLength)
        {
            throw new ArgumentException($"Sequence length {t} exceeds PositionalEncoding's maximum of {MaxLength}.");
        }

        return input + (t == MaxLength ? _table : _table.Narrow(0, 0, t));
    }

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context) =>
        input + _table.EmbeddingLookup(context.Positions ?? throw new InvalidOperationException("Call DecodingContext.BeginStep first."));

    /// <inheritdoc />
    protected internal override void MoveTo(Device device) => _table = MoveTensor(_table, device);

    /// <inheritdoc />
    public override void Dispose()
    {
        _table.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"PositionalEncoding({MaxLength} x {Dim})";
}
