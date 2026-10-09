// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers.Abstractions;

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
    private (Tensor Starts, Tensor Ends)? _spans;

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

    private MultiHeadAttention(Linear qkv, Linear output, int heads, bool causal)
    {
        Dim = output.OutFeatures;
        Heads = heads;
        Causal = causal;
        _qkv = qkv;
        _output = output;
    }

    /// <summary>
    /// A layer around existing projections (for example loaded weights), without dropout; the layer takes ownership.
    /// </summary>
    /// <param name="qkv">The query, key and value projections side by side, [dim, 3 · dim]: the queries' columns first, then the keys', then the values' (within each, head after head).</param>
    /// <param name="output">The output projection, [dim, dim].</param>
    /// <param name="heads">Number of heads (a divisor of dim); scores are scaled by the head size^-0.5.</param>
    /// <param name="causal">Mask future positions.</param>
    public static MultiHeadAttention FromWeights(Linear qkv, Linear output, int heads, bool causal = false)
    {
        ArgumentNullException.ThrowIfNull(qkv);
        ArgumentNullException.ThrowIfNull(output);
        int dim = output.OutFeatures;
        if (heads <= 0 || dim % heads != 0 || output.InFeatures != dim || qkv.InFeatures != dim || qkv.OutFeatures != 3 * dim)
        {
            throw new ArgumentException($"MultiHeadAttention takes qkv [dim, 3·dim] and output [dim, dim] with dim divisible by {heads} heads; "
                + $"got qkv [{qkv.InFeatures}, {qkv.OutFeatures}] and output [{output.InFeatures}, {output.OutFeatures}].");
        }

        return new MultiHeadAttention(qkv, output, heads, causal);
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
        Tensor context;
        if (!Causal && (_dropout is null || !_dropout.IsTraining))
        {
            // Every position sees every other: one attention over key ranges (all keys for each row), tiled, so the
            // [T, T] scores of each head are never stored (the gradient recomputes them); or the full scores where the
            // device measured them faster and has the memory for them (Tensor.AttentionFastest).
            var (starts, ends) = AllKeys(t, input.Device);
            context = Tensor.AttentionFastest(q, k, v, starts, ends, 1f / MathF.Sqrt(dh), everyKey: true);
            return _output.Forward(context
                .Reshape(n, Heads, t, dh)
                .Permute(0, 2, 1, 3)
                .Reshape(n, t, Dim));
        }

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

        context = weights.MatMul(v)                                       // [N·H, T, dh]
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
        cache.Layout.Write(SplitHeads(1), SplitHeads(2), cache, context.Position);
        var output = cache.Layout.Attend(q, cache, context, t, 1f / MathF.Sqrt(dh), decoderKernels: false);
        output = output
            .Reshape(n, Heads, t, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n, t, Dim);
        return _output.Forward(output);
    }

    /// <summary>The key ranges of a bidirectional pass over <paramref name="t"/> positions (every row sees all), cached per length and device.</summary>
    private (Tensor Starts, Tensor Ends) AllKeys(int t, Device device)
    {
        if (_spans is { } cached && cached.Starts.Shape[0] == t && cached.Starts.Device == device)
        {
            return cached;
        }

        _spans?.Starts.Dispose();
        _spans?.Ends.Dispose();
        var spans = KeySpans.Bidirectional(t, t);
        _spans = (CreateBuffer([.. spans.Starts.Select(s => (float)s)], [t], device), CreateBuffer([.. spans.Ends.Select(e => (float)e)], [t], device));
        return _spans.Value;
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
        _spans?.Starts.Dispose();
        _spans?.Ends.Dispose();
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
    private readonly Module? _activation;

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

    private TransformerEncoderLayer(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2, Module? activation)
    {
        _activation = activation;
        _norm1 = norm1;
        _attention = attention;
        _norm2 = norm2;
        _feedForward1 = feedForward1;
        _feedForward2 = feedForward2;
        Dim = attention.Dim;
    }

    /// <summary>
    /// A block around existing layers (for example loaded weights), without dropout; the block takes ownership. Each
    /// LayerNorm keeps its own epsilon; the feed-forward is GELU with the tanh approximation between
    /// <paramref name="feedForward1"/> [dim, ffDim] and <paramref name="feedForward2"/> [ffDim, dim]. SigLIP's encoder
    /// layers are exactly this.
    /// </summary>
    public static TransformerEncoderLayer FromLayers(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2) =>
        Create(norm1, attention, norm2, feedForward1, feedForward2, null);

    /// <summary>
    /// A block around existing layers, as <see cref="FromLayers(LayerNorm, MultiHeadAttention, LayerNorm, Linear, Linear)"/>, with
    /// the feed-forward's activation <paramref name="activation"/> (for example <see cref="QuickGELU"/>, CLIP's, or
    /// <see cref="ExactGELU"/>) between <paramref name="feedForward1"/> and <paramref name="feedForward2"/>.
    /// </summary>
    public static TransformerEncoderLayer FromLayers(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2, Module activation)
    {
        ArgumentNullException.ThrowIfNull(activation);
        return Create(norm1, attention, norm2, feedForward1, feedForward2, activation);
    }

    private static TransformerEncoderLayer Create(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2, Module? activation)
    {
        ArgumentNullException.ThrowIfNull(norm1);
        ArgumentNullException.ThrowIfNull(attention);
        ArgumentNullException.ThrowIfNull(norm2);
        ArgumentNullException.ThrowIfNull(feedForward1);
        ArgumentNullException.ThrowIfNull(feedForward2);
        int dim = attention.Dim;
        if (norm1.Features != dim || norm2.Features != dim || feedForward1.InFeatures != dim || feedForward2.InFeatures != feedForward1.OutFeatures
            || feedForward2.OutFeatures != dim)
        {
            throw new ArgumentException($"The layers of a TransformerEncoderLayer of width {dim} do not fit: norms of {norm1.Features} and {norm2.Features}, "
                + $"feed-forward [{feedForward1.InFeatures}, {feedForward1.OutFeatures}] and [{feedForward2.InFeatures}, {feedForward2.OutFeatures}].");
        }

        return new TransformerEncoderLayer(norm1, attention, norm2, feedForward1, feedForward2, activation);
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

    /// <summary>
    /// GELU(x·W1 + b1) (fused into one kernel, plus the product, during inference; not in a checkpointed pass, whose values
    /// the recompute with gradients must give again), or the block's own activation.
    /// </summary>
    private Tensor FeedForwardHidden(Tensor x) =>
        _activation is not null ? _activation.Forward(_feedForward1.Forward(x))
        : !Autograd.IsEnabled && !Checkpointing.FirstPass && _feedForward1.Bias is { } bias
            ? _feedForward1.ProjectWithoutBias(x).BiasGelu(bias)
            : _feedForward1.Forward(x).Gelu();

    /// <inheritdoc />
    public override IEnumerable<Module> Children() =>
        _dropout is null
            ? _activation is null ? [_norm1, _attention, _norm2, _feedForward1, _feedForward2] : [_norm1, _attention, _norm2, _feedForward1, _activation, _feedForward2]
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
    protected override void MoveTo(Device device) => _table = MoveTensor(_table, device);

    /// <inheritdoc />
    public override void Dispose()
    {
        _table.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"PositionalEncoding({MaxLength} x {Dim})";
}
