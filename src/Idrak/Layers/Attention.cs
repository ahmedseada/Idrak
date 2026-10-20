// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Diagnostics;
using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// Multi-head scaled dot-product self-attention over [batch, time, dim]: every position attends to every
/// other (or only to earlier ones when <see cref="Causal"/>). Heads run as one batched matrix product.
/// <para>
/// Options, each off by default (and then the layer computes exactly as without it): the score scale
/// (<see cref="ScoreScale"/>, 1 / √head size unless set), a bias by the query's and the key's positions added to the
/// scores (<see cref="PositionBias"/>), and padding: inside a <see cref="PaddedSequences.Use"/> scope of the input's
/// batch (or through <see cref="Forward(Tensor, PaddedSequences?)"/>) no position attends to another sequence's padding,
/// so a sequence gives the same output alone and in a padded batch (a causal layer's real positions never see the
/// padding after them, so it changes nothing there). The heads may be narrower or wider than the model
/// (<see cref="FromWeights"/>: [dim, 3 · heads · headDim]).
/// </para>
/// </summary>
public sealed class MultiHeadAttention : Module, ICachedModule
{
    private readonly Linear _qkv;
    private readonly Linear _output;
    private readonly Dropout? _dropout;
    private readonly float _scale;
    private readonly IAttentionPositionBias? _bias;
    private Tensor? _mask;
    private (Tensor Starts, Tensor Ends)? _spans;
    private (Tensor Starts, Tensor Ends, int Length, int[] Lengths, Device Device)? _paddedSpans;

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
        Inner = dim;
        Heads = heads;
        HeadDim = dim / heads;
        Causal = causal;
        _scale = 1f / MathF.Sqrt(HeadDim);
        _qkv = new Linear(dim, 3 * dim, device: device, random: random);
        _output = new Linear(dim, dim, device: device, random: random);
        _dropout = dropout > 0f ? new Dropout(dropout, random) : null;
    }

    /// <summary>
    /// A layer around existing projections (for example loaded weights), without dropout; the layer takes ownership. The
    /// same as <see cref="FromWeights"/>, for setting the options at once:
    /// <c>new MultiHeadAttention(qkv, output, heads) { ScoreScale = 1f, PositionBias = bias }</c>.
    /// </summary>
    /// <param name="qkv">The query, key and value projections side by side, [dim, 3 · inner] (inner = heads · head size).</param>
    /// <param name="output">The output projection, [inner, dim].</param>
    /// <param name="heads">Number of heads (a divisor of inner).</param>
    /// <param name="causal">Mask future positions.</param>
    /// <exception cref="ArgumentException">The projections do not fit together.</exception>
    public MultiHeadAttention(Linear qkv, Linear output, int heads, bool causal = false)
    {
        ArgumentNullException.ThrowIfNull(qkv);
        ArgumentNullException.ThrowIfNull(output);
        if (heads <= 0 || output.InFeatures % heads != 0 || qkv.InFeatures != output.OutFeatures || qkv.OutFeatures != 3 * output.InFeatures)
        {
            throw new ArgumentException($"MultiHeadAttention takes qkv [dim, 3·inner] and output [inner, dim] with inner divisible by {heads} heads; "
                + $"got qkv [{qkv.InFeatures}, {qkv.OutFeatures}] and output [{output.InFeatures}, {output.OutFeatures}].");
        }

        Dim = output.OutFeatures;
        Inner = output.InFeatures;
        Heads = heads;
        HeadDim = Inner / heads;
        Causal = causal;
        _scale = 1f / MathF.Sqrt(HeadDim);
        _qkv = qkv;
        _output = output;
    }

    /// <summary>
    /// A layer around existing projections (for example loaded weights), without dropout; the layer takes ownership.
    /// </summary>
    /// <param name="qkv">The query, key and value projections side by side, [dim, 3 · inner] (inner = heads · head size, usually dim): the queries' columns first, then the keys', then the values' (within each, head after head).</param>
    /// <param name="output">The output projection, [inner, dim].</param>
    /// <param name="heads">Number of heads (a divisor of inner); scores are scaled by the head size^-0.5 unless <see cref="ScoreScale"/> is set.</param>
    /// <param name="causal">Mask future positions.</param>
    public static MultiHeadAttention FromWeights(Linear qkv, Linear output, int heads, bool causal = false) => new(qkv, output, heads, causal);

    /// <summary>Model width.</summary>
    public int Dim { get; }

    /// <summary>The heads' width side by side, heads · <see cref="HeadDim"/> (the model width unless built otherwise).</summary>
    public int Inner { get; }

    /// <summary>Number of heads.</summary>
    public int Heads { get; }

    /// <summary>Size of each head.</summary>
    public int HeadDim { get; }

    /// <summary>Whether future positions are masked.</summary>
    public bool Causal { get; }

    /// <summary>
    /// The factor on each query·key product before the softmax: 1 / √<see cref="HeadDim"/> unless set (a family that
    /// scales nothing sets 1).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not finite and positive.</exception>
    public float ScoreScale
    {
        get => _scale;
        init => _scale = float.IsFinite(value) && value > 0f ? value
            : throw new ArgumentOutOfRangeException(nameof(ScoreScale), value, "The attention setting ScoreScale is a finite positive factor (1 / √head size by default).");
    }

    /// <summary>
    /// A bias by the query's and the key's positions, [heads, queries, keys], added to the scaled scores before the softmax
    /// (null, the default: none). The scores are then composed whole ([batch · heads, time, time]); in cached decoding each
    /// step reads its rows of the bias over every cached slot. When it is a <see cref="Module"/> its parameters are this
    /// layer's, unless <see cref="SharesPositionBias"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The bias is for another number of heads.</exception>
    public IAttentionPositionBias? PositionBias
    {
        get => _bias;
        init => _bias = value is null || value.Heads == Heads ? value
            : throw new ArgumentException($"The attention setting PositionBias gives {value.Heads} heads a bias; the layer has {Heads}.", nameof(PositionBias));
    }

    /// <summary>
    /// Whether <see cref="PositionBias"/> belongs to another layer (a bias shared by every layer, owned by the first): its
    /// parameters are then not among this layer's. False by default.
    /// </summary>
    public bool SharesPositionBias { get; init; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Attend(input, null, fused: false);

    /// <summary>
    /// The layer's output for a padded batch: sequence b of <paramref name="input"/> [batch, time, dim] is its first
    /// <paramref name="padding"/>.Lengths[b] positions and no position attends to its padding (null: nothing padded).
    /// </summary>
    public Tensor Forward(Tensor input, PaddedSequences? padding)
    {
        ArgumentNullException.ThrowIfNull(input);
        using (padding is null ? null : PaddedSequences.Use(padding))
        {
            return Forward(input);
        }
    }

    /// <summary>
    /// residual + this layer's output, for inference: the projections' biases and the residual in their products'
    /// epilogues where the layers can (<see cref="Linear.ProjectBiased"/>, <see cref="Linear.ProjectAdding"/>; the same
    /// values). The caller has checked that nothing is recorded, offloaded or traced per layer: the inner layers' products
    /// are called directly.
    /// </summary>
    internal Tensor ForwardAdding(Tensor input, Tensor residual) => Attend(input, residual, fused: true);

    // The layer's output (plus the residual when given; `fused`: the projections through their epilogues).
    private Tensor Attend(Tensor input, Tensor? residual, bool fused)
    {
        if (input.Rank != 3 || input.Shape[2] != Dim)
        {
            throw new ArgumentException($"MultiHeadAttention expects [batch, time, {Dim}], got {Tensor.FormatShape(input.Shape)}.");
        }

        int n = input.Shape[0], t = input.Shape[1], dh = HeadDim;
        // The batch's padding, when it is this input's and pads something (a causal layer's real positions see none of it).
        var padding = !Causal && PaddedSequences.Current is { } current && current.Matches(n, t) && current.Padded ? current : null;
        var qkv = fused ? _qkv.ProjectBiased(input) : _qkv.Forward(input); // [N, T, 3·inner]
        Tensor Output(Tensor context) => residual is null ? _output.Forward(context) : _output.ProjectAdding(context, residual);
        Tensor SplitHeads(int part) => qkv.Narrow(2, part * Inner, Inner)
            .Reshape(n, t, Heads, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n * Heads, t, dh);                                   // [N·H, T, dh]

        Tensor context;
        if (!Causal && _bias is null && (_dropout is null || !_dropout.IsTraining))
        {
            // Every position sees every other: one attention over key ranges (all keys for each row), tiled, so the
            // [T, T] scores of each head are never stored (the gradient recomputes them); or the full scores where the
            // device measured them faster and has the memory for them (Tensor.AttentionFastest). Without a gradient the
            // key ranges read the heads in place from the projections and write [N, T, inner] (no copies of the heads).
            // A padded batch: each sequence's keys end at its length (a table of ranges per sequence, its heads' rows).
            var (starts, ends) = padding is null ? AllKeys(t, input.Device) : PaddedKeys(padding, input.Device);
            bool everyKey = padding is null;
            if (Tensor.AttentionFastestPacked(qkv, Heads, Heads, dh, starts, ends, _scale, everyKey) is { } attended)
            {
                return Output(attended);
            }

            context = Tensor.AttentionFastest(SplitHeads(0), SplitHeads(1), SplitHeads(2), starts, ends, _scale, everyKey);
            return Output(context
                .Reshape(n, Heads, t, dh)
                .Permute(0, 2, 1, 3)
                .Reshape(n, t, Inner));
        }

        var (q, k, v) = (SplitHeads(0), SplitHeads(1), SplitHeads(2));
        var raw = q.MatMul(k, transposeB: true);                           // [N·H, T, T]
        var bias = _bias?.For(t, t, 0, input.Device);                      // [H, T, T]
        Tensor weights;
        if (!Autograd.IsEnabled)
        {
            // Inference: scale, mask (the bias, causal and padding masks added into one) and softmax in one kernel.
            weights = raw.ScaleMaskSoftmax(_scale, InferenceMask(bias, padding, n, t, input.Device));
        }
        else
        {
            var scores = raw * _scale;
            if (bias is not null)
            {
                scores = (scores.Reshape(n, Heads, t, t) + bias).Reshape(n * Heads, t, t);
            }

            if (Causal)
            {
                scores = scores + CausalMask(t, input.Device);
            }

            if (padding is not null)
            {
                scores = scores + padding.Keys(Heads);
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
            .Reshape(n, t, Inner);
        return Output(context);
    }

    // The additive mask of an inference pass over the full scores: none, the causal mask [T, T], the bias [H, T, T] (with
    // the causal mask added), or with padding [N, H, T, T]; ScaleMaskSoftmax repeats its rows over the scores' rows.
    private Tensor? InferenceMask(Tensor? bias, PaddedSequences? padding, int n, int t, Device device)
    {
        var mask = bias;
        if (Causal)
        {
            mask = mask is null ? CausalMask(t, device) : mask + CausalMask(t, device);
        }

        if (padding is not null)
        {
            var keys = padding.Keys(Heads);                                // [N·H, T, T]
            mask = mask is null ? keys : keys.Reshape(n, Heads, t, t) + mask;
        }

        return mask;
    }

    /// <summary>
    /// Cached attention for the new positions of [batch, newSteps, dim]: their keys and values are appended to this
    /// layer's <see cref="KeyValueCache"/>, and each new query attends to every cached position up to its own
    /// (the causal mask comes from the context), so a decoding step costs O(capacity) instead of O(steps²). With a
    /// <see cref="PositionBias"/>, each step adds its rows of the bias over every cached slot (the scores composed; not
    /// while a step is recorded as a graph, since the rows follow the host's position).
    /// </summary>
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        int n = input.Shape[0], t = input.Shape[1], dh = HeadDim;
        var cache = context.CacheFor(this, n * Heads, dh);
        var qkv = _qkv.Forward(input);
        Tensor SplitHeads(int part) => qkv.Narrow(2, part * Inner, Inner)
            .Reshape(n, t, Heads, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n * Heads, t, dh);

        var q = SplitHeads(0);
        cache.Layout.Write(SplitHeads(1), SplitHeads(2), cache, context.Position);
        Tensor output;
        if (_bias is not null)
        {
            if (ComputeGraph.IsCapturing)
            {
                throw new NotSupportedException("A step of an attention layer with a position bias cannot be recorded as a graph (its bias rows follow the position on the host); decode it step by step.");
            }

            var keys = cache.Layout.Expand(cache, keys: true);
            var values = cache.Layout.Expand(cache, keys: false);
            int slots = keys.Shape[1];
            var mask = _bias.For(t, slots, context.LengthOf(cache), input.Device) + context.MaskOf(cache)!;   // [H, t, slots]
            output = q.MatMul(keys, transposeB: true).ScaleMaskSoftmax(_scale, mask).MatMul(values);
        }
        else
        {
            output = cache.Layout.Attend(q, cache, context, t, _scale, decoderKernels: false);
        }

        output = output
            .Reshape(n, Heads, t, dh)
            .Permute(0, 2, 1, 3)
            .Reshape(n, t, Inner);
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

    // The key ranges of a padded batch: one table per sequence (its heads' rows), every row of sequence b seeing keys
    // [0, length b). Cached per lengths and device: a training pass's backward, and the next batch of the same lengths,
    // read the same tables.
    private (Tensor Starts, Tensor Ends) PaddedKeys(PaddedSequences padding, Device device)
    {
        int n = padding.Batch, t = padding.Length;
        if (_paddedSpans is { } cached && cached.Length == t && cached.Device == device && Same(cached.Lengths, padding.Lengths))
        {
            return (cached.Starts, cached.Ends);
        }

        _paddedSpans?.Starts.Dispose();
        _paddedSpans?.Ends.Dispose();
        var ends = new float[n * t];
        for (int b = 0; b < n; b++)
        {
            ends.AsSpan(b * t, t).Fill(padding.Lengths[b]);
        }

        var starts = CreateBuffer(new float[n * t], [n * t], device);
        var endsBuffer = CreateBuffer(ends, [n * t], device);
        _paddedSpans = (starts, endsBuffer, t, [.. padding.Lengths], device);
        return (starts, endsBuffer);
    }

    private static bool Same(int[] kept, IReadOnlyList<int> lengths)
    {
        if (kept.Length != lengths.Count)
        {
            return false;
        }

        for (int i = 0; i < kept.Length; i++)
        {
            if (kept[i] != lengths[i])
            {
                return false;
            }
        }

        return true;
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
    public override IEnumerable<Module> Children()
    {
        yield return _qkv;
        yield return _output;
        if (_dropout is not null)
        {
            yield return _dropout;
        }

        if (_bias is Module owned && !SharesPositionBias)
        {
            yield return owned;
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _mask?.Dispose();
        _spans?.Starts.Dispose();
        _spans?.Ends.Dispose();
        _paddedSpans?.Starts.Dispose();
        _paddedSpans?.Ends.Dispose();
        if (_bias is Module owned && !SharesPositionBias)
        {
            owned.Dispose();                                               // its kept bias too (its parameters again: a no-op)
        }

        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"MultiHeadAttention(dim {Dim}, {Heads} heads{(Inner != Dim ? $", inner {Inner}" : "")}{(Causal ? ", causal" : "")}{(_bias is null ? "" : ", position bias")})";
}

/// <summary>Where a transformer block's normalizations sit (<see cref="TransformerEncoderLayer.Placement"/>).</summary>
public enum NormPlacement
{
    /// <summary>Before each sublayer ("pre-norm"): x + Sublayer(Norm(x)). The default.</summary>
    Before,

    /// <summary>After each residual sum ("post-norm", the original transformer's): Norm(x + Sublayer(x)).</summary>
    After,
}

/// <summary>
/// One transformer encoder block over [batch, time, dim], its LayerNorms placed by <see cref="Placement"/>: pre-norm (the
/// default), x + Attention(LayerNorm(x)), then x + FeedForward(LayerNorm(x)); or post-norm, x = LayerNorm(x + Attention(x)),
/// then LayerNorm(x + FeedForward(x)). The feed-forward is GELU (or the block's own activation) of width ffDim.
/// </summary>
public sealed class TransformerEncoderLayer : Module, ICachedModule
{
    private readonly NormPlacement _placement;
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

    private TransformerEncoderLayer(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2, Module? activation,
        NormPlacement placement)
    {
        _placement = placement;
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
        Create(norm1, attention, norm2, feedForward1, feedForward2, null, NormPlacement.Before);

    /// <summary>
    /// A block around existing layers, as <see cref="FromLayers(LayerNorm, MultiHeadAttention, LayerNorm, Linear, Linear)"/>, with
    /// the feed-forward's activation <paramref name="activation"/> (for example <see cref="QuickGELU"/>, CLIP's, or
    /// <see cref="ExactGELU"/>) between <paramref name="feedForward1"/> and <paramref name="feedForward2"/>.
    /// </summary>
    public static TransformerEncoderLayer FromLayers(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2, Module activation)
    {
        ArgumentNullException.ThrowIfNull(activation);
        return Create(norm1, attention, norm2, feedForward1, feedForward2, activation, NormPlacement.Before);
    }

    /// <summary>
    /// A block around existing layers (for example loaded weights), without dropout, its norms placed by
    /// <paramref name="placement"/>; the block takes ownership. Post-norm (<see cref="NormPlacement.After"/>):
    /// x = <paramref name="norm1"/>(x + attention(x)), then <paramref name="norm2"/>(x + feed-forward(x)), the
    /// feed-forward <paramref name="feedForward2"/>(activation(<paramref name="feedForward1"/>(x))) with
    /// <paramref name="activation"/>, or GELU with the tanh approximation when null.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The placement is not a defined one.</exception>
    public static TransformerEncoderLayer FromLayers(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2,
        Module? activation, NormPlacement placement) =>
        Create(norm1, attention, norm2, feedForward1, feedForward2, activation, Checked(placement));

    private static TransformerEncoderLayer Create(LayerNorm norm1, MultiHeadAttention attention, LayerNorm norm2, Linear feedForward1, Linear feedForward2, Module? activation,
        NormPlacement placement)
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

        return new TransformerEncoderLayer(norm1, attention, norm2, feedForward1, feedForward2, activation, placement);
    }

    private static NormPlacement Checked(NormPlacement placement) => Enum.IsDefined(placement) ? placement
        : throw new ArgumentOutOfRangeException(nameof(placement), placement, "The block setting Placement is NormPlacement.Before or NormPlacement.After.");

    /// <summary>Model width.</summary>
    public int Dim { get; }

    /// <summary>
    /// Where the block's two LayerNorms sit: <see cref="NormPlacement.Before"/> each sublayer (pre-norm, the default) or
    /// <see cref="NormPlacement.After"/> each residual sum (post-norm). Set it on a new block
    /// (<c>new TransformerEncoderLayer(dim, heads) { Placement = NormPlacement.After }</c>) or through
    /// <see cref="FromLayers(LayerNorm, MultiHeadAttention, LayerNorm, Linear, Linear, Module, NormPlacement)"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not a defined placement.</exception>
    public NormPlacement Placement
    {
        get => _placement;
        init => _placement = Checked(value);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (_placement == NormPlacement.After)
        {
            return ForwardPostNorm(input);
        }

        if (_dropout is null && _activation is null && FusedPasses(input))
        {
            // Inference: each residual added in its output projection's epilogue and the GELU in the first feed-forward
            // projection's, where the layers hold packed weights (else the same operations one by one): the same values.
            var sum = _attention.ForwardAdding(_norm1.Forward(input), input);
            return _feedForward2.ProjectAdding(_feedForward1.ProjectGelu(_norm2.Forward(sum)), sum);
        }

        var attended = _attention.Forward(_norm1.Forward(input));
        var x = input + (_dropout?.Forward(attended) ?? attended);
        var hidden = _feedForward2.Forward(FeedForwardHidden(_norm2.Forward(x)));
        return x + (_dropout?.Forward(hidden) ?? hidden);
    }

    // Post-norm: x = norm1(x + attention(x)), then norm2(x + feed-forward(x)).
    private Tensor ForwardPostNorm(Tensor input)
    {
        if (_dropout is null && FusedPasses(input))
        {
            // Inference: each residual added in its output projection's epilogue, the bias (and the tanh GELU) in the first
            // feed-forward projection's, where the layers hold packed weights (else the same operations one by one): the
            // same values as below.
            var sum = _norm1.Forward(_attention.ForwardAdding(input, input));
            var hidden = _activation is null ? _feedForward1.ProjectGelu(sum) : _activation.Forward(_feedForward1.ProjectBiased(sum));
            return _norm2.Forward(_feedForward2.ProjectAdding(hidden, sum));
        }

        var attended = _attention.Forward(input);
        var x = _norm1.Forward(input + (_dropout?.Forward(attended) ?? attended));
        var fed = _feedForward2.Forward(FeedForwardHidden(x));
        return _norm2.Forward(x + (_dropout?.Forward(fed) ?? fed));
    }

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        if (_placement == NormPlacement.After)
        {
            var y = _norm1.Forward(input + _attention.ForwardCached(input, context));
            return _norm2.Forward(y + _feedForward2.Forward(FeedForwardHidden(y)));
        }

        var x = input + _attention.ForwardCached(_norm1.Forward(input), context);
        return x + _feedForward2.Forward(FeedForwardHidden(_norm2.Forward(x)));
    }

    // The fused inference passes: no gradient, not a checkpointed pass's first run (its values must be the recompute's),
    // nothing offloaded and no per-layer telemetry (they call the inner layers' products, not their Forward).
    private static bool FusedPasses(Tensor input) =>
        !Autograd.IsEnabled && !Checkpointing.FirstPass && input.Device.Backend.Offload is null && !Telemetry.IsEnabled(TelemetryLevel.Layers);

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
    public override string ToString() => $"TransformerEncoderLayer(dim {Dim}{(_placement == NormPlacement.After ? ", post-norm" : "")})";
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
