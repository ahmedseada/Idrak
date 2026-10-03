// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// RMS normalization over the last dimension: x / sqrt(mean(x²) + eps) · (gain + offset). The offset is 0 for most models
/// (1 for models that store the gain as a difference from 1).
/// </summary>
public sealed class RMSNorm : Module
{
    /// <summary>Creates the layer with a gain of ones.</summary>
    public RMSNorm(int features, float epsilon = 1e-6f, float offset = 0f, Device? device = null)
        : this(Tensor.Persistent(Enumerable.Repeat(1f - offset, features).ToArray(), [features], device ?? Device.Default, requiresGrad: true), epsilon, offset)
    {
    }

    private RMSNorm(Tensor gain, float epsilon, float offset)
    {
        Gain = gain;
        Epsilon = epsilon;
        Offset = offset;
    }

    /// <summary>A layer around an existing gain [features]; the layer takes ownership.</summary>
    public static RMSNorm FromWeights(Tensor gain, float epsilon, float offset = 0f) => new(gain, epsilon, offset);

    /// <summary>Size of the normalized dimension.</summary>
    public int Features => Gain.Size;

    /// <summary>Added to the variance before the square root.</summary>
    public float Epsilon { get; }

    /// <summary>Added to the gain (the effective scale is gain + offset).</summary>
    public float Offset { get; }

    /// <summary>The learned scale, [features].</summary>
    public Tensor Gain { get; private set; }

    // A normalization computed ahead by the layer before (a decoder block's residual addition fused with this norm):
    // used when this norm is next called with exactly that input on the same thread.
    [ThreadStatic]
    private static (RMSNorm Norm, Tensor Input, Tensor Output)? t_handoff;

    internal static void HandOff(RMSNorm norm, Tensor input, Tensor output) => t_handoff = (norm, input, output);

    // The normalization handed off for `input`, still to be read by its norm, or null.
    internal static Tensor? HandedOff(Tensor input) =>
        t_handoff is { } handoff && ReferenceEquals(handoff.Input, input) ? handoff.Output : null;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Shape[^1] != Features)
        {
            throw new ArgumentException($"RMSNorm({Features}) expects [..., {Features}], got {Tensor.FormatShape(input.Shape)}.");
        }

        if (t_handoff is { } handoff && ReferenceEquals(handoff.Norm, this))
        {
            t_handoff = null;
            if (ReferenceEquals(handoff.Input, input) && !Autograd.IsEnabled && !handoff.Output.IsDisposed)
            {
                return handoff.Output;                                          // computed with the residual addition before it
            }
        }

        if (!Autograd.IsEnabled || !input.RequiresGrad && !Gain.RequiresGrad)
        {
            return input.RmsNormAffine(Gain, Epsilon, Offset);                // one kernel when nothing needs gradients
        }

        var scale = Offset == 0f ? Gain : Gain + Offset;
        return input.RmsNormalize(Epsilon).GroupAffine(scale, null, Features, 1);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [Gain];

    /// <inheritdoc />
    protected internal override void MoveTo(Device device) => Gain = MoveTensor(Gain, device);

    /// <inheritdoc />
    public override string ToString() => $"RMSNorm({Features})";
}

/// <summary>Rotary position embedding settings.</summary>
/// <param name="Theta">The base of the frequencies (10000 in the original paper; larger for long-context models).</param>
/// <param name="RotaryDim">Dimensions of each head that rotate (all when null).</param>
/// <param name="Interleaved">Pairs are (2i, 2i+1) instead of (i, i + rotaryDim / 2).</param>
/// <param name="Scaling">Frequency scaling for long contexts, or null.</param>
public sealed record RopeSettings(float Theta, int? RotaryDim = null, bool Interleaved = false, RopeScaling? Scaling = null)
{
    /// <summary>The rotation frequency of each pair (after scaling; for a scaling whose frequencies depend on the position, those of the first positions).</summary>
    public double[] Frequencies(int headDim) => Scaled(headDim).Frequencies;

    /// <summary>
    /// The frequencies of each pair and the factor on the cos and sin tables, after <see cref="Scaling"/> (computed by the
    /// method registered for its type in <see cref="RopeScalings"/>).
    /// </summary>
    public RopeScalingResult Scaled(int headDim)
    {
        int rotary = RotaryDim ?? headDim, half = rotary / 2;
        var frequencies = new double[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = 1.0 / Math.Pow(Theta, 2.0 * i / rotary);
        }

        if (Scaling is null)
        {
            return new RopeScalingResult(frequencies);
        }

        var result = RopeScalings.Get(Scaling.Type)(new RopeScalingInput(frequencies, Theta, rotary, Scaling.Parameters));
        if (result.Frequencies.Length != half)
        {
            throw new InvalidOperationException($"RoPE scaling '{Scaling.Type}' returned {result.Frequencies.Length} frequencies; {half} expected.");
        }

        return result;
    }
}

/// <summary>
/// Causal self-attention for decoder-only language models: separate query, key and value projections (named "q", "k",
/// "v"; output "o"), grouped-query attention (fewer key/value heads, each shared by a group of query heads),
/// a head size independent of the model width, optional biases, optional RMS normalization of each head's queries and
/// keys, and rotary position embeddings. Works with the KV cache (float32 or int8) for incremental decoding.
/// <para>
/// Optionally (<see cref="SlidingWindow"/>, <see cref="ScoreSoftcap"/>): each position attends only to the last positions
/// up to a window (Mistral, Qwen2, Gemma 2 and 3), and the scores are soft-capped, cap · tanh(score / cap) (Gemma 2).
/// The device attention kernels mask causally only, so a windowed layer whose window falls inside the sequence, and a
/// soft-capped layer, attend through basic operations every backend has (the scores of all positions, masked); a window
/// longer than the sequence uses the kernels, with identical results.
/// </para>
/// </summary>
public sealed class CausalSelfAttention : Module, ICachedModule
{
    private Tensor _cos = null!;
    private Tensor _sin = null!;
    private readonly Dictionary<int, Tensor> _masks = [];
    private Tensor? _iota;

    /// <summary>Creates the layer with random projections.</summary>
    /// <param name="dim">Model width.</param>
    /// <param name="heads">Query heads.</param>
    /// <param name="kvHeads">Key/value heads (a divisor of <paramref name="heads"/>; equal to it for standard attention).</param>
    /// <param name="headDim">Size of each head.</param>
    /// <param name="rope">Rotary embedding settings, or null for none.</param>
    /// <param name="maxPositions">Longest sequence (the size of the rotary tables).</param>
    /// <param name="qkvBias">Biases on the query, key and value projections.</param>
    /// <param name="outputBias">A bias on the output projection.</param>
    /// <param name="qkNormEpsilon">RMS-normalize each head's queries and keys (with this epsilon), or null.</param>
    /// <param name="device">Device.</param>
    /// <param name="random">Initialization.</param>
    public CausalSelfAttention(int dim, int heads, int kvHeads, int headDim, RopeSettings? rope, int maxPositions, bool qkvBias = false,
        bool outputBias = false, float? qkNormEpsilon = null, Device? device = null, Random? random = null)
        : this(new Linear(dim, heads * headDim, qkvBias, device, random), new Linear(dim, kvHeads * headDim, qkvBias, device, random),
            new Linear(dim, kvHeads * headDim, qkvBias, device, random), new Linear(heads * headDim, dim, outputBias, device, random),
            qkNormEpsilon is { } eps ? new RMSNorm(headDim, eps, device: device) : null,
            qkNormEpsilon is { } eps2 ? new RMSNorm(headDim, eps2, device: device) : null,
            heads, kvHeads, headDim, rope, maxPositions)
    {
    }

    /// <summary>Creates the layer from existing projections (for example loaded weights); the layer takes ownership.</summary>
    public CausalSelfAttention(Linear query, Linear key, Linear value, Linear output, RMSNorm? queryNorm, RMSNorm? keyNorm,
        int heads, int kvHeads, int headDim, RopeSettings? rope, int maxPositions)
    {
        if (heads % kvHeads != 0)
        {
            throw new ArgumentException($"Query heads ({heads}) must be a multiple of key/value heads ({kvHeads}).");
        }

        if (query.OutFeatures != heads * headDim || key.OutFeatures != kvHeads * headDim || value.OutFeatures != kvHeads * headDim
            || output.InFeatures != heads * headDim)
        {
            throw new ArgumentException("The projection sizes do not match heads × head size.");
        }

        (Query, Key, Value, Output, QueryNorm, KeyNorm) = (query, key, value, output, queryNorm, keyNorm);
        query.Name = "q";
        key.Name = "k";
        value.Name = "v";
        output.Name = "o";
        if (queryNorm is not null)
        {
            queryNorm.Name = "q_norm";
        }

        if (keyNorm is not null)
        {
            keyNorm.Name = "k_norm";
        }

        Heads = heads;
        KvHeads = kvHeads;
        HeadDim = headDim;
        ScoreScale = 1f / MathF.Sqrt(headDim);
        Rope = rope;
        MaxPositions = maxPositions;
        if (rope is not null)
        {
            (_cos, _sin) = RopeTables.Acquire(rope, headDim, maxPositions, query.Device);
        }
    }

    /// <summary>Query heads.</summary>
    public int Heads { get; }

    /// <summary>Key/value heads.</summary>
    public int KvHeads { get; }

    /// <summary>Size of each head.</summary>
    public int HeadDim { get; }

    /// <summary>Rotary embedding settings, or null.</summary>
    public RopeSettings? Rope { get; }

    /// <summary>Longest supported sequence.</summary>
    public int MaxPositions { get; }

    /// <summary>Query projection ("q").</summary>
    public Linear Query { get; }

    /// <summary>Key projection ("k").</summary>
    public Linear Key { get; }

    /// <summary>Value projection ("v").</summary>
    public Linear Value { get; }

    /// <summary>Output projection ("o").</summary>
    public Linear Output { get; }

    /// <summary>RMS normalization of each query head, or null.</summary>
    public RMSNorm? QueryNorm { get; }

    /// <summary>RMS normalization of each key head, or null.</summary>
    public RMSNorm? KeyNorm { get; }

    /// <summary>The factor on each query·key product before the softmax (1 / √<see cref="HeadDim"/> unless set).</summary>
    public float ScoreScale { get; init; }

    /// <summary>
    /// How many positions each query attends to, itself included (query position minus key position below the window),
    /// or null for every earlier position. A window of <see cref="MaxPositions"/> or more is the same as none.
    /// </summary>
    public int? SlidingWindow
    {
        get => _window;
        init => _window = value is null or > 0 ? value : throw new ArgumentOutOfRangeException(nameof(SlidingWindow), "The window must be positive.");
    }

    private readonly int? _window;

    /// <summary>Soft-caps the scaled scores before the mask and softmax, cap · tanh(score / cap), or null for none.</summary>
    public float? ScoreSoftcap
    {
        get => _softcap;
        init => _softcap = value is null or > 0f ? value : throw new ArgumentOutOfRangeException(nameof(ScoreSoftcap), "The cap must be positive.");
    }

    private readonly float? _softcap;

    // A window shorter than the longest sequence (else it never masks anything).
    private bool Windowed => _window is { } w && w < MaxPositions;

    /// <summary>Plain causal attention: no window that can mask and no soft-capping (what the segmented and per-row kernels compute).</summary>
    internal bool PlainCausal => !Windowed && _softcap is null;

    // Whether attention over keys at positions [0, keys) for queries ending at keys - 1 needs the composed path: soft-capped
    // scores, or a window some query reaches past.
    private bool Composed(int keys) => _softcap is not null || Windowed && keys > _window!.Value;

    private int Group => Heads / KvHeads;

    /// <inheritdoc />
    public override IEnumerable<Module> Children()
    {
        yield return Query;
        yield return Key;
        yield return Value;
        yield return Output;
        if (QueryNorm is not null)
        {
            yield return QueryNorm;
        }

        if (KeyNorm is not null)
        {
            yield return KeyNorm;
        }
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        int n = input.Shape[0], t = input.Shape[1];
        if (t > MaxPositions)
        {
            throw new ArgumentException($"{t} positions exceed the layer's maximum of {MaxPositions}.");
        }

        float scale = ScoreScale;
        var packing = PackedSequences.Current is { } current && current.Matches(n, t) ? current : null;
        bool composed = Composed(t);
        if (packing is not null && !PlainCausal)
        {
            throw new NotSupportedException("Packed sequences need plain causal attention; this layer has a sliding window or soft-capped scores (pad the batches instead).");
        }

        if (packing is null && !composed && Rope is null && QueryNorm is null && KeyNorm is null && FusedTraining.Enabled && input.Backend.Capabilities.MatrixUnitAttentionHeadDim(HeadDim)
            && input.Backend.Capabilities.MatrixUnits && MixedPrecision.UsesTensorCores
            && Linear.PlainFloat(Query) && Linear.PlainFloat(Key) && Linear.PlainFloat(Value)
            && Tensor.ProjectPacked(input, [Query, Key, Value]) is { } packed)
        {
            // Queries, keys and values side by side per position, read in place by the attention kernels, which write
            // [n, t, heads·dim] for the output projection: no head rearrangement either way.
            if (Tensor.CausalAttentionPacked(packed, Heads, KvHeads, HeadDim, scale) is { } attended)
            {
                return Output.Forward(attended);
            }

            packed.Dispose();
        }

        var positions = packing?.Positions ?? Positions(t);
        var (q, keys, values) = Project(input, positions, packed: packing is not null);
        Tensor k = keys!, v = values!;
        if (packing is not null)
        {
            // Several sequences per row: each position attends within its own sequence only.
            var segmented = Tensor.CausalAttentionSegmented(q, k, v, packing, KvHeads, scale)
                ?? throw new NotSupportedException($"Packed sequences need attention within each sequence, which {input.Device} does not provide for head size {HeadDim} (on CUDA: bfloat16 tensor cores, head size 64 or 128).");
            ActivationMemory.Compress(q, k, v);
            return Merge(segmented, n, t);
        }

        if (!composed && HeadDim <= input.Backend.Capabilities.TiledAttentionHeadDim)
        {
            // Tiled attention, forward and backward: no [t, t] weights stored (positions[0] = 0 is the causal offset).
            var attended = Tensor.CausalAttention(q, k, v, positions, t, scale);
            ActivationMemory.Compress(q, k, v);
            return Merge(attended, n, t);
        }

        var raw = q.MatMul(k, transposeB: true);                                  // [n·kv, group·t, t]
        if (_softcap is { } cap)
        {
            (raw, scale) = (Softcap(raw, scale, cap), 1f);
        }

        Tensor weights;
        if (!Autograd.IsEnabled)
        {
            weights = raw.ScaleMaskSoftmax(scale, CausalMask(t, 1));             // mask row = query row % t
        }
        else
        {
            weights = (raw * scale + CausalMask(t, Group)).Softmax();            // the mask repeated for each group
        }

        return Merge(weights.MatMul(v), n, t);
    }

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context) => Output.Forward(HeadsCached(input, context));

    // ForwardCached without the output projection: the heads' results side by side, [n, t, heads·d].
    internal Tensor HeadsCached(Tensor input, DecodingContext context)
    {
        int n = input.Shape[0], t = input.Shape[1];
        var positions = context.Positions ?? throw new InvalidOperationException("Call DecodingContext.BeginStep first.");
        var cache = context.CacheFor(this, n * KvHeads, HeadDim);
        if (context.RowStarts is not null)
        {
            // Rows of different lengths: positions per token, and each row attends from its own start.
            if (!cache.Layout.RowStarts)
            {
                throw new NotSupportedException("Rows of different lengths need a float32 key/value cache.");
            }

            if (!PlainCausal)
            {
                throw new NotSupportedException("Rows of different lengths need plain causal attention; this layer has a sliding window or soft-capped scores.");
            }

            var (rq, rk, rv) = Project(input, context.TokenPositions!, packed: true);
            Tensor.WriteKeyValues(rk!, cache.Keys, context.Position);
            Tensor.WriteKeyValues(rv!, cache.Values, context.Position);
            var rows = Tensor.AttentionRows(q: rq, cache.Keys, cache.Values, context.Position, t, ScoreScale, context.TokenStarts!, KvHeads)
                       ?? throw new NotSupportedException($"Rows of different lengths need attention from per-row starts, which {input.Device} does not provide for head size {HeadDim} (on CUDA: bfloat16 tensor cores, head size 64 or 128).");
            return MergeHeads(rows, n, t);
        }

        var (q, k, v) = Project(input, positions, cache, context.Position);   // k and v null: already in the cache
        float scale = ScoreScale;
        if (k is not null)
        {
            cache.Layout.Write(k, v!, cache, context.Position);
        }

        // A step being recorded is replayed at later positions, so it takes the composed path whenever the window can mask.
        if (_softcap is not null || Windowed && (ComputeGraph.IsCapturing || Composed(context.Length + t)))
        {
            return MergeHeads(AttendComposed(q, cache, context, t, scale), n, t);
        }

        var context8 = cache.Layout.Attend(q, cache, context, t, scale, decoderKernels: true);
        return MergeHeads(context8, n, t);
    }

    // Attention over every cached slot from basic operations (any layout, any backend; device-side masks, so recordable):
    // the scores [rows, group·t, capacity], soft-capped, with the causal mask and, for a window, the positions before it
    // masked too.
    private Tensor AttendComposed(Tensor q, KeyValueCache cache, DecodingContext context, int t, float scale)
    {
        var keys = cache.Layout.Expand(cache, keys: true);
        var values = cache.Layout.Expand(cache, keys: false);
        var raw = q.MatMul(keys, transposeB: true);
        if (_softcap is { } cap)
        {
            (raw, scale) = (Softcap(raw, scale, cap), 1f);
        }

        var mask = context.Mask!;
        if (Windowed)
        {
            // Key j is before query i's window when j <= position + i - window: the causal mask of a cache `window` slots
            // wider, read from slot `window` on, is 0 exactly there (-1e9 elsewhere), so -1e9 - it masks those keys.
            int capacity = keys.Shape[1], window = _window!.Value;
            var wider = Tensor.DecoderMask(context.Position, t, capacity + window);
            mask = mask + (wider.Narrow(1, window, capacity) * -1f + -1e9f);
        }

        return raw.ScaleMaskSoftmax(scale, mask).MatMul(values);
    }

    // cap · tanh(scale · scores / cap): the scaled scores, soft-capped.
    private static Tensor Softcap(Tensor scores, float scale, float cap) => (scores * (scale / cap)).Tanh() * cap;

    // Projections → q [n·kv, group·t, d] (the query heads sharing a key/value head are stacked), k and v [n·kv, t, d];
    // with a (float32 or bfloat16) cache, inference writes k and v into it in the same pass and returns them null.
    // packed: positions hold one entry per token of the [n, t] batch (PackedSequences) instead of one per step.
    private (Tensor Q, Tensor? K, Tensor? V) Project(Tensor input, Tensor positions, KeyValueCache? cache = null, Tensor? position = null, bool packed = false)
    {
        int n = input.Shape[0], t = input.Shape[1], d = HeadDim;
        var projected = Linear.ForwardMany(input, Query, Key, Value);
        if (!Autograd.IsEnabled && !packed && Tensor.NormRopeHeads(projected[0], projected[1], projected[2], Heads, KvHeads, d, QueryNorm, KeyNorm,
                Rope is null ? null : _cos, Rope is null ? null : _sin, positions, Rope?.Interleaved ?? false,
                cache is { Layout.FusedWrite: true } ? cache : null, cache is { Layout.FusedWrite: true } ? position : null) is { } heads)
        {
            return heads;                           // normalization, rotation and head layout (or cache writes) in one pass
        }

        var q = projected[0].Reshape(n, t, Heads, d);
        var k = projected[1].Reshape(n, t, KvHeads, d);
        var v = projected[2].Reshape(n, t, KvHeads, d);
        Tensor? normedQ = null, normedK = null;
        if (Rope is not null && QueryNorm is not null && KeyNorm is not null && !Autograd.IsEnabled && !packed)
        {
            // Inference: each head's normalization and rotation in one pass.
            int half = _cos.Shape[1];
            (q, k) = Tensor.RmsNormRopePair(q, QueryNorm.Gain, QueryNorm.Epsilon, QueryNorm.Offset, k, KeyNorm.Gain, KeyNorm.Epsilon,
                KeyNorm.Offset, _cos, _sin, positions, half, Rope.Interleaved);
        }
        else
        {
            if (QueryNorm is not null)
            {
                q = normedQ = QueryNorm.Forward(q);
            }

            if (KeyNorm is not null)
            {
                k = normedK = KeyNorm.Forward(k);
            }

            if (Rope is not null)
            {
                int half = _cos.Shape[1];
                if (packed)
                {
                    // One position per token: the batch as a single row of n·t steps.
                    q = q.Reshape(1, n * t, Heads, d).Rope(_cos, _sin, positions, half, Rope.Interleaved).Reshape(n, t, Heads, d);
                    k = k.Reshape(1, n * t, KvHeads, d).Rope(_cos, _sin, positions, half, Rope.Interleaved).Reshape(n, t, KvHeads, d);
                }
                else
                {
                    q = q.Rope(_cos, _sin, positions, half, Rope.Interleaved);
                    k = k.Rope(_cos, _sin, positions, half, Rope.Interleaved);
                }
            }
        }

        var queries = q.Reshape(n, t, KvHeads, Group, d).Permute(0, 2, 3, 1, 4).Reshape(n * KvHeads, Group * t, d);
        var keys = k.Permute(0, 2, 1, 3).Reshape(n * KvHeads, t, d);
        var values = v.Permute(0, 2, 1, 3).Reshape(n * KvHeads, t, d);

        // Training: the projections, their normalized and rotated forms and the pre-rearrangement layout are read by no
        // backward step (rotation, rearrangement, bias and the projections themselves read only their inputs; a norm
        // reads its own normalized values, kept inside it): released now instead of at the end of the step.
        ActivationMemory.Release([queries, keys, values], projected[0], projected[1], projected[2], normedQ, normedK, q, k, v);
        return (queries, keys, values);
    }

    // [n·kv, group·t, d] → [n, t, heads·d] → output projection (query head h = kv · group + g, as the heads were split).
    private Tensor Merge(Tensor context, int n, int t)
    {
        var merged = MergeHeads(context, n, t);
        var output = Output.Forward(merged);
        ActivationMemory.Compress(context, merged);                  // read again only by the attention's and projection's backward
        return output;
    }

    private Tensor MergeHeads(Tensor context, int n, int t) =>
        context.Reshape(n, KvHeads, Group, t, HeadDim).Permute(0, 3, 1, 2, 4).Reshape(n, t, Heads * HeadDim);

    private Tensor Positions(int t)
    {
        if (_iota is null || _iota.Size < t || _iota.Device != Query.Device)
        {
            _iota?.Dispose();
            _iota = Tensor.Persistent([.. Enumerable.Range(0, MaxPositions).Select(i => (float)i)], [MaxPositions], Query.Device, requiresGrad: false);
        }

        return _iota;
    }

    // A [repeats·t, t] causal mask (0 on and below the diagonal, -1e9 above; with a window, also -1e9 from the window's
    // length below the diagonal on), cached per size.
    private Tensor CausalMask(int t, int repeats)
    {
        int key = t * 1024 + repeats;
        if (_masks.TryGetValue(key, out var mask) && mask.Device == Query.Device)
        {
            return mask;
        }

        var values = new float[repeats * t * t];
        for (int r = 0; r < repeats; r++)
        {
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < t; j++)
                {
                    if (j > i || Windowed && i - j >= _window!.Value)
                    {
                        values[(r * t + i) * t + j] = -1e9f;
                    }
                }
            }
        }

        mask?.Dispose();
        _masks[key] = mask = Tensor.Persistent(values, [repeats * t, t], Query.Device, requiresGrad: false);
        return mask;
    }

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        base.MoveTo(device);
        if (Rope is not null && _cos.Device != device)
        {
            RopeTables.Release(Rope, HeadDim, MaxPositions, _cos.Device);
            (_cos, _sin) = RopeTables.Acquire(Rope, HeadDim, MaxPositions, device);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (Rope is not null && _cos is not null)
        {
            RopeTables.Release(Rope, HeadDim, MaxPositions, _cos.Device);
            _cos = _sin = null!;
        }

        foreach (var mask in _masks.Values)
        {
            mask.Dispose();
        }

        _iota?.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"CausalSelfAttention({Heads} heads{(KvHeads != Heads ? $", {KvHeads} kv heads" : "")}, head size {HeadDim}{(Rope is null ? "" : ", rope")}{(QueryNorm is null ? "" : ", qk-norm")}"
        + $"{(_window is { } w ? $", window {w}" : "")}{(_softcap is { } c ? $", softcap {c}" : "")})";
}

/// <summary>The activation of a <see cref="FeedForward"/> block.</summary>
public enum FeedForwardActivation
{
    /// <summary>x · sigmoid(x) (also called swish; with a gate: SwiGLU).</summary>
    Silu,

    /// <summary>GELU, tanh approximation (with a gate: GeGLU).</summary>
    Gelu,

    /// <summary>max(0, x).</summary>
    Relu,
}

/// <summary>
/// A transformer feed-forward block: gated, down(act(gate(x)) · up(x)) (SwiGLU / GeGLU), or plain, down(act(up(x))).
/// Projections are named "gate", "up" and "down".
/// </summary>
public sealed class FeedForward : Module
{
    /// <summary>Creates the block with random projections.</summary>
    public FeedForward(int dim, int hidden, bool gated, FeedForwardActivation activation, bool bias = false, Device? device = null, Random? random = null)
        : this(gated ? new Linear(dim, hidden, bias, device, random) : null, new Linear(dim, hidden, bias, device, random),
            new Linear(hidden, dim, bias, device, random), activation)
    {
    }

    /// <summary>Creates the block from existing projections (gate null for a plain block); the block takes ownership.</summary>
    public FeedForward(Linear? gate, Linear up, Linear down, FeedForwardActivation activation)
    {
        (Gate, Up, Down, Activation) = (gate, up, down, activation);
        if (gate is not null)
        {
            gate.Name = "gate";
        }

        up.Name = "up";
        down.Name = "down";
    }

    /// <summary>The gate projection ("gate"), or null for a plain block.</summary>
    public Linear? Gate { get; }

    /// <summary>The up projection ("up").</summary>
    public Linear Up { get; }

    /// <summary>The down projection ("down").</summary>
    public Linear Down { get; }

    /// <summary>The activation.</summary>
    public FeedForwardActivation Activation { get; }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => Gate is null ? [Up, Down] : [Gate, Up, Down];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        Tensor hidden;
        if (Gate is null && Activation == FeedForwardActivation.Gelu && FusedTraining.Enabled && Linear.PlainFloat(Up) && Linear.PlainFloat(Down)
            && input.Backend.Capabilities.MatrixUnits && MixedPrecision.UsesTensorCores
            && Tensor.FeedForwardGelu(input, Up, Down) is { } geluBlock)
        {
            return geluBlock;                                       // GELU inside the products (tensor cores)
        }

        if (Gate is null)
        {
            hidden = Activate(Up.Forward(input));
        }
        else if (!Autograd.IsEnabled && Down.PackedWeight is not { ActivationInDownProjection: true } && Tensor.MatMulPackedGatedPair(input, Gate, Up, (int)Activation) is { } pair)
        {
            hidden = pair;                                          // act(gate) · up written by the gate/up product
        }
        else
        {
            var projected = Linear.ForwardMany(input, Gate, Up);
            if (!Autograd.IsEnabled && Down.Adapter is null && Activation is FeedForwardActivation.Silu or FeedForwardActivation.Gelu
                && Tensor.MatMulPackedGated(projected[0], projected[1], (int)Activation, Down) is { } fused)
            {
                return Down.Bias is null ? fused : fused + Down.Bias;                        // activation read by the down projection
            }

            var (gate, up, kind) = (projected[0], projected[1], (int)Activation);
            bool recompute = ActivationMemory.RecomputeFeedForward && Autograd.IsEnabled;
            Backends.Storage? packedHidden = null;
            if (ActivationMemory.BFloat16 && Autograd.IsEnabled
                && Tensor.GatedActivationCompressed(gate, up, kind, packOutput: !recompute, out packedHidden) is { } compressed)
            {
                hidden = compressed;                                        // gate, up (and the result) kept as bfloat16 by the kernel
            }
            else
            {
                hidden = Tensor.GatedActivation(gate, up, kind);           // act(gate) · up in one kernel
                ActivationMemory.Compress(gate, up);                        // read again only by its backward
            }

            if (recompute)
            {
                // Released after the down projection reads it, recomputed from gate and up when a backward step needs it.
                var down = Down.Forward(hidden);
                hidden.Evict(h => Tensor.RecomputeGatedActivation(gate, up, h, kind));
                return down;
            }

            var result = Down.Forward(hidden);
            if (packedHidden is not null)
            {
                hidden.EvictToPacked(packedHidden);
            }
            else
            {
                ActivationMemory.Compress(hidden);
            }

            return result;
        }
        return Down.Forward(hidden);
    }

    /// <summary>
    /// Whether <see cref="DownInput"/> and a fused down projection (packed weights, no bias or adapter) can stand in for
    /// <see cref="Module.Forward"/> at inference (formats whose down projection reads the activation keep that product). A
    /// format of one's own qualifies too: its down projection is its own product, and the residual addition and the next
    /// normalization after it still run as one pass.
    /// </summary>
    internal bool DownFusable => Down.PackedWeight is { ActivationInDownProjection: false } && Down.Bias is null && Down.Adapter is null;

    // The down projection's input at inference: act(gate) · up, or act(up).
    internal Tensor DownInput(Tensor input)
    {
        if (Gate is null)
        {
            return Activate(Up.Forward(input));
        }

        if (Tensor.MatMulPackedGatedPair(input, Gate, Up, (int)Activation) is { } pair)
        {
            return pair;                                            // act(gate) · up written by the gate/up product
        }

        var projected = Linear.ForwardMany(input, Gate, Up);
        return Tensor.GatedActivation(projected[0], projected[1], (int)Activation);
    }

    private Tensor Activate(Tensor x) => Activation switch
    {
        FeedForwardActivation.Silu => x * x.Sigmoid(),
        FeedForwardActivation.Gelu => x.Gelu(),
        _ => x.Relu(),
    };

    /// <inheritdoc />
    public override string ToString() => $"FeedForward({Up.InFeatures} -> {Up.OutFeatures}, {(Gate is null ? "" : "gated ")}{Activation})";
}

/// <summary>
/// One decoder layer. Sequential (the usual): x + attention(norm1(x)), then + feedForward(norm2(·)); optionally with
/// normalizations after each block too. Parallel: x + attention(norm1(x)) + feedForward(norm1(x)).
/// </summary>
public sealed class DecoderBlock : Module, ICachedModule
{
    /// <summary>Creates the block from its parts; the block takes ownership.</summary>
    /// <param name="attentionNorm">Normalization before attention ("attn_norm").</param>
    /// <param name="attention">The attention layer.</param>
    /// <param name="feedForwardNorm">Normalization before the feed-forward block ("mlp_norm"); null for a parallel block.</param>
    /// <param name="feedForward">The feed-forward block.</param>
    /// <param name="postAttentionNorm">Normalization of the attention output before the residual addition, or null.</param>
    /// <param name="postFeedForwardNorm">Normalization of the feed-forward output before the residual addition, or null.</param>
    /// <param name="residualDropout">Dropout on the attention and feed-forward outputs before each residual addition (training only), or null.</param>
    public DecoderBlock(Module attentionNorm, CausalSelfAttention attention, Module? feedForwardNorm, FeedForward feedForward,
        Module? postAttentionNorm = null, Module? postFeedForwardNorm = null, Dropout? residualDropout = null)
    {
        (AttentionNorm, Attention, FeedForwardNorm, FeedForward, PostAttentionNorm, PostFeedForwardNorm) =
            (attentionNorm, attention, feedForwardNorm, feedForward, postAttentionNorm, postFeedForwardNorm);
        ResidualDropout = residualDropout;
        if (residualDropout is not null)
        {
            residualDropout.Name = "drop";
        }

        attentionNorm.Name = "attn_norm";
        attention.Name = "attn";
        feedForward.Name = "mlp";
        if (feedForwardNorm is not null)
        {
            feedForwardNorm.Name = "mlp_norm";
        }

        if (postAttentionNorm is not null)
        {
            postAttentionNorm.Name = "post_attn_norm";
        }

        if (postFeedForwardNorm is not null)
        {
            postFeedForwardNorm.Name = "post_mlp_norm";
        }
    }

    /// <summary>Normalization before attention.</summary>
    public Module AttentionNorm { get; }

    /// <summary>The attention layer.</summary>
    public CausalSelfAttention Attention { get; }

    /// <summary>Normalization before the feed-forward block, or null for a parallel block.</summary>
    public Module? FeedForwardNorm { get; }

    /// <summary>The feed-forward block.</summary>
    public FeedForward FeedForward { get; }

    /// <summary>Normalization after attention, or null.</summary>
    public Module? PostAttentionNorm { get; }

    /// <summary>Normalization after the feed-forward block, or null.</summary>
    public Module? PostFeedForwardNorm { get; }

    /// <summary>Dropout on each branch's output before its residual addition (active in training mode), or null.</summary>
    public Dropout? ResidualDropout { get; }

    private bool Dropping => ResidualDropout is { Probability: > 0f } d && d.IsTraining;

    /// <summary>
    /// The RMS normalization that follows this block (the next block's attention norm, or the final norm), set by
    /// <see cref="DecoderSpec"/>: when nothing records gradients, the block's last residual addition and that
    /// normalization run as one kernel and the norm reuses the result.
    /// </summary>
    internal RMSNorm? NextNorm { get; set; }

    /// <summary>True when attention and the feed-forward block read the same normalized input and are added together.</summary>
    public bool Parallel => FeedForwardNorm is null;

    /// <inheritdoc />
    public override IEnumerable<Module> Children() =>
        new Module?[] { AttentionNorm, Attention, FeedForwardNorm, FeedForward, PostAttentionNorm, PostFeedForwardNorm, ResidualDropout }.OfType<Module>();

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Run(input, x => Attention.Forward(x));

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context) =>
        Run(input, x => Attention.ForwardCached(x, context), x => Attention.HeadsCached(x, context));

    // attendHeads: attention without its output projection, which inference then runs together with the residual
    // addition and the next normalization where the device can (few rows, packed weights).
    private Tensor Run(Tensor input, Func<Tensor, Tensor> attend, Func<Tensor, Tensor>? attendHeads = null)
    {
        var normalized = AttentionNorm.Forward(input);
        using var compressNormalized = new CompressAfter(normalized);
        bool inference = !Autograd.IsEnabled && !Dropping;
        Tensor x, fed;
        if (!Parallel && inference && FeedForwardNorm is RMSNorm norm)
        {
            var (sum, fedInput) = attendHeads is not null && PostAttentionNorm is null
                ? AddProjected(attendHeads(normalized), Attention.Output, input, norm)
                : Tensor.AddRmsNormAffine(input, Attend(normalized, attend), norm.Gain, norm.Epsilon, norm.Offset);   // one pass
            x = sum;
            if (PostFeedForwardNorm is null && NextNorm is { } following && FeedForward.DownFusable)
            {
                var (output, nextInput) = AddProjected(FeedForward.DownInput(fedInput), FeedForward.Down, x, following);
                RMSNorm.HandOff(following, output, nextInput);
                return output;
            }

            fed = FeedForward.Forward(fedInput);
        }
        else
        {
            var attended = Attend(normalized, attend);
            if (Parallel)
            {
                var parallel = FeedForward.Forward(normalized);
                var partial = Dropping ? ResidualDropout!.AddTo(input, attended) : input + attended;
                var sum = Dropping ? ResidualDropout!.AddTo(partial, parallel) : partial + parallel;
                ActivationMemory.Release([sum, normalized], attended, parallel, partial);   // their backward reads no values
                return sum;
            }

            x = Dropping ? ResidualDropout!.AddTo(input, attended) : input + attended;          // dropout fused in
            ActivationMemory.Release([x, normalized], attended);
            var fedInput = FeedForwardNorm!.Forward(x);
            fed = FeedForward.Forward(fedInput);
            ActivationMemory.Compress(fedInput);
        }

        if (PostFeedForwardNorm is not null)
        {
            var unnormed = fed;
            fed = PostFeedForwardNorm.Forward(fed);
            if (PostFeedForwardNorm is RMSNorm)
            {
                ActivationMemory.Release(unnormed);
            }
        }

        if (NextNorm is { } next && !Autograd.IsEnabled && !Dropping)
        {
            var (output, nextInput) = Tensor.AddRmsNormAffine(x, fed, next.Gain, next.Epsilon, next.Offset);
            RMSNorm.HandOff(next, output, nextInput);
            return output;
        }

        var result = Dropping ? ResidualDropout!.AddTo(x, fed) : x + fed;
        // The residual sum is read by no backward step when its norm is an RMS norm (which keeps its own normalized values;
        // a layer norm reads its input).
        ActivationMemory.Release([result], fed, FeedForwardNorm is RMSNorm ? x : null);
        return result;
    }

    private Tensor Attend(Tensor normalized, Func<Tensor, Tensor> attend)
    {
        var attended = attend(normalized);
        if (PostAttentionNorm is null)
        {
            return attended;
        }

        var normed = PostAttentionNorm.Forward(attended);
        if (PostAttentionNorm is RMSNorm)
        {
            ActivationMemory.Release(attended);                       // an RMS norm keeps its own normalized values
        }

        return normed;
    }

    // residual + projection(h) and its normalization: one pass for packed weights and few rows, else the projection
    // followed by the fused addition and normalization.
    private static (Tensor Sum, Tensor Normalized) AddProjected(Tensor h, Linear projection, Tensor residual, RMSNorm norm) =>
        Tensor.MatMulPackedAddRmsNorm(h, projection, residual, norm)
        ?? Tensor.AddRmsNormAffine(residual, projection.Forward(h), norm.Gain, norm.Epsilon, norm.Offset);

    /// <inheritdoc />
    public override string ToString() => $"DecoderBlock({(Parallel ? "parallel" : "sequential")})";
}

/// <summary>
/// Adds a learned vector per position to [batch, time, dim] embeddings (GPT-2 style absolute positions, "pos.weight"
/// [maxPositions, dim]). Supports sequences up to <see cref="MaxPositions"/>.
/// </summary>
public sealed class PositionEmbedding : Module, ICachedModule
{
    /// <summary>Wraps a [maxPositions, dim] table; the layer takes ownership.</summary>
    public PositionEmbedding(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"Position embeddings must be [maxPositions, dim], got {Tensor.FormatShape(weight.Shape)}.");
        }

        Weight = weight;
    }

    /// <summary>The [maxPositions, dim] table.</summary>
    public Tensor Weight { get; private set; }

    /// <summary>Longest supported sequence.</summary>
    public int MaxPositions => Weight.Shape[0];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        int t = input.Shape[^2];
        if (t > MaxPositions)
        {
            throw new ArgumentException($"Sequence length {t} exceeds the {MaxPositions} learned positions.");
        }

        if (PackedSequences.Current is { } packing && input.Rank == 3 && packing.Matches(input.Shape[0], t))
        {
            // Packed sequences: each token's position within its own sequence.
            return input + Weight.EmbeddingLookup(packing.Positions).Reshape(input.Shape);
        }

        return input + (t == MaxPositions ? Weight : Weight.Narrow(0, 0, t));
    }

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context) => context.TokenPositions is { } positions
        ? input + Weight.EmbeddingLookup(positions).Reshape(input.Shape)                     // rows of different lengths
        : input + Weight.EmbeddingLookup(context.Positions ?? throw new InvalidOperationException("Call DecodingContext.BeginStep first."));

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [Weight];

    /// <inheritdoc />
    protected internal override void MoveTo(Device device) => Weight = MoveTensor(Weight, device);

    /// <inheritdoc />
    public override void Dispose()
    {
        Weight.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"PositionEmbedding({MaxPositions} x {Weight.Shape[1]})";
}

/// <summary>Switch for the fused tensor-core training paths of the decoder layers (tests compare them with the composed ones).</summary>
internal static class FusedTraining
{
    public static bool Enabled = true;
}

/// <summary>Multiplies its input by a constant (for example the √dim embedding scale of some models).</summary>
public sealed class Scale(float factor) : Module
{
    /// <summary>The factor.</summary>
    public float Factor { get; } = factor;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input * Factor;

    /// <inheritdoc />
    public override string ToString() => $"Scale({Factor})";
}

/// <summary>
/// The rotary cos/sin tables ([positions, half]), one per settings, head size, length and device, shared by every
/// attention layer that uses them (a model's layers all do) and freed when the last of those layers is disposed.
/// </summary>
internal static class RopeTables
{
    private sealed class Entry(Tensor cos, Tensor sin)
    {
        public Tensor Cos { get; } = cos;
        public Tensor Sin { get; } = sin;
        public int Users;
    }

    private static readonly Dictionary<(RopeSettings, int, int, Device), Entry> Tables = new();
    private static readonly Lock Gate = new();

    public static (Tensor Cos, Tensor Sin) Acquire(RopeSettings rope, int headDim, int maxPositions, Device device)
    {
        var key = (rope, headDim, maxPositions, device);
        lock (Gate)
        {
            if (!Tables.TryGetValue(key, out var entry))
            {
                // The scaled frequencies (per position for a scaling that depends on it), and the attention factor on both
                // tables, as transformers multiplies cos and sin by it.
                var scaled = rope.Scaled(headDim);
                int half = scaled.Frequencies.Length;
                double factor = scaled.AttentionFactor;
                var cos = new float[maxPositions * half];
                var sin = new float[maxPositions * half];
                Parallel.For(0, maxPositions, ComputeResources.ParallelOptions, p =>
                {
                    var frequencies = scaled.FrequenciesAt?.Invoke(p) ?? scaled.Frequencies;
                    for (int i = 0; i < half; i++)
                    {
                        double angle = p * frequencies[i];
                        cos[p * half + i] = (float)(Math.Cos(angle) * factor);
                        sin[p * half + i] = (float)(Math.Sin(angle) * factor);
                    }
                });
                entry = new Entry(Tensor.Persistent(cos, [maxPositions, half], device, requiresGrad: false),
                    Tensor.Persistent(sin, [maxPositions, half], device, requiresGrad: false));
                Tables[key] = entry;
            }

            entry.Users++;
            return (entry.Cos, entry.Sin);
        }
    }

    public static void Release(RopeSettings rope, int headDim, int maxPositions, Device device)
    {
        var key = (rope, headDim, maxPositions, device);
        lock (Gate)
        {
            if (Tables.TryGetValue(key, out var entry) && --entry.Users == 0)
            {
                Tables.Remove(key);
                entry.Cos.Dispose();
                entry.Sin.Dispose();
            }
        }
    }
}
