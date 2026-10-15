// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.IdrakAudio;

/// <summary>
/// The IdrakAudio encoder and CTC head: log-mel features [batch, frames, mels] through <see cref="IdrakAudioConfig.SubsampleLayers"/>
/// convolutions of (1, k) stride (1, 2) over time with the mels as channels (each output past an utterance's frames set
/// to zero, so a batch gives each utterance what it gives alone), sinusoidal positions, pre-norm blocks whose attention
/// sees each utterance's own steps (one key range per utterance, through the library's span attention), a final layer
/// normalization and a linear head to the classes: [batch, steps, classes] logits.
/// </summary>
public sealed class IdrakAudioEncoder : Module
{
    // The batch the blocks are running on (its key ranges and its padding), set per pass on the calling thread; a block
    // recomputed in the backward pass (activation checkpointing) reads the same, since the backward runs on that thread
    // before the next pass sets another.
    [ThreadStatic]
    private static BatchLayout? t_layout;

    private readonly Conv2d[] _subsample;
    private readonly IdrakAudioBlock[] _blocks;
    private readonly LayerNorm _norm;
    private readonly Linear? _head;
    private readonly Lock _gate = new();
    private readonly List<Tensor> _tables = [];
    private Tensor? _positions;

    internal IdrakAudioEncoder(IdrakAudioConfig config, IdrakAudioWeights weights)
    {
        Config = config;
        int d = config.Dim, k = config.SubsampleKernel;
        _subsample = new Conv2d[config.SubsampleLayers];
        for (int i = 0; i < _subsample.Length; i++)
        {
            int inputs = i == 0 ? config.Mels : d;
            _subsample[i] = Conv2d.FromWeights(weights.Convolution($"subsample.{i}.weight", d, inputs, k), weights.Bias($"subsample.{i}.bias", d),
                (1, k), (1, 2), (0, k / 2));
            _subsample[i].Name = $"subsample.{i}";
        }

        _blocks = new IdrakAudioBlock[config.Layers];
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i] = new IdrakAudioBlock(config, weights, $"layers.{i}.");
        }

        int width = config.SubsampleLayers == 0 ? config.Mels : d;
        if (width != d)
        {
            throw new InvalidDataException($"Without a subsampler the features ({config.Mels} mels) are the blocks' input; give dim {config.Mels} or a subsampler.");
        }

        _norm = weights.Norm("norm", d, config.NormEpsilon);
        _norm.Name = "norm";
        if (!config.Sequence)
        {
            _head = weights.Linear("head", d, config.Classes);                // a seq2seq model's decoder reads the normed states
            _head.Name = "head";
        }
    }

    /// <summary>The settings it was built for.</summary>
    public IdrakAudioConfig Config { get; }

    internal static BatchLayout Layout => t_layout ?? throw new InvalidOperationException("An IdrakAudio block runs inside its encoder's pass.");

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _head is null ? [.. _subsample, .. _blocks, _norm] : [.. _subsample, .. _blocks, _norm, _head];

    /// <summary>Every utterance of <paramref name="input"/> [batch, frames, mels] taken as all frames.</summary>
    protected override Tensor ForwardCore(Tensor input) => Run(input, null, checkpointing: false);

    /// <summary>
    /// The logits of features [batch, frames, mels] whose utterance n has <paramref name="frames"/>[n] frames (all when
    /// null), zeros after them: [batch, steps, classes]; a "seq2seq" encoder's normed states [batch, steps, dim].
    /// </summary>
    public Tensor Run(Tensor features, IReadOnlyList<int>? frames, bool checkpointing)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Rank != 3 || features.Shape[2] != Config.Mels)
        {
            throw new ArgumentException($"The IdrakAudio encoder takes features [batch, frames, {Config.Mels}], got {Tensor.FormatShape(features.Shape)}.");
        }

        int n = features.Shape[0], t = features.Shape[1], d = Config.Dim, k = Config.SubsampleKernel;
        var steps = new int[n];
        for (int i = 0; i < n; i++)
        {
            steps[i] = frames is null ? t : frames[i];
            if ((uint)steps[i] > (uint)t)
            {
                throw new ArgumentException($"Utterance {i} has {steps[i]} frames; the batch holds {t}.");
            }
        }

        Tensor x;
        if (_subsample.Length == 0)
        {
            x = features;
        }
        else
        {
            x = features.Permute(0, 2, 1).Reshape(n, Config.Mels, 1, t);       // [n, mels, 1, t]: the mels as channels
            for (int s = 0; s < _subsample.Length; s++)
            {
                x = Activate(_subsample[s].Forward(x));                       // [n, d, 1, t']
                t = x.Shape[3];
                for (int i = 0; i < n; i++)
                {
                    steps[i] = (steps[i] + 2 * (k / 2) - k) / 2 + 1;
                }

                var rows = x.Reshape(n, d, t).Permute(0, 2, 1);                 // [n, t', d]
                if (steps.Any(v => v < t))
                {
                    rows = rows.GroupAffine(Keep(steps, t, x.Device), null, n * t, d);    // padding zeroed: the next convolution reads zeros there
                }

                x = s + 1 < _subsample.Length ? rows.Permute(0, 2, 1).Reshape(n, d, 1, t) : rows;
            }
        }

        if (Config.Positions == "sinusoidal")
        {
            x = x + Positions(t, x.Device);
        }

        // Left set after the pass: a checkpointed block runs again in the backward pass, after this returns.
        t_layout = new BatchLayout(n, t, steps, x.Device);
        foreach (var block in _blocks)
        {
            x = checkpointing && IsTraining ? block.ForwardCheckpointed(x) : block.Forward(x);
        }

        var normed = _norm.Forward(x);
        return _head is null ? normed : _head.Forward(normed);
    }

    internal Tensor Activate(Tensor x) => Config.Activation == "gelu" ? ExactGELU.Apply(x) : x.Gelu();

    // 1 on each utterance's steps, 0 after them: one value per [utterance, step] row.
    internal static Tensor Keep(int[] steps, int t, Device device)
    {
        var keep = new float[steps.Length * t];
        for (int i = 0; i < steps.Length; i++)
        {
            keep.AsSpan(i * t, steps[i]).Fill(1f);
        }

        return Tensor.From(keep, [keep.Length], device);
    }

    // sin(p / 10000^(2⌊i/2⌋/d)) at even i, cos at odd, [t, d]: built for twice the longest length asked so far and narrowed;
    // a table outgrown stays alive until Dispose (another thread may be reading it), so tables are made a few times at most.
    private Tensor Positions(int t, Device device)
    {
        Tensor table;
        lock (_gate)
        {
            if (_positions is null || _positions.Shape[0] < t || _positions.Device != device)
            {
                int d = Config.Dim, rows = Math.Max(t, 2 * (_positions?.Shape[0] ?? 0));
                var values = new float[rows * d];
                for (int p = 0; p < rows; p++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        double angle = p / Math.Pow(10000, 2 * (i / 2) / (double)d);
                        values[p * d + i] = (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
                    }
                }

                _positions = CreateBuffer(values, [rows, d], device);
                _tables.Add(_positions);
            }

            table = _positions;
        }

        return table.Shape[0] == t ? table : table.Narrow(0, 0, t);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => _positions is null ? base.Buffers() : [_positions, .. base.Buffers()];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        Release();
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        Release();
        base.Dispose();
    }

    private void Release()
    {
        lock (_gate)
        {
            _tables.ForEach(t => t.Dispose());
            _tables.Clear();
            _positions = null;
        }
    }

    /// <inheritdoc />
    public override string ToString() => $"IdrakAudioEncoder(dim {Config.Dim}, {Config.Layers} blocks{(_head is null ? "" : $", {Config.Classes} classes")})";

    // A batch's layout for the blocks: each utterance's steps as one key range per utterance ([n · t] starts and ends, a
    // table per utterance), or one table when every utterance has every step; the padding's row mask when there is one.
    internal sealed class BatchLayout
    {
        public BatchLayout(int n, int t, int[] steps, Device device)
        {
            (Batch, Steps, Lengths) = (n, t, steps);
            Full = steps.All(s => s == t);
            int tables = Full ? 1 : n;
            var starts = new float[tables * t];
            var ends = new float[tables * t];
            for (int i = 0; i < tables; i++)
            {
                ends.AsSpan(i * t, t).Fill(Full ? t : steps[i]);
            }

            Starts = Tensor.From(starts, [starts.Length], device);
            Ends = Tensor.From(ends, [ends.Length], device);
            Keep = Full ? null : IdrakAudioEncoder.Keep(steps, t, device);
        }

        public int Batch { get; }

        public int Steps { get; }

        public int[] Lengths { get; }

        public bool Full { get; }

        public Tensor Starts { get; }

        public Tensor Ends { get; }

        public Tensor? Keep { get; }
    }
}

/// <summary>
/// One pre-norm IdrakAudio block over [batch, steps, dim]: x + Attention(LayerNorm(x)) with each utterance's queries seeing
/// its own steps only; then, with <see cref="IdrakAudioConfig.ConvolutionModule"/>, x + Convolution(LayerNorm(x)) (pointwise
/// to 2 · dim, GLU, padding zeroed, a depthwise (1, k) convolution over time, LayerNorm, SiLU, pointwise); then
/// x + FeedForward(LayerNorm(x)).
/// </summary>
public sealed class IdrakAudioBlock : Module
{
    private readonly IdrakAudioConfig _config;
    private readonly LayerNorm _norm1, _norm2;
    private readonly Linear _qkv, _output, _feedForward1, _feedForward2;
    private readonly LayerNorm? _convNorm, _depthwiseNorm;
    private readonly Linear? _pointwise1, _pointwise2;
    private readonly Conv2d? _depthwise;
    private readonly Dropout? _dropout;

    internal IdrakAudioBlock(IdrakAudioConfig config, IdrakAudioWeights weights, string prefix)
    {
        _config = config;
        int d = config.Dim;
        _norm1 = weights.Norm(prefix + "norm1", d, config.NormEpsilon);
        _qkv = weights.Linear(prefix + "self_attn.in_proj", d, 3 * d, weightName: prefix + "self_attn.in_proj_weight", biasName: prefix + "self_attn.in_proj_bias");
        _output = weights.Linear(prefix + "self_attn.out_proj", d, d);
        if (config.ConvolutionModule)
        {
            int k = config.ConvolutionKernel;
            _convNorm = weights.Norm(prefix + "conv_norm", d, config.NormEpsilon);
            _pointwise1 = weights.Linear(prefix + "conv.pointwise1", d, 2 * d);
            _depthwise = Conv2d.FromWeights(weights.Convolution(prefix + "conv.depthwise.weight", d, 1, k), weights.Bias(prefix + "conv.depthwise.bias", d),
                (1, k), (1, 1), (0, k / 2), groups: d);
            _depthwiseNorm = weights.Norm(prefix + "conv.depthwise_norm", d, config.NormEpsilon);
            _pointwise2 = weights.Linear(prefix + "conv.pointwise2", d, d);
        }

        _norm2 = weights.Norm(prefix + "norm2", d, config.NormEpsilon);
        _feedForward1 = weights.Linear(prefix + "linear1", d, config.FfnDim);
        _feedForward2 = weights.Linear(prefix + "linear2", config.FfnDim, d);
        _dropout = config.Dropout > 0 ? new Dropout(config.Dropout) : null;
        Name = prefix.TrimEnd('.');
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children()
    {
        var children = new List<Module> { _norm1, _qkv, _output };
        if (_depthwise is not null)
        {
            children.AddRange([_convNorm!, _pointwise1!, _depthwise, _depthwiseNorm!, _pointwise2!]);
        }

        children.AddRange([_norm2, _feedForward1, _feedForward2]);
        if (_dropout is not null)
        {
            children.Add(_dropout);
        }

        return children;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        var layout = IdrakAudioEncoder.Layout;
        int n = input.Shape[0], t = input.Shape[1], d = _config.Dim, heads = _config.Heads, dh = d / heads;
        var qkv = _qkv.Forward(_norm1.Forward(input));                         // [n, t, 3d]
        Tensor Heads(int part) => qkv.Narrow(2, part * d, d).Reshape(n, t, heads, dh).Permute(0, 2, 1, 3).Reshape(n * heads, t, dh);
        var context = Tensor.AttentionFastest(Heads(0), Heads(1), Heads(2), layout.Starts, layout.Ends, 1f / MathF.Sqrt(dh), everyKey: layout.Full)
            .Reshape(n, heads, t, dh).Permute(0, 2, 1, 3).Reshape(n, t, d);
        var x = input + Drop(_output.Forward(context));
        if (_depthwise is not null)
        {
            var h = _pointwise1!.Forward(_convNorm!.Forward(x));                // [n, t, 2d]
            h = h.Narrow(2, 0, d) * h.Narrow(2, d, d).Sigmoid();               // GLU
            if (layout.Keep is { } keep)
            {
                h = h.GroupAffine(keep, null, n * t, d);                       // padding zeroed before the convolution over time
            }

            h = _depthwise.Forward(h.Permute(0, 2, 1).Reshape(n, d, 1, t)).Reshape(n, d, t).Permute(0, 2, 1);
            x = x + Drop(_pointwise2!.Forward(_depthwiseNorm!.Forward(h).Silu()));
        }

        var hidden = _feedForward1.Forward(_norm2.Forward(x));
        hidden = _config.Activation == "gelu" ? ExactGELU.Apply(hidden) : hidden.Gelu();
        return x + Drop(_feedForward2.Forward(hidden));
    }

    private Tensor Drop(Tensor x) => _dropout is null ? x : _dropout.Forward(x);

    /// <inheritdoc />
    public override string ToString() => $"IdrakAudioBlock(dim {_config.Dim}{(_depthwise is null ? "" : ", convolution module")})";
}
