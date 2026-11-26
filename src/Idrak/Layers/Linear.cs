// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// Fully connected layer: y = x · W + b, mapping [..., inFeatures] to [..., outFeatures]
/// (any leading dimensions, e.g. [batch, time, features] for sequences).
/// Weights start Xavier/Glorot-uniform and the bias starts at zero.
/// </summary>
public sealed partial class Linear : Module, IAdaptableLinear, ITiedWeights, INamedWeights, IOffloadableLayer
{
    /// <summary>Creates the layer on <paramref name="device"/> (default: <see cref="Device.Default"/>).</summary>
    /// <param name="inFeatures">Size of each input row.</param>
    /// <param name="outFeatures">Size of each output row.</param>
    /// <param name="bias">Whether to learn an additive bias.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Source of the initial weights; pass a seeded <see cref="System.Random"/> for reproducible runs.</param>
    public Linear(int inFeatures, int outFeatures, bool bias = true, Device? device = null, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        device ??= Device.Default;
        random ??= Random.Shared;

        float limit = MathF.Sqrt(6f / (inFeatures + outFeatures));
        _weight = CreateParameter(UniformValues(inFeatures * outFeatures, limit, random), [inFeatures, outFeatures], device);
        Bias = bias ? CreateParameter(new float[outFeatures], [outFeatures], device) : null;
    }

    private Linear(int inFeatures, int outFeatures, Tensor? weight, Tensor? bias)
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = weight;
        Bias = bias;
    }

    /// <summary>
    /// A layer around existing weights [in, out] (and optionally a bias [out]); the layer takes ownership. Used to build
    /// models from loaded weights without allocating random ones first.
    /// </summary>
    public static Linear FromWeights(Tensor weight, Tensor? bias = null)
    {
        if (weight.Rank != 2 || (bias is not null && (bias.Rank != 1 || bias.Shape[0] != weight.Shape[1])))
        {
            throw new ArgumentException($"Linear weights must be [in, out] with a bias [out]; got {Tensor.FormatShape(weight.Shape)} and {(bias is null ? "no bias" : Tensor.FormatShape(bias.Shape))}.");
        }

        return new Linear(weight.Shape[0], weight.Shape[1], weight, bias);
    }

    /// <summary>
    /// An output layer tied to <paramref name="embedding"/>: y = x · Eᵀ (+ b) with the embedding's [vocabulary, dim] table,
    /// read in place (one copy of the table; gradients reach the shared table, as in tied language models). Merging an
    /// adapter gives the layer its own copy first.
    /// </summary>
    public static Linear Tied(Embedding embedding, Tensor? bias = null) =>
        new(embedding.Dim, embedding.Vocabulary, null, bias) { _tiedTo = embedding };

    /// <summary>The embedding this layer reads its weights from (see <see cref="Tied"/>), else null.</summary>
    public Embedding? TiedTo => _tiedTo;

    /// <inheritdoc />
    Module? ITiedWeights.WeightsOf => _tiedTo;

    private Embedding? _tiedTo;

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>The [inFeatures, outFeatures] weight matrix.</summary>
    public Tensor Weight => _weight ?? throw new InvalidOperationException($"{this} reads the embedding table of {_tiedTo} (transposed); use TiedTo.Weight.");

    private Tensor? _weight;

    /// <summary>The [outFeatures] bias, or null when created with <c>bias: false</c>.</summary>
    public Tensor? Bias { get; private set; }

    /// <summary>
    /// The adapter on this layer, or null: a LoRA adapter (<see cref="LoraAdapter"/>, added by
    /// <see cref="ModuleExtensions.AddLora"/>: the output is <c>x·W + b + (x·A·B)·scale</c>), a DoRA adapter
    /// (<see cref="DoraAdapter"/>, <see cref="ModuleExtensions.AddDora"/>) or one of your own implementing
    /// <see cref="ILinearAdapter"/>. Its parameters come after W and b in <see cref="Parameters"/>. Setting it does not
    /// freeze anything; create the optimizer afterwards.
    /// </summary>
    public ILinearAdapter? Adapter { get; set; }

    /// <summary>The adapter when it is a plain LoRA adapter, else null.</summary>
    public LoraAdapter? Lora => Adapter as LoraAdapter;

    /// <summary>
    /// Soft-caps the outputs, cap · tanh(y / cap), or null for none: Gemma 2's final logit soft-capping on an output head,
    /// applied wherever the layer runs (generation, losses, scoring).
    /// </summary>
    public float? OutputSoftcap
    {
        get => _softcap;
        set => _softcap = value is null or > 0f ? value : throw new ArgumentOutOfRangeException(nameof(OutputSoftcap), "The cap must be positive.");
    }

    private float? _softcap;

    /// <summary>A float32 weight of its own, no adapter: the layer is x·W (+ b), which fused operations may compute themselves.</summary>
    internal static bool PlainFloat(Linear layer) => layer._weight is not null && layer._tiedTo is null && layer.Adapter is null && layer._softcap is null;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        using var tied = _tiedTo is { } table ? ForwardHooks.Enter(table) : null;   // a sharded run gathers the table it reads (plan 23, C6)
        var y = Project(input);
        return _softcap is { } cap ? (y * (1f / cap)).Tanh() * cap : y;
    }

    // x·W + b (and the adapter's update).
    private Tensor Project(Tensor input)
    {
        if (Bias is not null && _weight is not null && _tiedTo is null && Adapter is null && input.Rank >= 2)
        {
            return Tensor.MatMulBias(input, _weight, Bias);                 // one pass on tensor cores
        }

        var product = ProjectWithoutBias(input);
        if (Bias is null)
        {
            return product;
        }

        var biased = product + Bias;
        ActivationMemory.Release(product);                              // the bias's backward does not read it
        return biased;
    }

    /// <summary>
    /// The outputs of several layers applied to the same input. When nothing needs gradients and the layers are plain
    /// float projections (no adapter) with few input rows, they run as one device pass over the input
    /// (token-by-token decoding: query/key/value, gate/up); otherwise each layer runs on its own.
    /// </summary>
    internal static Tensor[] ForwardMany(Tensor input, params Linear[] layers)
    {
        using var gathered = ForwardHooks.Listening ? Offloading.Gather(layers) : null;   // a sharded run's weights of every layer (plan 23, C6)
        using var offload = Offloading.EnterMany(layers, input);       // offloaded weights of all the layers staged together
        int k = input.Shape[^1], rows = input.Size / Math.Max(1, k);
        var capabilities = input.Backend.Capabilities;
        bool fusedKernels = input.Backend.FusedKernels;
        // The checks are plain loops: no closure or delegate is allocated per call (this runs per block and decoding step).
        bool fused = !Autograd.IsEnabled && layers.Length is > 1 and <= 3 && rows <= capabilities.FewRows && fusedKernels
            && AllPlainFloat(layers, k);
        if (fused)
        {
            var weights = new Tensor[layers.Length];
            var biases = new Tensor?[layers.Length];
            for (int j = 0; j < layers.Length; j++)
            {
                (weights[j], biases[j]) = (layers[j].Weight, layers[j].Bias);
            }

            return Tensor.MatMulMany(input, weights, biases);
        }

        var results = new Tensor[layers.Length];

        // Training: the layers read one flattened view of the input, so their input gradients add up in its one buffer
        // (each product adds into it) instead of each view's gradient being added into the input by a separate pass.
        if (Autograd.IsEnabled && input.RequiresGrad && layers.Length > 1 && input.Rank > 2)
        {
            var flat = input.Reshape(-1, k);
            for (int j = 0; j < layers.Length; j++)
            {
                var y = layers[j].Forward(flat);
                results[j] = y.Rank == 2 ? y.Reshape([.. input.Shape[..^1], layers[j].OutFeatures]) : y;
            }

            return results;
        }

        for (int j = 0; j < layers.Length; j++)
        {
            results[j] = layers[j].Forward(input);
        }

        return results;
    }

    // Float32 weights of their own, no adapter, reading k features, and wide enough for the one-pass product.
    private static bool AllPlainFloat(Linear[] layers, int k)
    {
        foreach (var l in layers)
        {
            if (l.TiedTo is not null || l.Adapter is not null || l.InFeatures != k || (long)l.OutFeatures * k < 1 << 16)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>gelu(x·W + b) for inference (a feed-forward block's first projection): x·W then <see cref="Tensor.BiasGelu"/> (without a bias, the layer then gelu).</summary>
    internal Tensor ProjectGelu(Tensor input) => Bias is { } bias ? ProjectWithoutBias(input).BiasGelu(bias) : Forward(input).Gelu();

    /// <summary>x·W + b for inference: the layer.</summary>
    internal Tensor ProjectBiased(Tensor input) => Forward(input);

    /// <summary>residual + (x·W + b) for inference (an output projection and its residual): the layer, then the sum.</summary>
    internal Tensor ProjectAdding(Tensor input, Tensor residual) => residual + Forward(input);

    /// <summary>x·W, adapted by the adapter when one is attached.</summary>
    internal Tensor ProjectWithoutBias(Tensor input)
    {
        var product = BaseProduct(input);
        return Adapter is { } a ? a.Forward(this, input, product) : product;
    }

    /// <summary>x·W with the layer's own weights (float or tied), without bias or adapter.</summary>
    /// <inheritdoc cref="ILinearLayer.BaseProduct"/>
    Tensor ILinearLayer.BaseProduct(Tensor input) => BaseProduct(input);

    /// <inheritdoc cref="ILinearLayer.WeightValues"/>
    float[] ILinearLayer.WeightValues() => WeightValues();

    /// <inheritdoc />
    Device IAdaptableLinear.Device => Device;

    /// <inheritdoc />
    void IAdaptableLinear.MergeAdapter() => MergeAdapter();

    /// <inheritdoc />
    /// <remarks>"weight" [in, out] and "bias" when the layer has one; nothing for a tied layer (the embedding stores the table).</remarks>
    IEnumerable<(string Name, Tensor Tensor)> INamedWeights.NamedWeights() =>
        _tiedTo is not null ? [] : Bias is { } bias ? [("weight", Weight), ("bias", bias)] : [("weight", Weight)];

    /// <inheritdoc />
    IEnumerable<Tensor> IOffloadableLayer.StagedParameters() => Parameters();

    /// <inheritdoc />
    IEnumerable<Tensor> IOffloadableLayer.StagedBuffers() => Buffers();

    internal Tensor BaseProduct(Tensor input) => _tiedTo is { } e ? TiedProduct(input, e.Weight) : input.MatMul(Weight);

    // The tied head (x · Eᵀ), the table read as stored.
    private static Tensor TiedProduct(Tensor input, Tensor table) => input.MatMul(table, transposeB: true);

    /// <inheritdoc />
    public override void Dispose()
    {
        (Adapter as IDisposable)?.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters()
    {
        IEnumerable<Tensor> own = (_weight, Bias) switch
        {
            (null, null) => [],
            (null, { } b) => [b],
            ({ } w, null) => [w],
            ({ } w, { } b) => [w, b],
        };
        return Adapter is { } a ? own.Concat(a.Parameters) : own;
    }

    // The float weight values.
    internal float[] WeightValues() => _tiedTo is not null ? TiedValues() : Weight.ToArray();

    internal Device Device => _tiedTo?.Device ?? _weight!.Device;

    // The tied table transposed to [in, out] on the host.
    private float[] TiedValues() => HostParallel.Transpose(_tiedTo!.WeightValues(), OutFeatures, InFeatures);

    // A tied layer gets its own float copy of the (transposed) table before it is converted or merged.
    private void Untie()
    {
        if (_tiedTo is { } e)
        {
            _weight = Tensor.Persistent(TiedValues(), [InFeatures, OutFeatures], e.Device, requiresGrad: false);
            _tiedTo = null;
        }
    }

    internal void MergeAdapter()
    {
        if (Adapter is not { } a)
        {
            return;
        }

        Untie();
        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            var merged = a.Merge(this, Weight);
            Weight.Load(merged.ToArray());
        }

        Adapter = null;
        foreach (var p in a.Parameters)
        {
            p.Dispose();
        }

        (a as IDisposable)?.Dispose();
    }

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        _weight = _weight is null ? null : MoveTensor(_weight, device);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
        if (Adapter is { } a)
        {
            Adapter = a.MoveTo(t => MoveTensor(t, device));
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"Linear({InFeatures} -> {OutFeatures}{(Bias is null ? ", no bias" : "")}{(_tiedTo is null ? "" : ", tied")}{Adapter switch { null => "", LoraAdapter a => $", LoRA rank {a.Rank}", DoraAdapter d => $", DoRA rank {d.Rank}", var other => $", {other.GetType().Name}" }}{(_softcap is { } c ? $", softcap {c}" : "")})";
}
