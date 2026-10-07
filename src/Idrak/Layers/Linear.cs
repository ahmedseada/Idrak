// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// Fully connected layer: y = x · W + b, mapping [..., inFeatures] to [..., outFeatures]
/// (any leading dimensions, e.g. [batch, time, features] for sequences).
/// Weights start Xavier/Glorot-uniform and the bias starts at zero.
/// </summary>
public sealed partial class Linear : Module, ILinearLayer
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

    private Linear(int inFeatures, int outFeatures, Tensor? weight, PackedWeight? packed, Tensor? bias)
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = weight;
        _packed = packed;
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

        return new Linear(weight.Shape[0], weight.Shape[1], weight, null, bias);
    }

    /// <summary>
    /// A layer around existing packed weights (int8, 4-bit, bfloat16, or a format of your own derived from
    /// <see cref="Abstraction.Generation.PackedWeight"/>); the layer takes ownership.
    /// </summary>
    public static Linear FromPacked(PackedWeight weight, Tensor? bias = null) => new(weight.Rows, weight.Columns, null, weight, bias);

    /// <summary>A layer around existing int8 weights (see <see cref="ModuleExtensions.QuantizeInt8"/>); the layer takes ownership.</summary>
    public static Linear FromInt8(Int8Weight weight, Tensor? bias = null) => new(weight.Rows, weight.Columns, null, weight, bias);

    /// <summary>A layer around existing bfloat16 weights (see <see cref="ModuleExtensions.ToBFloat16"/>); the layer takes ownership.</summary>
    public static Linear FromBFloat16(BFloat16Weight weight, Tensor? bias = null) => new(weight.Rows, weight.Columns, null, weight, bias);

    /// <summary>
    /// An output layer tied to <paramref name="embedding"/>: y = x · Eᵀ (+ b) with the embedding's [vocabulary, dim] table,
    /// read in place (one copy of the table; gradients reach the shared table, as in tied language models). Converting
    /// the layer (int8, int4, bfloat16) or merging an adapter gives it its own copy first.
    /// </summary>
    public static Linear Tied(Embedding embedding, Tensor? bias = null) =>
        new(embedding.Dim, embedding.Vocabulary, null, null, bias) { _tiedTo = embedding };

    /// <summary>The embedding this layer reads its weights from (see <see cref="Tied"/>), else null.</summary>
    public Embedding? TiedTo => _tiedTo;

    private Embedding? _tiedTo;

    /// <summary>A layer around existing 4-bit weights (see <see cref="ModuleExtensions.QuantizeInt4"/>); the layer takes ownership.</summary>
    public static Linear FromInt4(Int4Weight weight, Tensor? bias = null) => new(weight.Rows, weight.Columns, null, weight, bias);

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>The [inFeatures, outFeatures] weight matrix.</summary>
    public Tensor Weight => _weight ?? throw new InvalidOperationException(_tiedTo is not null
        ? $"{this} reads the embedding table of {_tiedTo} (transposed); use TiedTo.Weight."
        : $"{this} holds {_packed!.Description} weights (see {_packed.GetType().Name}); call {_packed.FloatMethod} on the model to get float weights back.");

    /// <summary>The packed weights (int8, int4, bfloat16 or a format of your own) when the layer holds them, else null (float weights).</summary>
    public PackedWeight? PackedWeight => _packed;

    private PackedWeight? _packed;

    /// <summary>The 4-bit weights when the layer holds them (see <see cref="ModuleExtensions.QuantizeInt4"/>), else null.</summary>
    public Int4Weight? Int4 => _packed as Int4Weight;

    // Whether the weights are packed (int8, int4 or bfloat16) rather than a float tensor.
    internal bool Packed => _packed is not null;

    /// <summary>The bfloat16 weights when the layer holds them (see <see cref="ModuleExtensions.ToBFloat16"/>), else null.</summary>
    public BFloat16Weight? BFloat16 => _packed as BFloat16Weight;

    /// <summary>The int8 weights when the layer was quantized with <see cref="ModuleExtensions.QuantizeInt8"/>, else null.</summary>
    public Int8Weight? Int8 => _packed as Int8Weight;

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

    /// <summary>The adapter when it is a plain LoRA adapter (the fused LoRA products read only those), else null.</summary>
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

        if (Bias is { RequiresGrad: false } && Lora is not null && Linear.LoraProducts(input, [this]) is [var withBias])
        {
            return withBias;                                            // the frozen bias added inside the fused product
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
    /// float projections (no adapter, not int8) with few input rows, they run as one device pass over the input
    /// (token-by-token decoding: query/key/value, gate/up); otherwise each layer runs on its own.
    /// </summary>
    internal static Tensor[] ForwardMany(Tensor input, params Linear[] layers)
    {
        using var offload = Offloading.EnterMany(layers, input);       // offloaded weights of all the layers staged together
        int k = input.Shape[^1], rows = input.Size / Math.Max(1, k);
        var capabilities = input.Backend.Capabilities;
        bool fused = !Autograd.IsEnabled && layers.Length is > 1 and <= 3 && rows <= capabilities.FewRows && capabilities.FusedKernels
            && layers.All(l => !l.Packed && l.TiedTo is null && l.Adapter is null && l.InFeatures == k && (long)l.OutFeatures * k >= 1 << 16);
        if (fused)
        {
            return Tensor.MatMulMany(input, [.. layers.Select(l => l.Weight)], [.. layers.Select(l => l.Bias)]);
        }

        // Packed weights of one built-in kind (int8, int4, bfloat16): one pass where the device has one (few rows; for
        // prompts, one tensor-core launch over all the layers' columns). Formats of one's own have no Format and no such
        // pass: each layer runs its own product.
        var format = layers[0].PackedWeight?.Format;
        bool packed = format is not null && !Autograd.IsEnabled && layers.Length is > 1 and <= 3
            && (rows <= capabilities.FewRows || layers.All(l => l.Bias is null))                  // prompts: one launch
            && layers.All(l => l.Adapter is null && l.InFeatures == k && l.PackedWeight?.Format == format);
        if (packed && Linear.MatMulPackedMany(input, format!.Value, layers) is { } outputs)
        {
            return outputs;
        }

        // Adapters on every layer (LoRA / QLoRA, training or evaluation): one pass with each low-rank term inside its product.
        if (layers.All(l => l.Lora is not null && l.Bias is not { RequiresGrad: true }) && Linear.LoraProducts(input, layers) is { } lora)
        {
            return lora;
        }

        // Training over frozen packed layers (LoRA / QLoRA): the base products still run as one pass, recorded, and each
        // layer's adapter adds its low-rank term into its output.
        bool training = format is not null && Autograd.IsEnabled && layers.Length is > 1 and <= 3 && capabilities.FusedKernels
            && layers.All(l => l.Bias is null && l.InFeatures == k && l.PackedWeight?.Format == format);
        if (training && Linear.MatMulPackedManyRecorded(input, format!.Value, layers) is { } products)
        {
            return [.. products.Select((product, j) => layers[j].Adapter is { } a ? a.Forward(layers[j], input, product) : product)];
        }

        // Training: the layers read one flattened view of the input, so their input gradients add up in its one buffer
        // (each product adds into it) instead of each view's gradient being added into the input by a separate pass.
        if (Autograd.IsEnabled && input.RequiresGrad && layers.Length > 1 && input.Rank > 2)
        {
            var flat = input.Reshape(-1, k);
            return [.. layers.Select(l => l.Forward(flat) is var y && y.Rank == 2 ? y.Reshape([.. input.Shape[..^1], l.OutFeatures]) : y)];
        }

        return [.. layers.Select(l => l.Forward(input))];
    }

    /// <summary>x·W, adapted by the adapter when one is attached.</summary>
    internal Tensor ProjectWithoutBias(Tensor input)
    {
        if (Lora is not null && Linear.LoraProducts(input, [this], withBias: false) is [var fused])
        {
            return fused;
        }

        var product = BaseProduct(input);
        return Adapter is { } a ? a.Forward(this, input, product) : product;
    }

    /// <summary>x·W with the layer's own weights (float, packed or tied), without bias or adapter.</summary>
    /// <inheritdoc cref="ILinearLayer.BaseProduct"/>
    Tensor ILinearLayer.BaseProduct(Tensor input) => BaseProduct(input);

    /// <inheritdoc cref="ILinearLayer.WeightValues"/>
    float[] ILinearLayer.WeightValues() => WeightValues();

    internal Tensor BaseProduct(Tensor input) =>
        _packed is { } packed ? packed.MatMul(input) : _tiedTo is { } e ? TiedProduct(input, e.Weight) : input.MatMul(Weight);

    // The tied head (x · Eᵀ). While the table is frozen, large products read a transposed bfloat16 copy made once (the
    // tensor-core kernels would otherwise copy the whole table transposed on every call; bfloat16 is what they multiply in
    // anyway, at half the memory of a float copy: 0.45 GB instead of 0.9 GB for a 152k × 1536 table). The input's gradient
    // reads the table as stored.
    private Tensor TiedProduct(Tensor input, Tensor table)
    {
        int rows = input.Size / Math.Max(1, InFeatures);
        if (table.RequiresGrad || rows < 64 || !table.Backend.Capabilities.MatrixUnits || !MixedPrecision.UsesTensorCores)
        {
            _tiedTransposed?.Dispose();
            _tiedTransposed = null;
            return input.MatMul(table, transposeB: true);
        }

        _tiedTransposed ??= BFloat16Weight.FromValues(HostParallel.Transpose(table.ToArray(), OutFeatures, InFeatures), InFeatures, OutFeatures, table.Device);
        return Tensor.MatMulFrozenTransposed(input, table, _tiedTransposed);
    }

    private BFloat16Weight? _tiedTransposed;

    /// <summary>An FP8 copy of the frozen weight used for forward products (see <see cref="AttachFloat8"/>), else null.</summary>
    internal Float8Weight? Float8 { get; private set; }

    /// <summary>
    /// Gives the layer an FP8 (e4m3) copy of its frozen weight, used for its forward products with an adapter (the
    /// backward pass keeps the layer's own weights). False when the weight trains, the layer is tied or int8, or the
    /// device has no FP8 products.
    /// </summary>
    internal bool AttachFloat8()
    {
        if (Float8 is not null)
        {
            return true;
        }

        if (_tiedTo is not null || _packed is { Float8Copy: false } || _weight is { RequiresGrad: true })
        {
            return false;
        }

        using var dense = _weight is null ? _packed!.Dequantize() : null;
        Float8 = Float8Weight.Create(dense ?? _weight!);
        return Float8 is not null;
    }

    /// <summary>Removes the FP8 copy (<see cref="AttachFloat8"/>).</summary>
    internal void DetachFloat8()
    {
        Float8?.Dispose();
        Float8 = null;
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        (Adapter as IDisposable)?.Dispose();
        _tiedTransposed?.Dispose();
        _tiedTransposed = null;
        DetachFloat8();
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

    /// <summary>Folds the adapter into the weight (<c>W += A·B·scale</c>) and removes it; the outputs stay the same.</summary>
    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => _packed?.Buffers() ?? [];

    // The float weight values (expanded when the layer holds packed weights).
    internal float[] WeightValues()
    {
        if (_tiedTo is not null)
        {
            return TiedValues();
        }

        if (!Packed)
        {
            return Weight.ToArray();
        }

        using var w = _packed!.Dequantize();
        return w.ToArray();
    }

    internal Device Device => _tiedTo?.Device ?? _weight?.Device ?? _packed!.Device;

    // The tied table transposed to [in, out] on the host.
    private float[] TiedValues() => HostParallel.Transpose(_tiedTo!.WeightValues(), OutFeatures, InFeatures);

    // A tied layer gets its own float copy of the (transposed) table before it is converted or merged.
    private void Untie()
    {
        if (_tiedTo is { } e)
        {
            _weight = Tensor.Persistent(TiedValues(), [InFeatures, OutFeatures], e.Device, requiresGrad: false);
            _tiedTo = null;
            _tiedTransposed?.Dispose();
            _tiedTransposed = null;
        }
    }

    internal void ToBFloat16()
    {
        if (BFloat16 is not null)
        {
            return;
        }

        Untie();
        DequantizeInt8(trainable: false);
        ToFloat32(trainable: false);
        _packed = BFloat16Weight.Convert(Weight);
        _weight!.Dispose();
        _weight = null;
    }

    // Back to float weights from bfloat16, 4-bit or other packed ones (formats of one's own).
    internal void ToFloat32(bool trainable)
    {
        if (_packed is null or Int8Weight)
        {
            return;                                                        // float already, or int8 (DequantizeInt8)
        }

        using (var w = _packed.Dequantize())
        {
            _weight = Tensor.Persistent(w.ToArray(), [InFeatures, OutFeatures], w.Device, trainable);
        }

        _packed.Dispose();
        _packed = null;
    }

    // Loading packed weights (Module.Load): room for them, allocated without quantizing the current weights that the
    // file's values replace anyway. The layer drops its float / tied / other packed weights, as the conversions do.
    internal void LoadAsInt8()
    {
        if (Int8 is null)
        {
            _packed = Int8Weight.Empty(InFeatures, OutFeatures, ReleaseWeights());
        }
    }

    internal void LoadAsInt4()
    {
        if (Int4 is null)
        {
            _packed = Int4Weight.Empty(InFeatures, OutFeatures, ReleaseWeights());
        }
    }

    internal void LoadAsBFloat16()
    {
        if (BFloat16 is null)
        {
            _packed = BFloat16Weight.Empty(InFeatures, OutFeatures, ReleaseWeights());
        }
    }

    // Releases the weights (a tied layer only lets go of the table) and returns their device.
    private Device ReleaseWeights()
    {
        var device = Device;
        _tiedTo = null;
        _tiedTransposed?.Dispose();
        _tiedTransposed = null;
        _weight?.Dispose();
        _weight = null;
        _packed?.Dispose();
        _packed = null;
        return device;
    }

    internal void QuantizeInt4()
    {
        if (Int4 is not null)
        {
            return;
        }

        Untie();
        DequantizeInt8(trainable: false);
        ToFloat32(trainable: false);
        _packed = Int4Weight.Quantize(Weight);
        _weight!.Dispose();
        _weight = null;
    }

    internal void QuantizeInt8()
    {
        if (Int8 is not null)
        {
            return;
        }

        Untie();
        ToFloat32(trainable: false);
        _packed = Int8Weight.Quantize(Weight);
        _weight!.Dispose();
        _weight = null;
    }

    internal void DequantizeInt8(bool trainable)
    {
        if (Int8 is not { } q)
        {
            return;
        }

        using (var w = q.Dequantize())
        {
            _weight = Tensor.Persistent(w.ToArray(), [InFeatures, OutFeatures], w.Device, trainable);
        }

        q.Dispose();
        _packed = null;
    }

    internal void MergeAdapter()
    {
        if (Adapter is not { } a)
        {
            return;
        }

        if (Packed)
        {
            throw new InvalidOperationException($"{this}: merging an adapter into {_packed!.Description} weights would lose precision; call {_packed.FloatMethod} first, or keep the adapter.");
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
    protected internal override void MoveTo(Device device)
    {
        _weight = _weight is null ? null : MoveTensor(_weight, device);
        _packed?.MoveWeights(device, MoveTensor);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
        if (Adapter is { } a)
        {
            Adapter = a.MoveTo(t => MoveTensor(t, device));
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"Linear({InFeatures} -> {OutFeatures}{(Bias is null ? ", no bias" : "")}{(_packed is null ? "" : $", {_packed.ShortName}")}{(_tiedTo is null ? "" : ", tied")}{Adapter switch { null => "", LoraAdapter a => $", LoRA rank {a.Rank}", DoraAdapter d => $", DoRA rank {d.Rank}", var other => $", {other.GetType().Name}" }}{(_softcap is { } c ? $", softcap {c}" : "")})";
}
