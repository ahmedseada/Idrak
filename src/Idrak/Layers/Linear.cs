namespace Idrak.Layers;

/// <summary>
/// Fully connected layer: y = x · W + b, mapping [..., inFeatures] to [..., outFeatures]
/// (any leading dimensions, e.g. [batch, time, features] for sequences).
/// Weights start Xavier/Glorot-uniform and the bias starts at zero.
/// </summary>
public sealed class Linear : Module
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

    private Linear(int inFeatures, int outFeatures, Tensor? weight, Int8Weight? int8, Tensor? bias, BFloat16Weight? half = null, Int4Weight? int4 = null)
    {
        Int4 = int4;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = weight;
        Int8 = int8;
        BFloat16 = half;
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

    /// <summary>A layer around existing int8 weights (see <see cref="ModuleExtensions.QuantizeInt8"/>); the layer takes ownership.</summary>
    public static Linear FromInt8(Int8Weight weight, Tensor? bias = null) => new(weight.Rows, weight.Columns, null, weight, bias);

    /// <summary>A layer around existing bfloat16 weights (see <see cref="ModuleExtensions.ToBFloat16"/>); the layer takes ownership.</summary>
    public static Linear FromBFloat16(BFloat16Weight weight, Tensor? bias = null) => new(weight.Rows, weight.Columns, null, null, bias, weight);

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
    public static Linear FromInt4(Int4Weight weight, Tensor? bias = null) => new(weight.Rows, weight.Columns, null, null, bias, null, weight);

    /// <summary>Number of input features.</summary>
    public int InFeatures { get; }

    /// <summary>Number of output features.</summary>
    public int OutFeatures { get; }

    /// <summary>The [inFeatures, outFeatures] weight matrix.</summary>
    public Tensor Weight => _weight ?? throw new InvalidOperationException(_tiedTo is not null
        ? $"{this} reads the embedding table of {_tiedTo} (transposed); use TiedTo.Weight."
        : Int8 is not null
        ? $"{this} holds int8 weights (see Int8); call DequantizeInt8() on the model to get float weights back."
        : Int4 is not null ? $"{this} holds 4-bit weights (see Int4); call ToFloat32() on the model to get float weights back."
        : $"{this} holds bfloat16 weights (see BFloat16); call ToFloat32() on the model to get float weights back.");

    /// <summary>The 4-bit weights when the layer holds them (see <see cref="ModuleExtensions.QuantizeInt4"/>), else null.</summary>
    public Int4Weight? Int4 { get; private set; }

    // Whether the weights are packed (int8, int4 or bfloat16) rather than a float tensor.
    internal bool Packed => Int8 is not null || Int4 is not null || BFloat16 is not null;

    /// <summary>The bfloat16 weights when the layer holds them (see <see cref="ModuleExtensions.ToBFloat16"/>), else null.</summary>
    public BFloat16Weight? BFloat16 { get; private set; }

    /// <summary>The int8 weights when the layer was quantized with <see cref="ModuleExtensions.QuantizeInt8"/>, else null.</summary>
    public Int8Weight? Int8 { get; private set; }

    private Tensor? _weight;

    /// <summary>The [outFeatures] bias, or null when created with <c>bias: false</c>.</summary>
    public Tensor? Bias { get; private set; }

    /// <summary>
    /// A low-rank adapter (LoRA) added to this layer by <see cref="ModuleExtensions.AddLora"/>, or null. When present the
    /// output is <c>x·W + b + (x·A·B)·scale</c>; its A and B come after W and b in <see cref="Parameters"/>.
    /// </summary>
    public LoraAdapter? Adapter { get; internal set; }

    /// <summary>A float32 weight of its own, no adapter: the layer is x·W (+ b), which fused operations may compute themselves.</summary>
    internal static bool PlainFloat(Linear layer) => layer._weight is not null && layer._tiedTo is null && layer.Adapter is null;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (Bias is not null && _weight is not null && _tiedTo is null && Adapter is null && input.Rank >= 2)
        {
            return Tensor.MatMulBias(input, _weight, Bias);                 // one pass on tensor cores
        }

        if (Bias is { RequiresGrad: false } && Adapter is not null && Tensor.LoraProducts(input, [this]) is [var withBias])
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
        bool fused = !Autograd.IsEnabled && layers.Length is > 1 and <= 3 && rows <= Backends.Cuda.PtxKernels.GemvRows
            && input.Device.Type == DeviceType.Cuda
            && layers.All(l => !l.Packed && l.TiedTo is null && l.Adapter is null && l.InFeatures == k && (long)l.OutFeatures * k >= 1 << 16);
        if (fused)
        {
            return Tensor.MatMulMany(input, [.. layers.Select(l => l.Weight)], [.. layers.Select(l => l.Bias)]);
        }

        // Packed weights of one kind (int8, int4, bfloat16): one pass where the device has one (few rows; for prompts, one
        // tensor-core launch over all the layers' columns).
        int kind = layers[0].Int8 is not null ? 0 : layers[0].Int4 is not null ? 1 : layers[0].BFloat16 is not null ? 2 : -1;
        bool packed = kind >= 0 && !Autograd.IsEnabled && layers.Length is > 1 and <= 3
            && (rows <= Backends.Cuda.PtxKernels.GemvRows || layers.All(l => l.Bias is null))                  // prompts: one launch
            && layers.All(l => l.Adapter is null && l.InFeatures == k && (kind == 0 ? l.Int8 is not null : kind == 1 ? l.Int4 is not null : l.BFloat16 is not null));
        if (packed && Tensor.MatMulPackedMany(input, kind, layers) is { } outputs)
        {
            return outputs;
        }

        // Adapters on every layer (LoRA / QLoRA, training or evaluation): one pass with each low-rank term inside its product.
        if (layers.All(l => l.Adapter is not null && l.Bias is not { RequiresGrad: true }) && Tensor.LoraProducts(input, layers) is { } lora)
        {
            return lora;
        }

        // Training over frozen packed layers (LoRA / QLoRA): the base products still run as one pass, recorded, and each
        // layer's adapter adds its low-rank term into its output.
        bool training = kind >= 0 && Autograd.IsEnabled && layers.Length is > 1 and <= 3 && input.Device.Type == DeviceType.Cuda
            && layers.All(l => l.Bias is null && l.InFeatures == k && (kind == 0 ? l.Int8 is not null : kind == 1 ? l.Int4 is not null : l.BFloat16 is not null));
        if (training && Tensor.MatMulPackedManyRecorded(input, kind, layers) is { } products)
        {
            return [.. products.Select((product, j) => layers[j].Adapter is { } a ? Tensor.AddLowRank(product, input, a.A, a.B, a.Scale) : product)];
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

    /// <summary>x·W, plus the adapter's low-rank term when an adapter is attached.</summary>
    internal Tensor ProjectWithoutBias(Tensor input)
    {
        if (Adapter is not null && Tensor.LoraProducts(input, [this], withBias: false) is [var fused])
        {
            return fused;
        }

        var product = Int8 is { } q ? input.MatMulInt8(q) : Int4 is { } q4 ? input.MatMulInt4(q4) : BFloat16 is { } h ? input.MatMulBFloat16(h)
            : _tiedTo is { } e ? TiedProduct(input, e.Weight) : input.MatMul(Weight);
        return Adapter is { } a ? Tensor.AddLowRank(product, input, a.A, a.B, a.Scale) : product;
    }

    // The tied head (x · Eᵀ). While the table is frozen, large products read a transposed bfloat16 copy made once (the
    // tensor-core kernels would otherwise copy the whole table transposed on every call; bfloat16 is what they multiply in
    // anyway, at half the memory of a float copy: 0.45 GB instead of 0.9 GB for a 152k × 1536 table). The input's gradient
    // reads the table as stored.
    private Tensor TiedProduct(Tensor input, Tensor table)
    {
        int rows = input.Size / Math.Max(1, InFeatures);
        if (table.RequiresGrad || rows < 64 || table.Device.Type != DeviceType.Cuda || !MixedPrecision.UsesTensorCores)
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

        if (_tiedTo is not null || Int8 is not null || _weight is { RequiresGrad: true })
        {
            return false;
        }

        using var dense = _weight is null ? Int4?.Dequantize() ?? BFloat16!.Dequantize() : null;
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
        return Adapter is { } a ? own.Concat([a.A, a.B]) : own;
    }

    /// <summary>Folds the adapter into the weight (<c>W += A·B·scale</c>) and removes it; the outputs stay the same.</summary>
    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => Int8 is { } q ? [q.Packed, q.Scales] : Int4 is { } q4 ? [q4.Packed, q4.Scales] : BFloat16 is { } h ? [h.Packed] : [];

    // The float weight values (dequantized when the layer holds int8 weights).
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

        using var w = Int8?.Dequantize() ?? Int4?.Dequantize() ?? BFloat16!.Dequantize();
        return w.ToArray();
    }

    internal Device Device => _tiedTo?.Device ?? (_weight ?? Int8?.Packed ?? Int4?.Packed ?? BFloat16!.Packed).Device;

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
        BFloat16 = BFloat16Weight.Convert(Weight);
        _weight!.Dispose();
        _weight = null;
    }

    // Back to float weights from bfloat16 or 4-bit ones.
    internal void ToFloat32(bool trainable)
    {
        if (BFloat16 is null && Int4 is null)
        {
            return;
        }

        using (var w = BFloat16?.Dequantize() ?? Int4!.Dequantize())
        {
            _weight = Tensor.Persistent(w.ToArray(), [InFeatures, OutFeatures], w.Device, trainable);
        }

        BFloat16?.Dispose();
        Int4?.Dispose();
        BFloat16 = null;
        Int4 = null;
    }

    // Loading packed weights (Module.Load): room for them, allocated without quantizing the current weights that the
    // file's values replace anyway. The layer drops its float / tied / other packed weights, as the conversions do.
    internal void LoadAsInt8()
    {
        if (Int8 is null)
        {
            Int8 = Int8Weight.Empty(InFeatures, OutFeatures, ReleaseWeights());
        }
    }

    internal void LoadAsInt4()
    {
        if (Int4 is null)
        {
            Int4 = Int4Weight.Empty(InFeatures, OutFeatures, ReleaseWeights());
        }
    }

    internal void LoadAsBFloat16()
    {
        if (BFloat16 is null)
        {
            BFloat16 = BFloat16Weight.Empty(InFeatures, OutFeatures, ReleaseWeights());
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
        Int8?.Dispose();
        Int4?.Dispose();
        BFloat16?.Dispose();
        (Int8, Int4, BFloat16) = (null, null, null);
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
        Int4 = Int4Weight.Quantize(Weight);
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
        Int8 = Int8Weight.Quantize(Weight);
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
        Int8 = null;
    }

    internal void MergeAdapter()
    {
        if (Adapter is not { } a)
        {
            return;
        }

        if (Packed)
        {
            throw new InvalidOperationException($"{this}: merging a LoRA adapter into {(Int8 is not null ? "int8" : Int4 is not null ? "4-bit" : "bfloat16")} weights would lose precision; call {(Int8 is null ? "ToFloat32()" : "DequantizeInt8()")} first, or keep the adapter.");
        }

        Untie();
        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            var merged = Weight + a.A.MatMul(a.B) * a.Scale;
            Weight.Load(merged.ToArray());
        }

        Adapter = null;
        a.A.Dispose();
        a.B.Dispose();
    }

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        _weight = _weight is null ? null : MoveTensor(_weight, device);
        Int8?.MoveTo(device, MoveTensor);
        BFloat16?.MoveTo(device, MoveTensor);
        Int4?.MoveTo(device, MoveTensor);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
        if (Adapter is { } a)
        {
            Adapter = a with { A = MoveTensor(a.A, device), B = MoveTensor(a.B, device) };
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"Linear({InFeatures} -> {OutFeatures}{(Bias is null ? ", no bias" : "")}{(Int8 is null ? "" : ", int8")}{(BFloat16 is null ? "" : ", bf16")}{(Int4 is null ? "" : ", int4")}{(_tiedTo is null ? "" : ", tied")}{(Adapter is { } a ? $", LoRA rank {a.Rank}" : "")})";
}

/// <summary>
/// A LoRA adapter: two small trainable matrices A [in, rank] and B [rank, out] whose product, times
/// <see cref="Scale"/> = alpha / rank, is added to a <see cref="Linear"/> layer's weight. B starts at zero, so adding
/// an adapter does not change the model's outputs until it is trained.
/// </summary>
/// <param name="A">[inFeatures, rank], small random values.</param>
/// <param name="B">[rank, outFeatures], zeros at creation.</param>
/// <param name="Rank">The rank r.</param>
/// <param name="Scale">alpha / r.</param>
public sealed record LoraAdapter(Tensor A, Tensor B, int Rank, float Scale);
