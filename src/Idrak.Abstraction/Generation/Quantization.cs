// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// What a packed format packs (<see cref="IPackedFormat"/>): float weights [rows, columns] (host values, row-major) and the
/// device the packed weights are made on.
/// </summary>
/// <param name="values">The float weights, <paramref name="rows"/> × <paramref name="columns"/> values.</param>
/// <param name="rows">Input features (rows of the weight matrix).</param>
/// <param name="columns">Output features (columns).</param>
/// <param name="device">Where the packed weights are created.</param>
public readonly ref struct PackRequest(ReadOnlySpan<float> values, int rows, int columns, Device device)
{
    /// <summary>The float weights, <see cref="Rows"/> × <see cref="Columns"/> values.</summary>
    public ReadOnlySpan<float> Values { get; } = values;

    /// <summary>Input features (rows of the weight matrix).</summary>
    public int Rows { get; } = rows;

    /// <summary>Output features (columns).</summary>
    public int Columns { get; } = columns;

    /// <summary>Where the packed weights are created.</summary>
    public Device Device { get; } = device;
}

/// <summary>
/// A packed weight format, registered by its <see cref="IKind.Name"/> in <see cref="PackedWeightFormats"/> ("int8", "int4",
/// "bfloat16", or one of your own): how float weights are packed (<see cref="IKind{TRequest, T}.Create"/>), so options and
/// tools choose the format by name (<c>DecoderBuildOptions.PackedFormatName</c>).
/// </summary>
public interface IPackedFormat : IKind<PackRequest, IPackedWeight>;

/// <summary>
/// Weights of a <c>Linear</c> layer held in a packed format: the built-in <see cref="Int8Weight"/>,
/// <see cref="Int4Weight"/> and <see cref="BFloat16Weight"/>, or a format defined outside Idrak (NF4, FP8, …). The layer
/// and the fused products go through this contract instead of checking which format a layer holds: each format says how
/// it multiplies, expands and moves. What a format can do beyond it is an interface of its own, tested once when the
/// layer takes the weights: <see cref="IRecordable"/> when its <see cref="MatMul"/> only issues device work (a decoding
/// step recorded as a device graph records it), <see cref="IKernelPackedWeight"/> for the built-in formats the device
/// kernels read directly. <see cref="PackedWeight"/> is a base that expands the weights for each product.
/// </summary>
public interface IPackedWeight : IDisposable
{
    /// <summary>The format's name, as registered in <see cref="PackedWeightFormats"/> ("int8", "int4", "bfloat16" for the built-in ones).</summary>
    string Name { get; }

    /// <summary>Input features (rows of the [rows, columns] weight).</summary>
    int Rows { get; }

    /// <summary>Output features (columns).</summary>
    int Columns { get; }

    /// <summary>Device memory used, in bytes.</summary>
    long Bytes { get; }

    /// <summary>The device the weights are on.</summary>
    Device Device { get; }

    /// <summary>The format in messages: "int8", "4-bit", "bfloat16".</summary>
    string Description { get; }

    /// <summary>The format's short name, as summaries show it.</summary>
    string ShortName { get; }

    /// <summary>How to get float weights back, as messages name it ("ToFloat32()").</summary>
    string FloatMethod { get; }

    /// <summary>The float weights, [rows, columns], on the same device, as a new tensor the caller disposes.</summary>
    Tensor Dequantize();

    /// <summary>Writes the float weights [rows, columns] into <paramref name="destination"/> (on the same device).</summary>
    void DequantizeInto(Storage destination);

    /// <summary>
    /// The tensors holding the weights (packed words, scales): the layer lists them as its <see cref="Module.Buffers"/>, so
    /// <c>Idrak.ModuleFiles.Save</c> writes them exactly as they are (and <c>Idrak.ModuleFiles.Load</c> reads them
    /// back into a model holding the same format), and offloading treats them as frozen weights.
    /// </summary>
    IEnumerable<Tensor> Buffers();

    /// <summary>
    /// Moves the weights to <paramref name="device"/> (what a layer holding them calls from its own <c>MoveTo</c>): each
    /// tensor held is replaced by <c>move(tensor, device)</c>, which returns it on the device and releases the old one.
    /// </summary>
    /// <param name="device">The destination.</param>
    /// <param name="move">Moves one tensor: returns it on the device and releases the old one.</param>
    void MoveWeights(Device device, Func<Tensor, Device, Tensor> move);

    /// <summary>
    /// input [..., <see cref="Rows"/>] · W → [..., <see cref="Columns"/>], gradients flowing to the input (the weights are
    /// frozen). A format that only issues device work here implements <see cref="IRecordable"/>, so a decoding step
    /// recorded as a device graph records the product; one that does not runs such steps without a graph.
    /// </summary>
    /// <param name="input">[..., <see cref="Rows"/>] on the weights' device.</param>
    Tensor MatMul(Tensor input);
}

/// <summary>
/// Packed weights in a built-in <see cref="PackedFormat"/>, which the device kernels read directly: the fused projections,
/// gate/up pairs, low-rank (LoRA) products and the activation read by the down projection take them. Their products only
/// issue device work (<see cref="IRecordable"/>). A format defined outside Idrak is not one: every layer holding it runs
/// its <see cref="IPackedWeight.MatMul"/>.
/// </summary>
public interface IKernelPackedWeight : IPackedWeight, IRecordable
{
    /// <summary>The built-in format the device kernels read.</summary>
    PackedFormat Format { get; }

    /// <summary>The packed words the device kernels read.</summary>
    Tensor PackedValues { get; }

    /// <summary>The scales the device kernels read with <see cref="PackedValues"/>, or null when the format has none.</summary>
    Tensor? ScaleValues { get; }

    /// <summary>Whether the fused low-rank (LoRA) products read this format packed; otherwise the base product runs on its own.</summary>
    bool LowRankProducts { get; }

    /// <summary>Whether an FP8 copy may be made for training products (the fused low-rank products read one).</summary>
    bool Float8Copy { get; }

    /// <summary>
    /// Whether a gated feed-forward block applies the activation as the down projection reads its input (4-bit: eight
    /// columns per word repay that work), rather than in the gate/up product.
    /// </summary>
    bool ActivationInDownProjection { get; }

    /// <summary>
    /// dx = beta·dx + g · Wᵀ (+ dt · Aᵀ when <paramref name="dt"/> is given) for g [m, n], reading W [k, n] as stored: the
    /// input gradient through the frozen weights; false when the format or the device has no such product.
    /// </summary>
    /// <param name="g">The output gradient [m, n].</param>
    /// <param name="dx">The input gradient [m, k].</param>
    /// <param name="m">Rows.</param>
    /// <param name="k">Input features.</param>
    /// <param name="n">Output features.</param>
    /// <param name="beta">0 to overwrite <paramref name="dx"/>, 1 to add to it.</param>
    /// <param name="dt">The low-rank term's gradient [m, rank], or null.</param>
    /// <param name="a">The adapter's down projection [k, rank], or null.</param>
    /// <param name="rank">The adapter's rank.</param>
    bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank);
}

/// <summary>
/// A base for a packed weight format of one's own (<see cref="IPackedWeight"/>): it implements <see cref="Name"/>,
/// <see cref="Rows"/>, <see cref="Columns"/>, <see cref="Bytes"/>, <see cref="Dequantize"/>, <see cref="Buffers"/>,
/// <see cref="MoveTo"/> and <see cref="Dispose"/>; <see cref="MatMul"/> expands the weights for each product unless it is
/// overridden with a product that reads the packed form (a product that only issues device work makes the format an
/// <see cref="IRecordable"/>). Register the format (<see cref="IPackedFormat"/>) with <see cref="PackedWeightFormats.Register"/>
/// to create it by name (<see cref="PackedWeightFormats.Pack(string, ReadOnlySpan{float}, int, int, Device)"/>,
/// <c>DecoderBuildOptions.PackedFormatName</c>).
/// </summary>
public abstract class PackedWeight : IPackedWeight
{
    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public abstract int Rows { get; }

    /// <inheritdoc />
    public abstract int Columns { get; }

    /// <inheritdoc />
    public abstract long Bytes { get; }

    /// <summary>The device the weights are on (by default, that of the first of <see cref="Buffers"/>).</summary>
    public virtual Device Device => Buffers().First().Device;

    /// <inheritdoc />
    public abstract Tensor Dequantize();

    /// <inheritdoc />
    public abstract IEnumerable<Tensor> Buffers();

    /// <summary>
    /// Moves the weights to <paramref name="device"/> (<see cref="Module.To"/>): replace each tensor held by
    /// <c>move(tensor, device)</c>, which returns it on the device and releases the old one.
    /// </summary>
    /// <param name="device">The destination.</param>
    /// <param name="move">Moves one tensor.</param>
    protected abstract void MoveTo(Device device, Func<Tensor, Device, Tensor> move);

    /// <inheritdoc />
    public void MoveWeights(Device device, Func<Tensor, Device, Tensor> move) => MoveTo(device, move);

    /// <summary>
    /// input [..., <see cref="Rows"/>] · W → [..., <see cref="Columns"/>], gradients flowing to the input (the weights are
    /// frozen). By default the weights are expanded with <see cref="Dequantize"/> for the product (and again for the
    /// input's gradient) and freed after it, so a format works without a kernel of its own; override this with a product
    /// that reads the packed form. The default refuses to be recorded into a device graph, since it cannot tell whether
    /// <see cref="Dequantize"/> reads the host: such steps run without a graph.
    /// </summary>
    /// <param name="input">[..., <see cref="Rows"/>] on the weights' device.</param>
    public virtual Tensor MatMul(Tensor input) => input.MatMulExpanded(this);

    /// <inheritdoc />
    public abstract void Dispose();

    /// <summary>Writes the float weights [rows, columns] into <paramref name="destination"/> (on the same device).</summary>
    public virtual void DequantizeInto(Storage destination)
    {
        using var w = Dequantize();
        w.Backend.Copy(w.Storage, destination, Rows * Columns);
    }

    /// <summary>The format in messages: "int8", "4-bit", "bfloat16".</summary>
    public virtual string Description => Name;

    /// <summary>The format's short name, as summaries show it.</summary>
    public virtual string ShortName => Name;

    /// <summary>How to get float weights back, as messages name it ("ToFloat32()").</summary>
    public virtual string FloatMethod => "ToFloat32()";
}

/// <summary>
/// The packed weight formats by name (ignoring case): "int8", "int4" and "bfloat16" are registered; add a format of one's
/// own (<see cref="IPackedFormat"/>) with <see cref="Register"/>, so options and tools can choose it by name. Under a
/// built-in name the app's format overrides the library's, which stays behind it (see <see cref="SetPolicy"/>). The
/// overload of <see cref="Pack(PackedFormat, ReadOnlySpan{float}, int, int, Device)"/> taking a
/// <see cref="PackedFormat"/> always makes the built-in weights.
/// </summary>
public static class PackedWeightFormats
{
    // The built-in formats, by enum: the enum overload of Pack always makes these (a registered name may replace its entry
    // in the registry below, not here), and the registry starts from them.
    private static readonly IReadOnlyDictionary<PackedFormat, IPackedFormat> BuiltIn = new Dictionary<PackedFormat, IPackedFormat>
    {
        [PackedFormat.Int8] = new BuiltInFormat("int8", "one signed byte a weight and a scale a column", request => Int8Weight.Quantize(request.Values, request.Rows, request.Columns, request.Device)),
        [PackedFormat.Int4] = new BuiltInFormat("int4", "4-bit weights, a scale per 32 rows of a column", request => Int4Weight.Quantize(request.Values, request.Rows, request.Columns, request.Device)),
        [PackedFormat.BFloat16] = new BuiltInFormat("bfloat16", "bfloat16 weights, half the float32 memory", request => BFloat16Weight.FromValues(request.Values, request.Rows, request.Columns, request.Device)),
    };

    private static readonly KindTable<IPackedFormat> Registry = Formats();

    private static KindTable<IPackedFormat> Formats()
    {
        var table = new KindTable<IPackedFormat>(nameof(PackedWeightFormats), "Packed format", "add it with PackedWeightFormats.Register", Guard, StringComparer.OrdinalIgnoreCase);
        foreach (var format in BuiltIn.Values)
        {
            table.RegisterDefault(format);
        }

        return table;
    }

    // Packing makes the weights once: an app's format falls back when it fails to make them. Under Shadow both pack the
    // sampled weights, which agree when their shapes do; the app's are then freed.
    private static IPackedFormat Guard(Slot slot, IPackedFormat app, IPackedFormat library) => new GuardedFormat(slot, app, library);

    /// <summary>Packs [rows, columns] float values in the built-in <paramref name="format"/> on <paramref name="device"/>.</summary>
    public static IKernelPackedWeight Pack(PackedFormat format, ReadOnlySpan<float> values, int rows, int columns, Device device) =>
        (IKernelPackedWeight)Pack(BuiltIn.TryGetValue(format, out var builtIn) ? builtIn
            : throw new ArgumentOutOfRangeException(nameof(format), format, "Not a built-in packed format."), values, rows, columns, device);

    /// <summary>
    /// Packs [rows, columns] float values in the format registered as <paramref name="format"/> (see <see cref="Names"/>)
    /// on <paramref name="device"/>.
    /// </summary>
    public static IPackedWeight Pack(string format, ReadOnlySpan<float> values, int rows, int columns, Device device) =>
        Pack(Get(format), values, rows, columns, device);

    /// <summary>The weights <paramref name="format"/> packs, checked against the [rows, columns] shape asked for.</summary>
    /// <exception cref="ArgumentException">The values do not fill the shape.</exception>
    /// <exception cref="InvalidOperationException">The format made weights of another shape.</exception>
    public static IPackedWeight Pack(IPackedFormat format, ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        ArgumentNullException.ThrowIfNull(format);
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        var weight = format.Create(new PackRequest(values, rows, columns, device));
        if (weight.Rows != rows || weight.Columns != columns)
        {
            weight.Dispose();
            throw new InvalidOperationException($"The '{format.Name}' format made [{weight.Rows}, {weight.Columns}] weights for [{rows}, {columns}] values.");
        }

        return weight;
    }

    /// <summary>
    /// Registers <paramref name="format"/> under its <see cref="IKind.Name"/> (names ignore case), so options and tools can
    /// choose it by name; under a built-in name it overrides the library's format, which stays behind it (see
    /// <see cref="SetPolicy"/>).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPackedFormat format) => Registry.Register(format, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's format <paramref name="format"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string format) => Registry.Unregister(format);

    /// <summary>The registered format names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The format registered as <paramref name="format"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">No format of that name is registered.</exception>
    public static IPackedFormat Get(ReadOnlySpan<char> format) =>
        Registry.TryGet(format, out var found) ? found
            : throw new NotRegisteredException(nameof(PackedWeightFormats), format.ToString(), Registry.Names,
                $"No packed format '{format}' is registered ({string.Join(", ", Registry.Names)}); add it with PackedWeightFormats.Register.");

    /// <summary>The library's format <paramref name="format"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IPackedFormat? Default(string format) => Registry.Default(format);

    /// <summary>Who registered the format <paramref name="format"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string format) => Registry.Origin(format);

    /// <summary>
    /// What happens when the app's format <paramref name="format"/> fails to pack (<see cref="SlotPolicy.Throw"/> unless set:
    /// the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's). Packed weights are made once, so the fallback is at packing; their products do not fall back.
    /// </summary>
    public static void SetPolicy(string format, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(format, policy, shadowRate);

    // A built-in format: its weights from the request.
    private sealed class BuiltInFormat(string name, string summary, Func<PackRequest, IPackedWeight> pack) : IPackedFormat
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public IPackedWeight Create(PackRequest request) => pack(request);

        public override string ToString() => name;
    }

    private sealed class GuardedFormat(Slot slot, IPackedFormat app, IPackedFormat library) : IPackedFormat
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public IPackedWeight Create(PackRequest request)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Create(request);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        using var other = app.Create(request);
                        run.Done(Comparisons.Exact((answer.Rows, answer.Columns), (other.Rows, other.Columns)));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Create(request);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Create(request);
            }
        }
    }
}

/// <summary>
/// The int8 weights of a quantized <c>Linear</c> layer: each weight is a signed byte times its output column's
/// scale (symmetric, per-column: scale = max |w| / 127). A quarter of the float32 memory; created by
/// <c>ModuleExtensions.QuantizeInt8</c>. Inputs, outputs, biases and LoRA adapters stay float32.
/// </summary>
public sealed class Int8Weight : PackedWeight, IKernelPackedWeight
{
    /// <inheritdoc />
    public override string Name => "int8";

    /// <inheritdoc />
    public PackedFormat Format => PackedFormat.Int8;

    /// <inheritdoc />
    public override Device Device => Packed.Device;

    /// <inheritdoc />
    public Tensor PackedValues => Packed;

    /// <inheritdoc />
    public Tensor? ScaleValues => Scales;

    /// <summary>input [..., rows] · W → [..., columns], reading the bytes (few rows) or expanding them once (more).</summary>
    public override Tensor MatMul(Tensor input) => input.MatMulInt8(this);

    /// <inheritdoc />
    public bool LowRankProducts => false;

    /// <inheritdoc />
    public bool Float8Copy => false;

    /// <inheritdoc />
    public bool ActivationInDownProjection => false;

    /// <inheritdoc />
    public bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank) => false;

    /// <inheritdoc />
    public override string Description => "int8";

    /// <inheritdoc />
    public override string ShortName => "int8";

    /// <inheritdoc />
    public override string FloatMethod => "DequantizeInt8()";

    private Int8Weight(Tensor packed, Tensor scales, int rows, int columns)
    {
        Packed = packed;
        Scales = scales;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Input features (rows of the weight matrix).</summary>
    public override int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public override int Columns { get; }

    /// <summary>Device memory used, in bytes (weights and scales).</summary>
    public override long Bytes => 4L * (Packed.Size + Scales.Size);

    /// <summary>The bytes, packed four per element along each row (rows padded to a multiple of four columns).</summary>
    public Tensor Packed { get; private set; }

    /// <summary>One scale per column.</summary>
    public Tensor Scales { get; private set; }

    /// <summary>Quantizes a float weight matrix [rows, columns] (on its device).</summary>
    public static Int8Weight Quantize(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"Int8 quantization needs a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return Quantize(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>
    /// Quantizes weights given as host values [rows, columns] and uploads only the bytes to <paramref name="device"/>
    /// (large models never exist as float32 on the device).
    /// </summary>
    public static unsafe Int8Weight Quantize(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        // The worker lambdas cannot capture a span: they read the pinned values through a pointer (no copy).
        fixed (float* pinned = values)
        {
            return Quantize(pinned, rows, columns, device);
        }
    }

    /// <summary>
    /// Packed bytes and scales for [rows, columns] allocated on <paramref name="device"/> without computing anything, for
    /// loading stored values into (<c>Idrak.ModuleFiles.Load</c>).
    /// </summary>
    public static Int8Weight Empty(int rows, int columns, Device device)
    {
        int stride = (columns + 3) / 4 * 4;
        return new Int8Weight(
            Tensor.PersistentZeros([rows * stride / 4], device),
            Tensor.PersistentZeros([columns], device),
            rows, columns);
    }

    private static unsafe Int8Weight Quantize(float* source, int rows, int columns, Device device)
    {
        int stride = (columns + 3) / 4 * 4;

        // Column maxima per chunk of rows (all cores), then combined.
        var scales = new float[columns];
        var gate = new Lock();
        HostParallel.For(rows, Math.Max(1, Abstraction.Devices.Cpu.CpuTuning.ParallelElements / Math.Max(1, columns)), (first, last) =>
        {
            var local = new float[columns];
            for (int r = first; r < last; r++)
            {
                for (int j = 0; j < columns; j++)
                {
                    local[j] = MathF.Max(local[j], MathF.Abs(source[r * columns + j]));
                }
            }

            lock (gate)
            {
                for (int j = 0; j < columns; j++)
                {
                    scales[j] = MathF.Max(scales[j], local[j]);
                }
            }
        });
        for (int j = 0; j < columns; j++)
        {
            scales[j] = scales[j] > 0f ? scales[j] / 127f : 1f;
        }

        var bytes = new sbyte[rows * stride];
        HostParallel.For(rows, Math.Max(1, Abstraction.Devices.Cpu.CpuTuning.ParallelElements / Math.Max(1, columns)), (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                for (int j = 0; j < columns; j++)
                {
                    bytes[r * stride + j] = (sbyte)Math.Clamp(MathF.Round(source[r * columns + j] / scales[j]), -127f, 127f);
                }
            }
        });

        var packed = MemoryMarshal.Cast<sbyte, float>(bytes);
        return new Int8Weight(
            Tensor.Persistent(packed, [packed.Length], device, requiresGrad: false),
            Tensor.Persistent(scales, [columns], device, requiresGrad: false),
            rows, columns);
    }

    /// <summary>The float weights these bytes stand for, [rows, columns], on the same device.</summary>
    public override Tensor Dequantize()
    {
        var w = Tensor.Empty([Rows, Columns], Packed.Device, track: false);   // the kernel writes every value: not zeroed first
        DequantizeInto(w.Storage);
        return w;
    }

    /// <inheritdoc />
    public override void DequantizeInto(Storage destination) => Packed.Backend.Int8Dequantize(Packed.Storage, Scales.Storage, destination, Rows, Columns);

    /// <inheritdoc />
    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move)
    {
        Packed = move(Packed, device);
        Scales = move(Scales, device);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [Packed, Scales];

    /// <inheritdoc />
    public override void Dispose()
    {
        Packed.Dispose();
        Scales.Dispose();
    }
}

/// <summary>
/// The 4-bit weights of a <c>Linear</c> layer: each group of 32 input rows of a column shares one float scale and
/// every weight is a signed nibble in -8..7 (the scale is searched per group for the least squared error), about
/// 5 bits per weight; decoding reads 8× less than float32. Created by <c>ModuleExtensions.QuantizeInt4</c> or when
/// loading with 4-bit weights; inputs, outputs, biases and LoRA adapters stay float32 (QLoRA-style fine-tuning).
/// </summary>
public sealed class Int4Weight : PackedWeight, IKernelPackedWeight
{
    /// <inheritdoc />
    public override string Name => "int4";

    /// <inheritdoc />
    public PackedFormat Format => PackedFormat.Int4;

    /// <inheritdoc />
    public override Device Device => Packed.Device;

    /// <inheritdoc />
    public Tensor PackedValues => Packed;

    /// <inheritdoc />
    public Tensor? ScaleValues => Scales;

    /// <summary>input [..., rows] · W → [..., columns], reading the nibbles.</summary>
    public override Tensor MatMul(Tensor input) => input.MatMulInt4(this);

    /// <inheritdoc />
    public bool LowRankProducts => true;

    /// <inheritdoc />
    public bool Float8Copy => true;

    /// <inheritdoc />
    public bool ActivationInDownProjection => true;

    /// <inheritdoc />
    public bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank) => false;

    /// <inheritdoc />
    public override string Description => "4-bit";

    /// <inheritdoc />
    public override string ShortName => "int4";

    /// <summary>Weight rows that share one scale per column.</summary>
    public const int GroupSize = 32;

    private Int4Weight(Tensor packed, Tensor scales, int rows, int columns)
    {
        Packed = packed;
        Scales = scales;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Input features (rows of the weight matrix).</summary>
    public override int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public override int Columns { get; }

    /// <summary>Device memory used, in bytes (weights and scales).</summary>
    public override long Bytes => 4L * (Packed.Size + Scales.Size);

    /// <summary>The nibbles, eight per element along each row (nibble c of element w is column 8w + c).</summary>
    public Tensor Packed { get; private set; }

    /// <summary>One scale per group of <see cref="GroupSize"/> rows and column: [⌈rows / 32⌉, 8·⌈columns / 8⌉].</summary>
    public Tensor Scales { get; private set; }

    /// <summary>Quantizes a float weight matrix [rows, columns] (on its device).</summary>
    public static Int4Weight Quantize(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"Int4 quantization needs a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return Quantize(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>Quantizes host values [rows, columns] and uploads only the nibbles and scales to <paramref name="device"/>.</summary>
    public static unsafe Int4Weight Quantize(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        // The worker lambdas cannot capture a span: they read the pinned values through a pointer (no copy).
        fixed (float* pinned = values)
        {
            return Quantize(pinned, rows, columns, device);
        }
    }

    /// <summary>
    /// Nibbles and scales for [rows, columns] allocated on <paramref name="device"/> without computing anything, for
    /// loading stored values into (<c>Idrak.ModuleFiles.Load</c>).
    /// </summary>
    public static Int4Weight Empty(int rows, int columns, Device device)
    {
        int words = (columns + 7) / 8, groups = (rows + GroupSize - 1) / GroupSize;
        return new Int4Weight(
            Tensor.PersistentZeros([rows * words], device),
            Tensor.PersistentZeros([groups * words * 8], device),
            rows, columns);
    }

    // Whether the quantizer takes its 128-bit vector path (four columns at once): where the runtime accelerates 128-bit
    // vectors and the setting allows vectors (ComputeResources.CpuVectors). The same bits either way.
    internal static bool VectorQuantize => Vector128.IsHardwareAccelerated && ComputeResources.CpuVectors != CpuVectorPaths.Scalar;

    // The candidate divisors around -8 the scale search tries (the group's extreme maps near -8), as the scalar rule
    // writes them: -8 + 0.1 · step for step -10..10.
    private const int SearchSteps = 10;

    private static float Divisor(int step) => -8f + 0.1f * step;

    // The smallest group extreme the vector path takes: below it a divisor over the extreme could leave float's range, and
    // the scalar rule (the reference) answers.
    private const float VectorSmallest = 1e-30f;

    private static unsafe Int4Weight Quantize(float* source, int rows, int columns, Device device)
    {
        int words = (columns + 7) / 8, groups = (rows + GroupSize - 1) / GroupSize;
        var packed = new uint[rows * words];
        var scales = new float[groups * words * 8];
        bool vectors = VectorQuantize;
        HostParallel.For(groups, Math.Max(1, 2 * Abstraction.Devices.Cpu.CpuTuning.ParallelElements / GroupSize / Math.Max(1, columns)), (first, last) =>
        {
            Span<float> w = stackalloc float[GroupSize];
            Span<sbyte> q = stackalloc sbyte[GroupSize];
            Span<sbyte> best = stackalloc sbyte[GroupSize];
            Span<Vector128<float>> wv = stackalloc Vector128<float>[GroupSize];
            Span<Vector128<float>> qv = stackalloc Vector128<float>[GroupSize];
            for (int g = first; g < last; g++)
            {
                int r0 = g * GroupSize, count = Math.Min(rows, r0 + GroupSize) - r0;
                int j = 0;
                if (vectors)
                {
                    // Four adjacent columns at once, read row by row (each row's four values are contiguous).
                    for (; j + 4 <= columns; j += 4)
                    {
                        if (!Columns4(source, columns, r0, count, j, words, g, packed, scales, wv, qv))
                        {
                            for (int c = j; c < j + 4; c++)
                            {
                                Column(source, columns, r0, count, c, words, g, packed, scales, w, q, best);
                            }
                        }
                    }
                }

                for (; j < columns; j++)
                {
                    Column(source, columns, r0, count, j, words, g, packed, scales, w, q, best);
                }
            }
        });

        return new Int4Weight(
            Tensor.Persistent(MemoryMarshal.Cast<uint, float>(packed), [packed.Length], device, requiresGrad: false),
            Tensor.Persistent(scales, [scales.Length], device, requiresGrad: false),
            rows, columns);
    }

    // One column of a group (the reference rule): the scale with the least squared error among the divisors, and its
    // nibbles.
    private static unsafe void Column(float* source, int columns, int r0, int count, int j, int words, int g, uint[] packed, float[] scales,
        Span<float> w, Span<sbyte> q, Span<sbyte> best)
    {
        float extreme = 0f;
        for (int i = 0; i < count; i++)
        {
            w[i] = source[(r0 + i) * columns + j];
            extreme = MathF.Abs(w[i]) > MathF.Abs(extreme) ? w[i] : extreme;
        }

        // Zero nibbles unless a scale is taken: a group no candidate fits (a weight that is not finite) has scale 0 and
        // zero nibbles, whatever the previous column held.
        float scale = 0f;
        best.Clear();
        if (extreme != 0f)
        {
            // Divisors t around -8 (the group's extreme maps near -8); for each, the least-squares scale of
            // the rounded values; keep the smallest squared error (t = -8 is the plain rule, never worse).
            double bestError = double.MaxValue;
            for (int step = -SearchSteps; step <= SearchSteps; step++)
            {
                float t = Divisor(step), inverse = t / extreme;
                double wq = 0, qq = 0;
                for (int i = 0; i < count; i++)
                {
                    q[i] = (sbyte)Math.Clamp((int)MathF.Round(w[i] * inverse), -8, 7);
                    wq += w[i] * q[i];
                    qq += q[i] * q[i];
                }

                if (qq == 0)
                {
                    continue;
                }

                float d = (float)(wq / qq);
                double error = 0;
                for (int i = 0; i < count; i++)
                {
                    double e = w[i] - d * q[i];
                    error += e * e;
                }

                if (error < bestError)
                {
                    (bestError, scale) = (error, d);
                    q[..count].CopyTo(best);
                }
            }
        }

        scales[g * words * 8 + j] = scale;
        for (int i = 0; i < count; i++)
        {
            packed[(r0 + i) * words + (j >> 3)] |= (uint)(best[i] & 15) << (4 * (j & 7));
        }
    }

    // Columns j..j+3 of a group in 128-bit vectors: every lane computes what Column computes for its column, in the same
    // order (float products, double sums over the rows in turn, round half to even, the same comparisons), so the nibbles
    // and scales are the same bits. False (nothing written) for a block the reference must answer: a value that is not
    // finite, or a group extreme so small that a divisor over it could leave float's range.
    private static unsafe bool Columns4(float* source, int columns, int r0, int count, int j, int words, int g, uint[] packed, float[] scales,
        Span<Vector128<float>> wv, Span<Vector128<float>> qv)
    {
        var extreme = Vector128<float>.Zero;
        var finite = Vector128<float>.Zero;
        for (int i = 0; i < count; i++)
        {
            var w = Vector128.Load(source + (r0 + i) * columns + j);
            wv[i] = w;
            finite |= w - w;                                                        // 0 for finite values, NaN otherwise
            extreme = Vector128.ConditionalSelect(Vector128.GreaterThan(Vector128.Abs(w), Vector128.Abs(extreme)), w, extreme);
        }

        var zero = Vector128.Equals(extreme, Vector128<float>.Zero);
        if (!Vector128.EqualsAll(finite.AsInt32(), Vector128<int>.Zero)
            || !Vector128.EqualsAll((zero | Vector128.GreaterThanOrEqual(Vector128.Abs(extreme), Vector128.Create(VectorSmallest))).AsInt32(), Vector128<int>.AllBitsSet))
        {
            return false;
        }

        // A zero extreme divides by one here (its lane ends with scale 0 and zero nibbles, as the reference's).
        var divisor = Vector128.ConditionalSelect(zero, Vector128<float>.One, extreme);
        var bestLow = Vector128.Create(double.MaxValue);
        var bestHigh = Vector128.Create(double.MaxValue);
        var bestScale = Vector128<float>.Zero;
        var bestStep = Vector128.Create(int.MinValue);
        var low8 = Vector128.Create(-8);
        var high7 = Vector128.Create(7);
        for (int step = -SearchSteps; step <= SearchSteps; step++)
        {
            var inverse = Vector128.Create(Divisor(step)) / divisor;
            Vector128<double> wqLow = Vector128<double>.Zero, wqHigh = Vector128<double>.Zero;
            var qq = Vector128<int>.Zero;
            for (int i = 0; i < count; i++)
            {
                var qi = Vector128.Min(Vector128.Max(Vector128.ConvertToInt32(Vector128.Round(wv[i] * inverse)), low8), high7);
                var qf = Vector128.ConvertToSingle(qi);
                qv[i] = qf;
                var product = wv[i] * qf;
                wqLow += Vector128.WidenLower(product);
                wqHigh += Vector128.WidenUpper(product);
                qq += qi * qi;
            }

            var (qqLow, qqHigh) = (Vector128.ConvertToDouble(Vector128.WidenLower(qq)), Vector128.ConvertToDouble(Vector128.WidenUpper(qq)));
            var d = Vector128.Narrow(wqLow / qqLow, wqHigh / qqHigh);
            Vector128<double> errorLow = Vector128<double>.Zero, errorHigh = Vector128<double>.Zero;
            for (int i = 0; i < count; i++)
            {
                var e = wv[i] - d * qv[i];
                var (eLow, eHigh) = (Vector128.WidenLower(e), Vector128.WidenUpper(e));
                errorLow += eLow * eLow;
                errorHigh += eHigh * eHigh;
            }

            // A lane whose nibbles are all zero (qq 0) takes nothing, as the reference skips it.
            var takenLow = Vector128.LessThan(errorLow, bestLow) & ~Vector128.Equals(qqLow, Vector128<double>.Zero);
            var takenHigh = Vector128.LessThan(errorHigh, bestHigh) & ~Vector128.Equals(qqHigh, Vector128<double>.Zero);
            bestLow = Vector128.ConditionalSelect(takenLow, errorLow, bestLow);
            bestHigh = Vector128.ConditionalSelect(takenHigh, errorHigh, bestHigh);
            var taken = Vector128.Narrow(takenLow.AsInt64(), takenHigh.AsInt64());
            bestScale = Vector128.ConditionalSelect(taken.AsSingle(), d, bestScale);
            bestStep = Vector128.ConditionalSelect(taken, Vector128.Create(step), bestStep);
        }

        for (int lane = 0; lane < 4; lane++)
        {
            int column = j + lane, chosen = bestStep.GetElement(lane);
            bool none = zero.AsInt32().GetElement(lane) != 0 || chosen == int.MinValue;
            scales[g * words * 8 + column] = none ? 0f : bestScale.GetElement(lane);
            if (none)
            {
                continue;                                                           // zero nibbles
            }

            // The chosen divisor's nibbles, made again as the search made them.
            float inverse = Divisor(chosen) / extreme.GetElement(lane);
            for (int i = 0; i < count; i++)
            {
                int nibble = Math.Clamp((int)MathF.Round(wv[i].GetElement(lane) * inverse), -8, 7);
                packed[(r0 + i) * words + (column >> 3)] |= (uint)(nibble & 15) << (4 * (column & 7));
            }
        }

        return true;
    }

    /// <summary>The float weights these nibbles stand for, [rows, columns], on the same device.</summary>
    public override Tensor Dequantize()
    {
        var w = Tensor.Empty([Rows, Columns], Packed.Device, track: false);   // the kernel writes every value: not zeroed first
        DequantizeInto(w.Storage);
        return w;
    }

    /// <inheritdoc />
    public override void DequantizeInto(Storage destination) => Packed.Backend.Int4Dequantize(Packed.Storage, Scales.Storage, destination, Rows, Columns);

    /// <inheritdoc />
    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move)
    {
        Packed = move(Packed, device);
        Scales = move(Scales, device);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [Packed, Scales];

    /// <inheritdoc />
    public override void Dispose()
    {
        Packed.Dispose();
        Scales.Dispose();
    }
}

/// <summary>
/// The bfloat16 weights of a <c>Linear</c> layer: each weight keeps float32's range with an 8-bit mantissa (about
/// 3 significant digits), in half the memory; decoding reads half the bytes of float32. Hugging Face checkpoints are
/// usually stored this way, so loading them as bfloat16 is exact. Created by <c>ModuleExtensions.ToBFloat16</c>
/// or when loading with bfloat16 weights; inputs, outputs, biases and LoRA adapters stay float32.
/// </summary>
public sealed class BFloat16Weight : PackedWeight, IKernelPackedWeight
{
    /// <inheritdoc />
    public override string Name => "bfloat16";

    /// <inheritdoc />
    public PackedFormat Format => PackedFormat.BFloat16;

    /// <inheritdoc />
    public override Device Device => Packed.Device;

    /// <inheritdoc />
    public Tensor PackedValues => Packed;

    /// <inheritdoc />
    public Tensor? ScaleValues => null;

    /// <summary>input [..., rows] · W → [..., columns], reading the bfloat16 values.</summary>
    public override Tensor MatMul(Tensor input) => input.MatMulBFloat16(this);

    /// <inheritdoc />
    public bool LowRankProducts => true;

    /// <inheritdoc />
    public bool Float8Copy => true;

    /// <inheritdoc />
    public bool ActivationInDownProjection => false;

    /// <inheritdoc />
    public bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank) =>
        g.Backend.BFloat16TransposedMatMul(g.Storage, Packed.Storage, dx, m, k, n, beta, dt, a, rank);

    /// <inheritdoc />
    public override string Description => "bfloat16";

    /// <inheritdoc />
    public override string ShortName => "bf16";

    private BFloat16Weight(Tensor packed, int rows, int columns)
    {
        Packed = packed;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Input features (rows of the weight matrix).</summary>
    public override int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public override int Columns { get; }

    /// <summary>Device memory used, in bytes.</summary>
    public override long Bytes => 4L * Packed.Size;

    /// <summary>The values, two per element along each row (rows padded to an even number of columns).</summary>
    public Tensor Packed { get; private set; }

    /// <summary>Rounds a float weight matrix [rows, columns] (on its device) to bfloat16.</summary>
    public static BFloat16Weight Convert(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"bfloat16 weights need a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return FromValues(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>Rounds host values [rows, columns] to bfloat16 (to nearest, ties to even) and uploads only those.</summary>
    public static unsafe BFloat16Weight FromValues(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        // The worker lambdas cannot capture a span: they read the pinned values through a pointer (no copy).
        fixed (float* pinned = values)
        {
            return FromValues(pinned, rows, columns, device);
        }
    }

    /// <summary>
    /// bfloat16 weights [rows, columns] whose bits <paramref name="write"/> puts in place: a checkpoint's stored bfloat16
    /// values copied as they are (through <see cref="CopyStoredBits"/>), which are the weights
    /// <see cref="FromValues(ReadOnlySpan{float}, int, int, Device)"/> makes from their float32 copies, bit for bit, without
    /// those copies. On the CPU the bits are written straight into the weights' memory; on another device into one host
    /// buffer of 16-bit values (half a float32 copy), uploaded once.
    /// </summary>
    /// <param name="rows">Input features (rows of the weight matrix).</param>
    /// <param name="columns">Output features (columns).</param>
    /// <param name="device">Where the weights are created.</param>
    /// <param name="write">
    /// Writes row r's <paramref name="columns"/> values at [r · stride] of the span it gets (stride, its second argument:
    /// the columns rounded up to an even number; the padding after each row is zeroed after it) and returns true; returns
    /// false when it wrote nothing (the source does not hold these bits), and then the memory is freed and the result is null.
    /// </param>
    public static BFloat16Weight? FromBits(int rows, int columns, Device device, Func<Span<ushort>, int, bool> write)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(write);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        int stride = (columns + 1) / 2 * 2, count = checked(rows * stride);
        var packed = Tensor.Empty([count / 2], device, track: false);   // every value is written below: not zeroed first
        try
        {
            var host = packed.Storage as IHostStorage;                       // in host memory: the words are written in place
            ushort[]? staged = host is null ? GC.AllocateUninitializedArray<ushort>(count) : null;
            var bits = host is not null ? MemoryMarshal.Cast<float, ushort>(host.HostMemory) : staged.AsSpan();
            if (!write(bits, stride))
            {
                packed.Dispose();
                return null;
            }

            if (stride != columns)
            {
                for (int r = 0; r < rows; r++)
                {
                    bits[r * stride + columns] = 0;
                }
            }

            if (staged is not null)
            {
                packed.Load(MemoryMarshal.Cast<ushort, float>(staged.AsSpan()));
            }

            return new BFloat16Weight(packed, rows, columns);
        }
        catch
        {
            packed.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Copies stored bfloat16 values as the bits <see cref="FromValues(ReadOnlySpan{float}, int, int, Device)"/> gives their
    /// float32 copies: each value as it is, except that a NaN is made quiet, as rounding makes it. <paramref name="stored"/>
    /// and <paramref name="destination"/> may be the same memory.
    /// </summary>
    /// <param name="stored">The stored values' bits.</param>
    /// <param name="destination">At least as long as <paramref name="stored"/>.</param>
    public static void CopyStoredBits(ReadOnlySpan<ushort> stored, Span<ushort> destination)
    {
        if (destination.Length < stored.Length)
        {
            throw new ArgumentException($"{destination.Length} values cannot hold {stored.Length}.", nameof(destination));
        }

        // A widened bfloat16 value rounds back to itself (its low 16 bits are zero, so the rounding adds nothing to the
        // high half); a NaN keeps its bits with the quiet bit (0x40 of the high half) set, as BFloat16Bits.Round does.
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            var magnitude = new System.Numerics.Vector<ushort>(0x7FFF);
            var infinity = new System.Numerics.Vector<ushort>(0x7F80);
            var quiet = new System.Numerics.Vector<ushort>(0x40);
            for (; i <= stored.Length - System.Numerics.Vector<ushort>.Count; i += System.Numerics.Vector<ushort>.Count)
            {
                var v = new System.Numerics.Vector<ushort>(stored[i..]);
                (v | (System.Numerics.Vector.GreaterThan(v & magnitude, infinity) & quiet)).CopyTo(destination[i..]);
            }
        }

        for (; i < stored.Length; i++)
        {
            int v = stored[i];
            destination[i] = (ushort)((v & 0x7FFF) > 0x7F80 ? v | 0x40 : v);
        }
    }

    /// <summary>
    /// Room for bfloat16 values [rows, columns] on <paramref name="device"/>, allocated without computing anything, for
    /// loading stored values into (<c>Idrak.ModuleFiles.Load</c>).
    /// </summary>
    public static BFloat16Weight Empty(int rows, int columns, Device device)
    {
        int stride = (columns + 1) / 2 * 2;
        return new BFloat16Weight(Tensor.PersistentZeros([rows * stride / 2], device), rows, columns);
    }

    private static unsafe BFloat16Weight FromValues(float* source, int rows, int columns, Device device)
    {
        int stride = (columns + 1) / 2 * 2;
        var halves = new ushort[rows * stride];
        HostParallel.For(rows, Math.Max(1, Abstraction.Devices.Cpu.CpuTuning.ParallelElements / Math.Max(1, columns)), (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                for (int j = 0; j < columns; j++)
                {
                    halves[r * stride + j] = Round(source[r * columns + j]);
                }
            }
        });

        var packed = MemoryMarshal.Cast<ushort, float>(halves);
        return new BFloat16Weight(Tensor.Persistent(packed, [packed.Length], device, requiresGrad: false), rows, columns);
    }

    /// <summary>A value's bfloat16 bits (round to nearest, ties to even; NaN stays NaN).</summary>
    internal static ushort Round(float value) => BFloat16Bits.Round(value);

    /// <summary>The float weights, [rows, columns], on the same device.</summary>
    public override Tensor Dequantize()
    {
        var w = Tensor.Empty([Rows, Columns], Packed.Device, track: false);   // the kernel writes every value: not zeroed first
        DequantizeInto(w.Storage);
        return w;
    }

    /// <inheritdoc />
    public override void DequantizeInto(Storage destination) => Packed.Backend.BFloat16Dequantize(Packed.Storage, destination, Rows, Columns);

    /// <inheritdoc />
    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move) => Packed = move(Packed, device);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [Packed];

    /// <inheritdoc />
    public override void Dispose() => Packed.Dispose();
}

/// <summary>
/// IEEE half-precision weights of a <c>Linear</c> layer, as a float16 checkpoint stores them (about 3 significant
/// digits, range ±65504), in half the memory of float32. Every product widens them exactly
/// (<c>Backend.Float16Dequantize</c>) and runs the float32 one, so the layer computes what a float32 layer holding the
/// checkpoint's values computes, bit for bit. Not an <see cref="IKernelPackedWeight"/>: the fused projections of the
/// built-in formats never see it, and every layer holding it runs <see cref="MatMul"/>, which only issues device work
/// (<see cref="IRecordable"/>). Two values per 32-bit
/// word along each row, as <see cref="BFloat16Weight"/> packs bfloat16.
/// </summary>
public sealed class Float16Weight : PackedWeight, IRecordable
{
    /// <summary>The format's name.</summary>
    public const string FormatName = "float16";

    private Float16Weight(Tensor packed, int rows, int columns)
    {
        Packed = packed;
        Rows = rows;
        Columns = columns;
    }

    /// <inheritdoc />
    public override string Name => FormatName;

    /// <inheritdoc />
    public override Device Device => Packed.Device;

    /// <inheritdoc />
    public override int Rows { get; }

    /// <inheritdoc />
    public override int Columns { get; }

    /// <inheritdoc />
    public override long Bytes => 4L * Packed.Size;

    /// <summary>The values, two per element along each row (rows padded to an even number of columns).</summary>
    public Tensor Packed { get; private set; }

    /// <summary>input [..., rows] · W → [..., columns]: the half values widened on the device, then the float32 product.</summary>
    public override Tensor MatMul(Tensor input) => input.MatMulFloat16(this);

    /// <summary>
    /// Rounds host values [rows, columns] to half precision (to nearest, ties to even: exact for values a float16
    /// checkpoint stores) and uploads only those.
    /// </summary>
    public static Float16Weight FromValues(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        int stride = (columns + 1) / 2 * 2;
        var halves = new ushort[rows * stride];
        for (int r = 0; r < rows; r++)
        {
            var row = values.Slice(r * columns, columns);
            var target = halves.AsSpan(r * stride, columns);
            for (int j = 0; j < columns; j++)
            {
                target[j] = BitConverter.HalfToUInt16Bits((Half)row[j]);
            }
        }

        var packed = MemoryMarshal.Cast<ushort, float>(halves);
        return new Float16Weight(Tensor.Persistent(packed, [packed.Length], device, requiresGrad: false), rows, columns);
    }

    /// <summary>Rounds a float weight matrix [rows, columns] (on its device) to half precision.</summary>
    public static Float16Weight Convert(Tensor weight)
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"float16 weights need a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return FromValues(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>Room for half values [rows, columns] on <paramref name="device"/>, for loading stored values into.</summary>
    public static Float16Weight Empty(int rows, int columns, Device device)
    {
        int stride = (columns + 1) / 2 * 2;
        return new Float16Weight(Tensor.PersistentZeros([rows * stride / 2], device), rows, columns);
    }

    /// <summary>The float weights, [rows, columns], on the same device.</summary>
    public override Tensor Dequantize()
    {
        var w = Tensor.Empty([Rows, Columns], Packed.Device, track: false);   // the kernel writes every value: not zeroed first
        DequantizeInto(w.Storage);
        return w;
    }

    /// <inheritdoc />
    public override void DequantizeInto(Storage destination) => Packed.Backend.Float16Dequantize(Packed.Storage, destination, Rows, Columns);

    /// <inheritdoc />
    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move) => Packed = move(Packed, device);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [Packed];

    /// <inheritdoc />
    public override void Dispose() => Packed.Dispose();
}
