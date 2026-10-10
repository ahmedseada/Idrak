// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// Packs float weights [rows, columns] (host values, row-major) into a <see cref="PackedWeight"/> on
/// <paramref name="device"/>; registered under a format name with <see cref="PackedWeight.Register"/>.
/// </summary>
/// <param name="values">The float weights, <paramref name="rows"/> × <paramref name="columns"/> values.</param>
/// <param name="rows">Input features (rows of the weight matrix).</param>
/// <param name="columns">Output features (columns).</param>
/// <param name="device">Where the packed weights are created.</param>
public delegate PackedWeight PackedWeightFactory(ReadOnlySpan<float> values, int rows, int columns, Device device);

/// <summary>
/// Weights of a <c>Linear</c> layer held in a packed format: the built-in <see cref="Int8Weight"/>,
/// <see cref="Int4Weight"/> and <see cref="BFloat16Weight"/>, or a format defined outside Idrak (NF4, FP8, …). The layer
/// and the fused products go through this type instead of checking which format a layer holds: each format says how it
/// multiplies, expands and moves, and which fused paths it takes.
/// <para>
/// A format of your own derives from this class and implements <see cref="Name"/>, <see cref="Rows"/>,
/// <see cref="Columns"/>, <see cref="Bytes"/>, <see cref="Dequantize"/>, <see cref="Buffers"/>, <see cref="MoveTo"/> and
/// <see cref="Dispose"/>; <see cref="MatMul"/> expands the weights for each product unless it is overridden with a
/// product that reads the packed form. Its <see cref="Format"/> is null: the device kernels for the built-in formats
/// (fused projections, gate/up pairs, LoRA products, the activation read by the down projection) never see it, and every
/// layer holding it runs <see cref="MatMul"/>. Register a factory with <see cref="Register"/> to create it by name
/// (<see cref="FromValues(string, ReadOnlySpan{float}, int, int, Device)"/>, <c>DecoderBuildOptions.PackedFormatName</c>).
/// </para>
/// </summary>
public abstract class PackedWeight : IDisposable
{
    /// <summary>A format defined outside Idrak (<see cref="Format"/> is null).</summary>
    protected PackedWeight()
    {
    }

    private protected PackedWeight(PackedFormat format) => Format = format;

    /// <summary>
    /// The built-in format (which the device kernels read directly), or null for a format defined outside Idrak, which
    /// layers multiply through <see cref="MatMul"/>.
    /// </summary>
    public PackedFormat? Format { get; }

    /// <summary>The format's name, as registered with <see cref="Register"/> ("int8", "int4", "bfloat16" for the built-in ones).</summary>
    public abstract string Name { get; }

    /// <summary>Input features (rows of the [rows, columns] weight).</summary>
    public abstract int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public abstract int Columns { get; }

    /// <summary>Device memory used, in bytes.</summary>
    public abstract long Bytes { get; }

    /// <summary>The device the weights are on (by default, that of the first of <see cref="Buffers"/>).</summary>
    public virtual Device Device => Buffers().First().Device;

    /// <summary>
    /// The float weights, [rows, columns], on the same device, as a new tensor the caller disposes.
    /// </summary>
    public abstract Tensor Dequantize();

    /// <summary>
    /// The tensors holding the weights (packed words, scales): the layer lists them as its <see cref="Module.Buffers"/>, so
    /// <c>Idrak.ModuleFiles.Save</c> writes them exactly as they are (and <c>Idrak.ModuleFiles.Load</c> reads them
    /// back into a model holding the same format), and offloading treats them as frozen weights.
    /// </summary>
    public abstract IEnumerable<Tensor> Buffers();

    /// <summary>
    /// Moves the weights to <paramref name="device"/> (<see cref="Module.To"/>): replace each tensor held by
    /// <c>move(tensor, device)</c>, which returns it on the device and releases the old one.
    /// </summary>
    /// <param name="device">The destination.</param>
    /// <param name="move">Moves one tensor.</param>
    protected abstract void MoveTo(Device device, Func<Tensor, Device, Tensor> move);

    /// <summary>Moves the weights to <paramref name="device"/> (what a layer holding them calls from its own <c>MoveTo</c>).</summary>
    /// <param name="device">The destination.</param>
    /// <param name="move">Moves one tensor: returns it on the device and releases the old one.</param>
    public void MoveWeights(Device device, Func<Tensor, Device, Tensor> move) => MoveTo(device, move);

    /// <summary>
    /// input [..., <see cref="Rows"/>] · W → [..., <see cref="Columns"/>], gradients flowing to the input (the weights are
    /// frozen). By default the weights are expanded with <see cref="Dequantize"/> for the product (and again for the
    /// input's gradient) and freed after it, so a format works without a kernel of its own; override this with a product
    /// that reads the packed form. A decoding step recorded as a device graph (CUDA) records this product, so an override
    /// must only issue device work (no host round trip); the default refuses to be recorded, since it cannot tell whether
    /// <see cref="Dequantize"/> does, and such steps run without a graph.
    /// </summary>
    /// <param name="input">[..., <see cref="Rows"/>] on the weights' device.</param>
    public virtual Tensor MatMul(Tensor input) => input.MatMulExpanded(this);

    /// <inheritdoc />
    public abstract void Dispose();

    /// <summary>
    /// The packed words the device kernels read (a built-in <see cref="Format"/> only; callers check it first).
    /// </summary>
    /// <exception cref="InvalidOperationException">The weights are not in a built-in format.</exception>
    public virtual Tensor PackedValues => throw new InvalidOperationException($"{Name} weights are not a built-in format; the device kernels cannot read them.");

    /// <summary>The scales the device kernels read with <see cref="PackedValues"/>, or null when the format has none.</summary>
    public virtual Tensor? ScaleValues => null;

    /// <summary>Writes the float weights [rows, columns] into <paramref name="destination"/> (on the same device).</summary>
    public virtual void DequantizeInto(Storage destination)
    {
        using var w = Dequantize();
        w.Backend.Copy(w.Storage, destination, Rows * Columns);
    }

    /// <summary>Whether the fused low-rank (LoRA) products read this format packed; otherwise the base product runs on its own.</summary>
    public virtual bool LowRankProducts => false;

    /// <summary>
    /// Whether an FP8 copy may be made for training products: only the fused low-rank products read one, and they take
    /// built-in formats only.
    /// </summary>
    public virtual bool Float8Copy => Format is not null;

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
    public virtual bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank) => false;

    /// <summary>
    /// Whether a gated feed-forward block applies the activation as the down projection reads its input (4-bit: eight
    /// columns per word repay that work), rather than in the gate/up product.
    /// </summary>
    public virtual bool ActivationInDownProjection => false;

    /// <summary>Packs [rows, columns] float values in <paramref name="format"/> on <paramref name="device"/>.</summary>
    public static PackedWeight FromValues(PackedFormat format, ReadOnlySpan<float> values, int rows, int columns, Device device) =>
        (BuiltIn.TryGetValue(format, out var builtIn) ? builtIn.Factory
            : throw new ArgumentOutOfRangeException(nameof(format), format, "Not a built-in packed format."))(values, rows, columns, device);

    // The built-in formats, by enum and name: the enum overload above always makes these (a registered name may replace
    // its entry in the registry below, not here), and the registry starts from them.
    private static readonly IReadOnlyDictionary<PackedFormat, (string Name, PackedWeightFactory Factory)> BuiltIn =
        new Dictionary<PackedFormat, (string, PackedWeightFactory)>
        {
            [PackedFormat.Int8] = ("int8", Int8Weight.Quantize),
            [PackedFormat.Int4] = ("int4", Int4Weight.Quantize),
            [PackedFormat.BFloat16] = ("bfloat16", BFloat16Weight.FromValues),
        };

    /// <summary>
    /// Packs [rows, columns] float values in the format registered as <paramref name="format"/> (see
    /// <see cref="FormatNames"/>) on <paramref name="device"/>.
    /// </summary>
    public static PackedWeight FromValues(string format, ReadOnlySpan<float> values, int rows, int columns, Device device) =>
        Pack(Factory(format), format, values, rows, columns, device);

    /// <summary>The weights <paramref name="factory"/> packs, checked against the [rows, columns] shape asked for.</summary>
    /// <exception cref="ArgumentException">The values do not fill the shape.</exception>
    /// <exception cref="InvalidOperationException">The factory made weights of another shape.</exception>
    public static PackedWeight Pack(PackedWeightFactory factory, string format, ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        var weight = factory(values, rows, columns, device);
        if (weight.Rows != rows || weight.Columns != columns)
        {
            weight.Dispose();
            throw new InvalidOperationException($"The '{format}' factory made [{weight.Rows}, {weight.Columns}] weights for [{rows}, {columns}] values.");
        }

        return weight;
    }

    private static readonly SlotTable<string, PackedWeightFactory> Registry = Formats();

    private static SlotTable<string, PackedWeightFactory> Formats()
    {
        var table = new SlotTable<string, PackedWeightFactory>(nameof(PackedWeight), GuardFactory, StringComparer.OrdinalIgnoreCase);
        foreach (var (name, factory) in BuiltIn.Values)
        {
            table.RegisterDefault(name, factory);
        }

        return table;
    }

    // Packing makes the weights once: an app's factory falls back when it fails to make them. Under Shadow both pack the
    // sampled weights, which agree when their shapes do; the app's are then freed.
    private static PackedWeightFactory GuardFactory(Slot slot, PackedWeightFactory app, PackedWeightFactory library) => (values, rows, columns, device) =>
    {
        if (slot.Policy == SlotPolicy.Shadow)
        {
            var run = slot.Shadow();
            var answer = library(values, rows, columns, device);
            if (run is not null)
            {
                run.Answered();
                try
                {
                    using var other = app(values, rows, columns, device);
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
            return app(values, rows, columns, device);
        }
        catch (Exception e) when (slot.Failed(e))
        {
            return library(values, rows, columns, device);
        }
    };

    /// <summary>
    /// Registers how to pack weights in the format <paramref name="format"/> (names ignore case), so options and tools can
    /// choose it by name. "int8", "int4" and "bfloat16" are registered; under one of their names the app's factory
    /// overrides the library's, which stays behind it (see <see cref="SetPolicy"/>). The overload of
    /// <see cref="FromValues(PackedFormat, ReadOnlySpan{float}, int, int, Device)"/> taking a <see cref="PackedFormat"/>
    /// always makes the built-in weights.
    /// </summary>
    /// <param name="format">The name to choose the format by.</param>
    /// <param name="factory">Packs float values in the format.</param>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string format, PackedWeightFactory factory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(format);
        ArgumentNullException.ThrowIfNull(factory);
        Registry.Register(format, factory, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="format"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string format) => Registry.Unregister(format);

    /// <summary>The registered format names.</summary>
    public static IReadOnlyCollection<string> FormatNames => Registry.Keys;

    /// <summary>The factory registered as <paramref name="format"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">No format of that name is registered.</exception>
    public static PackedWeightFactory Factory(string format) =>
        Registry.TryGet(format, out var factory) ? factory
            : throw new NotSupportedException($"No packed format '{format}' is registered ({string.Join(", ", Registry.Keys)}); add it with PackedWeight.Register.");

    /// <summary>The library's factory of <paramref name="format"/>, whatever an app registered over it; null when the library has none.</summary>
    public static PackedWeightFactory? Default(string format) => Registry.Default(format);

    /// <summary>Who registered the format <paramref name="format"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string format) => Registry.Origin(format);

    /// <summary>
    /// What happens when the app's factory <paramref name="format"/> fails to pack (<see cref="SlotPolicy.Throw"/> unless set:
    /// the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's). Packed weights are made once, so the fallback is at packing; their products do not fall back.
    /// </summary>
    public static void SetPolicy(string format, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(format, policy, shadowRate);

    /// <summary>The format in messages: "int8", "4-bit", "bfloat16".</summary>
    public virtual string Description => Name;

    /// <summary>The format's short name, as summaries show it.</summary>
    public virtual string ShortName => Name;

    /// <summary>How to get float weights back, as messages name it ("ToFloat32()").</summary>
    public virtual string FloatMethod => "ToFloat32()";
}

/// <summary>
/// The int8 weights of a quantized <c>Linear</c> layer: each weight is a signed byte times its output column's
/// scale (symmetric, per-column: scale = max |w| / 127). A quarter of the float32 memory; created by
/// <c>ModuleExtensions.QuantizeInt8</c>. Inputs, outputs, biases and LoRA adapters stay float32.
/// </summary>
public sealed class Int8Weight : PackedWeight
{
    /// <inheritdoc />
    public override string Name => "int8";

    /// <inheritdoc />
    public override Device Device => Packed.Device;

    /// <inheritdoc />
    public override Tensor PackedValues => Packed;

    /// <inheritdoc />
    public override Tensor? ScaleValues => Scales;

    /// <summary>input [..., rows] · W → [..., columns], reading the bytes (few rows) or expanding them once (more).</summary>
    public override Tensor MatMul(Tensor input) => input.MatMulInt8(this);

    /// <inheritdoc />
    public override bool Float8Copy => false;

    /// <inheritdoc />
    public override string Description => "int8";

    /// <inheritdoc />
    public override string ShortName => "int8";

    /// <inheritdoc />
    public override string FloatMethod => "DequantizeInt8()";

    private Int8Weight(Tensor packed, Tensor scales, int rows, int columns)
        : base(PackedFormat.Int8)
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
public sealed class Int4Weight : PackedWeight
{
    /// <inheritdoc />
    public override string Name => "int4";

    /// <inheritdoc />
    public override Device Device => Packed.Device;

    /// <inheritdoc />
    public override Tensor PackedValues => Packed;

    /// <inheritdoc />
    public override Tensor? ScaleValues => Scales;

    /// <summary>input [..., rows] · W → [..., columns], reading the nibbles.</summary>
    public override Tensor MatMul(Tensor input) => input.MatMulInt4(this);

    /// <inheritdoc />
    public override bool LowRankProducts => true;

    /// <inheritdoc />
    public override bool ActivationInDownProjection => true;

    /// <inheritdoc />
    public override string Description => "4-bit";

    /// <inheritdoc />
    public override string ShortName => "int4";

    /// <summary>Weight rows that share one scale per column.</summary>
    public const int GroupSize = 32;

    private Int4Weight(Tensor packed, Tensor scales, int rows, int columns)
        : base(PackedFormat.Int4)
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

    private static unsafe Int4Weight Quantize(float* source, int rows, int columns, Device device)
    {
        int words = (columns + 7) / 8, groups = (rows + GroupSize - 1) / GroupSize;
        var packed = new uint[rows * words];
        var scales = new float[groups * words * 8];
        HostParallel.For(groups, Math.Max(1, 2 * Abstraction.Devices.Cpu.CpuTuning.ParallelElements / GroupSize / Math.Max(1, columns)), (first, last) =>
        {
            Span<float> w = stackalloc float[GroupSize];
            Span<sbyte> q = stackalloc sbyte[GroupSize];
            Span<sbyte> best = stackalloc sbyte[GroupSize];
            for (int g = first; g < last; g++)
            {
                int r0 = g * GroupSize, count = Math.Min(rows, r0 + GroupSize) - r0;
                for (int j = 0; j < columns; j++)
                {
                    float extreme = 0f;
                    for (int i = 0; i < count; i++)
                    {
                        w[i] = source[(r0 + i) * columns + j];
                        extreme = MathF.Abs(w[i]) > MathF.Abs(extreme) ? w[i] : extreme;
                    }

                    float scale = 0f;
                    if (extreme != 0f)
                    {
                        // Divisors t around -8 (the group's extreme maps near -8); for each, the least-squares scale of
                        // the rounded values; keep the smallest squared error (t = -8 is the plain rule, never worse).
                        double bestError = double.MaxValue;
                        for (int step = -10; step <= 10; step++)
                        {
                            float t = -8f + 0.1f * step, inverse = t / extreme;
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
                    else
                    {
                        best.Clear();
                    }

                    scales[g * words * 8 + j] = scale;
                    for (int i = 0; i < count; i++)
                    {
                        packed[(r0 + i) * words + (j >> 3)] |= (uint)(best[i] & 15) << (4 * (j & 7));
                    }
                }
            }
        });

        return new Int4Weight(
            Tensor.Persistent(MemoryMarshal.Cast<uint, float>(packed), [packed.Length], device, requiresGrad: false),
            Tensor.Persistent(scales, [scales.Length], device, requiresGrad: false),
            rows, columns);
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
public sealed class BFloat16Weight : PackedWeight
{
    /// <inheritdoc />
    public override string Name => "bfloat16";

    /// <inheritdoc />
    public override Device Device => Packed.Device;

    /// <inheritdoc />
    public override Tensor PackedValues => Packed;

    /// <summary>input [..., rows] · W → [..., columns], reading the bfloat16 values.</summary>
    public override Tensor MatMul(Tensor input) => input.MatMulBFloat16(this);

    /// <inheritdoc />
    public override bool LowRankProducts => true;

    /// <inheritdoc />
    public override bool TransposedProduct(Tensor g, Storage dx, int m, int k, int n, float beta, Storage? dt, Storage? a, int rank) =>
        g.Backend.BFloat16TransposedMatMul(g.Storage, Packed.Storage, dx, m, k, n, beta, dt, a, rank);

    /// <inheritdoc />
    public override string Description => "bfloat16";

    /// <inheritdoc />
    public override string ShortName => "bf16";

    private BFloat16Weight(Tensor packed, int rows, int columns)
        : base(PackedFormat.BFloat16)
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
