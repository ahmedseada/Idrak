// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Formats;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A weight kept in its file's format (plan 37, Q0): the stored bytes of a registered element type
/// (<see cref="ElementTypes"/>: a GGUF block format such as q4_k, or float8 / int8 codes with their scales) as one typed
/// storage, [columns, rows] as stored (each output's row of inputs), so a low-bit model takes about its file's size in
/// memory. The product (<see cref="Backend.StoredMatMul"/>) expands the weights a tile of outputs at a time inside the
/// kernel and multiplies them as the float32 weights would be: the same values and the same arithmetic as the weights
/// expanded at load, bit for bit. A product that records gradients expands the whole matrix for it
/// (<see cref="Dequantize"/>), as the float path computes them. Frozen. Its stored bytes are in the device's memory
/// (<see cref="FromStored"/>), or read in place from a checkpoint's file mapping (plan 37, W3: <see cref="FromMapped"/>),
/// the file's own float32 rows among them.
/// </summary>
public sealed class StoredWeight : PackedWeight, IRecordable
{
    private Tensor _values;
    private Tensor? _scales, _zeros;
    private int _disposed;

    private StoredWeight(Tensor values, Tensor? scales, Tensor? zeros, StoredScales layout, int rows, int columns)
    {
        (_values, _scales, _zeros, Layout, Rows, Columns) = (values, scales, zeros, layout, rows, columns);
    }

    /// <summary>
    /// The weight [<paramref name="rows"/>, <paramref name="columns"/>] (inputs, outputs) from its stored bytes,
    /// <paramref name="stored"/>: <paramref name="columns"/> rows of <paramref name="rows"/> values of
    /// <paramref name="type"/> each (the file's [outputs, inputs] order), with their <paramref name="scaling"/> or none.
    /// </summary>
    /// <exception cref="ArgumentException">The type has no conversion to float32, a row is not whole blocks or bytes, the bytes or scales do not fit.</exception>
    public static StoredWeight FromStored(ElementType type, ReadOnlySpan<byte> stored, int rows, int columns, StoredScaling? scaling, Device device)
    {
        ArgumentNullException.ThrowIfNull(type);
        ArgumentNullException.ThrowIfNull(device);
        CheckRows(type, rows);
        long bytes = type.ByteCount((long)rows * columns);
        if (stored.Length != bytes)
        {
            throw new ArgumentException($"[{columns}, {rows}] {type.Name} values take {bytes} bytes; {stored.Length} given.", nameof(stored));
        }

        var layout = CheckScaling(scaling, rows, columns);
        Tensor? values = null;
        try
        {
            values = TensorScope.Untrack(Tensor.FromBytes(stored, type, [columns, rows], device));
            return WithScales(values, scaling, layout, rows, columns, device);
        }
        catch
        {
            values?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The weight [<paramref name="rows"/>, <paramref name="columns"/>] (inputs, outputs) read in place from
    /// <paramref name="mapping"/> (plan 37, W3): its stored bytes as <see cref="FromStored"/> takes them (the file's
    /// [outputs, inputs] rows of <paramref name="type"/>; float32 among the types), with their <paramref name="scaling"/> or
    /// none, on <paramref name="device"/>, whose kernels read the mapping where the file's pages lie
    /// (<see cref="IMappedStorageBackend"/>: the CPU), so the weights are not copied a second time into memory: the
    /// system's page cache holds them once. The products are <see cref="FromStored"/>'s bit for bit. The weight takes the
    /// mapping (closed when the weight is disposed, or at once when this throws); a copy taken of it later (a move to
    /// another device, an expansion to float32) is counted in <see cref="MappedWeights"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The type has no conversion to float32, a row is not whole blocks or bytes, the mapping does not hold the values, the scales do not fit.</exception>
    /// <exception cref="NotSupportedException">The device does not read a mapping in place.</exception>
    public static StoredWeight FromMapped(ElementType type, FileMapping mapping, int rows, int columns, StoredScaling? scaling, Device device)
    {
        ArgumentNullException.ThrowIfNull(mapping);
        Tensor? values = null;
        try
        {
            ArgumentNullException.ThrowIfNull(type);
            ArgumentNullException.ThrowIfNull(device);
            CheckRows(type, rows);
            var layout = CheckScaling(scaling, rows, columns);
            if (device.Backend is not IMappedStorageBackend mapped)
            {
                throw new NotSupportedException($"{device} does not read a file's mapping in place: read the weight into its memory (FromStored).");
            }

            // The storage takes the mapping (it closes it when it throws too; closing twice does nothing).
            values = Tensor.OverStorage(mapped.MapStorage(mapping, type, (long)rows * columns), [columns, rows], device);
            return WithScales(values, scaling, layout, rows, columns, device);
        }
        catch
        {
            if (values is not null)
            {
                values.Dispose();                                           // its storage closes the mapping
            }
            else
            {
                mapping.Dispose();
            }

            throw;
        }
    }

    // Which scale each value takes: none without scaling, else the scaling's layout, checked to cover [columns, rows].
    private static StoredScales CheckScaling(StoredScaling? scaling, int rows, int columns)
    {
        var layout = scaling?.Layout ?? default;
        if (scaling is not null)
        {
            bool shaped = layout.HasScales && layout.BlockColumns > 0 && layout.ScaleColumns > 0;
            int needed = !shaped || columns == 0 || rows == 0 ? 0 : layout.IndexOf(columns - 1, rows - 1) + 1;
            if (!shaped || scaling.Scales.Length < needed || scaling.Zeros is { } z && z.Length != scaling.Scales.Length)
            {
                throw new ArgumentException($"{scaling.Scales.Length} scales in blocks of [{layout.BlockRows}, {layout.BlockColumns}] do not cover [{columns}, {rows}].", nameof(scaling));
            }
        }

        return layout;
    }

    // The weight over `values` with its scales and zero points made on the device (values disposed by the caller on failure).
    private static StoredWeight WithScales(Tensor values, StoredScaling? scaling, StoredScales layout, int rows, int columns, Device device)
    {
        Tensor? scales = null, zeros = null;
        try
        {
            scales = scaling is null ? null : Tensor.Persistent(scaling.Scales, [scaling.Scales.Length], device);
            zeros = scaling?.Zeros is { } zeroPoints ? Tensor.Persistent(zeroPoints, [zeroPoints.Length], device) : null;
            return new StoredWeight(values, scales, zeros, layout, rows, columns);
        }
        catch
        {
            scales?.Dispose();
            zeros?.Dispose();
            throw;
        }
    }

    // A stored row of `rows` values must start on a byte (and a block) boundary, and the type must widen to float32.
    internal static void CheckRows(ElementType type, int rows)
    {
        if (type.ToFloat32 is null)
        {
            throw new ArgumentException($"The element type {type.Name} has no conversion to float32: its weights cannot be multiplied.", nameof(type));
        }

        if (rows % type.BlockSize != 0 || (long)rows / type.BlockSize * type.BlockBits % 8 != 0)
        {
            throw new ArgumentException($"A row of {rows} {type.Name} values is not whole blocks of {type.BlockSize} on a byte boundary.", nameof(rows));
        }
    }

    /// <summary>The element type the weights are stored in.</summary>
    public ElementType Type => _values.ElementType;

    /// <summary>The stored values [columns, rows] (outputs, inputs), of <see cref="Type"/>.</summary>
    public Tensor StoredValues => _values;

    /// <summary>The scales (<see cref="Layout"/>), or null.</summary>
    public Tensor? Scales => _scales;

    /// <summary>The zero points, one per scale, or null.</summary>
    public Tensor? Zeros => _zeros;

    /// <summary>Which scale each stored value takes.</summary>
    public StoredScales Layout { get; }

    /// <summary>
    /// The file mapping the stored values are read from in place (plan 37, W3: <see cref="FromMapped"/>), or null when
    /// they are in the device's memory.
    /// </summary>
    public FileMapping? Mapping => (_values.Storage as Devices.Cpu.MappedStorage)?.Mapping;

    /// <inheritdoc />
    public override string Name => Type.Name;

    /// <inheritdoc />
    public override int Rows { get; }

    /// <inheritdoc />
    public override int Columns { get; }

    /// <inheritdoc />
    public override long Bytes => _values.Storage.ByteLength + (_scales?.Storage.ByteLength ?? 0) + (_zeros?.Storage.ByteLength ?? 0);

    /// <inheritdoc />
    public override Device Device => _values.Device;

    /// <inheritdoc />
    public override string Description => Mapping is { } mapping ? $"{Type.Name} (read in place from {System.IO.Path.GetFileName(mapping.Path)})" : $"{Type.Name} (as stored)";

    /// <inheritdoc />
    public override string ShortName => Type.Name;

    /// <inheritdoc />
    public override Tensor Dequantize()
    {
        var w = Tensor.Empty([Rows, Columns], Device);
        DequantizeInto(w.Storage);
        return w;
    }

    /// <inheritdoc />
    public override void DequantizeInto(Storage destination) =>
        Device.Backend.ExpandStored(_values.Storage, _scales?.Storage, _zeros?.Storage, destination, Columns, Rows, Layout);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers()
    {
        yield return _values;
        if (_scales is not null)
        {
            yield return _scales;
        }

        if (_zeros is not null)
        {
            yield return _zeros;
        }
    }

    /// <inheritdoc />
    protected override void MoveTo(Device device, Func<Tensor, Device, Tensor> move)
    {
        _values = move(_values, device);
        _scales = _scales is null ? null : move(_scales, device);
        _zeros = _zeros is null ? null : move(_zeros, device);
    }

    /// <summary>
    /// input [..., <see cref="Rows"/>] · W: the device's stored product (expanded a tile at a time, the float32 product's
    /// bits); with gradients recorded, the whole matrix expanded for it as the float path computes them.
    /// </summary>
    public override Tensor MatMul(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        input.ThrowIfDisposed();
        int k = Rows, n = Columns;
        if (Tensor.WillRecord(input) || input.Device != Device || input.Rank < 2 || input.Shape[^1] != k)
        {
            return input.MatMulExpanded(this);
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = input.Rank == 2 ? input : input.Reshape(-1, k);
        int m = flat.Shape[0];
        var y = Tensor.Empty([m, n], Device);
        Device.Backend.StoredMatMul(flat.Storage, _values.Storage, _scales?.Storage, _zeros?.Storage, y.Storage, m, n, k, Layout);
        var result = input.Rank == 2 ? y : y.Reshape([.. input.Shape[..^1], n]);
        return Tensor.Traced("matmul_stored", result, start);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _values.Dispose();
        _scales?.Dispose();
        _zeros?.Dispose();
    }
}

/// <summary>
/// What a load did with the weights stored in a format of their own (plan 37, Q0): how many matrices it kept as stored
/// (<see cref="StoredWeight"/>) and their bytes, how many it expanded to float32 and their bytes, the formats, and why it
/// expanded (the setting, a device without the format's product), for the load's report.
/// </summary>
/// <param name="Kept">Matrices kept as stored.</param>
/// <param name="KeptBytes">Their bytes in memory.</param>
/// <param name="Expanded">Matrices stored in such a format but expanded to float32.</param>
/// <param name="ExpandedBytes">Their bytes in memory (float32).</param>
/// <param name="Formats">The stored formats seen, by name.</param>
/// <param name="Why">Why some were expanded, or null.</param>
public sealed record StoredWeightsSummary(int Kept, long KeptBytes, int Expanded, long ExpandedBytes, IReadOnlyList<string> Formats, string? Why)
{
    /// <summary>"weights kept packed: 14 matrices (q8_0), 1.2 MB; 0 expanded to float32 (0 MB)", or "weights expanded: ..." when none was kept.</summary>
    public override string ToString()
    {
        string formats = Formats.Count == 0 ? "" : $" ({string.Join(", ", Formats)})";
        string kept = string.Create(CultureInfo.InvariantCulture, $"{Kept} matrices{formats}, {Megabytes(KeptBytes)}");
        string expanded = string.Create(CultureInfo.InvariantCulture, $"{Expanded} expanded to float32 ({Megabytes(ExpandedBytes)}){(Why is null ? "" : ": " + Why)}");
        return Kept > 0 ? $"weights kept packed: {kept}; {expanded}" : $"weights expanded: {expanded}{formats}";
    }

    private static string Megabytes(long bytes) => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e6:0.##} MB");
}

/// <summary>
/// The setting that keeps weights in their file's format (plan 37, Q0): <see cref="EnvironmentVariable"/>
/// (IDRAK_KEEP_PACKED), on unless "0", "false", "off" or "no"; a load's option (<c>PretrainedOptions.KeepPacked</c>) wins.
/// Off, every weight is expanded to float32 at load, as before Q0 (the same results, to check against).
/// </summary>
public static class WeightStorage
{
    /// <summary>The variable that turns keeping weights as stored off when "0", "false", "off" or "no": "IDRAK_KEEP_PACKED".</summary>
    public const string EnvironmentVariable = "IDRAK_KEEP_PACKED";

    /// <summary>Whether weights are kept in their file's format: on unless <see cref="EnvironmentVariable"/> turns it off.</summary>
    public static bool DefaultKeep => Environment.GetEnvironmentVariable(EnvironmentVariable)?.Trim().ToLowerInvariant() is not ("0" or "false" or "off" or "no");

    /// <summary>
    /// Whether <paramref name="backend"/> multiplies weights stored as <paramref name="type"/> with a kernel of its own
    /// (<see cref="BackendCapabilities.StoresAnyFormat"/>, <see cref="BackendCapabilities.StoredFormats"/>): where it does
    /// not, a load expands them to float32 (a product through the host fallback would copy the weights on every call).
    /// </summary>
    public static bool Multiplies(Backend backend, ElementType type)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(type);
        var capabilities = backend.Capabilities;
        return type.ToFloat32 is not null && (capabilities.StoresAnyFormat || capabilities.StoredFormats.Contains(type.Name));
    }

    /// <summary>Publishes a load's <paramref name="summary"/> on <paramref name="device"/> to telemetry (<see cref="WeightsKept"/>, at <see cref="TelemetryLevel.Devices"/>).</summary>
    public static void Report(StoredWeightsSummary summary, Device device)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentNullException.ThrowIfNull(device);
        if (Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            Telemetry.WeightsKept(new WeightsKept(device.ToString(), summary.Kept, summary.KeptBytes, summary.Expanded, summary.ExpandedBytes,
                string.Join(", ", summary.Formats), summary.Why));
        }
    }
}
