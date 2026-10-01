// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>A module with an incremental (cached) forward pass for autoregressive decoding.</summary>
public interface ICachedModule
{
    /// <summary>
    /// Processes only the new positions in <paramref name="input"/> ([batch, newSteps, ...]), reading and
    /// extending the state kept in <paramref name="context"/>. Inference only (no gradients).
    /// </summary>
    Tensor ForwardCached(Tensor input, DecodingContext context);
}

/// <summary>How a <see cref="KeyValueCache"/> stores keys and values.</summary>
public enum KeyValueFormat
{
    /// <summary>32-bit floats (exact).</summary>
    Float32,

    /// <summary>
    /// One signed byte per value plus one float scale per cached row (a head at one position): about a quarter of the
    /// memory, so longer contexts or larger batches fit; attention reads the bytes directly.
    /// </summary>
    Int8,

    /// <summary>
    /// bfloat16 values (float32's range, about 3 significant digits): half the memory of <see cref="Float32"/> and nearly
    /// the same results; attention reads the halves directly. Decoder models (<see cref="DecoderSpec"/>) only.
    /// </summary>
    BFloat16,
}

/// <summary>
/// Keys and values of one attention layer for every position decoded so far, [batch·heads, capacity, headDim]. With
/// <see cref="KeyValueFormat.Int8"/>, <see cref="Keys"/> and <see cref="Values"/> hold packed bytes
/// ([batch·heads, capacity, ceil(headDim / 4)] elements of four bytes) and <see cref="KeyScales"/>/<see cref="ValueScales"/>
/// one scale per row.
/// </summary>
public sealed class KeyValueCache : IDisposable
{
    internal KeyValueCache(int rows, int capacity, int headDim, Device device, KeyValueFormat format)
    {
        // Created outside any TensorScope: caches are usually created lazily inside a scoped prefill step.
        Format = format;
        HeadDim = headDim;
        int width = format switch
        {
            KeyValueFormat.Int8 => (headDim + 3) / 4,
            KeyValueFormat.BFloat16 => (headDim + 1) / 2,
            _ => headDim,
        };
        Keys = Tensor.Empty([rows, capacity, width], device, zeroed: true, track: false);
        Values = Tensor.Empty([rows, capacity, width], device, zeroed: true, track: false);
        if (format == KeyValueFormat.Int8)
        {
            KeyScales = Tensor.Empty([rows, capacity], device, zeroed: true, track: false);
            ValueScales = Tensor.Empty([rows, capacity], device, zeroed: true, track: false);
        }
    }

    /// <summary>How keys and values are stored.</summary>
    public KeyValueFormat Format { get; }

    /// <summary>Values per head.</summary>
    public int HeadDim { get; }

    /// <summary>Cached keys, [batch·heads, capacity, headDim] (packed bytes for int8, bfloat16 pairs for bfloat16).</summary>
    public Tensor Keys { get; }

    /// <summary>Cached values, [batch·heads, capacity, headDim] (packed bytes for int8).</summary>
    public Tensor Values { get; }

    /// <summary>For int8: the scale of each cached key row, [batch·heads, capacity].</summary>
    public Tensor? KeyScales { get; }

    /// <summary>For int8: the scale of each cached value row, [batch·heads, capacity].</summary>
    public Tensor? ValueScales { get; }

    /// <summary>Device memory used, in bytes.</summary>
    public long Bytes => 4L * (Keys.Size + Values.Size + (KeyScales?.Size ?? 0) + (ValueScales?.Size ?? 0));

    /// <inheritdoc />
    public void Dispose()
    {
        Keys.Dispose();
        Values.Dispose();
        KeyScales?.Dispose();
        ValueScales?.Dispose();
    }
}

/// <summary>
/// State for incremental decoding with <see cref="ICachedModule"/> layers: the number of positions already
/// processed (kept in device memory, so a decoding step can be recorded once as a <see cref="ComputeGraph"/>
/// and replayed at every position), one <see cref="KeyValueCache"/> per attention layer, and the current step's
/// causal mask and position indices.
/// </summary>
/// <example>
/// <code>
/// using var context = new DecodingContext(device, batch: 1, capacity: 64);
/// using (Autograd.NoGrad())
/// {
///     var logits = model.ForwardCached(promptIds, context);   // prefill: all prompt positions at once
///     // then, per new token: model.ForwardCached(nextId, context) processes one position
/// }
/// </code>
/// </example>
public sealed class DecodingContext : IDisposable
{
    private readonly Dictionary<object, KeyValueCache> _caches = new(ReferenceEqualityComparer.Instance);
    private Tensor? _iota;

    /// <summary>Creates an empty context.</summary>
    /// <param name="device">Where caches and counters live (the model's device).</param>
    /// <param name="batch">Sequences decoded together.</param>
    /// <param name="capacity">Maximum positions (the model's context length).</param>
    /// <param name="format">How the attention layers' keys and values are cached (int8: about a quarter of the memory).</param>
    public DecodingContext(Device device, int batch, int capacity, KeyValueFormat format = KeyValueFormat.Float32)
    {
        Format = format;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        Device = device;
        Batch = batch;
        Capacity = capacity;
        Position = Tensor.Persistent([0f], [1], device, requiresGrad: false);
    }

    /// <summary>How keys and values are cached.</summary>
    public KeyValueFormat Format { get; }

    /// <summary>Device memory used by the key/value caches created so far, in bytes.</summary>
    public long CacheBytes => _caches.Values.Sum(c => c.Bytes);

    /// <summary>The device of the caches.</summary>
    public Device Device { get; }

    /// <summary>Sequences decoded together.</summary>
    public int Batch { get; }

    /// <summary>Maximum number of cached positions.</summary>
    public int Capacity { get; }

    /// <summary>Positions processed so far, as a one-element device tensor (advanced by <see cref="EndStep"/>).</summary>
    public Tensor Position { get; }

    /// <summary>Positions processed so far, as tracked on the host.</summary>
    public int Length { get; private set; }

    /// <summary>
    /// The current step's [newSteps, capacity] causal mask, or null outside a step. Built on first use (the tiled and
    /// decoding attention kernels never need it: for a long prompt it would be newSteps × capacity floats).
    /// </summary>
    public Tensor? Mask => InStep ? _mask ??= Tensor.DecoderMask(Position, _steps, Capacity, track: false) : null;

    private Tensor? _mask;
    private int _steps;

    /// <summary>Whether a step is open (between <see cref="BeginStep"/> and <see cref="EndStep"/>).</summary>
    public bool InStep { get; private set; }

    /// <summary>
    /// When set, <see cref="Sequential.ForwardCached(Tensor, DecodingContext)"/> returns only the last new position's outputs: the layers after the
    /// last cached (attention) layer — final norm and output head — run on one position instead of every prompt position
    /// (sampling reads only the last one; a long prompt's full logits would be prompt × vocabulary floats).
    /// </summary>
    public bool LastPositionOnly { get; set; }

    /// <summary>The current step's position indices, [newSteps] (set by <see cref="BeginStep"/>).</summary>
    public Tensor? Positions { get; private set; }

    /// <summary>
    /// Rows of different lengths decoded together: row b's sequence starts at position RowStarts[b] (its prompt padded
    /// on the left so every row's prompt ends at the same position). Its tokens are numbered from there and see no earlier
    /// position. Null (the default): every row starts at 0. Needs a <see cref="KeyValueFormat.Float32"/> cache.
    /// </summary>
    public IReadOnlyList<int>? RowStarts { get; private set; }

    /// <summary>Sets <see cref="RowStarts"/> (one start per row; null: all rows start at 0).</summary>
    public void SetRowStarts(IReadOnlyList<int>? starts)
    {
        if (starts is not null && (starts.Count != Batch || starts.Any(s => s < 0 || s >= Capacity)))
        {
            throw new ArgumentException($"Row starts: one per row ({Batch}), each within the capacity ({Capacity}).");
        }

        RowStarts = starts?.ToArray();
    }

    /// <summary>With <see cref="RowStarts"/>: the current step's position of each new token within its row's sequence, [batch · newSteps].</summary>
    internal Tensor? TokenPositions { get; private set; }

    /// <summary>With <see cref="RowStarts"/>: each new token's row start, [batch · newSteps].</summary>
    internal Tensor? TokenStarts { get; private set; }

    /// <summary>Prepares mask and positions for <paramref name="steps"/> new positions (all computed on the device).</summary>
    public void BeginStep(int steps)
    {
        if (Length + steps > Capacity)
        {
            throw new InvalidOperationException($"Decoding past the context capacity ({Capacity}); call Reset and re-feed a shorter window.");
        }

        _steps = steps;
        InStep = true;
        if (RowStarts is { } starts)
        {
            var positions = new float[Batch * steps];
            var first = new float[Batch * steps];
            for (int b = 0; b < Batch; b++)
            {
                for (int t = 0; t < steps; t++)
                {
                    positions[b * steps + t] = Math.Max(0, Length + t - starts[b]);      // padding: position 0 (it attends to nothing)
                    first[b * steps + t] = starts[b];
                }
            }

            TokenPositions = Tensor.Persistent(positions, [Batch * steps], Device, requiresGrad: false);
            TokenStarts = Tensor.Persistent(first, [Batch * steps], Device, requiresGrad: false);
        }

        if (steps == 1)
        {
            Positions = Position;
        }
        else
        {
            if (_iota is null || _iota.Size < steps)
            {
                _iota?.Dispose();
                _iota = Tensor.Persistent([.. Enumerable.Range(0, Capacity).Select(i => (float)i)], [Capacity], Device, requiresGrad: false);
            }

            var positions = _iota.Narrow(0, 0, steps);
            positions.Backend.AddBroadcastScalar(Position.Storage, positions.Storage, steps, 1f);
            Positions = positions;
        }
    }

    /// <summary>Marks <paramref name="steps"/> positions as processed (device counter and host mirror).</summary>
    public void EndStep(int steps)
    {
        Position.AddInPlace(steps);
        Length += steps;
        ClearStep();
    }

    /// <summary>
    /// Records one decoding step (which must call <see cref="BeginStep"/>/<see cref="EndStep"/>, e.g. via
    /// <see cref="Sequential.ForwardCached(Tensor, DecodingContext)"/>) as a <see cref="ComputeGraph"/>. Because positions, masks and cache
    /// offsets are computed on the device from <see cref="Position"/>, the same graph is valid at every position.
    /// </summary>
    public ComputeGraph CaptureStep(Action step)
    {
        if (RowStarts is not null)
        {
            throw new InvalidOperationException("Steps with row starts upload positions from the host and cannot be recorded.");
        }

        int length = Length;
        var graph = ComputeGraph.Capture(Device, step);
        Length = length;   // recording does not execute the step
        ClearStep();
        return graph;
    }

    private void ClearStep()
    {
        _mask?.Dispose();
        _mask = null;
        TokenPositions?.Dispose();
        TokenPositions = null;
        TokenStarts?.Dispose();
        TokenStarts = null;
        InStep = false;
        Positions = null;
    }

    /// <summary>Replays a step recorded with <see cref="CaptureStep"/> and advances the host-side length.</summary>
    public void ReplayStep(ComputeGraph graph, int steps = 1)
    {
        if (Length + steps > Capacity)
        {
            throw new InvalidOperationException($"Decoding past the context capacity ({Capacity}); call Reset and re-feed a shorter window.");
        }

        int length = Length;
        graph.Replay();
        Length = length + steps;
    }

    /// <summary>Forgets all positions; caches are overwritten from position 0.</summary>
    public void Reset()
    {
        Position.FillInPlace(0f);
        Length = 0;
    }

    /// <summary>
    /// Keeps the first <paramref name="length"/> positions and forgets the rest (their cache entries are overwritten by
    /// the next steps): reuses the cached keys and values of a shared prompt prefix.
    /// </summary>
    public void Truncate(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Length);
        Position.FillInPlace(length);
        Length = length;
    }

    /// <summary>The cache for <paramref name="owner"/> (one per attention layer), created on first use.</summary>
    internal KeyValueCache CacheFor(object owner, int rows, int headDim)
    {
        if (!_caches.TryGetValue(owner, out var cache))
        {
            _caches[owner] = cache = new KeyValueCache(rows, Capacity, headDim, Device, Format);
        }

        return cache;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (var cache in _caches.Values)
        {
            cache.Dispose();
        }

        _caches.Clear();
        _iota?.Dispose();
        Position.Dispose();
    }
}
