// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction.Generation;

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
    /// the same results; decoder models (<c>DecoderSpec</c>) read the halves directly, the multi-head attention
    /// layer expands them to float32.
    /// </summary>
    BFloat16,

    /// <summary>
    /// A format of one's own (an <see cref="IKeyValueLayout"/> registered with <see cref="KeyValueLayouts.Register"/>): what
    /// <see cref="KeyValueCache.Format"/> and <see cref="DecodingContext.Format"/> report for it. Not a format to choose;
    /// pass the layout itself.
    /// </summary>
    Custom,
}

/// <summary>
/// Keys and values of one attention layer for every position decoded so far, [batch·heads, capacity, headDim]. With
/// <see cref="KeyValueFormat.Int8"/>, <see cref="Keys"/> and <see cref="Values"/> hold packed bytes
/// ([batch·heads, capacity, ceil(headDim / 4)] elements of four bytes) and <see cref="KeyScales"/>/<see cref="ValueScales"/>
/// one scale per row. A format of one's own (<see cref="IKeyValueLayout"/>) chooses its row width and scales.
/// </summary>
public sealed class KeyValueCache : IDisposable
{
    internal KeyValueCache(int rows, int capacity, int headDim, Device device, KeyValueFormat format)
        : this(rows, capacity, headDim, device, KeyValueLayouts.For(format))
    {
    }

    /// <summary>
    /// Creates an empty (zeroed) cache stored by <paramref name="layout"/>: usually made by <see cref="DecodingContext"/>
    /// for each attention layer; directly, e.g. to test a format of one's own.
    /// </summary>
    /// <param name="rows">Rows of cached heads (batch · key/value heads).</param>
    /// <param name="capacity">Maximum positions.</param>
    /// <param name="headDim">Values per head.</param>
    /// <param name="device">Where the cache lives.</param>
    /// <param name="layout">How keys and values are stored and attended over.</param>
    public KeyValueCache(int rows, int capacity, int headDim, Device device, IKeyValueLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        // Created outside any TensorScope: caches are usually created lazily inside a scoped prefill step.
        Layout = layout;
        Format = KeyValueLayouts.FormatOf(layout);
        RowStartsLayout = layout as IRowStartsLayout;
        SlidingWindowLayout = layout as ISlidingWindowLayout;
        FusedWriteLayout = layout as IFusedWriteLayout;
        HeadDim = headDim;
        int width = layout.RowWidth(headDim);
        if (width <= 0)
        {
            throw new InvalidOperationException($"The key/value format '{layout.Name}' gave a row width of {width} for {headDim} values per head.");
        }

        Keys = Tensor.Empty([rows, capacity, width], device, zeroed: true, track: false);
        Values = Tensor.Empty([rows, capacity, width], device, zeroed: true, track: false);
        if (layout is IScaledLayout)
        {
            KeyScales = Tensor.Empty([rows, capacity], device, zeroed: true, track: false);
            ValueScales = Tensor.Empty([rows, capacity], device, zeroed: true, track: false);
        }
    }

    /// <summary>How keys and values are stored: a built-in format, or <see cref="KeyValueFormat.Custom"/> (see <see cref="Layout"/>).</summary>
    public KeyValueFormat Format { get; }

    /// <summary>Values per head.</summary>
    public int HeadDim { get; }

    /// <summary>How keys and values are written and attended over.</summary>
    public IKeyValueLayout Layout { get; }

    /// <summary>The layout's attention over rows of different lengths, or null when it has none (tested once, when the cache is made).</summary>
    public IRowStartsLayout? RowStartsLayout { get; }

    /// <summary>The layout's sliding-window and soft-capped attention, or null when it has none (tested once, when the cache is made).</summary>
    public ISlidingWindowLayout? SlidingWindowLayout { get; }

    /// <summary>The layout's fused writes from the projection kernel, or null when it takes none (tested once, when the cache is made).</summary>
    public IFusedWriteLayout? FusedWriteLayout { get; }

    /// <summary>
    /// Cached keys, [batch·heads, capacity, headDim] (packed bytes for int8, bfloat16 pairs for bfloat16; a format of one's
    /// own: [batch·heads, capacity, <see cref="IKeyValueLayout.RowWidth"/>]).
    /// </summary>
    public Tensor Keys { get; }

    /// <summary>Cached values, [batch·heads, capacity, headDim] (packed bytes for int8).</summary>
    public Tensor Values { get; }

    /// <summary>For int8 (or another <see cref="IScaledLayout"/>): the scale of each cached key row, [batch·heads, capacity].</summary>
    public Tensor? KeyScales { get; }

    /// <summary>For int8 (or another <see cref="IScaledLayout"/>): the scale of each cached value row, [batch·heads, capacity].</summary>
    public Tensor? ValueScales { get; }

    /// <summary>Device memory used, in bytes.</summary>
    public long Bytes => 4L * (Keys.Size + Values.Size + (KeyScales?.Size ?? 0) + (ValueScales?.Size ?? 0));

    /// <summary>Positions the cache holds (its slots).</summary>
    public int Capacity => Keys.Shape[1];

    // The window this cache slides with (DecodingContext.SlidingWindowSteps), or null for a cache of the context's capacity.
    internal SlidingWindow? Sliding { get; init; }

    // Moves the positions in slots [from, from + count) to slots [0, count), keys, values and scales alike (a copy first:
    // the ranges may overlap). Every element is copied as it is, whatever the format packs into it.
    internal void MoveToFront(int from, int count)
    {
        Move(Keys, from, count);
        Move(Values, from, count);
        Move(KeyScales, from, count);
        Move(ValueScales, from, count);

        static void Move(Tensor? rows, int from, int count)
        {
            if (rows is null || count == 0 || from == 0)
            {
                return;
            }

            int n = rows.Shape[0], capacity = rows.Shape[1], inner = rows.Size / (n * capacity);
            using var moved = rows.Narrow(1, from, count);
            rows.Backend.Copy2D(moved.Storage, 0, count * inner, rows.Storage, 0, capacity * inner, n, count * inner, accumulate: false);
        }
    }

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
/// State for incremental decoding with <c>ICachedModule</c> layers: the number of positions already
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

    // What the layout can do, tested once (IRecordableLayout, ISlidingWindowLayout).
    private readonly bool _recordableLayout, _slidingLayout;
    private Tensor? _iota;

    // The memory of the cross-attention layers (an encoder's output projected to each layer's keys and values, once per
    // sequence), by layer: read by every decoding step, owned by this context.
    private readonly Dictionary<object, (Tensor Keys, Tensor Values)> _memories = new(ReferenceEqualityComparer.Instance);

    // Each memory's lengths (a padded batch of memories: row b sees its first lengths[b] positions), by layer; none when
    // every row sees the whole memory.
    private readonly Dictionary<object, int[]> _memoryLengths = new(ReferenceEqualityComparer.Instance);

    // The sliding windows' caches (SlidingWindowSteps), one group per window length.
    private readonly List<SlidingWindow> _windows = [];
    private int _slidingSteps;

    // The rotary offsets (RotaryOffsets): host values, the device copy a uniform offset is added from, and each pass that
    // set them (its first image's length, the length when set, the offsets before) for Truncate.
    private int[]? _rotary;
    private Tensor? _rotaryOffset;
    private bool _rotaryOwned;
    private readonly List<(int From, int End, int[]? Before)> _rotaryPasses = [];

    // With RowStarts: the starts on the device, [batch] (uploaded once by SetRowStarts), and the buffer a one-token step's
    // positions are computed into from Position (written in place, so a recorded step reads the position it is replayed at);
    // whether the step's TokenPositions and TokenStarts are its own uploads (a step of several positions) to free.
    private Tensor? _rowStarts;
    private Tensor? _rowPositions;
    private bool _tokensOwned;

    /// <summary>Creates an empty context.</summary>
    /// <param name="device">Where caches and counters live (the model's device).</param>
    /// <param name="batch">Sequences decoded together.</param>
    /// <param name="capacity">Maximum positions (the model's context length).</param>
    /// <param name="format">How the attention layers' keys and values are cached (int8: about a quarter of the memory).</param>
    public DecodingContext(Device device, int batch, int capacity, KeyValueFormat format = KeyValueFormat.Float32)
        : this(device, batch, capacity, KeyValueLayouts.For(format))
    {
    }

    /// <summary>Creates an empty context whose caches are stored by <paramref name="layout"/> (a built-in or one's own format).</summary>
    /// <param name="device">Where caches and counters live (the model's device).</param>
    /// <param name="batch">Sequences decoded together.</param>
    /// <param name="capacity">Maximum positions (the model's context length).</param>
    /// <param name="layout">How the attention layers' keys and values are cached (e.g. <see cref="KeyValueLayouts.Get"/>).</param>
    public DecodingContext(Device device, int batch, int capacity, IKeyValueLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        Layout = layout;
        Format = KeyValueLayouts.FormatOf(layout);
        _recordableLayout = layout is IRecordableLayout;
        _slidingLayout = layout is ISlidingWindowLayout;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        Device = device;
        Batch = batch;
        Capacity = capacity;
        Position = Tensor.Persistent([0f], [1], device, requiresGrad: false);
    }

    /// <summary>How keys and values are cached: a built-in format, or <see cref="KeyValueFormat.Custom"/> (see <see cref="Layout"/>).</summary>
    public KeyValueFormat Format { get; }

    /// <summary>How keys and values are cached and attended over.</summary>
    public IKeyValueLayout Layout { get; }

    /// <summary>Device memory used by the key/value caches created so far, in bytes.</summary>
    public long CacheBytes => _caches.Values.Sum(c => c.Bytes);

    /// <summary>Device memory used by the cross-attention memories (<see cref="SetMemory(object, Tensor, Tensor)"/>), in bytes.</summary>
    public long MemoryBytes => _memories.Values.Sum(m => 4L * (m.Keys.Size + m.Values.Size));

    /// <summary>
    /// Holds the memory of <paramref name="owner"/> (a cross-attention layer): an encoder's output projected to that layer's
    /// keys and values, [batch · heads, memory, headDim] each, computed once per sequence and read by every decoding step
    /// (<see cref="MemoryOf"/>). The context takes ownership; an earlier memory of the same layer is disposed. It stays
    /// through <see cref="Reset"/> (the same sound decoded again from position 0) until <see cref="ClearMemory"/> or
    /// <see cref="Dispose"/>.
    /// </summary>
    /// <exception cref="ArgumentException">Keys and values are not [rows, memory, headDim] alike, or not on this context's device.</exception>
    public void SetMemory(object owner, Tensor keys, Tensor values) => SetMemory(owner, keys, values, null);

    /// <summary>
    /// <see cref="SetMemory(object, Tensor, Tensor)"/> for a padded batch of memories: row b of the context sees the first
    /// <paramref name="lengths"/>[b] memory positions only (all of them when null), read back by <see cref="MemoryLengthsOf"/>.
    /// </summary>
    /// <exception cref="ArgumentException">Keys and values do not fit, or the lengths are not one per row of the context, each 1 to the memory's positions.</exception>
    public void SetMemory(object owner, Tensor keys, Tensor values, IReadOnlyList<int>? lengths)
    {
        ArgumentNullException.ThrowIfNull(owner);
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        if (keys.Rank != 3 || !keys.Shape.SequenceEqual(values.Shape) || keys.Device != Device || values.Device != Device)
        {
            throw new ArgumentException($"A memory's keys and values are [rows, memory, headDim] alike on {Device}; got {Tensor.FormatShape(keys.Shape)} on {keys.Device} and {Tensor.FormatShape(values.Shape)} on {values.Device}.");
        }

        if (lengths is not null)
        {
            if (lengths.Count != Batch)
            {
                throw new ArgumentException($"{lengths.Count} memory lengths for a context of {Batch} rows.", nameof(lengths));
            }

            foreach (int length in lengths)
            {
                if (length < 1 || length > keys.Shape[1])
                {
                    throw new ArgumentException($"A memory length of {length} is outside 1 to the memory's {keys.Shape[1]} positions.", nameof(lengths));
                }
            }
        }

        if (_memories.TryGetValue(owner, out var earlier))
        {
            earlier.Keys.Dispose();
            earlier.Values.Dispose();
        }

        _memories[owner] = (keys, values);
        if (lengths is null || lengths.All(l => l == keys.Shape[1]))
        {
            _memoryLengths.Remove(owner);
        }
        else
        {
            _memoryLengths[owner] = [.. lengths];
        }
    }

    /// <summary>The lengths the memory of <paramref name="owner"/> was set with (one per row), or null when every row sees the whole memory.</summary>
    public IReadOnlyList<int>? MemoryLengthsOf(object owner)
    {
        ArgumentNullException.ThrowIfNull(owner);
        return _memoryLengths.GetValueOrDefault(owner);
    }

    /// <summary>The memory held for <paramref name="owner"/> (<see cref="SetMemory(object, Tensor, Tensor)"/>), or null when none is.</summary>
    public (Tensor Keys, Tensor Values)? MemoryOf(object owner)
    {
        ArgumentNullException.ThrowIfNull(owner);
        return _memories.TryGetValue(owner, out var memory) ? memory : null;
    }

    /// <summary>Frees every memory set with <see cref="SetMemory(object, Tensor, Tensor)"/> (a new sound's encoder output replaces them).</summary>
    public void ClearMemory()
    {
        foreach (var (keys, values) in _memories.Values)
        {
            keys.Dispose();
            values.Dispose();
        }

        _memories.Clear();
        _memoryLengths.Clear();
    }

    /// <summary>
    /// Reorders the rows of every cache: row b takes the positions of row <paramref name="sources"/>[b] (a beam search
    /// keeping the best beams; a row may be taken by several). Every format's rows are copied as they are packed (keys,
    /// values and scales), so it works for the int8 and bfloat16 formats and one's own. Between steps only; the
    /// cross-attention memories, the same for every row, are not touched.
    /// </summary>
    /// <exception cref="ArgumentException">Not one source per row, or a source outside the batch.</exception>
    /// <exception cref="InvalidOperationException">A step is open.</exception>
    public void ReorderRows(IReadOnlyList<int> sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        if (sources.Count != Batch)
        {
            throw new ArgumentException($"One source row per row ({Batch}), got {sources.Count}.", nameof(sources));
        }

        if (InStep)
        {
            throw new InvalidOperationException("Reorder the rows between steps.");
        }

        bool identity = true;
        for (int b = 0; b < Batch; b++)
        {
            if (sources[b] < 0 || sources[b] >= Batch)
            {
                throw new ArgumentException($"Row {b} takes row {sources[b]}, outside the {Batch} rows.", nameof(sources));
            }

            identity &= sources[b] == b;
        }

        if (identity)
        {
            return;
        }

        foreach (var cache in _caches.Values)
        {
            int rows = cache.Keys.Shape[0];
            if (rows % Batch != 0)
            {
                throw new InvalidOperationException($"A cache of {rows} rows does not split over {Batch} sequences.");
            }

            // Row r of the cache belongs to sequence r / perRow: it takes the same head of the source sequence.
            int perRow = rows / Batch;
            var index = new float[rows];
            for (int r = 0; r < rows; r++)
            {
                index[r] = sources[r / perRow] * perRow + r % perRow;
            }

            using var indices = Tensor.From(index, [rows], Device);
            Gather(cache.Keys, indices);
            Gather(cache.Values, indices);
            Gather(cache.KeyScales, indices);
            Gather(cache.ValueScales, indices);
        }

        // The rows gathered whole, whatever each element packs (a copy: the gather reads every row before the write).
        static void Gather(Tensor? rows, Tensor indices)
        {
            if (rows is null)
            {
                return;
            }

            int n = rows.Shape[0];
            using var table = rows.Reshape(n, rows.Size / n);
            using var gathered = table.EmbeddingLookup(indices);
            rows.CopyFrom(gathered);
        }
    }

    /// <summary>
    /// The key/value cache of <paramref name="layer"/> (an attention layer this context has decoded through), or null
    /// before its first step: e.g. <c>context[block.Attention].Bytes</c> for one layer's share of <see cref="CacheBytes"/>.
    /// </summary>
    public KeyValueCache? this[Module layer] => _caches.GetValueOrDefault(layer);

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
    /// When positive, an attention layer with a sliding window shorter than the capacity (<see cref="CacheFor(object, int, int, int)"/>
    /// with its window) keeps window - 1 + SlidingWindowSteps positions instead of <see cref="Capacity"/>: before a step
    /// that would not fit, the window's last window - 1 positions move to the front of its cache (a copy of those rows,
    /// once every SlidingWindowSteps positions or so), so every key a query can see is there, in order. Memory in
    /// proportion to the window instead of the context, and the attention reads the same keys in the same order. Each
    /// step then feeds at most SlidingWindowSteps positions (<see cref="BeginStep"/> refuses more). Only formats that read
    /// a cache's own position (an <see cref="ISlidingWindowLayout"/>), never with <see cref="RowStarts"/>, and
    /// <see cref="Truncate"/> only before the first move. Default 0: every cache holds <see cref="Capacity"/> positions.
    /// Set before the first step.
    /// </summary>
    /// <exception cref="InvalidOperationException">Caches were made already.</exception>
    public int SlidingWindowSteps
    {
        get => _slidingSteps;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            if (_caches.Count > 0 && value != _slidingSteps)
            {
                throw new InvalidOperationException("SlidingWindowSteps is set before the first step (the caches are made with it).");
            }

            _slidingSteps = value;
        }
    }

    /// <summary>
    /// The slot the next position goes to in <paramref name="cache"/>, as a one-element device tensor: <see cref="Position"/>
    /// for a cache of the context's capacity, its window's own for a sliding one (<see cref="SlidingWindowSteps"/>). What
    /// a layout writes and attends from.
    /// </summary>
    public Tensor PositionOf(KeyValueCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        return cache.Sliding?.Position ?? Position;
    }

    /// <summary>The slots of <paramref name="cache"/> holding positions so far, as tracked on the host (<see cref="Length"/> for a cache of the context's capacity).</summary>
    public int LengthOf(KeyValueCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        return Length - (cache.Sliding?.Base ?? 0);
    }

    /// <summary>The current step's causal mask over <paramref name="cache"/>'s slots, [newSteps, its capacity] (<see cref="Mask"/> for a cache of the context's capacity), or null outside a step.</summary>
    public Tensor? MaskOf(KeyValueCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        return cache.Sliding is not { } window ? Mask
            : InStep ? window.Mask ??= Tensor.DecoderMask(window.Position, _steps, window.Capacity, track: false) : null;
    }

    // Before `steps` new positions: each sliding window whose cache would overflow keeps its last window - 1 positions,
    // moved to its front (outside a recording: a recorded step is replayed after the move, see ReplayStep).
    private void Slide(int steps)
    {
        foreach (var window in _windows)
        {
            if (steps > _slidingSteps)
            {
                throw new InvalidOperationException($"A step of {steps} positions; the sliding windows take at most {_slidingSteps} (SlidingWindowSteps).");
            }

            if (Length - window.Base + steps <= window.Capacity)
            {
                continue;
            }

            if (ComputeGraph.IsCapturing)
            {
                throw new InvalidOperationException("A sliding window must move before the step is recorded (CaptureStep moves it).");
            }

            int keep = Math.Min(window.Length - 1, Length);            // the positions a query from Length on still sees
            foreach (var cache in window.Caches)
            {
                cache.MoveToFront(Length - keep - window.Base, keep);
            }

            window.Base = Length - keep;
            window.Position.FillInPlace(keep);
        }
    }

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
    /// When set, <c>Sequential.ForwardCached</c> returns only the last new position's outputs: the layers after the
    /// last cached (attention) layer — final norm and output head — run on one position instead of every prompt position
    /// (sampling reads only the last one; a long prompt's full logits would be prompt × vocabulary floats).
    /// </summary>
    public bool LastPositionOnly { get; set; }

    /// <summary>
    /// Stops a long pass early: a cached forward pass (<c>Sequential.ForwardCached</c>) checks it before each layer, so a
    /// prompt's prefill (an image's hundreds of tokens) ends at the next layer rather than at the next generated token.
    /// The step is then withdrawn (<see cref="CancelStep"/>) and <see cref="OperationCanceledException"/> thrown.
    /// </summary>
    public CancellationToken Cancellation { get; set; }

    /// <summary>The current step's position indices, [newSteps] (set by <see cref="BeginStep"/>).</summary>
    public Tensor? Positions { get; private set; }

    /// <summary>
    /// Rows of different lengths decoded together: row b's sequence starts at position RowStarts[b] (its prompt padded
    /// on the left so every row's prompt ends at the same position). Its tokens are numbered from there and see no earlier
    /// position. Null (the default): every row starts at 0. Needs a <see cref="KeyValueFormat.Float32"/> cache for decoder models.
    /// </summary>
    public IReadOnlyList<int>? RowStarts { get; private set; }

    /// <summary>Sets <see cref="RowStarts"/> (one start per row; null: all rows start at 0).</summary>
    public void SetRowStarts(IReadOnlyList<int>? starts)
    {
        if (starts is not null && (starts.Count != Batch || starts.Any(s => s < 0 || s >= Capacity)))
        {
            throw new ArgumentException($"Row starts: one per row ({Batch}), each within the capacity ({Capacity}).");
        }

        if (starts is not null && _windows.Count > 0)
        {
            throw new InvalidOperationException("Rows of different lengths do not attend from sliding windows' caches (SlidingWindowSteps): make the context without them.");
        }

        if (InStep)
        {
            throw new InvalidOperationException("Set the row starts between steps.");
        }

        RowStarts = starts?.ToArray();
        _rowStarts?.Dispose();
        _rowPositions?.Dispose();
        (_rowStarts, _rowPositions) = (null, null);
        if (starts is not null)
        {
            // Fixed for the generation: on the device once, read by every one-token step (recorded or not).
            var values = new float[Batch];
            for (int b = 0; b < Batch; b++)
            {
                values[b] = starts[b];
            }

            _rowStarts = Tensor.Persistent(values, [Batch], Device, requiresGrad: false);
            _rowPositions = Tensor.PersistentZeros([Batch], Device, requiresGrad: false);
        }
    }

    /// <summary>
    /// With <see cref="RowStarts"/>: the current step's position of each new token within its row's sequence, [batch · newSteps]
    /// (padding: 0). A one-token step computes them on the device from <see cref="Position"/>, so it can be recorded.
    /// </summary>
    public Tensor? TokenPositions { get; private set; }

    /// <summary>With <see cref="RowStarts"/>: each new token's row start, [batch · newSteps] (a one-token step: the starts uploaded once by <see cref="SetRowStarts"/>).</summary>
    public Tensor? TokenStarts { get; private set; }

    /// <summary>
    /// For a decoder with several rotary axes (<see cref="RopeSettings.Sections"/>) after a prompt whose image tokens were
    /// placed by an image position rule (<see cref="ImagePositionRules"/>): each row's rotary position minus its position
    /// in its sequence (Qwen2-VL's <c>rope_deltas</c>, negative when images take fewer positions than tokens), which the
    /// tokens after it rotate by. Null (the default): rotary positions are the sequence's positions.
    /// </summary>
    public IReadOnlyList<int>? RotaryOffsets => _rotary;

    /// <summary>
    /// With <see cref="RotaryOffsets"/>: the current step's rotary positions, the indices of the rotary tables, one per step
    /// ([newSteps], every row alike) or, when the rows' offsets differ or they have <see cref="RowStarts"/>, one per token
    /// ([batch · newSteps]); null otherwise (the attention layers rotate by <see cref="Positions"/>).
    /// </summary>
    public Tensor? RotaryPositions { get; private set; }

    /// <summary>
    /// Sets <see cref="RotaryOffsets"/> (one per row; null: none) after a pass that placed images: the offsets hold for the
    /// tokens from <paramref name="from"/> (the pass's first image token, counted like <see cref="Length"/>) on, and the
    /// ones before keep those in effect before. <see cref="Truncate"/> to <paramref name="from"/> or earlier brings those
    /// back; truncating between <paramref name="from"/> and <see cref="Length"/> is refused (the tokens there were placed
    /// by the pass). Not while a step is open.
    /// </summary>
    /// <exception cref="ArgumentException">Not one offset per row, or <paramref name="from"/> outside [0, <see cref="Length"/>].</exception>
    public void SetRotaryOffsets(IReadOnlyList<int>? offsets, int from)
    {
        if (offsets is not null && offsets.Count != Batch)
        {
            throw new ArgumentException($"Rotary offsets: one per row ({Batch}), got {offsets.Count}.", nameof(offsets));
        }

        if (from < 0 || from > Length)
        {
            throw new ArgumentException($"The offsets hold from position {from}, which is outside the {Length} positions processed.", nameof(from));
        }

        if (InStep)
        {
            throw new InvalidOperationException("Set the rotary offsets between steps.");
        }

        _rotaryPasses.Add((from, Length, _rotary));
        ApplyRotary(offsets is null ? null : [.. offsets]);
    }

    // The offsets in effect from now on, and the device copy of a uniform one (written in place, so a recorded step adds
    // the value set last when replayed).
    private void ApplyRotary(int[]? offsets)
    {
        _rotary = offsets;
        if (offsets is not null && offsets.All(o => o == offsets[0]))
        {
            _rotaryOffset ??= Tensor.Persistent([0f], [1], Device, requiresGrad: false);
            _rotaryOffset.FillInPlace(offsets[0]);
        }
    }

    /// <summary>
    /// Whether a step can be recorded (<see cref="CaptureStep"/>): the cache format is recordable, and the rows share one
    /// rotary offset (or have none). Rows with <see cref="RowStarts"/> are recordable too (a one-token step reads its
    /// positions and starts on the device), except with rotary offsets: their per-token rotary positions are computed on
    /// the host each step.
    /// </summary>
    public bool Recordable => _recordableLayout && (_rotary is null || RowStarts is null && _rotary.All(o => o == _rotary[0]));

    /// <summary>
    /// Opens a step of <paramref name="steps"/> new positions, or joins the step already open (a module inside a network
    /// that opened it): the scope a cached pass runs in (see <c>ICachedModule</c>). The scope that opened the step counts
    /// its positions when it is completed (<see cref="IDecodingStep.Complete"/>) and withdraws the step when it is disposed
    /// without that (a pass stopped part-way); one that joined does neither. So any module may open its step and only the
    /// outermost one brackets it. The scope is the context's own (one step is open at a time): nothing is allocated per step.
    /// </summary>
    public IDecodingStep OpenStep(int steps)
    {
        if (InStep)
        {
            return _joinedStep ??= new StepScope(this, owner: false);
        }

        BeginStep(steps);
        var scope = _ownStep ??= new StepScope(this, owner: true);
        scope.Open(steps);
        return scope;
    }

    // The scopes OpenStep hands out: the step's opener (reused for every step) and a joiner.
    private StepScope? _ownStep, _joinedStep;

    private sealed class StepScope(DecodingContext context, bool owner) : IDecodingStep
    {
        private int _steps;
        private bool _open;

        public int Steps => owner ? _steps : context._steps;

        public bool Opened => owner;

        public void Open(int steps) => (_steps, _open) = (steps, true);

        public void Complete()
        {
            if (owner && _open)
            {
                _open = false;
                context.EndStep(_steps);
            }
        }

        public void Dispose()
        {
            if (owner && _open)
            {
                _open = false;
                context.CancelStep();                                    // the stopped step's positions are not counted
            }
        }
    }

    /// <summary>
    /// Prepares mask and positions for <paramref name="steps"/> new positions (all computed on the device). The primitive
    /// under <see cref="OpenStep"/>: a step begun here is ended with <see cref="EndStep"/> or <see cref="CancelStep"/>.
    /// </summary>
    public void BeginStep(int steps)
    {
        if (Length + steps > Capacity)
        {
            throw new InvalidOperationException($"Decoding past the context capacity ({Capacity}); call Reset and re-feed a shorter window.");
        }

        Slide(steps);
        _steps = steps;
        InStep = true;
        if (RowStarts is { } && steps == 1)
        {
            // One token a row (a decoding step): its position max(0, Position - start) on the device from the counter and
            // the starts uploaded once, so the step records and replays at every position as the cache fills.
            var backend = _rowPositions!.Backend;
            backend.Affine(_rowStarts!.Storage, _rowPositions.Storage, Batch, -1f, 0f);
            backend.AddBroadcastScalar(Position.Storage, _rowPositions.Storage, Batch, 1f);
            backend.Clamp(_rowPositions.Storage, _rowPositions.Storage, Batch, 0f, Capacity);      // padding: position 0 (it attends to nothing)
            (TokenPositions, TokenStarts, _tokensOwned) = (_rowPositions, _rowStarts, false);
        }
        else if (RowStarts is { } starts)
        {
            // Several positions (a prompt's pass, run as it is): uploaded from the host.
            if (ComputeGraph.IsCapturing)
            {
                throw new InvalidOperationException("A step of several positions over rows with starts uploads its positions from the host and cannot be recorded; record one-token steps.");
            }

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
            _tokensOwned = true;
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

        if (_rotary is { } offsets)
        {
            RotaryPositions = RotaryStep(offsets, steps);
        }
    }

    // The step's rotary positions: on the device from Position for a uniform offset (the same graph valid at every
    // position, adding the offset set last), else per token on the host (rows with starts, or different offsets).
    private Tensor RotaryStep(int[] offsets, int steps)
    {
        if (RowStarts is null && offsets.All(o => o == offsets[0]))
        {
            _rotaryOwned = false;
            var rotary = Positions!.Narrow(0, 0, steps);
            rotary.Backend.AddBroadcastScalar(_rotaryOffset!.Storage, rotary.Storage, steps, 1f);
            return rotary;
        }

        var values = new float[Batch * steps];
        for (int b = 0; b < Batch; b++)
        {
            for (int t = 0; t < steps; t++)
            {
                int position = Length + t - (RowStarts?[b] ?? 0);
                values[b * steps + t] = position < 0 ? 0 : position + offsets[b];     // padding: position 0 (it attends to nothing)
            }
        }

        _rotaryOwned = true;
        return Tensor.Persistent(values, [Batch * steps], Device, requiresGrad: false);
    }

    /// <summary>
    /// Withdraws the open step without counting its positions (a pass stopped part-way): <see cref="Length"/> stays where
    /// it was, and the next steps overwrite whatever the stopped one wrote to the caches.
    /// </summary>
    public void CancelStep() => ClearStep();

    /// <summary>Marks <paramref name="steps"/> positions as processed (device counter and host mirror).</summary>
    public void EndStep(int steps)
    {
        Position.AddInPlace(steps);
        foreach (var window in _windows)
        {
            window.Position.AddInPlace(steps);
        }

        Length += steps;
        ClearStep();
    }

    /// <summary>
    /// Records one decoding step (which must call <see cref="BeginStep"/>/<see cref="EndStep"/>, e.g. via
    /// <c>Sequential.ForwardCached</c>) as a <see cref="ComputeGraph"/>. Because positions, masks and cache
    /// offsets are computed on the device from <see cref="Position"/>, the same graph is valid at every position; with
    /// <see cref="RowStarts"/>, a one-token step's positions are too (from the starts uploaded once). Needs a
    /// recordable cache format (<see cref="IRecordableLayout"/>).
    /// </summary>
    public ComputeGraph CaptureStep(Action step)
    {
        if (!_recordableLayout)
        {
            throw new InvalidOperationException($"The key/value cache format '{Layout.Name}' is not recordable (IRecordableLayout); run its steps as they are.");
        }

        if (!Recordable)
        {
            throw new InvalidOperationException("Rows with different rotary offsets (images placed differently), or rotary offsets with row starts, upload their rotary positions from the host and cannot be recorded.");
        }

        Slide(1);                                     // a sliding window moves now: the recording cannot
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
        foreach (var window in _windows)
        {
            window.Mask?.Dispose();
            window.Mask = null;
        }

        if (_tokensOwned)
        {
            TokenPositions?.Dispose();
            TokenStarts?.Dispose();
            _tokensOwned = false;
        }

        TokenPositions = null;
        TokenStarts = null;
        if (_rotaryOwned)
        {
            RotaryPositions?.Dispose();
            _rotaryOwned = false;
        }

        RotaryPositions = null;
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

        Slide(steps);
        int length = Length;
        graph.Replay();
        Length = length + steps;
    }

    /// <summary>Forgets all positions (and the <see cref="RotaryOffsets"/>); caches are overwritten from position 0.</summary>
    public void Reset()
    {
        Position.FillInPlace(0f);
        _rotaryPasses.Clear();
        ApplyRotary(null);
        foreach (var window in _windows)
        {
            window.Base = 0;
            window.Position.FillInPlace(0f);
        }

        Length = 0;
    }

    /// <summary>
    /// Keeps the first <paramref name="length"/> positions and forgets the rest (their cache entries are overwritten by
    /// the next steps): reuses the cached keys and values of a shared prompt prefix.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// A sliding window's cache has moved (its first positions are gone), or <paramref name="length"/> falls inside the
    /// images of a pass that set <see cref="RotaryOffsets"/> (<see cref="SetRotaryOffsets"/>).
    /// </exception>
    public void Truncate(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Length);
        if (_windows.Any(w => w.Base > 0))
        {
            throw new InvalidOperationException("A sliding window's cache has moved past the first positions (SlidingWindowSteps); Reset and feed the prompt again.");
        }

        // The rotary offsets in effect at `length`: those set before every pass whose images start at or after it.
        while (_rotaryPasses.Count > 0 && _rotaryPasses[^1].End > length)
        {
            var pass = _rotaryPasses[^1];
            if (length > pass.From)
            {
                throw new InvalidOperationException($"Truncating to {length} falls inside a pass whose images (from position {pass.From} to {pass.End}) set the rotary "
                    + "offsets; truncate to before its first image and feed the rest again.");
            }

            _rotaryPasses.RemoveAt(_rotaryPasses.Count - 1);
            ApplyRotary(pass.Before);
        }

        Position.FillInPlace(length);
        foreach (var window in _windows)
        {
            window.Position.FillInPlace(length);
        }

        Length = length;
    }

    /// <summary>The cache for <paramref name="owner"/> (one per attention layer), created on first use.</summary>
    public KeyValueCache CacheFor(object owner, int rows, int headDim)
    {
        if (!_caches.TryGetValue(owner, out var cache))
        {
            long start = Telemetry.StartStage();
            using (Device.Backend.LongLivedAllocations())                  // a read's caches live as long as it
            {
                _caches[owner] = cache = new KeyValueCache(rows, Capacity, headDim, Device, Layout);
            }

            Telemetry.InferenceStage(InferenceStageKind.CacheAllocation, Device, start, Capacity, cache.Bytes);
        }

        return cache;
    }

    /// <summary>
    /// The cache for <paramref name="owner"/>, an attention layer whose queries see the last <paramref name="window"/>
    /// positions (0: every earlier one), created on first use: a sliding window's cache of window - 1 +
    /// <see cref="SlidingWindowSteps"/> positions when that is set and smaller than <see cref="Capacity"/> (and the format
    /// reads a cache's own position, rows start together, and nothing is decoded yet), else one of the context's capacity.
    /// The layer then writes and attends from <see cref="PositionOf"/>, <see cref="LengthOf"/> and <see cref="MaskOf"/>.
    /// </summary>
    public KeyValueCache CacheFor(object owner, int rows, int headDim, int window)
    {
        if (_caches.TryGetValue(owner, out var cache))
        {
            return cache;
        }

        int slots = window - 1 + _slidingSteps;
        if (_slidingSteps <= 0 || window <= 0 || slots >= Capacity || Length > 0 || RowStarts is not null || !_slidingLayout)
        {
            return CacheFor(owner, rows, headDim);
        }

        var group = _windows.Find(w => w.Length == window);
        if (group is null)
        {
            _windows.Add(group = new SlidingWindow(window, slots, Device));
        }

        _caches[owner] = cache = new KeyValueCache(rows, slots, headDim, Device, Layout) { Sliding = group };
        group.Caches.Add(cache);
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
        ClearMemory();
        foreach (var window in _windows)
        {
            window.Dispose();
        }

        _windows.Clear();
        _iota?.Dispose();
        _rotaryOffset?.Dispose();
        _rowStarts?.Dispose();
        _rowPositions?.Dispose();
        Position.Dispose();
    }
}

// The caches of one window length sliding together (DecodingContext.SlidingWindowSteps): their capacity, the position
// slot 0 holds (host), the slot the next position goes to (device, advanced with the context's position), and the
// current step's mask over their slots.
internal sealed class SlidingWindow(int length, int capacity, Device device) : IDisposable
{
    public int Length { get; } = length;

    public int Capacity { get; } = capacity;

    public int Base { get; set; }

    public Tensor Position { get; } = Tensor.Persistent([0f], [1], device, requiresGrad: false);

    public Tensor? Mask { get; set; }

    public List<KeyValueCache> Caches { get; } = [];

    public void Dispose()
    {
        Mask?.Dispose();
        Position.Dispose();
    }
}

/// <summary>
/// One decoding step over a <see cref="DecodingContext"/>, opened by <see cref="DecodingContext.OpenStep"/>: the scope a
/// cached pass runs in. Dispose it when the pass ends; complete it first when the pass ran to its end.
/// </summary>
public interface IDecodingStep : IDisposable
{
    /// <summary>The new positions of the step.</summary>
    int Steps { get; }

    /// <summary>Whether this scope opened the step (it counts or withdraws it); false when it joined a step already open.</summary>
    bool Opened { get; }

    /// <summary>The pass ran to its end: the scope that opened the step counts its positions (<see cref="DecodingContext.EndStep"/>).</summary>
    void Complete();
}
