// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// Multi-head attention of a decoder's rows over an encoder's output (the memory): queries from the decoder's
/// [batch, time, dim], keys and values from the memory [batch, memory, memoryDim], every query seeing every memory
/// position of its sequence (no rotary positions: encoder-decoder families use learned or sinusoidal positions); a padded
/// batch of memories gives each sequence's length, and its padding is not attended to. The scores are
/// never stored whole (<see cref="Tensor.AttentionFastest"/>: tiled over the keys, or the composed scores where the device
/// measured them faster); gradients flow to the queries, the memory and the projections for training.
/// <para>
/// Inference computes the memory's projections once per sequence: <see cref="SetMemory"/> projects it and hands the keys
/// and values to the <see cref="DecodingContext"/> (<see cref="DecodingContext.SetMemory"/>), and every cached step
/// (<see cref="ForwardCached"/>) reads them from there. A training pass gives the memory with the input
/// (<see cref="Forward(Tensor, Tensor)"/>), or opens an <see cref="EncoderMemory"/> scope around a pass through blocks that
/// call <see cref="Module.Forward"/> alone.
/// </para>
/// </summary>
public sealed class CrossAttention : Module, ICachedModule
{
    private readonly Linear[] _keyValue;
    private (Tensor Starts, Tensor Ends, int Rows, int Keys, Device Device, int[]? Lengths)? _spans;

    /// <summary>Creates the layer with random projections.</summary>
    /// <param name="dim">The decoder's width (the queries' and the output's).</param>
    /// <param name="heads">Attention heads (dim must be divisible by them).</param>
    /// <param name="memoryDim">The memory's width (the keys' and values' input), or null for <paramref name="dim"/>.</param>
    /// <param name="bias">Biases on the projections.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Initialization.</param>
    public CrossAttention(int dim, int heads, int? memoryDim = null, bool bias = true, Device? device = null, Random? random = null)
        : this(new Linear(dim, dim, bias, device, random), new Linear(memoryDim ?? dim, dim, bias, device, random),
            new Linear(memoryDim ?? dim, dim, bias, device, random), new Linear(dim, dim, bias, device, random), heads)
    {
    }

    /// <summary>Creates the layer from existing projections (for example loaded weights); the layer takes ownership.</summary>
    /// <param name="query">The query projection, [dim, heads · headDim] ("q").</param>
    /// <param name="key">The key projection, [memoryDim, heads · headDim] ("k").</param>
    /// <param name="value">The value projection, [memoryDim, heads · headDim] ("v").</param>
    /// <param name="output">The output projection, [heads · headDim, dim] ("o").</param>
    /// <param name="heads">Attention heads.</param>
    /// <exception cref="ArgumentException">The projections do not fit together or the heads do not divide them.</exception>
    public CrossAttention(Linear query, Linear key, Linear value, Linear output, int heads)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        int inner = query.OutFeatures;
        if (inner % heads != 0 || key.OutFeatures != inner || value.OutFeatures != inner || output.InFeatures != inner || key.InFeatures != value.InFeatures)
        {
            throw new ArgumentException($"CrossAttention takes q [dim, inner], k and v [memoryDim, inner] and o [inner, dim] with inner divisible by {heads} heads; "
                + $"got q [{query.InFeatures}, {query.OutFeatures}], k [{key.InFeatures}, {key.OutFeatures}], v [{value.InFeatures}, {value.OutFeatures}] and o [{output.InFeatures}, {output.OutFeatures}].");
        }

        (Query, Key, Value, Output) = (query, key, value, output);
        query.Name = "q";
        key.Name = "k";
        value.Name = "v";
        output.Name = "o";
        _keyValue = [key, value];
        Heads = heads;
        HeadDim = inner / heads;
        ScoreScale = 1f / MathF.Sqrt(HeadDim);
    }

    /// <summary>Attention heads.</summary>
    public int Heads { get; }

    /// <summary>Size of each head.</summary>
    public int HeadDim { get; }

    /// <summary>The decoder's width (the queries' input and the output).</summary>
    public int Dim => Query.InFeatures;

    /// <summary>The memory's width (the keys' and values' input).</summary>
    public int MemoryDim => Key.InFeatures;

    /// <summary>Query projection ("q").</summary>
    public Linear Query { get; }

    /// <summary>Key projection ("k").</summary>
    public Linear Key { get; }

    /// <summary>Value projection ("v").</summary>
    public Linear Value { get; }

    /// <summary>Output projection ("o").</summary>
    public Linear Output { get; }

    /// <summary>The factor on each query·key product before the softmax (1 / √<see cref="HeadDim"/> unless set).</summary>
    public float ScoreScale { get; init; }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [Query, Key, Value, Output];

    /// <summary>
    /// The memory [batch, memory, memoryDim] projected to this layer's keys and values, [batch · heads, memory, headDim]
    /// each (head h of sequence b at row b · heads + h).
    /// </summary>
    /// <exception cref="ArgumentException">The memory is not [batch, memory, <see cref="MemoryDim"/>].</exception>
    public (Tensor Keys, Tensor Values) Project(Tensor memory)
    {
        ArgumentNullException.ThrowIfNull(memory);
        if (memory.Rank != 3 || memory.Shape[2] != MemoryDim)
        {
            throw new ArgumentException($"CrossAttention's memory is [batch, memory, {MemoryDim}], got {Tensor.FormatShape(memory.Shape)}.");
        }

        int n = memory.Shape[0], m = memory.Shape[1];
        var projected = Linear.ForwardMany(memory, _keyValue);
        return (Heads4(projected[0], n, m), Heads4(projected[1], n, m));
    }

    /// <summary>
    /// Projects <paramref name="memory"/> once and hands the keys and values to <paramref name="context"/>
    /// (<see cref="DecodingContext.SetMemory"/>) for every step of the sequence. A memory of one sequence serves a context
    /// of several rows (a beam search's beams): its projections are repeated for each row.
    /// </summary>
    /// <param name="memory">The encoder's output, [1 or batch, memory, memoryDim].</param>
    /// <param name="context">The decoding context (its batch is the rows decoded together).</param>
    /// <exception cref="ArgumentException">The memory's batch is neither 1 nor the context's.</exception>
    public void SetMemory(Tensor memory, DecodingContext context)
    {
        ArgumentNullException.ThrowIfNull(memory);
        ArgumentNullException.ThrowIfNull(context);
        int n = memory.Shape.Length == 3 ? memory.Shape[0] : -1;
        if (n != 1 && n != context.Batch)
        {
            throw new ArgumentException($"The memory holds {n} sequences for a context of {context.Batch} rows (one sequence serves every row).");
        }

        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var (keys, values) = Project(memory);
        if (n == 1 && context.Batch > 1)
        {
            var copies = new Tensor[context.Batch];
            Array.Fill(copies, keys);
            keys = Tensor.Concat(copies);
            Array.Fill(copies, values);
            values = Tensor.Concat(copies);
        }

        context.SetMemory(this, TensorScope.Untrack(keys), TensorScope.Untrack(values));
    }

    /// <summary>Attention of <paramref name="input"/> [batch, time, dim] over <paramref name="memory"/> [batch, memory, memoryDim] → [batch, time, dim].</summary>
    public Tensor Forward(Tensor input, Tensor memory) => Forward(input, memory, null);

    /// <summary>
    /// Attention of <paramref name="input"/> [batch, time, dim] over a padded batch of memories [batch, memory, memoryDim]
    /// → [batch, time, dim]: sequence b attends to its first <paramref name="lengths"/>[b] memory positions only (all when
    /// null), so a sequence gives the same output in any batch and its padding gets no gradient.
    /// </summary>
    /// <exception cref="ArgumentException">There is not one length per sequence, or one is outside 1 to the memory's positions.</exception>
    public Tensor Forward(Tensor input, Tensor memory, IReadOnlyList<int>? lengths)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(memory);
        var own = Lengths(lengths, memory);
        var (keys, values) = Project(memory);
        return Attend(input, keys, values, own);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) =>
        Forward(input, EncoderMemory.Current ?? throw new InvalidOperationException(
            "CrossAttention needs the encoder's output: call Forward(input, memory), open an EncoderMemory scope around the pass, or decode with a context given the memory (SetMemory)."),
            EncoderMemory.CurrentLengths);

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var memory = context.MemoryOf(this) ?? throw new InvalidOperationException("The decoding context holds no memory for this CrossAttention: call SetMemory(memory, context) before the first step.");
        return Attend(input, memory.Keys, memory.Values, null);
    }

    // The lengths checked against the memory [n, m, memoryDim]: null when every sequence has all m positions.
    private static int[]? Lengths(IReadOnlyList<int>? lengths, Tensor memory)
    {
        if (lengths is null)
        {
            return null;
        }

        int n = memory.Rank == 3 ? memory.Shape[0] : -1, m = memory.Rank == 3 ? memory.Shape[1] : -1;
        if (lengths.Count != n)
        {
            throw new ArgumentException($"{lengths.Count} memory lengths for a memory of {n} sequences ({Tensor.FormatShape(memory.Shape)}).", nameof(lengths));
        }

        bool full = true;
        for (int b = 0; b < n; b++)
        {
            if (lengths[b] < 1 || lengths[b] > m)
            {
                throw new ArgumentException($"Sequence {b}'s memory length {lengths[b]} is outside 1 to {m}, the memory's positions.", nameof(lengths));
            }

            full &= lengths[b] == m;
        }

        return full ? null : [.. lengths];
    }

    // q from the input, [n·heads, t, d]; attention over keys and values [n·heads, m, d], each query seeing its sequence's
    // keys (all of them, or its first lengths[b]).
    private Tensor Attend(Tensor input, Tensor keys, Tensor values, int[]? lengths)
    {
        if (input.Rank != 3 || input.Shape[2] != Dim)
        {
            throw new ArgumentException($"CrossAttention expects [batch, time, {Dim}], got {Tensor.FormatShape(input.Shape)}.");
        }

        int n = input.Shape[0], t = input.Shape[1], m = keys.Shape[1];
        if (keys.Shape[0] != n * Heads)
        {
            throw new ArgumentException($"The memory's {keys.Shape[0]} rows do not fit {n} sequences of {Heads} heads.");
        }

        var q = Heads4(Query.Forward(input), n, t);
        var (starts, ends) = Spans(t, m, input.Device, lengths);
        var context = Tensor.AttentionFastest(q, keys, values, starts, ends, ScoreScale, everyKey: lengths is null);   // [n·heads, t, d]
        var merged = context.Reshape(n, Heads, t, HeadDim).Permute(0, 2, 1, 3).Reshape(n, t, Heads * HeadDim);
        return Output.Forward(merged);
    }

    // [n, rows, heads·d] → [n·heads, rows, d].
    private Tensor Heads4(Tensor projected, int n, int rows) =>
        projected.Reshape(n, rows, Heads, HeadDim).Permute(0, 2, 1, 3).Reshape(n * Heads, rows, HeadDim);

    // Every query row's range of keys: all of them (one table), or sequence b's first lengths[b] (a table per sequence, its
    // heads' rows; AttentionFastest gives a table each run of heads). Cached per shape, device and lengths: a training
    // pass's backward, and a checkpointed block's second forward, read the same tables.
    private (Tensor Starts, Tensor Ends) Spans(int rows, int keys, Device device, int[]? lengths)
    {
        if (_spans is { } cached && cached.Rows == rows && cached.Keys == keys && cached.Device == device
            && (cached.Lengths is null ? lengths is null : lengths is not null && cached.Lengths.AsSpan().SequenceEqual(lengths)))
        {
            return (cached.Starts, cached.Ends);
        }

        _spans?.Starts.Dispose();
        _spans?.Ends.Dispose();
        int tables = lengths?.Length ?? 1;
        var ends = new float[tables * rows];
        for (int b = 0; b < tables; b++)
        {
            ends.AsSpan(b * rows, rows).Fill(lengths?[b] ?? keys);
        }

        var starts = CreateBuffer(new float[tables * rows], [tables * rows], device);
        var endsBuffer = CreateBuffer(ends, [tables * rows], device);
        _spans = (starts, endsBuffer, rows, keys, device, lengths);
        return (starts, endsBuffer);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _spans?.Starts.Dispose();
        _spans?.Ends.Dispose();
        _spans = null;
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"CrossAttention({Heads} heads, head size {HeadDim}{(MemoryDim != Dim ? $", memory {MemoryDim}" : "")})";
}

/// <summary>
/// The encoder's output for a pass through decoder blocks called with <see cref="Module.Forward"/> alone (a training
/// step through a <see cref="Sequential"/>): every <see cref="CrossAttention"/> on this thread attends over it until the
/// scope is disposed. Decoding with a <see cref="DecodingContext"/> uses <see cref="CrossAttention.SetMemory"/> instead.
/// </summary>
/// <example>
/// <code>
/// using (EncoderMemory.Use(encoded))
/// {
///     var logits = decoder.Forward(ids);
/// }
/// </code>
/// </example>
public sealed class EncoderMemory : IDisposable
{
    [ThreadStatic]
    private static EncoderMemory? t_current;

    private readonly EncoderMemory? _previous;

    private EncoderMemory(Tensor memory, IReadOnlyList<int>? lengths)
    {
        Memory = memory;
        Lengths = lengths;
        _previous = t_current;
        t_current = this;
    }

    /// <summary>The memory the current scope holds, [batch, memory, memoryDim], or null outside one.</summary>
    public static Tensor? Current => t_current?.Memory;

    /// <summary>The current scope's memory lengths (each sequence's own positions), or null: every position of every sequence.</summary>
    public static IReadOnlyList<int>? CurrentLengths => t_current?.Lengths;

    /// <summary>The scope's memory.</summary>
    public Tensor Memory { get; }

    /// <summary>Each sequence's memory positions (the first ones; padding after them), or null for all of them.</summary>
    public IReadOnlyList<int>? Lengths { get; }

    /// <summary>Opens a scope over <paramref name="memory"/> (not owned: the caller keeps and disposes it).</summary>
    public static EncoderMemory Use(Tensor memory) => Use(memory, null);

    /// <summary>
    /// Opens a scope over a padded batch of memories (not owned): sequence b's first <paramref name="lengths"/>[b]
    /// positions are its own, the rest padding no <see cref="CrossAttention"/> attends to (all of them when null).
    /// </summary>
    public static EncoderMemory Use(Tensor memory, IReadOnlyList<int>? lengths)
    {
        ArgumentNullException.ThrowIfNull(memory);
        return new EncoderMemory(memory, lengths is null ? null : [.. lengths]);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (ReferenceEquals(t_current, this))
        {
            t_current = _previous;
        }
    }
}
