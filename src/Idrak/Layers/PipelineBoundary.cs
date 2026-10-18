// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Cluster;
using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// The edge between two stages of a pipeline (plan 23, phase C7), as a module of the stage on one side of it. At the end
/// of a stage (<see cref="Sends"/>) its forward pass sends the activation to the next stage and its backward pass receives
/// that activation's gradient back; at the start of a stage it receives the activation and its backward pass sends the
/// gradient. Both are an <see cref="Autograd.Function"/>, so a stage's backward pass reaches the boundary like any other
/// operation, and the micro-batch (<see cref="MicroBatch"/>) keeps the messages of micro-batches in flight apart. Without
/// autograd, the first stage's boundary runs generation: its cached forward sends the hidden state and the decoding
/// context's position down the stages (each keeping its own key/value cache) and returns the last stage's logits. Values
/// travel as float32 through host memory over any <see cref="ITransport"/>, tags 0 and up (the collectives use negative ones).
/// </summary>
public sealed class PipelineBoundary : Module, ICachedModule
{
    // Tags: each kind of message a range of its own; a micro-batch m takes two tags (its header, then its values).
    internal const int ActivationTag = 1 << 20, GradientTag = 2 << 20, InferenceTag = 3 << 20, ReturnTag = (3 << 20) + 8;
    internal const int CommandStop = 0, CommandCached = 1, CommandPlain = 2;
    private const int HeaderBytes = 32, MaxRank = 8;

    private readonly Tensor _token;
    private long _blockedTicks;

    /// <summary>A boundary between this rank's stage and <paramref name="peer"/>'s.</summary>
    /// <param name="transport">The transport joining the stages' ranks.</param>
    /// <param name="peer">The neighbouring stage's rank: the next stage's when <paramref name="sends"/>, else the previous one's.</param>
    /// <param name="sends">True at the end of a stage (its output goes on), false at its start (its input arrives).</param>
    /// <param name="device">This stage's device.</param>
    /// <param name="answerer">The rank of the pipeline's last stage, whose logits a generation's forward waits for (the sending side of the first stage); null elsewhere.</param>
    public PipelineBoundary(ITransport transport, int peer, bool sends, Device device, int? answerer = null)
    {
        ArgumentNullException.ThrowIfNull(transport);
        ArgumentNullException.ThrowIfNull(device);
        if (peer == transport.Rank || (uint)peer >= (uint)transport.WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(peer), peer, $"a boundary joins rank {transport.Rank} to another rank of 0 to {transport.WorldSize - 1}");
        }

        Transport = transport;
        Peer = peer;
        Sends = sends;
        Answerer = answerer;
        Device = device;
        _token = Tensor.Persistent([0f], [1], device, requiresGrad: true);       // what makes autograd record the receive
        Name = sends ? $"send to rank {peer}" : $"receive from rank {peer}";
    }

    /// <summary>The transport joining the stages.</summary>
    public ITransport Transport { get; }

    /// <summary>The neighbouring stage's rank.</summary>
    public int Peer { get; }

    /// <summary>True at the end of a stage, false at its start.</summary>
    public bool Sends { get; }

    /// <summary>The last stage's rank, whose logits a generation's forward waits for; null on a boundary that does not answer generation.</summary>
    public int? Answerer { get; }

    /// <summary>This stage's device.</summary>
    public Device Device { get; }

    /// <summary>The micro-batch the next forward belongs to (set by the stage before each forward).</summary>
    public int MicroBatch { get; set; }

    /// <summary>Bytes sent since <see cref="ResetCounters"/>.</summary>
    public long BytesSent { get; private set; }

    /// <summary>Bytes received since <see cref="ResetCounters"/>.</summary>
    public long BytesReceived { get; private set; }

    /// <summary>Time spent waiting for the neighbour's messages since <see cref="ResetCounters"/>.</summary>
    public TimeSpan Blocked => Stopwatch.GetElapsedTime(0, _blockedTicks);

    /// <summary>Whether this boundary ran a generation's forward (its stages then serve until told to stop).</summary>
    public bool Generated { get; private set; }

    /// <summary>The tensor a receiving boundary's forward is given (any input is ignored: the activation arrives).</summary>
    public Tensor Token => _token;

    /// <summary>Starts the byte and wait counters again.</summary>
    public void ResetCounters() => (BytesSent, BytesReceived, _blockedTicks) = (0, 0, 0);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        int m = MicroBatch;
        if (Autograd.IsEnabled)
        {
            // Training: the activation goes on (or arrives) now; its gradient comes back (or goes back) in the backward pass.
            return Sends
                ? Autograd.Function("pipeline-send", x =>
                {
                    SendTensor(x[0], Peer, ActivationTag + (2 * m));
                    return Tensor.Zeros([1], x[0].Device);                      // the stage's root: its backward pass starts here
                }, (x, _, _) => [ReceiveTensor(Peer, GradientTag + (2 * m), x[0].Device, x[0].Shape)]).Apply(input)
                : Autograd.Function("pipeline-receive", _ => ReceiveTensor(Peer, ActivationTag + (2 * m), Device), (_, _, g) =>
                {
                    SendTensor(g, Peer, GradientTag + (2 * m));
                    return [null];
                }).Apply(_token);
        }

        if (!Sends)
        {
            throw new InvalidOperationException($"{Name}: without autograd a stage's input is served (PipelineStage.Serve), not run as a module.");
        }

        return RoundTrip(input, null);
    }

    /// <summary>
    /// A generation's step through the following stages: the hidden state <paramref name="input"/> and the context's
    /// position go down the stages, each of which keeps its own cache in step with <paramref name="context"/> (reset or cut
    /// back with it), and the last stage's logits come back.
    /// </summary>
    /// <exception cref="NotSupportedException">The context caches keys and values in a format of one's own, or holds a cross-attention memory: neither is mirrored to the other stages.</exception>
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);
        if (!Sends)
        {
            throw new InvalidOperationException($"{Name}: a stage's input in generation is served (PipelineStage.Serve), not run as a module.");
        }

        if (context.Format == KeyValueFormat.Custom || context.MemoryBytes > 0)
        {
            throw new NotSupportedException("A pipeline's later stages keep caches of the built-in formats for text: a cache layout of one's own and a cross-attention memory "
                + "are not mirrored to them; generate with a built-in KeyValueFormat.");
        }

        return RoundTrip(input, context);
    }

    // Generation: the step's header and hidden state to the next stage, the logits back from the last.
    private Tensor RoundTrip(Tensor input, DecodingContext? context)
    {
        int answerer = Answerer ?? throw new InvalidOperationException($"{Name} does not answer generation: only the first stage's boundary does.");
        Generated = true;
        var header = new GenerationHeader(context is null ? CommandPlain : CommandCached, context?.Batch ?? 0, context?.Capacity ?? 0, context?.Length ?? 0,
            context?.LastPositionOnly ?? false, context?.SlidingWindowSteps ?? 0, (int)(context?.Format ?? KeyValueFormat.Float32), context?.RowStarts);
        SendGeneration(header, input, Peer);
        return ReceiveTensor(answerer, ReturnTag, Device);
    }

    // A generation step's header (and the rows' starts, when set) then the hidden state, to `to`.
    internal void SendGeneration(in GenerationHeader header, Tensor? hidden, int to)
    {
        Span<byte> bytes = stackalloc byte[HeaderBytes];
        bytes.Clear();
        BinaryPrimitives.WriteInt32LittleEndian(bytes, header.Command);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[4..], header.Batch);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[8..], header.Capacity);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[12..], header.Length);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[16..], header.LastPositionOnly ? 1 : 0);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[20..], header.SlidingWindowSteps);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[24..], header.Format);
        BinaryPrimitives.WriteInt32LittleEndian(bytes[28..], header.RowStarts?.Count ?? 0);
        Transport.Send(to, bytes, InferenceTag);
        BytesSent += HeaderBytes;
        if (header.RowStarts is { Count: > 0 } starts)
        {
            var values = new byte[starts.Count * sizeof(int)];
            for (int i = 0; i < starts.Count; i++)
            {
                BinaryPrimitives.WriteInt32LittleEndian(values.AsSpan(i * sizeof(int)), starts[i]);
            }

            Transport.Send(to, values, InferenceTag + 2);
            BytesSent += values.Length;
        }

        if (hidden is not null)
        {
            SendTensor(hidden, to, InferenceTag + 4);
        }
    }

    // The next generation step's header (and rows' starts) from `from`; its hidden state follows (ReceiveTensor, InferenceTag + 4).
    internal GenerationHeader ReceiveGeneration(int from)
    {
        Span<byte> bytes = stackalloc byte[HeaderBytes];
        long start = Stopwatch.GetTimestamp();
        int length = Transport.Receive(from, bytes, InferenceTag);
        _blockedTicks += Stopwatch.GetTimestamp() - start;
        BytesReceived += length;
        if (length != HeaderBytes)
        {
            throw new InvalidDataException($"{Name}: a generation header of {length} bytes from rank {from}, not {HeaderBytes}.");
        }

        int rows = BinaryPrimitives.ReadInt32LittleEndian(bytes[28..]);
        int[]? starts = null;
        if (rows > 0)
        {
            var values = new byte[rows * sizeof(int)];
            int got = Transport.Receive(from, values, InferenceTag + 2);
            BytesReceived += got;
            starts = new int[rows];
            for (int i = 0; i < rows; i++)
            {
                starts[i] = BinaryPrimitives.ReadInt32LittleEndian(values.AsSpan(i * sizeof(int)));
            }
        }

        return new GenerationHeader(BinaryPrimitives.ReadInt32LittleEndian(bytes), BinaryPrimitives.ReadInt32LittleEndian(bytes[4..]),
            BinaryPrimitives.ReadInt32LittleEndian(bytes[8..]), BinaryPrimitives.ReadInt32LittleEndian(bytes[12..]), BinaryPrimitives.ReadInt32LittleEndian(bytes[16..]) != 0,
            BinaryPrimitives.ReadInt32LittleEndian(bytes[20..]), BinaryPrimitives.ReadInt32LittleEndian(bytes[24..]), starts);
    }

    /// <summary>Sends <paramref name="tensor"/>'s shape (under <paramref name="tag"/>) then its values (under tag + 1) to <paramref name="to"/>.</summary>
    internal void SendTensor(Tensor tensor, int to, int tag)
    {
        var shape = tensor.Shape;
        if (shape.Length > MaxRank)
        {
            throw new NotSupportedException($"{Name}: a tensor of {shape.Length} dimensions crosses a pipeline boundary; at most {MaxRank}.");
        }

        Span<byte> header = stackalloc byte[sizeof(int) * (MaxRank + 1)];
        BinaryPrimitives.WriteInt32LittleEndian(header, shape.Length);
        for (int i = 0; i < shape.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(header[(sizeof(int) * (i + 1))..], shape[i]);
        }

        int headerLength = sizeof(int) * (shape.Length + 1);
        Transport.Send(to, header[..headerLength], tag);
        int count = tensor.Size;
        var values = ArrayPool<float>.Shared.Rent(Math.Max(1, count));
        try
        {
            tensor.Backend.Download(tensor.Storage, values.AsSpan(0, count));
            var bytes = MemoryMarshal.AsBytes(values.AsSpan(0, count));        // float32 in the machine's order: little-endian on every supported platform
            Transport.Send(to, bytes, tag + 1);
            BytesSent += headerLength + bytes.Length;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }

    /// <summary>Receives a tensor <see cref="SendTensor"/> sent under <paramref name="tag"/>, onto <paramref name="device"/>; when <paramref name="expected"/> is given its shape must match.</summary>
    internal Tensor ReceiveTensor(int from, int tag, Device device, ReadOnlySpan<int> expected = default)
    {
        Span<byte> header = stackalloc byte[sizeof(int) * (MaxRank + 1)];
        long start = Stopwatch.GetTimestamp();
        int length = Transport.Receive(from, header, tag);
        _blockedTicks += Stopwatch.GetTimestamp() - start;
        int rank = length >= sizeof(int) ? BinaryPrimitives.ReadInt32LittleEndian(header) : -1;
        if (rank < 0 || rank > MaxRank || length != sizeof(int) * (rank + 1))
        {
            throw new InvalidDataException($"{Name}: a tensor header of {length} bytes from rank {from} (tag {tag}) does not hold a shape.");
        }

        var shape = new int[rank];
        long count = 1;
        for (int i = 0; i < rank; i++)
        {
            shape[i] = BinaryPrimitives.ReadInt32LittleEndian(header[(sizeof(int) * (i + 1))..]);
            count *= shape[i];
        }

        if (!expected.IsEmpty && !expected.SequenceEqual(shape))
        {
            throw new InvalidDataException($"{Name}: rank {from} sent a tensor of shape {Tensor.FormatShape(shape)} where {Tensor.FormatShape(expected)} was expected (tag {tag}).");
        }

        int bytes = checked((int)count * sizeof(float));
        var buffer = ArrayPool<byte>.Shared.Rent(Math.Max(1, bytes));
        try
        {
            start = Stopwatch.GetTimestamp();
            int got = Transport.Receive(from, buffer.AsSpan(0, bytes), tag + 1);
            _blockedTicks += Stopwatch.GetTimestamp() - start;
            if (got != bytes)
            {
                throw new InvalidDataException($"{Name}: {got} bytes of values from rank {from} for a tensor of shape {Tensor.FormatShape(shape)} ({bytes} bytes).");
            }

            BytesReceived += length + bytes;
            return Tensor.From(MemoryMarshal.Cast<byte, float>(buffer.AsSpan(0, bytes)), shape, device);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _token.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"PipelineBoundary({Name})";
}

// What a generation step tells the following stages: the command, the decoding context's shape and position (each stage
// keeps its own context in step with it), and the rows' starts of a batch of different lengths.
internal readonly record struct GenerationHeader(int Command, int Batch, int Capacity, int Length, bool LastPositionOnly, int SlidingWindowSteps, int Format,
    IReadOnlyList<int>? RowStarts);
