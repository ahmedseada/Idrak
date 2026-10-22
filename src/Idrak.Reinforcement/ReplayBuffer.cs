// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.InteropServices;

namespace Idrak.Reinforcement;

/// <summary>
/// An experience replay memory (Lin 1992; Mnih et al. 2015): the last <see cref="Capacity"/> transitions (observation,
/// action, reward, next observation, whether the episode terminated) of float observations and discrete actions, in a ring
/// that overwrites the oldest once full, sampled uniformly from a seed.
/// <list type="bullet">
/// <item><b>Memory that does not grow:</b> every array is rented from <see cref="ArrayPool{T}"/> once, at its full size, when
/// the buffer is made (observations and next observations as two flat float arrays, struct of arrays, rule 37), and
/// returned by <see cref="Dispose"/>; adding and sampling allocate nothing. <see cref="Bytes"/> is what it holds.</item>
/// <item><b>Reproducible:</b> <see cref="Sample"/> draws its indices from the <see cref="Random"/> it is given, in order (with
/// replacement, as DQN samples), so a seeded draw gives the same mini-batch on every machine; <see cref="Save"/> and
/// <see cref="Load"/> keep the contents bit for bit (little-endian), so a resumed run samples what it would have.</item>
/// </list>
/// Not thread-safe: one trainer fills and samples it.
/// </summary>
public sealed class ReplayBuffer : IDisposable
{
    private const string Magic = "idrak-replay/1";

    private float[] _observations, _next, _rewards;
    private int[] _actions;
    private bool[] _terminated;
    private bool _disposed;

    /// <summary>A buffer of <paramref name="capacity"/> transitions of observations of <paramref name="observationSize"/> floats.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A capacity or size that is not positive, or a buffer larger than an array holds.</exception>
    public ReplayBuffer(int capacity, int observationSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(observationSize);
        if ((long)capacity * observationSize > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, string.Create(CultureInfo.InvariantCulture,
                $"{capacity} transitions of {observationSize} floats hold more than an array can; give a smaller capacity."));
        }

        Capacity = capacity;
        ObservationSize = observationSize;
        _observations = ArrayPool<float>.Shared.Rent(capacity * observationSize);
        _next = ArrayPool<float>.Shared.Rent(capacity * observationSize);
        _rewards = ArrayPool<float>.Shared.Rent(capacity);
        _actions = ArrayPool<int>.Shared.Rent(capacity);
        _terminated = ArrayPool<bool>.Shared.Rent(capacity);
    }

    /// <summary>The most transitions it keeps.</summary>
    public int Capacity { get; }

    /// <summary>The floats of one observation.</summary>
    public int ObservationSize { get; }

    /// <summary>The transitions it holds (up to <see cref="Capacity"/>).</summary>
    public int Count { get; private set; }

    /// <summary>Where the next transition is written (the oldest one once full).</summary>
    public int Position { get; private set; }

    /// <summary>The transitions added since it was made (or loaded), including those overwritten.</summary>
    public long Added { get; private set; }

    /// <summary>The bytes its arrays hold, fixed from when it was made.</summary>
    public long Bytes => (long)(_observations.Length + _next.Length + _rewards.Length) * sizeof(float) + (long)_actions.Length * sizeof(int) + _terminated.Length;

    /// <summary>Adds a transition, over the oldest one when full.</summary>
    /// <param name="observation">The observation acted on (<see cref="ObservationSize"/> floats).</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward it got, finite.</param>
    /// <param name="next">The observation after it (the last one of an episode that ended).</param>
    /// <param name="terminated">The episode terminated at this step, so nothing is bootstrapped from <paramref name="next"/> (a truncated one is not terminated).</param>
    /// <exception cref="ArgumentException">Observations of another size, or a reward that is not finite.</exception>
    public void Add(ReadOnlySpan<float> observation, int action, float reward, ReadOnlySpan<float> next, bool terminated)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (observation.Length != ObservationSize || next.Length != ObservationSize)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"A transition's observations hold {ObservationSize} floats each, not {observation.Length} and {next.Length}."));
        }

        if (!float.IsFinite(reward))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"A transition's reward is finite, not {reward}."), nameof(reward));
        }

        int at = Position;
        observation.CopyTo(_observations.AsSpan(at * ObservationSize, ObservationSize));
        next.CopyTo(_next.AsSpan(at * ObservationSize, ObservationSize));
        _actions[at] = action;
        _rewards[at] = reward;
        _terminated[at] = terminated;
        Position = at + 1 == Capacity ? 0 : at + 1;
        Count = Math.Min(Count + 1, Capacity);
        Added++;
    }

    /// <summary>Fills <paramref name="indices"/> with transitions drawn uniformly (with replacement) by <paramref name="random"/>, one draw each, in order.</summary>
    /// <exception cref="InvalidOperationException">The buffer is empty.</exception>
    public void Sample(Random random, Span<int> indices)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(random);
        if (Count == 0)
        {
            throw new InvalidOperationException("The replay buffer is empty: add transitions before sampling.");
        }

        for (int i = 0; i < indices.Length; i++)
        {
            indices[i] = random.Next(Count);
        }
    }

    /// <summary>
    /// Copies the transitions <paramref name="indices"/> into the caller's arrays, in that order: observations and next
    /// observations as rows of <see cref="ObservationSize"/> floats, then actions, rewards and terminations.
    /// </summary>
    /// <exception cref="ArgumentException">An array too small for the indices.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An index of no transition held.</exception>
    public void Gather(ReadOnlySpan<int> indices, Span<float> observations, Span<int> actions, Span<float> rewards, Span<float> next, Span<bool> terminated)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int n = indices.Length, size = ObservationSize;
        if (observations.Length < n * size || next.Length < n * size || actions.Length < n || rewards.Length < n || terminated.Length < n)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"Gathering {n} transitions needs {n * size} floats of observations and {n} of the rest."));
        }

        for (int i = 0; i < n; i++)
        {
            int at = indices[i];
            if ((uint)at >= (uint)Count)
            {
                throw new ArgumentOutOfRangeException(nameof(indices), at, string.Create(CultureInfo.InvariantCulture, $"The buffer holds {Count} transitions."));
            }

            _observations.AsSpan(at * size, size).CopyTo(observations.Slice(i * size, size));
            _next.AsSpan(at * size, size).CopyTo(next.Slice(i * size, size));
            actions[i] = _actions[at];
            rewards[i] = _rewards[at];
            terminated[i] = _terminated[at];
        }
    }

    /// <summary>Forgets every transition (the memory stays held).</summary>
    public void Clear()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Count = Position = 0;
        Added = 0;
    }

    /// <summary>
    /// Writes the buffer to <paramref name="stream"/>: a header (format, capacity, observation size, count, position,
    /// added), then the transitions held, little-endian, so <see cref="Load"/> restores it bit for bit.
    /// </summary>
    public void Save(Stream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(stream);
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);   // BinaryWriter is little-endian on every platform
        writer.Write(Magic);
        writer.Write(Capacity);
        writer.Write(ObservationSize);
        writer.Write(Count);
        writer.Write(Position);
        writer.Write(Added);
        writer.Flush();
        Write(stream, MemoryMarshal.AsBytes(_observations.AsSpan(0, Count * ObservationSize)), sizeof(float));
        Write(stream, MemoryMarshal.AsBytes(_next.AsSpan(0, Count * ObservationSize)), sizeof(float));
        Write(stream, MemoryMarshal.AsBytes(_rewards.AsSpan(0, Count)), sizeof(float));
        Write(stream, MemoryMarshal.AsBytes(_actions.AsSpan(0, Count)), sizeof(int));
        Write(stream, MemoryMarshal.AsBytes(_terminated.AsSpan(0, Count)), 1);
    }

    /// <summary>Restores what <see cref="Save"/> wrote into this buffer (of the same capacity and observation size).</summary>
    /// <exception cref="InvalidDataException">The stream holds no replay buffer, or one of another capacity or size.</exception>
    public void Load(Stream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(stream);
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        string magic;
        try
        {
            magic = reader.ReadString();
        }
        catch (Exception e) when (e is EndOfStreamException or FormatException or IOException)
        {
            throw new InvalidDataException($"The stream holds no replay buffer ({Magic}).", e);
        }

        int capacity = reader.ReadInt32(), size = reader.ReadInt32(), count = reader.ReadInt32(), position = reader.ReadInt32();
        long added = reader.ReadInt64();
        if (magic != Magic || capacity != Capacity || size != ObservationSize || (uint)count > (uint)capacity || (uint)position >= (uint)capacity || added < count)
        {
            throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
                $"The stream holds a replay buffer of {capacity} transitions of {size} floats ({magic}), not one of {Capacity} and {ObservationSize} ({Magic})."));
        }

        Read(stream, MemoryMarshal.AsBytes(_observations.AsSpan(0, count * size)), sizeof(float));
        Read(stream, MemoryMarshal.AsBytes(_next.AsSpan(0, count * size)), sizeof(float));
        Read(stream, MemoryMarshal.AsBytes(_rewards.AsSpan(0, count)), sizeof(float));
        Read(stream, MemoryMarshal.AsBytes(_actions.AsSpan(0, count)), sizeof(int));
        Read(stream, MemoryMarshal.AsBytes(_terminated.AsSpan(0, count)), 1);
        (Count, Position, Added) = (count, position, added);
    }

    /// <summary>Returns the arrays to the pool; more than once is harmless.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ArrayPool<float>.Shared.Return(_observations);
        ArrayPool<float>.Shared.Return(_next);
        ArrayPool<float>.Shared.Return(_rewards);
        ArrayPool<int>.Shared.Return(_actions);
        ArrayPool<bool>.Shared.Return(_terminated);
        _observations = _next = _rewards = [];
        _actions = [];
        _terminated = [];
    }

    // Raw values in little-endian order (as they are on a little-endian machine; byte-swapped in a copy elsewhere, rule 48).
    private static void Write(Stream stream, Span<byte> bytes, int width)
    {
        if (BitConverter.IsLittleEndian || width == 1)
        {
            stream.Write(bytes);
            return;
        }

        byte[] copy = ArrayPool<byte>.Shared.Rent(bytes.Length);
        try
        {
            bytes.CopyTo(copy);
            Swap(copy.AsSpan(0, bytes.Length), width);
            stream.Write(copy, 0, bytes.Length);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(copy);
        }
    }

    private static void Read(Stream stream, Span<byte> bytes, int width)
    {
        stream.ReadExactly(bytes);
        if (!BitConverter.IsLittleEndian && width > 1)
        {
            Swap(bytes, width);
        }
    }

    private static void Swap(Span<byte> bytes, int width)
    {
        var words = MemoryMarshal.Cast<byte, int>(bytes);
        for (int i = 0; i < words.Length; i++)
        {
            words[i] = BinaryPrimitives.ReverseEndianness(words[i]);
        }
    }
}
