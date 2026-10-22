// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>What kind of values a <see cref="Space"/> holds.</summary>
public enum SpaceKind
{
    /// <summary>One of n choices: the integers 0 to n - 1 (<see cref="Space.Discrete"/>).</summary>
    Discrete,

    /// <summary>An array of floats of a fixed shape, each within its bounds (<see cref="Space.Box(IReadOnlyList{int}, float, float)"/>).</summary>
    Box,

    /// <summary>Text of any length (<see cref="Space.Text"/>): a prompt, a reply.</summary>
    Text,

    /// <summary>A conversation (<see cref="Space.Conversation"/>): chat messages, with tool calls and their results.</summary>
    Conversation,
}

/// <summary>
/// The values an environment's observations or actions take (<see cref="IEnvironment.ObservationSpace"/>,
/// <see cref="IEnvironment.ActionSpace"/>), as Gymnasium's spaces describe them: one of n choices, a box of floats with
/// bounds, text, or a conversation. A policy reads it to shape its output (a categorical head of n, a Gaussian of the box's
/// size); the contract suites check that observations lie in it. Immutable; equal when kind, count, shape and bounds are.
/// </summary>
public sealed class Space : IEquatable<Space>
{
    private readonly int[] _shape;
    private readonly float[] _low, _high;

    private Space(SpaceKind kind, int count, int[] shape, float[] low, float[] high)
    {
        Kind = kind;
        Count = count;
        _shape = shape;
        _low = low;
        _high = high;
    }

    /// <summary>Text of any length: an observation that is a prompt, an action that is a reply.</summary>
    public static Space Text { get; } = new(SpaceKind.Text, 0, [], [], []);

    /// <summary>A conversation: an observation that is the chat so far, an action that is the assistant's next message.</summary>
    public static Space Conversation { get; } = new(SpaceKind.Conversation, 0, [], [], []);

    /// <summary>One of <paramref name="count"/> choices, the integers 0 to <paramref name="count"/> - 1.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is not positive.</exception>
    public static Space Discrete(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        return new(SpaceKind.Discrete, count, [], [], []);
    }

    /// <summary>
    /// Floats of <paramref name="shape"/> (outermost first), every one within [<paramref name="low"/>,
    /// <paramref name="high"/>]; an infinite bound leaves that side open.
    /// </summary>
    /// <exception cref="ArgumentException">A dimension that is not positive, a NaN bound, or <paramref name="low"/> above <paramref name="high"/>.</exception>
    public static Space Box(IReadOnlyList<int> shape, float low, float high)
    {
        int size = SizeOf(shape);
        var lows = new float[size];
        var highs = new float[size];
        lows.AsSpan().Fill(low);
        highs.AsSpan().Fill(high);
        return Box([.. shape], lows, highs);
    }

    /// <summary>Floats of <paramref name="shape"/>, element i within [<paramref name="low"/>[i], <paramref name="high"/>[i]] (row-major order).</summary>
    /// <exception cref="ArgumentException">
    /// A dimension that is not positive, bounds of another length than the shape's elements, a NaN bound, or a low bound
    /// above its high one.
    /// </exception>
    public static Space Box(IReadOnlyList<int> shape, IReadOnlyList<float> low, IReadOnlyList<float> high)
    {
        ArgumentNullException.ThrowIfNull(low);
        ArgumentNullException.ThrowIfNull(high);
        int size = SizeOf(shape);
        if (low.Count != size || high.Count != size)
        {
            throw new ArgumentException($"A box of {size} elements needs {size} low and high bounds, not {low.Count} and {high.Count}.");
        }

        float[] lows = [.. low], highs = [.. high];
        for (int i = 0; i < size; i++)
        {
            if (float.IsNaN(lows[i]) || float.IsNaN(highs[i]) || lows[i] > highs[i])
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"Element {i} of the box has the bounds {lows[i]} to {highs[i]}: a low bound above its high one, or NaN."));
            }
        }

        return new(SpaceKind.Box, 0, [.. shape], lows, highs);
    }

    /// <summary>What kind of values the space holds.</summary>
    public SpaceKind Kind { get; }

    /// <summary>The number of choices of a <see cref="SpaceKind.Discrete"/> space; 0 for the others.</summary>
    public int Count { get; }

    /// <summary>The shape of a <see cref="SpaceKind.Box"/> space, outermost first; empty for the others.</summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>The floats of one value of a box (the product of <see cref="Shape"/>); 1 for a discrete space, 0 for text and conversation.</summary>
    public int Size => Kind switch { SpaceKind.Box => _low.Length, SpaceKind.Discrete => 1, _ => 0 };

    /// <summary>The low bound of each element of a box, in row-major order; empty for the others.</summary>
    public IReadOnlyList<float> Low => _low;

    /// <summary>The high bound of each element of a box, in row-major order; empty for the others.</summary>
    public IReadOnlyList<float> High => _high;

    /// <summary>Whether <paramref name="choice"/> is one of a discrete space's choices (false for the other kinds).</summary>
    public bool Contains(int choice) => Kind == SpaceKind.Discrete && (uint)choice < (uint)Count;

    /// <summary>Whether <paramref name="values"/> is a value of a box: as many floats as <see cref="Size"/>, none NaN, each within its bounds (false for the other kinds).</summary>
    public bool Contains(ReadOnlySpan<float> values)
    {
        if (Kind != SpaceKind.Box || values.Length != _low.Length)
        {
            return false;
        }

        for (int i = 0; i < values.Length; i++)
        {
            if (!(values[i] >= _low[i] && values[i] <= _high[i]))                   // NaN fails both
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Whether <paramref name="value"/> is a value of the space: an <see cref="int"/> choice of a discrete space, a
    /// <see cref="float"/> array of a box, a <see cref="string"/> of a text space, and anything but null of a
    /// conversation (Idrak.Abstraction cannot name the conversation types of the packages that make them).
    /// </summary>
    public bool ContainsValue(object? value) => Kind switch
    {
        SpaceKind.Discrete => value is int choice && Contains(choice),
        SpaceKind.Box => value is float[] values && Contains(values.AsSpan()),
        SpaceKind.Text => value is string,
        _ => value is not null,
    };

    /// <inheritdoc />
    public bool Equals(Space? other) =>
        other is not null && Kind == other.Kind && Count == other.Count && _shape.AsSpan().SequenceEqual(other._shape)
        && _low.AsSpan().SequenceEqual(other._low) && _high.AsSpan().SequenceEqual(other._high);

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Space other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Kind, Count, _shape.Length, _low.Length);

    /// <summary>"discrete(4)", "box([3], -1 to 1)", "box([2, 2], bounds per element)", "text" or "conversation".</summary>
    public override string ToString() => Kind switch
    {
        SpaceKind.Discrete => string.Create(CultureInfo.InvariantCulture, $"discrete({Count})"),
        SpaceKind.Box when _low.AsSpan().IndexOfAnyExcept(_low[0]) < 0 && _high.AsSpan().IndexOfAnyExcept(_high[0]) < 0 =>
            string.Create(CultureInfo.InvariantCulture, $"box([{string.Join(", ", _shape)}], {_low[0]} to {_high[0]})"),
        SpaceKind.Box => $"box([{string.Join(", ", _shape)}], bounds per element)",
        SpaceKind.Text => "text",
        _ => "conversation",
    };

    // The elements of a box of `shape`: every dimension positive, the product within an array's length.
    private static int SizeOf(IReadOnlyList<int> shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Count == 0)
        {
            throw new ArgumentException("A box has at least one dimension.", nameof(shape));
        }

        long size = 1;
        foreach (int dimension in shape)
        {
            if (dimension <= 0)
            {
                throw new ArgumentException($"The box's shape [{string.Join(", ", shape)}] has a dimension that is not positive.", nameof(shape));
            }

            size *= dimension;
            if (size > Array.MaxLength)
            {
                throw new ArgumentException($"The box's shape [{string.Join(", ", shape)}] holds more elements than an array can.", nameof(shape));
            }
        }

        return (int)size;
    }
}
