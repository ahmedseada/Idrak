// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>
/// A row of a per-element type's values with a float32 scale of its own (what int8's and int16's
/// <see cref="IElementType.ForRow"/> give: they hold no fraction): one block of the row's values, its scale first, then
/// each value's code as the element type stores it (the block form the described lanes read, ggml's Q8_0 with a block a
/// row); a value is its code's number times its row's scale. Read and written by the interface's defaults from that
/// description alone, with no tensor of scales beside it.
/// </summary>
public sealed class RowScale : IElementType
{
    private static readonly NumberType Float32 = NumberType.Floating(8, 23);

    internal RowScale(IElementType element, int length)
    {
        var e = element.Description;
        if (e is not { Values: 1, Group: 1, Scales: null or [], Mins: null or [], Sums: null or [], Planes: [{ Offset: 0, Group: 1, Stride: 0, Stream: false, Places: null, Radix: 0 } plane] }
            || plane.Bits != 8 * e.Bytes)
        {
            throw new ArgumentException($"A row scale is given to a type of one value a block, its code whole bytes; {element} is not one.", nameof(element));
        }

        Element = element;
        Length = length;
        Description = e with { Values = length, Bytes = 4 + length * e.Bytes, Planes = [plane with { Offset = 4 }], Scales = [new BlockScale(Float32, length, 0)] };
        Name = $"{element.Name} rows of {length}, a float32 scale a row";
    }

    /// <summary>The type whose codes a row holds.</summary>
    public IElementType Element { get; }

    /// <summary>Values a row holds.</summary>
    public int Length { get; }

    /// <inheritdoc />
    public TypeDescription Description { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <summary>Itself for its own rows' length; for another, the element type's row of that length.</summary>
    /// <param name="length">Values a row holds.</param>
    public IElementType ForRow(int length) => length == Length ? this : Element.ForRow(length);

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>
/// The row types (<see cref="RowScale"/>) one element type gives: each row length's made once and given again, so a row
/// type, and its description, is one object (a device's plans and its measured lanes, kept a description, made once).
/// </summary>
/// <param name="element">The element type whose rows these are.</param>
public sealed class RowScales(IElementType element)
{
    private readonly Lock _gate = new();
    private RowScale[] _made = [];

    /// <summary>The row of <paramref name="length"/> values.</summary>
    /// <param name="length">Values a row holds.</param>
    public IElementType Of(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        if (Find(Volatile.Read(ref _made), length) is { } made)
        {
            return made;
        }

        lock (_gate)
        {
            if (Find(_made, length) is { } again)
            {
                return again;
            }

            var row = new RowScale(element, length);
            Volatile.Write(ref _made, [.. _made, row]);
            return row;
        }
    }

    private static RowScale? Find(RowScale[] made, int length)
    {
        foreach (var row in made)
        {
            if (row.Length == length)
            {
                return row;
            }
        }

        return null;
    }
}
