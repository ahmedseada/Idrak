// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

/// <summary>
/// Tensors held in memory by name as an <see cref="ITensorStore"/>: what a reader of stored weights takes before a first
/// save (new layers drawn from a seed, a checkpoint's tensors renamed), or a test's fixture.
/// </summary>
public sealed class TensorDictionary : ITensorStore
{
    private readonly Dictionary<string, (int[] Shape, float[] Values)> _tensors = new(StringComparer.Ordinal);

    /// <inheritdoc />
    public IEnumerable<string> Names => _tensors.Keys;

    /// <summary>The tensors held.</summary>
    public int Count => _tensors.Count;

    /// <summary>Adds (or replaces) <paramref name="name"/> with <paramref name="values"/> of <paramref name="shape"/>, row-major.</summary>
    /// <exception cref="ArgumentException">The values do not fill the shape.</exception>
    public void Add(string name, ReadOnlySpan<int> shape, float[] values)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(values);
        long count = 1;
        foreach (int d in shape)
        {
            count *= d;
        }

        if (count != values.Length)
        {
            throw new ArgumentException($"{name}: {values.Length} values for {Tensor.FormatShape(shape)}.", nameof(values));
        }

        _tensors[name] = (shape.ToArray(), values);
    }

    /// <summary>Adds (or replaces) <paramref name="name"/> with <paramref name="tensor"/>'s values and shape (read from its device).</summary>
    public void Add(string name, Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        Add(name, tensor.Shape, tensor.ToArray());
    }

    /// <summary>Takes <paramref name="name"/> out; false when it was not held.</summary>
    public bool Remove(string name) => _tensors.Remove(name);

    /// <inheritdoc />
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <inheritdoc />
    public int[] ShapeOf(string name) => Entry(name).Shape;

    /// <inheritdoc />
    public float[] Read(string name) => Entry(name).Values;

    /// <inheritdoc />
    public void Dispose()
    {
    }

    private (int[] Shape, float[] Values) Entry(string name) =>
        _tensors.TryGetValue(name, out var entry) ? entry : throw new KeyNotFoundException($"The tensor '{name}' is not held ({_tensors.Count} tensors are).");
}
