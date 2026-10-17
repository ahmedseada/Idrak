// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>How <see cref="Tensor.Pad"/> fills the positions it adds.</summary>
public enum PaddingMode
{
    /// <summary>Zeros.</summary>
    Zeros,

    /// <summary>The values mirrored about the edge, the edge itself not repeated (torch's "reflect": 1 2 3 → 3 2 | 1 2 3 | 2 1).</summary>
    Reflect,

    /// <summary>The edge value repeated (torch's "replicate": 1 2 3 → 1 1 | 1 2 3 | 3 3).</summary>
    Replicate,
}

public sealed partial class Tensor
{
    /// <summary>
    /// This tensor with <paramref name="before"/> positions added in front and <paramref name="after"/> after along
    /// <paramref name="dim"/> (negative counts from the end), filled as <paramref name="mode"/> says. Zeros are made and
    /// joined; mirrored and repeated edges are gathered from the tensor (only the edges, so the gradient of a padded
    /// position reaches the position it copies without a pass over the whole tensor), then joined. Reflection needs at
    /// most size − 1 positions a side.
    /// </summary>
    public Tensor Pad(int dim, int before, int after, PaddingMode mode = PaddingMode.Zeros)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegative(before);
        ArgumentOutOfRangeException.ThrowIfNegative(after);
        dim = NormalizeDim(dim);
        if (before == 0 && after == 0)
        {
            return this;
        }

        int size = _shape[dim];
        var parts = new List<Tensor>(3);
        if (mode == PaddingMode.Zeros)
        {
            if (before > 0)
            {
                parts.Add(Zeros(WithSize(dim, before), Device));
            }

            parts.Add(this);
            if (after > 0)
            {
                parts.Add(Zeros(WithSize(dim, after), Device));
            }

            return Concat(parts, dim);
        }

        if (size == 0 || mode == PaddingMode.Reflect && (before > size - 1 || after > size - 1))
        {
            throw new ArgumentException($"Padding {before} before and {after} after by reflection needs at most {Math.Max(0, size - 1)} a side along dimension {dim} of size {size}.");
        }

        if (before > 0)
        {
            var positions = new int[before];
            for (int k = 0; k < before; k++)
            {
                positions[k] = mode == PaddingMode.Reflect ? before - k : 0;
            }

            parts.Add(GatherAlong(dim, positions));
        }

        parts.Add(this);
        if (after > 0)
        {
            var positions = new int[after];
            for (int k = 0; k < after; k++)
            {
                positions[k] = mode == PaddingMode.Reflect ? size - 2 - k : size - 1;
            }

            parts.Add(GatherAlong(dim, positions));
        }

        return Concat(parts, dim);
    }

    /// <summary>This tensor with the order along <paramref name="dim"/> reversed (torch.flip over one axis), with its gradient.</summary>
    public Tensor Flip(int dim)
    {
        ThrowIfDisposed();
        dim = NormalizeDim(dim);
        int size = _shape[dim];
        var positions = new int[size];
        for (int k = 0; k < size; k++)
        {
            positions[k] = size - 1 - k;
        }

        return GatherAlong(dim, positions);
    }

    // The positions `positions` of dimension `dim`, in that order: the tensor as rows [outer · size, inner] (the leading
    // dimensions and the axis as rows, the trailing ones as a row), one gather of the chosen rows, the result reshaped.
    private Tensor GatherAlong(int dim, int[] positions)
    {
        int outer = 1, inner = 1, size = _shape[dim];
        for (int d = 0; d < dim; d++)
        {
            outer *= _shape[d];
        }

        for (int d = dim + 1; d < _shape.Length; d++)
        {
            inner *= _shape[d];
        }

        var indices = new float[outer * positions.Length];
        for (int o = 0; o < outer; o++)
        {
            for (int k = 0; k < positions.Length; k++)
            {
                int position = positions[k];
                if (position < 0 || position >= size)
                {
                    throw new ArgumentOutOfRangeException(nameof(positions), position, $"Position {position} is outside dimension {dim} of size {size}.");
                }

                indices[o * positions.Length + k] = o * size + position;
            }
        }

        var rows = Reshape(outer * size, inner).EmbeddingLookup(From(indices, [indices.Length], Device));
        return rows.Reshape(WithSize(dim, positions.Length));
    }

    private int[] WithSize(int dim, int size)
    {
        int[] shape = [.. _shape];
        shape[dim] = size;
        return shape;
    }
}
