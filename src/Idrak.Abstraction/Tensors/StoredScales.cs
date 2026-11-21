// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// Which scale each value of a weight stored with scales takes (plan 37, Q0): over the stored [rows, columns] matrix (rows
/// the outputs), value (r, c) takes scale (r / <see cref="BlockRows"/>) · <see cref="ScaleColumns"/> + c /
/// <see cref="BlockColumns"/>. One scale for the tensor is one block of every row and column; one per row a block of one
/// row and every column. The default (all 0) is a weight with no scales.
/// </summary>
/// <param name="BlockRows">Rows a scale covers.</param>
/// <param name="BlockColumns">Columns a scale covers.</param>
/// <param name="ScaleColumns">Scales along a row of blocks.</param>
public readonly record struct StoredScales(int BlockRows, int BlockColumns, int ScaleColumns)
{
    /// <summary>Whether the weight has scales.</summary>
    public bool HasScales => BlockRows > 0;

    /// <summary>The index of the scale of stored value (<paramref name="row"/>, <paramref name="column"/>).</summary>
    public int IndexOf(int row, int column) => row / BlockRows * ScaleColumns + column / BlockColumns;

    /// <summary>
    /// Whether every one of <paramref name="rows"/> outputs shares the scales (one block of rows over them all), each scale a
    /// column or a run of columns of <paramref name="columns"/> (one for the whole tensor too): scales on x's columns, which a
    /// product applies to x once a call (m · columns multiplies) instead of to the weight's values.
    /// </summary>
    public bool OnColumns(int rows, int columns) =>
        HasScales && BlockRows >= rows && BlockColumns >= 1 && ScaleColumns >= (columns + BlockColumns - 1) / BlockColumns;
}
