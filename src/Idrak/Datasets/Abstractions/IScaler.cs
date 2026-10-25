// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Inference;

namespace Idrak.Datasets.Abstractions;

/// <summary>A column-wise transformation fitted on training data and applied to any data with the same columns.</summary>
public interface IScaler
{
    /// <summary>Transforms row-major data with <paramref name="columns"/> columns in place.</summary>
    void Transform(Span<float> data, int columns);

    /// <summary>Undoes <see cref="Transform"/> in place, e.g. to turn scaled predictions back into prices.</summary>
    void InverseTransform(Span<float> data, int columns);
}

/// <summary>
/// A scaler that a model package stores (<see cref="ModelPackageWriter.Scaler"/>): written as text under its entry kind,
/// which names the reader that loads it back (<see cref="ModelPackageReader.Scaler"/>).
/// </summary>
public interface ISavableScaler : IScaler
{
    /// <summary>The package entry kind it is stored as (<see cref="PackageEntryKind.StandardScaler"/>, <see cref="PackageEntryKind.MinMaxScaler"/>).</summary>
    PackageEntryKind EntryKind { get; }

    /// <summary>Writes the fitted values as text, as its reader reads them back.</summary>
    void Save(TextWriter writer);
}
