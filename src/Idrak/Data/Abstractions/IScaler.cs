// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Abstractions;

/// <summary>A column-wise transformation fitted on training data and applied to any data with the same columns.</summary>
public interface IScaler
{
    /// <summary>Transforms row-major data with <paramref name="columns"/> columns in place.</summary>
    void Transform(Span<float> data, int columns);

    /// <summary>Undoes <see cref="Transform"/> in place, e.g. to turn scaled predictions back into prices.</summary>
    void InverseTransform(Span<float> data, int columns);
}
