// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// Provides weights by name for <c>DecoderSpec.Build</c>, in Idrak's layout (Linear weights [in, out]). The
/// names are listed in <c>DecoderSpec</c>. Implementations read files (for example a pretrained-model package
/// translating another framework's names and layouts) or anything else.
/// </summary>
public interface IWeightSource
{
    /// <summary>The values of the tensor <paramref name="name"/> with <paramref name="shape"/>, or null when the source does not have it.</summary>
    float[]? Read(string name, IReadOnlyList<int> shape);
}
