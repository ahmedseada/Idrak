// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

/// <summary>
/// The scales a weight is stored with (plan 37, Q0): <see cref="Scales"/> (and <see cref="Zeros"/>, or null) laid out as
/// <see cref="Layout"/> says, so value(r, c) = (decoded − zero) · scale over the stored [rows, columns] matrix, as the store
/// computes it when it reads float32 values.
/// </summary>
/// <param name="Scales">The scales, in the layout's order.</param>
/// <param name="Zeros">The zero points (one per scale), or null.</param>
/// <param name="Layout">Which scale each stored value takes.</param>
public sealed record StoredScaling(float[] Scales, float[]? Zeros, StoredScales Layout);
