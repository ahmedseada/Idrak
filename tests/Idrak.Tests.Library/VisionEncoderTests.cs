// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Data;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Imaging;
using Idrak.Layers;
using Idrak.Models;

// Gemma 3's image side (plan 11, phase 3b): the SigLIP encoder and the projector of the tiny reference in
// tests/Idrak.Tests/data/vlm give transformers' embeddings, patch outputs and projected features, from the reference's
// pixel values and end to end from the PNG.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] VisionEncoderGroup =
    [
    ];

    internal static float MaxDifference(float[] expected, float[] actual) =>
        expected.Length != actual.Length ? float.PositiveInfinity : expected.Zip(actual, (a, b) => MathF.Abs(a - b)).Max();
}
