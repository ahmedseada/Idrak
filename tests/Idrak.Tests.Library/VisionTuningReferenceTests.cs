// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Plan 12, phase 0: the tiny vision fine-tune of tools/vlm/make_tiny_tuning.py (tests/Idrak.Tests/data/vlm-tuning): its
// folder, whose records the ShareGPT files are written from.
internal abstract partial class Tests
{
    internal static string TuningData(string name) => TestData($"vlm-tuning/{name}");
}
