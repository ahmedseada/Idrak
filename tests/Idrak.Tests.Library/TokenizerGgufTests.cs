// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T3: GGUF tokenizers beyond gpt2 (llama, t5, bert), through the GGUF tokenizers.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TokenizerGgufGroup =
    [
    ];
}
