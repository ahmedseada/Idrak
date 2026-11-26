// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction;
using Idrak.Abstraction.Generation;
using Idrak.Abstraction.Testing;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase T1: the Unigram model, against the fixtures' references (data/tokenizers/README.md): tokenizers for each
// tokenizer.json, sentencepiece for each tokenizer.model.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TokenizerUnigramGroup =
    [
    ];

    internal static int[] Ints(JsonNode? array) => [.. array!.AsArray().Select(x => (int)x!)];
}
