// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// A tokenizer.json's normalizer and pre-tokenizer on spans (plan 24, L1): text in, the pieces a tokenizer model encodes
/// out, as ranges of one normalized text that knows where each character came from. Built once per tokenizer; safe to run
/// from several threads at once, each with its own <see cref="PreTokenizedText"/> (which keeps its buffers between runs, so
/// a warm run allocates nothing).
/// </summary>
internal sealed class TokenizerPipeline
{
    private readonly Action<NormalizedText>? _normalizer;
    private readonly Action<PreTokenizedText, bool>? _preTokenizer;

    /// <summary>The pipeline of <paramref name="json"/> (a tokenizer.json, or its definition's JSON).</summary>
    public TokenizerPipeline(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        _normalizer = json["normalizer"] is JsonObject normalizer ? SpanComponents.Normalizer(normalizer) : null;
        _preTokenizer = json["pre_tokenizer"] is JsonObject preTokenizer ? SpanComponents.PreTokenizer(preTokenizer) : null;
    }

    /// <summary>
    /// Normalizes and pre-tokenizes <paramref name="text"/>[<paramref name="start"/>..<paramref name="start"/> + <paramref name="length"/>]
    /// into <paramref name="output"/>; <paramref name="atStart"/> says whether the range begins the input (not after an
    /// added token).
    /// </summary>
    public void Run(string text, int start, int length, bool atStart, PreTokenizedText output)
    {
        ArgumentNullException.ThrowIfNull(output);
        output.Text.Reset(text, start, length);
        Run(atStart, output);
    }

    /// <summary>
    /// <see cref="Run(string, int, int, bool, PreTokenizedText)"/> on <paramref name="text"/>[<paramref name="start"/>..<paramref name="start"/> + <paramref name="length"/>]
    /// of a caller's buffer whose first <paramref name="textLength"/> characters are the text (left unchanged until the
    /// next run on <paramref name="output"/>): no string is made of it.
    /// </summary>
    public void Run(char[] text, int textLength, int start, int length, bool atStart, PreTokenizedText output)
    {
        ArgumentNullException.ThrowIfNull(output);
        output.Text.Reset(text, textLength, start, length);
        Run(atStart, output);
    }

    private void Run(bool atStart, PreTokenizedText output)
    {
        _normalizer?.Invoke(output.Text);
        output.OnePiece();
        _preTokenizer?.Invoke(output, atStart);
    }
}
