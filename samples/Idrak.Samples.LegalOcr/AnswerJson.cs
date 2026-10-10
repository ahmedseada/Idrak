// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Nlp.Abstractions;

namespace Idrak.Samples.LegalOcr;

/// <summary>How the page shows a model's JSON answer: read through the library's lenient JSON repair, indented, Arabic as itself.</summary>
public static class AnswerJson
{
    private static readonly JsonSerializerOptions Indented = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>The answer's JSON indented and whether it needed repair; null when the answer has none.</summary>
    public static (string? Json, bool Repaired) Show(string answer) =>
        JsonRepairs.Parse(answer) is { } found ? (found.Json?.ToJsonString(Indented) ?? "null", found.Repaired) : (null, false);
}
