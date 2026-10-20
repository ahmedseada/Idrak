// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// How the app writes and compares text. Transcriptions are kept in logical Unicode order (the order they are typed and
/// stored in, which for Arabic is the reading order, right to left on the page), NFC-normalized, white space collapsed.
/// </summary>
internal static class OcrText
{
    /// <summary>UTF-8 without a byte order mark: what every file the app writes uses.</summary>
    public static readonly Encoding Utf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

    /// <summary>
    /// A transcription as the app keeps it: NFC, a byte order mark dropped, every run of white space (line breaks
    /// included) one space, trimmed.
    /// </summary>
    public static string Normalize(string? text) => AnswerTexts.Normalize(text);

    /// <summary>A text file's transcription (<see cref="Normalize"/>), "" when the file is missing.</summary>
    public static string ReadTranscription(string path) => File.Exists(path) ? Normalize(File.ReadAllText(path, Encoding.UTF8)) : "";

    /// <summary>
    /// Writes <paramref name="text"/> as UTF-8 without a BOM; lines end with <see cref="Environment.NewLine"/> (as the
    /// console's WriteLine does), so a file and the console text it was printed from agree on every platform.
    /// </summary>
    public static void Write(string path, string text)
    {
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        File.WriteAllText(path, EndLines(text), Utf8);
    }

    /// <summary>Every line ending (\r\n, \r or \n) as <see cref="Environment.NewLine"/>.</summary>
    public static string EndLines(string text) => text.ReplaceLineEndings(Environment.NewLine);

    /// <summary>
    /// The text an answer stands for. <c>raw</c>: the answer as it is. <c>values</c>: when the answer is JSON (a
    /// structured page, as the legal-documents fine-tune answers; a ```json fence is allowed), its string and number
    /// values in document order, one per line; an answer that is not JSON is kept as it is.
    /// </summary>
    public static string AnswerText(string answer, string mode) => mode switch
    {
        "raw" or "values" => AnswerTexts.Get(mode).Text(answer),
        _ => throw new UsageException($"--truth-text is raw or values, not '{mode}'."),
    };
}

/// <summary>JSON writing options of the app.</summary>
internal static class Json
{
    public static readonly JsonSerializerOptions Indented = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,                 // Arabic as itself, not \u escapes
    };
}
