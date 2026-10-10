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
    public static string Normalize(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return "";
        }

        var builder = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in text.Normalize(NormalizationForm.FormC))
        {
            if (c == '﻿')
            {
                continue;
            }

            if (char.IsWhiteSpace(c))
            {
                space = builder.Length > 0;
                continue;
            }

            if (space)
            {
                builder.Append(' ');
                space = false;
            }

            builder.Append(c);
        }

        return builder.ToString();
    }

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
        "raw" => answer,
        "values" => JsonValues(answer) ?? answer,
        _ => throw new UsageException($"--truth-text is raw or values, not '{mode}'."),
    };

    // The values of a JSON answer, or null when it is not JSON.
    private static string? JsonValues(string answer)
    {
        string text = answer.Trim();
        if (text.StartsWith("```", StringComparison.Ordinal))
        {
            int start = text.IndexOf('\n');
            int end = text.LastIndexOf("```", StringComparison.Ordinal);
            if (start < 0 || end <= start)
            {
                return null;
            }

            text = text[(start + 1)..end].Trim();
        }

        if (text.Length == 0 || text[0] is not ('{' or '['))
        {
            return null;
        }

        JsonNode? node;
        try
        {
            node = JsonNode.Parse(text);
        }
        catch (JsonException)
        {
            return null;
        }

        var values = new List<string>();
        Collect(node, values);
        return string.Join("\n", values);
    }

    private static void Collect(JsonNode? node, List<string> values)
    {
        switch (node)
        {
            case JsonObject obj:
                foreach (var (_, value) in obj)
                {
                    Collect(value, values);
                }

                break;
            case JsonArray array:
                foreach (var item in array)
                {
                    Collect(item, values);
                }

                break;
            case JsonValue value when value.GetValueKind() is JsonValueKind.String:
                if (value.GetValue<string>() is { Length: > 0 } s)
                {
                    values.Add(s);
                }

                break;
            case JsonValue value when value.GetValueKind() is JsonValueKind.Number:
                values.Add(value.ToJsonString());
                break;
        }
    }
}

/// <summary>
/// Reading order of a line for the recognizer. The recognizer reads a line image's columns one after another; for a
/// right-to-left script the line image is flipped horizontally first, so its columns run in reading order and the labels
/// are the transcription in logical order. One exception is undone here: inside a right-to-left line, a run of digits or
/// left-to-right letters (a number, a Latin word) is drawn left to right, so in the flipped image it appears reversed.
/// <see cref="ToColumns"/> reverses such runs in a transcription (what the flipped image shows, in its column order) and
/// <see cref="FromColumns"/> reverses them back (the same operation: it is its own inverse).
/// </summary>
internal static class ReadingOrder
{
    /// <summary>The transcription in the flipped line's column order (left-to-right runs reversed).</summary>
    public static string ToColumns(string logical) => ReverseLeftToRightRuns(logical);

    /// <summary>A reading in column order back in logical order.</summary>
    public static string FromColumns(string columns) => ReverseLeftToRightRuns(columns);

    /// <summary>Whether a character is drawn left to right inside a right-to-left line: a digit (Arabic-Indic too) or a letter of a left-to-right script.</summary>
    public static bool IsLeftToRight(Rune rune) =>
        Rune.IsDigit(rune) || Rune.IsLetter(rune) && !IsRightToLeftLetter(rune);

    private static bool IsRightToLeftLetter(Rune rune) => rune.Value is >= 0x0590 and <= 0x08FF or >= 0xFB1D and <= 0xFDFF or >= 0xFE70 and <= 0xFEFF
        or >= 0x10800 and <= 0x10FFF or >= 0x1E800 and <= 0x1EFFF;

    // A run: left-to-right characters, with the spaces and punctuation between two of them (not at its ends).
    private static string ReverseLeftToRightRuns(string text)
    {
        var runes = text.EnumerateRunes().ToArray();
        var builder = new StringBuilder(text.Length);
        Span<char> pair = stackalloc char[2];
        for (int i = 0; i < runes.Length;)
        {
            if (!IsLeftToRight(runes[i]))
            {
                builder.Append(pair[..runes[i++].EncodeToUtf16(pair)]);
                continue;
            }

            int end = i, j = i;
            while (j < runes.Length && (IsLeftToRight(runes[j]) || IsJoiner(runes[j])))
            {
                if (IsLeftToRight(runes[j]))
                {
                    end = j;
                }

                j++;
            }

            for (int k = end; k >= i; k--)
            {
                builder.Append(pair[..runes[k].EncodeToUtf16(pair)]);
            }

            i = end + 1;
        }

        return builder.ToString();
    }

    // Inside a number or a Latin phrase: a space, a separator or punctuation that keeps the run together.
    private static bool IsJoiner(Rune rune) => rune.Value is ' ' or '.' or ',' or ':' or '/' or '-' or '٫' or '٬';
}

/// <summary>Where a line's text lies in its page's known text (the draft snapped to the truth).</summary>
/// <param name="Start">The span's first character in the page text.</param>
/// <param name="Length">The span's length.</param>
/// <param name="Distance">The edit distance between the draft and the span.</param>
internal readonly record struct TextSpan(int Start, int Length, int Distance);

/// <summary>Approximate matching of a line's draft inside its page's text (Sellers' semi-global edit distance).</summary>
internal static class TextAlignment
{
    /// <summary>
    /// The span of <paramref name="text"/>, starting at or after <paramref name="from"/>, that is nearest
    /// <paramref name="draft"/> by edit distance (characters as Unicode scalars), its ends moved out to word boundaries;
    /// null when the draft or the rest of the text is empty.
    /// </summary>
    public static TextSpan? Best(string draft, string text, int from)
    {
        if (from >= text.Length || draft.Length == 0)
        {
            return null;
        }

        // One rented buffer: the draft's scalars, the text's scalars from `from` with each one's char index, and two rows
        // of the table (only the last row is read, so the rows before it are not kept).
        int size = text.Length - from + 1;
        int[] buffer = ArrayPool<int>.Shared.Rent(draft.Length + 6 * size);
        try
        {
            var d = buffer.AsSpan(0, draft.Length);
            int n = 0;
            foreach (var rune in draft.EnumerateRunes())
            {
                d[n++] = rune.Value;
            }

            int m = 0;
            var t = buffer.AsSpan(draft.Length, size);
            var index = buffer.AsSpan(draft.Length + size, size);
            for (int i = from; i < text.Length;)
            {
                var rune = Rune.GetRuneAt(text, i);
                (t[m], index[m]) = (rune.Value, i);
                m++;
                i += rune.Utf16SequenceLength;
            }

            // cost[j]: distance of d[..i] to the best span ending at t[j - 1]; start[j]: where that span starts. Row i - 1
            // in the previous pair, row i in the current one.
            var cost = buffer.AsSpan(draft.Length + 2 * size, m + 1);
            var start = buffer.AsSpan(draft.Length + 3 * size, m + 1);
            var nextCost = buffer.AsSpan(draft.Length + 4 * size, m + 1);
            var nextStart = buffer.AsSpan(draft.Length + 5 * size, m + 1);
            for (int j = 0; j <= m; j++)
            {
                (cost[j], start[j]) = (0, j);
            }

            for (int i = 1; i <= n; i++)
            {
                (nextCost[0], nextStart[0]) = (i, 0);
                int letter = d[i - 1];
                for (int j = 1; j <= m; j++)
                {
                    int substitute = cost[j - 1] + (letter == t[j - 1] ? 0 : 1);
                    int delete = cost[j] + 1;
                    int insert = nextCost[j - 1] + 1;
                    if (substitute <= delete && substitute <= insert)
                    {
                        (nextCost[j], nextStart[j]) = (substitute, start[j - 1]);
                    }
                    else if (delete <= insert)
                    {
                        (nextCost[j], nextStart[j]) = (delete, start[j]);
                    }
                    else
                    {
                        (nextCost[j], nextStart[j]) = (insert, nextStart[j - 1]);
                    }
                }

                var swap = cost;
                cost = nextCost;
                nextCost = swap;
                swap = start;
                start = nextStart;
                nextStart = swap;
            }

            int bestEnd = 1;
            for (int j = 2; j <= m; j++)
            {
                if (cost[j] < cost[bestEnd])
                {
                    bestEnd = j;
                }
            }

            int first = index[start[bestEnd]], last = bestEnd >= m ? text.Length : index[bestEnd];
            first = Math.Min(first, last);
            // Out to word boundaries: a line breaks between words.
            while (first > from && !char.IsWhiteSpace(text[first - 1]) && first < last)
            {
                first--;
            }

            while (last < text.Length && !char.IsWhiteSpace(text[last]))
            {
                last++;
            }

            return new TextSpan(first, last - first, cost[bestEnd]);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Each draft's span in <paramref name="text"/>, top to bottom: each starts where the previous accepted one ended. A
    /// draft whose best span differs in more than <paramref name="maxError"/> of its characters is left unaligned (null),
    /// and the next draft searches from the same place.
    /// </summary>
    public static IReadOnlyList<(string Text, double Match)?> Align(IReadOnlyList<string> drafts, string text, double maxError = 0.5)
    {
        var result = new List<(string, double)?>();
        int from = 0;
        foreach (string draft in drafts)
        {
            if (Best(draft, text, from) is { } span && span.Distance <= maxError * draft.EnumerateRunes().Count())
            {
                string aligned = OcrText.Normalize(text.Substring(span.Start, span.Length));
                double match = 1 - span.Distance / (double)Math.Max(1, draft.EnumerateRunes().Count());
                result.Add((aligned, match));
                from = span.Start + span.Length;
            }
            else
            {
                result.Add(null);
            }
        }

        return result;
    }
}
