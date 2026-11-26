// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Abstraction.Text;

namespace Idrak.Text.Abstractions;

/// <summary>
/// What of a model's answer a metric compares with the reference: the answer as it is, the values of its JSON, or
/// another reading.
/// </summary>
public interface IAnswerText
{
    /// <summary>Its name (<see cref="AnswerTexts.Raw"/>, <see cref="AnswerTexts.Values"/>).</summary>
    string Name { get; }

    /// <summary>The text of <paramref name="answer"/> to compare (not normalized: <see cref="AnswerTexts.Normalize"/> does that).</summary>
    string Text(string answer);
}

/// <summary>
/// The names of the answer texts and how metrics compare text. The library's: <see cref="Raw"/> (the answer as it is)
/// and <see cref="Values"/> (its JSON's string and number values in document order, one a line, the JSON read through
/// the lenient <see cref="JsonRepairs"/>; an answer without JSON as it is).
/// </summary>
public static class AnswerTexts
{
    /// <summary>The answer as it is.</summary>
    public const string Raw = "raw";

    /// <summary>The values of the answer's JSON in document order, one a line (an answer without JSON as it is).</summary>
    public const string Values = "values";

    /// <summary>
    /// Text as metrics compare it: Unicode composed (NFC), byte-order marks dropped, every whitespace run one space,
    /// trimmed (so line breaks and spacing do not count as errors).
    /// </summary>
    public static string Normalize(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return "";
        }

        var builder = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in UnicodeNormalization.Normalize(text, NormalizationForm.FormC))
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
}
