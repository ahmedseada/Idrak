// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

/// <summary>
/// A writing system, told from a character's Unicode block: its name, whether it runs right to left, and whether its
/// letters have case. Digits belong to the script they are written in (0-9 to Latin, ٠-٩ to Arabic).
/// </summary>
public readonly record struct WritingScript(string Name, bool RightToLeft, bool HasCase)
{
    /// <summary>Latin letters and the digits 0-9.</summary>
    public static readonly WritingScript Latin = new("Latin", false, true);

    /// <summary>Arabic letters and Arabic-Indic digits, right to left.</summary>
    public static readonly WritingScript Arabic = new("Arabic", true, false);

    /// <summary>Hebrew letters, right to left.</summary>
    public static readonly WritingScript Hebrew = new("Hebrew", true, false);

    /// <summary>Greek letters.</summary>
    public static readonly WritingScript Greek = new("Greek", false, true);

    /// <summary>Cyrillic letters.</summary>
    public static readonly WritingScript Cyrillic = new("Cyrillic", false, true);

    /// <summary>Devanagari letters and digits.</summary>
    public static readonly WritingScript Devanagari = new("Devanagari", false, false);

    /// <summary>Chinese, Japanese and Korean characters.</summary>
    public static readonly WritingScript Han = new("Han", false, false);

    /// <summary>Everything else (punctuation, symbols).</summary>
    public static readonly WritingScript Common = new("Common", false, false);

    /// <summary>The script of a character.</summary>
    public static WritingScript Of(char c) => c switch
    {
        >= '0' and <= '9' => Latin,
        >= 'A' and <= 'Z' or >= 'a' and <= 'z' or >= 'À' and <= 'ɏ' and not '×' and not '÷' => Latin,
        >= 'Ͱ' and <= 'Ͽ' => Greek,
        >= 'Ѐ' and <= 'ԯ' => Cyrillic,
        >= '֐' and <= '׿' or >= 'יִ' and <= 'ﭏ' => Hebrew,
        >= '؀' and <= 'ۿ' or >= 'ݐ' and <= 'ݿ' or >= 'ࢠ' and <= 'ࣿ'
            or >= 'ﭐ' and <= '﷿' or >= 'ﹰ' and <= '﻿' => Arabic,
        >= 'ऀ' and <= 'ॿ' => Devanagari,
        >= '぀' and <= 'ヿ' or >= '一' and <= '鿿' or >= '가' and <= '힯' => Han,
        _ => Common,
    };

    /// <summary>The script of a text: that of its first character that has one (Common if none).</summary>
    public static WritingScript Of(string text)
    {
        foreach (char c in text)
        {
            var script = Of(c);
            if (script != Common)
            {
                return script;
            }
        }

        return Common;
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}
