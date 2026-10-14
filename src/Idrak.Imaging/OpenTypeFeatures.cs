// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging.Shaping;

namespace Idrak.Imaging;

/// <summary>The layout features a font has for a script: what its GSUB and GPOS tables offer that language system.</summary>
public static class OpenTypeFeatures
{
    /// <summary>
    /// The feature tags of <paramref name="font"/>'s GSUB (substitution) and GPOS (positioning) for
    /// <paramref name="script"/> (ISO 15924, "Arab") and <paramref name="language"/> (BCP 47, or null for the
    /// script's default), in the tables' order, each once; empty where the font has no table or no system for it.
    /// </summary>
    public static (IReadOnlyList<string> Substitution, IReadOnlyList<string> Positioning) Of(IFontFace font, string script, string? language = null)
    {
        ArgumentNullException.ThrowIfNull(font);
        ArgumentException.ThrowIfNullOrWhiteSpace(script);
        var layout = FontLayout.For(font);
        var scripts = OpenTypeTags.Scripts(script);
        string? languageTag = OpenTypeTags.Language(language);
        return (Tags(layout.Substitutions, scripts, languageTag), Tags(layout.Positions, scripts, languageTag));
    }

    private static List<string> Tags(LayoutTable table, IReadOnlyList<string> scripts, string? language)
    {
        var tags = new List<string>();
        if (table.LanguageSystem(scripts, language) is not var (features, required))
        {
            return tags;
        }

        foreach (int index in required >= 0 ? [required, .. features] : features)
        {
            uint tag = table.FeatureTag(index);
            string text = new([(char)(tag >> 24), (char)((tag >> 16) & 0xFF), (char)((tag >> 8) & 0xFF), (char)(tag & 0xFF)]);
            if (!tags.Contains(text))
            {
                tags.Add(text);
            }
        }

        return tags;
    }
}
