// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// One glyph of shaped text, in font units: which glyph, the text position it came from, how far it moves the pen and
/// how far it sits from the pen (y up).
/// </summary>
/// <param name="Glyph">The glyph id in the font.</param>
/// <param name="Cluster">The index (UTF-16) in the text of the first character the glyph stands for; glyphs of one cluster draw one unit of text.</param>
/// <param name="XAdvance">How far the pen moves after it.</param>
/// <param name="YAdvance">The same, vertically (0 for horizontal text).</param>
/// <param name="XOffset">Where it is drawn, from the pen.</param>
/// <param name="YOffset">The same, vertically (a mark above a letter is drawn higher).</param>
public readonly record struct ShapedGlyph(int Glyph, int Cluster, int XAdvance, int YAdvance, int XOffset, int YOffset);

/// <summary>
/// A run of text one shaper turns into glyphs: one script, one direction.
/// </summary>
/// <param name="Start">Its first index (UTF-16) in the text.</param>
/// <param name="Length">Its length (UTF-16).</param>
/// <param name="Script">Its script, an ISO 15924 code ("Arab", "Latn", "Hebr", "Hani"; "Zyyy" for text of no script, such as digits and punctuation alone).</param>
/// <param name="RightToLeft">Whether it reads right to left.</param>
/// <param name="Language">Its language, a BCP 47 tag ("ar", "fa", "ur"), or null: the font's default rules for the script.</param>
public readonly record struct TextRun(int Start, int Length, string Script, bool RightToLeft, string? Language = null)
{
    /// <summary>How characters share clusters (<see cref="TextClusters.Graphemes"/> unless set).</summary>
    public TextClusters Clusters { get; init; }
}

/// <summary>How a shaper groups characters into clusters (HarfBuzz's monotone cluster levels).</summary>
public enum TextClusters
{
    /// <summary>A letter and its marks are one cluster from the start (HarfBuzz's default, MONOTONE_GRAPHEMES).</summary>
    Graphemes,

    /// <summary>Each character is its own cluster until shaping joins them (MONOTONE_CHARACTERS).</summary>
    Characters,
}

/// <summary>
/// Turns a run of text into positioned glyphs of a font, for the scripts it is registered under in
/// <see cref="TextShapers"/>: the characters mapped to glyphs, substituted and positioned by the font's layout tables as the
/// script's rules say. The glyphs come back in the order they are drawn, left to right (a right-to-left run reversed).
/// </summary>
public interface ITextShaper
{
    /// <summary>The shaper's name ("general", "arabic").</summary>
    string Name { get; }

    /// <summary>
    /// Shapes <paramref name="run"/> of <paramref name="text"/> with <paramref name="font"/>. The text around the run is
    /// context (an Arabic letter joins across a run's edge). <paramref name="features"/> turn OpenType features on
    /// ("+smcp") or off ("-liga") over the script's defaults; null for the defaults.
    /// </summary>
    ShapedGlyph[] Shape(IFontFace font, string text, TextRun run, IReadOnlyCollection<string>? features = null);
}

/// <summary>
/// The text shapers, by script (ISO 15924 code, ignoring case). A script with no shaper of its own is shaped by the one
/// registered under <see cref="AnyScript"/>. Built in (Idrak.Imaging): "general" under <see cref="AnyScript"/> (the
/// character map, then the font's GSUB and GPOS features for the script: right for scripts that neither join nor reorder,
/// such as Latin, Greek, Cyrillic, Armenian, Georgian, Hebrew and the CJK scripts) and "arabic" for Arab, Syrc, Nkoo,
/// Mong, Mand, Mani, Phlp, Adlm, Rohg, Sogd and Ougr (cursive joining). Register a shaper for another script with
/// <see cref="Register"/>.
/// </summary>
public static class TextShapers
{
    /// <summary>The key of the shaper for every script without its own.</summary>
    public const string AnyScript = "*";

    private static readonly SlotTable<string, ITextShaper> Registry =
        new(nameof(TextShapers), (slot, app, library) => new GuardedShaper(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static TextShapers() => LibraryDefaults.Ensure(typeof(TextShapers));

    /// <summary>Registers <paramref name="shaper"/> for <paramref name="script"/> (or <see cref="AnyScript"/>), over the library's for that script, which stays behind it.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string script, ITextShaper shaper)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(script);
        ArgumentNullException.ThrowIfNull(shaper);
        Registry.Register(script, shaper, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's shaper for <paramref name="script"/> (the library's, if any, answers again).</summary>
    public static bool Unregister(string script) => Registry.Unregister(script);

    /// <summary>The scripts with a shaper of their own, and <see cref="AnyScript"/>.</summary>
    public static IReadOnlyCollection<string> Scripts => Registry.Keys;

    /// <summary>The shaper of <paramref name="script"/>, else the one for <see cref="AnyScript"/>.</summary>
    /// <exception cref="NotSupportedException">Neither is registered.</exception>
    public static ITextShaper Get(string script) => Registry.Find(script) ?? Registry.Find(AnyScript)
        ?? throw new NotSupportedException($"No text shaper is registered for '{script}' or for every script ('{AnyScript}'); add one with TextShapers.Register.");

    /// <summary>The library's shaper registered for <paramref name="script"/>, or null.</summary>
    public static ITextShaper? Default(string script) => Registry.Default(script);

    /// <summary>Who registered the shaper for <paramref name="script"/>.</summary>
    public static string? Origin(string script) => Registry.Origin(script);

    /// <summary>What happens when the app's shaper for <paramref name="script"/> fails.</summary>
    public static void SetPolicy(string script, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(script, policy, shadowRate);

    /// <summary>Shapes a run with the shaper of its script.</summary>
    public static ShapedGlyph[] Shape(IFontFace font, string text, TextRun run, IReadOnlyCollection<string>? features = null) =>
        Get(run.Script).Shape(font, text, run, features);

    // An app's shaper over the library's: a failed run is shaped by the library's, or both are compared.
    private sealed class GuardedShaper(Slot slot, ITextShaper app, ITextShaper library) : ITextShaper
    {
        public string Name => app.Name;

        public ShapedGlyph[] Shape(IFontFace font, string text, TextRun run, IReadOnlyCollection<string>? features = null)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var shadow = slot.Shadow();
                var answer = library.Shape(font, text, run, features);
                if (shadow is not null)
                {
                    shadow.Answered();
                    try
                    {
                        shadow.Done(Comparisons.Exact(string.Join(" ", answer), string.Join(" ", app.Shape(font, text, run, features))));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        shadow.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Shape(font, text, run, features);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Shape(font, text, run, features);
            }
        }
    }
}
