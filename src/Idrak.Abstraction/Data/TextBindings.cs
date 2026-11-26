// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A text binding's settings, read once per page from its options (<see cref="ITextBinding.Read"/>): the options as
/// given, and whatever a binding parses them into (a binding's own settings derive from this class).
/// </summary>
public class TextBindingSettings
{
    /// <summary>Settings of <paramref name="options"/> (none when null).</summary>
    public TextBindingSettings(IReadOnlyDictionary<string, string>? options = null) => Options = options ?? PageOptions.None;

    /// <summary>No options: a binding's defaults.</summary>
    public static TextBindingSettings None { get; } = new();

    /// <summary>The options as given, by key.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }
}

/// <summary>Text put into regions: each region's text (empty for a region left empty), and the label, the drawn text one line a group in reading order.</summary>
/// <param name="Texts">Each region's text, in the order the regions were given ("" for one that draws nothing).</param>
/// <param name="Label">The drawn text: one line per group that has text, in reading order; the units of each line as its regions hold them.</param>
public sealed record BoundText(IReadOnlyList<string> Texts, IReadOnlyList<string> Label);

/// <summary>
/// Puts a page's text into the regions its shapes make (<see cref="ShapeRegion"/>): which units (whitespace-separated
/// tokens) of which line go into which region. Its settings are read once per page (<see cref="Read"/>). The contract:
/// a region left empty draws nothing and its units are absent from the label; units placed nowhere are absent from the
/// label; the label is exactly the drawn text, one line per group that has text, in reading order (groups in the order
/// of their numbers, a group's regions in the order of their indexes). The same input and seed give the same binding.
/// Register one with <c>Register</c>.
/// </summary>
public interface ITextBinding
{
    /// <summary>The binding's settings from <paramref name="options"/> (its own keys, <c>Keys</c>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad
    /// value.</exception>
    TextBindingSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>Puts <paramref name="lines"/> into <paramref name="regions"/> with <paramref name="settings"/>, its choices drawn from <paramref name="seed"/>.</summary>
    BoundText Bind(IReadOnlyList<string> lines, IReadOnlyList<ShapeRegion> regions, TextBindingSettings settings, int seed);
}

/// <summary>
/// The text bindings, by name (ignoring case). Built in (Idrak.Imaging): <c>sequence</c> (each line fills one group of
/// regions in reading order, a unit a region unless <c>take</c> says otherwise; <c>blank</c> and <c>blank-at</c> leave
/// regions empty).
/// </summary>
public static class TextBindings
{
    /// <summary>The binding that fills a group of regions per line, in reading order.</summary>
    public const string Sequence = "sequence";
}
