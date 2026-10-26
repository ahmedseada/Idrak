// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Microsoft.Maui.Controls.Shapes;

namespace Idrak.Maui.Theme;

/// <summary>The icons as path geometries (see <see cref="IdrakIcons"/>), merged into <see cref="IdrakStyles"/>.</summary>
public partial class IdrakIconDictionary : ResourceDictionary
{
    /// <summary>The icons.</summary>
    public IdrakIconDictionary() => InitializeComponent();
}

/// <summary>
/// Idrak's icons by name ("home", "chat", "mic"): the approved design's 24-unit stroke icons (Theme/IdrakIcons.xaml, keys
/// "IdrakIcon" and the name with its first letter capitalised). Code that binds an icon by name takes its geometry here;
/// XAML names the key directly ({DynamicResource IdrakIconHome}).
/// </summary>
public static class IdrakIcons
{
    private static readonly Lazy<IdrakIconDictionary> Dictionary = new(() => new IdrakIconDictionary());

    /// <summary>The resource key of the icon <paramref name="name"/> ("home" → "IdrakIconHome").</summary>
    public static string Key(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        return string.Create(9 + name.Length, name, static (span, n) =>
        {
            "IdrakIcon".CopyTo(span);
            n.AsSpan().CopyTo(span[9..]);
            span[9] = char.ToUpperInvariant(span[9]);
        });
    }

    /// <summary>The names of every icon, in the dictionary's order.</summary>
    public static IReadOnlyList<string> Names => [.. Dictionary.Value.Keys.Select(k => char.ToLowerInvariant(k[9]) + k[10..])];

    /// <summary>The geometry of the icon <paramref name="name"/>, or null when there is none (an empty name included).</summary>
    public static Geometry? Get(string? name) =>
        string.IsNullOrWhiteSpace(name) ? null : Dictionary.Value.TryGetValue(Key(name), out var value) ? value as Geometry : null;
}
