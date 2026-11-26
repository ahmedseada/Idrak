// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui;

/// <summary>
/// What a registry listed or picked by name until the registries are rebuilt on real interfaces: asking for it fails, and
/// the code that used what it gave stays as it was (as Idrak's own).
/// </summary>
internal static class NotOnTheNewDesign
{
    /// <summary>Throws: <paramref name="what"/> is not on the new design yet.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public static T Yet<T>(string what) => throw new NotSupportedException($"{what} is not on the new design yet.");

    /// <summary>Throws: the names a registry listed (<paramref name="what"/>) are not on the new design yet.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public static IReadOnlyCollection<string> Names(string what) => Yet<IReadOnlyCollection<string>>($"The registered {what}");
}
