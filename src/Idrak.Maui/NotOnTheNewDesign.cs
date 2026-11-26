// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui;

/// <summary>
/// What is not on the new design yet (a list or a choice by name, until it is rebuilt on real interfaces): asking for it
/// fails, and the code that used what it gave stays as it was (as Idrak's own).
/// </summary>
internal static class NotOnTheNewDesign
{
    /// <summary>Throws: <paramref name="what"/> is not on the new design yet.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public static T Yet<T>(string what) => throw new NotSupportedException($"{what} is not on the new design yet.");

    /// <summary>Throws: the names listed (<paramref name="what"/>) are not on the new design yet.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public static IReadOnlyCollection<string> Names(string what) => Yet<IReadOnlyCollection<string>>($"Listing the {what}");
}
