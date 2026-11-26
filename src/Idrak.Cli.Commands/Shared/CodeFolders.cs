// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Abstractions;
using Idrak.Text.Abstractions;

namespace Idrak.Cli.Shared;

/// <summary>
/// The folders the data commands skip in code folders and archives: Idrak.Data's own (<see cref="ReadOptions.DefaultSkippedFolders"/>:
/// .git and the editors'). The coding languages' build output and dependency folders came from the languages
/// (<see cref="CodingLanguages"/>): listing them is not on the new design yet.
/// </summary>
internal static class CodeFolders
{
    /// <summary>The folders skipped, in order.</summary>
    public static IReadOnlyList<string> Skipped => CodingLanguages.Merge([ReadOptions.DefaultSkippedFolders], StringComparer.OrdinalIgnoreCase);

    /// <summary>The read options of a file or folder read by the data commands: the defaults, skipping <see cref="Skipped"/>.</summary>
    public static ReadOptions Read => new() { SkippedFolders = Skipped };
}
