// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Abstractions;
using Idrak.Text.Abstractions;

namespace Idrak.Cli.Shared;

/// <summary>
/// The folders the data commands skip in code folders and archives: Idrak.Data's own (<see cref="ReadOptions.DefaultSkippedFolders"/>:
/// .git and the editors') and every registered coding language's build output and dependency folders
/// (<see cref="CodingLanguages"/>; the tool registers Idrak.Text.Plugins' at startup), which Idrak.Data does not know.
/// </summary>
internal static class CodeFolders
{
    /// <summary>The folders skipped, in order.</summary>
    public static IReadOnlyList<string> Skipped =>
        CodingLanguages.Merge([ReadOptions.DefaultSkippedFolders, .. CodingLanguages.Registered.Select(l => l.IgnoredDirectories), .. CodingLanguages.Registered.Select(l => l.LinkedDirectories)],
            StringComparer.OrdinalIgnoreCase);

    /// <summary>The read options of a file or folder read by the data commands: the defaults, skipping <see cref="Skipped"/>.</summary>
    public static ReadOptions Read => new() { SkippedFolders = Skipped };
}
