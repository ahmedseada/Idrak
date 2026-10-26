// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;

namespace Idrak.Maui;

/// <summary>
/// The source an assembly was built from: the git branch and commit read at build time (Idrak.Maui.csproj's
/// IdrakSourceCommit target writes them as <see cref="AssemblyMetadataAttribute"/>s "IdrakSourceBranch" and
/// "IdrakSourceCommit"; "unknown" when git was not there, and -p:IdrakSourceCommit=… sets them by hand). Home shows it
/// ("main · 628b6bb") in place of a version.
/// </summary>
/// <param name="Branch">The branch ("main"), or "" when detached or unknown.</param>
/// <param name="Commit">The commit's short hash ("628b6bb"), or "unknown".</param>
public sealed record SourceInfo(string Branch, string Commit)
{
    /// <summary>The metadata key of the commit.</summary>
    public const string CommitKey = "IdrakSourceCommit";

    /// <summary>The metadata key of the branch.</summary>
    public const string BranchKey = "IdrakSourceBranch";

    /// <summary>Idrak.Maui's own source.</summary>
    public static SourceInfo Library { get; } = Of(typeof(SourceInfo).Assembly);

    /// <summary>Whether the commit is known.</summary>
    public bool IsKnown => Commit.Length > 0 && Commit != "unknown";

    /// <summary>"main · 628b6bb", or the commit alone when the branch is not known.</summary>
    public string Text => Branch.Length > 0 ? $"{Branch} · {Commit}" : Commit;

    /// <summary>The source <paramref name="assembly"/> was built from ("unknown" when it does not say).</summary>
    public static SourceInfo Of(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        string? Read(string key) => assembly.GetCustomAttributes<AssemblyMetadataAttribute>().FirstOrDefault(a => a.Key == key)?.Value?.Trim();
        string branch = Read(BranchKey) ?? "";
        return new SourceInfo(branch is "HEAD" or "unknown" ? "" : branch, Read(CommitKey) is { Length: > 0 } commit ? commit : "unknown");
    }
}
