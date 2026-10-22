// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Text.Abstractions;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// The folders a coding task runs in: a fresh copy of its project (<see cref="Prepare"/>), the hidden files and a
/// solution laid over it (<see cref="Overlay"/>), its starting state committed for <c>git diff</c>
/// (<see cref="CommitStart"/>), a prepared copy cloned for each episode (<see cref="Clone"/>), and the files the agent
/// changed where the hidden tests go (<see cref="TestEdits"/>) or that a glob protects (<see cref="ProtectedEdits"/>).
/// What is build output, a dependency folder or a git exclude comes from the coding languages given
/// (<see cref="CodingLanguages"/>; every registered one when none are given): no language is named here.
/// </summary>
public static class CodingWorkspace
{
    // Build output, dependencies and tool state of the languages: not copied from a task's folders (a dependency folder is linked instead).
    private static HashSet<string> NotCopied(IReadOnlyList<ICodingLanguage> languages) =>
        new([".git", .. languages.SelectMany(l => l.IgnoredDirectories), .. languages.SelectMany(l => l.LinkedDirectories)], StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Fills <paramref name="workFolder"/> (created; it must be empty) with the task's project: <see cref="CodingTask.Base"/>,
    /// then <see cref="CodingTask.Workspace"/> over it. The languages' build output is not copied; their dependency
    /// folders (<see cref="ICodingLanguage.LinkedDirectories"/>) are linked, not copied (copied where links are not allowed).
    /// </summary>
    /// <param name="task">The task.</param>
    /// <param name="workFolder">The folder to fill.</param>
    /// <param name="languages">The coding languages whose folders are skipped and linked; null for every registered one.</param>
    /// <exception cref="IOException">The work folder is not empty.</exception>
    public static void Prepare(CodingTask task, string workFolder, IReadOnlyList<ICodingLanguage>? languages = null)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(workFolder);
        if (Directory.Exists(workFolder) && Directory.EnumerateFileSystemEntries(workFolder).Any())
        {
            throw new IOException($"The work folder '{workFolder}' is not empty.");
        }

        Directory.CreateDirectory(workFolder);
        languages ??= CodingLanguages.Registered;
        var skipped = NotCopied(languages);
        var linked = CodingLanguages.Merge([.. languages.Select(l => l.LinkedDirectories)], StringComparer.OrdinalIgnoreCase);
        foreach (var source in new[] { task.Base, task.Workspace })
        {
            if (source is null)
            {
                continue;
            }

            CopyTree(source, workFolder, overwrite: true, skipped);
            foreach (var name in linked)
            {
                string dependencies = Path.Combine(source, name), target = Path.Combine(workFolder, name);
                if (Directory.Exists(dependencies) && !Directory.Exists(target))
                {
                    try
                    {
                        Directory.CreateSymbolicLink(target, dependencies);
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        CopyTree(dependencies, target, overwrite: false, null);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Copies the files of <paramref name="source"/> over <paramref name="workFolder"/> (replacing files of the same path;
    /// the languages' build output not copied; every registered language's when <paramref name="languages"/> is null): hidden tests, a solution.
    /// </summary>
    public static void Overlay(string source, string workFolder, IReadOnlyList<ICodingLanguage>? languages = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(workFolder);
        CopyTree(source, workFolder, overwrite: true, NotCopied(languages ?? CodingLanguages.Registered));
    }

    /// <summary>
    /// Makes <paramref name="folder"/> a git repository with its files committed as "start", so the agent can review its
    /// changes with <c>git diff</c>, the languages' git excludes (<see cref="ICodingLanguage.GitExcludes"/>; every
    /// registered language's when <paramref name="languages"/> is null) left out; best effort (nothing happens without git,
    /// or when it is a repository already).
    /// </summary>
    public static void CommitStart(string folder, IReadOnlyList<ICodingLanguage>? languages = null)
    {
        ArgumentNullException.ThrowIfNull(folder);
        if (Directory.Exists(Path.Combine(folder, ".git")))
        {
            return;
        }

        var excludes = CodingLanguages.Merge([.. (languages ?? CodingLanguages.Registered).Select(l => l.GitExcludes)]);
        try
        {
            Git(folder, "init", "-q");
            if (excludes.Count > 0)
            {
                File.AppendAllText(Path.Combine(folder, ".git", "info", "exclude"), "\n" + string.Join("\n", excludes) + "\n");
            }

            Git(folder, "add", "-A");
            Git(folder, "-c", "user.name=agent", "-c", "user.email=agent@localhost", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "start");
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
        }
    }

    /// <summary>
    /// Copies everything in <paramref name="prepared"/> (a workspace after its setup: build output, the git repository and
    /// all) into <paramref name="workFolder"/> (created; it must be empty); a linked folder is linked again, not copied.
    /// </summary>
    /// <exception cref="IOException">The work folder is not empty.</exception>
    public static void Clone(string prepared, string workFolder)
    {
        ArgumentNullException.ThrowIfNull(prepared);
        ArgumentNullException.ThrowIfNull(workFolder);
        if (Directory.Exists(workFolder) && Directory.EnumerateFileSystemEntries(workFolder).Any())
        {
            throw new IOException($"The work folder '{workFolder}' is not empty.");
        }

        CloneTree(prepared, workFolder);
    }

    /// <summary>Deletes <paramref name="folder"/> and everything in it (read-only files too, such as git's objects); nothing when it does not exist.</summary>
    public static void Delete(string folder)
    {
        ArgumentNullException.ThrowIfNull(folder);
        if (!Directory.Exists(folder))
        {
            return;
        }

        foreach (var file in Directory.EnumerateFiles(folder, "*", new EnumerationOptions { RecurseSubdirectories = true, AttributesToSkip = FileAttributes.ReparsePoint, IgnoreInaccessible = true }))
        {
            File.SetAttributes(file, FileAttributes.Normal);
        }

        Directory.Delete(folder, recursive: true);
    }

    /// <summary>
    /// The paths of the task's hidden files (<see cref="CodingTask.VerifyFiles"/>), relative to the workspace with '/', in
    /// order (the languages' build output left out; every registered language's when <paramref name="languages"/> is null); none without them.
    /// </summary>
    public static IReadOnlyList<string> HiddenFiles(CodingTask task, IReadOnlyList<ICodingLanguage>? languages = null)
    {
        ArgumentNullException.ThrowIfNull(task);
        if (task.VerifyFiles is not { } hidden)
        {
            return [];
        }

        var files = new List<string>();
        Collect(hidden, hidden, files, NotCopied(languages ?? CodingLanguages.Registered));
        files.Sort(StringComparer.Ordinal);
        return files;
    }

    /// <summary>
    /// The files the agent changed where the task's hidden tests go: a hidden file's path, or any file in a folder of the
    /// hidden files other than the workspace's top, that <paramref name="workspace"/> holds with other content than
    /// <paramref name="start"/> (the workspace before the agent), adds, or removes. Relative paths with '/', in order.
    /// </summary>
    public static IReadOnlyList<string> TestEdits(CodingTask task, string start, string workspace, IReadOnlyList<ICodingLanguage>? languages = null)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(start);
        ArgumentNullException.ThrowIfNull(workspace);
        var hidden = HiddenFiles(task, languages);
        if (hidden.Count == 0)
        {
            return [];
        }

        var folders = hidden.Select(f => Path.GetDirectoryName(f)?.Replace('\\', '/') ?? "").Where(d => d.Length > 0).Distinct(StringComparer.Ordinal).ToList();
        var candidates = new SortedSet<string>(hidden, StringComparer.Ordinal);
        foreach (var root in new[] { start, workspace })
        {
            foreach (var folder in folders)
            {
                string full = Path.Combine(root, folder);
                if (Directory.Exists(full))
                {
                    foreach (var file in Directory.EnumerateFiles(full, "*", SearchOption.AllDirectories))
                    {
                        candidates.Add(Path.GetRelativePath(root, file).Replace('\\', '/'));
                    }
                }
            }
        }

        return [.. candidates.Where(path => Differs(Path.Combine(start, path), Path.Combine(workspace, path)))];
    }

    /// <summary>
    /// The files matching one of <paramref name="globs"/> (paths relative to the workspace with '/'; <c>*</c> within a
    /// folder, <c>**</c> across folders, as <see cref="DataFiles.Glob"/>) that <paramref name="workspace"/> holds with other
    /// content than <paramref name="start"/>, adds, or removes: the agent's edits to files a task protects (test projects,
    /// the build's shared files). The languages' build output and <c>.git</c> are not looked in (every registered
    /// language's when <paramref name="languages"/> is null). Relative paths with '/', in order; none without globs.
    /// </summary>
    public static IReadOnlyList<string> ProtectedEdits(string start, string workspace, IReadOnlyList<string> globs, IReadOnlyList<ICodingLanguage>? languages = null)
    {
        ArgumentNullException.ThrowIfNull(start);
        ArgumentNullException.ThrowIfNull(workspace);
        ArgumentNullException.ThrowIfNull(globs);
        if (globs.Count == 0)
        {
            return [];
        }

        var patterns = globs.Select(DataFiles.Glob).ToList();
        var skipped = NotCopied(languages ?? CodingLanguages.Registered);
        var candidates = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var root in new[] { start, workspace })
        {
            var files = new List<string>();
            if (Directory.Exists(root))
            {
                Collect(root, root, files, skipped);
            }

            candidates.UnionWith(files.Where(f => patterns.Any(p => p.IsMatch(f))));
        }

        return [.. candidates.Where(path => Differs(Path.Combine(start, path), Path.Combine(workspace, path)))];
    }

    // Whether two files differ: one exists and the other not, or their bytes differ.
    private static bool Differs(string a, string b)
    {
        bool hasA = File.Exists(a), hasB = File.Exists(b);
        if (hasA != hasB)
        {
            return true;
        }

        if (!hasA)
        {
            return false;
        }

        var (left, right) = (new FileInfo(a), new FileInfo(b));
        return left.Length != right.Length || !File.ReadAllBytes(a).AsSpan().SequenceEqual(File.ReadAllBytes(b));
    }

    private static void Collect(string root, string folder, List<string> files, HashSet<string> skipped)
    {
        foreach (var file in Directory.EnumerateFiles(folder))
        {
            files.Add(Path.GetRelativePath(root, file).Replace('\\', '/'));
        }

        foreach (var child in Directory.EnumerateDirectories(folder))
        {
            if (!skipped.Contains(Path.GetFileName(child)) && new DirectoryInfo(child).LinkTarget is null)
            {
                Collect(root, child, files, skipped);
            }
        }
    }

    // Copies a tree; folders named in `skipped` are not copied (null: every folder is).
    private static void CopyTree(string source, string target, bool overwrite, HashSet<string>? skipped)
    {
        Directory.CreateDirectory(target);
        foreach (var file in Directory.EnumerateFiles(source))
        {
            File.Copy(file, Path.Combine(target, Path.GetFileName(file)), overwrite);
        }

        foreach (var folder in Directory.EnumerateDirectories(source))
        {
            if (skipped is null || !skipped.Contains(Path.GetFileName(folder)))
            {
                CopyTree(folder, Path.Combine(target, Path.GetFileName(folder)), overwrite, skipped);
            }
        }
    }

    private static void CloneTree(string source, string target)
    {
        Directory.CreateDirectory(target);
        foreach (var file in Directory.EnumerateFiles(source))
        {
            File.Copy(file, Path.Combine(target, Path.GetFileName(file)));
        }

        foreach (var folder in Directory.EnumerateDirectories(source))
        {
            string to = Path.Combine(target, Path.GetFileName(folder));
            if (new DirectoryInfo(folder).LinkTarget is { } link)
            {
                Directory.CreateSymbolicLink(to, link);
            }
            else
            {
                CloneTree(folder, to);
            }
        }
    }

    private static void Git(string folder, params string[] arguments)
    {
        var start = new ProcessStartInfo("git") { WorkingDirectory = folder, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("git did not start.");
        process.StandardOutput.ReadToEnd();
        process.StandardError.ReadToEnd();
        process.WaitForExit();
    }
}
