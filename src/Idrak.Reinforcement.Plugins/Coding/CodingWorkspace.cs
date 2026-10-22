// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// The folders a coding task runs in: a fresh copy of its project (<see cref="Prepare"/>), the hidden files and a
/// solution laid over it (<see cref="Overlay"/>), its starting state committed for <c>git diff</c>
/// (<see cref="CommitStart"/>), a prepared copy cloned for each episode (<see cref="Clone"/>), and the files the agent
/// changed where the hidden tests go (<see cref="TestEdits"/>).
/// </summary>
public static class CodingWorkspace
{
    // Build output, dependencies and tool state: not copied from a task's folders (a dependency folder is linked instead).
    private static readonly HashSet<string> BuildOutput = new(["bin", "obj", "node_modules", ".angular", "dist", ".vs", ".git"], StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Fills <paramref name="workFolder"/> (created; it must be empty) with the task's project: <see cref="CodingTask.Base"/>,
    /// then <see cref="CodingTask.Workspace"/> over it. Build output is not copied; a node_modules folder is linked, not
    /// copied (copied where links are not allowed).
    /// </summary>
    /// <exception cref="IOException">The work folder is not empty.</exception>
    public static void Prepare(CodingTask task, string workFolder)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(workFolder);
        if (Directory.Exists(workFolder) && Directory.EnumerateFileSystemEntries(workFolder).Any())
        {
            throw new IOException($"The work folder '{workFolder}' is not empty.");
        }

        Directory.CreateDirectory(workFolder);
        foreach (var source in new[] { task.Base, task.Workspace })
        {
            if (source is null)
            {
                continue;
            }

            CopyTree(source, workFolder, overwrite: true);
            string modules = Path.Combine(source, "node_modules"), target = Path.Combine(workFolder, "node_modules");
            if (Directory.Exists(modules) && !Directory.Exists(target))
            {
                try
                {
                    Directory.CreateSymbolicLink(target, modules);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    CopyTree(modules, target, overwrite: false, skipBuildOutput: false);
                }
            }
        }
    }

    /// <summary>Copies the files of <paramref name="source"/> over <paramref name="workFolder"/> (replacing files of the same path; build output not copied): hidden tests, a solution.</summary>
    public static void Overlay(string source, string workFolder)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(workFolder);
        CopyTree(source, workFolder, overwrite: true);
    }

    /// <summary>
    /// Makes <paramref name="folder"/> a git repository with its files committed as "start", so the agent can review its
    /// changes with <c>git diff</c>; best effort (nothing happens without git, or when it is a repository already).
    /// </summary>
    public static void CommitStart(string folder)
    {
        ArgumentNullException.ThrowIfNull(folder);
        if (Directory.Exists(Path.Combine(folder, ".git")))
        {
            return;
        }

        try
        {
            Git(folder, "init", "-q");
            File.AppendAllText(Path.Combine(folder, ".git", "info", "exclude"), "\nbin/\nobj/\nnode_modules\n.angular/\ndist/\n");
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

    /// <summary>The paths of the task's hidden files (<see cref="CodingTask.VerifyFiles"/>), relative to the workspace with '/', in order; none without them.</summary>
    public static IReadOnlyList<string> HiddenFiles(CodingTask task)
    {
        ArgumentNullException.ThrowIfNull(task);
        if (task.VerifyFiles is not { } hidden)
        {
            return [];
        }

        var files = new List<string>();
        Collect(hidden, hidden, files);
        files.Sort(StringComparer.Ordinal);
        return files;
    }

    /// <summary>
    /// The files the agent changed where the task's hidden tests go: a hidden file's path, or any file in a folder of the
    /// hidden files other than the workspace's top, that <paramref name="workspace"/> holds with other content than
    /// <paramref name="start"/> (the workspace before the agent), adds, or removes. Relative paths with '/', in order.
    /// </summary>
    public static IReadOnlyList<string> TestEdits(CodingTask task, string start, string workspace)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(start);
        ArgumentNullException.ThrowIfNull(workspace);
        var hidden = HiddenFiles(task);
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

    private static void Collect(string root, string folder, List<string> files)
    {
        foreach (var file in Directory.EnumerateFiles(folder))
        {
            files.Add(Path.GetRelativePath(root, file).Replace('\\', '/'));
        }

        foreach (var child in Directory.EnumerateDirectories(folder))
        {
            if (!BuildOutput.Contains(Path.GetFileName(child)))
            {
                Collect(root, child, files);
            }
        }
    }

    private static void CopyTree(string source, string target, bool overwrite, bool skipBuildOutput = true)
    {
        Directory.CreateDirectory(target);
        foreach (var file in Directory.EnumerateFiles(source))
        {
            File.Copy(file, Path.Combine(target, Path.GetFileName(file)), overwrite);
        }

        foreach (var folder in Directory.EnumerateDirectories(source))
        {
            if (!skipBuildOutput || !BuildOutput.Contains(Path.GetFileName(folder)))
            {
                CopyTree(folder, Path.Combine(target, Path.GetFileName(folder)), overwrite, skipBuildOutput);
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
