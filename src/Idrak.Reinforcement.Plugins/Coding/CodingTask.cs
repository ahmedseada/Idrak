// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Text.Abstractions;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// A coding task: a prompt, the project it starts from, and the commands that decide whether it was done. On disk a task
/// is a folder with <c>task.json</c> (<c>{"prompt", "language", "base", "setup": [...], "verify": [...], "tags": [...]}</c>), a
/// <c>workspace/</c> folder with the starting files (laid over <c>base</c>, a shared project folder relative to the task),
/// optionally a <c>verify/</c> folder whose files (hidden tests) are copied in after the agent finishes, and a
/// <c>solution/</c> folder with a reference solution laid over the workspace (<see cref="CodingVerifier.CheckTaskAsync"/>
/// checks that verification fails without it and passes with it).
/// </summary>
/// <param name="Id">Name (the folder's path within the suite).</param>
/// <param name="Prompt">What the user asks for.</param>
/// <param name="Language">The task's language, as the task names it (for reports only).</param>
/// <param name="Base">A project folder copied first, or null.</param>
/// <param name="Workspace">Starting files laid over <paramref name="Base"/>, or null.</param>
/// <param name="VerifyFiles">Files copied in before verification, or null.</param>
/// <param name="Setup">Commands run before the agent starts (a restore, an install), not shown to it.</param>
/// <param name="Verify">Commands that must all exit with 0 for the task to count as done.</param>
/// <param name="Tags">Free-form labels.</param>
/// <param name="Solution">A reference solution laid over the workspace, or null.</param>
public sealed record CodingTask(string Id, string Prompt, string? Language = null, string? Base = null, string? Workspace = null, string? VerifyFiles = null,
    IReadOnlyList<string>? Setup = null, IReadOnlyList<string>? Verify = null, IReadOnlyList<string>? Tags = null, string? Solution = null)
{
    /// <summary>The task in <paramref name="folder"/> (its <c>task.json</c>), named <paramref name="id"/> (null: its "id", else the folder's name).</summary>
    /// <exception cref="InvalidDataException">task.json is not an object, or has no prompt.</exception>
    /// <exception cref="DirectoryNotFoundException">Its base folder does not exist.</exception>
    public static CodingTask Load(string folder, string? id = null)
    {
        ArgumentNullException.ThrowIfNull(folder);
        folder = Path.GetFullPath(folder);
        var json = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "task.json"))) as JsonObject
                   ?? throw new InvalidDataException($"{folder}/task.json is not a JSON object.");
        static IReadOnlyList<string>? List(JsonNode? node) => node is JsonArray a ? [.. a.Select(n => (string?)n ?? "")] : null;
        string? Folder(string? path) => path is not null && Directory.Exists(Path.Combine(folder, path)) ? Path.GetFullPath(Path.Combine(folder, path)) : null;
        string? baseFolder = (string?)json["base"];
        if (baseFolder is not null && Folder(baseFolder) is null)
        {
            throw new DirectoryNotFoundException($"{folder}/task.json: base folder '{baseFolder}' does not exist.");
        }

        return new CodingTask(id ?? (string?)json["id"] ?? Path.GetFileName(folder),
            (string?)json["prompt"] ?? throw new InvalidDataException($"{folder}/task.json has no prompt."),
            (string?)json["language"], Folder(baseFolder), Folder("workspace"), Folder("verify"),
            List(json["setup"]), List(json["verify"]), List(json["tags"]), Folder("solution"));
    }
}

/// <summary>A task suite: every folder under a suite folder that holds a <c>task.json</c> (<see cref="CodingTask"/>).</summary>
public static class CodingSuite
{
    /// <summary>
    /// Every task under <paramref name="folder"/>, ordered by id (its folder's path in the suite, with '/'); <c>.git</c> and
    /// the coding languages' build output and dependency folders (every registered language's when
    /// <paramref name="languages"/> is null) are not searched. With <paramref name="filter"/>, only the tasks whose id holds
    /// it (ignoring case).
    /// </summary>
    public static IReadOnlyList<CodingTask> Load(string folder, string? filter = null, IReadOnlyList<ICodingLanguage>? languages = null)
    {
        ArgumentNullException.ThrowIfNull(folder);
        folder = Path.GetFullPath(folder);
        var skipped = new HashSet<string>([".git", .. (languages ?? CodingLanguages.Registered).SelectMany(l => l.IgnoredDirectories.Concat(l.LinkedDirectories))],
            StringComparer.OrdinalIgnoreCase);
        var found = new List<string>();
        var pending = new Stack<string>();
        pending.Push(folder);
        while (pending.Count > 0)
        {
            string current = pending.Pop();
            if (File.Exists(Path.Combine(current, "task.json")))
            {
                found.Add(current);
            }

            foreach (var child in Directory.EnumerateDirectories(current))
            {
                if (!skipped.Contains(Path.GetFileName(child)))
                {
                    pending.Push(child);
                }
            }
        }

        return [.. found
            .Select(f => CodingTask.Load(f, Path.GetRelativePath(folder, f).Replace('\\', '/')))
            .Where(t => filter is null || t.Id.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(t => t.Id, StringComparer.Ordinal)];
    }
}
