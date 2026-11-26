// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Formats;

/// <summary>Settings a model source receives with the name it resolves (Idrak's <c>ModelSource.Resolve</c> passes them).</summary>
public sealed record ModelSourceOptions
{
    /// <summary>The revision (branch, tag or commit) of a hub model.</summary>
    public string Revision { get; init; } = "main";

    /// <summary>The access token for gated and private models (null: HF_TOKEN or the saved login).</summary>
    public string? Token { get; init; }

    /// <summary>Whether a model that is not cached may be downloaded (false: only the caches are searched).</summary>
    public bool Download { get; init; } = true;

    /// <summary>
    /// What fetches, caches and logs for the source (null: the source's own; Idrak.Data's <c>Downloader.Shared</c> for the Hub); pass
    /// your own for a mirror, an authenticated proxy or an offline cache.
    /// </summary>
    public IDownloader? Downloader { get; init; }

    /// <summary>
    /// Told how a download of the model goes (Craft's status bar, 2026-11-01): the bytes received of all the files to
    /// download, and the files done of them, several times a second and once per file; null for none. A model found in a
    /// cache downloads nothing and reports nothing.
    /// </summary>
    public IProgress<ModelPullProgress>? Progress { get; init; }
}

/// <summary>How far a model's download is: the bytes received of all its files to download, the files done, the file now.</summary>
/// <param name="Model">The model being downloaded.</param>
/// <param name="Received">The bytes received so far, of every file.</param>
/// <param name="Total">The bytes of every file to download.</param>
/// <param name="FilesDone">The files done.</param>
/// <param name="Files">The files to download.</param>
/// <param name="File">The file being downloaded now ("" between files).</param>
public sealed record ModelPullProgress(string Model, long Received, long Total, int FilesDone, int Files, string File)
{
    /// <summary>The share received, from 0 to 1 (1 when there was nothing to download).</summary>
    public double Fraction => Total > 0 ? Math.Clamp((double)Received / Total, 0, 1) : 1;
}

/// <summary>
/// A kind of model name ("store:qwen3:8b", "owner/name", a folder …) that loading a model resolves (Idrak's
/// <c>ModelSource.Resolve</c> asks the registered sources): it says which names are its own and turns one into a local
/// folder (or a path a checkpoint format reads, such as a .gguf file's prepared folder). Register new ones with <c>Register</c>.
/// </summary>
public interface IModelSource
{
    /// <summary>The source's name ("folder", "store", "gguf", "huggingface", …): registering another under it replaces this one.</summary>
    string Name { get; }

    /// <summary>Whether <paramref name="model"/> is a name this source resolves.</summary>
    bool CanResolve(string model);

    /// <summary>The local folder of <paramref name="model"/> (fetched or prepared as needed).</summary>
    string Resolve(string model, ModelSourceOptions options);
}

