// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

/// <summary>Settings a model source receives with the name it resolves (Idrak.LanguageModels' <c>ModelSource.Resolve</c> passes them).</summary>
public sealed record ModelSourceOptions
{
    /// <summary>The revision (branch, tag or commit) of a hub model.</summary>
    public string Revision { get; init; } = "main";

    /// <summary>The access token for gated and private models (null: HF_TOKEN or the saved login).</summary>
    public string? Token { get; init; }


    /// <summary>Whether a model that is not cached may be downloaded (false: only the caches are searched).</summary>
    public bool Download { get; init; } = true;

    /// <summary>
    /// What the application offers sources beyond these settings, by type (null: nothing). The built-in sources ask it for
    /// the downloader that fetches, caches and logs (Idrak.Datasets' <c>Downloader</c>; the shared one when it offers
    /// none); a source of your own may ask it for anything the application put there.
    /// </summary>
    public IServiceProvider? Services { get; init; }
}

/// <summary>
/// A kind of model name ("store:qwen3:8b", "owner/name", a folder …) that loading a model resolves (Idrak.LanguageModels'
/// <c>ModelSource.Resolve</c> asks the registered sources): it says which names are its own and turns one into a local
/// folder (or a path a checkpoint format reads, such as a .gguf file's prepared folder). Register new ones with
/// <see cref="ModelSources.Register"/>.
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

/// <summary>
/// The sources a model name is resolved by, in order. Idrak.LanguageModels registers an existing folder, "store:name" (the
/// local model store), a .gguf file, then a Hugging Face id ("owner/name"). A name goes to the first source that can
/// resolve it, the most recently registered first, so a new source (for example a "myhub:" prefix) is asked before the
/// built-in ones.
/// </summary>
public static class ModelSources
{
    private static readonly List<IModelSource> Registry = [];

    static ModelSources() => LibraryDefaults.Ensure();

    /// <summary>
    /// Registers <paramref name="source"/>: it replaces the source of the same name (in its place), or is asked before
    /// every source registered so far.
    /// </summary>
    public static void Register(IModelSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        lock (Registry)
        {
            int at = Registry.FindIndex(s => s.Name == source.Name);
            if (at >= 0)
            {
                Registry[at] = source;
            }
            else
            {
                Registry.Insert(0, source);
            }
        }
    }

    /// <summary>Removes the source registered as <paramref name="name"/>; false when there is none.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.RemoveAll(s => s.Name == name) > 0;
        }
    }

    /// <summary>The registered source names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Select(s => s.Name)];
            }
        }
    }

    /// <summary>The source registered as <paramref name="name"/>.</summary>
    public static IModelSource Get(string name)
    {
        lock (Registry)
        {
            return Registry.Find(s => s.Name == name)
                ?? throw new NotSupportedException($"No model source '{name}' is registered ({string.Join(", ", Registry.Select(s => s.Name))}); add it with ModelSources.Register.");
        }
    }

    /// <summary>The source that resolves <paramref name="model"/>, or null when none does.</summary>
    public static IModelSource? For(string model)
    {
        IModelSource[] sources;
        lock (Registry)
        {
            sources = [.. Registry];
        }

        return sources.FirstOrDefault(s => s.CanResolve(model));                // asked outside the lock: sources look at the disk
    }
}
