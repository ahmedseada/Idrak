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
}

/// <summary>
/// A kind of model name ("store:qwen3:8b", "owner/name", a folder …) that loading a model resolves (Idrak's
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
/// The sources a model name is resolved by, in order. Idrak registers an existing folder, "store:name" (the
/// local model store) and a .gguf file; Idrak.Data adds a Hugging Face id ("owner/name"). A name goes to the first source that can
/// resolve it, the most recently registered first, so a new source (for example a "myhub:" prefix) is asked before the
/// built-in ones.
/// </summary>
public static class ModelSources
{
    private static readonly SlotTable<string, IModelSource> Table = new(nameof(ModelSources), (slot, app, library) => new GuardedSource(slot, app, library),
        StringComparer.Ordinal, newestFirst: true);

    static ModelSources() => LibraryDefaults.Ensure(typeof(ModelSources));

    /// <summary>
    /// Registers <paramref name="source"/>: it takes the place of the source of the same name (the library's stays behind
    /// it, see <see cref="SetPolicy"/>), or is asked before every source registered so far.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IModelSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        Table.Register(source.Name, source, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's source <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered source names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The source registered as <paramref name="name"/>.</summary>
    public static IModelSource Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"No model source '{name}' is registered ({string.Join(", ", Table.Keys)}); add it with ModelSources.Register.");

    /// <summary>The first source that resolves <paramref name="model"/> (the most recently registered first), or null.</summary>
    public static IModelSource? For(string model) => Table.Values.FirstOrDefault(s => s.CanResolve(model));   // sources look at the disk

    /// <summary>The library's source <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IModelSource? Default(string name) => Table.Default(name);

    /// <summary>Who registered the source <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's source <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's). Under <see cref="SlotPolicy.Shadow"/> only <see cref="IModelSource.CanResolve"/> is compared: resolving
    /// fetches files, which is not done twice.
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    private sealed class GuardedSource(Slot slot, IModelSource app, IModelSource library) : IModelSource
    {
        public string Name => app.Name;

        public bool CanResolve(string model) => slot.Call(() => app.CanResolve(model), () => library.CanResolve(model), Comparisons.Exact);

        public string Resolve(string model, ModelSourceOptions options) => slot.Call(() => app.Resolve(model, options), () => library.Resolve(model, options), effects: true);
    }
}
