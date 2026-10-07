// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>
/// The readers of a model folder's own chat template, which loading a pretrained model asks (Idrak's
/// <c>PretrainedModel.Load</c>). Idrak.Nlp registers "jinja": the Jinja template of tokenizer_config.json,
/// chat_template.jinja or chat_template.json, rendered as Hugging Face renders it. Without a reader, a loaded model has no
/// chat template. Add a reader of your own with <see cref="Register"/>.
/// </summary>
public static class ChatTemplates
{
    private sealed record Entry(string Name, Func<string, ITokenizer?, ChatTemplate?> Load);

    private static readonly List<Entry> Registry = [];

    static ChatTemplates() => LibraryDefaults.Ensure(typeof(ChatTemplates));

    /// <summary>
    /// Registers the reader <paramref name="name"/>: <paramref name="load"/> reads the chat template of a model folder
    /// (given the model's tokenizer, or null), or returns null when the folder holds none it reads. It replaces the reader
    /// of the same name (in its place), or is asked before every reader registered so far.
    /// </summary>
    public static void Register(string name, Func<string, ITokenizer?, ChatTemplate?> load)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(load);
        lock (Registry)
        {
            int at = Registry.FindIndex(e => e.Name == name);
            if (at >= 0)
            {
                Registry[at] = new Entry(name, load);
            }
            else
            {
                Registry.Insert(0, new Entry(name, load));
            }
        }
    }

    /// <summary>Removes the reader registered as <paramref name="name"/>; false when there is none.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.RemoveAll(e => e.Name == name) > 0;
        }
    }

    /// <summary>The registered reader names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Select(e => e.Name)];
            }
        }
    }

    /// <summary>
    /// The chat template of the model in <paramref name="folder"/>: the first one a registered reader returns (the most
    /// recently registered first), or null when none reads one.
    /// </summary>
    public static ChatTemplate? Load(string folder, ITokenizer? tokenizer = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        Entry[] entries;
        lock (Registry)
        {
            entries = [.. Registry];
        }

        // Asked outside the lock: a reader looks at the disk.
        foreach (var entry in entries)
        {
            if (entry.Load(folder, tokenizer) is { } template)
            {
                return template;
            }
        }

        return null;
    }
}
