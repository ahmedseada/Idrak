// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// The readers of a model folder's own chat template, which loading a pretrained model asks (Idrak's
/// <c>PretrainedModel.Load</c>). Idrak.Nlp registers "jinja": the Jinja template of tokenizer_config.json,
/// chat_template.jinja or chat_template.json, rendered as Hugging Face renders it. Without a reader, a loaded model has no
/// chat template. Add a reader of your own with <see cref="Register"/>.
/// </summary>
public static class ChatTemplates
{
    private static readonly SlotTable<string, Func<string, ITokenizer?, ChatTemplate?>> Table =
        new(nameof(ChatTemplates), Guard, StringComparer.Ordinal, newestFirst: true);

    static ChatTemplates() => LibraryDefaults.Ensure(typeof(ChatTemplates));

    // Reading a folder is one call; templates agree when both readers find one of the same type, or both find none.
    private static Func<string, ITokenizer?, ChatTemplate?> Guard(Slot slot, Func<string, ITokenizer?, ChatTemplate?> app, Func<string, ITokenizer?, ChatTemplate?> library) =>
        (folder, tokenizer) => slot.Call(() => app(folder, tokenizer), () => library(folder, tokenizer),
            (a, b) => Comparisons.Exact(a?.GetType().Name ?? "none", b?.GetType().Name ?? "none"));

    /// <summary>
    /// Registers the reader <paramref name="name"/>: <paramref name="load"/> reads the chat template of a model folder
    /// (given the model's tokenizer, or null), or returns null when the folder holds none it reads. Under a registered
    /// name it takes that reader's place (the library's stays behind it as its fallback, see <see cref="SetPolicy"/>); a
    /// new name is asked before every reader registered so far.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, Func<string, ITokenizer?, ChatTemplate?> load)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(load);
        Table.Register(name, load, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's reader <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered reader names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The library's reader <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static Func<string, ITokenizer?, ChatTemplate?>? Default(string name) => Table.Default(name);

    /// <summary>Who registered the reader <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's reader <paramref name="name"/> fails (<see cref="SlotPolicy.FallBack"/> to the library's unless set).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>
    /// The chat template of the model in <paramref name="folder"/>: the first one a registered reader returns (the most
    /// recently registered first), or null when none reads one.
    /// </summary>
    public static ChatTemplate? Load(string folder, ITokenizer? tokenizer = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        foreach (var load in Table.Values)
        {
            if (load(folder, tokenizer) is { } template)
            {
                return template;
            }
        }

        return null;
    }
}
