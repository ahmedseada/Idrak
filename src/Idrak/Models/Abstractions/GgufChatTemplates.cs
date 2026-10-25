// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// A chat template for GGUF files that carry none (no <c>tokenizer.chat_template</c>): the Jinja template written into the
/// prepared folder, and which files it is for, told by their tokens. Registered by name in <see cref="GgufChatTemplates"/>.
/// </summary>
public interface IGgufChatTemplate
{
    /// <summary>
    /// Whether a file without a template gets this one: the file and its <c>tokenizer.ggml.tokens</c> (a template is for
    /// the files whose vocabulary holds the tokens it writes).
    /// </summary>
    bool Matches(GgufFile file, IReadOnlyList<string> tokens);

    /// <summary>The Jinja chat template, as tokenizer_config.json's <c>chat_template</c> holds one.</summary>
    string Template { get; }
}

/// <summary>A chat template for GGUF files without one (<see cref="IGgufChatTemplate"/>), from its text and a test of the file's tokens.</summary>
public sealed class GgufChatTemplate : IGgufChatTemplate
{
    /// <summary>Whether a file without a template gets this one: the file and its <c>tokenizer.ggml.tokens</c>.</summary>
    public required Func<GgufFile, IReadOnlyList<string>, bool> Matches { get; init; }

    /// <inheritdoc />
    public required string Template { get; init; }

    bool IGgufChatTemplate.Matches(GgufFile file, IReadOnlyList<string> tokens) => Matches(file, tokens);
}

/// <summary>
/// The chat templates a GGUF file without its own is given, by name, asked in the order they were registered (the
/// library's first): the first whose <see cref="IGgufChatTemplate.Matches"/> takes the file's tokens is written into the
/// prepared folder, and the prepared model notes which. The library registers "chatml" (the ChatML turns of
/// <c>&lt;|im_start|&gt;</c> and <c>&lt;|im_end|&gt;</c>, the layout Idrak's own created models use, for a file whose
/// vocabulary holds <c>&lt;|im_start|&gt;</c>); a family's own template is a plug-in's (Idrak.Text.Plugins). With none
/// matching, the prepared model has no chat template, as a model folder without one: chat is refused, plain text
/// generation works, and the note names this registry.
/// </summary>
public static class GgufChatTemplates
{
    private static readonly SlotTable<string, IGgufChatTemplate> Registry = new(nameof(GgufChatTemplates), comparer: StringComparer.Ordinal,
        unguarded: "a template is a description read once, when a file is prepared");

    static GgufChatTemplates() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterGgufChatTemplates);   // the built-in templates, on first use

    /// <summary>
    /// Registers the template <paramref name="name"/> for files without one, asked after those registered before it;
    /// under a built-in name it takes the library's place (asked where the library's was) until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, IGgufChatTemplate template)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(template);
        ArgumentException.ThrowIfNullOrEmpty(template.Template, nameof(template));
        Registry.Register(name, template, System.Reflection.Assembly.GetCallingAssembly(), name);
    }

    /// <summary>Removes the app's template <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered template names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The template registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">No template is registered under that name.</exception>
    public static IGgufChatTemplate Get(ReadOnlySpan<char> name) => Find(name)
        ?? throw new NotSupportedException($"GGUF chat template '{name}' is not registered (registered: {string.Join(", ", Names)}); add it with GgufChatTemplates.Register, "
                                          + "or reference Idrak.Text.Plugins (the families' templates) and call TextPlugins.RegisterAll().");

    /// <summary>The template registered as <paramref name="name"/>, or null when there is none.</summary>
    public static IGgufChatTemplate? Find(ReadOnlySpan<char> name) => Registry.Find(name);

    /// <summary>The library's template <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IGgufChatTemplate? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the template <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// The first registered template whose <see cref="IGgufChatTemplate.Matches"/> takes <paramref name="file"/> and its
    /// <paramref name="tokens"/>, with its name; null when none does.
    /// </summary>
    public static (string Name, IGgufChatTemplate Template)? For(GgufFile file, IReadOnlyList<string> tokens)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(tokens);
        var names = Registry.Keys;
        var templates = Registry.Values;
        for (int i = 0; i < templates.Count; i++)
        {
            if (templates[i].Matches(file, tokens))
            {
                return (names[i], templates[i]);
            }
        }

        return null;
    }
}
