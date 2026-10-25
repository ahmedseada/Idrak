// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Generation;

/// <summary>What a chat template reader is asked with: the model folder and the model's tokenizer (or null).</summary>
/// <param name="Folder">The model folder.</param>
/// <param name="Tokenizer">The model's tokenizer, or null.</param>
public sealed record ChatTemplateRequest(string Folder, ITokenizer? Tokenizer = null);

/// <summary>
/// A reader of a model folder's own chat template, registered by its <see cref="IKind.Name"/> in
/// <see cref="ChatTemplates"/>: <see cref="IKind{TRequest, T}.Create"/> reads the template of the folder, or returns null
/// when the folder holds none it reads.
/// </summary>
public interface IChatTemplateReader : IKind<ChatTemplateRequest, IChatTemplate?>;

/// <summary>
/// The readers of a model folder's own chat template, which loading a pretrained model asks (Idrak's
/// <c>PretrainedModel.Load</c>). Idrak.Text registers "jinja": the Jinja template of tokenizer_config.json,
/// chat_template.jinja or chat_template.json, rendered as Hugging Face renders it. Without a reader, a loaded model has no
/// chat template. Add a reader of your own with <see cref="Register"/>.
/// </summary>
public static class ChatTemplates
{
    private static readonly KindTable<IChatTemplateReader> Table =
        new(nameof(ChatTemplates), "Chat template reader", "add it with ChatTemplates.Register", Guard, StringComparer.Ordinal, newestFirst: true);

    static ChatTemplates() => LibraryDefaults.Ensure(typeof(ChatTemplates));

    // Reading a folder is one call; templates agree when both readers find one of the same type, or both find none.
    private static IChatTemplateReader Guard(Slot slot, IChatTemplateReader app, IChatTemplateReader library) => new GuardedReader(slot, app, library);

    /// <summary>
    /// Registers <paramref name="reader"/> under its <see cref="IKind.Name"/>. Under a registered name it takes that
    /// reader's place (the library's stays behind it, see <see cref="SetPolicy"/>); a new name is asked before every
    /// reader registered so far.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IChatTemplateReader reader) => Table.Register(reader, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's reader <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered reader names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The library's reader <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IChatTemplateReader? Default(string name) => Table.Default(name);

    /// <summary>Who registered the reader <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's reader <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>
    /// The chat template of the model in <paramref name="folder"/>: the first one a registered reader returns (the most
    /// recently registered first), or null when none reads one.
    /// </summary>
    public static IChatTemplate? Load(string folder, ITokenizer? tokenizer = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        var request = new ChatTemplateRequest(folder, tokenizer);
        foreach (var reader in Table.Kinds)
        {
            if (reader.Create(request) is { } template)
            {
                return template;
            }
        }

        return null;
    }

    /// <summary>The prompt <paramref name="template"/> renders for the next assistant turn, as a string (see <see cref="IChatTemplate.Render"/>).</summary>
    /// <exception cref="NotSupportedException">A message part whose kind the template does not render.</exception>
    public static string Render(this IChatTemplate template, IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think)
    {
        ArgumentNullException.ThrowIfNull(template);
        var prompt = new StringBuilder();
        template.Render(messages, tools, think, prompt);
        return prompt.ToString();
    }

    private sealed class GuardedReader(Slot slot, IChatTemplateReader app, IChatTemplateReader library) : IChatTemplateReader
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public IChatTemplate? Create(ChatTemplateRequest request) => slot.Call(() => app.Create(request), () => library.Create(request),
            (a, b) => Comparisons.Exact(a?.GetType().Name ?? "none", b?.GetType().Name ?? "none"));
    }
}
