// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using System.Runtime.CompilerServices;
using Idrak.Generation;

namespace Idrak.Inference;

/// <summary>
/// The chat kind of the inference engine: answers chat requests with <see cref="ChatGenerator"/> copies (and continues
/// raw prompts with their text generators), one generation per copy at a time.
/// </summary>
internal sealed class ChatEngineModel(GenerativeModelBuilder settings, Func<Device?, TextGenerator> load, bool owns, bool reloadable)
    : EngineModel<ChatGenerator>(KindName, settings.Hosting(reloadable)), IToolChatModel, ITextModel
{
    /// <summary>The kind's name in the engine's status.</summary>
    public const string KindName = "chat";

    /// <summary>The tools registered with the model, or null.</summary>
    public IToolRegistry? Tools => settings.ToolRegistry;

    public override ChatGenerator LoadCopy()
    {
        var generator = settings.Load(load);
        if (settings.ImageReader is not { } images)
        {
            return new ChatGenerator(generator, settings.ChatTemplate);
        }

        try
        {
            return new ChatGenerator(generator, settings.ChatTemplate) { Images = images(generator) };
        }
        catch
        {
            if (owns)
            {
                generator.Model.Dispose();
            }

            throw;
        }
    }

    public override void UnloadCopy(ChatGenerator copy)
    {
        copy.Images?.Owner?.Dispose();
        if (owns)
        {
            copy.Generator.Model.Dispose();
        }
    }

    public override ModelDescription Describe(ChatGenerator copy) => TextEngineModel.Describe(Name, Kind, copy.Generator);

    /// <summary>
    /// The kinds of message parts it takes, known before a copy loads: text, and images when it was given
    /// <see cref="GenerativeModelBuilder.Images"/> and its chat template renders them (<see cref="ChatTemplate.PartKinds"/>).
    /// </summary>
    public IReadOnlySet<string> PartKinds => _partKinds ??= settings.ImageReader is not null
        && (settings.ChatTemplate ?? new ChatMLTemplate()).PartKinds.Contains(ChatParts.Image)
            ? FrozenSet.Create(StringComparer.Ordinal, ChatParts.Text, ChatParts.Image) : ChatParts.TextOnly;

    private IReadOnlySet<string>? _partKinds;

    public async IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ChatParts.ThrowIfUnsupported(this, request, Name);   // before a copy is taken or loaded
        using var lease = await Host.AcquireAsync(cancellationToken).ConfigureAwait(false);
        await foreach (var chunk in lease.Guard(lease.Copy.StreamAsync(request, lease.Token)).ConfigureAwait(false))
        {
            if (chunk.Done)
            {
                lease.Completed(1, chunk.Stats!.GeneratedTokens, chunk.Stats.GenerationDuration);
            }

            yield return chunk;
        }
    }

    public IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        TextEngineModel.StreamText(Host, copy => copy.Generator, prompt, options, cancellationToken);
}
