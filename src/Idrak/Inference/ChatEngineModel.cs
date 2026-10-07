// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Generation;

namespace Idrak.Inference;

/// <summary>
/// The chat kind of the inference engine: answers chat requests with <see cref="ChatGenerator"/> copies (and continues
/// raw prompts with their text generators), one generation per copy at a time.
/// </summary>
internal sealed class ChatEngineModel(GenerativeModelBuilder settings, Func<Device?, TextGenerator> load, bool owns, bool reloadable)
    : EngineModel<ChatGenerator>(KindName, settings.Hosting(reloadable)), IChatModel, ITextModel
{
    /// <summary>The kind's name in the engine's status.</summary>
    public const string KindName = "chat";

    /// <summary>The tools registered with the model, or null.</summary>
    public ToolRegistry? Tools => settings.ToolRegistry;

    public override ChatGenerator LoadCopy() => new(settings.Load(load), settings.ChatTemplate);

    public override void UnloadCopy(ChatGenerator copy)
    {
        if (owns)
        {
            copy.Generator.Model.Dispose();
        }
    }

    public override ModelDescription Describe(ChatGenerator copy) => TextEngineModel.Describe(Name, Kind, copy.Generator);

    public async IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
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
