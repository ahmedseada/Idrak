// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Generation;

namespace Idrak.Inference;

/// <summary>The text kind of the inference engine: continues prompts with <see cref="TextGenerator"/> copies, one generation per copy at a time.</summary>
internal sealed class TextEngineModel(GenerativeModelBuilder settings, Func<Device?, TextGenerator> load, bool owns, bool reloadable)
    : EngineModel<TextGenerator>(KindName, settings.Hosting(reloadable)), ITextModel
{
    /// <summary>The kind's name in the engine's status.</summary>
    public const string KindName = "text";

    public override TextGenerator LoadCopy() => settings.Load(load);

    public override void UnloadCopy(TextGenerator copy)
    {
        if (owns)
        {
            copy.Model.Dispose();
        }
    }

    public override ModelDescription Describe(TextGenerator copy) => Describe(Name, Kind, copy);

    public IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        StreamText(Host, copy => copy, prompt, options, cancellationToken);

    internal static ModelDescription Describe(string name, string kind, TextGenerator generator) =>
        new(name, kind, generator.Model.ParameterCount, generator.Device, generator.ContextLength);

    /// <summary>One text generation on a free copy, counted in the engine's statistics when it ends.</summary>
    internal static async IAsyncEnumerable<GenerationChunk> StreamText<TCopy>(IEngineHost<TCopy> host, Func<TCopy, TextGenerator> generator, string prompt,
        GenerationOptions options, [EnumeratorCancellation] CancellationToken cancellationToken)
        where TCopy : class
    {
        using var lease = await host.AcquireAsync(cancellationToken).ConfigureAwait(false);
        await foreach (var chunk in lease.Guard(generator(lease.Copy).StreamAsync(prompt, options, lease.Token)).ConfigureAwait(false))
        {
            if (chunk.Done)
            {
                lease.Completed(1, chunk.Stats!.GeneratedTokens, chunk.Stats.GenerationDuration);
            }

            yield return chunk;
        }
    }
}
