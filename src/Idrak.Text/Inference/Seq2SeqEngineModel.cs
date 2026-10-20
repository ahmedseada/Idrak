// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Generation;
using Idrak.Inference;

namespace Idrak.Text.Inference;

/// <summary>
/// The encoder-decoder kind of the inference engine ("seq2seq"): answers prompts with <see cref="Seq2SeqGenerator"/>
/// copies, one generation per copy at a time (a copy owns its decoding cache), through <see cref="ITextModel"/>. Added
/// with <see cref="GenerativeModels.Seq2SeqModel(InferenceEngineBuilder, string, string, Func{GenerativeModelBuilder, GenerativeModelBuilder}?)"/>.
/// </summary>
internal sealed class Seq2SeqEngineModel(GenerativeModelBuilder settings, Func<Device?, Seq2SeqGenerator> load, bool owns, bool reloadable)
    : EngineModel<Seq2SeqGenerator>(KindName, settings.Hosting(reloadable)), ITextModel
{
    /// <summary>The kind's name in the engine's status.</summary>
    public const string KindName = "seq2seq";

    public override Seq2SeqGenerator LoadCopy()
    {
        var generator = load(settings.TargetDevice);
        if (settings.WarmUpPrompt is { } prompt)
        {
            generator.Generate(prompt, new GenerationOptions { NumPredict = 1 });
        }

        return generator;
    }

    public override void UnloadCopy(Seq2SeqGenerator copy)
    {
        if (owns)
        {
            copy.Model.Dispose();
        }
    }

    public override ModelDescription Describe(Seq2SeqGenerator copy) =>
        new(Name, Kind, copy.Model.Network.ParameterCount, copy.Model.Device, copy.Model.Encoder.MaxPositions == int.MaxValue ? null : copy.Model.Encoder.MaxPositions);

    /// <summary>One generation on a free copy, counted in the engine's statistics when it ends.</summary>
    public async IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var lease = await Host.AcquireAsync(cancellationToken).ConfigureAwait(false);
        await foreach (var chunk in lease.Guard(lease.Copy.StreamAsync(prompt, options, lease.Token)).ConfigureAwait(false))
        {
            if (chunk.Done)
            {
                lease.Completed(1, chunk.Stats!.GeneratedTokens, chunk.Stats.GenerationDuration);
            }

            yield return chunk;
        }
    }
}
