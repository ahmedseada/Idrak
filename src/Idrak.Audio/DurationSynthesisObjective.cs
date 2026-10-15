// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio.Abstractions;

namespace Idrak.Audio;

/// <summary>
/// The "durations" synthesis objective (<see cref="SpeechSynthesisObjectives"/>, plan 21 S6): an
/// <see cref="ITrainableSpeechSynthesizer"/>'s log-mel with each label repeated for its given frames (from a CTC
/// alignment, <see cref="SpeechAlignments"/>), each utterance's mean absolute error over its frames and mels, plus
/// <see cref="SpeechTrainingOptions.DurationLossWeight"/> times its mean squared error of the predicted log(frames +
/// offset) over its labels, each averaged over the batch, so an utterance's loss is the same in any batch. An utterance
/// without a label or a frame is not trained on.
/// </summary>
public sealed class DurationSynthesisObjective : ISpeechSynthesisObjective
{
    /// <inheritdoc />
    public string Name => SpeechSynthesisObjectives.Durations;

    /// <inheritdoc />
    public string Refused => "have no label or no frame";

    /// <inheritdoc />
    public bool Trains(ITrainableSpeechSynthesizer model) => model is not null;

    /// <inheritdoc />
    public bool Accepts(ITrainableSpeechSynthesizer model, int frames, int[] labels) => frames >= 1 && labels is { Length: >= 1 };

    /// <inheritdoc />
    public Tensor Loss(ITrainableSpeechSynthesizer model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, IReadOnlyList<int[]> durations,
        SpeechForwardOptions forward, SpeechTrainingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return Loss(model, features, frames, labels, durations, options.DurationLossWeight, forward);
    }

    /// <summary>The batch's loss as the trainer computes it: the log-mel's mean absolute error plus <paramref name="durationWeight"/> times the log durations' mean squared error.</summary>
    public static Tensor Loss(ITrainableSpeechSynthesizer model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, IReadOnlyList<int[]> durations,
        float durationWeight = 1f, SpeechForwardOptions? forward = null)
    {
        var (mel, duration) = Parts(model, features, frames, labels, durations, forward);
        return durationWeight == 0 ? mel : mel + duration * durationWeight;
    }

    /// <summary>
    /// The loss's two parts: the mean over the batch of each utterance's mean absolute log-mel error over its frames and
    /// mels, and of its mean squared log-duration error over its labels.
    /// </summary>
    /// <exception cref="ArgumentException">The batch's lists disagree, a text's durations are not one per label, or they do not sum to its frames.</exception>
    public static (Tensor Mel, Tensor Duration) Parts(ITrainableSpeechSynthesizer model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels,
        IReadOnlyList<int[]> durations, SpeechForwardOptions? forward = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentNullException.ThrowIfNull(durations);
        int n = labels.Count;
        if (frames.Count != n || durations.Count != n || features.Rank != 3 || features.Shape[0] != n)
        {
            throw new ArgumentException($"{n} texts with {frames.Count} frame counts, {durations.Count} duration lists and features {Tensor.FormatShape(features.Shape)}: one each per utterance.", nameof(labels));
        }

        int length = 0;
        for (int b = 0; b < n; b++)
        {
            if (durations[b].Length != labels[b].Length || durations[b].Sum() != frames[b])
            {
                throw new ArgumentException($"Utterance {b}: {durations[b].Length} durations summing to {durations[b].Sum()} for {labels[b].Length} labels and {frames[b]} frames.", nameof(durations));
            }

            length = Math.Max(length, labels[b].Length);
        }

        int mels = features.Shape[2], longest = features.Shape[1];
        var ids = new float[n * length];
        var targets = new float[n * length];
        var labelWeights = new float[n * length];
        var frameWeights = new float[n * longest];
        var lengths = new int[n];
        float offset = model.DurationOffset;
        for (int b = 0; b < n; b++)
        {
            lengths[b] = labels[b].Length;
            float labelWeight = 1f / (lengths[b] * n), frameWeight = 1f / ((float)frames[b] * mels * n);
            for (int i = 0; i < lengths[b]; i++)
            {
                ids[b * length + i] = labels[b][i];
                targets[b * length + i] = MathF.Log(durations[b][i] + offset);
                labelWeights[b * length + i] = labelWeight;
            }

            frameWeights.AsSpan(b * longest, frames[b]).Fill(frameWeight);
        }

        var device = features.Device;
        var hidden = model.Encode(Tensor.From(ids, [n, length], device), lengths, forward);
        var logDurations = model.LogDurations(hidden, lengths);                     // [n, length]
        var mel = model.Decode(hidden, lengths, durations, forward);               // [n, frames, mels]
        if (mel.Rank != 3 || mel.Shape[0] != n || mel.Shape[1] != longest || mel.Shape[2] != mels)
        {
            throw new InvalidOperationException($"The decoder made {Tensor.FormatShape(mel.Shape)} for targets {Tensor.FormatShape(features.Shape)}.");
        }

        // Each row's error weighted by 1 / (its utterance's frames · mels · batch), the padding by 0, then added up.
        var melLoss = (mel - features).Abs().Reshape(n * longest, mels).GroupAffine(Tensor.From(frameWeights, [frameWeights.Length], device), null, n * longest, mels).Sum();
        var error = logDurations.Reshape(n * length) - Tensor.From(targets, [n * length], device);
        var durationLoss = error.Square().GroupAffine(Tensor.From(labelWeights, [labelWeights.Length], device), null, n * length, 1).Sum();
        return (melLoss, durationLoss);
    }
}
