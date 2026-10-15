// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Generation;
using Idrak.Audio.Abstractions;

namespace Idrak.Audio;

/// <summary>
/// The "ctc" training objective (<see cref="SpeechTrainingObjectives"/>): an <see cref="ICtcSpeechModel"/>'s logits, a
/// log-softmax over its classes, each utterance's steps (<see cref="ICtcSpeechModel.StepsFor"/> of its frames) and labels,
/// PyTorch's <c>ctc_loss</c> with mean reduction and zero_infinity; evaluation decodes through <see cref="CtcDecoders"/>.
/// An utterance whose steps are fewer than its labels need (<see cref="CtcAlignment.MinimumSteps"/>) is not trained on.
/// </summary>
public sealed class CtcObjective : ISpeechTrainingObjective
{
    /// <inheritdoc />
    public string Name => SpeechTrainingObjectives.Ctc;

    /// <inheritdoc />
    public string Refused => "give fewer steps than their labels need (CTC cannot read them)";

    /// <inheritdoc />
    public bool Trains(ITrainableSpeechModel model) => model is ICtcSpeechModel;

    /// <inheritdoc />
    public bool Accepts(ITrainableSpeechModel model, int frames, int[] labels) => Ctc(model).StepsFor(frames) >= CtcAlignment.MinimumSteps(labels);

    /// <inheritdoc />
    public Tensor Loss(ITrainableSpeechModel model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, SpeechForwardOptions forward) =>
        Loss(Ctc(model), features, frames, labels, forward);

    /// <summary>The mean CTC loss of a batch, as the trainer computes it.</summary>
    public static Tensor Loss(ICtcSpeechModel model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, SpeechForwardOptions? forward = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(labels);
        var logProbs = model.Logits(features, frames, forward).LogSoftmax();
        var steps = new int[frames.Count];
        for (int i = 0; i < steps.Length; i++)
        {
            steps[i] = model.StepsFor(frames[i]);
        }

        return Losses.Ctc(logProbs, labels, steps, model.Blank, LossReduction.Mean, zeroInfinity: true, batchFirst: true);
    }

    /// <inheritdoc />
    public IReadOnlyList<int[]> Decode(ITrainableSpeechModel model, Tensor features, IReadOnlyList<int> frames, string decoder, int beamWidth)
    {
        var ctc = Ctc(model);
        var logProbs = ctc.Logits(features, frames).LogSoftmax();
        var readings = CtcDecoders.Decode(logProbs, [.. frames.Select(ctc.StepsFor)], decoder, new CtcDecodeOptions { Blank = ctc.Blank, BeamWidth = beamWidth }, batchFirst: true);
        return [.. readings.Select(r => r[0].Labels)];
    }

    private static ICtcSpeechModel Ctc(ITrainableSpeechModel model) =>
        model as ICtcSpeechModel ?? throw new ArgumentException($"The ctc objective trains an ICtcSpeechModel, not a {model.GetType().Name}.", nameof(model));
}
