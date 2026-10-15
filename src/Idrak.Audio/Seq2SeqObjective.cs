// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Generation;
using Idrak.Audio.Abstractions;

namespace Idrak.Audio;

/// <summary>
/// The "seq2seq" training objective (<see cref="SpeechTrainingObjectives"/>): an <see cref="ISequenceSpeechModel"/>'s
/// encoder over the batch, its decoder read with teacher forcing (the start token and each utterance's labels in, each
/// next label and then the end token taught), the cross-entropy of every taught position through the fused, chunked
/// <see cref="Losses.TokenCrossEntropy"/> (the [positions, vocabulary] logits never stored), averaged over each
/// utterance's positions and then over the batch, so an utterance's loss is the same in any batch. Evaluation generates
/// each utterance's labels: greedy for the "greedy" decoder, a beam search of the beam width otherwise. An utterance
/// whose labels and end token do not fit the decoder's positions, or too short for one encoder step, is not trained on.
/// </summary>
public sealed class Seq2SeqObjective : ISpeechTrainingObjective
{
    /// <inheritdoc />
    public string Name => SpeechTrainingObjectives.Seq2Seq;

    /// <inheritdoc />
    public string Refused => "have more labels than the decoder's positions hold, or too few frames for one encoder step";

    /// <inheritdoc />
    public bool Trains(ITrainableSpeechModel model) => model is ISequenceSpeechModel;

    /// <inheritdoc />
    public bool Accepts(ITrainableSpeechModel model, int frames, int[] labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        var sequence = Sequence(model);
        return sequence.StepsFor(frames) >= 1 && labels.Length + 1 <= sequence.MaxPositions;
    }

    /// <inheritdoc />
    public Tensor Loss(ITrainableSpeechModel model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, SpeechForwardOptions forward) =>
        Loss(Sequence(model), features, frames, labels, forward);

    /// <summary>The batch's loss as the trainer computes it: each utterance's mean next-token cross-entropy, averaged over the batch.</summary>
    /// <exception cref="ArgumentException">Not one label sequence per utterance, or one too long for the decoder's positions.</exception>
    public static Tensor Loss(ISequenceSpeechModel model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, SpeechForwardOptions? forward = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(labels);
        int n = labels.Count;
        if (frames.Count != n)
        {
            throw new ArgumentException($"{frames.Count} utterances' frames for {n} label sequences.", nameof(labels));
        }

        int length = 0;
        foreach (var row in labels)
        {
            length = Math.Max(length, row.Length + 1);
        }

        if (length > model.MaxPositions)
        {
            throw new ArgumentException($"A label sequence of {length - 1} labels and the end token does not fit the decoder's {model.MaxPositions} positions.", nameof(labels));
        }

        // Inputs: the start token, the labels, then the end token as padding (each position sees only the tokens up to it,
        // so the padding after an utterance's own positions changes nothing they give). Targets: the labels and the end
        // token, weighted 1 / (positions · batch); the padding weighted 0.
        var tokens = new float[n * length];
        var targets = new float[n * length];
        var weights = new float[n * length];
        var steps = new int[n];
        for (int b = 0; b < n; b++)
        {
            var row = labels[b];
            int at = b * length;
            tokens.AsSpan(at, length).Fill(model.EndToken);
            tokens[at] = model.StartToken;
            float weight = 1f / ((row.Length + 1) * n);
            for (int i = 0; i < row.Length; i++)
            {
                tokens[at + i + 1] = row[i];
                targets[at + i] = row[i];
                weights[at + i] = weight;
            }

            targets[at + row.Length] = model.EndToken;
            weights[at + row.Length] = weight;
            steps[b] = model.StepsFor(frames[b]);
        }

        var device = features.Device;
        var memory = model.Encode(features, frames, forward);
        var hidden = model.Hidden(Tensor.From(tokens, [n, length], device), memory, steps);
        return Losses.TokenCrossEntropy(hidden.Reshape(n * length, hidden.Shape[2]), model.Head, Tensor.From(targets, [n * length], device),
            Tensor.From(weights, [n * length], device), 1f);
    }

    /// <inheritdoc />
    public IReadOnlyList<int[]> Decode(ITrainableSpeechModel model, Tensor features, IReadOnlyList<int> frames, string decoder, int beamWidth)
    {
        var sequence = Sequence(model);
        ArgumentNullException.ThrowIfNull(frames);
        int beams = decoder == CtcDecoders.Greedy ? 1 : Math.Max(1, beamWidth);
        using var noGrad = Autograd.NoGrad();
        var memory = sequence.Encode(features, frames);
        var readings = new int[frames.Count][];
        for (int b = 0; b < readings.Length; b++)
        {
            using var scope = new TensorScope();
            var own = memory.Narrow(0, b, 1).Narrow(1, 0, sequence.StepsFor(frames[b]));   // its own steps: no padding to attend to
            readings[b] = sequence.Generate(own, beams);
        }

        return readings;
    }

    private static ISequenceSpeechModel Sequence(ITrainableSpeechModel model) =>
        model as ISequenceSpeechModel ?? throw new ArgumentException($"The seq2seq objective trains an ISequenceSpeechModel, not a {model?.GetType().Name ?? "null"}.", nameof(model));
}
