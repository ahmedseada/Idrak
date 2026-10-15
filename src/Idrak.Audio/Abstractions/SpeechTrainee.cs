// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio.Abstractions;

/// <summary>
/// What the speech trainer's loop (<c>SpeechTrainer</c>) trains: the layers, the features the utterances are read as,
/// the labels of a text, which utterances it trains on, the loss of a group of them, the evaluation's scores and how it is
/// saved. A recognizer with its objective is one (<c>SpeechTrainer.Train</c>), a synthesizer with its own another
/// (<c>SpeechSynthesisTrainer.Train</c>).
/// </summary>
internal abstract class SpeechTrainee
{
    public abstract string Architecture { get; }

    public abstract string Objective { get; }

    public abstract Module Network { get; }

    public abstract IAudioFeatureExtractor Features { get; }

    public abstract Device Device { get; }

    // Why the utterances Accepts (and Evaluates) refuse are left out, for the run's note.
    public abstract string Refused { get; }

    // The evaluation's score, for the trace ("cer").
    public abstract string Metric { get; }

    public abstract int[] Labels(string text);

    public abstract bool Accepts(ISpeechReader reader, int utterance, int frames, int[] labels);

    // Whether utterance `utterance` of the evaluation source can be scored.
    public virtual bool Evaluates(ISpeechReader reader, int utterance) => true;

    // The mean loss of `group` (read from `reader`), its features [count, longest, mels] on the device.
    public abstract Tensor Loss(ISpeechReader reader, Tensor features, SpeechTrainer.Prepared[] group, SpeechForwardOptions forward);

    // Each utterance's score of `group` (gradients off); the group's labels are not read.
    public abstract IEnumerable<TuningScore> Score(ISpeechReader reader, Tensor features, SpeechTrainer.Prepared[] group);

    public abstract void Save(string folder);
}
