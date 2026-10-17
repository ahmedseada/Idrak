// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio;

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

    // The width of an utterance's feature rows (the mels, for a log-mel model).
    public virtual int FeatureWidth => Features.Options.Mels;

    // The last step's loss terms by name, for the progress (null: the loss alone).
    public virtual IReadOnlyDictionary<string, float>? Terms => null;

    // The optimizer of the network's parameters.
    public virtual Optimizer CreateOptimizer(List<Tensor> parameters, SpeechTrainingOptions options) => SpeechTrainer.CreateOptimizer(parameters, options);

    // Before the first step: the trainee's own optimizers (and their state from `resume`, `step` steps done); disposed after the run.
    public virtual IDisposable? Start(SpeechTrainingOptions options, string? resume, int step) => null;

    // After each optimizer step: the trainee's own schedules.
    public virtual void Stepped()
    {
    }

    // Writes the trainee's own optimizers' state into a checkpoint folder.
    public virtual void SaveState(string folder)
    {
    }

    // The bytes the trainee's parameters take (the network's and any of its own, such as discriminators): what the
    // optimizers' state is sized by.
    public virtual long ParameterBytes => Network.Parameters().Sum(p => (long)p.Size) * sizeof(float);

    // One micro-batch through a step's passes, forward and backward, without an optimizer step, its gradients released
    // after: the memory probe an automatic batch is sized by (SpeechTrainingOptions.BatchSeconds 0).
    public virtual void Probe(ISpeechReader reader, SpeechTrainer.Prepared[] group, SpeechTrainingOptions options, SpeechForwardOptions forward)
    {
        using (var scope = new TensorScope())
        using (var precision = options.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null)
        {
            var loss = Loss(reader, SpeechTrainer.Padded(group, this), group, forward);
            loss.Backward();
        }

        foreach (var parameter in Network.Parameters())
        {
            parameter.ReleaseGrad();
        }
    }

    // One optimizer step over the step's micro-batches (`step` counts from 1): each micro-batch's mean loss, scaled by its
    // utterances' share of the step, backward; then the optimizer's step. Returns the step's mean loss.
    public virtual double Step(ISpeechReader reader, IReadOnlyList<SpeechTrainer.Prepared[]> groups, Optimizer optimizer, SpeechTrainingOptions options,
        SpeechForwardOptions forward, int step)
    {
        int total = groups.Sum(g => g.Length);
        optimizer.ZeroGrad();
        double sum = 0;
        foreach (var group in groups)
        {
            using var scope = new TensorScope();
            using var precision = options.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null;
            var loss = Loss(reader, SpeechTrainer.Padded(group, this), group, forward);
            var scaled = group.Length == total ? loss : loss * (group.Length / (float)total);
            scaled.Backward();
            sum += loss.Item() * group.Length;
        }

        optimizer.ClipAndStep(options.MaxGradientNorm);
        return sum / total;
    }
}
