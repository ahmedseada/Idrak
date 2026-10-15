// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// Trains a text-to-speech model (<see cref="ITrainableSpeechSynthesizer"/>) on transcribed speech with its objective
/// (<see cref="SpeechSynthesisObjectives"/>: "durations"), plan 21 S6: the speech trainer's loop
/// (<see cref="SpeechTrainer"/>: the same batches by length, features read ahead and cached, augmentations, micro-batches
/// halved when the device runs out of memory, <c>checkpoint-STEP</c> folders resumed bit for bit on the CPU) with the
/// synthesis loss: each utterance's log-mel features are the target, its text the input, and its label durations come
/// from the alignments file <see cref="SpeechTrainingOptions.Durations"/> names (<see cref="SpeechAlignments"/>, a CTC
/// model's best paths), scaled to the utterance's frames (<see cref="SpeechAlignment.FrameDurations"/>). An utterance with
/// no alignment, or whose alignment's labels are not as many as the model's labels of its text (the synthesizer and the
/// CTC model read the text with different tokenizers), is left out with a note. The evaluation's score is the mean loss
/// of the evaluation utterances that have alignments (lower is better).
/// </summary>
public static class SpeechSynthesisTrainer
{
    /// <summary>
    /// Trains <paramref name="model"/> on <paramref name="train"/> for <see cref="SpeechTrainingOptions.Steps"/> steps (see
    /// <see cref="SpeechTrainer.Train"/>), its durations from <see cref="SpeechTrainingOptions.Durations"/>; or goes on from
    /// <paramref name="resume"/> (see <see cref="Resume"/>).
    /// </summary>
    /// <exception cref="ArgumentException">An option is out of range or names nothing registered, the options name no alignments file, or no utterance can be trained on.</exception>
    /// <exception cref="FileNotFoundException">The alignments file does not exist.</exception>
    /// <exception cref="InvalidOperationException">The loss stayed not finite (<see cref="SpeechTrainingOptions.NonFiniteSteps"/>).</exception>
    /// <exception cref="OperationCanceledException">Cancelled: a checkpoint of the last finished step is written first.</exception>
    public static SpeechTrainingResult Train(ITrainableSpeechSynthesizer model, SpeechSource train, SpeechSource? evaluation, SpeechTrainingOptions options, string outputFolder,
        SpeechTrainingCheckpoint? resume = null, IProgress<SpeechTrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        if (string.IsNullOrWhiteSpace(options.Durations))
        {
            throw new ArgumentException("A synthesizer trains on label durations: give SpeechTrainingOptions.Durations, an alignments file (SpeechAlignments.AlignToFile, or idrak data audio --align MODEL).", nameof(options));
        }

        if (options.Augmentations.Select(AudioAugmentations.Get).FirstOrDefault(a => a.Stage == AudioAugmentationStage.Features) is { } masking)
        {
            throw new ArgumentException($"The '{masking.Name}' augmentation changes the features after they are taken, and a synthesizer's features are its targets: "
                + "give a synthesizer sound augmentations only (\"speed\", \"noise\").", nameof(options));
        }

        var objective = SpeechSynthesisObjectives.Get(model.Objective);
        if (!objective.Trains(model))
        {
            throw new ArgumentException($"The '{objective.Name}' synthesis objective does not train a {model.GetType().Name} ({model.Architecture}).", nameof(model));
        }

        var alignments = SpeechAlignments.Read(options.Durations);
        return SpeechTrainer.Run(new SynthesisTrainee(model, objective, options, SpeechAlignments.Index(alignments)), train, evaluation, options, outputFolder, resume, progress,
            cancellationToken, trace);
    }

    /// <summary>
    /// Goes on with the run a checkpoint folder holds: its model read back (<see cref="SpeechSynthesisModels.Load"/>, a
    /// family whose model is an <see cref="ITrainableSpeechSynthesizer"/>), its optimizer's state, sources, position and
    /// options (or <paramref name="overrides"/>), writing later checkpoints beside it.
    /// </summary>
    /// <exception cref="FileNotFoundException">The folder holds no training state.</exception>
    /// <exception cref="InvalidDataException">The folder's model is not one the trainer trains.</exception>
    public static SpeechTrainingResult Resume(string checkpointFolder, SpeechTrainingOptions? overrides = null, Device? device = null, IProgress<SpeechTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default, Action<string>? trace = null)
    {
        var state = SpeechTrainer.ReadCheckpoint(checkpointFolder);
        using var synthesizer = SpeechSynthesisModels.Load(checkpointFolder, new SpeechSynthesisLoadOptions { Device = device });
        var model = synthesizer as ITrainableSpeechSynthesizer
            ?? throw new InvalidDataException($"{checkpointFolder} holds a {synthesizer.GetType().Name}, not a model the synthesis trainer trains (an ITrainableSpeechSynthesizer).");
        string output = Path.GetDirectoryName(Path.GetFullPath(checkpointFolder))!;
        var options = overrides ?? state.Options;
        return Train(model, state.Train, state.Evaluation, options, output, state with { Options = options }, progress, cancellationToken, trace);
    }

    // A synthesizer and its objective: the utterances with an alignment of as many labels as the model reads, their
    // durations in frames, the objective's loss; the evaluation scores each utterance's loss.
    private sealed class SynthesisTrainee(ITrainableSpeechSynthesizer model, ISpeechSynthesisObjective objective, SpeechTrainingOptions options,
        Func<SpeechUtterance, SpeechAlignment?> alignments) : SpeechTrainee
    {
        public override string Architecture => model.Architecture;

        public override string Objective => model.Objective;

        public override Module Network => model.Network;

        public override IAudioFeatureExtractor Features => model.Features;

        public override Device Device => model.Device;

        public override string Refused => $"have no alignment in {options.Durations} with as many labels as the model reads in their text, or {objective.Refused}";

        public override string Metric => "loss";

        public override int[] Labels(string text) => model.Labels(text);

        public override bool Accepts(ISpeechReader reader, int utterance, int frames, int[] labels) =>
            Aligned(reader, utterance, labels.Length) is not null && objective.Accepts(model, frames, labels);

        public override bool Evaluates(ISpeechReader reader, int utterance)
        {
            var labels = model.Labels(reader[utterance].Text);
            return Aligned(reader, utterance, labels.Length) is not null && objective.Accepts(model, 1, labels);
        }

        public override Tensor Loss(ISpeechReader reader, Tensor features, SpeechTrainer.Prepared[] group, SpeechForwardOptions forward)
        {
            var (frames, labels, durations) = Targets(reader, group, group.Select(u => u.Labels).ToArray());
            return objective.Loss(model, features, frames, labels, durations, forward, options);
        }

        public override IEnumerable<TuningScore> Score(ISpeechReader reader, Tensor features, SpeechTrainer.Prepared[] group)
        {
            var (frames, labels, durations) = Targets(reader, group, [.. group.Select(u => model.Labels(reader[u.Utterance].Text))]);
            float loss = objective.Loss(model, features, frames, labels, durations, new SpeechForwardOptions(), options).Item();
            return [new TuningScore(loss * group.Length, group.Length)];
        }

        public override void Save(string folder) => model.Save(folder);

        private SpeechAlignment? Aligned(ISpeechReader reader, int utterance, int labels) =>
            alignments(reader[utterance]) is { } found && found.Labels.Length == labels ? found : null;

        // Each utterance's frames, labels and frames per label (its alignment scaled to its features' frames).
        private (int[] Frames, int[][] Labels, int[][] Durations) Targets(ISpeechReader reader, SpeechTrainer.Prepared[] group, int[][] labels)
        {
            var frames = new int[group.Length];
            var durations = new int[group.Length][];
            for (int i = 0; i < group.Length; i++)
            {
                frames[i] = group[i].Frames;
                var alignment = Aligned(reader, group[i].Utterance, labels[i].Length)
                    ?? throw new InvalidOperationException($"{reader[group[i].Utterance].Name} has no alignment of {labels[i].Length} labels in {options.Durations}.");
                durations[i] = alignment.FrameDurations(frames[i]);
            }

            return (frames, labels, durations);
        }
    }
}
