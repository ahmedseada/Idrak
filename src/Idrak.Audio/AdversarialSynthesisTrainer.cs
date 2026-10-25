// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Optimizers;

namespace Idrak.Audio;

/// <summary>
/// Trains a synthesizer against discriminators (<see cref="IAdversarialSpeechSynthesizer"/>, plan 21 S9 route C: a VITS
/// checkpoint fine-tuned on 30 to 60 minutes of one person) on transcribed speech, in the speech trainer's loop
/// (<see cref="SpeechTrainer"/>: the same batches by length, the training rows read ahead and cached, micro-batches halved
/// when the device runs out of memory, <c>checkpoint-STEP</c> folders resumed bit for bit on the CPU). Each step is
/// VITS's: the discriminators' least-squares loss on the real sound and the sound the generator makes (detached), their
/// optimizer's step; then the generator's loss (mel, KL, duration, adversarial, feature matching, weighted as
/// <see cref="SpeechTrainingOptions.Adversarial"/> says) with the stepped discriminators, and its step. Both optimizers
/// are the options' (Adam's betas and epsilon from <see cref="AdversarialSynthesisOptions"/>), with the same schedule; a
/// checkpoint holds the discriminators and their optimizer's state beside the generator's. The evaluation's score is the
/// mean absolute log-mel error of the evaluation utterances' segments (lower is better).
/// </summary>
public static class AdversarialSynthesisTrainer
{
    /// <summary>The objective's name in a checkpoint's state.</summary>
    public const string Objective = "adversarial";

    /// <summary>The discriminators' optimizer state in a checkpoint folder.</summary>
    public const string DiscriminatorOptimizerFile = "discriminator-optimizer.safetensors";

    /// <summary>The files of a checkpoint folder that only resuming reads (the run's state, both optimizers' state).</summary>
    public static IReadOnlyList<string> StateFiles { get; } = [SpeechTrainer.StateFile, SpeechTrainer.OptimizerFile, DiscriminatorOptimizerFile];

    /// <summary>
    /// Trains <paramref name="model"/> on <paramref name="train"/> for <see cref="SpeechTrainingOptions.Steps"/> steps (see
    /// <see cref="SpeechTrainer.Train"/>), or goes on from <paramref name="resume"/> (see <see cref="Resume"/>).
    /// </summary>
    /// <exception cref="ArgumentException">An option is out of range or names nothing registered, a feature augmentation is asked for, or no utterance can be trained on.</exception>
    /// <exception cref="InvalidOperationException">The loss stayed not finite (<see cref="SpeechTrainingOptions.NonFiniteSteps"/>).</exception>
    /// <exception cref="OperationCanceledException">Cancelled: a checkpoint of the last finished step is written first.</exception>
    public static SpeechTrainingResult Train(IAdversarialSpeechSynthesizer model, SpeechSource train, SpeechSource? evaluation, SpeechTrainingOptions options, string outputFolder,
        SpeechTrainingCheckpoint? resume = null, IProgress<SpeechTrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        if (options.Augmentations.Select(AudioAugmentations.Get).FirstOrDefault(a => a.Stage == AudioAugmentationStage.Features) is { } masking)
        {
            throw new ArgumentException($"The '{masking.Name}' augmentation changes the training rows after they are taken, and a synthesizer's rows are its targets: "
                + "give a synthesizer sound augmentations only (\"speed\", \"noise\").", nameof(options));
        }

        var adversarial = options.Adversarial ?? new AdversarialSynthesisOptions();
        model.Freeze(adversarial.Frozen);
        if (adversarial.Frozen.Count > 0)
        {
            trace?.Invoke($"frozen: {string.Join(", ", adversarial.Frozen)}");
        }

        return SpeechTrainer.Run(new AdversarialTrainee(model, options, adversarial), train, evaluation, options, outputFolder, resume, progress, cancellationToken, trace);
    }

    /// <summary>
    /// Goes on with the run a checkpoint folder holds: its model read back to be trained
    /// (<see cref="SpeechSynthesisModels.Load"/> with <see cref="SpeechSynthesisLoadOptions.Training"/>, a family whose model
    /// is an <see cref="IAdversarialSpeechSynthesizer"/>, its discriminators from the folder), both optimizers' state, the
    /// sources, position and options (or <paramref name="overrides"/>), writing later checkpoints beside it.
    /// </summary>
    /// <exception cref="FileNotFoundException">The folder holds no training state.</exception>
    /// <exception cref="InvalidDataException">The folder's model is not one this trainer trains.</exception>
    public static SpeechTrainingResult Resume(string checkpointFolder, SpeechTrainingOptions? overrides = null, Device? device = null, IProgress<SpeechTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default, Action<string>? trace = null)
    {
        var state = SpeechTrainer.ReadCheckpoint(checkpointFolder);
        using var synthesizer = SpeechSynthesisModels.Load(checkpointFolder, new SpeechSynthesisLoadOptions { Device = device, Training = true });
        var model = synthesizer as IAdversarialSpeechSynthesizer
            ?? throw new InvalidDataException($"{checkpointFolder} holds a {synthesizer.GetType().Name}, not a model the adversarial trainer trains (an IAdversarialSpeechSynthesizer).");
        string output = Path.GetDirectoryName(Path.GetFullPath(checkpointFolder))!;
        var options = overrides ?? state.Options;
        return Train(model, state.Train, state.Evaluation, options, output, state with { Options = options }, progress, cancellationToken, trace);
    }

    // The optimizer the options name over `parameters` at `rate`: Adam and AdamW with the adversarial options' betas and
    // epsilon (VITS's 0.8, 0.99, 1e-9), the others as the speech trainer makes them.
    internal static Optimizer Optimizer(List<Tensor> parameters, SpeechTrainingOptions options, AdversarialSynthesisOptions adversarial, float rate)
    {
        var settings = new OptimizerSettings(rate, options.WeightDecay, options.Momentum);
        if (options.Optimizer.Equals("adamw", StringComparison.OrdinalIgnoreCase) || options.Optimizer.Equals("adam", StringComparison.OrdinalIgnoreCase))
        {
            settings = settings with { Beta1 = adversarial.Beta1, Beta2 = adversarial.Beta2, Epsilon = adversarial.Epsilon };
        }

        var make = OptimizerKinds.Factory(options.Optimizer, settings);
        return options.HostOptimizer ? new HostOptimizer(parameters, make) : make(parameters);
    }

    // The synthesizer and its discriminators in the speech trainer's loop: the generator is the network the loop's
    // optimizer steps, the discriminators get their own optimizer and schedule here, and a step runs the two halves.
    private sealed class AdversarialTrainee(IAdversarialSpeechSynthesizer model, SpeechTrainingOptions options, AdversarialSynthesisOptions adversarial) : SpeechTrainee
    {
        private Module? _discriminators;
        private Optimizer? _optimizer;
        private ILearningRateScheduler? _schedule;
        private Dictionary<string, float>? _terms;
        private int _steppedAt = -1;
        private float _discriminatorLoss;

        public override string Architecture => model.Architecture;

        public override string Objective => AdversarialSynthesisTrainer.Objective;

        public override Module Network => model.Generator;

        public override IAudioFeatureExtractor Features => model.Features;

        public override Device Device => model.Device;

        public override string Refused => model.Refused;

        public override string Metric => "mel";

        public override int FeatureWidth => model.FeatureWidth;

        public override IReadOnlyDictionary<string, float>? Terms => _terms;

        public override int[] Labels(string text) => model.Labels(text);

        public override bool Accepts(ISpeechReader reader, int utterance, int frames, int[] labels) => model.Accepts(frames, labels, adversarial);

        public override bool Evaluates(ISpeechReader reader, int utterance)
        {
            var labels = model.Labels(reader[utterance].Text);
            int frames = model.Features.FramesFor((long)Math.Round(reader[utterance].Seconds * model.Features.Options.SampleRate));
            return model.Accepts(frames, labels, adversarial);
        }

        public override Tensor Loss(ISpeechReader reader, Tensor features, SpeechTrainer.Prepared[] group, SpeechForwardOptions forward) =>
            throw new NotSupportedException("An adversarial synthesizer's step has two halves (AdversarialSynthesisTrainer runs them); it has no single loss.");

        public override IEnumerable<TuningScore> Score(ISpeechReader reader, Tensor features, SpeechTrainer.Prepared[] group)
        {
            var labels = group.Select(u => model.Labels(reader[u.Utterance].Text)).ToArray();
            var batch = new AdversarialSynthesisBatch(features, [.. group.Select(u => u.Frames)], labels, [.. group.Select(u => u.Utterance)], 0, options.Seed,
                group.Length, group.Sum(u => (long)u.Frames), labels.Sum(l => (long)l.Length));
            return [new TuningScore(model.MelError(batch, adversarial) * group.Length, group.Length)];
        }

        public override void Save(string folder) => model.Save(folder);

        public override Optimizer CreateOptimizer(List<Tensor> parameters, SpeechTrainingOptions run) => Optimizer(parameters, run, adversarial, run.LearningRate);

        public override IDisposable? Start(SpeechTrainingOptions run, string? resume, int step)
        {
            _discriminators = model.Discriminators(adversarial, run.Seed);
            _discriminators.Train();
            _optimizer = Optimizer([.. _discriminators.Parameters()], run, adversarial, adversarial.DiscriminatorLearningRate ?? run.LearningRate);
            if (resume is not null)
            {
                string path = Path.Combine(resume, DiscriminatorOptimizerFile);
                if (!File.Exists(path))
                {
                    throw new FileNotFoundException($"{resume} holds no discriminator optimizer state ({DiscriminatorOptimizerFile}); it was not written by AdversarialSynthesisTrainer.", path);
                }

                _optimizer.RequireState().LoadState(SpeechTrainer.ReadState(path));
            }

            _schedule = LearningRateSchedules.Factory(run.Schedule, new LearningRateScheduleSettings(run.WarmupFraction, run.MinLearningRate))(_optimizer, run.Steps);
            for (int i = 0; i < step; i++)
            {
                _schedule.Step();
            }

            return _optimizer;
        }

        public override void Stepped() => _schedule?.Step();

        public override long ParameterBytes => base.ParameterBytes + (_discriminators?.Parameters().Sum(p => (long)p.Size) ?? 0) * sizeof(float);

        // Both halves of a step over the group, as a step runs them (the generator's pass kept between them when the
        // options say so), no optimizer stepped, every gradient released.
        public override void Probe(ISpeechReader reader, SpeechTrainer.Prepared[] group, SpeechTrainingOptions run, SpeechForwardOptions forward)
        {
            using var features = SpeechTrainer.Padded(group, this);
            var batch = new AdversarialSynthesisBatch(features, [.. group.Select(u => u.Frames)], [.. group.Select(u => u.Labels)], [.. group.Select(u => u.Utterance)],
                1, run.Seed, group.Length, group.Sum(u => (long)u.Frames), group.Sum(u => (long)u.Labels.Length), forward.Checkpointing);
            using (var scope = new TensorScope())
            using (var precision = run.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null)
            using (var step = model.Begin(batch, adversarial, adversarial.KeepGeneratorPass))
            {
                step.Discriminate();
                step.Generate();
            }

            foreach (var parameter in model.Generator.Parameters().Concat(_discriminators?.Parameters() ?? []))
            {
                parameter.ReleaseGrad();
            }
        }

        public override void SaveState(string folder) => SpeechTrainer.WriteState(Path.Combine(folder, DiscriminatorOptimizerFile), _optimizer!);

        // The discriminators' half of every micro-batch and their step, then the generator's half and its step. A step run
        // again after the device ran out of memory in the generator's half (smaller micro-batches) does not step the
        // discriminators twice: their half is done once a step.
        public override double Step(ISpeechReader reader, IReadOnlyList<SpeechTrainer.Prepared[]> groups, Optimizer optimizer, SpeechTrainingOptions run,
            SpeechForwardOptions forward, int step)
        {
            int utterances = groups.Sum(g => g.Length);
            long frames = groups.Sum(g => g.Sum(u => (long)u.Frames)), labels = groups.Sum(g => g.Sum(u => (long)u.Labels.Length));
            bool keep = adversarial.KeepGeneratorPass && groups.Count == 1;
            var features = new Tensor?[groups.Count];
            var steps = new IAdversarialSynthesisStep?[groups.Count];
            using var outer = keep ? new TensorScope() : null;
            using var precision = run.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null;
            try
            {
                for (int i = 0; i < groups.Count; i++)
                {
                    var group = groups[i];
                    features[i] = TensorScope.Untrack(SpeechTrainer.Padded(group, this));
                    var batch = new AdversarialSynthesisBatch(features[i]!, [.. group.Select(u => u.Frames)], [.. group.Select(u => u.Labels)], [.. group.Select(u => u.Utterance)],
                        step, run.Seed, utterances, frames, labels, forward.Checkpointing);
                    steps[i] = model.Begin(batch, adversarial, keep);
                }

                if (_steppedAt != step)
                {
                    _optimizer!.ZeroGrad();
                    float loss = 0;
                    foreach (var part in steps)
                    {
                        using var scope = keep ? null : new TensorScope();
                        loss += part!.Discriminate();
                    }

                    _optimizer.ClipAndStep(run.MaxGradientNorm);
                    (_steppedAt, _discriminatorLoss) = (step, loss);
                }

                optimizer.ZeroGrad();
                var terms = new Dictionary<string, float>(StringComparer.Ordinal);
                foreach (var part in steps)
                {
                    using var scope = keep ? null : new TensorScope();
                    foreach (var (name, value) in part!.Generate())
                    {
                        terms[name] = terms.GetValueOrDefault(name) + value;
                    }
                }

                optimizer.ClipAndStep(run.MaxGradientNorm);
                terms["discriminator"] = _discriminatorLoss;
                _terms = terms;
                return terms.TryGetValue("generator", out float generator) ? generator : float.NaN;
            }
            finally
            {
                foreach (var part in steps)
                {
                    part?.Dispose();
                }

                foreach (var tensor in features)
                {
                    tensor?.Dispose();
                }
            }
        }
    }
}
