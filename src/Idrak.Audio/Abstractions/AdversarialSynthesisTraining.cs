// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// How a synthesizer is trained against discriminators (<c>AdversarialSynthesisTrainer</c>, plan 21 S9 route C: a VITS
/// checkpoint fine-tuned on one person's recordings). Every weight, size and rate is a setting; the defaults are VITS's
/// (its c_mel 45, c_kl 1, AdamW with betas 0.8 and 0.99, segments of 8192 samples at a hop of 256), a starting point and
/// not values tied to a device. Carried in <c>SpeechTrainingOptions.Adversarial</c>, so a checkpoint resumes with them.
/// </summary>
public sealed record AdversarialSynthesisOptions
{
    /// <summary>The frames of the random slice of each utterance's latents the decoder makes sound of (VITS trains its decoder on segments, so its memory does not grow with an utterance's length).</summary>
    public int SegmentFrames { get; init; } = 32;

    /// <summary>The weight of the mean absolute log-mel error of the segment's sound.</summary>
    public float MelWeight { get; init; } = 45f;

    /// <summary>The weight of the KL divergence of the posterior from the text's prior, over the batch's frames.</summary>
    public float KlWeight { get; init; } = 1f;

    /// <summary>The weight of the duration predictor's loss (the stochastic predictor's negative log-likelihood), over the batch's labels.</summary>
    public float DurationWeight { get; init; } = 1f;

    /// <summary>The weight of the generator's least-squares adversarial loss.</summary>
    public float AdversarialWeight { get; init; } = 1f;

    /// <summary>The weight of the feature-matching loss (the sum over the discriminators' layers of the mean absolute difference; VITS's feature_loss doubles it, hence 2).</summary>
    public float FeatureMatchingWeight { get; init; } = 2f;

    /// <summary>The mel bands of the mel loss.</summary>
    public int Mels { get; init; } = 80;

    /// <summary>The mel filterbank's lowest frequency, Hz.</summary>
    public float MelMinFrequency { get; init; }

    /// <summary>The mel filterbank's highest frequency, Hz; null: half the rate.</summary>
    public float? MelMaxFrequency { get; init; }

    /// <summary>The discriminators' learning rate; null: the generator's (<c>SpeechTrainingOptions.LearningRate</c>).</summary>
    public float? DiscriminatorLearningRate { get; init; }

    /// <summary>Adam's first-moment decay, for both optimizers ("adam" and "adamw").</summary>
    public float Beta1 { get; init; } = 0.8f;

    /// <summary>Adam's second-moment decay.</summary>
    public float Beta2 { get; init; } = 0.99f;

    /// <summary>Adam's epsilon.</summary>
    public float Epsilon { get; init; } = 1e-9f;

    /// <summary>
    /// With one micro-batch a step, keep the generator's pass from the discriminators' step to its own (one forward pass,
    /// as VITS runs it). False, or a step of several micro-batches: the generator's pass is run again with the same draws
    /// (the same values, an extra forward pass, the memory of one micro-batch's pass at a time).
    /// </summary>
    public bool KeepGeneratorPass { get; init; } = true;

    /// <summary>
    /// The family's description of new discriminators (for VITS: the periods, scales, channels, kernels, strides, groups;
    /// the plug-in's README lists them); null: the ones the model's folder holds, else the family's defaults.
    /// </summary>
    public JsonObject? Discriminator { get; init; }

    /// <summary>
    /// A file of trained discriminators to start from, as the family reads it (VITS: a <c>discriminator.safetensors</c>,
    /// with a <c>discriminator.json</c> beside it when it has one, or the original VITS and MMS releases' <c>D_*.pth</c>
    /// checkpoint, whose <c>weight_g</c> and <c>weight_v</c> names it reads). Null: the model folder's, else new ones of
    /// <see cref="Discriminator"/>. A model folder's own discriminators (a checkpoint's) train as they are, whatever this
    /// names. New discriminators tell nothing apart for thousands of steps against a trained generator, which then trains
    /// on the mel loss alone (a buzz at the frame rate); fine-tuning a released model takes its released discriminators
    /// (MMS ships them apart from its HF checkpoints).
    /// </summary>
    public string? DiscriminatorWeights { get; init; }

    /// <summary>
    /// The generator's parts kept as the base has them (<see cref="IAdversarialSpeechSynthesizer.Parts"/> names them; for
    /// VITS "decoder" keeps the base's waveform generator): their weights take no gradient and no optimizer state. Empty,
    /// the default: every part trains.
    /// </summary>
    public IReadOnlyList<string> Frozen { get; init; } = [];

    /// <summary>The options as JSON (<see cref="FromJson"/> reads it back).</summary>
    public JsonObject ToJson() => new()
    {
        ["segment_frames"] = SegmentFrames, ["mel_weight"] = MelWeight, ["kl_weight"] = KlWeight, ["duration_weight"] = DurationWeight,
        ["adversarial_weight"] = AdversarialWeight, ["feature_matching_weight"] = FeatureMatchingWeight, ["mels"] = Mels,
        ["mel_min_frequency"] = MelMinFrequency, ["mel_max_frequency"] = MelMaxFrequency, ["discriminator_learning_rate"] = DiscriminatorLearningRate,
        ["beta1"] = Beta1, ["beta2"] = Beta2, ["epsilon"] = Epsilon, ["keep_generator_pass"] = KeepGeneratorPass,
        ["discriminator"] = Discriminator?.DeepClone(), ["discriminator_weights"] = DiscriminatorWeights,
        ["frozen"] = new JsonArray([.. Frozen.Select(p => (JsonNode?)p)]),
    };

    /// <summary>Options from their JSON (<see cref="ToJson"/>); a key missing keeps the default.</summary>
    public static AdversarialSynthesisOptions FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var d = new AdversarialSynthesisOptions();
        return new AdversarialSynthesisOptions
        {
            SegmentFrames = (int?)json["segment_frames"] ?? d.SegmentFrames,
            MelWeight = (float?)json["mel_weight"] ?? d.MelWeight,
            KlWeight = (float?)json["kl_weight"] ?? d.KlWeight,
            DurationWeight = (float?)json["duration_weight"] ?? d.DurationWeight,
            AdversarialWeight = (float?)json["adversarial_weight"] ?? d.AdversarialWeight,
            FeatureMatchingWeight = (float?)json["feature_matching_weight"] ?? d.FeatureMatchingWeight,
            Mels = (int?)json["mels"] ?? d.Mels,
            MelMinFrequency = (float?)json["mel_min_frequency"] ?? d.MelMinFrequency,
            MelMaxFrequency = (float?)json["mel_max_frequency"],
            DiscriminatorLearningRate = (float?)json["discriminator_learning_rate"],
            Beta1 = (float?)json["beta1"] ?? d.Beta1,
            Beta2 = (float?)json["beta2"] ?? d.Beta2,
            Epsilon = (float?)json["epsilon"] ?? d.Epsilon,
            KeepGeneratorPass = (bool?)json["keep_generator_pass"] ?? d.KeepGeneratorPass,
            Discriminator = json["discriminator"] is JsonObject description ? (JsonObject)description.DeepClone() : null,
            DiscriminatorWeights = (string?)json["discriminator_weights"],
            Frozen = json["frozen"] is JsonArray frozen ? [.. frozen.Select(p => (string)p!)] : [],
        };
    }

    /// <summary>Refuses a value out of range, naming it.</summary>
    /// <exception cref="ArgumentException">A size is not positive, a weight negative, a rate not positive, a beta outside [0, 1).</exception>
    public void Check()
    {
        if (SegmentFrames < 1 || Mels < 1)
        {
            throw new ArgumentException($"Adversarial synthesis: SegmentFrames ({SegmentFrames}) and Mels ({Mels}) are positive.");
        }

        if (!(MelWeight >= 0) || !(KlWeight >= 0) || !(DurationWeight >= 0) || !(AdversarialWeight >= 0) || !(FeatureMatchingWeight >= 0))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"Adversarial synthesis: the loss weights are not negative (mel {MelWeight}, kl {KlWeight}, duration {DurationWeight}, adversarial {AdversarialWeight}, feature matching {FeatureMatchingWeight})."));
        }

        if (DiscriminatorLearningRate is { } rate && !(rate > 0) || Beta1 is < 0 or >= 1 || Beta2 is < 0 or >= 1 || !(Epsilon > 0))
        {
            throw new ArgumentException("Adversarial synthesis: DiscriminatorLearningRate and Epsilon are positive, Beta1 and Beta2 in [0, 1).");
        }

        if (MelMinFrequency < 0 || MelMaxFrequency is { } top && !(top > MelMinFrequency))
        {
            throw new ArgumentException("Adversarial synthesis: MelMinFrequency is not negative and MelMaxFrequency above it.");
        }
    }
}

/// <summary>
/// One micro-batch of a step, as the trainer hands it to the model: each utterance's training rows (the model's
/// <see cref="IAdversarialSpeechSynthesizer.Features"/>, padded to the longest), frames and labels, which utterances of
/// the source they are (the model's draws are seeded by them, so a step does not change when its micro-batches are
/// split differently), and the whole step's sizes (its losses are VITS's over the whole batch: the KL over its frames, the
/// duration loss over its labels, the rest averaged over its utterances, so the micro-batches' shares add up to it).
/// </summary>
/// <param name="Features">[count, longest, <see cref="IAdversarialSpeechSynthesizer.FeatureWidth"/>] on the model's device; zeros after an utterance's frames.</param>
/// <param name="Frames">Each utterance's frames.</param>
/// <param name="Labels">Each utterance's labels.</param>
/// <param name="Utterances">Each utterance's index in the training source.</param>
/// <param name="Step">The step (1 for the first).</param>
/// <param name="Seed">The run's seed.</param>
/// <param name="StepUtterances">The utterances of the whole step.</param>
/// <param name="StepFrames">Their frames.</param>
/// <param name="StepLabels">Their labels.</param>
public sealed record AdversarialSynthesisBatch(Tensor Features, IReadOnlyList<int> Frames, IReadOnlyList<int[]> Labels, IReadOnlyList<int> Utterances, int Step, int Seed,
    int StepUtterances, long StepFrames, long StepLabels);

/// <summary>
/// One micro-batch's part of a training step: first the discriminators' (their loss on the real sound and on the sound the
/// generator makes, detached), then, after the trainer has stepped the discriminators, the generator's (its losses with
/// the stepped discriminators). Each adds its share of the step's gradients and returns its share of the step's losses.
/// </summary>
public interface IAdversarialSynthesisStep : IDisposable
{
    /// <summary>Adds the discriminators' gradients of their least-squares loss over this micro-batch; returns its share of the step's discriminator loss.</summary>
    float Discriminate();

    /// <summary>
    /// Adds the generator's gradients of its loss over this micro-batch (the discriminators' parameters get none); returns
    /// this micro-batch's share of the step's terms: "mel", "kl", "duration", "adversarial", "feature_matching" (each
    /// weighted) and "generator" (their sum).
    /// </summary>
    IReadOnlyDictionary<string, float> Generate();
}

/// <summary>
/// A synthesizer trained against discriminators (plan 21, S9 route C), whatever its layers: the trainer
/// (<c>AdversarialSynthesisTrainer</c>) reads utterances through <see cref="Features"/>, steps the discriminators and the
/// generator in turn through <see cref="Begin"/>'s steps, and saves it with <see cref="Save"/> into a folder
/// <c>SpeechSynthesisModels.Load</c> reads, so the trained model speaks like any other and installs as a voice of kind
/// <see cref="VoiceKinds.Checkpoint"/>. The VITS plug-in's synthesizer is one (loaded with <see cref="SpeechSynthesisLoadOptions.Training"/>).
/// </summary>
public interface IAdversarialSpeechSynthesizer : ISpeechSynthesizer
{
    /// <summary>The architecture name its folder's config.json gives.</summary>
    string Architecture { get; }

    /// <summary>The generator's trained layers (their parameters).</summary>
    Module Generator { get; }

    /// <summary>
    /// The training rows of an utterance's sound, [frames, <see cref="FeatureWidth"/>] (the trainer caches them, so a
    /// second pass takes none): what the model reads its targets from (VITS: the linear spectrogram and the frame's samples).
    /// </summary>
    IAudioFeatureExtractor Features { get; }

    /// <summary>The width of a training row.</summary>
    int FeatureWidth { get; }

    /// <summary>The labels of <paramref name="text"/> as the model reads it.</summary>
    int[] Labels(string text);

    /// <summary>Whether an utterance of <paramref name="frames"/> frames and these labels can be trained on.</summary>
    bool Accepts(int frames, int[] labels, AdversarialSynthesisOptions options);

    /// <summary>Why the utterances <see cref="Accepts"/> refuses are left out, for the run's note.</summary>
    string Refused { get; }

    /// <summary>
    /// The discriminators trained beside the generator: the ones the model's folder holds, else the file
    /// <see cref="AdversarialSynthesisOptions.DiscriminatorWeights"/> names, else new ones of the options' description
    /// drawn from <paramref name="seed"/>; the same object on later calls.
    /// </summary>
    Module Discriminators(AdversarialSynthesisOptions options, int seed);

    /// <summary>
    /// A micro-batch's step (see <see cref="IAdversarialSynthesisStep"/>). With <paramref name="keep"/> the generator's pass
    /// of <see cref="IAdversarialSynthesisStep.Discriminate"/> is kept for <see cref="IAdversarialSynthesisStep.Generate"/>;
    /// otherwise each runs it (with the same draws).
    /// </summary>
    IAdversarialSynthesisStep Begin(AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options, bool keep);

    /// <summary>The batch's mean absolute log-mel error of the segments' sound (unweighted, gradients off): the evaluation's score.</summary>
    float MelError(AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options);

    /// <summary>The files of a saved folder only training reads (the discriminators), left out when the folder is installed as a voice.</summary>
    IReadOnlyList<string> TrainingFiles { get; }

    /// <summary>The generator's parts <see cref="Freeze"/> can keep as they are, in the order the model reads them (none: it trains whole).</summary>
    IReadOnlyList<string> Parts => [];

    /// <summary>
    /// Keeps <paramref name="parts"/> (of <see cref="Parts"/>) as they are for training: their weights take no gradient
    /// and leave <see cref="Generator"/>'s parameters, so the optimizer neither steps nor holds state for them; they are
    /// saved unchanged. Called once, before the trainer reads the parameters.
    /// </summary>
    /// <exception cref="ArgumentException">A part is not one of <see cref="Parts"/>.</exception>
    void Freeze(IReadOnlyCollection<string> parts)
    {
        if (parts.Count > 0)
        {
            throw new ArgumentException($"A {Architecture} model keeps no part fixed (it names no parts); train it whole, without frozen parts.", nameof(parts));
        }
    }

    /// <summary>Writes the model (and its discriminators, as <see cref="TrainingFiles"/>) as a folder <c>SpeechSynthesisModels.Load</c> reads, the weights as they are now.</summary>
    void Save(string folder);
}
