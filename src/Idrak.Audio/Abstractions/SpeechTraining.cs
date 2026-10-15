// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Audio.Abstractions;

/// <summary>How a trainable speech model runs a training pass (<see cref="ICtcSpeechModel.Logits"/>, a sequence model's logits).</summary>
public sealed record SpeechForwardOptions
{
    /// <summary>Keep only each block's output and recompute the rest in the backward pass (activation checkpointing).</summary>
    public bool Checkpointing { get; init; }
}

/// <summary>
/// A speech model the speech trainer (<c>SpeechTrainer</c>) trains and saves, whatever its objective: a family's
/// recognizer implements it (the IdrakAudio plug-in's tasks do), so a model made by <c>SpeechModels.Create</c>, trained,
/// and read back by <c>SpeechModels.Load</c> is one kind of object throughout. The trainer knows nothing of the layers:
/// the model's <see cref="Objective"/> (a <see cref="SpeechTrainingObjectives"/> name) gives the loss and the decoding.
/// </summary>
public interface ITrainableSpeechModel : ISpeechRecognizer
{
    /// <summary>The architecture name its folder's config.json gives (its <see cref="SpeechFamilies"/> registration).</summary>
    string Architecture { get; }

    /// <summary>The <see cref="SpeechTrainingObjectives"/> name it trains with ("ctc", "seq2seq").</summary>
    string Objective { get; }

    /// <summary>The trained layers (their parameters, and training or evaluation mode).</summary>
    Module Network { get; }

    /// <summary>How sound becomes the features the model reads ([frames, mels] per utterance, no padding to a segment).</summary>
    IAudioFeatureExtractor Features { get; }

    /// <summary>The labels of <paramref name="text"/> (token ids, special ones aside), as the model is trained to read it.</summary>
    int[] Labels(string text);

    /// <summary>The text of a sequence of labels.</summary>
    string Text(ReadOnlySpan<int> labels);

    /// <summary>
    /// Writes the model as a folder <c>SpeechModels.Load</c> reads (config.json, the weights as float32 safetensors, the
    /// tokenizer and the family's other files), the trained weights as they are now.
    /// </summary>
    void Save(string folder);
}

/// <summary>
/// What a speech training objective does for the trainer: which utterances it can train on, the loss of a batch, and the
/// labels it reads for evaluation. Registered in <see cref="SpeechTrainingObjectives"/>; a model names its own
/// (<see cref="ITrainableSpeechModel.Objective"/>).
/// </summary>
public interface ISpeechTrainingObjective
{
    /// <summary>The name it is registered under.</summary>
    string Name { get; }

    /// <summary>Whether <paramref name="model"/> is one it trains (CTC: an <see cref="ICtcSpeechModel"/>).</summary>
    bool Trains(ITrainableSpeechModel model);

    /// <summary>Whether an utterance of <paramref name="frames"/> feature frames and these labels can be trained on.</summary>
    bool Accepts(ITrainableSpeechModel model, int frames, int[] labels);

    /// <summary>Why the utterances <see cref="Accepts"/> refuses are left out, for the run's note ("give fewer steps than their labels need").</summary>
    string Refused { get; }

    /// <summary>
    /// The mean loss of a batch: features [batch, frames, mels] on the model's device (each utterance's frames first, zeros
    /// after), each utterance's frames and labels. An utterance's loss is the same in any batch.
    /// </summary>
    Tensor Loss(ITrainableSpeechModel model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, SpeechForwardOptions forward);

    /// <summary>Each utterance's labels as the model reads them (gradients off), for scoring against its transcript.</summary>
    IReadOnlyList<int[]> Decode(ITrainableSpeechModel model, Tensor features, IReadOnlyList<int> frames, string decoder, int beamWidth);
}

/// <summary>
/// The speech training objectives, by name (ordinal). Built in (Idrak.Audio): "ctc" (<see cref="Idrak.Audio.CtcObjective"/>:
/// PyTorch's CTC loss over an <see cref="ICtcSpeechModel"/>'s head, CtcDecoders for evaluation). Register another with
/// <see cref="Register"/> (one under a built-in name takes its place until <see cref="Unregister"/>).
/// </summary>
public static class SpeechTrainingObjectives
{
    /// <summary>"ctc".</summary>
    public const string Ctc = "ctc";

    private static readonly SlotTable<string, ISpeechTrainingObjective> Registry = BuiltIn();

    private static SlotTable<string, ISpeechTrainingObjective> BuiltIn()
    {
        var table = new SlotTable<string, ISpeechTrainingObjective>(nameof(SpeechTrainingObjectives), comparer: StringComparer.Ordinal,
            unguarded: "an objective's loss and decoding belong to its models, so two registrations cannot stand in for each other");
        table.RegisterDefault(Ctc, new Idrak.Audio.CtcObjective());
        return table;
    }

    /// <summary>Registers <paramref name="objective"/> under its <see cref="ISpeechTrainingObjective.Name"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISpeechTrainingObjective objective)
    {
        ArgumentNullException.ThrowIfNull(objective);
        ArgumentException.ThrowIfNullOrEmpty(objective.Name);
        Registry.Register(objective.Name, objective, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's objective <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The objective <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under it; the message names those that are.</exception>
    public static ISpeechTrainingObjective Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"Speech training objective '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); add it with SpeechTrainingObjectives.Register.");

    /// <summary>The library's objective <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ISpeechTrainingObjective? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the objective <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}

/// <summary>
/// A speech recognizer with a CTC head, trained with the "ctc" objective (<see cref="Idrak.Audio.CtcObjective"/>): the
/// IdrakAudio plug-in's "ctc" task is one.
/// </summary>
public interface ICtcSpeechModel : ITrainableSpeechModel
{
    /// <inheritdoc />
    string ITrainableSpeechModel.Objective => SpeechTrainingObjectives.Ctc;

    /// <summary>The head's classes: the labels and the blank.</summary>
    int Classes { get; }

    /// <summary>The blank class.</summary>
    int Blank { get; }

    /// <summary>The steps the head gives for <paramref name="frames"/> feature frames (after the model's subsampling).</summary>
    int StepsFor(int frames);

    /// <summary>
    /// The head's logits for a batch of features [batch, frames, mels] on <see cref="ISpeechRecognizer.Device"/>, each
    /// utterance's frames first and zeros after them (<paramref name="frames"/> gives how many are its own): [batch, steps,
    /// classes], an utterance's steps past <see cref="StepsFor"/> of its frames not read. An utterance gives the same logits
    /// in any batch (its padding is not attended to).
    /// </summary>
    Tensor Logits(Tensor features, IReadOnlyList<int> frames, SpeechForwardOptions? options = null);
}

