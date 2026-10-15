// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// A text-to-speech model the speech trainer trains (<c>SpeechSynthesisTrainer</c>, plan 21 S6) and saves, whatever its
/// layers: a text encoder over the labels of a text, a duration predictor (each label's log frames), length regulation
/// to the frames and a decoder to a log-mel spectrogram, spoken through a <see cref="Vocoders"/> vocoder. The IdrakAudio
/// plug-in's "tts" task is one: made by <c>SpeechSynthesisModels.Create</c>, trained, and read back by
/// <c>SpeechSynthesisModels.Load</c>, it is one kind of object throughout. Its <see cref="Objective"/> (a
/// <see cref="SpeechSynthesisObjectives"/> name) gives the loss.
/// </summary>
public interface ITrainableSpeechSynthesizer : ISpeechSynthesizer
{
    /// <summary>The architecture name its folder's config.json gives (its <see cref="SpeechSynthesisFamilies"/> registration).</summary>
    string Architecture { get; }

    /// <summary>The <see cref="SpeechSynthesisObjectives"/> name it trains with ("durations").</summary>
    string Objective => SpeechSynthesisObjectives.Durations;

    /// <summary>The trained layers (their parameters, and training or evaluation mode).</summary>
    Module Network { get; }

    /// <summary>How sound becomes the log-mel frames the model makes ([frames, mels] per utterance, no padding to a segment): its training targets.</summary>
    IAudioFeatureExtractor Features { get; }

    /// <summary>The offset of the durations the predictor gives: it predicts log(frames + offset), so a label of no frames has a finite target.</summary>
    float DurationOffset { get; }

    /// <summary>The labels of <paramref name="text"/> (token ids), as the model reads it; at least one for a text with anything to say.</summary>
    int[] Labels(string text);

    /// <summary>
    /// The text encoder's states for <paramref name="labels"/> [batch, length] (label ids as floats, any value after a
    /// text's own labels) on <see cref="ISpeechSynthesizer.Device"/>, text n having <paramref name="lengths"/>[n] labels:
    /// [batch, length, width], a text's own positions the same in any batch (its padding is not attended to).
    /// </summary>
    Tensor Encode(Tensor labels, IReadOnlyList<int> lengths, SpeechForwardOptions? options = null);

    /// <summary>The duration predictor's log(frames + <see cref="DurationOffset"/>) of each label: [batch, length] from the encoder's states.</summary>
    Tensor LogDurations(Tensor hidden, IReadOnlyList<int> lengths);

    /// <summary>
    /// The log-mel spectrogram [batch, frames, mels] of the encoder's states with each label repeated for its
    /// <paramref name="durations"/> (text n's frames the sum of its durations, frames after them padding): the decoder sees
    /// each utterance's own frames only, so an utterance gives the same spectrogram in any batch.
    /// </summary>
    Tensor Decode(Tensor hidden, IReadOnlyList<int> lengths, IReadOnlyList<int[]> durations, SpeechForwardOptions? options = null);

    /// <summary>
    /// Writes the model as a folder <c>SpeechSynthesisModels.Load</c> reads (config.json, the weights as float32
    /// safetensors, the tokenizer and the family's other files), the trained weights as they are now.
    /// </summary>
    void Save(string folder);
}

/// <summary>
/// What a speech synthesis training objective does for the trainer: which utterances it can train on and the loss of a
/// batch. Registered in <see cref="SpeechSynthesisObjectives"/>; a model names its own
/// (<see cref="ITrainableSpeechSynthesizer.Objective"/>).
/// </summary>
public interface ISpeechSynthesisObjective
{
    /// <summary>The name it is registered under.</summary>
    string Name { get; }

    /// <summary>Whether <paramref name="model"/> is one it trains.</summary>
    bool Trains(ITrainableSpeechSynthesizer model);

    /// <summary>Whether an utterance of <paramref name="frames"/> target frames and these labels can be trained on.</summary>
    bool Accepts(ITrainableSpeechSynthesizer model, int frames, int[] labels);

    /// <summary>Why the utterances <see cref="Accepts"/> refuses are left out, for the run's note.</summary>
    string Refused { get; }

    /// <summary>
    /// The mean loss of a batch: the target log-mels [batch, frames, mels] on the model's device (each utterance's frames
    /// first, zeros after), each utterance's frames, labels and frames per label (summing to its frames), and the run's
    /// options (an objective's weights are settings there). An utterance's loss is the same in any batch.
    /// </summary>
    Tensor Loss(ITrainableSpeechSynthesizer model, Tensor features, IReadOnlyList<int> frames, IReadOnlyList<int[]> labels, IReadOnlyList<int[]> durations,
        SpeechForwardOptions forward, SpeechTrainingOptions options);
}

/// <summary>
/// The speech synthesis training objectives, by name (ordinal). Built in (Idrak.Audio): "durations"
/// (<see cref="Idrak.Audio.DurationSynthesisObjective"/>: the mean absolute log-mel error with the given durations, plus
/// the mean squared log-duration error weighted by <see cref="SpeechTrainingOptions.DurationLossWeight"/>). Register
/// another with <see cref="Register"/> (one under a built-in name takes its place until <see cref="Unregister"/>).
/// </summary>
public static class SpeechSynthesisObjectives
{
    /// <summary>"durations".</summary>
    public const string Durations = "durations";

    private static readonly SlotTable<string, ISpeechSynthesisObjective> Registry = BuiltIn();

    private static SlotTable<string, ISpeechSynthesisObjective> BuiltIn()
    {
        var table = new SlotTable<string, ISpeechSynthesisObjective>(nameof(SpeechSynthesisObjectives), comparer: StringComparer.Ordinal,
            unguarded: "an objective's loss belongs to its models, so two registrations cannot stand in for each other");
        table.RegisterDefault(Durations, new Idrak.Audio.DurationSynthesisObjective());
        return table;
    }

    /// <summary>Registers <paramref name="objective"/> under its <see cref="ISpeechSynthesisObjective.Name"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISpeechSynthesisObjective objective)
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
    public static ISpeechSynthesisObjective Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"Speech synthesis objective '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); add it with SpeechSynthesisObjectives.Register.");

    /// <summary>The library's objective <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ISpeechSynthesisObjective? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the objective <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}
