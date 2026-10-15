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
/// PyTorch's CTC loss over an <see cref="ICtcSpeechModel"/>'s head, CtcDecoders for evaluation) and "seq2seq"
/// (<see cref="Idrak.Audio.Seq2SeqObjective"/>: an <see cref="ISequenceSpeechModel"/>'s decoder read with teacher forcing,
/// each utterance's mean token cross-entropy; greedy or beam generation for evaluation). Register another with
/// <see cref="Register"/> (one under a built-in name takes its place until <see cref="Unregister"/>).
/// </summary>
public static class SpeechTrainingObjectives
{
    /// <summary>"ctc".</summary>
    public const string Ctc = "ctc";

    /// <summary>"seq2seq".</summary>
    public const string Seq2Seq = "seq2seq";

    private static readonly SlotTable<string, ISpeechTrainingObjective> Registry = BuiltIn();

    private static SlotTable<string, ISpeechTrainingObjective> BuiltIn()
    {
        var table = new SlotTable<string, ISpeechTrainingObjective>(nameof(SpeechTrainingObjectives), comparer: StringComparer.Ordinal,
            unguarded: "an objective's loss and decoding belong to its models, so two registrations cannot stand in for each other");
        table.RegisterDefault(Ctc, new Idrak.Audio.CtcObjective());
        table.RegisterDefault(Seq2Seq, new Idrak.Audio.Seq2SeqObjective());
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

/// <summary>
/// A speech recognizer that reads the sound with an encoder and writes the text with a decoder attending to it, trained
/// with the "seq2seq" objective (<see cref="Idrak.Audio.Seq2SeqObjective"/>): the decoder reads the start token and an
/// utterance's labels and is taught each next label and then the end token. The IdrakAudio plug-in's "seq2seq" task is one.
/// </summary>
public interface ISequenceSpeechModel : ITrainableSpeechModel
{
    /// <inheritdoc />
    string ITrainableSpeechModel.Objective => SpeechTrainingObjectives.Seq2Seq;

    /// <summary>The decoder's vocabulary (the head's outputs): the labels and the start and end tokens.</summary>
    int Vocabulary { get; }

    /// <summary>The token the decoder starts from.</summary>
    int StartToken { get; }

    /// <summary>The token that ends a text.</summary>
    int EndToken { get; }

    /// <summary>The decoder's positions: the start token, the labels and the end token fit in them (labels at most this less 1).</summary>
    int MaxPositions { get; }

    /// <summary>The encoder's steps for <paramref name="frames"/> feature frames (each utterance's memory length).</summary>
    int StepsFor(int frames);

    /// <summary>
    /// The encoder's output for a batch of features [batch, frames, mels] on <see cref="ISpeechRecognizer.Device"/>, each
    /// utterance's frames first and zeros after them: [batch, steps, width], utterance n's first <see cref="StepsFor"/> of
    /// its frames its own (the same in any batch), the rest padding the decoder does not attend to.
    /// </summary>
    Tensor Encode(Tensor features, IReadOnlyList<int> frames, SpeechForwardOptions? options = null);

    /// <summary>
    /// The decoder's last hidden states (before the head) for <paramref name="tokens"/> [batch, length] (token ids as floats)
    /// over <paramref name="memory"/> [batch, steps, width], sequence n attending to its first <paramref name="steps"/>[n]
    /// memory positions: [batch, length, decoder width]. Each position sees the tokens up to it only.
    /// </summary>
    Tensor Hidden(Tensor tokens, Tensor memory, IReadOnlyList<int> steps);

    /// <summary>The head: hidden states [rows, decoder width] to logits [rows, <see cref="Vocabulary"/>].</summary>
    Tensor Head(Tensor hidden);

    /// <summary>
    /// The labels the decoder writes for one utterance's memory [1, steps, width] (its own steps, no padding), from the start
    /// token to the end token (neither returned): greedy for one beam, else a beam search of <paramref name="beams"/>.
    /// </summary>
    int[] Generate(Tensor memory, int beams, CancellationToken cancellationToken = default);
}
