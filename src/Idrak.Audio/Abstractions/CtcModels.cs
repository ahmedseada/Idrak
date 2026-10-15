// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio.Abstractions;

/// <summary>How <see cref="ICtcSpeechModel.Logits"/> runs a training pass.</summary>
public sealed record CtcForwardOptions
{
    /// <summary>Keep only each block's output and recompute the rest in the backward pass (activation checkpointing).</summary>
    public bool Checkpointing { get; init; }
}

/// <summary>
/// A speech recognizer with a CTC head: what <c>CtcTrainer</c> trains and saves. A family's recognizer implements it (the
/// IdrakAudio plug-in's "ctc" task does), so a model made by <c>SpeechModels.Create</c>, trained, and read back by
/// <c>SpeechModels.Load</c> is one kind of object throughout. The trainer knows nothing of the layers: it asks for logits
/// and labels and saves through <see cref="Save"/>.
/// </summary>
public interface ICtcSpeechModel : ISpeechRecognizer
{
    /// <summary>The architecture name its folder's config.json gives (its <see cref="SpeechFamilies"/> registration).</summary>
    string Architecture { get; }

    /// <summary>The trained layers (their parameters, and training or evaluation mode).</summary>
    Module Network { get; }

    /// <summary>How sound becomes the features <see cref="Logits"/> reads ([frames, mels] per utterance, no padding to a segment).</summary>
    IAudioFeatureExtractor Features { get; }

    /// <summary>The head's classes: the labels and the blank.</summary>
    int Classes { get; }

    /// <summary>The blank class.</summary>
    int Blank { get; }

    /// <summary>The steps the head gives for <paramref name="frames"/> feature frames (after the model's subsampling).</summary>
    int StepsFor(int frames);

    /// <summary>The labels of <paramref name="text"/> (classes other than the blank), as the model is trained to read it.</summary>
    int[] Labels(string text);

    /// <summary>The text of a sequence of labels (blanks dropped, repeats collapsed already).</summary>
    string Text(ReadOnlySpan<int> labels);

    /// <summary>
    /// The head's logits for a batch of features [batch, frames, mels] on <see cref="ISpeechRecognizer.Device"/>, each
    /// utterance's frames first and zeros after them (<paramref name="frames"/> gives how many are its own): [batch, steps,
    /// classes], an utterance's steps past <see cref="StepsFor"/> of its frames not read. An utterance gives the same logits
    /// in any batch (its padding is not attended to).
    /// </summary>
    Tensor Logits(Tensor features, IReadOnlyList<int> frames, CtcForwardOptions? options = null);

    /// <summary>
    /// Writes the model as a folder <c>SpeechModels.Load</c> reads (config.json, the weights as float32 safetensors, the
    /// tokenizer and the family's other files), the trained weights as they are now.
    /// </summary>
    void Save(string folder);
}
