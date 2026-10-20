// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json;
using Idrak.Audio;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;

namespace Idrak.Audio.Plugins.IdrakAudio;

/// <summary>
/// An IdrakAudio "seq2seq" model: the encoder (<see cref="IdrakAudioEncoder"/> without the CTC head) and a text decoder
/// attending to it (<see cref="IdrakAudioDecoder"/>), its tokenizer and its features, as a speech recognizer (greedy, or a
/// beam search when <see cref="TranscriptionOptions.Beams"/> is above 1, from the start token to the end token) and as a
/// model <c>SpeechTrainer</c> trains with the "seq2seq" objective and saves. Token i of the tokenizer is class i; the
/// start and end tokens are the config's (by default one class past the tokenizer's tokens).
/// </summary>
public sealed class IdrakAudioSeq2SeqModel : ISequenceSpeechModel
{
    private readonly IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> _layout;
    private readonly Module[] _body;
    private readonly Linear _head;
    private readonly CrossAttention[] _crosses;
    private readonly IdrakAudioTranscriber _transcriber;

    internal IdrakAudioSeq2SeqModel(IdrakAudioConfig config, IdrakAudioWeights weights, IPretrainedTokenizer tokenizer, Device device)
    {
        if (tokenizer.VocabularySize > config.Classes)
        {
            throw new InvalidDataException($"The IdrakAudio decoder has {config.Classes} classes, fewer than the tokenizer's {tokenizer.VocabularySize} tokens.");
        }

        Config = config;
        Device = device;
        Tokenizer = tokenizer;
        Encoder = new IdrakAudioEncoder(config, weights);
        try
        {
            (Decoder, _head) = IdrakAudioDecoder.Build(config, weights);
        }
        catch
        {
            Encoder.Dispose();
            throw;
        }

        Network = new Seq2SeqNetwork(Encoder, Decoder);
        _body = [.. Decoder.Take(Decoder.Count() - 1)];
        _crosses = [.. Decoder.Descendants().OfType<CrossAttention>()];
        _layout = [.. weights.Entries];
        Features = AudioFeatureExtractors.Create(config.FeatureExtractor, config.Features);
        _transcriber = new IdrakAudioTranscriber(config, Features, device, Read, labels => Text(labels));
    }

    /// <summary>The settings.</summary>
    public IdrakAudioConfig Config { get; }

    /// <summary>The encoder (its normed states, no head).</summary>
    public IdrakAudioEncoder Encoder { get; }

    /// <summary>The decoder: embedding, positions, blocks, norm and head, for <see cref="DecodingContext"/> steps.</summary>
    public Sequential Decoder { get; }

    /// <summary>The tokenizer: token i is class i.</summary>
    public IPretrainedTokenizer Tokenizer { get; }

    /// <inheritdoc />
    public string Architecture => IdrakAudioFamily.Architecture;

    /// <inheritdoc />
    public Module Network { get; }

    /// <inheritdoc />
    public IAudioFeatureExtractor Features { get; }

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public IReadOnlySet<string> Languages { get; } = new HashSet<string>();

    /// <inheritdoc />
    public int Vocabulary => Config.Classes;

    /// <inheritdoc />
    public int StartToken => Config.StartToken;

    /// <inheritdoc />
    public int EndToken => Config.EndToken;

    /// <inheritdoc />
    public int MaxPositions => Config.MaxTargetPositions;

    /// <inheritdoc />
    public int StepsFor(int frames) => Config.StepsFor(frames);

    /// <inheritdoc />
    public int[] Labels(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return [.. Tokenizer.Encode(text)];
    }

    /// <inheritdoc />
    public string Text(ReadOnlySpan<int> labels)
    {
        var ids = new List<int>(labels.Length);
        foreach (int label in labels)
        {
            if (label != StartToken && label != EndToken && label < Tokenizer.VocabularySize)
            {
                ids.Add(label);
            }
        }

        return Tokenizer.Decode(ids);
    }

    /// <inheritdoc />
    public Tensor Encode(Tensor features, IReadOnlyList<int> frames, SpeechForwardOptions? options = null) =>
        Encoder.Run(features, frames, options?.Checkpointing ?? false);

    /// <inheritdoc />
    /// <remarks>
    /// Activation checkpointing (<see cref="SpeechForwardOptions.Checkpointing"/>) covers the encoder's blocks; the
    /// decoder's keep their activations (a text's tokens are far fewer than its sound's steps).
    /// </remarks>
    public Tensor Hidden(Tensor tokens, Tensor memory, IReadOnlyList<int> steps)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(memory);
        using var scope = EncoderMemory.Use(memory, steps);
        var x = tokens;
        foreach (var module in _body)
        {
            x = module.Forward(x);
        }

        return x;
    }

    /// <inheritdoc />
    public Tensor Head(Tensor hidden) => _head.Forward(hidden);

    /// <inheritdoc />
    public int[] Generate(Tensor memory, int beams, CancellationToken cancellationToken = default) =>
        [.. Search(memory, new BeamOptions { Beams = Math.Max(1, beams), MaxTokens = MaxPositions - 1, EndToken = EndToken }, cancellationToken).Tokens];

    // The best hypothesis from the start token over one utterance's memory [1, steps, dim], the memory projected once.
    private BeamHypothesis Search(Tensor memory, BeamOptions options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(memory);
        using var noGrad = Autograd.NoGrad();
        using var context = new DecodingContext(Device, options.Beams, MaxPositions) { LastPositionOnly = true, Cancellation = cancellationToken };
        foreach (var cross in _crosses)
        {
            cross.SetMemory(memory, context);
        }

        return BeamSearch.Decode((ids, ctx) => Decoder.ForwardCached(ids, ctx), context, [StartToken], options, null, cancellationToken)[0];
    }

    /// <inheritdoc />
    public void Save(string folder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "config.json"), Config.ToJson().ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), Export(), SafeTensorType.F32);
        Tokenizer.Save(Path.Combine(folder, "tokenizer.json"));
    }

    /// <summary>Each parameter (the encoder's, then the decoder's, in the order they were made) with its checkpoint name and whether it is stored transposed.</summary>
    internal IEnumerable<(string Name, Tensor Parameter, bool Transposed)> NamedParameters()
    {
        var parameters = Network.Parameters().ToList();
        if (parameters.Count != _layout.Count)
        {
            throw new InvalidOperationException($"The network holds {parameters.Count} parameters; {_layout.Count} were read.");
        }

        for (int i = 0; i < parameters.Count; i++)
        {
            yield return (_layout[i].Name, parameters[i], _layout[i].Transposed);
        }
    }

    // The parameters as they are now, under their checkpoint names and layouts.
    internal IEnumerable<(string Name, int[] Shape, float[] Values)> Export() =>
        NamedParameters().Select((p, i) => (p.Name, _layout[i].Stored, IdrakAudioWeights.Stored(p.Parameter, p.Transposed)));

    /// <inheritdoc />
    public Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default) =>
        _transcriber.Transcribe(audio, options, progress, cancellationToken);

    /// <inheritdoc />
    public IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null, CancellationToken cancellationToken = default) =>
        _transcriber.TranscribeStream(chunks, options, cancellationToken);

    // One stretch's features read: the encoder, then the decoder from the start token (greedy, or a beam search above one beam).
    private (int[] Labels, float Score) Read(Tensor features, TranscriptionOptions options, ref TimeSpan encoder, ref TimeSpan decoder)
    {
        var clock = Stopwatch.StartNew();
        var memory = Encoder.Run(features, null, checkpointing: false);
        encoder += clock.Elapsed;
        clock.Restart();
        var best = Search(memory, new BeamOptions
        {
            Beams = Math.Max(1, options.Beams),
            LengthPenalty = options.LengthPenalty,
            MaxTokens = options.MaxTokens > 0 ? Math.Min(options.MaxTokens, MaxPositions - 1) : MaxPositions - 1,
            MinTokens = options.MinTokens,
            EndToken = EndToken,
        }, default);
        decoder += clock.Elapsed;
        int[] labels = [.. best.Tokens];
        return (labels, best.LogProb / (labels.Length + 1));
    }

    /// <inheritdoc />
    public void Dispose() => Network.Dispose();

    /// <inheritdoc />
    public override string ToString() =>
        $"IdrakAudio seq2seq (encoder {Config.Dim} wide, {Config.Layers} blocks; decoder {Config.DecoderDim} wide, {Config.DecoderLayers} blocks; {Config.Classes} classes) on {Device}";

    // The trained layers: the encoder, then the decoder (their parameters in that order, as the weights were made).
    private sealed class Seq2SeqNetwork(IdrakAudioEncoder encoder, Sequential decoder) : Module
    {
        public override IEnumerable<Module> Children() => [encoder, decoder];

        protected override Tensor ForwardCore(Tensor input) =>
            throw new NotSupportedException("An IdrakAudio seq2seq network runs in two parts: the model's Encode, then its Hidden and Head (or Generate).");

        // Every layer through its own Dispose (the encoder's position tables, the attention layers' masks and key ranges),
        // which a container's parameters-and-buffers Dispose does not reach; disposing a tensor twice is a no-op.
        public override void Dispose()
        {
            encoder.Dispose();
            foreach (var module in decoder.Descendants().Reverse())
            {
                module.Dispose();
            }

            base.Dispose();
        }
    }
}
