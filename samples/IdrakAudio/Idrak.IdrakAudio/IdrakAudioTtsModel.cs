// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Idrak.Abstraction.Diagnostics;
using Idrak.Audio;
using Idrak.Layers;
using Idrak.Models;

namespace Idrak.IdrakAudio;

/// <summary>
/// An IdrakAudio "tts" model: a synthesizer trained from scratch (plan 21, S6). The text's labels (the tokenizer's ids)
/// through the token embedding and sinusoidal positions, <see cref="IdrakAudioConfig.Layers"/> pre-norm blocks over each
/// text's own labels and a layer norm (the text's states); a duration predictor (two (1, k) convolutions over the labels,
/// each with a ReLU and a layer norm, the padding zeroed before each, and a linear to log(frames + offset)); the states
/// projected to the decoder's width when it differs and repeated for each label's frames (<see cref="LengthRegulation"/>),
/// sinusoidal positions, <see cref="IdrakAudioConfig.DecoderLayers"/> pre-norm blocks over each utterance's own frames, a
/// layer norm and a linear to the mels: a log-mel spectrogram in the layout of <see cref="Features"/>, spoken through a
/// <see cref="Vocoders"/> vocoder ("griffin-lim" unless the call, the config or <see cref="Vocoder"/> gives another of
/// the same mel layout). Trained by <c>SpeechSynthesisTrainer</c> with the "durations" objective.
/// </summary>
public sealed class IdrakAudioTtsModel : ITrainableSpeechSynthesizer
{
    private readonly IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> _layout;
    private readonly IdrakAudioTtsNetwork _network;
    private readonly ConcurrentDictionary<Rune, bool> _readable = new();
    private bool _disposed;

    internal IdrakAudioTtsModel(IdrakAudioConfig config, IdrakAudioWeights weights, BpeTokenizer tokenizer, Device device)
    {
        if (tokenizer.VocabularySize > config.Classes)
        {
            throw new InvalidDataException($"The IdrakAudio text embedding has {config.Classes} classes, fewer than the tokenizer's {tokenizer.VocabularySize} tokens.");
        }

        Config = config;
        Device = device;
        Tokenizer = tokenizer;
        _network = new IdrakAudioTtsNetwork(config, weights);
        _layout = [.. weights.Entries];
        Features = AudioFeatureExtractors.Create(config.FeatureExtractor, config.Features);
    }

    /// <summary>The settings.</summary>
    public IdrakAudioConfig Config { get; }

    /// <summary>The tokenizer: token i is label i.</summary>
    public BpeTokenizer Tokenizer { get; }

    /// <summary>
    /// A vocoder the app made (its own settings, a Griffin-Lim of more iterations say), used when a call names none
    /// (<see cref="SpeechSynthesisOptions.Vocoder"/>); null: the config's vocoder or the library's default.
    /// </summary>
    public IVocoder? Vocoder { get; set; }

    /// <summary>
    /// When speaking, leave out the characters the tokenizer has no token for (punctuation the transcripts did not hold,
    /// say), as a character-vocabulary synthesizer does (true, the default); false: such a text is refused naming the character.
    /// </summary>
    public bool DropUnknownCharacters { get; set; } = true;

    /// <inheritdoc />
    public string Architecture => IdrakAudioFamily.Architecture;

    /// <inheritdoc />
    public Module Network => _network;

    /// <inheritdoc />
    public IAudioFeatureExtractor Features { get; }

    /// <inheritdoc />
    public float DurationOffset => Config.DurationOffset;

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public int SampleRate => Features.Options.SampleRate;

    /// <inheritdoc />
    public IReadOnlyList<string> Voices { get; } = [];

    /// <inheritdoc />
    public int[] Labels(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return [.. Tokenizer.Encode(text)];
    }

    /// <inheritdoc />
    /// <remarks>The text's blocks keep their activations (a text's labels are far fewer than its frames); activation checkpointing covers the decoder's.</remarks>
    public Tensor Encode(Tensor labels, IReadOnlyList<int> lengths, SpeechForwardOptions? options = null) => _network.Encode(labels, lengths);

    /// <inheritdoc />
    public Tensor LogDurations(Tensor hidden, IReadOnlyList<int> lengths) => _network.LogDurations(hidden, lengths);

    /// <inheritdoc />
    public Tensor Decode(Tensor hidden, IReadOnlyList<int> lengths, IReadOnlyList<int[]> durations, SpeechForwardOptions? options = null) =>
        _network.Decode(hidden, lengths, durations, options?.Checkpointing ?? false);

    /// <summary>
    /// The log-mel spectrogram [frames, mels] (row after row) of <paramref name="labels"/>: the durations predicted (at
    /// <paramref name="rate"/>: 2 speaks twice as fast) or the given <paramref name="durations"/>, one device pass; with the
    /// frames per label used.
    /// </summary>
    /// <exception cref="ArgumentException">There is no label, or the durations are not one per label.</exception>
    public (float[] LogMel, int Frames, int[] Durations) Spectrogram(ReadOnlySpan<int> labels, float rate = 1f, int[]? durations = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (labels.IsEmpty)
        {
            throw new ArgumentException("There are no labels to speak.", nameof(labels));
        }

        if (durations is not null && durations.Length != labels.Length)
        {
            throw new ArgumentException($"{durations.Length} durations for {labels.Length} labels.", nameof(durations));
        }

        var ids = new float[labels.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = labels[i];
        }

        bool training = _network.IsTraining;
        _network.Eval();
        try
        {
            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            int[] lengths = [labels.Length];
            var hidden = _network.Encode(Tensor.From(ids, [1, ids.Length], Device), lengths);
            durations ??= LengthRegulation.Durations(_network.LogDurations(hidden, lengths).ToArray(), Config.DurationOffset, rate);
            var mel = _network.Decode(hidden, lengths, [durations], checkpointing: false);
            return (mel.ToArray(), mel.Shape[1], durations);
        }
        finally
        {
            _network.Train(training);
        }
    }

    /// <inheritdoc />
    public (AudioData Audio, SynthesisStats Stats) Speak(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new SpeechSynthesisOptions();
        Device.Backend.ResetPeakMemoryUsage();
        long started = Stopwatch.GetTimestamp();
        var vocoder = VocoderFor(options);
        var pieces = Pieces(text, options);
        var textTime = Stopwatch.GetElapsedTime(started);
        var samples = new List<float[]>(pieces.Count);
        TimeSpan model = TimeSpan.Zero, vocoding = TimeSpan.Zero;
        long total = 0;
        foreach (var labels in pieces)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (piece, modelTime, vocoderTime) = SpeakPiece(labels, options, vocoder);
            samples.Add(piece);
            total += piece.Length;
            model += modelTime;
            vocoding += vocoderTime;
        }

        var joined = new float[total];
        int at = 0;
        foreach (var piece in samples)
        {
            piece.CopyTo(joined, at);
            at += piece.Length;
        }

        var audio = new AudioData(joined, 1, SampleRate);
        if (options.SampleRate is { } rate && rate != SampleRate)
        {
            audio = Resampler.Resample(audio, rate);
        }

        double seconds = (double)total / SampleRate;
        var elapsed = Stopwatch.GetElapsedTime(started);
        return (audio, new SynthesisStats(textTime, model, vocoding, Device.Backend.GetMemoryUsage().Peak, seconds > 0 ? elapsed.TotalSeconds / seconds : 0));
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<AudioData> SpeakStream(string text, SpeechSynthesisOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new SpeechSynthesisOptions();
        var vocoder = VocoderFor(options);
        foreach (var labels in Pieces(text, options))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (samples, _, _) = await Task.Run(() => SpeakPiece(labels, options, vocoder), cancellationToken).ConfigureAwait(false);
            var audio = new AudioData(samples, 1, SampleRate);
            yield return options.SampleRate is { } rate && rate != SampleRate ? Resampler.Resample(audio, rate) : audio;
        }
    }

    // The labels of each piece of the text (SpeechSynthesisText's split), pieces without a label left out.
    private List<int[]> Pieces(string text, SpeechSynthesisOptions options)
    {
        if (options.Voice is not null)
        {
            throw new ArgumentException($"This model has one voice; '{options.Voice}' was asked for.", nameof(options));
        }

        if (!(options.Rate > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options), $"SpeechSynthesisOptions.Rate is positive, got {options.Rate}.");
        }

        var pieces = new List<int[]>();
        foreach (var piece in SpeechSynthesisText.Pieces(text, options.MaxCharacters))
        {
            var labels = Labels(DropUnknownCharacters ? Readable(piece) : piece);
            if (labels.Length > 0)
            {
                pieces.Add(labels);
            }
        }

        return pieces;
    }

    // The piece without the characters the tokenizer cannot read (each character tried once per model).
    private string Readable(string piece)
    {
        var kept = new StringBuilder(piece.Length);
        foreach (var rune in piece.EnumerateRunes())
        {
            if (Rune.IsWhiteSpace(rune) || _readable.GetOrAdd(rune, r => CanRead(r.ToString())))
            {
                kept.Append(rune.ToString());
            }
        }

        return kept.ToString();
    }

    private bool CanRead(string character)
    {
        try
        {
            return Tokenizer.Encode(character).Count > 0;
        }
        catch (InvalidDataException)
        {
            return false;
        }
    }

    // One piece: the spectrogram (a device pass), then the vocoder; the samples and the two times.
    private (float[] Samples, TimeSpan Model, TimeSpan Vocoder) SpeakPiece(int[] labels, SpeechSynthesisOptions options, IVocoder vocoder)
    {
        long started = Stopwatch.GetTimestamp();
        long telemetryStart = Telemetry.Start(TelemetryLevel.Inference);
        var (logMel, frames, _) = Spectrogram(labels, options.Rate);
        Telemetry.InferenceStage(InferenceStageKind.Encoder, Device, telemetryStart, labels.Length, (long)logMel.Length * sizeof(float));
        var modelTime = Stopwatch.GetElapsedTime(started);
        long vocoderStart = Stopwatch.GetTimestamp();
        using var scope = new TensorScope();
        var audio = vocoder.Synthesize(Tensor.From(logMel, [frames, Config.Mels], Device), Device);
        return (audio.Samples, modelTime, Stopwatch.GetElapsedTime(vocoderStart));
    }

    // The call's vocoder, else the app's (Vocoder), else the config's or the library's default, made for this model's mel
    // layout; one that reads another layout or makes another rate is refused.
    private IVocoder VocoderFor(SpeechSynthesisOptions options)
    {
        var vocoder = options.Vocoder is null && Vocoder is not null ? Vocoder : Vocoders.Create(options.Vocoder ?? Config.Vocoder, Features.Options);
        if (vocoder.Mels != Config.Mels || vocoder.SampleRate != SampleRate)
        {
            throw new NotSupportedException($"The vocoder '{vocoder.Name}' reads {vocoder.Mels} mel bands at {vocoder.SampleRate} Hz; this model makes {Config.Mels} at {SampleRate} Hz. "
                + "Give a vocoder of this model's mel layout (SpeechSynthesisOptions.Vocoder, the config's \"vocoder\"), or the library's \"griffin-lim\".");
        }

        return vocoder;
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

    /// <summary>Each parameter (in the order they were made) with its checkpoint name and whether it is stored transposed.</summary>
    internal IEnumerable<(string Name, Tensor Parameter, bool Transposed)> NamedParameters()
    {
        var parameters = _network.Parameters().ToList();
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
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _network.Dispose();
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"IdrakAudio tts (text {Config.Dim} wide, {Config.Layers} blocks; decoder {Config.DecoderDim} wide, {Config.DecoderLayers} blocks; {Config.Mels} mels) on {Device}";
}

/// <summary>
/// The layers of an IdrakAudio "tts" model, in the order their weights are made (and their parameters listed): the text
/// embedding and blocks ("text.embed", "text.layers.N", "text.norm"), the duration predictor ("duration.conv1",
/// "duration.norm1", "duration.conv2", "duration.norm2", "duration.linear"), the decoder's projection when the widths
/// differ ("decoder.proj"), blocks ("decoder.layers.N"), norm ("decoder.norm") and the mel head ("decoder.mel").
/// </summary>
internal sealed class IdrakAudioTtsNetwork : Module
{
    private readonly IdrakAudioConfig _config;
    private readonly Embedding _embed;
    private readonly IdrakAudioBlock[] _text;
    private readonly LayerNorm _textNorm;
    private readonly Conv2d _conv1, _conv2;
    private readonly LayerNorm _durationNorm1, _durationNorm2;
    private readonly Linear _durationLinear;
    private readonly Linear? _projection;
    private readonly IdrakAudioBlock[] _decoder;
    private readonly LayerNorm _decoderNorm;
    private readonly Linear _mel;
    private readonly Dropout? _dropout;
    private readonly SinusoidTable _textPositions, _framePositions;

    public IdrakAudioTtsNetwork(IdrakAudioConfig config, IdrakAudioWeights weights)
    {
        _config = config;
        int d = config.Dim, dd = config.DecoderDim, f = config.DurationFilters, k = config.DurationKernel;
        _embed = Embedding.FromWeights(weights.Table("text.embed.weight", config.Classes, d));
        _embed.Name = "text.embed";
        _text = new IdrakAudioBlock[config.Layers];
        for (int i = 0; i < _text.Length; i++)
        {
            _text[i] = new IdrakAudioBlock(config, weights, $"text.layers.{i}.");
        }

        _textNorm = weights.Norm("text.norm", d, config.NormEpsilon);
        _conv1 = Conv2d.FromWeights(weights.Convolution("duration.conv1.weight", f, d, k), weights.Bias("duration.conv1.bias", f), (1, k), (1, 1), (0, k / 2));
        _durationNorm1 = weights.Norm("duration.norm1", f, config.NormEpsilon);
        _conv2 = Conv2d.FromWeights(weights.Convolution("duration.conv2.weight", f, f, k), weights.Bias("duration.conv2.bias", f), (1, k), (1, 1), (0, k / 2));
        _durationNorm2 = weights.Norm("duration.norm2", f, config.NormEpsilon);
        _durationLinear = weights.Linear("duration.linear", f, 1);
        _projection = d != dd ? weights.Linear("decoder.proj", d, dd) : null;
        var decoderConfig = config with { Dim = dd, Heads = config.DecoderHeads, FfnDim = config.DecoderFfnDim };
        _decoder = new IdrakAudioBlock[config.DecoderLayers];
        for (int i = 0; i < _decoder.Length; i++)
        {
            _decoder[i] = new IdrakAudioBlock(decoderConfig, weights, $"decoder.layers.{i}.");
        }

        _decoderNorm = weights.Norm("decoder.norm", dd, config.NormEpsilon);
        _mel = weights.Linear("decoder.mel", dd, config.Mels);
        _dropout = config.Dropout > 0 ? new Dropout(config.Dropout) : null;
        _textPositions = new SinusoidTable(d);
        _framePositions = new SinusoidTable(dd);
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children()
    {
        var children = new List<Module> { _embed };
        children.AddRange(_text);
        children.AddRange([_textNorm, _conv1, _durationNorm1, _conv2, _durationNorm2, _durationLinear]);
        if (_projection is not null)
        {
            children.Add(_projection);
        }

        children.AddRange(_decoder);
        children.AddRange([_decoderNorm, _mel]);
        if (_dropout is not null)
        {
            children.Add(_dropout);
        }

        return children;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) =>
        throw new NotSupportedException("An IdrakAudio tts network runs in parts: the model's Encode, LogDurations and Decode (or Spectrogram).");

    // The text's states [n, L, dim]: embedding, positions, blocks over each text's own labels, norm. The blocks keep their
    // activations: checkpointed blocks read the layout set last in the forward pass, which is the decoder's.
    public Tensor Encode(Tensor labels, IReadOnlyList<int> lengths)
    {
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentNullException.ThrowIfNull(lengths);
        if (labels.Rank != 2 || labels.Shape[0] != lengths.Count)
        {
            throw new ArgumentException($"The IdrakAudio text encoder takes labels [{lengths.Count}, length], got {Tensor.FormatShape(labels.Shape)}.", nameof(labels));
        }

        int n = labels.Shape[0], length = labels.Shape[1];
        var own = Lengths(lengths, length);
        var x = _embed.Forward(labels);
        if (_config.Positions == "sinusoidal")
        {
            x = x + _textPositions.For(length, x.Device);
        }

        IdrakAudioEncoder.Use(new IdrakAudioEncoder.BatchLayout(n, length, own, x.Device));
        foreach (var block in _text)
        {
            x = block.Forward(x);
        }

        return _textNorm.Forward(x);
    }

    // log(frames + offset) of each label [n, L]: the padding zeroed before each convolution over the labels.
    public Tensor LogDurations(Tensor hidden, IReadOnlyList<int> lengths)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        int n = hidden.Shape[0], length = hidden.Shape[1];
        var own = Lengths(lengths, length);
        var keep = own.All(l => l == length) ? null : IdrakAudioEncoder.Keep(own, length, hidden.Device);
        var h = Drop(_durationNorm1.Forward(Convolve(_conv1, hidden, keep)));
        h = Drop(_durationNorm2.Forward(Convolve(_conv2, h, keep)));
        return _durationLinear.Forward(h).Reshape(n, length);
    }

    // The log-mel [n, frames, mels] of the states with each label repeated for its durations.
    public Tensor Decode(Tensor hidden, IReadOnlyList<int> lengths, IReadOnlyList<int[]> durations, bool checkpointing)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(durations);
        int length = hidden.Shape[1];
        var own = Lengths(lengths, length);
        for (int b = 0; b < own.Length; b++)
        {
            if (durations[b].Length != own[b])
            {
                throw new ArgumentException($"Text {b} has {own[b]} labels and {durations[b].Length} durations.", nameof(durations));
            }
        }

        var states = _projection is null ? hidden : _projection.Forward(hidden);   // projected per label, before the repeat
        var (x, frames) = LengthRegulation.Expand(states, durations);
        int n = x.Shape[0], t = x.Shape[1];
        if (_config.Positions == "sinusoidal")
        {
            x = x + _framePositions.For(t, x.Device);
        }

        IdrakAudioEncoder.Use(new IdrakAudioEncoder.BatchLayout(n, t, frames, x.Device));   // left set: the checkpointed blocks recompute with it
        foreach (var block in _decoder)
        {
            x = checkpointing && IsTraining ? block.ForwardCheckpointed(x) : block.Forward(x);
        }

        return _mel.Forward(_decoderNorm.Forward(x));
    }

    // A (1, k) convolution over the labels of rows [n, L, c] (the padding zeroed first when `keep` is given), then a ReLU: [n, L, filters].
    private static Tensor Convolve(Conv2d convolution, Tensor rows, Tensor? keep)
    {
        int n = rows.Shape[0], length = rows.Shape[1], channels = rows.Shape[2];
        if (keep is not null)
        {
            rows = rows.GroupAffine(keep, null, n * length, channels);
        }

        var y = convolution.Forward(rows.Permute(0, 2, 1).Reshape(n, channels, 1, length)).Relu();
        return y.Reshape(n, y.Shape[1], length).Permute(0, 2, 1);
    }

    private Tensor Drop(Tensor x) => _dropout is null ? x : _dropout.Forward(x);

    private static int[] Lengths(IReadOnlyList<int> lengths, int length)
    {
        ArgumentNullException.ThrowIfNull(lengths);
        var own = new int[lengths.Count];
        for (int i = 0; i < own.Length; i++)
        {
            own[i] = lengths[i];
            if (own[i] < 1 || own[i] > length)
            {
                throw new ArgumentException($"Text {i} has {own[i]} labels; the batch holds 1 to {length}.", nameof(lengths));
            }
        }

        return own;
    }

    // Every layer through its own Dispose (the blocks' key ranges, the position tables), then the parameters.
    public override void Dispose()
    {
        _textPositions.Dispose();
        _framePositions.Dispose();
        foreach (var module in Children())
        {
            module.Dispose();
        }

        base.Dispose();
    }
}

/// <summary>
/// sin(p / 10000^(2⌊i/2⌋/width)) at even i, cos at odd, [rows, width]: built for twice the longest length asked so far
/// and narrowed, so a table is made a few times at most (rule 75); a table outgrown stays alive until Dispose (another
/// thread may be reading it).
/// </summary>
internal sealed class SinusoidTable(int width) : IDisposable
{
    private readonly Lock _gate = new();
    private readonly List<Tensor> _tables = [];
    private Tensor? _current;

    public Tensor For(int rows, Device device)
    {
        Tensor table;
        lock (_gate)
        {
            if (_current is null || _current.Shape[0] < rows || _current.Device != device)
            {
                int made = Math.Max(rows, 2 * (_current?.Shape[0] ?? 0));
                var values = new float[made * width];
                for (int p = 0; p < made; p++)
                {
                    for (int i = 0; i < width; i++)
                    {
                        double angle = p / Math.Pow(10000, 2 * (i / 2) / (double)width);
                        values[p * width + i] = (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
                    }
                }

                _current = Tensor.Persistent(values, [made, width], device);
                _tables.Add(_current);
            }

            table = _current;
        }

        return table.Shape[0] == rows ? table : table.Narrow(0, 0, rows);
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _tables.ForEach(t => t.Dispose());
            _tables.Clear();
            _current = null;
        }
    }
}
