// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Idrak.Audio.Abstractions;

namespace Idrak.Audio;

/// <summary>
/// A synthesizer that also speaks in installed voices (plan 21, S9). <see cref="SpeechSynthesisOptions.Voice"/> resolves
/// in this order: a speaker of the synthesizer's checkpoint; else a voice of the store, by kind:
/// <see cref="VoiceKinds.Speaker"/> speaks the checkpoint speaker voice.json names, <see cref="VoiceKinds.Checkpoint"/>
/// loads the voice's own model folder and speaks with it, <see cref="VoiceKinds.Embedding"/> speaks in the synthesizer's
/// voice and converts the speech to the voice's embedding (<see cref="SpeechSynthesisOptions.VoiceConversion"/>). An
/// unknown name is refused, listing the checkpoint's speakers and the installed voices. Converters and checkpoints it
/// loads are kept for later calls and disposed with it.
/// </summary>
public sealed class VoicedSynthesizer : ISpeechSynthesizer
{
    private readonly ISpeechSynthesizer _synthesizer;
    private readonly IVoiceStore? _store;
    private readonly IVoiceConverter? _converter;
    private readonly bool _ownsSynthesizer;
    private readonly Dictionary<string, IVoiceConverter> _converters = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ISpeechSynthesizer> _checkpoints = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float[]> _embeddings = new(StringComparer.Ordinal);
    private readonly Lock _gate = new();
    private bool _disposed;

    /// <summary>
    /// Wraps <paramref name="synthesizer"/>. <paramref name="store"/>: where voices are looked up (null: the
    /// <see cref="VoiceStores"/> store the options name, "folder" by default). <paramref name="converter"/>: the converter
    /// for embedding voices made by its family (null: loaded from the folder voice.json names, on the synthesizer's device).
    /// </summary>
    public VoicedSynthesizer(ISpeechSynthesizer synthesizer, IVoiceStore? store = null, IVoiceConverter? converter = null, bool ownsSynthesizer = true)
    {
        ArgumentNullException.ThrowIfNull(synthesizer);
        _synthesizer = synthesizer;
        _store = store;
        _converter = converter;
        _ownsSynthesizer = ownsSynthesizer;
    }

    /// <summary>The synthesizer it speaks with.</summary>
    public ISpeechSynthesizer Synthesizer => _synthesizer;

    /// <inheritdoc />
    public Device Device => _synthesizer.Device;

    /// <inheritdoc />
    public int SampleRate => _synthesizer.SampleRate;

    /// <summary>The checkpoint's speakers, then the installed voices of the store (the given one, or "folder").</summary>
    public IReadOnlyList<string> Voices => [.. _synthesizer.Voices, .. (_store ?? VoiceStores.Get()).List().Select(v => v.Name)];

    // A speaker of the checkpoint: one of its voices (any case), or the index of one.
    private bool IsSpeaker(string name) => _synthesizer.Voices.Contains(name, StringComparer.OrdinalIgnoreCase)
        || int.TryParse(name, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int index) && index < _synthesizer.Voices.Count;

    /// <summary>How <paramref name="options"/>' voice resolves: who speaks, with what options, and the conversion after (null converter: none).</summary>
    public (ISpeechSynthesizer Speaker, SpeechSynthesisOptions Options, IVoiceConverter? Converter, float[]? Target) Resolve(SpeechSynthesisOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new SpeechSynthesisOptions();
        if (options.Voice is not { } name || IsSpeaker(name))
        {
            return (_synthesizer, options, null, null);
        }

        var store = _store ?? VoiceStores.Get(options.VoiceStore);
        var voice = store.Find(name) ?? throw new ArgumentException(
            $"Voice '{name}' is not a speaker of this model (speakers: {(_synthesizer.Voices.Count == 0 ? "none, it has one voice" : string.Join(", ", _synthesizer.Voices))}) "
            + $"nor installed in {store.Location} (installed: {(store.List() is { Count: > 0 } all ? string.Join(", ", all.Select(v => v.Name)) : "none")}).", nameof(options));
        switch (voice.Kind)
        {
            case VoiceKinds.Speaker:
                string speaker = voice.Manifest["speaker"]?.GetValue<string>() ?? throw new InvalidDataException($"Voice '{voice.Name}' is of kind speaker but its voice.json names no \"speaker\".");
                return (_synthesizer, options with { Voice = speaker }, null, null);
            case VoiceKinds.Checkpoint:
                return (Checkpoint(voice), options with { Voice = voice.Manifest["voice"]?.GetValue<string>() }, null, null);
            case VoiceKinds.Embedding:
                if (string.Equals(options.VoiceConversion, "off", StringComparison.OrdinalIgnoreCase))
                {
                    return (_synthesizer, options with { Voice = null }, null, null);
                }

                string family = voice.Encoder ?? throw new InvalidDataException($"Voice '{voice.Name}' names no encoder family in its voice.json (\"encoder.family\").");
                if (!string.Equals(options.VoiceConversion, "auto", StringComparison.OrdinalIgnoreCase) && !string.Equals(options.VoiceConversion, family, StringComparison.Ordinal))
                {
                    throw new ArgumentException($"Voice '{voice.Name}' was made by the encoder '{family}', and VoiceConversion asks for '{options.VoiceConversion}': "
                        + "a converter reads only its own family's embeddings (use \"auto\", or reinstall the voice with that family).", nameof(options));
                }

                var converter = Converter(voice, family);
                var target = Embedding(store, voice);
                if (target.Length != converter.EmbeddingSize)
                {
                    throw new InvalidDataException($"Voice '{voice.Name}' holds an embedding of {target.Length}, converter '{converter.Name}' takes {converter.EmbeddingSize}.");
                }

                return (_synthesizer, options with { Voice = null, SampleRate = null }, converter, target);
            default:
                throw new NotSupportedException($"Voice '{voice.Name}' is of kind '{voice.Kind}', which this library does not speak ({VoiceKinds.Speaker}, {VoiceKinds.Embedding}, {VoiceKinds.Checkpoint}).");
        }
    }

    /// <summary>Speaks <paramref name="text"/> in the options' voice; a converted voice is at the converter's rate unless the options set one.</summary>
    public (AudioData Audio, SynthesisStats Stats) Speak(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new SpeechSynthesisOptions();
        var (speaker, inner, converter, target) = Resolve(options);
        long started = Stopwatch.GetTimestamp();
        var (audio, stats) = speaker.Speak(text, inner, cancellationToken);
        if (converter is null)
        {
            return (audio, stats);
        }

        long converting = Stopwatch.GetTimestamp();
        audio = Finish(converter.Convert(audio, target!, options.Conversion ?? new VoiceConversionOptions { Seed = options.Seed }, cancellationToken), options);
        var conversion = Stopwatch.GetElapsedTime(converting);
        double seconds = audio.Duration.TotalSeconds;
        return (audio, stats with
        {
            Conversion = conversion,
            PeakDeviceBytes = Math.Max(stats.PeakDeviceBytes, converter.Device.Backend.GetMemoryUsage().Peak),
            RealTimeFactor = seconds > 0 ? Stopwatch.GetElapsedTime(started).TotalSeconds / seconds : 0,
        });
    }

    /// <summary>Speaks <paramref name="text"/> piece by piece in the options' voice, each piece converted as soon as it is made.</summary>
    public async IAsyncEnumerable<AudioData> SpeakStream(string text, SpeechSynthesisOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new SpeechSynthesisOptions();
        var (speaker, inner, converter, target) = Resolve(options);
        int index = 0;
        await foreach (var piece in speaker.SpeakStream(text, inner, cancellationToken).ConfigureAwait(false))
        {
            if (converter is null)
            {
                yield return piece;
                continue;
            }

            var conversion = options.Conversion ?? new VoiceConversionOptions { Seed = options.Seed };
            conversion = conversion with { Seed = conversion.Seed + index++ };
            yield return await Task.Run(() => Finish(converter.Convert(piece, target!, conversion, cancellationToken), options), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Disposes the converters and checkpoints it loaded, and the synthesizer when it owns it.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        lock (_gate)
        {
            foreach (var converter in _converters.Values) converter.Dispose();
            foreach (var checkpoint in _checkpoints.Values) checkpoint.Dispose();
            _converters.Clear();
            _checkpoints.Clear();
        }

        if (_ownsSynthesizer)
        {
            _synthesizer.Dispose();
        }
    }

    private static AudioData Finish(AudioData audio, SpeechSynthesisOptions options) =>
        options.SampleRate is { } rate && rate != audio.SampleRate ? Resampler.Resample(audio, rate) : audio;

    private IVoiceConverter Converter(VoiceProfile voice, string family)
    {
        if (_converter is not null && string.Equals(_converter.Name, family, StringComparison.Ordinal))
        {
            return _converter;
        }

        string folder = voice.EncoderFolder ?? throw new InvalidOperationException(
            $"Voice '{voice.Name}' needs a '{family}' converter and its voice.json names no folder to load one from; pass the converter to VoicedSynthesizer.");
        lock (_gate)
        {
            string key = family + "|" + folder;
            if (!_converters.TryGetValue(key, out var converter))
            {
                if (!Directory.Exists(folder) && !File.Exists(folder))
                {
                    throw new DirectoryNotFoundException($"Voice '{voice.Name}' was made with the '{family}' converter at {folder}, which is not there now; "
                        + "load the converter (VoiceConverters.Load) and pass it to VoicedSynthesizer.");
                }

                converter = VoiceConverters.Load(folder, new VoiceConverterLoadOptions { Device = Device, Family = family });
                _converters[key] = converter;
            }

            return converter;
        }
    }

    private float[] Embedding(IVoiceStore store, VoiceProfile voice)
    {
        lock (_gate)
        {
            string key = voice.Folder + "|" + voice.Manifest["created"];          // a voice installed again under the name is read again
            if (!_embeddings.TryGetValue(key, out var embedding))
            {
                embedding = store.ReadEmbedding(voice);
                _embeddings[key] = embedding;
            }

            return embedding;
        }
    }

    private ISpeechSynthesizer Checkpoint(VoiceProfile voice)
    {
        string folder = voice.CheckpointFolder!;
        lock (_gate)
        {
            if (!_checkpoints.TryGetValue(folder, out var model))
            {
                model = SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = Device });
                _checkpoints[folder] = model;
            }

            return model;
        }
    }
}
