// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Idrak.Audio.Abstractions;

/// <summary>What a sound file's header says.</summary>
/// <param name="SampleRate">Samples a second.</param>
/// <param name="Channels">Channels, interleaved in the file.</param>
/// <param name="Samples">Samples per channel; -1 when the header read does not reach the sound's data (<see cref="AudioCodecs.ReadInfo(string)"/> then asks again with the whole file).</param>
/// <param name="BitsPerSample">Bits per sample as stored (8, 16, 24, 32, 64).</param>
/// <param name="IsFloat">Whether the samples are stored as floating point.</param>
/// <param name="Format">The codec's name, e.g. "wav".</param>
public readonly record struct AudioInfo(int SampleRate, int Channels, long Samples, int BitsPerSample, bool IsFloat, string Format);

/// <summary>
/// Reads one sound format; register it with <see cref="AudioCodecs.Register"/>. A codec decides from a file's first
/// bytes whether the file is in its format, so the extension only decides which files a folder source picks up.
/// </summary>
public interface IAudioCodec
{
    /// <summary>The format's name, e.g. "wav" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions of the format, with the dot, e.g. ".wav".</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// The rate, channels and length from the start of a file (its first <see cref="AudioCodecs.HeaderBytes"/>, or all
    /// of a smaller file), or null when the bytes are not this format. A file whose data chunk lies further in answers
    /// -1 samples; <see cref="AudioCodecs.ReadInfo(string)"/> (and its in-memory sibling) then asks again with the whole file.
    /// </summary>
    AudioInfo? ReadInfo(ReadOnlySpan<byte> header);

    /// <summary>
    /// Decodes a whole file: a damaged file is an <see cref="InvalidDataException"/>, a valid file in a variant the codec
    /// does not read (a compressed WAV, say) a <see cref="NotSupportedException"/>.
    /// </summary>
    AudioData Decode(ReadOnlySpan<byte> file);
}

/// <summary>
/// The sound formats the library reads, by name (ignoring case). Built in (Idrak.Audio): "wav" (RIFF/WAVE: PCM 8, 16,
/// 24 and 32 bits, float 32 and 64, WAVE_FORMAT_EXTENSIBLE; not compressed variants or RF64). Register a codec for
/// another format with <see cref="Register"/>; an app's codec over a library one falls back on it or is compared with it
/// by its policy.
/// </summary>
public static class AudioCodecs
{
    /// <summary>The bytes of a file's start that <see cref="IAudioCodec.ReadInfo"/> receives.</summary>
    public const int HeaderBytes = 64 * 1024;

    // The most recently registered first; the library's wav, flac and mp3 behind them (registered as a library default on
    // the registry's first use).
    private static readonly SlotTable<string, IAudioCodec> Registry = BuiltIn();

    private static SlotTable<string, IAudioCodec> BuiltIn()
    {
        var table = new SlotTable<string, IAudioCodec>(nameof(AudioCodecs), (slot, app, library) => new GuardedCodec(slot, app, library), StringComparer.OrdinalIgnoreCase, newestFirst: true);
        table.RegisterDefault("wav", new WavCodec());
        table.RegisterDefault("flac", new FlacCodec());
        table.RegisterDefault("mp3", new Mp3Codec());
        return table;
    }

    /// <summary>
    /// Registers <paramref name="codec"/>: one of a registered name takes its place (a built-in stays behind it as its
    /// fallback, see <see cref="SetPolicy"/>), a new one is asked first.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioCodec codec)
    {
        ArgumentNullException.ThrowIfNull(codec);
        ArgumentException.ThrowIfNullOrWhiteSpace(codec.Name);
        Registry.Register(codec.Name, codec, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's codec <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered codec names, the first asked first.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The extensions of the registered codecs (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions => [.. Registry.Values.SelectMany(c => c.Extensions).Select(e => e.ToLowerInvariant()).Distinct()];

    /// <summary>The codec registered as <paramref name="name"/> (any case).</summary>
    /// <exception cref="NotSupportedException">None is; the message names those that are.</exception>
    public static IAudioCodec Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No audio codec '{name}' is registered ({string.Join(", ", Names)}); add it with AudioCodecs.Register.");

    /// <summary>The library's codec <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IAudioCodec? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the codec <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's codec <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the
    /// error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs
    /// beside the library's (<see cref="SlotPolicy.Shadow"/>: the library's answer is used, the two are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    // The codecs in the order a file is offered to them: those that list the path's extension first (a .wav file is the
    // WAV codec's before a codec that finds a frame sync inside its samples is asked), then the rest, each group in the
    // registry's order.
    private static IEnumerable<IAudioCodec> ForExtension(string path)
    {
        string extension = Path.GetExtension(path);
        if (extension.Length == 0)
        {
            return Registry.Values;
        }

        var listing = Registry.Values.Where(c => c.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase)).ToList();
        return listing.Count == 0 ? Registry.Values : listing.Concat(Registry.Values.Where(c => !listing.Contains(c)));
    }

    /// <summary>Whether a registered codec lists the extension of <paramref name="path"/>.</summary>
    public static bool CanDecode(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Length > 0 && Registry.Values.Any(c => c.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase));
    }

    /// <summary>The rate, channels and length of a sound file, or null when no registered codec knows its format.</summary>
    public static AudioInfo? ReadInfo(string path)
    {
        byte[] head = ArrayPool<byte>.Shared.Rent(HeaderBytes);              // one per file of a folder: pooled, not allocated
        try
        {
            int length;
            using (var stream = File.OpenRead(path))
            {
                length = stream.ReadAtLeast(head.AsSpan(0, HeaderBytes), HeaderBytes, throwOnEndOfStream: false);
            }

            byte[]? whole = null;
            foreach (var codec in ForExtension(path))
            {
                if (codec.ReadInfo(head.AsSpan(0, length)) is not { } info)
                {
                    continue;
                }

                // The data lies past the first bytes (a long LIST chunk, a large ID3 tag): ask with all of them; a codec that
                // then finds the file is not its own leaves it to the next.
                if (info.Samples >= 0 || length < HeaderBytes)
                {
                    return info;
                }

                whole ??= File.ReadAllBytes(path);
                if (codec.ReadInfo(whole) is { } full)
                {
                    return full;
                }
            }

            return null;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(head);
        }
    }

    /// <summary>
    /// The rate, channels and length of a sound file held in memory, or null when no registered codec knows its format:
    /// from its first bytes, or all of them when the data lies further on. Nothing is decoded.
    /// </summary>
    public static AudioInfo? ReadInfo(ReadOnlySpan<byte> file)
    {
        var header = file[..Math.Min(file.Length, HeaderBytes)];
        foreach (var codec in Registry.Values)
        {
            if (codec.ReadInfo(header) is not { } info)
            {
                continue;
            }

            if (info.Samples >= 0 || file.Length == header.Length)
            {
                return info;
            }

            if (codec.ReadInfo(file) is { } full)
            {
                return full;                                                    // not its own after all: the next codec is asked
            }
        }

        return null;
    }

    /// <summary>Decodes a sound file with the codec that knows its format.</summary>
    /// <exception cref="InvalidDataException">No registered codec reads the file, or it is damaged; the message names the file.</exception>
    /// <exception cref="NotSupportedException">The file is in a variant its codec does not read; the message names the file.</exception>
    public static AudioData Decode(string path)
    {
        // The file's bytes in a pooled buffer: a data loader decodes one file per sample, most past the large-object size.
        using var stream = File.OpenRead(path);
        long size = stream.Length;
        if (size > Array.MaxLength)
        {
            throw new IOException($"{path}: {size} bytes is too large for a sound file.");
        }

        byte[] bytes = ArrayPool<byte>.Shared.Rent((int)size);
        try
        {
            long start = Telemetry.Start(TelemetryLevel.Inference);
            int length = stream.ReadAtLeast(bytes.AsSpan(0, (int)size), (int)size, throwOnEndOfStream: false);
            var audio = Decode(bytes.AsSpan(0, length), path);
            Telemetry.InferenceStage(InferenceStageKind.AudioDecode, null, start, (long)audio.Channels * audio.Length, length);
            return audio;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    /// <summary>Decodes a sound file held in memory (a chat message's audio, a download) with the codec that knows its format.</summary>
    /// <exception cref="InvalidDataException">No registered codec reads the bytes, or they are damaged.</exception>
    public static AudioData Decode(ReadOnlySpan<byte> file) => Decode(file, "audio");

    private static AudioData Decode(ReadOnlySpan<byte> bytes, string path)
    {
        var header = bytes[..Math.Min(bytes.Length, HeaderBytes)];
        foreach (var codec in ForExtension(path))
        {
            // A codec that knows the first bytes decodes the file; one that needs all of them to be sure (a large tag
            // before the sound) and then finds it is not its own leaves it to the next.
            if (codec.ReadInfo(header) is { } info && (info.Samples >= 0 || bytes.Length == header.Length || codec.ReadInfo(bytes) is not null))
            {
                try
                {
                    return codec.Decode(bytes);
                }
                catch (Exception e) when (e is IndexOutOfRangeException or ArgumentOutOfRangeException or ArgumentException or EndOfStreamException)
                {
                    throw new InvalidDataException($"{path}: a damaged {codec.Name} file ({e.Message}).", e);
                }
                catch (InvalidDataException e)
                {
                    throw new InvalidDataException($"{path}: {e.Message}", e);
                }
                catch (NotSupportedException e)
                {
                    throw new NotSupportedException($"{path}: {e.Message}", e);
                }
            }
        }

        throw new InvalidDataException($"{path}: not an audio format of the registered codecs ({string.Join(", ", Names)}); register one with AudioCodecs.Register.");
    }

    /// <summary>
    /// Decodes a sound file to <paramref name="channels"/> channels: 1 is the mean of the file's channels, a file's own
    /// count is returned as it is, any other count is refused (channels are not invented).
    /// </summary>
    /// <exception cref="ArgumentException">The file has neither one nor <paramref name="channels"/> channels and <paramref name="channels"/> is not 1.</exception>
    public static AudioData Load(string path, int channels)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        var audio = Decode(path);
        if (audio.Channels == channels)
        {
            return audio;
        }

        if (channels == 1)
        {
            var mono = GC.AllocateUninitializedArray<float>(audio.Length);
            audio.ToMono(mono);
            return new AudioData(mono, 1, audio.SampleRate);
        }

        throw new ArgumentException($"{path} has {audio.Channels} channels, not {channels} (1 mixes a file's channels down; others are not invented).", nameof(channels));
    }

    // An app's codec over a built-in: each call falls back on its own, or is compared with the built-in's.
    private sealed class GuardedCodec(Slot slot, IAudioCodec app, IAudioCodec library) : IAudioCodec
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Extensions => app.Extensions;

        public AudioInfo? ReadInfo(ReadOnlySpan<byte> header)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.ReadInfo(header);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        run.Done(Comparisons.Exact(answer, app.ReadInfo(header)));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.ReadInfo(header);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.ReadInfo(header);
            }
        }

        public AudioData Decode(ReadOnlySpan<byte> file)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Decode(file);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        var other = app.Decode(file);
                        run.Done(Comparisons.Exact((answer.Channels, answer.SampleRate, answer.Length), (other.Channels, other.SampleRate, other.Length))
                                 ?? Comparisons.Difference(answer.Samples, other.Samples, 1e-6f));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Decode(file);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Decode(file);
            }
        }
    }
}
