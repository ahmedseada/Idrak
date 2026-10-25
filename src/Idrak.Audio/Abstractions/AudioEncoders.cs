// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Audio.Abstractions;

/// <summary>How a sound file is written (<see cref="IAudioEncoder.Encode"/>).</summary>
public sealed record AudioEncodeOptions
{
    /// <summary>Bits per sample: 8, 16, 24 or 32 for integer samples, 32 for float ones.</summary>
    public int BitsPerSample { get; init; } = 16;

    /// <summary>Whether to store floating-point samples (32-bit) instead of integers.</summary>
    public bool Float { get; init; }

    /// <summary>
    /// The format's own options by key, each value as text (<see cref="IAudioEncoder.Parameters"/> declares them: MP3's
    /// bit rate, say), over the encoder's own settings; null or empty: those settings alone. A key the encoder does not
    /// declare is refused.
    /// </summary>
    public IReadOnlyDictionary<string, string>? Parameters { get; init; }
}

/// <summary>
/// Writes one sound format; register it with <see cref="AudioEncoders.Register"/>. The reading side of a format is its
/// <see cref="IAudioCodec"/> in <see cref="AudioCodecs"/>.
/// </summary>
public interface IAudioEncoder
{
    /// <summary>The format's name, e.g. "wav" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extension it writes, with the dot, e.g. ".wav"; <see cref="AudioEncoders.Save"/> picks an encoder by it.</summary>
    string Extension { get; }

    /// <summary>
    /// The options the format takes beyond the sample format, declared with their types, ranges and defaults: the keys of
    /// <see cref="AudioEncodeOptions.Parameters"/>, which the encoder parses with it. None by default
    /// (<see cref="AudioEncoders.NoParameters"/>).
    /// </summary>
    OptionSchema Parameters => AudioEncoders.NoParameters;

    /// <summary>Writes <paramref name="audio"/> to <paramref name="output"/> (16-bit integers unless <paramref name="options"/> say otherwise).</summary>
    void Encode(AudioData audio, Stream output, AudioEncodeOptions? options = null);
}

/// <summary>
/// The sound formats files are written in, by name (ignoring case). Built in (Idrak.Audio): "wav" (PCM 8, 16, 24 and
/// 32 bits and float 32; WAVE_FORMAT_EXTENSIBLE past two channels or 16 bits). Register another format, or replace
/// these, with <see cref="Register"/>; an app's encoder over a library one falls back on it or is compared with it by
/// its policy.
/// </summary>
public static class AudioEncoders
{
    private static readonly SlotTable<string, IAudioEncoder> Registry = BuiltIn();

    /// <summary>The options of an encoder that takes none beyond the sample format (<see cref="IAudioEncoder.Parameters"/>'s default): every key refused.</summary>
    public static OptionSchema NoParameters { get; } = new("audio encoder");

    private static SlotTable<string, IAudioEncoder> BuiltIn()
    {
        var table = new SlotTable<string, IAudioEncoder>(nameof(AudioEncoders), (slot, app, library) => new GuardedEncoder(slot, app, library), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault("wav", new WavEncoder());
        table.RegisterDefault("mp3", new Mp3Encoder());
        return table;
    }

    /// <summary>Registers <paramref name="encoder"/> under its name; under a library name it takes that encoder's place, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentException.ThrowIfNullOrWhiteSpace(encoder.Name);
        Registry.Register(encoder.Name, encoder, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's encoder <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered encoder names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The extensions the registered encoders write (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions => [.. Registry.Values.Select(e => e.Extension.ToLowerInvariant()).Distinct()];

    /// <summary>The encoder registered as <paramref name="name"/> (any case).</summary>
    /// <exception cref="NotSupportedException">None is; the message names those that are.</exception>
    public static IAudioEncoder Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(AudioEncoders), name, Names,
            $"Audio encoder '{name}' is not registered (registered: {string.Join(", ", Names)}); add it with AudioEncoders.Register.");

    /// <summary>The encoder that writes files named like <paramref name="path"/> (by its extension).</summary>
    /// <exception cref="NotSupportedException">No registered encoder lists the extension; the message names the extensions there are.</exception>
    public static IAudioEncoder For(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string extension = Path.GetExtension(path);
        return Registry.Values.FirstOrDefault(e => extension.Length > 0 && string.Equals(e.Extension, extension, StringComparison.OrdinalIgnoreCase))
            ?? throw new NotRegisteredException(nameof(AudioEncoders), extension, Extensions,
                $"An audio encoder of '{extension}' files is not registered (registered: {string.Join(", ", Extensions)}); add one with AudioEncoders.Register.");
    }

    /// <summary>The library's encoder <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IAudioEncoder? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the encoder <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's encoder <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's bytes are written, the two are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Writes <paramref name="audio"/> to <paramref name="path"/> with the encoder of its extension (the folder is created).</summary>
    /// <exception cref="NotSupportedException">No registered encoder writes the extension.</exception>
    public static void Save(string path, AudioData audio, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var encoder = For(path);
        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = File.Create(path);
        encoder.Encode(audio, stream, options);
    }

    /// <summary>The bytes <paramref name="encoder"/> writes for <paramref name="audio"/>.</summary>
    public static byte[] Encode(this IAudioEncoder encoder, AudioData audio, AudioEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        using var stream = new MemoryStream();
        encoder.Encode(audio, stream, options);
        return stream.ToArray();
    }

    // An app's encoder over a library one: each call falls back on its own, or is compared with the library's.
    private sealed class GuardedEncoder(Slot slot, IAudioEncoder app, IAudioEncoder library) : IAudioEncoder
    {
        public string Name => app.Name;

        public string Extension => app.Extension;

        public OptionSchema Parameters => app.Parameters;

        public void Encode(AudioData audio, Stream output, AudioEncodeOptions? options = null)
        {
            var bytes = slot.Call(() => app.Encode(audio, options), () => library.Encode(audio, options),
                (a, b) => a.AsSpan().SequenceEqual(b) ? null : $"{a.Length} and {b.Length} bytes, {FirstDifference(a, b)}");
            output.Write(bytes);
        }

        private static string FirstDifference(byte[] a, byte[] b)
        {
            int i = a.AsSpan().CommonPrefixLength(b);
            return i < Math.Min(a.Length, b.Length) ? $"first difference at byte {i}" : "one is a prefix of the other";
        }
    }
}
