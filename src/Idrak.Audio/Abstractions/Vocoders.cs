// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// Turns a spectrogram a model produced into sound. The frames are rows (<c>[frames, mels]</c> log-mel as
/// <see cref="AudioFeatureExtractors"/> "log-mel" lays them out, with the <see cref="AudioFeatureOptions"/> the vocoder
/// was made with saying the rate, window, hop, mel count and normalisation).
/// </summary>
public interface IVocoder
{
    /// <summary>The vocoder's name, e.g. "griffin-lim".</summary>
    string Name { get; }

    /// <summary>The rate of the sound it makes.</summary>
    int SampleRate { get; }

    /// <summary>The mel bands a frame holds.</summary>
    int Mels { get; }

    /// <summary>The sound of <paramref name="logMel"/> (<c>[frames, mels]</c>): one channel at <see cref="SampleRate"/>, frames × hop samples.</summary>
    AudioData Synthesize(Tensor logMel, Device device);
}

/// <summary>
/// A kind of vocoder, registered in <see cref="Vocoders"/> under its <see cref="IKind.Name"/>: whether it takes the mel
/// layout of some feature options (<see cref="Accepts"/>: one made for any layout takes every one; one whose weights fix
/// its layout takes only that), and a vocoder made for them (<see cref="IKind{TRequest, T}.Create"/>).
/// </summary>
public interface IVocoderKind : IKind<AudioFeatureOptions, IVocoder>
{
    /// <summary>Whether it makes sound from frames of the mel layout of <paramref name="options"/> (the rate, the mel count).</summary>
    bool Accepts(AudioFeatureOptions options);
}

/// <summary>
/// The vocoders, by name (ignoring case). Built in (Idrak.Audio): "griffin-lim" (<see cref="GriffinLimVocoder"/>: no
/// weights, the phases recovered by iteration, made for any mel layout). Register another kind (a neural one a plug-in
/// loads), or one over a library name (which stays behind it, see <see cref="SetPolicy"/>), with <see cref="Register"/>.
/// </summary>
public static class Vocoders
{
    private static readonly KindTable<IVocoderKind> Registry = BuiltIn();

    private static KindTable<IVocoderKind> BuiltIn()
    {
        var table = new KindTable<IVocoderKind>(nameof(Vocoders), "Vocoder", "add it with Vocoders.Register",
            (slot, app, library) => new Kind(app.Name, app.Summary, app.Accepts, options => new GuardedVocoder(slot, app.Create(options), library.Create(options))),
            StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(new Kind(GriffinLimVocoder.DefaultName, "no weights: the phases recovered by iteration, for any mel layout", static _ => true,
            options => new GriffinLimVocoder(options)));
        return table;
    }

    /// <summary>The library's default vocoder, "griffin-lim".</summary>
    public static string DefaultName => GriffinLimVocoder.DefaultName;

    /// <summary>Registers <paramref name="kind"/> under its name; under a library name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IVocoderKind kind) => Registry.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's vocoder <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The vocoder kind <paramref name="name"/>, or null.</summary>
    public static IVocoderKind? Find(ReadOnlySpan<char> name) => Registry.Find(name);

    /// <summary>
    /// The vocoder <paramref name="name"/> (null: the library's default) made for the mel layout of <paramref name="options"/>;
    /// given options are checked against the kind first (<see cref="IVocoderKind.Accepts"/>).
    /// </summary>
    /// <exception cref="NotSupportedException">None is registered (the message names those that are), or it does not take the mel layout of the options given.</exception>
    public static IVocoder Create(string? name, AudioFeatureOptions? options = null)
    {
        name ??= DefaultName;
        var kind = Registry.Find(name)
            ?? throw new NotSupportedException($"No vocoder '{name}' is registered ({string.Join(", ", Names)}); add it with Vocoders.Register.");
        if (options is not null && !kind.Accepts(options))
        {
            throw new NotSupportedException($"The vocoder '{name}' does not take frames of {options.Mels} mel bands at {options.SampleRate} Hz.");
        }

        return kind.Create(options ?? new AudioFeatureOptions());
    }

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IVocoderKind? Default(string name) => Registry.Default(name);

    /// <summary>Who registered <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's vocoder <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's sound is used, the two compared by length and samples within 1e-3).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    // A kind as functions of the options.
    private sealed class Kind(string name, string summary, Func<AudioFeatureOptions, bool> accepts, Func<AudioFeatureOptions, IVocoder> create) : IVocoderKind
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public bool Accepts(AudioFeatureOptions options) => accepts(options);

        public IVocoder Create(AudioFeatureOptions request) => create(request);
    }

    // An app's vocoder over a library one: each call falls back on its own, or is compared with the library's.
    private sealed class GuardedVocoder(Slot slot, IVocoder app, IVocoder library) : IVocoder
    {
        public string Name => app.Name;

        public int SampleRate => app.SampleRate;

        public int Mels => app.Mels;

        public AudioData Synthesize(Tensor logMel, Device device) =>
            slot.Call(() => app.Synthesize(logMel, device), () => library.Synthesize(logMel, device),
                (a, b) => Comparisons.Exact(a.Length, b.Length) ?? Comparisons.Difference(a.Samples, b.Samples, 1e-3f));
    }
}
