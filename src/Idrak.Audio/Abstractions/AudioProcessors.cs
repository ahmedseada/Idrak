// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// One step done to sound (filter it, take the noise off, set its level): sound in, sound out, as images have
/// <c>IImageTransform</c>. Register one with <see cref="AudioProcessors.Register"/>; a pipeline
/// (<see cref="AudioPipeline"/>) names processors in the order they run, as text such as
/// <c>highpass=80, denoise=1.5 quantile=0.2, normalize=-20</c>.
/// <para>
/// A processor runs in one of two ways, declared by <see cref="Live"/>. A <b>live</b> one works on the sound as it comes:
/// <see cref="Open"/> gives a stage (<see cref="IAudioStage"/>) that takes blocks in order and keeps what it needs
/// between them (a filter's memory, a level's envelope); it implements <see cref="Open"/>, and <see cref="Apply"/> runs
/// a whole recording through a stage. A <b>whole-recording</b> one needs the end before it can start (trimming the
/// silence at both ends, a level read over all the speech, a noise floor read from the pauses); it implements
/// <see cref="Apply"/>, and its stage keeps the blocks and gives the result when flushed, so a live pipeline (one that
/// cleans a microphone) refuses it, naming <see cref="LiveAlternative"/> when it has one.
/// </para>
/// <para>
/// What a processor takes is described by <see cref="Parameters"/>: its value (<c>80</c> in <c>highpass=80</c>) and
/// options (<c>quantile=0.2</c>), each with its kind, range, unit and the value used when left out. The default
/// <see cref="Keys"/> and <see cref="Check"/> follow from them, so a processor that describes its inputs gets its steps
/// checked when a pipeline is parsed. A processor knows nothing of models: one that runs a model (a learned denoiser)
/// is a plug-in.
/// </para>
/// </summary>
public interface IAudioProcessor
{
    /// <summary>The name a pipeline uses (<c>highpass</c>); names compare ignoring case and hold no ',', '=' or spaces.</summary>
    string Name { get; }

    /// <summary>One line on what it does and what its value is (listed with the registered processors).</summary>
    string Summary { get; }

    /// <summary>
    /// Whether it runs on sound as it comes, block by block (true), or needs the whole recording (false: its stage gives
    /// everything when flushed, and a live pipeline refuses it).
    /// </summary>
    bool Live { get; }

    /// <summary>A short name for people ("High-pass filter"); <see cref="Name"/> unless the processor gives one.</summary>
    string Label => Name;

    /// <summary>
    /// Its inputs: the value (key "") and each option, with kind, range, unit and the value used when left out; none
    /// unless the processor describes them (then it takes no value and no option).
    /// </summary>
    IReadOnlyList<AudioProcessorParameter> Parameters => [];

    /// <summary>The option keys it takes besides its value: by default, those of <see cref="Parameters"/>.</summary>
    IReadOnlyCollection<string> Keys => [.. Parameters.Where(p => p.Key.Length > 0).Select(p => p.Key)];

    /// <summary>For a whole-recording processor, the live one to use on live sound instead (named when a live pipeline refuses it); null when there is none.</summary>
    string? LiveAlternative => null;

    /// <summary>
    /// Throws <see cref="ArgumentException"/> when <paramref name="step"/>'s value or options are not ones it takes
    /// (checked when a pipeline is made). By default, against <see cref="Parameters"/> (<see cref="AudioProcessorStep.Validate"/>).
    /// </summary>
    void Check(AudioProcessorStep step) => step.Validate(Parameters);

    /// <summary>
    /// A stage running <paramref name="step"/> over blocks of <paramref name="channels"/> channels at
    /// <paramref name="sampleRate"/>. A live processor implements it; a whole-recording one's default keeps the blocks
    /// and runs <see cref="Apply"/> over them when flushed.
    /// </summary>
    /// <exception cref="ArgumentException">The step cannot run at that rate or channel count (a cutoff over half the rate).</exception>
    IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) => Live
        ? throw new NotSupportedException($"The audio processor {Name} is live but does not implement Open.")
        : new BufferedStage(this, step, sampleRate, channels);

    /// <summary>
    /// The whole recording after <paramref name="step"/>; <paramref name="audio"/> is not changed (the result is new
    /// samples, or <paramref name="audio"/> itself when nothing changes). A whole-recording processor implements it; a
    /// live one's default runs a copy through one <see cref="Open"/>ed stage.
    /// </summary>
    AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (!Live)
        {
            throw new NotSupportedException($"The audio processor {Name} needs the whole recording but does not implement Apply.");
        }

        using var stage = Open(step, audio.SampleRate, audio.Channels);
        return AudioStages.Run(stage, AudioStages.Copy(audio));
    }
}

/// <summary>
/// A processor opened for a rate and a channel count (<see cref="IAudioProcessor.Open"/>): blocks go in, in order, and
/// sound comes out, delayed by <see cref="Latency"/>; <see cref="Flush"/> gives what is left at the end. Not thread-safe:
/// one reader feeds it.
/// </summary>
public interface IAudioStage : IDisposable
{
    /// <summary>The rate of the sound it gives (a resampler changes it; most keep the rate they were opened for).</summary>
    int SampleRate { get; }

    /// <summary>The channels of the sound it gives (a mix to mono changes them).</summary>
    int Channels { get; }

    /// <summary>How many samples, at <see cref="SampleRate"/>, its output lags its input (a look-ahead); 0 for most, and for a stage that gives everything when flushed.</summary>
    int Latency => 0;

    /// <summary>
    /// The next block (at the rate and channels the stage was opened for); returns the sound ready so far, which may be
    /// shorter, longer or empty. The block is the stage's after the call: it may write the result into the block's
    /// samples and return it, or keep the block; pass a copy to keep the original.
    /// </summary>
    AudioData Process(AudioData block);

    /// <summary>What is left at the end (a look-ahead's tail, a whole-recording stage's result); empty when nothing is.</summary>
    AudioData Flush();

    void IDisposable.Dispose()
    {
    }
}

/// <summary>What kind of input an audio processor's value or option is.</summary>
public enum AudioProcessorInput
{
    /// <summary>A number (a frequency, a level in dB, seconds).</summary>
    Number,

    /// <summary>A whole number (an order, a count).</summary>
    Integer,

    /// <summary>One of <see cref="AudioProcessorParameter.Choices"/>.</summary>
    Choice,

    /// <summary>true or false.</summary>
    Switch,
}

/// <summary>
/// One input of an audio processor (<see cref="IAudioProcessor.Parameters"/>): its value (<see cref="Key"/> "") or an
/// option, how it is entered, its range and unit, and the value used when it is left out.
/// </summary>
/// <param name="Key">"" for the step's value (<c>highpass=80</c>), else the option's key (<c>quantile</c>).</param>
/// <param name="Label">A short name for people.</param>
/// <param name="Input">How it is entered.</param>
public sealed record AudioProcessorParameter(string Key, string Label, AudioProcessorInput Input)
{
    /// <summary>The smallest value taken, for numbers (taken itself unless <see cref="MinExclusive"/>).</summary>
    public double? Min { get; init; }

    /// <summary>Whether <see cref="Min"/> itself is refused (a length that must be above 0).</summary>
    public bool MinExclusive { get; init; }

    /// <summary>The largest value taken, for numbers.</summary>
    public double? Max { get; init; }

    /// <summary>A sensible increment for a number control.</summary>
    public double? Step { get; init; }

    /// <summary>The values taken, for <see cref="AudioProcessorInput.Choice"/>.</summary>
    public IReadOnlyList<string> Choices { get; init; } = [];

    /// <summary>The value used when it is left out, as a pipeline writes it; null when it must be given or has none.</summary>
    public string? Default { get; init; }

    /// <summary>Whether it may be left out (then <see cref="Default"/> is used, or what the summary says).</summary>
    public bool Optional { get; init; }

    /// <summary>The unit, for people ("Hz", "dB", "dBFS", "s").</summary>
    public string? Unit { get; init; }

    /// <summary>One line on what it changes.</summary>
    public string? Help { get; init; }

    /// <summary>The range as people read it ("0 to 1", "above 0"); "" when it has none.</summary>
    public string Range => (Min, Max) switch
    {
        ({ } min, { } max) => $"{(MinExclusive ? "above " : "")}{Format(min)} to {Format(max)}",
        ({ } min, null) => $"{(MinExclusive ? "above" : "at least")} {Format(min)}",
        (null, { } max) => $"at most {Format(max)}",
        _ => "",
    };

    private static string Format(double value) => value.ToString(CultureInfo.InvariantCulture);
}

/// <summary>
/// One step of an audio pipeline: a processor's name, its value (<c>80</c> in <c>highpass=80</c>; null when none is
/// given) and its options (<c>quantile=0.2</c>). Immutable.
/// </summary>
public sealed class AudioProcessorStep
{
    private readonly SortedDictionary<string, string> _options;

    /// <summary>The step <paramref name="name"/> with <paramref name="value"/> and <paramref name="options"/> (keys ignore case).</summary>
    public AudioProcessorStep(string name, string? value = null, IEnumerable<KeyValuePair<string, string>>? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name.Trim().ToLowerInvariant();
        Value = value?.Trim() is { Length: > 0 } v ? v : null;
        _options = new SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, text) in options ?? [])
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(key);
            _options[key.Trim().ToLowerInvariant()] = text?.Trim() is { Length: > 0 } t ? t
                : throw new ArgumentException($"The option '{key}' of {Name} has no value.", nameof(options));
        }
    }

    /// <summary>The processor's name (lower case).</summary>
    public string Name { get; }

    /// <summary>The value, as written; null when the step has none.</summary>
    public string? Value { get; }

    /// <summary>The options, by key (lower case).</summary>
    public IReadOnlyDictionary<string, string> Options => _options;

    /// <summary>This step with the option <paramref name="key"/> set to <paramref name="value"/>.</summary>
    public AudioProcessorStep With(string key, string value) => new(Name, Value, _options.Append(KeyValuePair.Create(key, value)));

    /// <summary>The value as a finite number; <paramref name="otherwise"/> when there is none (null: a value is required).</summary>
    /// <exception cref="ArgumentException">No value where one is required, or a value that is not a finite number.</exception>
    public double Number(double? otherwise = null) => Value is null
        ? otherwise ?? throw new ArgumentException($"{Name} needs a value ({Name}=NUMBER).")
        : ParseNumber(Value, Name);

    /// <summary>The value as a whole number; <paramref name="otherwise"/> when there is none (null: a value is required).</summary>
    /// <exception cref="ArgumentException">No value where one is required, or a value that is not a whole number.</exception>
    public int Integer(int? otherwise = null) => Value is null
        ? otherwise ?? throw new ArgumentException($"{Name} needs a value ({Name}=N).")
        : ParseInteger(Value, Name);

    /// <summary>The option <paramref name="key"/> as a finite number, or <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">The option is not a finite number.</exception>
    public double Number(string key, double otherwise) => _options.TryGetValue(key, out var text) ? ParseNumber(text, $"{Name} {key}") : otherwise;

    /// <summary>The option <paramref name="key"/> as a whole number, or <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">The option is not a whole number.</exception>
    public int Integer(string key, int otherwise) => _options.TryGetValue(key, out var text) ? ParseInteger(text, $"{Name} {key}") : otherwise;

    /// <summary>The option <paramref name="key"/> as true or false, or <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">The option is not true or false.</exception>
    public bool Flag(string key, bool otherwise) => _options.TryGetValue(key, out var text) ? ParseFlag(text, $"{Name} {key}") : otherwise;

    /// <summary>The option <paramref name="key"/>, or <paramref name="otherwise"/> when not given.</summary>
    public string? Option(string key, string? otherwise = null) => _options.TryGetValue(key, out var v) ? v : otherwise;

    /// <summary>
    /// Throws unless the step fits <paramref name="parameters"/>: no option they do not name, a value only when they
    /// describe one (and one when it is not optional), each given input of its kind, in its range or among its choices.
    /// </summary>
    /// <exception cref="ArgumentException">The step does not fit, naming what it takes.</exception>
    public void Validate(IReadOnlyList<AudioProcessorParameter> parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        AudioProcessorParameter? value = null;
        var keys = new List<string>();
        foreach (var parameter in parameters)
        {
            if (parameter.Key.Length == 0)
            {
                value = parameter;
            }
            else
            {
                keys.Add(parameter.Key);
            }
        }

        ThrowIfUnknownOptions(keys);
        if (value is null)
        {
            if (Value is not null)
            {
                throw new ArgumentException($"{Name} takes no value (given '{Value}').");
            }
        }
        else if (Value is null)
        {
            if (!value.Optional)
            {
                throw new ArgumentException($"{Name} needs a value: {Describe(value, Name)}.");
            }
        }
        else
        {
            CheckInput(value, Value, Name);
        }

        foreach (var parameter in parameters)
        {
            if (parameter.Key.Length > 0 && _options.TryGetValue(parameter.Key, out var text))
            {
                CheckInput(parameter, text, $"{Name} {parameter.Key}");
            }
        }
    }

    /// <summary>Throws unless every option is one of <paramref name="keys"/>, naming them.</summary>
    /// <exception cref="ArgumentException">An option is not in <paramref name="keys"/>.</exception>
    public void ThrowIfUnknownOptions(IReadOnlyCollection<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        foreach (string key in _options.Keys)
        {
            if (!keys.Contains(key, StringComparer.OrdinalIgnoreCase))
            {
                throw new ArgumentException(keys.Count == 0
                    ? $"The audio processor {Name} takes no options (given '{key}')."
                    : $"The audio processor {Name} does not take '{key}'; it takes: {string.Join(", ", keys)}.");
            }
        }
    }

    /// <summary>As a pipeline writes it: <c>name</c> or <c>name=value</c>, then <c> key=value</c> per option.</summary>
    public override string ToString()
    {
        var text = new StringBuilder(Name);
        if (Value is not null)
        {
            text.Append('=').Append(Value);
        }

        foreach (var (key, value) in _options)
        {
            text.Append(' ').Append(key).Append('=').Append(value);
        }

        return text.ToString();
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is AudioProcessorStep other && other.ToString() == ToString();

    /// <inheritdoc />
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(ToString());

    private static void CheckInput(AudioProcessorParameter parameter, string text, string what)
    {
        switch (parameter.Input)
        {
            case AudioProcessorInput.Number:
            case AudioProcessorInput.Integer:
            {
                double number = parameter.Input == AudioProcessorInput.Number ? ParseNumber(text, what) : ParseInteger(text, what);
                bool low = parameter.Min is { } min && (parameter.MinExclusive ? number <= min : number < min);
                bool high = parameter.Max is { } max && number > max;
                if (low || high)
                {
                    throw new ArgumentException($"{what} is {parameter.Range}{Unit(parameter)}, not {text}.");
                }

                break;
            }

            case AudioProcessorInput.Choice:
                if (!parameter.Choices.Contains(text, StringComparer.OrdinalIgnoreCase))
                {
                    throw new ArgumentException($"{what} is one of {string.Join(", ", parameter.Choices)}, not '{text}'.");
                }

                break;
            case AudioProcessorInput.Switch:
                _ = ParseFlag(text, what);
                break;
        }
    }

    private static string Describe(AudioProcessorParameter parameter, string name) => parameter.Input switch
    {
        AudioProcessorInput.Choice => $"{name}={string.Join("|", parameter.Choices)}",
        AudioProcessorInput.Switch => $"{name}=true|false",
        _ => $"{name}={parameter.Label.ToUpperInvariant().Replace(' ', '_')}" + (parameter.Range.Length > 0 ? $" ({parameter.Range}{Unit(parameter)})" : ""),
    };

    private static string Unit(AudioProcessorParameter parameter) => parameter.Unit is { Length: > 0 } unit ? " " + unit : "";

    private static double ParseNumber(string text, string what) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && double.IsFinite(v) ? v
        : throw new ArgumentException($"{what} takes a number, not '{text}'.");

    private static int ParseInteger(string text, string what) =>
        int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v
        : throw new ArgumentException($"{what} takes a whole number, not '{text}'.");

    private static bool ParseFlag(string text, string what) =>
        bool.TryParse(text, out bool v) ? v : throw new ArgumentException($"{what} is true or false, not '{text}'.");
}

/// <summary>
/// The audio processors a pipeline (<see cref="AudioPipeline"/>) can name, by name (ignoring case). The library's:
/// <c>highpass</c> (live), <c>denoise</c>, <c>trim</c> and <c>normalize</c> (whole recording), the stages
/// <see cref="AudioCleaners"/> runs. Register another with <see cref="Register"/>; one of a library name shadows the
/// library's, which <see cref="Unregister"/> brings back.
/// </summary>
public static class AudioProcessors
{
    /// <summary>A second-order Butterworth high-pass: the DC offset and rumble go (live).</summary>
    public const string HighPass = "highpass";

    /// <summary>Spectral gating against the noise floor read from the pauses (whole recording).</summary>
    public const string Denoise = "denoise";

    /// <summary>The silence at both ends cut (whole recording).</summary>
    public const string Trim = "trim";

    /// <summary>The speech level set, the peak kept under a ceiling (whole recording).</summary>
    public const string Normalize = "normalize";

    private static readonly SlotTable<string, IAudioProcessor> Table = BuiltIn();

    private static SlotTable<string, IAudioProcessor> BuiltIn()
    {
        var table = new SlotTable<string, IAudioProcessor>(nameof(AudioProcessors), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a processor's stage keeps its state between blocks, so another cannot stand in for one block unnoticed");
        table.RegisterDefault(HighPass, new HighPassProcessor());
        table.RegisterDefault(Denoise, new DenoiseProcessor());
        table.RegisterDefault(Trim, new TrimProcessor());
        table.RegisterDefault(Normalize, new NormalizeProcessor());
        return table;
    }

    /// <summary>Registers <paramref name="processor"/> under its name (over the library's of that name, which stays behind it).</summary>
    /// <exception cref="ArgumentException">The name is empty or holds a ',', '=' or a space.</exception>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioProcessor processor)
    {
        ArgumentNullException.ThrowIfNull(processor);
        ArgumentException.ThrowIfNullOrWhiteSpace(processor.Name);
        if (processor.Name.AsSpan().IndexOfAny(", =\t") >= 0)
        {
            throw new ArgumentException($"An audio processor's name has no ',', '=' or space ('{processor.Name}').", nameof(processor));
        }

        Table.Register(processor.Name, processor, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's processor <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The processor registered as <paramref name="name"/>, or null.</summary>
    public static IAudioProcessor? Find(string name) => Table.Find(name);

    /// <summary>The processor registered as <paramref name="name"/>.</summary>
    /// <exception cref="ArgumentException">None is: the message names the registered ones.</exception>
    public static IAudioProcessor Get(string name) => Table.Find(name)
        ?? throw new ArgumentException($"Unknown audio processor '{name}' (registered: {string.Join(", ", Names)}); add one with AudioProcessors.Register.");

    /// <summary>The library's processor <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IAudioProcessor? Default(string name) => Table.Default(name);

    /// <summary>Who registered the processor <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered processor as "name: summary" (with "(whole recording)" for those that are not live), one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(p => $"{p.Name}: {p.Summary}{(p.Live ? "" : " (whole recording)")}"));
}
