// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Formats;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// How text is spoken (plan 21, S5). Every value is a setting; a null one means "the family's configured value" (a
/// VITS checkpoint's <c>noise_scale</c>, say), so a family's defaults are never the only way.
/// </summary>
public sealed record SpeechSynthesisOptions
{
    /// <summary>
    /// A speaker id or name when the family has several; null: the family's first. Through a
    /// <see cref="VoicedSynthesizer"/> it can also name an installed voice (<see cref="IVoiceStore"/>): a checkpoint's
    /// speaker is looked for first, then the store's voices.
    /// </summary>
    public string? Voice { get; init; }

    /// <summary>
    /// How an installed embedding voice is used (<see cref="VoicedSynthesizer"/>): "auto" converts the speech with the
    /// converter family the voice was made with; "off" speaks in the synthesizer's own voice, unconverted (to compare);
    /// a <see cref="VoiceConverters"/> name requires that family (a voice made by another is refused).
    /// </summary>
    public string VoiceConversion { get; init; } = "auto";

    /// <summary>The conversion's own settings (tau, seed, pieces); null: <see cref="VoiceConversionOptions"/>' defaults with <see cref="Seed"/>.</summary>
    public VoiceConversionOptions? Conversion { get; init; }

    /// <summary>The <see cref="VoiceStores"/> name installed voices are looked up in when the <see cref="VoicedSynthesizer"/> was given no store; null: "folder".</summary>
    public string? VoiceStore { get; init; }

    /// <summary>The speaking rate: 2 speaks twice as fast (half the frames per token), 0.5 half as fast; 1 / the length scale.</summary>
    public float Rate { get; init; } = 1f;

    /// <summary>The scale of the noise added to the acoustic latents; 0 gives a deterministic output; null: the family's configured value.</summary>
    public float? NoiseScale { get; init; }

    /// <summary>The scale of the noise the duration predictor draws; 0 gives deterministic durations; null: the family's configured value.</summary>
    public float? DurationNoiseScale { get; init; }

    /// <summary>The seed the noise is drawn from: the same text and seed give the same samples.</summary>
    public int Seed { get; init; }

    /// <summary>A <see cref="Vocoders"/> name for a family that produces a spectrogram; null: the family's own vocoder or the library's default.</summary>
    public string? Vocoder { get; init; }

    /// <summary>Resample the output to this rate when set; null: the family's rate.</summary>
    public int? SampleRate { get; init; }

    /// <summary>0: the text is spoken in one pass; otherwise it is split at sentence ends into pieces of at most this many characters, spoken one by one and joined.</summary>
    public int MaxCharacters { get; init; }

    /// <summary>
    /// The seconds of sound a family's decoder makes in one piece (each piece with its context, so the samples are a
    /// whole pass's): the device's memory follows the piece, not the text's length. 0: one pass; null: the model's own setting.
    /// </summary>
    public double? DecoderSeconds { get; init; }
}

/// <summary>What a synthesis took.</summary>
/// <param name="Text">Tokenising the text.</param>
/// <param name="Model">The model's pass (the text encoder, durations, the flow).</param>
/// <param name="Vocoder">The samples from the model's output (the decoder or a vocoder).</param>
/// <param name="PeakDeviceBytes">The device's peak memory over the call (0 when the device does not report it).</param>
/// <param name="RealTimeFactor">The time taken over the sound's duration: below 1 is faster than real time.</param>
/// <param name="Conversion">Converting the speech to an installed voice (<see cref="VoicedSynthesizer"/>; zero when it was not converted).</param>
public sealed record SynthesisStats(TimeSpan Text, TimeSpan Model, TimeSpan Vocoder, long PeakDeviceBytes, double RealTimeFactor, TimeSpan Conversion = default);

/// <summary>A loaded text-to-speech model.</summary>
public interface ISpeechSynthesizer : IDisposable
{
    /// <summary>The device the model runs on.</summary>
    Device Device { get; }

    /// <summary>The rate of the sound it makes (the family's configured one).</summary>
    int SampleRate { get; }

    /// <summary>The voices it can speak with (empty for a single-speaker model).</summary>
    IReadOnlyList<string> Voices { get; }

    /// <summary>Speaks <paramref name="text"/>: one channel at <see cref="SampleRate"/> (or the options' rate), with what it took.</summary>
    (AudioData Audio, SynthesisStats Stats) Speak(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Speaks <paramref name="text"/> piece by piece (sentences, bounded by <see cref="SpeechSynthesisOptions.MaxCharacters"/>), each piece as soon as it is made.</summary>
    IAsyncEnumerable<AudioData> SpeakStream(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default);
}

/// <summary>What a family reads a synthesis checkpoint from (<see cref="ISpeechSynthesisFamily.Read"/>).</summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by (config.json's "architectures").</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while reading (do not keep it).</param>
/// <param name="Folder">The model folder (where vocab.json and the like are), or null.</param>
/// <param name="Open">Opens the checkpoint again (the caller disposes what it returns).</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record SpeechSynthesisCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Func<ITensorStore> Open, List<string> Notes);

/// <summary>How a synthesis model is loaded.</summary>
public sealed record SpeechSynthesisLoadOptions
{
    /// <summary>The device the model runs on; null: <see cref="Device.Default"/>.</summary>
    public Device? Device { get; init; }

    /// <summary>
    /// Load the model to be trained: its weights as parameters that take gradients, kept in the form training updates (a
    /// weight norm's gains and directions, not folded once as for speaking), read by the family's
    /// <see cref="ITrainableSpeechSynthesisFamily.ReadForTraining"/> into a model the trainers take (an
    /// <see cref="IAdversarialSpeechSynthesizer"/>; see <see cref="SpeechSynthesisModels.LoadForTraining"/>). A family that
    /// does not train refuses it.
    /// </summary>
    public bool Training { get; init; }
}

/// <summary>
/// A text-to-speech family: how its checkpoints are read into an <see cref="ISpeechSynthesizer"/>. Registered in
/// <see cref="SpeechSynthesisFamilies"/> under the architecture name its checkpoints give (config.json's "architectures").
/// </summary>
public interface ISpeechSynthesisFamily
{
    /// <summary>The architecture name it is registered under.</summary>
    string Name { get; }

    /// <summary>Reads the checkpoint (configuration, tokenizer, tensors) into a synthesizer on the options' device, to speak.</summary>
    /// <exception cref="InvalidDataException">A tensor or file is missing or disagrees with the configuration.</exception>
    ISpeechSynthesizer Read(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options);
}

/// <summary>
/// A text-to-speech family that makes models that did not exist from its own description of them
/// (<see cref="SpeechSynthesisModels.Create"/>); a family read from checkpoints only does not implement it.
/// </summary>
public interface ICreatableSynthesisFamily : ISpeechSynthesisFamily
{
    /// <summary>
    /// A model that did not exist, from the family's own description of it (<paramref name="spec"/>): its tensors drawn from
    /// <paramref name="seed"/>, its config.json and other files (<see cref="SpeechSynthesisModels.Create"/> writes them).
    /// </summary>
    CreatedSpeechModel Create(JsonObject spec, ITokenizer tokenizer, int seed);
}

/// <summary>
/// A text-to-speech family whose models train against discriminators: it reads a checkpoint to be trained
/// (<see cref="ReadForTraining"/>, <see cref="SpeechSynthesisLoadOptions.Training"/>) into a model the adversarial
/// trainer takes; a family that does not implement it is not trained.
/// </summary>
public interface ITrainableSpeechSynthesisFamily : ISpeechSynthesisFamily
{
    /// <summary>Reads the checkpoint to be trained: its weights as parameters, in the form training updates.</summary>
    /// <exception cref="InvalidDataException">A tensor or file is missing or disagrees with the configuration.</exception>
    /// <exception cref="NotSupportedException">The checkpoint is one the family does not train (a multi-speaker one, say).</exception>
    IAdversarialSpeechSynthesizer ReadForTraining(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options);
}

/// <summary>
/// The text-to-speech families, by architecture name (config.json's "architectures"). The library registers none: a
/// family (its configuration, tokenizer, layers and decoder) is an application of these contracts, registered by the
/// app or plug-in that brings it (Idrak.Audio.Plugins' VITS family is one). A checkpoint whose name has no family is refused
/// naming this registry: nothing falls back to another family, since families are not interchangeable.
/// </summary>
public static class SpeechSynthesisFamilies
{
    private static readonly SlotTable<string, ISpeechSynthesisFamily> Registry = new(nameof(SpeechSynthesisFamilies), comparer: StringComparer.Ordinal,
        unguarded: "a family's tokenizer, layers and decoder must agree with one another, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="ISpeechSynthesisFamily.Name"/>; under a library name it takes that family's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISpeechSynthesisFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrEmpty(family.Name);
        Registry.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static ISpeechSynthesisFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names <see cref="Register"/>.</exception>
    public static ISpeechSynthesisFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>The library's family <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ISpeechSynthesisFamily? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the family <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>The family that reads a checkpoint of architecture <paramref name="architecture"/>: the one registered under that exact name.</summary>
    /// <exception cref="NotSupportedException">No family is registered under the name; the message names this registry.</exception>
    public static ISpeechSynthesisFamily For(string architecture)
    {
        ArgumentException.ThrowIfNullOrEmpty(architecture);
        return Registry.Find(architecture) ?? throw NotRegistered(architecture);
    }

    internal static NotRegisteredException NotRegistered(string name) =>
        new(nameof(SpeechSynthesisFamilies), name, Registry.Keys,
            $"Speech synthesis family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "register it with SpeechSynthesisFamilies.Register (a plug-in or the app that brings the family; the library registers none), "
            + "or reference Idrak.Audio.Plugins and call AudioPlugins.RegisterAll().");
}

/// <summary>Loads a text-to-speech model from a folder: its config.json names the family, the checkpoint format reads its tensors.</summary>
public static class SpeechSynthesisModels
{
    /// <summary>
    /// Loads the model at <paramref name="path"/> (a folder with config.json and a checkpoint any registered
    /// <see cref="CheckpointFormats"/> format opens) through the family config.json's "architectures" names.
    /// </summary>
    /// <exception cref="FileNotFoundException">There is no config.json.</exception>
    /// <exception cref="InvalidDataException">config.json names no architecture.</exception>
    /// <exception cref="NotSupportedException">The architecture has no registered family.</exception>
    /// <exception cref="NotSupportedException">The architecture has no registered family, or the options ask to train and its family does not train.</exception>
    public static ISpeechSynthesizer Load(string path, SpeechSynthesisLoadOptions? options = null)
    {
        options ??= new SpeechSynthesisLoadOptions();
        return options.Training ? LoadForTraining(path, options) : Read(path, (family, checkpoint) => family.Read(checkpoint, options));
    }

    /// <summary>
    /// Loads the model at <paramref name="path"/> to be trained against discriminators: its family's
    /// <see cref="ITrainableSpeechSynthesisFamily.ReadForTraining"/> (the options' <see cref="SpeechSynthesisLoadOptions.Training"/> set).
    /// </summary>
    /// <exception cref="FileNotFoundException">There is no config.json.</exception>
    /// <exception cref="InvalidDataException">config.json names no architecture.</exception>
    /// <exception cref="NotSupportedException">The architecture has no registered family, or its family does not train (it is no <see cref="ITrainableSpeechSynthesisFamily"/>).</exception>
    public static IAdversarialSpeechSynthesizer LoadForTraining(string path, SpeechSynthesisLoadOptions? options = null)
    {
        var training = (options ?? new SpeechSynthesisLoadOptions()) with { Training = true };
        return Read(path, (family, checkpoint) => family is ITrainableSpeechSynthesisFamily trainable ? trainable.ReadForTraining(checkpoint, training)
            : throw new NotSupportedException($"Speech synthesis family '{family.Name}' does not train against discriminators (it is no ITrainableSpeechSynthesisFamily): "
                + "its models speak but are not trained here."));
    }

    // The checkpoint at `path` read by the family its config.json names.
    private static T Read<T>(string path, Func<ISpeechSynthesisFamily, SpeechSynthesisCheckpoint, T> read)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var format = CheckpointFormats.For(path);
        string folder = format.Prepare(path);
        string configPath = Path.Combine(folder, "config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"{folder} has no config.json.", configPath);
        }

        var config = JsonNode.Parse(File.ReadAllText(configPath), documentOptions: new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip })?.AsObject()
            ?? throw new InvalidDataException($"{configPath} is not a JSON object.");
        string architecture = config["architectures"] is JsonArray { Count: > 0 } a && a[0] is JsonValue v && v.TryGetValue(out string? name) && name.Length > 0 ? name
            : config["model_type"]?.GetValue<string>() ?? throw new InvalidDataException($"{configPath} names no architecture (\"architectures\" or \"model_type\").");
        var family = SpeechSynthesisFamilies.For(architecture);
        using var tensors = format.Open(folder);
        return read(family, new SpeechSynthesisCheckpoint(architecture, config, tensors, folder, () => format.Open(folder), []));
    }

    /// <summary>
    /// A text-to-speech model that did not exist: the family registered under the spec's architecture makes it from
    /// <paramref name="spec"/> (<see cref="ICreatableSynthesisFamily.Create"/>: its tensors drawn from the seed, its config.json
    /// and other files), which is written into <paramref name="folder"/> (config.json with "architectures",
    /// model.safetensors in float32, the tokenizer as tokenizer.json when it is an <c>IPretrainedTokenizer</c> and the family wrote
    /// none, the family's files) and loaded from there, so it is a model like any other: it speaks, trains and saves (plan 21, S6).
    /// </summary>
    /// <exception cref="NotSupportedException">No family is registered under the architecture, or it makes no models from a description.</exception>
    /// <exception cref="InvalidDataException">The spec names no architecture.</exception>
    /// <exception cref="IOException">The folder holds files and <see cref="SpeechCreateOptions.Overwrite"/> is off.</exception>
    public static ISpeechSynthesizer Create(string folder, JsonObject spec, ITokenizer tokenizer, SpeechCreateOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(tokenizer);
        options ??= new SpeechCreateOptions();
        string architecture = SpeechModels.Architecture(spec, options);
        var family = SpeechSynthesisFamilies.Get(architecture) as ICreatableSynthesisFamily
            ?? throw new NotSupportedException($"Speech synthesis family '{architecture}' cannot make a model from a description (it reads checkpoints only).");
        SpeechModels.CheckFolder(folder, options);
        SpeechModels.Write(folder, architecture, family.Create(spec, tokenizer, options.Seed), tokenizer);
        return Load(folder, new SpeechSynthesisLoadOptions { Device = options.Device });
    }
}
