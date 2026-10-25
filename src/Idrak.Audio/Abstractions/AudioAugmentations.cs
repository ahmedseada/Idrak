// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>Where an augmentation works: on the sound before its features, or on the features.</summary>
public enum AudioAugmentationStage
{
    /// <summary>The sound (one channel at the features' rate), before the features are taken: the features are taken again.</summary>
    Sound,

    /// <summary>The features [frames, mels], after the features are taken (or read from the feature cache).</summary>
    Features,
}

/// <summary>The settings of the augmentations (<see cref="AudioAugmentations"/>); each augmentation reads its own.</summary>
public sealed record AudioAugmentationOptions
{
    /// <summary>The chance that each listed augmentation changes a given utterance in a given pass (0 to 1).</summary>
    public double Probability { get; init; } = 1;

    /// <summary>"speed": the factors one is drawn from (sound played faster or slower, its pitch with it: Kaldi's speed perturbation).</summary>
    public IReadOnlyList<double> SpeedFactors { get; init; } = [0.9, 1.0, 1.1];

    /// <summary>"noise": the lowest signal-to-noise ratio drawn, dB.</summary>
    public double MinSnrDb { get; init; } = 10;

    /// <summary>"noise": the highest signal-to-noise ratio drawn, dB.</summary>
    public double MaxSnrDb { get; init; } = 40;

    /// <summary>"noise": sound files a stretch of noise is drawn from (any format a registered codec reads); empty for white noise.</summary>
    public IReadOnlyList<string> NoiseFiles { get; init; } = [];

    /// <summary>"spec-mask": time masks per utterance (SpecAugment).</summary>
    public int TimeMasks { get; init; } = 2;

    /// <summary>"spec-mask": the widest time mask, frames.</summary>
    public int TimeMaskFrames { get; init; } = 40;

    /// <summary>"spec-mask": the widest time mask as a share of the utterance's frames (the smaller of the two applies).</summary>
    public double TimeMaskFraction { get; init; } = 0.05;

    /// <summary>"spec-mask": frequency masks per utterance.</summary>
    public int FrequencyMasks { get; init; } = 2;

    /// <summary>"spec-mask": the widest frequency mask, mel bands.</summary>
    public int FrequencyMaskBands { get; init; } = 15;

    /// <summary>The options as JSON (<see cref="FromJson"/> reads it back).</summary>
    public JsonObject ToJson() => new()
    {
        ["probability"] = Probability,
        ["speed_factors"] = new JsonArray([.. SpeedFactors.Select(f => (JsonNode)f)]),
        ["min_snr_db"] = MinSnrDb,
        ["max_snr_db"] = MaxSnrDb,
        ["noise_files"] = new JsonArray([.. NoiseFiles.Select(f => (JsonNode)f)]),
        ["time_masks"] = TimeMasks,
        ["time_mask_frames"] = TimeMaskFrames,
        ["time_mask_fraction"] = TimeMaskFraction,
        ["frequency_masks"] = FrequencyMasks,
        ["frequency_mask_bands"] = FrequencyMaskBands,
    };

    /// <summary>Options from their JSON (<see cref="ToJson"/>); a key missing keeps the default.</summary>
    public static AudioAugmentationOptions FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var d = new AudioAugmentationOptions();
        return new AudioAugmentationOptions
        {
            Probability = (double?)json["probability"] ?? d.Probability,
            SpeedFactors = json["speed_factors"] is JsonArray speeds ? [.. speeds.Select(f => (double)f!)] : d.SpeedFactors,
            MinSnrDb = (double?)json["min_snr_db"] ?? d.MinSnrDb,
            MaxSnrDb = (double?)json["max_snr_db"] ?? d.MaxSnrDb,
            NoiseFiles = json["noise_files"] is JsonArray files ? [.. files.Select(f => (string)f!)] : d.NoiseFiles,
            TimeMasks = (int?)json["time_masks"] ?? d.TimeMasks,
            TimeMaskFrames = (int?)json["time_mask_frames"] ?? d.TimeMaskFrames,
            TimeMaskFraction = (double?)json["time_mask_fraction"] ?? d.TimeMaskFraction,
            FrequencyMasks = (int?)json["frequency_masks"] ?? d.FrequencyMasks,
            FrequencyMaskBands = (int?)json["frequency_mask_bands"] ?? d.FrequencyMaskBands,
        };
    }

    /// <summary>Refuses values out of range.</summary>
    /// <exception cref="ArgumentException">A value is out of range.</exception>
    public void Check()
    {
        if (Probability is < 0 or > 1 || SpeedFactors.Count == 0 || SpeedFactors.Any(f => !(f > 0)) || MinSnrDb > MaxSnrDb || TimeMasks < 0 || TimeMaskFrames < 0
            || TimeMaskFraction is < 0 or > 1 || FrequencyMasks < 0 || FrequencyMaskBands < 0)
        {
            throw new ArgumentException("Augmentation options: Probability is 0 to 1, SpeedFactors positive (at least one), MinSnrDb at most MaxSnrDb, "
                + "the masks' counts and widths not negative, TimeMaskFraction 0 to 1.");
        }
    }
}

/// <summary>
/// One way of varying training audio so a model sees more than the recordings it has. It draws everything from the
/// <see cref="Random"/> it is given, which the trainer seeds from the run's seed, the pass and the utterance
/// (<see cref="AudioAugmentations.RandomFor"/>): the same utterance in the same pass is varied the same way on every run
/// and on resume. Register one with <see cref="AudioAugmentations.Register"/>.
/// </summary>
public interface IAudioAugmentation
{
    /// <summary>The augmentation's name ("speed").</summary>
    string Name { get; }

    /// <summary>Whether it changes the sound or the features.</summary>
    AudioAugmentationStage Stage { get; }

    /// <summary>The sound varied (<see cref="AudioAugmentationStage.Sound"/>): <paramref name="audio"/> is one channel; the result one channel at the same rate.</summary>
    AudioData Augment(AudioData audio, Random random, AudioAugmentationOptions options) => audio;

    /// <summary>The features [<paramref name="frames"/>, <paramref name="mels"/>] varied in place (<see cref="AudioAugmentationStage.Features"/>).</summary>
    void Augment(Span<float> features, int frames, int mels, Random random, AudioAugmentationOptions options)
    {
    }
}

/// <summary>
/// The training augmentations, by name (ignoring case). Built in (Idrak.Audio): "speed" (the sound resampled by a factor
/// drawn from <see cref="AudioAugmentationOptions.SpeedFactors"/>, so it plays faster or slower), "noise" (white noise, or a
/// stretch of one of <see cref="AudioAugmentationOptions.NoiseFiles"/>, added at a signal-to-noise ratio drawn between
/// the options' limits) and "spec-mask" (SpecAugment's time and frequency masks on the features, Park et al. 2019,
/// "SpecAugment", arXiv:1904.08779, filled with the utterance's mean feature).
/// </summary>
public static class AudioAugmentations
{
    /// <summary>Speed perturbation.</summary>
    public const string Speed = "speed";

    /// <summary>Added noise.</summary>
    public const string Noise = "noise";

    /// <summary>Time and frequency masks.</summary>
    public const string SpecMask = "spec-mask";

    private static readonly SlotTable<string, IAudioAugmentation> Table = BuiltIn();

    private static SlotTable<string, IAudioAugmentation> BuiltIn()
    {
        var table = new SlotTable<string, IAudioAugmentation>(nameof(AudioAugmentations), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "an augmentation's draws are its own, so another cannot stand in for it and give the same varied audio");
        table.RegisterDefault(Speed, new SpeedAugmentation());
        table.RegisterDefault(Noise, new NoiseAugmentation());
        table.RegisterDefault(SpecMask, new SpecMaskAugmentation());
        return table;
    }

    /// <summary>Registers <paramref name="augmentation"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioAugmentation augmentation)
    {
        ArgumentNullException.ThrowIfNull(augmentation);
        ArgumentException.ThrowIfNullOrWhiteSpace(augmentation.Name);
        Table.Register(augmentation.Name, augmentation, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's augmentation of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The augmentation of that name, or null.</summary>
    public static IAudioAugmentation? Find(string name) => Table.Find(name);

    /// <summary>The augmentation of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IAudioAugmentation Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(AudioAugmentations), name, Table.Keys,
            $"Audio augmentation '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with AudioAugmentations.Register.");

    /// <summary>The library's augmentation of that name, or null.</summary>
    public static IAudioAugmentation? Default(string name) => Table.Default(name);

    /// <summary>Who registered the augmentation of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// The random draws of augmentation <paramref name="name"/> for utterance <paramref name="utterance"/> in pass
    /// <paramref name="epoch"/> of a run seeded with <paramref name="seed"/>: the same on every run (the name hashed
    /// with FNV-1a, not the process's string hash).
    /// </summary>
    public static Random RandomFor(int seed, int epoch, int utterance, string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        uint hash = 2166136261;
        foreach (char c in name.ToLowerInvariant())
        {
            hash = (hash ^ c) * 16777619;
        }

        return new Random(unchecked((((int)hash * 31 + seed) * 1_000_003 + epoch) * 10_007 + utterance));
    }
}
