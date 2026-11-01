// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// Sampling values, each one given or not (null): a model's own recommendation (<see cref="Read"/>: its
/// generation_config.json, as Hugging Face models ship it, or a GGUF file's general.sampling keys written there when the
/// file is prepared), or the values an app or a user set explicitly. <see cref="Resolve"/> lays them over
/// <see cref="GenerationOptions"/>: explicit values win, the model's recommendation comes next, the options' own values
/// (the library's defaults, or the app's baseline) last. A model's recommendation matters: Qwen2.5-7B-Instruct
/// recommends temperature 0.7, top_p 0.8, top_k 20 and repetition_penalty 1.05; with the generic defaults (0.8, 0.9, 40,
/// 1.1) a coding agent on it answered in another language and wrote code as prose instead of tool calls.
/// </summary>
public sealed record SamplingSettings
{
    /// <summary>The file a model's recommendation is read from: "generation_config.json".</summary>
    public const string FileName = "generation_config.json";

    /// <summary>Softmax temperature (<see cref="GenerationOptions.Temperature"/>; generation_config.json's "temperature"); 0 means greedy.</summary>
    public float? Temperature { get; init; }

    /// <summary>Top-k (<see cref="GenerationOptions.TopK"/>; "top_k"; 0 = all tokens).</summary>
    public int? TopK { get; init; }

    /// <summary>Nucleus sampling (<see cref="GenerationOptions.TopP"/>; "top_p"; 1 = off).</summary>
    public float? TopP { get; init; }

    /// <summary>Min-p (<see cref="GenerationOptions.MinP"/>; "min_p"; 0 = off).</summary>
    public float? MinP { get; init; }

    /// <summary>
    /// The repetition penalty (<see cref="GenerationOptions.RepeatPenalty"/>; "repetition_penalty"; 1 = off). Transformers
    /// applies it over the whole sequence; Idrak over <see cref="GenerationOptions.RepeatLastN"/> recent tokens.
    /// </summary>
    public float? RepeatPenalty { get; init; }

    /// <summary>
    /// Greedy decoding (always the most likely token): true for generation_config.json's "do_sample": false, false for
    /// "do_sample": true, null when it does not say (transformers then decodes greedily, but most model folders that omit
    /// it hold no sampling values at all, so it is read as not stated).
    /// </summary>
    public bool? Greedy { get; init; }

    /// <summary>Where the values came from (a file's path), or null.</summary>
    public string? Source { get; init; }

    /// <summary>True when no value is given.</summary>
    public bool IsEmpty => Temperature is null && TopK is null && TopP is null && MinP is null && RepeatPenalty is null && Greedy is null;

    // Any value that says how to choose tokens (an explicit one of these turns off a recommended greedy decoding).
    private bool ChoosesTokens => Temperature is not null || TopK is not null || TopP is not null || MinP is not null || Greedy is not null;

    /// <summary>
    /// The sampling the model in <paramref name="folder"/> recommends (its generation_config.json), or null when the folder
    /// has no such file or the file holds no sampling value. A value out of its range (a negative temperature, a top_p
    /// above 1) is left out, as if not given.
    /// </summary>
    /// <exception cref="InvalidDataException">The file is not JSON.</exception>
    public static SamplingSettings? Read(string folder)
    {
        ArgumentNullException.ThrowIfNull(folder);
        string path = Path.Combine(folder, FileName);
        if (!File.Exists(path))
        {
            return null;
        }

        JsonNode? json;
        try
        {
            json = JsonNode.Parse(File.ReadAllBytes(path));
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{path} is not JSON: {e.Message}", e);
        }

        return json is JsonObject config && FromJson(config) is { } read ? read with { Source = path } : null;
    }

    /// <summary>The sampling values of a generation_config.json object, or null when it holds none.</summary>
    public static SamplingSettings? FromJson(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var read = new SamplingSettings
        {
            Temperature = Number(config, "temperature") is { } t && t >= 0 ? t : null,
            TopK = Number(config, "top_k") is { } k && k >= 0 && k <= int.MaxValue && k == MathF.Floor(k) ? (int)k : null,
            TopP = Number(config, "top_p") is { } p && p is > 0 and <= 1 ? p : null,
            MinP = Number(config, "min_p") is { } m && m is >= 0 and <= 1 ? m : null,
            RepeatPenalty = Number(config, "repetition_penalty") is { } r && r > 0 ? r : null,
            Greedy = config["do_sample"] is JsonValue sample && sample.TryGetValue(out bool doSample) ? !doSample : null,
        };
        return read.IsEmpty ? null : read;

        static float? Number(JsonObject config, string key) =>
            config[key] is JsonValue value && (value.TryGetValue(out double d) || value.TryGetValue(out string? s)
                && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out d)) && double.IsFinite(d) ? (float)d : null;
    }

    /// <summary>
    /// <paramref name="baseline"/> (new <see cref="GenerationOptions"/> when null: the library's defaults) with the sampling
    /// values of <paramref name="recommended"/> (the model's) over it and those of <paramref name="given"/> (explicit: an
    /// app's or a user's) over both, value by value, and where each came from. Greedy decoding (<see cref="Greedy"/>, or a
    /// temperature of 0) becomes top-k 1 with temperature, top-p and min-p off; a recommended greedy decoding is not taken
    /// when <paramref name="given"/> says how to choose tokens (any of temperature, top-k, top-p, min-p or greedy).
    /// The other options (length, stops, penalties' window) are the baseline's.
    /// </summary>
    public static ResolvedSampling Resolve(GenerationOptions? baseline, SamplingSettings? given, SamplingSettings? recommended)
    {
        var options = baseline ?? new GenerationOptions();
        var sources = new List<SampledSetting>(5);
        bool givenGreedy = given?.Greedy == true || given?.Temperature == 0;
        bool modelGreedy = given?.ChoosesTokens != true && (recommended?.Greedy == true || recommended?.Temperature == 0);
        if (givenGreedy || modelGreedy)
        {
            options = options with { Temperature = 1f, TopK = 1, TopP = 1f, MinP = 0f };
            sources.Add(new("greedy", "on", givenGreedy ? Given : Model));
        }
        else
        {
            options = options with
            {
                Temperature = Pick("temperature", given?.Temperature, recommended?.Temperature, options.Temperature),
                TopK = Pick("top_k", given?.TopK, recommended?.TopK, options.TopK),
                TopP = Pick("top_p", given?.TopP, recommended?.TopP, options.TopP),
                MinP = Pick("min_p", given?.MinP, recommended?.MinP, options.MinP),
            };
        }

        options = options with { RepeatPenalty = Pick("repeat_penalty", given?.RepeatPenalty, recommended?.RepeatPenalty, options.RepeatPenalty) };
        return new ResolvedSampling(options, sources);

        T Pick<T>(string name, T? explicitValue, T? modelValue, T fallback) where T : struct, IFormattable
        {
            var (value, source) = explicitValue is { } e ? (e, Given) : modelValue is { } m ? (m, Model) : (fallback, Default);
            sources.Add(new(name, value.ToString(null, CultureInfo.InvariantCulture), source));
            return value;
        }
    }

    /// <summary>
    /// <paramref name="options"/> with this recommendation's values over its own (<see cref="Resolve"/> with nothing
    /// explicit): what a server applies before a request's own keys.
    /// </summary>
    public GenerationOptions ApplyTo(GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return Resolve(options, null, this).Options;
    }

    /// <summary>The source of a value given explicitly: "given".</summary>
    public const string Given = "given";

    /// <summary>The source of a value the model recommends: "model".</summary>
    public const string Model = "model";

    /// <summary>The source of a value from the baseline options (the library's defaults, or the app's): "default".</summary>
    public const string Default = "default";
}

/// <summary>One sampling value as <see cref="SamplingSettings.Resolve"/> chose it.</summary>
/// <param name="Name">The value's name: temperature, top_k, top_p, min_p, repeat_penalty, or greedy.</param>
/// <param name="Value">The value, in the invariant culture.</param>
/// <param name="Source">Where it came from: <see cref="SamplingSettings.Given"/>, <see cref="SamplingSettings.Model"/> or <see cref="SamplingSettings.Default"/>.</param>
public sealed record SampledSetting(string Name, string Value, string Source);

/// <summary>The options <see cref="SamplingSettings.Resolve"/> made, and where each sampling value came from.</summary>
/// <param name="Options">The options to generate with.</param>
/// <param name="Settings">The sampling values, each with its source.</param>
public sealed record ResolvedSampling(GenerationOptions Options, IReadOnlyList<SampledSetting> Settings)
{
    /// <summary>"temperature 0.7 (model), top_k 20 (model), top_p 0.8 (model), min_p 0 (default), repeat_penalty 1.05 (model)".</summary>
    public string Describe() => string.Join(", ", Settings.Select(s => $"{s.Name} {s.Value} ({s.Source})"));

    /// <summary>The values that came from the model's recommendation.</summary>
    public IEnumerable<SampledSetting> FromModel => Settings.Where(s => s.Source == SamplingSettings.Model);
}
