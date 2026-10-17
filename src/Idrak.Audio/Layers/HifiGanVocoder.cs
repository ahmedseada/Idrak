// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Audio.Layers;

/// <summary>
/// A HiFi-GAN vocoder's shape (transformers' <c>HifiGanConfig</c>, the keys of SpeechT5's and FastSpeech 2's vocoders):
/// the mel bands it reads, its rate, the generator's channels, upsampling and residual blocks, its slopes, and whether the
/// log-mel is standardized first (<c>normalize_before</c>, with the checkpoint's <c>mean</c> and <c>scale</c>). Every value a setting.
/// </summary>
public sealed record HifiGanVocoderConfig
{
    /// <summary>The mel bands of the input (<c>model_in_dim</c>).</summary>
    public int Mels { get; init; } = 80;

    /// <summary>The output rate (<c>sampling_rate</c>).</summary>
    public int SamplingRate { get; init; } = 16000;

    /// <summary>The generator's first channel count, halved a stage (<c>upsample_initial_channel</c>).</summary>
    public int UpsampleInitialChannel { get; init; } = 512;

    /// <summary>The upsampling factors; their product is the samples a frame (<c>upsample_rates</c>).</summary>
    public int[] UpsampleRates { get; init; } = [4, 4, 4, 4];

    /// <summary>The transposed convolutions' kernels (<c>upsample_kernel_sizes</c>).</summary>
    public int[] UpsampleKernelSizes { get; init; } = [8, 8, 8, 8];

    /// <summary>The residual blocks' kernels (<c>resblock_kernel_sizes</c>).</summary>
    public int[] ResblockKernelSizes { get; init; } = [3, 7, 11];

    /// <summary>The dilations of each residual block's convolutions (<c>resblock_dilation_sizes</c>).</summary>
    public int[][] ResblockDilationSizes { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>The leaky ReLU slope inside the generator (<c>leaky_relu_slope</c>).</summary>
    public float LeakyReluSlope { get; init; } = 0.1f;

    /// <summary>The slope of the last leaky ReLU before the output convolution (PyTorch's default; <c>final_leaky_relu_slope</c> when present).</summary>
    public float FinalLeakyReluSlope { get; init; } = 0.01f;

    /// <summary>Whether the log-mel is standardized with the checkpoint's <c>mean</c> and <c>scale</c> before the generator (<c>normalize_before</c>).</summary>
    public bool NormalizeBefore { get; init; } = true;

    /// <summary>The samples one frame makes: the product of <see cref="UpsampleRates"/>.</summary>
    public int Hop => UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <summary>Reads a config.json (transformers' keys; absent keys keep the defaults).</summary>
    /// <exception cref="InvalidDataException">A list key is not a list, or the lists disagree in length.</exception>
    public static HifiGanVocoderConfig FromJson(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var d = new HifiGanVocoderConfig();
        var result = d with
        {
            Mels = (int?)config["model_in_dim"] ?? d.Mels,
            SamplingRate = (int?)config["sampling_rate"] ?? d.SamplingRate,
            UpsampleInitialChannel = (int?)config["upsample_initial_channel"] ?? d.UpsampleInitialChannel,
            UpsampleRates = Ints(config, "upsample_rates", d.UpsampleRates),
            UpsampleKernelSizes = Ints(config, "upsample_kernel_sizes", d.UpsampleKernelSizes),
            ResblockKernelSizes = Ints(config, "resblock_kernel_sizes", d.ResblockKernelSizes),
            ResblockDilationSizes = config["resblock_dilation_sizes"] is JsonArray lists
                ? [.. lists.Select((l, i) => l is JsonArray a ? a.Select(v => (int)v!).ToArray() : throw new InvalidDataException($"resblock_dilation_sizes[{i}] is not a list."))]
                : d.ResblockDilationSizes,
            LeakyReluSlope = (float?)config["leaky_relu_slope"] ?? d.LeakyReluSlope,
            FinalLeakyReluSlope = (float?)config["final_leaky_relu_slope"] ?? d.FinalLeakyReluSlope,
            NormalizeBefore = (bool?)config["normalize_before"] ?? d.NormalizeBefore,
        };
        return result.Checked();
    }

    /// <summary>Checks the sizes fit together.</summary>
    /// <exception cref="InvalidDataException">They do not.</exception>
    public HifiGanVocoderConfig Checked()
    {
        if (Mels < 1 || SamplingRate < 1 || UpsampleInitialChannel < 1 || UpsampleRates.Length == 0 || ResblockKernelSizes.Length == 0)
        {
            throw new InvalidDataException("A HiFi-GAN vocoder's config needs positive sizes, an upsampling stage and a residual block.");
        }

        if (UpsampleRates.Length != UpsampleKernelSizes.Length || ResblockKernelSizes.Length != ResblockDilationSizes.Length)
        {
            throw new InvalidDataException($"upsample_rates ({UpsampleRates.Length}) and upsample_kernel_sizes ({UpsampleKernelSizes.Length}) pair up, as do "
                + $"resblock_kernel_sizes ({ResblockKernelSizes.Length}) and resblock_dilation_sizes ({ResblockDilationSizes.Length}).");
        }

        if (UpsampleInitialChannel >> UpsampleRates.Length == 0)
        {
            throw new InvalidDataException($"upsample_initial_channel {UpsampleInitialChannel} halved over {UpsampleRates.Length} stages leaves no channel.");
        }

        return this;
    }

    /// <summary>The description as a config.json.</summary>
    public JsonObject ToJson() => new()
    {
        ["model_in_dim"] = Mels, ["sampling_rate"] = SamplingRate, ["upsample_initial_channel"] = UpsampleInitialChannel,
        ["upsample_rates"] = new JsonArray([.. UpsampleRates.Select(v => (JsonNode)v)]), ["upsample_kernel_sizes"] = new JsonArray([.. UpsampleKernelSizes.Select(v => (JsonNode)v)]),
        ["resblock_kernel_sizes"] = new JsonArray([.. ResblockKernelSizes.Select(v => (JsonNode)v)]),
        ["resblock_dilation_sizes"] = new JsonArray([.. ResblockDilationSizes.Select(l => (JsonNode)new JsonArray([.. l.Select(v => (JsonNode)v)]))]),
        ["leaky_relu_slope"] = LeakyReluSlope, ["final_leaky_relu_slope"] = FinalLeakyReluSlope, ["normalize_before"] = NormalizeBefore,
    };

    private static int[] Ints(JsonObject config, string key, int[] fallback) => config[key] switch
    {
        null => fallback,
        JsonArray a => [.. a.Select(v => (int)v!)],
        _ => throw new InvalidDataException($"\"{key}\" is not a list."),
    };
}

/// <summary>
/// HiFi-GAN as a vocoder (<see cref="IVocoder"/>, "hifigan"): a log-mel spectrogram [frames, mels], standardized with
/// the checkpoint's mean and scale when it says so, through a <see cref="HifiGanGenerator"/> to samples at the
/// checkpoint's rate, long spectrograms decoded in pieces with context (<see cref="DecoderChunkSeconds"/>) so device
/// memory follows the piece. Read from a checkpoint folder (<see cref="Load"/>: transformers' <c>conv_pre</c>,
/// <c>upsampler.N</c>, <c>resblocks.N</c>, <c>conv_post</c>, <c>mean</c> and <c>scale</c> at the root, as SpeechT5's and
/// FastSpeech 2's HiFi-GAN ship) or built from any <see cref="StoredLayers"/>. An app registers one in
/// <see cref="Vocoders"/> with <see cref="RegisterFolder"/>, so a synthesizer picks it by name.
/// </summary>
public sealed class HifiGanVocoder : IVocoder, IDisposable
{
    /// <summary>The registry name an app gives it by default, "hifigan".</summary>
    public const string DefaultName = "hifigan";

    private readonly Tensor? _shift, _scaleInverse;                            // −mean / scale and 1 / scale per band: (x − mean) / scale as one affine
    private bool _disposed;

    /// <summary>Over <paramref name="generator"/> of <paramref name="config"/>, standardizing with <paramref name="mean"/> and <paramref name="scale"/> [mels] when given.</summary>
    public HifiGanVocoder(HifiGanGenerator generator, HifiGanVocoderConfig config, Tensor? mean = null, Tensor? scale = null)
    {
        Generator = generator ?? throw new ArgumentNullException(nameof(generator));
        Config = config ?? throw new ArgumentNullException(nameof(config));
        if ((mean is null) != (scale is null))
        {
            throw new ArgumentException("The mean and the scale come together, or neither.", nameof(scale));
        }

        if (mean is not null && (mean.Size != config.Mels || scale!.Size != config.Mels))
        {
            throw new ArgumentException($"The mean and scale hold {mean.Size} and {scale!.Size} values for {config.Mels} mel bands.", nameof(mean));
        }

        if (mean is not null)
        {
            float[] means = mean.ToArray(), inverse = scale!.ToArray(), shift = new float[inverse.Length];
            for (int i = 0; i < inverse.Length; i++)
            {
                inverse[i] = 1f / inverse[i];
                shift[i] = -means[i] * inverse[i];
            }

            _scaleInverse = TensorScope.Untrack(Tensor.From(inverse, [config.Mels], generator.WeightsDevice ?? Device.Default));
            _shift = TensorScope.Untrack(Tensor.From(shift, [config.Mels], _scaleInverse.Device));
        }
    }

    /// <summary>The generator.</summary>
    public HifiGanGenerator Generator { get; }

    /// <summary>The shape.</summary>
    public HifiGanVocoderConfig Config { get; }

    /// <inheritdoc />
    public string Name => DefaultName;

    /// <inheritdoc />
    public int SampleRate => Config.SamplingRate;

    /// <inheritdoc />
    public int Mels => Config.Mels;

    /// <summary>The device the generator's weights are on.</summary>
    public Device Device => Generator.WeightsDevice ?? Device.Default;

    /// <summary>Whether the log-mel is standardized before the generator (the checkpoint had its mean and scale, and the config asks).</summary>
    public bool Normalizes => _shift is not null;

    /// <summary>
    /// The seconds of sound made in one piece, each with the generator's context frames on both sides (the same
    /// samples as one pass); 0: the whole spectrogram in one pass.
    /// </summary>
    public double DecoderChunkSeconds { get; set; } = 10;

    /// <summary>
    /// Reads a HiFi-GAN checkpoint (<paramref name="path"/>: a folder with config.json and its tensors in any format
    /// <see cref="CheckpointFormats"/> opens, or a file in it) onto <paramref name="device"/> (default <see cref="Device.Default"/>).
    /// </summary>
    /// <exception cref="InvalidDataException">No config.json, or a tensor missing or of another shape than the config says.</exception>
    public static HifiGanVocoder Load(string path, Device? device = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string target = File.Exists(path) ? Path.GetDirectoryName(Path.GetFullPath(path))! : path;
        var format = CheckpointFormats.For(target);
        string folder = format.Prepare(target);
        string configFile = Path.Combine(folder, "config.json");
        if (!File.Exists(configFile))
        {
            throw new InvalidDataException($"{folder} has no config.json (a HiFi-GAN vocoder's shape).");
        }

        var config = HifiGanVocoderConfig.FromJson(JsonNode.Parse(File.ReadAllText(configFile)) as JsonObject ?? throw new InvalidDataException($"{configFile} is not a JSON object."));
        using var tensors = format.Open(folder);
        return FromStored(new StoredLayers(tensors, device ?? Device.Default), "", config);
    }

    /// <summary>
    /// The vocoder from <paramref name="stored"/> under <paramref name="prefix"/> (empty for the root): the generator's
    /// tensors (<see cref="HifiGanGenerator.FromStored"/>, no speaker conditioning) and, when the config standardizes,
    /// <c>mean</c> and <c>scale</c> [mels] (a checkpoint without them standardizes nothing).
    /// </summary>
    public static HifiGanVocoder FromStored(StoredLayers stored, string prefix, HifiGanVocoderConfig config)
    {
        ArgumentNullException.ThrowIfNull(stored);
        ArgumentNullException.ThrowIfNull(config);
        config.Checked();
        var generator = HifiGanGenerator.FromStored(stored, prefix, config.Mels, config.UpsampleInitialChannel, config.UpsampleRates, config.UpsampleKernelSizes,
            config.ResblockKernelSizes, config.ResblockDilationSizes, config.LeakyReluSlope, config.FinalLeakyReluSlope, 0);
        try
        {
            string meanName = StoredLayers.Join(prefix, "mean"), scaleName = StoredLayers.Join(prefix, "scale");
            if (!config.NormalizeBefore || !stored.Contains(meanName) || !stored.Contains(scaleName))
            {
                return new HifiGanVocoder(generator, config);
            }

            using var mean = Tensor.From(stored.Read(meanName), [config.Mels], Device.Cpu);
            using var scale = Tensor.From(stored.Read(scaleName), [config.Mels], Device.Cpu);
            return new HifiGanVocoder(generator, config, mean, scale);
        }
        catch
        {
            generator.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Registers a vocoder read from <paramref name="path"/> under <paramref name="name"/> in <see cref="Vocoders"/>: the
    /// checkpoint is read at the first <see cref="Vocoders.Create"/> and kept for the process (the registry's options are
    /// not read; the checkpoint fixes the mel layout).
    /// </summary>
    public static void RegisterFolder(string path, string name = DefaultName, Device? device = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var loaded = new Lazy<HifiGanVocoder>(() => Load(path, device), LazyThreadSafetyMode.ExecutionAndPublication);
        Vocoders.Register(name, _ => loaded.Value);
    }

    /// <inheritdoc />
    public AudioData Synthesize(Tensor logMel, Device device)
    {
        ArgumentNullException.ThrowIfNull(logMel);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (logMel.Rank != 2 || logMel.Shape[1] != Mels)
        {
            throw new ArgumentException($"The HiFi-GAN vocoder takes [frames, {Mels}], got {Tensor.FormatShape(logMel.Shape)}.", nameof(logMel));
        }

        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var x = logMel.To(Device);
        if (_shift is not null)
        {
            x = x.GroupAffine(_scaleInverse, _shift, Mels, 1);                   // (x − mean) / scale, each band's own, one kernel
        }

        int chunkFrames = DecoderChunkSeconds <= 0 ? 0 : Math.Max(1, (int)(DecoderChunkSeconds * SampleRate / Config.Hop));
        return new AudioData(Generator.Samples(x.Transpose(), null, chunkFrames), 1, SampleRate);
    }

    /// <summary>Disposes the generator and the normalization tensors.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Generator.Dispose();
        _shift?.Dispose();
        _scaleInverse?.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"HifiGanVocoder({Mels} mels at {SampleRate} Hz, hop {Config.Hop}{(Normalizes ? ", standardized" : "")})";
}
