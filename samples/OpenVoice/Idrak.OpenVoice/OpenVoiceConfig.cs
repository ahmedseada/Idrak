// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.OpenVoice;

/// <summary>
/// An OpenVoice converter's config.json (its "data" and "model" sections, as the official checkpoint ships them).
/// Every shape the converter uses comes from here; the posterior encoder's 16 WaveNet layers, the flow's 4 couplings
/// of 4 layers and their kernel 5 are OpenVoice's model code, settings here with those values.
/// </summary>
public sealed record OpenVoiceConfig
{
    /// <summary>The checkpoint's version ("v1" or "v2"; config.json's "_version_").</summary>
    public string Version { get; init; } = "v1";

    /// <summary>The rate the converter reads and writes sound at.</summary>
    public int SamplingRate { get; init; } = 22050;

    /// <summary>The spectrogram's FFT size.</summary>
    public int FilterLength { get; init; } = 1024;

    /// <summary>The spectrogram's hop (the decoder's samples per frame).</summary>
    public int HopLength { get; init; } = 256;

    /// <summary>The spectrogram's window (equal to <see cref="FilterLength"/> in OpenVoice's checkpoints).</summary>
    public int WinLength { get; init; } = 1024;

    /// <summary>The latents' channels.</summary>
    public int InterChannels { get; init; } = 192;

    /// <summary>The WaveNets' width.</summary>
    public int HiddenChannels { get; init; } = 192;

    /// <summary>The speaker embedding's width.</summary>
    public int GinChannels { get; init; } = 256;

    /// <summary>Whether the posterior encoder and the decoder see a zero embedding (v2's "zero_g"): only the flows are conditioned.</summary>
    public bool ZeroG { get; init; }

    /// <summary>The decoder's residual block type ("1"; "2" is not supported).</summary>
    public string Resblock { get; init; } = "1";

    /// <summary>The decoder's residual block kernels.</summary>
    public int[] ResblockKernelSizes { get; init; } = [3, 7, 11];

    /// <summary>The decoder's residual block dilations.</summary>
    public int[][] ResblockDilationSizes { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>The decoder's upsampling factors.</summary>
    public int[] UpsampleRates { get; init; } = [8, 8, 2, 2];

    /// <summary>The decoder's first channel count.</summary>
    public int UpsampleInitialChannel { get; init; } = 512;

    /// <summary>The decoder's transposed convolution kernels.</summary>
    public int[] UpsampleKernelSizes { get; init; } = [16, 16, 4, 4];

    /// <summary>The posterior encoder's WaveNet layers (OpenVoice's model code: 16).</summary>
    public int PosteriorLayers { get; init; } = 16;

    /// <summary>The flow's coupling layers (OpenVoice's model code: 4).</summary>
    public int Flows { get; init; } = 4;

    /// <summary>The WaveNet layers of each coupling (OpenVoice's model code: 4).</summary>
    public int FlowLayers { get; init; } = 4;

    /// <summary>The WaveNets' kernel (OpenVoice's model code: 5).</summary>
    public int WaveNetKernelSize { get; init; } = 5;

    /// <summary>The leaky ReLU slope inside the decoder (OpenVoice's model code: 0.1; config.json's "leaky_relu_slope" when present).</summary>
    public float LeakyReluSlope { get; init; } = 0.1f;

    /// <summary>The slope of the decoder's last leaky ReLU (PyTorch's default, 0.01; config.json's "final_leaky_relu_slope" when present).</summary>
    public float FinalLeakyReluSlope { get; init; } = 0.01f;

    /// <summary>The reference encoder's convolution filters (OpenVoice's model code: 32, 32, 64, 64, 128, 128).</summary>
    public int[] ReferenceFilters { get; init; } = [32, 32, 64, 64, 128, 128];

    /// <summary>The spectrogram's bins.</summary>
    public int Bins => FilterLength / 2 + 1;

    /// <summary>Whether <paramref name="config"/> has the shape of an OpenVoice config.json: "data" with the spectrogram's settings and "model" with "gin_channels".</summary>
    public static bool Recognizes(JsonObject config) =>
        config["data"] is JsonObject data && data["filter_length"] is not null && data["hop_length"] is not null
        && config["model"] is JsonObject model && model["gin_channels"] is not null && model["inter_channels"] is not null && config["architectures"] is null;

    /// <summary>Reads an OpenVoice config.json.</summary>
    /// <exception cref="InvalidDataException">A section is missing, it has several speakers, or a setting is unsupported.</exception>
    public static OpenVoiceConfig FromJson(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var data = config["data"] as JsonObject ?? throw new InvalidDataException("An OpenVoice config.json has a \"data\" section.");
        var model = config["model"] as JsonObject ?? throw new InvalidDataException("An OpenVoice config.json has a \"model\" section.");
        if (data["n_speakers"] is JsonValue speakers && speakers.GetValue<int>() != 0)
        {
            throw new InvalidDataException($"data.n_speakers is {speakers}: a converter with speaker ids (no reference encoder) is not supported; OpenVoice's converters have 0.");
        }

        var d = new OpenVoiceConfig();
        var result = d with
        {
            Version = config["_version_"]?.GetValue<string>() ?? d.Version,
            SamplingRate = Int(data, "sampling_rate", d.SamplingRate),
            FilterLength = Int(data, "filter_length", d.FilterLength),
            HopLength = Int(data, "hop_length", d.HopLength),
            WinLength = Int(data, "win_length", d.WinLength),
            InterChannels = Int(model, "inter_channels", d.InterChannels),
            HiddenChannels = Int(model, "hidden_channels", d.HiddenChannels),
            GinChannels = Int(model, "gin_channels", d.GinChannels),
            ZeroG = model["zero_g"]?.GetValue<bool>() ?? d.ZeroG,
            Resblock = model["resblock"] is JsonValue r ? (r.TryGetValue(out string? text) ? text : r.ToJsonString()) : d.Resblock,
            ResblockKernelSizes = Ints(model, "resblock_kernel_sizes", d.ResblockKernelSizes),
            ResblockDilationSizes = model["resblock_dilation_sizes"] is JsonArray lists
                ? [.. lists.Select(l => l is JsonArray a ? a.Select(v => v!.GetValue<int>()).ToArray() : throw new InvalidDataException("model.resblock_dilation_sizes holds lists."))]
                : d.ResblockDilationSizes,
            UpsampleRates = Ints(model, "upsample_rates", d.UpsampleRates),
            UpsampleInitialChannel = Int(model, "upsample_initial_channel", d.UpsampleInitialChannel),
            UpsampleKernelSizes = Ints(model, "upsample_kernel_sizes", d.UpsampleKernelSizes),
            LeakyReluSlope = model["leaky_relu_slope"] is JsonValue slope ? (float)slope.GetValue<double>() : d.LeakyReluSlope,
            FinalLeakyReluSlope = model["final_leaky_relu_slope"] is JsonValue finalSlope ? (float)finalSlope.GetValue<double>() : d.FinalLeakyReluSlope,
        };
        if (result.Resblock != "1")
        {
            throw new InvalidDataException($"model.resblock is \"{result.Resblock}\"; only \"1\" (the block OpenVoice's converters use) is supported.");
        }

        if (result.WinLength != result.FilterLength)
        {
            throw new InvalidDataException($"data.win_length ({result.WinLength}) differs from data.filter_length ({result.FilterLength}); only equal ones (OpenVoice's) are supported.");
        }

        if (result.UpsampleRates.Aggregate(1, (a, b) => a * b) != result.HopLength)
        {
            throw new InvalidDataException($"model.upsample_rates multiply to {result.UpsampleRates.Aggregate(1, (a, b) => a * b)}, data.hop_length is {result.HopLength}; they agree.");
        }

        return result;
    }

    private static int Int(JsonObject section, string key, int fallback) => section[key] is JsonValue v ? v.GetValue<int>() : fallback;

    private static int[] Ints(JsonObject section, string key, int[] fallback) =>
        section[key] is JsonArray a ? [.. a.Select(v => v!.GetValue<int>())] : fallback;
}
