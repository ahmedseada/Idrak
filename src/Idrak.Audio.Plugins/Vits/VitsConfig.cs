// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Audio.Plugins.Vits;

/// <summary>
/// A VITS checkpoint's config.json (transformers' <c>VitsConfig</c> keys, its defaults where a key is missing). Every
/// shape and scale the model uses comes from here; nothing is a constant of the plug-in.
/// </summary>
public sealed record VitsConfig
{
    /// <summary>The text encoder's width.</summary>
    public int HiddenSize { get; init; } = 192;

    /// <summary>The text encoder's layers.</summary>
    public int Layers { get; init; } = 6;

    /// <summary>The attention heads of each layer.</summary>
    public int Heads { get; init; } = 2;

    /// <summary>The relative-position window of the attention (0: none).</summary>
    public int WindowSize { get; init; } = 4;

    /// <summary>Whether the attention projections have biases.</summary>
    public bool UseBias { get; init; } = true;

    /// <summary>The feed-forward width.</summary>
    public int FfnDim { get; init; } = 768;

    /// <summary>The feed-forward convolutions' kernel.</summary>
    public int FfnKernelSize { get; init; } = 3;

    /// <summary>The feed-forward activation (relu, gelu, gelu_new, silu).</summary>
    public string HiddenAct { get; init; } = "relu";

    /// <summary>The layer norms' epsilon.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>The acoustic latents' channels (the flow's width, the decoder's input).</summary>
    public int FlowSize { get; init; } = 192;

    /// <summary>The speaker embedding's size (0: a single speaker).</summary>
    public int SpeakerEmbeddingSize { get; init; }

    /// <summary>The speakers.</summary>
    public int Speakers { get; init; } = 1;

    /// <summary>The decoder's first channel count, halved at each upsampling stage.</summary>
    public int UpsampleInitialChannel { get; init; } = 512;

    /// <summary>The decoder's upsampling factors; their product is the samples per frame (the hop).</summary>
    public int[] UpsampleRates { get; init; } = [8, 8, 2, 2];

    /// <summary>The transposed convolutions' kernels, one per rate.</summary>
    public int[] UpsampleKernelSizes { get; init; } = [16, 16, 4, 4];

    /// <summary>The residual blocks' kernels (one block per kernel at each stage).</summary>
    public int[] ResblockKernelSizes { get; init; } = [3, 7, 11];

    /// <summary>The dilations of each residual block's convolutions.</summary>
    public int[][] ResblockDilationSizes { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>The leaky ReLU slope inside the decoder (the last activation before the output convolution uses <see cref="FinalLeakyReluSlope"/>).</summary>
    public float LeakyReluSlope { get; init; } = 0.1f;

    /// <summary>The slope of the decoder's last leaky ReLU, which transformers' <c>VitsHifiGan</c> leaves at PyTorch's default (0.01); a setting here since it is not in config.json.</summary>
    public float FinalLeakyReluSlope { get; init; } = 0.01f;

    /// <summary>The channels of the duration predictor's flows (the halves split and transformed).</summary>
    public int DepthSeparableChannels { get; init; } = 2;

    /// <summary>The layers of each dilated depth-separable convolution.</summary>
    public int DepthSeparableLayers { get; init; } = 3;

    /// <summary>The bins of the rational-quadratic splines.</summary>
    public int DurationFlowBins { get; init; } = 10;

    /// <summary>The bound outside which the splines are the identity.</summary>
    public float DurationTailBound { get; init; } = 5f;

    /// <summary>The duration predictor's convolution kernel.</summary>
    public int DurationKernelSize { get; init; } = 3;

    /// <summary>The duration predictor's filter channels (the deterministic predictor's width).</summary>
    public int DurationFilterChannels { get; init; } = 256;

    /// <summary>The spline flows of the stochastic duration predictor.</summary>
    public int DurationFlows { get; init; } = 4;

    /// <summary>Whether durations come from the stochastic predictor (flows, noise) or the deterministic one.</summary>
    public bool StochasticDurations { get; init; } = true;

    /// <summary>The coupling layers of the prior flow.</summary>
    public int PriorFlows { get; init; } = 4;

    /// <summary>The WaveNet layers inside each coupling layer.</summary>
    public int PriorWaveNetLayers { get; init; } = 4;

    /// <summary>The linear spectrogram's bins the posterior encoder reads (n_fft / 2 + 1).</summary>
    public int SpectrogramBins { get; init; } = 513;

    /// <summary>The WaveNet layers of the posterior encoder.</summary>
    public int PosteriorWaveNetLayers { get; init; } = 16;

    /// <summary>The WaveNet convolutions' kernel.</summary>
    public int WaveNetKernelSize { get; init; } = 5;

    /// <summary>The WaveNet dilation growth per layer.</summary>
    public int WaveNetDilationRate { get; init; } = 1;

    /// <summary>The configured scale of the noise added to the acoustic latents.</summary>
    public float NoiseScale { get; init; } = 0.667f;

    /// <summary>The configured scale of the duration predictor's noise.</summary>
    public float NoiseScaleDuration { get; init; } = 0.8f;

    /// <summary>The configured speaking rate (1 / length scale).</summary>
    public float SpeakingRate { get; init; } = 1f;

    /// <summary>The output rate.</summary>
    public int SamplingRate { get; init; } = 16000;

    /// <summary>The vocabulary's size.</summary>
    public int VocabSize { get; init; } = 38;

    /// <summary>The samples one frame of latents makes: the product of <see cref="UpsampleRates"/>.</summary>
    public int HopLength => UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <summary>Reads a config.json (missing keys take transformers' defaults).</summary>
    /// <exception cref="InvalidDataException">A list key has a value of the wrong form, or the upsampling lists disagree in length.</exception>
    public static VitsConfig FromJson(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var defaults = new VitsConfig();
        var result = defaults with
        {
            HiddenSize = Int(config, "hidden_size", defaults.HiddenSize),
            Layers = Int(config, "num_hidden_layers", defaults.Layers),
            Heads = Int(config, "num_attention_heads", defaults.Heads),
            WindowSize = Int(config, "window_size", defaults.WindowSize),
            UseBias = config["use_bias"]?.GetValue<bool>() ?? defaults.UseBias,
            FfnDim = Int(config, "ffn_dim", defaults.FfnDim),
            FfnKernelSize = Int(config, "ffn_kernel_size", defaults.FfnKernelSize),
            HiddenAct = config["hidden_act"]?.GetValue<string>() ?? defaults.HiddenAct,
            LayerNormEps = Float(config, "layer_norm_eps", defaults.LayerNormEps),
            FlowSize = Int(config, "flow_size", defaults.FlowSize),
            SpeakerEmbeddingSize = Int(config, "speaker_embedding_size", defaults.SpeakerEmbeddingSize),
            Speakers = Int(config, "num_speakers", defaults.Speakers),
            UpsampleInitialChannel = Int(config, "upsample_initial_channel", defaults.UpsampleInitialChannel),
            UpsampleRates = Ints(config, "upsample_rates", defaults.UpsampleRates),
            UpsampleKernelSizes = Ints(config, "upsample_kernel_sizes", defaults.UpsampleKernelSizes),
            ResblockKernelSizes = Ints(config, "resblock_kernel_sizes", defaults.ResblockKernelSizes),
            ResblockDilationSizes = config["resblock_dilation_sizes"] is JsonArray lists
                ? [.. lists.Select((l, i) => l is JsonArray a ? a.Select(v => v!.GetValue<int>()).ToArray() : throw new InvalidDataException($"resblock_dilation_sizes[{i}] is not a list."))]
                : defaults.ResblockDilationSizes,
            LeakyReluSlope = Float(config, "leaky_relu_slope", defaults.LeakyReluSlope),
            FinalLeakyReluSlope = Float(config, "final_leaky_relu_slope", defaults.FinalLeakyReluSlope),
            DepthSeparableChannels = Int(config, "depth_separable_channels", defaults.DepthSeparableChannels),
            DepthSeparableLayers = Int(config, "depth_separable_num_layers", defaults.DepthSeparableLayers),
            DurationFlowBins = Int(config, "duration_predictor_flow_bins", defaults.DurationFlowBins),
            DurationTailBound = Float(config, "duration_predictor_tail_bound", defaults.DurationTailBound),
            DurationKernelSize = Int(config, "duration_predictor_kernel_size", defaults.DurationKernelSize),
            DurationFilterChannels = Int(config, "duration_predictor_filter_channels", defaults.DurationFilterChannels),
            DurationFlows = Int(config, "duration_predictor_num_flows", defaults.DurationFlows),
            StochasticDurations = config["use_stochastic_duration_prediction"]?.GetValue<bool>() ?? defaults.StochasticDurations,
            PriorFlows = Int(config, "prior_encoder_num_flows", defaults.PriorFlows),
            PriorWaveNetLayers = Int(config, "prior_encoder_num_wavenet_layers", defaults.PriorWaveNetLayers),
            SpectrogramBins = Int(config, "spectrogram_bins", defaults.SpectrogramBins),
            PosteriorWaveNetLayers = Int(config, "posterior_encoder_num_wavenet_layers", defaults.PosteriorWaveNetLayers),
            WaveNetKernelSize = Int(config, "wavenet_kernel_size", defaults.WaveNetKernelSize),
            WaveNetDilationRate = Int(config, "wavenet_dilation_rate", defaults.WaveNetDilationRate),
            NoiseScale = Float(config, "noise_scale", defaults.NoiseScale),
            NoiseScaleDuration = Float(config, "noise_scale_duration", defaults.NoiseScaleDuration),
            SpeakingRate = Float(config, "speaking_rate", defaults.SpeakingRate),
            SamplingRate = Int(config, "sampling_rate", defaults.SamplingRate),
            VocabSize = Int(config, "vocab_size", defaults.VocabSize),
        };
        if (result.UpsampleRates.Length != result.UpsampleKernelSizes.Length)
        {
            throw new InvalidDataException($"upsample_rates has {result.UpsampleRates.Length} entries and upsample_kernel_sizes {result.UpsampleKernelSizes.Length}; they pair up.");
        }

        if (result.ResblockKernelSizes.Length != result.ResblockDilationSizes.Length)
        {
            throw new InvalidDataException($"resblock_kernel_sizes has {result.ResblockKernelSizes.Length} entries and resblock_dilation_sizes {result.ResblockDilationSizes.Length}; they pair up.");
        }

        return result;
    }

    private static int Int(JsonObject config, string key, int fallback) => config[key] is JsonValue v ? v.GetValue<int>() : fallback;

    private static float Float(JsonObject config, string key, float fallback) => config[key] is JsonValue v ? (float)v.GetValue<double>() : fallback;

    private static int[] Ints(JsonObject config, string key, int[] fallback) =>
        config[key] is JsonArray a ? [.. a.Select(v => v!.GetValue<int>())] : fallback;
}
