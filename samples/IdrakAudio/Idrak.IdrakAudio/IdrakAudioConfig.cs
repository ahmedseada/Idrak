// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.IdrakAudio;

/// <summary>
/// An IdrakAudio model's settings, as its config.json holds them (and as a spec gives them to <c>SpeechModels.Create</c>,
/// a missing key taking the default below). Every shape is a setting; nothing is tied to a device.
/// </summary>
public sealed record IdrakAudioConfig
{
    /// <summary>
    /// What the model does: "ctc" (a recognizer with a CTC head), "seq2seq" (the encoder and a text decoder attending to
    /// it) or "tts" (a synthesizer: text encoder, duration predictor, length regulation, mel decoder).
    /// </summary>
    public string Task { get; init; } = IdrakAudioFamily.CtcTask;

    /// <summary>Mel bands of the features (the subsampler's input channels).</summary>
    public int Mels { get; init; } = 80;

    /// <summary>The encoder's width.</summary>
    public int Dim { get; init; } = 256;

    /// <summary>Encoder blocks.</summary>
    public int Layers { get; init; } = 6;

    /// <summary>Attention heads (a divisor of <see cref="Dim"/>).</summary>
    public int Heads { get; init; } = 4;

    /// <summary>The feed-forward's hidden width.</summary>
    public int FfnDim { get; init; } = 1024;

    /// <summary>Convolutions of stride 2 over time before the blocks (each halves the steps; 0 for none).</summary>
    public int SubsampleLayers { get; init; } = 2;

    /// <summary>Their kernel over time (odd: padded by half on each side).</summary>
    public int SubsampleKernel { get; init; } = 3;

    /// <summary>"sinusoidal" positions added after the subsampler, or "none".</summary>
    public string Positions { get; init; } = "sinusoidal";

    /// <summary>A Conformer-style convolution module in each block, between attention and feed-forward.</summary>
    public bool ConvolutionModule { get; init; }

    /// <summary>The convolution module's depthwise kernel over time (odd).</summary>
    public int ConvolutionKernel { get; init; } = 15;

    /// <summary>"gelu_tanh" (the fused GELU) or "gelu" (exact).</summary>
    public string Activation { get; init; } = "gelu_tanh";

    /// <summary>The layer normalizations' epsilon.</summary>
    public float NormEpsilon { get; init; } = 1e-5f;

    /// <summary>Dropout after attention, the convolution module and the feed-forward while training (0: none, and resumed runs bit for bit).</summary>
    public float Dropout { get; init; }

    /// <summary>The weights drawn at creation: a normal draw of this over the square root of each matrix's inputs (biases 0, norms 1).</summary>
    public float InitScale { get; init; } = 1f;

    /// <summary>The head's classes: the tokenizer's tokens and the blank ("ctc"), or the tokens and the start and end token ("seq2seq").</summary>
    public int Classes { get; init; }

    /// <summary>The blank class ("ctc"); token i is class i below it and i + 1 from it on.</summary>
    public int BlankId { get; init; }

    /// <summary>"seq2seq": the decoder's blocks.</summary>
    public int DecoderLayers { get; init; } = 4;

    /// <summary>"seq2seq": the decoder's width (its cross-attention reads the encoder's <see cref="Dim"/>); the spec's default is <see cref="Dim"/>.</summary>
    public int DecoderDim { get; init; } = 256;

    /// <summary>"seq2seq": the decoder's attention heads (a divisor of <see cref="DecoderDim"/>); the spec's default is <see cref="Heads"/>.</summary>
    public int DecoderHeads { get; init; } = 4;

    /// <summary>"seq2seq": the decoder's feed-forward width; the spec's default is <see cref="FfnDim"/>.</summary>
    public int DecoderFfnDim { get; init; } = 1024;

    /// <summary>"seq2seq": the decoder's learned positions (the start token, the labels and the end token fit in them).</summary>
    public int MaxTargetPositions { get; init; } = 448;

    /// <summary>"seq2seq": the head reads the token embedding's table (true) or has its own weight ("decoder.head.weight").</summary>
    public bool TieEmbeddings { get; init; } = true;

    /// <summary>"seq2seq": the token the decoder starts from; the spec's default is the last class (the one past the tokenizer's tokens).</summary>
    public int StartToken { get; init; }

    /// <summary>"seq2seq": the token that ends a text; the spec's default is the last class, as the start token.</summary>
    public int EndToken { get; init; }

    /// <summary>Whether the task is "seq2seq".</summary>
    public bool Sequence => Task == IdrakAudioFamily.Seq2SeqTask;

    /// <summary>Whether the task is "tts" (a synthesizer: the text encoder over <see cref="Layers"/> blocks, the mel decoder over the decoder's).</summary>
    public bool Synthesis => Task == IdrakAudioFamily.TtsTask;

    /// <summary>"tts": the duration predictor's channels (its two convolutions); the spec's default is <see cref="Dim"/>.</summary>
    public int DurationFilters { get; init; } = 256;

    /// <summary>"tts": the duration predictor's kernel over the labels (odd: padded by half on each side).</summary>
    public int DurationKernel { get; init; } = 3;

    /// <summary>"tts": the duration predictor gives log(frames + this), so a label of no frames has a finite target.</summary>
    public float DurationOffset { get; init; } = 1f;

    /// <summary>"tts": the <see cref="Vocoders"/> name it speaks through when the call names none; null: the library's default ("griffin-lim").</summary>
    public string? Vocoder { get; init; }

    /// <summary>The feature extractor (an <see cref="AudioFeatureExtractors"/> name).</summary>
    public string FeatureExtractor { get; init; } = "log-mel";

    /// <summary>The features: a family's rate, window, hop and mels; no padding to a segment (each utterance's own length).</summary>
    public AudioFeatureOptions Features { get; init; } = new() { PadSeconds = null };

    /// <summary>Seconds of sound transcribed at a time when the transcription's options give no chunk length.</summary>
    public double MaxSeconds { get; init; } = 30;

    /// <summary>The steps the head gives for <paramref name="frames"/> feature frames.</summary>
    public int StepsFor(int frames)
    {
        int steps = frames;
        for (int i = 0; i < SubsampleLayers; i++)
        {
            steps = (steps + 2 * (SubsampleKernel / 2) - SubsampleKernel) / 2 + 1;
        }

        return Math.Max(steps, 0);
    }

    /// <summary>The settings of a config.json or a spec; <paramref name="vocabulary"/> (a created model's tokenizer) sets the classes when the JSON has none.</summary>
    /// <exception cref="InvalidDataException">A value is out of range or of the wrong kind.</exception>
    public static IdrakAudioConfig FromJson(JsonObject json, int? vocabulary = null)
    {
        ArgumentNullException.ThrowIfNull(json);
        var d = new IdrakAudioConfig();
        try
        {
            int mels = (int?)json["mels"] ?? d.Mels;
            var features = json["features"] is JsonObject f ? ReadFeatures(f, mels) : d.Features with { Mels = mels };
            int dim = (int?)json["dim"] ?? d.Dim, heads = (int?)json["heads"] ?? d.Heads, ffn = (int?)json["ffn_dim"] ?? d.FfnDim;
            int classes = (int?)json["classes"] ?? (vocabulary is { } v ? v + ((string?)json["task"] == IdrakAudioFamily.TtsTask ? 0 : 1) : 0);   // tts: no class past the tokens
            var config = new IdrakAudioConfig
            {
                Task = (string?)json["task"] ?? d.Task,
                Mels = mels,
                Dim = dim,
                Layers = (int?)json["layers"] ?? d.Layers,
                Heads = heads,
                FfnDim = ffn,
                SubsampleLayers = (int?)json["subsample_layers"] ?? d.SubsampleLayers,
                SubsampleKernel = (int?)json["subsample_kernel"] ?? d.SubsampleKernel,
                Positions = (string?)json["positions"] ?? d.Positions,
                ConvolutionModule = (bool?)json["convolution_module"] ?? d.ConvolutionModule,
                ConvolutionKernel = (int?)json["convolution_kernel"] ?? d.ConvolutionKernel,
                Activation = (string?)json["activation"] ?? d.Activation,
                NormEpsilon = (float?)json["norm_epsilon"] ?? d.NormEpsilon,
                Dropout = (float?)json["dropout"] ?? d.Dropout,
                InitScale = (float?)json["init_scale"] ?? d.InitScale,
                Classes = classes,
                BlankId = (int?)json["blank_id"] ?? d.BlankId,
                DecoderLayers = (int?)json["decoder_layers"] ?? d.DecoderLayers,
                DecoderDim = (int?)json["decoder_dim"] ?? dim,
                DecoderHeads = (int?)json["decoder_heads"] ?? heads,
                DecoderFfnDim = (int?)json["decoder_ffn_dim"] ?? ffn,
                MaxTargetPositions = (int?)json["max_target_positions"] ?? d.MaxTargetPositions,
                TieEmbeddings = (bool?)json["tie_embeddings"] ?? d.TieEmbeddings,
                StartToken = (int?)json["start_token"] ?? classes - 1,
                EndToken = (int?)json["end_token"] ?? classes - 1,
                FeatureExtractor = (string?)json["feature_extractor"] ?? d.FeatureExtractor,
                Features = features,
                MaxSeconds = (double?)json["max_seconds"] ?? d.MaxSeconds,
                DurationFilters = (int?)json["duration_filters"] ?? dim,
                DurationKernel = (int?)json["duration_kernel"] ?? d.DurationKernel,
                DurationOffset = (float?)json["duration_offset"] ?? d.DurationOffset,
                Vocoder = (string?)json["vocoder"],
            };
            config.Check();
            return config;
        }
        catch (Exception e) when (e is InvalidOperationException or FormatException or ArgumentException)
        {
            throw new InvalidDataException($"An IdrakAudio config is not valid: {e.Message}", e);
        }
    }

    private static AudioFeatureOptions ReadFeatures(JsonObject json, int mels)
    {
        var d = new AudioFeatureOptions { PadSeconds = null };
        return new AudioFeatureOptions
        {
            SampleRate = (int?)json["sample_rate"] ?? d.SampleRate,
            WindowSize = (int?)json["window_size"] ?? d.WindowSize,
            HopSize = (int?)json["hop_size"] ?? d.HopSize,
            Window = (string?)json["window"] ?? d.Window,
            Center = (bool?)json["center"] ?? d.Center,
            Mels = (int?)json["mels"] ?? mels,
            MinFrequency = (float?)json["min_frequency"] ?? d.MinFrequency,
            MaxFrequency = json.ContainsKey("max_frequency") ? (float?)json["max_frequency"] : d.MaxFrequency,
            MelScale = (string?)json["mel_scale"] ?? d.MelScale,
            MelNorm = (string?)json["mel_norm"] ?? d.MelNorm,
            LogFloor = (float?)json["log_floor"] ?? d.LogFloor,
            DynamicRange = json.ContainsKey("dynamic_range") ? (float?)json["dynamic_range"] : d.DynamicRange,
            Normalize = json.ContainsKey("normalize")
                ? json["normalize"] is JsonArray { Count: 2 } n ? ((float)n[0]!, (float)n[1]!) : null
                : d.Normalize,
            PadSeconds = json.ContainsKey("pad_seconds") ? (double?)json["pad_seconds"] : d.PadSeconds,
            Path = (string?)json["path"] ?? d.Path,
        };
    }

    private void Check()
    {
        if (Mels < 1 || Dim < 1 || Layers < 0 || Heads < 1 || Dim % Heads != 0 || FfnDim < 1 || SubsampleLayers < 0 || SubsampleKernel < 1 || SubsampleKernel % 2 == 0
            || ConvolutionKernel < 1 || ConvolutionKernel % 2 == 0 || !(NormEpsilon > 0) || Dropout is < 0 or >= 1 || !(InitScale > 0) || !(MaxSeconds > 0))
        {
            throw new ArgumentException($"mels, dim, heads and ffn_dim are positive, dim a multiple of heads ({Dim}, {Heads}), the kernels odd ({SubsampleKernel}, {ConvolutionKernel}), "
                + "layers and subsample_layers not negative, dropout in [0, 1), norm_epsilon, init_scale and max_seconds positive.");
        }

        if (Positions is not ("sinusoidal" or "none") || Activation is not ("gelu_tanh" or "gelu"))
        {
            throw new ArgumentException($"positions is sinusoidal or none ({Positions}); activation gelu_tanh or gelu ({Activation}).");
        }

        if (Synthesis)
        {
            if (Classes < 1 || DurationFilters < 1 || DurationKernel < 1 || DurationKernel % 2 == 0 || !(DurationOffset > 0)
                || DecoderLayers < 0 || DecoderDim < 1 || DecoderHeads < 1 || DecoderDim % DecoderHeads != 0 || DecoderFfnDim < 1)
            {
                throw new ArgumentException($"tts: classes, duration_filters, decoder_dim, decoder_heads and decoder_ffn_dim are positive, decoder_dim a multiple of decoder_heads ({DecoderDim}, {DecoderHeads}), "
                    + $"duration_kernel odd ({DurationKernel}), duration_offset positive ({DurationOffset}) and decoder_layers not negative.");
            }
        }
        else if (Task == IdrakAudioFamily.CtcTask && (Classes < 2 || (uint)BlankId >= (uint)Classes))
        {
            throw new ArgumentException($"The head has {Classes} classes with the blank {BlankId}: at least a label and the blank, the blank one of them.");
        }

        if (Sequence)
        {
            if (Classes < 2 || (uint)StartToken >= (uint)Classes || (uint)EndToken >= (uint)Classes)
            {
                throw new ArgumentException($"The decoder has {Classes} classes with the start token {StartToken} and the end token {EndToken}: at least a label and the end token, both of them classes.");
            }

            if (DecoderLayers < 0 || DecoderDim < 1 || DecoderHeads < 1 || DecoderDim % DecoderHeads != 0 || DecoderFfnDim < 1 || MaxTargetPositions < 2)
            {
                throw new ArgumentException($"decoder_dim, decoder_heads and decoder_ffn_dim are positive, decoder_dim a multiple of decoder_heads ({DecoderDim}, {DecoderHeads}), "
                    + $"decoder_layers not negative and max_target_positions at least 2 ({DecoderLayers}, {MaxTargetPositions}).");
            }
        }

        if (Features.Mels != Mels)
        {
            throw new ArgumentException($"The features give {Features.Mels} mels; the model reads {Mels}.");
        }
    }

    /// <summary>The settings as config.json holds them (with "architectures" and "model_type"; the decoder's for "seq2seq" only).</summary>
    public JsonObject ToJson()
    {
        var json = Common();
        if (Synthesis)
        {
            foreach (string key in (string[])["blank_id", "subsample_layers", "subsample_kernel", "max_seconds"])
            {
                json.Remove(key);
            }

            json["decoder_layers"] = DecoderLayers;
            json["decoder_dim"] = DecoderDim;
            json["decoder_heads"] = DecoderHeads;
            json["decoder_ffn_dim"] = DecoderFfnDim;
            json["duration_filters"] = DurationFilters;
            json["duration_kernel"] = DurationKernel;
            json["duration_offset"] = DurationOffset;
            if (Vocoder is not null)
            {
                json["vocoder"] = Vocoder;
            }

            return json;
        }

        if (Sequence)
        {
            json.Remove("blank_id");
            json["decoder_layers"] = DecoderLayers;
            json["decoder_dim"] = DecoderDim;
            json["decoder_heads"] = DecoderHeads;
            json["decoder_ffn_dim"] = DecoderFfnDim;
            json["max_target_positions"] = MaxTargetPositions;
            json["tie_embeddings"] = TieEmbeddings;
            json["start_token"] = StartToken;
            json["end_token"] = EndToken;
        }

        return json;
    }

    private JsonObject Common() => new()
    {
        ["architectures"] = new JsonArray((JsonNode)IdrakAudioFamily.Architecture),
        ["model_type"] = "idrak-audio",
        ["task"] = Task,
        ["mels"] = Mels, ["dim"] = Dim, ["layers"] = Layers, ["heads"] = Heads, ["ffn_dim"] = FfnDim,
        ["subsample_layers"] = SubsampleLayers, ["subsample_kernel"] = SubsampleKernel, ["positions"] = Positions,
        ["convolution_module"] = ConvolutionModule, ["convolution_kernel"] = ConvolutionKernel, ["activation"] = Activation,
        ["norm_epsilon"] = NormEpsilon, ["dropout"] = Dropout, ["init_scale"] = InitScale, ["classes"] = Classes, ["blank_id"] = BlankId,
        ["feature_extractor"] = FeatureExtractor, ["features"] = Features.ToJson(), ["max_seconds"] = MaxSeconds,
    };

    /// <inheritdoc />
    public override string ToString() => ToJson().ToJsonString(new JsonSerializerOptions { WriteIndented = false });

    internal static string Invariant(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);
}
