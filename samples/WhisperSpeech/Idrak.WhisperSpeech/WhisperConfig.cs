// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.WhisperSpeech;

/// <summary>
/// A Whisper-style checkpoint's settings, every one read from its files: the sizes from config.json, the token ids from
/// generation_config.json (or, failing that, the tokenizer's added tokens), nothing fixed in code.
/// </summary>
public sealed partial record WhisperConfig
{
    /// <summary>Model width (d_model).</summary>
    public required int Dim { get; init; }

    /// <summary>Encoder layers.</summary>
    public required int EncoderLayers { get; init; }

    /// <summary>Decoder layers.</summary>
    public required int DecoderLayers { get; init; }

    /// <summary>Encoder attention heads.</summary>
    public required int EncoderHeads { get; init; }

    /// <summary>Decoder attention heads.</summary>
    public required int DecoderHeads { get; init; }

    /// <summary>The encoder's feed-forward width.</summary>
    public required int EncoderFfDim { get; init; }

    /// <summary>The decoder's feed-forward width.</summary>
    public required int DecoderFfDim { get; init; }

    /// <summary>Mel bands of the features (num_mel_bins).</summary>
    public required int Mels { get; init; }

    /// <summary>Encoder positions (frames / 2).</summary>
    public required int MaxSourcePositions { get; init; }

    /// <summary>Decoder positions (prompt and text).</summary>
    public required int MaxTargetPositions { get; init; }

    /// <summary>Vocabulary size.</summary>
    public required int Vocabulary { get; init; }

    /// <summary>The feed-forward activation (gelu, gelu_new, relu).</summary>
    public string Activation { get; init; } = "gelu";

    /// <summary>Layer normalization epsilon.</summary>
    public float NormEpsilon { get; init; } = 1e-5f;

    /// <summary>Multiply the token embeddings by √dim (scale_embedding).</summary>
    public bool ScaleEmbedding { get; init; }

    /// <summary>The token every transcript starts with (decoder_start_token_id, &lt;|startoftranscript|&gt;).</summary>
    public required int StartToken { get; init; }

    /// <summary>The end-of-text token.</summary>
    public required int EndToken { get; init; }

    /// <summary>The token asking for no timestamps (&lt;|notimestamps|&gt;); the timestamp tokens follow it.</summary>
    public required int NoTimestampsToken { get; init; }

    /// <summary>The first timestamp token (time 0); token t stands for (t - TimestampBegin) · <see cref="TimePrecision"/> seconds.</summary>
    public int TimestampBegin => NoTimestampsToken + 1;

    /// <summary>The token that opens a prompt of earlier text (&lt;|startofprev|&gt;), or null when the checkpoint has none.</summary>
    public int? PrevToken { get; init; }

    /// <summary>
    /// The token whose probability after &lt;|startoftranscript|&gt; says there is no speech (&lt;|nospeech|&gt;, or
    /// &lt;|nocaptions|&gt; in the first checkpoints): the one just before &lt;|notimestamps|&gt;, as transformers takes it.
    /// </summary>
    public required int NoSpeechToken { get; init; }

    /// <summary>The language tokens by code ("en" → the id of &lt;|en|&gt;).</summary>
    public required IReadOnlyDictionary<string, int> Languages { get; init; }

    /// <summary>The task tokens by name ("transcribe", "translate").</summary>
    public required IReadOnlyDictionary<string, int> Tasks { get; init; }

    /// <summary>Tokens never chosen at any step (suppress_tokens).</summary>
    public IReadOnlyList<int> SuppressTokens { get; init; } = [];

    /// <summary>Tokens never chosen at the first step (begin_suppress_tokens).</summary>
    public IReadOnlyList<int> BeginSuppressTokens { get; init; } = [];

    /// <summary>Seconds per timestamp step.</summary>
    public double TimePrecision { get; init; } = 0.02;

    /// <summary>How many timestamp steps into a segment its first timestamp may fall (max_initial_timestamp_index), or null for any.</summary>
    public int? MaxInitialTimestampIndex { get; init; } = 50;

    /// <summary>Each head's size.</summary>
    public int EncoderHeadDim => Dim / EncoderHeads;

    /// <summary>Each decoder head's size.</summary>
    public int DecoderHeadDim => Dim / DecoderHeads;

    /// <summary>
    /// Reads the settings: the sizes from <paramref name="config"/> (config.json), the token ids from
    /// <paramref name="generation"/> (generation_config.json) when given, else from config.json and the tokenizer's
    /// <paramref name="addedTokens"/> (content → id).
    /// </summary>
    /// <exception cref="InvalidDataException">A size is missing, or a token the family needs is nowhere.</exception>
    public static WhisperConfig Read(JsonObject config, JsonObject? generation, IReadOnlyDictionary<string, int> addedTokens)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(addedTokens);
        int Size(string key) => config[key] is { } v && v.GetValueKind() == JsonValueKind.Number ? (int)v
            : throw new InvalidDataException($"A Whisper config.json has \"{key}\".");
        int? Id(string key) => (generation?[key] ?? config[key]) is { } v && v.GetValueKind() == JsonValueKind.Number ? (int)v : null;
        int? Added(string content) => addedTokens.TryGetValue(content, out int id) ? id : null;
        int Token(string key, string content) => Id(key) ?? Added(content)
            ?? throw new InvalidDataException($"Neither generation_config.json nor config.json has \"{key}\", and the tokenizer has no \"{content}\" token.");

        var languages = new Dictionary<string, int>(StringComparer.Ordinal);
        if ((generation?["lang_to_id"] ?? config["lang_to_id"]) is JsonObject langToId)
        {
            foreach (var (token, id) in langToId)
            {
                languages[Bare(token)] = (int)id!;
            }
        }
        else
        {
            foreach (var (content, id) in addedTokens)
            {
                if (LanguageToken().IsMatch(content))
                {
                    languages[Bare(content)] = id;
                }
            }
        }

        var tasks = new Dictionary<string, int>(StringComparer.Ordinal);
        if ((generation?["task_to_id"] ?? config["task_to_id"]) is JsonObject taskToId)
        {
            foreach (var (task, id) in taskToId)
            {
                tasks[task] = (int)id!;
            }
        }
        else
        {
            foreach (string task in new[] { "transcribe", "translate" })
            {
                if (Added($"<|{task}|>") is { } id)
                {
                    tasks[task] = id;
                }
            }
        }

        int vocabulary = Size("vocab_size");
        List<int> Ids(string key)
        {
            var node = generation?[key] ?? config[key];
            return node is JsonArray array ? [.. array.Select(v => (int)v!).Where(id => (uint)id < (uint)vocabulary)] : [];
        }

        var noTimestamps = Token("no_timestamps_token_id", "<|notimestamps|>");
        return new WhisperConfig
        {
            Dim = Size("d_model"), EncoderLayers = Size("encoder_layers"), DecoderLayers = Size("decoder_layers"),
            EncoderHeads = Size("encoder_attention_heads"), DecoderHeads = Size("decoder_attention_heads"),
            EncoderFfDim = Size("encoder_ffn_dim"), DecoderFfDim = Size("decoder_ffn_dim"), Mels = Size("num_mel_bins"),
            MaxSourcePositions = Size("max_source_positions"), MaxTargetPositions = Size("max_target_positions"), Vocabulary = vocabulary,
            Activation = (string?)config["activation_function"] ?? "gelu",
            NormEpsilon = config["layer_norm_eps"] is { } eps && eps.GetValueKind() == JsonValueKind.Number ? (float)eps : 1e-5f,
            ScaleEmbedding = (bool?)config["scale_embedding"] ?? false,
            StartToken = Token("decoder_start_token_id", "<|startoftranscript|>"),
            EndToken = Token("eos_token_id", "<|endoftext|>"),
            NoTimestampsToken = noTimestamps,
            PrevToken = Id("prev_sot_token_id") ?? Added("<|startofprev|>"),
            NoSpeechToken = Added("<|nospeech|>") ?? Added("<|nocaptions|>") ?? noTimestamps - 1,
            Languages = languages,
            Tasks = tasks,
            SuppressTokens = Ids("suppress_tokens"),
            BeginSuppressTokens = Ids("begin_suppress_tokens"),
            TimePrecision = (generation?["time_precision"] ?? config["time_precision"]) is { } precision && precision.GetValueKind() == JsonValueKind.Number ? (double)precision : 0.02,
            MaxInitialTimestampIndex = (generation?["max_initial_timestamp_index"] ?? config["max_initial_timestamp_index"]) is { } initial
                ? initial.GetValueKind() == JsonValueKind.Number ? (int)initial : null : 50,
        };
    }

    /// <summary>The added tokens of a tokenizer.json (content → id), read without the vocabulary.</summary>
    public static IReadOnlyDictionary<string, int> AddedTokens(string tokenizerFile)
    {
        var added = new Dictionary<string, int>(StringComparer.Ordinal);
        using var document = JsonDocument.Parse(File.ReadAllBytes(tokenizerFile), new JsonDocumentOptions { MaxDepth = 256 });
        if (document.RootElement.TryGetProperty("added_tokens", out var tokens) && tokens.ValueKind == JsonValueKind.Array)
        {
            foreach (var token in tokens.EnumerateArray())
            {
                if (token.TryGetProperty("content", out var content) && token.TryGetProperty("id", out var id))
                {
                    added[content.GetString()!] = id.GetInt32();
                }
            }
        }

        return added;
    }

    // "<|en|>" → "en".
    private static string Bare(string token) => token.StartsWith("<|", StringComparison.Ordinal) && token.EndsWith("|>", StringComparison.Ordinal) ? token[2..^2] : token;

    [GeneratedRegex(@"^<\|[a-z]{2,3}\|>$")]
    private static partial Regex LanguageToken();
}
