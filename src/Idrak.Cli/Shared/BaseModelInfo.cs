// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.LanguageModels;

namespace Idrak.Cli.Shared;

/// <summary>
/// What the design rules need of a base model (<c>idrak suggest --base MODEL</c>) without loading its weights: the
/// parameter count, sizes and context from its config (a folder's config.json or a GGUF file's metadata), and its
/// tokenizer to count tokens. A Hugging Face id is looked up in the local caches only (no download: <c>idrak pull</c>
/// fetches it); when it is not there the rules continue with what they know and say so.
/// </summary>
internal sealed record BaseModelInfo(string Name, string? Folder, long? Parameters, int? Hidden, int? Layers, int? Context, BpeTokenizer? Tokenizer)
{
    /// <summary>Reads what is available about <paramref name="name"/> (after alias resolution).</summary>
    public static BaseModelInfo Inspect(string name)
    {
        string? folder = null;
        try
        {
            if (File.Exists(name) && name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                return FromGguf(name);
            }

            folder = Directory.Exists(name) ? name : ModelSource.Resolve(name, download: false);
        }
        catch (Exception e) when (e is DirectoryNotFoundException or FileNotFoundException or InvalidOperationException or NotSupportedException)
        {
            return new BaseModelInfo(name, null, null, null, null, null, null);
        }

        string config = Path.Combine(folder, "config.json");
        long? parameters = null;
        int? hidden = null, layers = null, context = null;
        if (File.Exists(config) && JsonNode.Parse(File.ReadAllText(config)) is JsonObject c)
        {
            var text = c["text_config"] as JsonObject ?? c;
            hidden = (int?)text["hidden_size"];
            layers = (int?)text["num_hidden_layers"];
            context = (int?)text["max_position_embeddings"];
            int? inter = (int?)text["intermediate_size"], vocab = (int?)text["vocab_size"], heads = (int?)text["num_attention_heads"];
            int? kvHeads = (int?)text["num_key_value_heads"] ?? heads;
            if (hidden is { } h && layers is { } l && inter is { } ff && vocab is { } v && heads is { } nh && kvHeads is { } kv)
            {
                long headDim = (int?)text["head_dim"] ?? h / nh;
                long attention = h * nh * headDim * 2 + h * kv * headDim * 2;   // q and o, k and v
                long feedForward = 3L * h * ff;                                  // gate, up and down
                bool tied = (bool?)c["tie_word_embeddings"] ?? (bool?)text["tie_word_embeddings"] ?? false;
                parameters = l * (attention + feedForward + 2L * h) + (long)v * h * (tied ? 1 : 2) + h;
            }
        }

        string tokenizer = Path.Combine(folder, "tokenizer.json");
        return new BaseModelInfo(name, folder, parameters, hidden, layers, context, File.Exists(tokenizer) ? BpeTokenizer.Load(tokenizer) : null);
    }

    private static BaseModelInfo FromGguf(string path)
    {
        using var file = GgufFile.Open(path);
        long parameters = file.Tensors.Values.Sum(t => t.Count);
        string arch = file.Get("general.architecture", "llama");
        int? Int(string key) => file.Metadata.TryGetValue($"{arch}.{key}", out var v) ? Convert.ToInt32(v, System.Globalization.CultureInfo.InvariantCulture) : null;
        return new BaseModelInfo(path, null, parameters, Int("embedding_length"), Int("block_count"), Int("context_length"), null);
    }
}
