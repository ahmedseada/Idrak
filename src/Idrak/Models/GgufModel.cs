// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

// The GGUF pre-tokenizer names that are no family's (GPT-2's own rule), registered by LibraryRegistrations in
// GgufPreTokenizers; the families' names and patterns are plug-ins' (Idrak.Text.Plugins).
internal static class GgufBuiltIns
{
    // The pre-tokenizer names of GPT-2's own rule (null: no split pattern).
    internal static IEnumerable<(string Name, string? Pattern)> PreTokenizers() =>
        new[] { "gpt2", "gpt-2", "default" }.Select(name => (name, (string?)null));
}

/// <summary>
/// Models from GGUF files (the format of llama.cpp and of the local model store). <see cref="Prepare"/> writes a small folder with what the
/// file's metadata describes, in the Hugging Face layout (config.json, tokenizer.json, tokenizer_config.json with the chat
/// template, generation_config.json), and <see cref="PretrainedModel.Load"/> reads the weights from the GGUF file itself,
/// dequantized tensor by tensor, with llama.cpp's names and layouts turned back into the Hugging Face ones (the family's
/// <see cref="GgufArchitecture.TensorName"/>, less what the converter added to their values, <see cref="GgufArchitecture.StoredOffset"/>).
/// Architectures: those registered with <see cref="GgufArchitectures.Register"/> (the library registers none;
/// Idrak.Text.Plugins registers its families'). Experts stored together (blk.N.ffn_gate_exps, [experts, ff, dim]) are read
/// one expert at a time.
/// Tokenizers: by tokenizer.ggml.model through GgufTokenizers: byte-level BPE ("gpt2"), SentencePiece BPE ("llama") and Unigram
/// ("t5"), WordPiece ("bert").
/// </summary>
public static class GgufModel
{
    private const string Marker = "gguf.json";
    private const int FormatVersion = 2;                            // bump when the prepared files change (2: llama, t5 and bert tokenizers)

    /// <summary>Whether <paramref name="folder"/> was made by <see cref="Prepare"/>.</summary>
    public static bool IsPrepared(string folder) => File.Exists(Path.Combine(folder, Marker));

    /// <summary>The GGUF file a prepared folder reads its weights from.</summary>
    public static string SourceOf(string folder) =>
        (string?)JsonNode.Parse(File.ReadAllText(Path.Combine(folder, Marker)))?["file"] ?? throw new InvalidDataException($"{folder}/{Marker} names no file.");

    /// <summary>
    /// The folder describing <paramref name="gguf"/> in the Hugging Face layout, made once per file (by path, size and
    /// time) under <paramref name="cacheRoot"/>/gguf (default: the library's cache folder, <see cref="ComputeResources.CacheFolder"/>).
    /// </summary>
    public static string Prepare(string gguf, string? cacheRoot = null)
    {
        string path = Path.GetFullPath(gguf);
        var info = new FileInfo(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"{path} does not exist.", path);
        }

        string fingerprint = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"{FormatVersion}|{path}|{info.Length}|{info.LastWriteTimeUtc.Ticks}"))))[..12];
        string name = Path.GetFileNameWithoutExtension(path);
        name = name.StartsWith("sha256-", StringComparison.Ordinal) ? name[..Math.Min(name.Length, 19)] : name;   // blobs of the local model store
        cacheRoot ??= ComputeResources.CacheFolder;                                                     // the downloads' cache root
        string folder = Path.Combine(cacheRoot, "gguf", $"{name}-{fingerprint}");
        if (IsPrepared(folder))
        {
            return folder;
        }

        using var file = GgufFile.Open(path);
        var notes = new List<string>();
        string arch = file.Get("general.architecture", "");
        var architecture = GgufArchitectures.Find(arch)
            ?? throw new NotRegisteredException(nameof(GgufArchitectures), arch, GgufArchitectures.Names,
                $"{path}: GGUF architecture '{arch}' is not registered (registered: {string.Join(", ", GgufArchitectures.Names)}); "
                + "add it with GgufArchitectures.Register, or reference Idrak.Text.Plugins and call TextPlugins.RegisterAll().");

        string temp = folder + ".tmp";
        if (Directory.Exists(temp))
        {
            Directory.Delete(temp, true);
        }

        Directory.CreateDirectory(temp);
        var tokens = file.Get<string[]>("tokenizer.ggml.tokens", []);
        WriteJson(Path.Combine(temp, "config.json"), Config(file, arch, architecture, tokens.Length, notes));
        WriteTokenizer(file, temp, tokens, notes);
        WriteJson(Path.Combine(temp, Marker), new JsonObject
        {
            ["file"] = path,
            ["architecture"] = arch,
            ["name"] = file.Get("general.name", name),
            ["notes"] = new JsonArray([.. notes.Select(n => (JsonNode?)JsonValue.Create(n))]),
            ["types"] = new JsonObject([.. file.Tensors.Values.GroupBy(t => GgufFile.TypeName(t.Type))
                .Select(g => KeyValuePair.Create(g.Key, (JsonNode?)JsonValue.Create(g.Count())))]),
        });
        if (Directory.Exists(folder))
        {
            Directory.Delete(folder, true);
        }

        Directory.Move(temp, folder);
        return folder;
    }

    /// <summary>The notes recorded while preparing (fallbacks taken), for <see cref="PretrainedModel.Notes"/>.</summary>
    internal static IEnumerable<string> NotesOf(string folder) =>
        (JsonNode.Parse(File.ReadAllText(Path.Combine(folder, Marker)))?["notes"] as JsonArray ?? []).Select(n => (string?)n ?? "");

    internal static ITensorStore OpenTensors(string folder) => new GgufTensors(GgufFile.Open(SourceOf(folder)));

    // The Hugging Face architecture of a file: the family's, or its one with experts when the file has experts.
    private static string HuggingFaceOf(GgufFile file, string arch, IGgufArchitecture architecture) =>
        file.Get($"{arch}.expert_count", 0) > 0 ? architecture.WithExperts ?? architecture.HuggingFace : architecture.HuggingFace;

    private static JsonObject Config(GgufFile file, string arch, IGgufArchitecture architecture, int vocabulary, List<string> notes)
    {
        string hfArchitecture = HuggingFaceOf(file, arch, architecture);
        int dim = file.Get($"{arch}.embedding_length", 0), heads = file.Get($"{arch}.attention.head_count", 0);
        var config = new JsonObject
        {
            ["architectures"] = new JsonArray(hfArchitecture),
            ["model_type"] = arch,
            ["vocab_size"] = file.Get($"{arch}.vocab_size", vocabulary),
            ["hidden_size"] = dim,
            ["intermediate_size"] = file.Get($"{arch}.feed_forward_length", 0),
            ["num_hidden_layers"] = file.Get($"{arch}.block_count", 0),
            ["num_attention_heads"] = heads,
            ["num_key_value_heads"] = file.Get($"{arch}.attention.head_count_kv", heads),
            ["head_dim"] = file.Get($"{arch}.attention.key_length", heads > 0 ? dim / heads : 0),
            ["max_position_embeddings"] = file.Get($"{arch}.context_length", 4096),
            ["rms_norm_eps"] = file.Get($"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
            ["rope_theta"] = file.Get($"{arch}.rope.freq_base", 10000.0),
            ["tie_word_embeddings"] = !file.Tensors.ContainsKey("output.weight"),
            ["hidden_act"] = "silu",
            ["torch_dtype"] = "bfloat16",
        };
        int experts = file.Get($"{arch}.expert_count", 0);
        if (experts > 0)
        {
            // The experts' keys as transformers names them (the family says what its expert count is called).
            config[architecture.ExpertCountKey] = experts;
            config["num_experts_per_tok"] = file.Get($"{arch}.expert_used_count", 0);
            if (file.Metadata.ContainsKey($"{arch}.expert_feed_forward_length"))
            {
                config["moe_intermediate_size"] = file.Get($"{arch}.expert_feed_forward_length", 0);
            }

            if (file.Metadata.ContainsKey($"{arch}.expert_shared_feed_forward_length"))
            {
                config["shared_expert_intermediate_size"] = file.Get($"{arch}.expert_shared_feed_forward_length", 0);
            }

            if (file.Metadata.ContainsKey($"{arch}.expert_weights_norm"))
            {
                config["norm_topk_prob"] = file.Get($"{arch}.expert_weights_norm", false);
            }
            else if (architecture.NormalizeTopK is { } normalize)
            {
                config["norm_topk_prob"] = normalize;
            }

            // Layers without a router are dense (Qwen's mlp_only_layers).
            int layers = file.Get($"{arch}.block_count", 0);
            var dense = Enumerable.Range(0, layers).Where(l => !file.Tensors.ContainsKey($"blk.{l}.ffn_gate_inp.weight")).ToList();
            if (dense.Count > 0)
            {
                config["mlp_only_layers"] = new JsonArray([.. dense.Select(l => (JsonNode)l)]);
            }
        }

        if (file.Metadata.ContainsKey("tokenizer.ggml.bos_token_id"))
        {
            config["bos_token_id"] = file.Get("tokenizer.ggml.bos_token_id", 0);
        }

        if (file.Metadata.ContainsKey("tokenizer.ggml.eos_token_id"))
        {
            config["eos_token_id"] = file.Get("tokenizer.ggml.eos_token_id", 0);
        }

        // The family's own keys (a RoPE scaling it stores its own way, …); the file's generic scaling keys only when it set none.
        architecture.Configure(file, config, notes);
        bool own = config["rope_scaling"] is not null;
        string scaling = file.Get($"{arch}.rope.scaling.type", "none");
        double factor = file.Get($"{arch}.rope.scaling.factor", 1.0);
        if (!own && scaling == "linear" && factor > 1)
        {
            config["rope_scaling"] = new JsonObject { ["rope_type"] = "linear", ["factor"] = factor };
        }
        else if (!own && scaling is "yarn" && factor > 1)
        {
            // llama.cpp's YaRN keys, as Hugging Face's rope_scaling (the scaling registered as "yarn" reads it): the context
            // the model was trained on, and the ramp's bounds when the file sets them. The context length stays the file's.
            var yarn = new JsonObject { ["rope_type"] = "yarn", ["factor"] = factor };
            int original = file.Get($"{arch}.rope.scaling.original_context_length", 0);
            if (original > 0)
            {
                yarn["original_max_position_embeddings"] = original;
            }

            foreach (var (key, name) in new[] { ("yarn_beta_fast", "beta_fast"), ("yarn_beta_slow", "beta_slow") })
            {
                if (file.Metadata.ContainsKey($"{arch}.rope.scaling.{key}"))
                {
                    yarn[name] = file.Get($"{arch}.rope.scaling.{key}", 0.0);
                }
            }

            config["rope_scaling"] = yarn;
        }
        else if (!own && scaling is not ("none" or "linear"))
        {
            notes.Add($"the file's RoPE scaling '{scaling}' is not read; the model is used within its stored context.");
        }

        return config;
    }

    // The tokenizer.json of the file's tokenizer (GgufTokenizers, by tokenizer.ggml.model), then tokenizer_config.json and
    // generation_config.json.
    internal static void WriteTokenizer(GgufFile file, string folder, string[] tokens, List<string> notes)
    {
        string model = file.Get("tokenizer.ggml.model", "");
        var definition = GgufTokenizers.Get(model).Create(new GgufTokenizerRequest(file, tokens, notes));
        TokenizerJsonReader.Write(definition, Path.Combine(folder, "tokenizer.json"));

        string? Token(string key) => file.Metadata.TryGetValue(key, out var v) && Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture) is var id
                                     && id >= 0 && id < tokens.Length ? tokens[id] : null;
        string? template = file.Get<string?>("tokenizer.chat_template", null);
        if (template is null)
        {
            // A registered fallback for the file's tokens (GgufChatTemplates: the library's ChatML, a family's from its
            // plug-in); with none, no template, as a model folder without one.
            var fallback = GgufChatTemplates.For(file, tokens);
            template = fallback?.Template.Template;
            notes.Add(fallback is { Name: var used }
                ? $"the file has no chat template; the fallback '{used}' registered for its tokens (GgufChatTemplates) was used."
                : $"the file has no chat template, and no fallback registered in GgufChatTemplates ({string.Join(", ", GgufChatTemplates.Names)}) is for its tokens: "
                  + "chat is not available (plain text generation and fine-tuning on text work); register one with GgufChatTemplates.Register, "
                  + "or reference Idrak.Text.Plugins (the families' templates) and call TextPlugins.RegisterAll().");
        }

        var tokenizerConfig = new JsonObject { ["bos_token"] = Token("tokenizer.ggml.bos_token_id"), ["eos_token"] = Token("tokenizer.ggml.eos_token_id") };
        if (template is not null)
        {
            tokenizerConfig["chat_template"] = template;
        }

        tokenizerConfig["add_bos_token"] = file.Get("tokenizer.ggml.add_bos_token", false);
        WriteJson(Path.Combine(folder, "tokenizer_config.json"), tokenizerConfig);

        var stops = new JsonArray();
        foreach (var key in new[] { "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id" })
        {
            if (file.Metadata.ContainsKey(key) && !stops.Any(s => (long)s! == file.Get(key, -1L)))
            {
                stops.Add((JsonNode)JsonValue.Create(file.Get(key, -1L)));
            }
        }

        WriteJson(Path.Combine(folder, "generation_config.json"), new JsonObject { ["eos_token_id"] = stops });
    }

    // The library's fallback chat template (GgufChatTemplates, "chatml"): ChatML turns.
    internal const string ChatMLName = "chatml";

    internal const string ChatML = "{%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{%- endfor %}"
                                   + "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}";

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    // The UTF-8 ToJsonString(JsonOptions) gives, written straight to the file (no string of a multi-MB tokenizer.json).
    private static void WriteJson(string path, JsonNode json)
    {
        using var stream = File.Create(path);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions
        {
            Encoder = JsonOptions.Encoder, Indented = JsonOptions.WriteIndented, IndentCharacter = JsonOptions.IndentCharacter,
            IndentSize = JsonOptions.IndentSize, NewLine = JsonOptions.NewLine, MaxDepth = JsonOptions.MaxDepth == 0 ? 64 : JsonOptions.MaxDepth,
        });
        json.WriteTo(writer, JsonOptions);
    }

    // The GGUF file's tensors under Hugging Face names, in Hugging Face layouts.
    private sealed class GgufTensors : ITensorStore
    {
        // A Hugging Face tensor: the GGUF tensor holding it, which expert of a tensor of all the experts (-1 for the whole
        // tensor), its shape, and what llama.cpp added to its values (subtracted on reading).
        private sealed record Source(string Gguf, int Expert, int[] Shape, float Offset = 0);

        private readonly GgufFile _file;
        private readonly Dictionary<string, Source> _names = new(StringComparer.Ordinal);    // Hugging Face name → GGUF tensor
        private readonly string _arch;
        private readonly bool _interleavedQueryKeys;
        private readonly int _heads, _kvHeads;

        public GgufTensors(GgufFile file)
        {
            _file = file;
            _arch = file.Get("general.architecture", "");
            var architecture = GgufArchitectures.Find(_arch);
            _interleavedQueryKeys = architecture is { InterleavedQueryKeys: true };
            _heads = file.Get($"{_arch}.attention.head_count", 0);
            _kvHeads = file.Get($"{_arch}.attention.head_count_kv", _heads);
            // Experts' tensors are named through the Hugging Face family's own names (Mixtral's and Qwen's differ).
            var family = architecture is null ? null
                : PretrainedArchitectures.Names.Contains(HuggingFaceOf(file, _arch, architecture)) ? PretrainedArchitectures.Get(HuggingFaceOf(file, _arch, architecture)) : null;
            foreach (var (name, info) in file.Tensors)
            {
                if ((architecture is null ? GgufArchitecture.CommonTensorName(name) : architecture.TensorName(name)) is { } hf)
                {
                    _names[hf] = new Source(name, -1, info.Shape);
                }
                else if (family is not null)
                {
                    AddExperts(name, info, family);
                }
            }

            if (architecture is not null)
            {
                foreach (var (hf, source) in _names.ToList())
                {
                    if (architecture.StoredOffset(hf) is var added && added != 0)
                    {
                        _names[hf] = source with { Offset = added };
                    }
                }
            }
        }

        // The experts' tensors under the family's names: the router (ffn_gate_inp), the experts stored together
        // (ffn_gate_exps, ffn_up_exps, ffn_down_exps: [experts, rows, columns], one Hugging Face tensor per expert) or one
        // by one (ffn_gate.j, older Mixtral files), and the shared expert (ffn_*_shexp) and its gate (ffn_gate_inp_shexp,
        // stored as a vector).
        private void AddExperts(string name, GgufTensorInfo info, IPretrainedArchitecture family)
        {
            void Add(string idrak, int expert, int[] shape)
            {
                if (family.TensorName(idrak) is { } hf)
                {
                    _names[hf] = new Source(name, expert, shape);
                }
            }

            string? Projection(string part) => part switch { "gate" => "gate", "up" => "up", "down" => "down", _ => null };
            var shape = info.Shape;
            switch (name.Split('.'))
            {
                case ["blk", var layer, "ffn_gate_inp", var kind]:
                    Add($"layers.{layer}.mlp.router.{kind}", -1, shape);
                    break;
                case ["blk", var layer, "ffn_gate_inp_shexp", var kind]:
                    Add($"layers.{layer}.mlp.shared_gate.{kind}", -1, shape.Length == 1 ? [1, shape[0]] : shape);
                    break;
                case ["blk", var layer, var part, var kind] when part.EndsWith("_exps", StringComparison.Ordinal) && shape.Length == 3
                                                                && Projection(part[4..^5]) is { } projection:
                    for (int j = 0; j < shape[0]; j++)
                    {
                        Add($"layers.{layer}.mlp.experts.{j}.{projection}.{kind}", j, [shape[1], shape[2]]);
                    }

                    break;
                case ["blk", var layer, var part, var kind] when part.EndsWith("_shexp", StringComparison.Ordinal) && Projection(part[4..^6]) is { } projection:
                    Add($"layers.{layer}.mlp.shared.{projection}.{kind}", -1, shape);
                    break;
                case ["blk", var layer, var part, var expert, var kind] when part.StartsWith("ffn_", StringComparison.Ordinal)
                                                                            && int.TryParse(expert, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out _)
                                                                            && Projection(part[4..]) is { } projection:
                    Add($"layers.{layer}.mlp.experts.{expert}.{projection}.{kind}", -1, shape);
                    break;
            }
        }

        public IEnumerable<string> Names => _names.Keys;

        public bool Contains(string name) => _names.ContainsKey(name);

        public int[] ShapeOf(string name) => _names[name].Shape;

        public float[] Read(string name)
        {
            var source = _names[name];
            string gguf = source.Gguf;
            if (source.Expert >= 0)
            {
                // One expert's rows of the tensor of all the experts.
                int rows = source.Shape[0], columns = source.Shape[1];
                var slice = GC.AllocateUninitializedArray<float>(checked(rows * columns));
                _file.ReadRows(gguf, source.Expert * rows, slice, new byte[checked((int)(rows * _file.RowBytes(gguf)))]);
                Subtract(slice, source.Offset);
                return slice;
            }

            var values = _file.Read(gguf);
            if (Permuted(gguf) is { } heads)
            {
                // llama.cpp stores Llama's q / k rows with each head's two rotary halves interleaved; put them back.
                var shape = ShapeOf(name);
                Unpermute(values, heads, shape[0], shape.Length > 1 ? shape[1] : 1);
            }

            Subtract(values, source.Offset);
            return values;
        }

        // What the converter added to every value taken back off (nothing to do for 0): a vector loop, then a scalar tail.
        private static void Subtract(Span<float> values, float offset)
        {
            if (offset == 0)
            {
                return;
            }

            int i = 0;
            if (System.Numerics.Vector.IsHardwareAccelerated && values.Length >= System.Numerics.Vector<float>.Count)
            {
                var by = new System.Numerics.Vector<float>(offset);
                for (; i <= values.Length - System.Numerics.Vector<float>.Count; i += System.Numerics.Vector<float>.Count)
                {
                    (new System.Numerics.Vector<float>(values[i..]) - by).CopyTo(values[i..]);
                }
            }

            for (; i < values.Length; i++)
            {
                values[i] -= offset;
            }
        }

        public float[] ReadTransposed(string name)
        {
            var source = _names[name];
            string gguf = source.Gguf;
            var shape = source.Shape;
            if (shape.Length != 2)
            {
                throw new ArgumentException($"'{name}' is [{string.Join(", ", shape)}], not a matrix.", nameof(name));
            }

            // Chunks of stored rows are dequantized through pooled buffers and scattered into the transposed result (with
            // Llama's q / k rows put back in order on the way), so the stored order never exists as a full array.
            int rows = shape[0], columns = shape[1], start = Math.Max(0, source.Expert) * rows;   // an expert's first row
            int? heads = Permuted(gguf);
            int headDim = heads is { } h ? rows / h : 1, half = headDim / 2;
            var values = GC.AllocateUninitializedArray<float>(checked(rows * columns));
            int chunkRows = Math.Max(1, (1 << 22) / Math.Max(1, columns));
            float[] chunk = System.Buffers.ArrayPool<float>.Shared.Rent(chunkRows * columns);
            byte[] raw = System.Buffers.ArrayPool<byte>.Shared.Rent(checked((int)(chunkRows * _file.RowBytes(gguf))));
            try
            {
                for (int r0 = 0; r0 < rows; r0 += chunkRows)
                {
                    int n = Math.Min(chunkRows, rows - r0), first = r0;
                    _file.ReadRows(gguf, start + r0, chunk.AsSpan(0, n * columns), raw);
                    Idrak.Abstraction.Devices.HostParallel.For(columns, Math.Max(1, (1 << 14) / n), (c0, c1) =>
                    {
                        const int Tile = 64;
                        for (int t0 = 0; t0 < n; t0 += Tile)
                        {
                            int t1 = Math.Min(n, t0 + Tile);
                            for (int c = c0; c < c1; c++)
                            {
                                int column = c * rows;
                                for (int r = t0; r < t1; r++)
                                {
                                    int from = first + r;
                                    int to = heads is null ? from : from - from % headDim + (from % headDim % 2) * half + from % headDim / 2;
                                    values[column + to] = chunk[r * columns + c];
                                }
                            }
                        }
                    });
                }
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(chunk);
                System.Buffers.ArrayPool<byte>.Shared.Return(raw);
            }

            Subtract(values, source.Offset);
            return values;
        }

        // The heads whose q / k rows llama.cpp interleaved in the tensor gguf (families that store them so), else null.
        private int? Permuted(string gguf) =>
            _interleavedQueryKeys && gguf.Contains(".attn_q.", StringComparison.Ordinal) ? _heads
            : _interleavedQueryKeys && gguf.Contains(".attn_k.", StringComparison.Ordinal) ? _kvHeads
            : null;

        public void Dispose() => _file.Dispose();

        // Row h·headDim + 2i + a moves to h·headDim + a·headDim/2 + i, in place (one head's rows are buffered at a time).
        private static void Unpermute(float[] values, int heads, int rows, int columns)
        {
            int headDim = rows / heads, half = headDim / 2;
            var head = new float[headDim * columns];
            for (int h = 0; h < heads; h++)
            {
                var block = values.AsSpan(h * headDim * columns, headDim * columns);
                block.CopyTo(head);
                for (int i = 0; i < half; i++)
                {
                    for (int a = 0; a < 2; a++)
                    {
                        head.AsSpan((i * 2 + a) * columns, columns).CopyTo(block[((a * half + i) * columns)..]);
                    }
                }
            }
        }
    }
}
