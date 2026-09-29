using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.LanguageModels;

/// <summary>
/// Models from GGUF files (llama.cpp's and Ollama's format). <see cref="Prepare"/> writes a small folder with what the
/// file's metadata describes, in the Hugging Face layout (config.json, tokenizer.json, tokenizer_config.json with the chat
/// template, generation_config.json), and <see cref="PretrainedModel.Load"/> reads the weights from the GGUF file itself,
/// dequantized tensor by tensor, with llama.cpp's names and layouts turned back into the Hugging Face ones.
/// Architectures: llama (Llama, Mistral), qwen2, qwen3. Tokenizers: byte-level BPE (tokenizer.ggml.model "gpt2").
/// </summary>
public static class GgufModel
{
    private const string Marker = "gguf.json";
    private const int FormatVersion = 1;                            // bump when the prepared files change

    private static readonly Dictionary<string, string> Architectures = new(StringComparer.Ordinal)
    {
        ["llama"] = "LlamaForCausalLM",
        ["qwen2"] = "Qwen2ForCausalLM",
        ["qwen3"] = "Qwen3ForCausalLM",
    };

    // tokenizer.ggml.pre → the pre-tokenizer's split pattern (llama.cpp's llama-vocab.cpp).
    private const string Llama3Pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    private const string Qwen2Pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    private const string TekkenPattern = @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    private static string? PreTokenizerPattern(string pre) => pre switch
    {
        "llama3" or "llama-bpe" or "llama-v3" or "smaug-bpe" or "falcon3" or "pixtral" => Llama3Pattern,
        "qwen2" or "qwen35" or "deepseek-r1-qwen" or "megrez" or "hunyuan" => Qwen2Pattern,
        "tekken" => TekkenPattern,
        "gpt2" or "gpt-2" or "default" => null,                    // GPT-2's own pattern (ByteLevel with its regex)
        _ => Llama3Pattern,
    };

    /// <summary>Whether <paramref name="folder"/> was made by <see cref="Prepare"/>.</summary>
    public static bool IsPrepared(string folder) => File.Exists(Path.Combine(folder, Marker));

    /// <summary>The GGUF file a prepared folder reads its weights from.</summary>
    public static string SourceOf(string folder) =>
        (string?)JsonNode.Parse(File.ReadAllText(Path.Combine(folder, Marker)))?["file"] ?? throw new InvalidDataException($"{folder}/{Marker} names no file.");

    /// <summary>
    /// The folder describing <paramref name="gguf"/> in the Hugging Face layout, made once per file (by path, size and
    /// time) under <paramref name="cacheRoot"/>/gguf (default: IDRAK_CACHE or ~/.cache/idrak).
    /// </summary>
    public static string Prepare(string gguf, string? cacheRoot = null)
    {
        string path = Path.GetFullPath(gguf);
        var info = new FileInfo(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"{path} does not exist.", path);
        }

        string fingerprint = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(
            $"{FormatVersion}|{path}|{info.Length}|{info.LastWriteTimeUtc.Ticks}")))[..12];
        string name = Path.GetFileNameWithoutExtension(path);
        name = name.StartsWith("sha256-", StringComparison.Ordinal) ? name[..Math.Min(name.Length, 19)] : name;   // Ollama blobs
        string folder = Path.Combine(cacheRoot ?? Idrak.Datasets.Downloader.DefaultCacheRoot, "gguf", $"{name}-{fingerprint}");
        if (IsPrepared(folder))
        {
            return folder;
        }

        using var file = GgufFile.Open(path);
        var notes = new List<string>();
        string arch = file.Get("general.architecture", "");
        if (!Architectures.TryGetValue(arch, out var hfArchitecture))
        {
            throw new NotSupportedException($"{path}: GGUF architecture '{arch}' is not supported yet (supported: {string.Join(", ", Architectures.Keys)}).");
        }

        string temp = folder + ".tmp";
        if (Directory.Exists(temp))
        {
            Directory.Delete(temp, true);
        }

        Directory.CreateDirectory(temp);
        var tokens = file.Get<string[]>("tokenizer.ggml.tokens", []);
        WriteJson(Path.Combine(temp, "config.json"), Config(file, arch, hfArchitecture, tokens.Length, notes));
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

    private static JsonObject Config(GgufFile file, string arch, string hfArchitecture, int vocabulary, List<string> notes)
    {
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
        if (file.Metadata.ContainsKey("tokenizer.ggml.bos_token_id"))
        {
            config["bos_token_id"] = file.Get("tokenizer.ggml.bos_token_id", 0);
        }

        if (file.Metadata.ContainsKey("tokenizer.ggml.eos_token_id"))
        {
            config["eos_token_id"] = file.Get("tokenizer.ggml.eos_token_id", 0);
        }

        string scaling = file.Get($"{arch}.rope.scaling.type", "none");
        double factor = file.Get($"{arch}.rope.scaling.factor", 1.0);
        if (file.Tensors.ContainsKey("rope_freqs.weight"))
        {
            config["rope_scaling"] = Llama3Scaling(file, config, notes);
        }
        else if (scaling == "linear" && factor > 1)
        {
            config["rope_scaling"] = new JsonObject { ["rope_type"] = "linear", ["factor"] = factor };
        }
        else if (scaling is "yarn" && factor > 1)
        {
            int original = file.Get($"{arch}.rope.scaling.original_context_length", 0);
            if (original > 0)
            {
                config["max_position_embeddings"] = original;
            }

            notes.Add($"the file extends its context with YaRN (factor {factor}); it is used within its original context of {original} positions.");
        }

        return config;
    }

    // Llama 3.1+ stores its RoPE scaling as per-frequency divisors (rope_freqs.weight): recover the parameters and check them.
    private static JsonObject Llama3Scaling(GgufFile file, JsonObject config, List<string> notes)
    {
        var divisors = file.Read("rope_freqs.weight");
        double theta = (double)config["rope_theta"]!;
        int headDim = (int)config["head_dim"]!;
        double factor = divisors.Max();
        foreach (int original in new[] { 8192, 4096, 16384, 32768 })
        {
            double worst = 0;
            for (int i = 0; i < divisors.Length; i++)
            {
                double wavelength = 2 * Math.PI * Math.Pow(theta, 2.0 * i / headDim);
                double low = original / 1.0, high = original / 4.0;
                double expected = wavelength < high ? 1 : wavelength > low ? factor
                    : 1 / ((1 - (original / wavelength - 1) / 3) / factor + (original / wavelength - 1) / 3);
                worst = Math.Max(worst, Math.Abs(expected - divisors[i]) / expected);
            }

            if (worst < 1e-3)
            {
                return new JsonObject { ["rope_type"] = "llama3", ["factor"] = factor, ["low_freq_factor"] = 1.0, ["high_freq_factor"] = 4.0,
                    ["original_max_position_embeddings"] = original };
            }
        }

        notes.Add("rope_freqs.weight does not match Llama 3's scaling with the usual parameters; they were assumed (factor from the file, original 8192).");
        return new JsonObject { ["rope_type"] = "llama3", ["factor"] = factor, ["low_freq_factor"] = 1.0, ["high_freq_factor"] = 4.0,
            ["original_max_position_embeddings"] = 8192 };
    }

    private static void WriteTokenizer(GgufFile file, string folder, string[] tokens, List<string> notes)
    {
        string model = file.Get("tokenizer.ggml.model", "");
        if (model != "gpt2")
        {
            throw new NotSupportedException($"{file.Path}: tokenizer '{model}' is not supported yet (byte-level BPE, 'gpt2', is: Llama 3, Qwen, Mistral Nemo …).");
        }

        var types = file.Get<long[]>("tokenizer.ggml.token_type", []);
        var merges = file.Get<string[]>("tokenizer.ggml.merges", []);
        var vocab = new JsonObject();
        var added = new JsonArray();
        for (int id = 0; id < tokens.Length; id++)
        {
            long type = id < types.Length ? types[id] : 1;
            if (type is 3 or 4)
            {
                added.Add((JsonNode)new JsonObject { ["id"] = id, ["content"] = tokens[id], ["special"] = type == 3 });
            }

            if (!vocab.ContainsKey(tokens[id]))
            {
                vocab[tokens[id]] = id;
            }
        }

        string pre = file.Get("tokenizer.ggml.pre", "default");
        string? pattern = PreTokenizerPattern(pre);
        if (pattern == Llama3Pattern && pre is not ("llama3" or "llama-bpe" or "llama-v3" or "smaug-bpe" or "falcon3" or "pixtral"))
        {
            notes.Add($"pre-tokenizer '{pre}' is not known; Llama 3's splitting rule is used (tokens may differ slightly from the original).");
        }

        var pretokenizers = new JsonArray();
        if (pattern is not null)
        {
            pretokenizers.Add((JsonNode)new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["Regex"] = pattern }, ["behavior"] = "Isolated", ["invert"] = false });
        }

        pretokenizers.Add((JsonNode)new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = false, ["use_regex"] = pattern is null });
        WriteJson(Path.Combine(folder, "tokenizer.json"), new JsonObject
        {
            ["added_tokens"] = added,
            ["normalizer"] = null,
            ["pre_tokenizer"] = new JsonObject { ["type"] = "Sequence", ["pretokenizers"] = pretokenizers },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JsonArray([.. merges.Select(m => (JsonNode?)JsonValue.Create(m))]) },
        });

        string? Token(string key) => file.Metadata.TryGetValue(key, out var v) && Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture) is var id
                                     && id >= 0 && id < tokens.Length ? tokens[id] : null;
        string? template = file.Get<string?>("tokenizer.chat_template", null);
        if (template is null)
        {
            template = tokens.Contains("<|im_start|>") ? ChatMl : tokens.Contains("<|start_header_id|>") ? Llama3Chat : null;
            notes.Add(template is null
                ? "the file has no chat template: chat is not available (plain text generation and fine-tuning on text work)."
                : "the file has no chat template; a standard one for its tokens was used.");
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

    private const string ChatMl = "{%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{%- endfor %}"
                                  + "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}";

    private const string Llama3Chat = "{{- bos_token }}{%- for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{%- endfor %}"
                                      + "{%- if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}";

    private static void WriteJson(string path, JsonNode json) =>
        File.WriteAllText(path, json.ToJsonString(new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));

    // The GGUF file's tensors under Hugging Face names, in Hugging Face layouts.
    private sealed class GgufTensors : ITensorStore
    {
        private readonly GgufFile _file;
        private readonly Dictionary<string, string> _names = new(StringComparer.Ordinal);    // Hugging Face name → GGUF name
        private readonly string _arch;
        private readonly int _heads, _kvHeads;

        public GgufTensors(GgufFile file)
        {
            _file = file;
            _arch = file.Get("general.architecture", "");
            _heads = file.Get($"{_arch}.attention.head_count", 0);
            _kvHeads = file.Get($"{_arch}.attention.head_count_kv", _heads);
            foreach (var name in file.Tensors.Keys)
            {
                if (HfName(name) is { } hf)
                {
                    _names[hf] = name;
                }
            }
        }

        public IEnumerable<string> Names => _names.Keys;

        public bool Contains(string name) => _names.ContainsKey(name);

        public int[] ShapeOf(string name) => _file.Tensors[_names[name]].Shape;

        public float[] Read(string name)
        {
            string gguf = _names[name];
            var values = _file.Read(gguf);
            if (_arch == "llama" && (gguf.Contains(".attn_q.", StringComparison.Ordinal) || gguf.Contains(".attn_k.", StringComparison.Ordinal)))
            {
                // llama.cpp stores Llama's q / k rows with each head's two rotary halves interleaved; put them back.
                int heads = gguf.Contains(".attn_q.", StringComparison.Ordinal) ? _heads : _kvHeads;
                var shape = ShapeOf(name);
                int rows = shape[0], columns = shape.Length > 1 ? shape[1] : 1;
                values = Unpermute(values, heads, rows, columns);
            }

            return values;
        }

        public void Dispose() => _file.Dispose();

        private static float[] Unpermute(float[] values, int heads, int rows, int columns)
        {
            var result = new float[values.Length];
            int headDim = rows / heads, half = headDim / 2;
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < half; i++)
                {
                    for (int a = 0; a < 2; a++)
                    {
                        int from = h * headDim + i * 2 + a, to = h * headDim + a * half + i;
                        Array.Copy(values, (long)from * columns, result, (long)to * columns, columns);
                    }
                }
            }

            return result;
        }

        private static string? HfName(string name)
        {
            switch (name)
            {
                case "token_embd.weight": return "model.embed_tokens.weight";
                case "output_norm.weight": return "model.norm.weight";
                case "output.weight": return "lm_head.weight";
            }

            var parts = name.Split('.');
            if (parts is not ["blk", var layer, var part, var kind])
            {
                return null;
            }

            string? module = part switch
            {
                "attn_norm" => "input_layernorm",
                "ffn_norm" => "post_attention_layernorm",
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_output" => "self_attn.o_proj",
                "attn_q_norm" => "self_attn.q_norm",
                "attn_k_norm" => "self_attn.k_norm",
                "ffn_gate" => "mlp.gate_proj",
                "ffn_up" => "mlp.up_proj",
                "ffn_down" => "mlp.down_proj",
                _ => null,
            };
            return module is null ? null : $"model.layers.{layer}.{module}.{kind}";
        }
    }
}
