using System.Text.Json.Nodes;
using Idrak;
using Idrak.Layers;
using Idrak.LanguageModels;

// Pretrained models: checkpoints written here in the Hugging Face layout, read back through the architecture registry.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Pretrained =
    [
        ("pretrained: safetensors F32/BF16 round trip; sharded index", SafeTensorsRoundTrip),
        ("pretrained: a Hugging Face-layout checkpoint loads through the registry and matches DecoderSpec; int8; custom architectures", PretrainedCheckpoint),
        ("pretrained: byte-level and SentencePiece-style BPE tokenizers (merges, special tokens, byte fallback, round trips)", BpeTokenizers),
    ];

    private static string TempFolder()
    {
        string folder = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void SafeTensorsRoundTrip(Device device)
    {
        _ = device;
        string folder = TempFolder();
        try
        {
            var a = Enumerable.Range(0, 12).Select(i => MathF.Sin(i) * 3).ToArray();
            var b = Enumerable.Range(0, 5).Select(i => (float)i - 2.5f).ToArray();
            SafeTensorsWriter.Write(Path.Combine(folder, "part1.safetensors"), [("a", [3, 4], a)], SafeTensorType.F32, new Dictionary<string, string> { ["format"] = "pt" });
            SafeTensorsWriter.Write(Path.Combine(folder, "part2.safetensors"), [("b", [5], b)], SafeTensorType.BF16);
            File.WriteAllText(Path.Combine(folder, "model.safetensors.index.json"),
                new JsonObject { ["weight_map"] = new JsonObject { ["a"] = "part1.safetensors", ["b"] = "part2.safetensors" } }.ToJsonString());
            using var reader = SafeTensorsReader.Open(folder);
            Check(reader.Tensors["a"].Shape.SequenceEqual([3, 4]) && reader.Tensors["b"].Type == SafeTensorType.BF16, "headers");
            Check(reader.Read("a").SequenceEqual(a), "F32 values are exact");
            AssertClose(b, reader.Read("b"), 0f, "these values are exact in bfloat16");
            Check(reader.Metadata["format"] == "pt", "metadata");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // Writes config.json and weights as Hugging Face checkpoints store them: PyTorch names, Linear weights [out, in].
    private static void WriteCheckpoint(string folder, DecoderSpec spec, RandomWeights weights, string architecture, SafeTensorType type, bool sharded)
    {
        var config = new JsonObject
        {
            ["architectures"] = new JsonArray(architecture),
            ["vocab_size"] = spec.Vocabulary, ["hidden_size"] = spec.Dim, ["num_hidden_layers"] = spec.Layers,
            ["num_attention_heads"] = spec.Heads, ["num_key_value_heads"] = spec.KvHeads, ["head_dim"] = spec.HeadDim,
            ["intermediate_size"] = spec.FfDim, ["max_position_embeddings"] = spec.MaxPositions, ["rms_norm_eps"] = spec.NormEpsilon,
            ["rope_theta"] = spec.Rope!.Theta, ["hidden_act"] = "silu", ["tie_word_embeddings"] = spec.TieEmbeddings,
        };
        File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
        var tensors = new List<(string, int[], float[])>();
        foreach (var (name, values) in weights.Values)
        {
            string stored = PretrainedArchitectures.LlamaTensorName(name)!;
            bool linear = (name.Contains(".attn.") && !name.Contains("_norm") || name.Contains(".mlp.") || name.StartsWith("head")) && name.EndsWith(".weight");
            int[] shape = ShapeOf(spec, name);
            tensors.Add(linear ? (stored, [shape[1], shape[0]], Transpose2D(values, shape[0], shape[1])) : (stored, shape, values));
        }

        tensors.Add(("model.layers.0.self_attn.rotary_emb.inv_freq", [2], [1f, 0.5f]));   // present in some checkpoints; ignored
        if (!sharded)
        {
            SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors, type);
            return;
        }

        var map = new JsonObject();
        for (int part = 0; part < 2; part++)
        {
            string file = $"model-0000{part + 1}-of-00002.safetensors";
            var share = tensors.Where((_, i) => i % 2 == part).ToList();
            SafeTensorsWriter.Write(Path.Combine(folder, file), share, type);
            foreach (var (name, _, _) in share)
            {
                map[name] = file;
            }
        }

        File.WriteAllText(Path.Combine(folder, "model.safetensors.index.json"), new JsonObject { ["weight_map"] = map }.ToJsonString());
    }

    private static int[] ShapeOf(DecoderSpec s, string name)
    {
        string last = name.Split('.')[^2];
        bool bias = name.EndsWith(".bias");
        int[] matrix = last switch
        {
            "embed" => [s.Vocabulary, s.Dim],
            "q" => [s.Dim, s.Heads * s.HeadDim],
            "k" or "v" => [s.Dim, s.KvHeads * s.HeadDim],
            "o" => [s.Heads * s.HeadDim, s.Dim],
            "gate" or "up" => [s.Dim, s.FfDim],
            "down" => [s.FfDim, s.Dim],
            "head" => [s.Dim, s.Vocabulary],
            "q_norm" or "k_norm" => [s.HeadDim],
            _ => [s.Dim],
        };
        return bias ? [matrix[^1]] : matrix;
    }

    private static float[] Transpose2D(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[c * rows + r] = values[r * columns + c];
            }
        }

        return result;
    }

    private static void PretrainedCheckpoint(Device device)
    {
        var spec = SmallSpec with { QkNorm = true };
        var weights = new RandomWeights(70);
        using var reference = spec.Build(weights, new DecoderBuildOptions { Device = device });
        int[] ids = [2, 9, 14, 3, 7];
        using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
        var expected = reference.Predict(input).ToArray();
        foreach (var (type, sharded) in new[] { (SafeTensorType.F32, false), (SafeTensorType.F32, true), (SafeTensorType.BF16, false) })
        {
            string folder = TempFolder();
            try
            {
                WriteCheckpoint(folder, spec, weights, "Qwen3ForCausalLM", type, sharded);
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                Check(model.Spec == spec with { NormEpsilon = spec.NormEpsilon, QkNorm = true }, $"spec read from config.json: {model.Spec}");
                Check(model.Notes.Count == 0, string.Join("; ", model.Notes));
                float tolerance = type == SafeTensorType.F32 ? 1e-4f : 0.05f * expected.Max(MathF.Abs);
                var loaded = model.Network.Predict(input).ToArray();
                AssertClose(expected, loaded, tolerance, $"{type}{(sharded ? " sharded" : "")}");
                if (type == SafeTensorType.BF16)
                {
                    // bfloat16 weights from a bfloat16 checkpoint are the checkpoint's values exactly.
                    using var half = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, BFloat16 = true });
                    Check(half.Network.Descendants().OfType<Linear>().All(l => l.BFloat16 is not null), "bf16 projections");
                    AssertClose(loaded, half.Network.Predict(input).ToArray(), 1e-4f, "bf16 weights from a BF16 checkpoint");
                }

                if (type == SafeTensorType.F32 && !sharded)
                {
                    using var int8 = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Int8 = true, MaxPositions = 16 });
                    Check(int8.Network.Descendants().OfType<Linear>().All(l => l.Int8 is not null) && int8.MaxPositions == 16, "int8, 16 positions");
                    AssertClose(expected, int8.Network.Predict(input).ToArray(), 0.05f * expected.Max(MathF.Abs), "int8 checkpoint");

                    // A family registered by the application: same naming, a different spec (no q/k norm, so those weights go unused).
                    PretrainedArchitectures.Register("TestNoQkNormForCausalLM", PretrainedArchitectures.LlamaStyle((c, s, notes) => s));
                    using var custom = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Architecture = "TestNoQkNormForCausalLM" });
                    Check(!custom.Spec.QkNorm && custom.Notes.Any(n => n.Contains("not used")), "a custom registration reports unused q/k norm weights");
                }
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }

        try
        {
            PretrainedArchitectures.Get("NoSuchModelForCausalLM");
            Check(false, "unknown architectures are rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("Register"), ex.Message);
        }
    }

    private static void BpeTokenizers(Device device)
    {
        _ = device;
        // Byte-level: every byte is a token (as GPT-2 writes bytes), plus merges for "ab", "abc" and "Ġab".
        var byteMap = new List<string>();
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            byteMap.Add(((char)(printable.Contains(b) ? b : 256 + extra++)).ToString());
        }

        var vocab = new JsonObject();
        foreach (var token in byteMap.Concat(["ab", "abc", "Ġa", "Ġab"]))
        {
            vocab[token] = vocab.Count;
        }

        var byteLevel = BpeTokenizer.FromJson(new JsonObject
        {
            ["added_tokens"] = new JsonArray(new JsonObject { ["id"] = 300, ["content"] = "<|end|>", ["special"] = true }),
            ["normalizer"] = null,
            ["pre_tokenizer"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JsonArray(
                    new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["Regex"] = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" }, ["behavior"] = "Isolated", ["invert"] = false },
                    new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["use_regex"] = false }),
            },
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JsonArray("Ġ a", "a b", "ab c", new JsonArray("Ġa", "b")) },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
        });
        var ids = byteLevel.Encode("abc ab<|end|>ab");
        int Id(string token) => byteLevel.IdOf(token)!.Value;
        Check(ids.SequenceEqual([Id("abc"), Id("Ġab"), 300, Id("ab")]), $"merges and special tokens: [{string.Join(", ", ids)}]");
        foreach (var text in new[] { "héllo 🌍 wörld\n\tfn main() { return 42; }", "  two  spaces", "<|end|><|end|>x" })
        {
            Check(byteLevel.Decode(byteLevel.Encode(text)) == text, $"byte-level round trip of '{text}'");
        }

        Check(byteLevel.IsSpecial(300) && byteLevel.VocabularySize == 301, "special tokens and vocabulary size");

        // SentencePiece-style: "▁" marks word starts, unknown characters fall back to byte tokens.
        var spVocab = new JsonObject();
        foreach (var token in new[] { "<unk>", "▁", "h", "e", "l", "o", "▁h", "el", "▁hel", "lo", "▁hello" }.Concat(Enumerable.Range(0, 256).Select(b => $"<0x{b:X2}>")))
        {
            spVocab[token] = spVocab.Count;
        }

        var sentencePiece = BpeTokenizer.FromJson(new JsonObject
        {
            ["normalizer"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["normalizers"] = new JsonArray(
                    new JsonObject { ["type"] = "Prepend", ["prepend"] = "▁" },
                    new JsonObject { ["type"] = "Replace", ["pattern"] = new JsonObject { ["String"] = " " }, ["content"] = "▁" }),
            },
            ["pre_tokenizer"] = null,
            ["model"] = new JsonObject
            {
                ["type"] = "BPE", ["vocab"] = spVocab, ["merges"] = new JsonArray("▁ h", "e l", "▁h el", "l o", "▁hel lo"), ["byte_fallback"] = true, ["unk_token"] = "<unk>",
            },
            ["decoder"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["decoders"] = new JsonArray(
                    new JsonObject { ["type"] = "Replace", ["pattern"] = new JsonObject { ["String"] = "▁" }, ["content"] = " " },
                    new JsonObject { ["type"] = "ByteFallback" },
                    new JsonObject { ["type"] = "Fuse" },
                    new JsonObject { ["type"] = "Strip", ["content"] = " ", ["start"] = 1, ["stop"] = 0 }),
            },
        });
        var sp = sentencePiece.Encode("hello é");
        int Sp(string token) => sentencePiece.IdOf(token)!.Value;
        Check(sp.SequenceEqual([Sp("▁hello"), Sp("▁"), Sp("<0xC3>"), Sp("<0xA9>")]), $"merges and byte fallback: [{string.Join(", ", sp)}]");
        Check(sentencePiece.Decode(sp) == "hello é", $"SentencePiece round trip: '{sentencePiece.Decode(sp)}'");
        Check(sentencePiece.Decode([Sp("<0xC3>"), Sp("<0x41>"), Sp("▁h")]) == "\uFFFD\uFFFD h", "invalid UTF-8 byte runs decode to one U+FFFD per byte, as the tokenizers library does");

        // Metaspace with prepend_scheme "first": "▁" is prepended where the input starts, not after an added token.
        var metaspace = BpeTokenizer.FromJson(new JsonObject
        {
            ["added_tokens"] = new JsonArray(new JsonObject { ["id"] = spVocab.Count, ["content"] = "<s>", ["special"] = true }),
            ["pre_tokenizer"] = new JsonObject { ["type"] = "Metaspace", ["replacement"] = "▁", ["prepend_scheme"] = "first", ["split"] = false },
            ["model"] = new JsonObject
            {
                ["type"] = "BPE", ["vocab"] = spVocab.DeepClone(), ["merges"] = new JsonArray("▁ h", "e l", "▁h el", "l o", "▁hel lo"),
                ["byte_fallback"] = true, ["unk_token"] = "<unk>",
            },
        });
        var first = metaspace.Encode("hello<s>hello");
        Check(first.SequenceEqual([Sp("▁hello"), spVocab.Count, Sp("h"), Sp("el"), Sp("lo")]),
            $"prepend_scheme first: [{string.Join(", ", first.Select(metaspace.TokenOf))}]");
    }
}
