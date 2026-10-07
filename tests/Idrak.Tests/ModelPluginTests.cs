// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Nlp;

// The language models' plug-in points: checkpoint formats, GGUF types and architectures, model sources and tokenizer
// components, each reached through its registry with the built-ins and with one registered here (removed afterwards).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ModelPluginGroup =
    [
        ("plugins: the built-in checkpoint formats, GGUF types and architectures, model sources and tokenizer components are in their registries", BuiltInModelPlugins),
        ("plugins: a registered checkpoint format (weights in a JSON file) loads a model whose loss matches the safetensors copy", CustomCheckpointFormat),
        ("plugins: a registered GGUF type dequantizes (whole tensors and blocks), and a registered GGUF architecture name loads like llama", CustomGgufTypeAndArchitecture),
        ("plugins: GGUF pre-tokenizers by name: the built-ins (llama3, qwen2, tekken, gpt2 families) are registered; an unregistered name uses Llama 3's rule with a note; a registered name gives its pattern and the original tokens", CustomGgufPreTokenizer),
        ("plugins: a registered model source ('test:' names → a local folder) is asked before the built-ins, without the network", CustomModelSource),
        ("plugins: registered normalizer, pre-tokenizer and decoder types used from tokenizer JSON; a registered built-in name replaces it until removed", CustomTokenizerComponents),
    ];

    private static void BuiltInModelPlugins(Device device)
    {
        _ = device;
        string gguf = TestData("gguf/tiny-qwen3-q8.gguf"), folder = TestData("gguf/tiny-qwen3-q8-hf");
        Check(CheckpointFormats.Names.SequenceEqual(["gguf", "safetensors"]), $"formats: {string.Join(", ", CheckpointFormats.Names)}");
        Check(CheckpointFormats.For(gguf).Name == "gguf" && CheckpointFormats.For(folder).Name == "safetensors"
              && CheckpointFormats.Get("safetensors").CanOpen(folder) && !CheckpointFormats.Get("gguf").CanOpen(folder), "formats claim their paths");
        try
        {
            CheckpointFormats.Get("nope");
            Check(false, "an unknown format should fail");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("CheckpointFormats.Register", StringComparison.Ordinal), ex.Message);
        }

        Check(new[] { 0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 30 }.All(GgufTypes.Ids.Contains), $"types: {string.Join(", ", GgufTypes.Ids)}");
        var q8 = GgufTypes.Get(8);
        Check(q8.Name == "Q8_0" && q8.BlockValues == 32 && q8.BlockBytes == 34 && GgufFile.BlockSize(14) == (256, 210) && GgufFile.TypeName(15) == "Q8_K",
            "Q8_0 and Q6_K blocks; names of types not registered");
        Check(GgufArchitectures.Names.ToHashSet().SetEquals(["llama", "qwen2", "qwen3", "qwen2moe", "qwen3moe"]) && GgufArchitectures.Get("llama") is { InterleavedQueryKeys: true, WithExperts: "MixtralForCausalLM" }
              && GgufArchitectures.Get("qwen3") is { HuggingFace: "Qwen3ForCausalLM", InterleavedQueryKeys: false }, "GGUF architectures");
        Check(ModelSources.Names.SequenceEqual(["folder", "store", "gguf", "huggingface"]), $"sources: {string.Join(", ", ModelSources.Names)}");
        Check(ModelSources.For(folder)?.Name == "folder" && ModelSources.For(gguf)?.Name == "gguf" && ModelSources.For("store:x")?.Name == "store"
              && ModelSources.For("Qwen/Qwen3-0.6B")?.Name == "huggingface" && ModelSources.For("no such thing") is null, "sources claim their names");
        Check(TokenizerComponents.NormalizerTypes.Contains("NFC") && TokenizerComponents.PreTokenizerTypes.Contains("ByteLevel")
              && TokenizerComponents.DecoderTypes.Contains("ByteFallback") && TokenizerComponents.DecoderTypes.Count == 7, "tokenizer component types");
    }

    // Weights as base64 float32 in weights.json, next to the Hugging Face config and tokenizer files.
    private sealed class JsonWeightsFormat : ICheckpointFormat
    {
        public string Name => "test-json";

        public bool CanOpen(string path) => File.Exists(Path.Combine(path, "weights.json"));

        public string Prepare(string path) => path;

        public ITensorStore Open(string folder) => new JsonWeights(JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "weights.json")))!.AsObject());

        public IEnumerable<string> Notes(string folder) => ["weights read from weights.json"];

        public static void Write(string path, SafeTensorsReader reader) => File.WriteAllText(path, new JsonObject([.. reader.Tensors.Keys.Select(name =>
            KeyValuePair.Create(name, (JsonNode?)new JsonObject
            {
                ["shape"] = new JsonArray([.. reader.Tensors[name].Shape.Select(d => (JsonNode?)d)]),
                ["data"] = Convert.ToBase64String(MemoryMarshal.AsBytes(reader.Read(name).AsSpan())),
            }))]).ToJsonString());
    }

    private sealed class JsonWeights(JsonObject tensors) : ITensorStore
    {
        public IEnumerable<string> Names => tensors.Select(t => t.Key);

        public bool Contains(string name) => tensors.ContainsKey(name);

        public int[] ShapeOf(string name) => [.. tensors[name]!["shape"]!.AsArray().Select(d => (int)d!)];

        public float[] Read(string name) => MemoryMarshal.Cast<byte, float>(Convert.FromBase64String((string)tensors[name]!["data"]!)).ToArray();

        public void Dispose()
        {
        }
    }

    private static float ProbeLoss(PretrainedModel model)
    {
        int[] ids = [.. model.Tokenizer!.Encode("the answer is in the thin rain, and then another answer")];
        return FineTuner.Evaluate(model, [new TrainingSequence(ids, [.. ids.Select(_ => true)])]);
    }

    private static void CustomCheckpointFormat(Device device)
    {
        string source = TestData("gguf/tiny-qwen3-q8-hf"), folder = TempFolder();
        try
        {
            foreach (var file in Directory.GetFiles(source, "*.json"))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            using (var reader = SafeTensorsReader.Open(source))
            {
                JsonWeightsFormat.Write(Path.Combine(folder, "weights.json"), reader);
            }

            try
            {
                PretrainedModel.Load(folder, new PretrainedOptions { Device = device }).Dispose();
                Check(false, "without the format, a folder without safetensors should fail");
            }
            catch (FileNotFoundException)
            {
            }

            CheckpointFormats.Register(new JsonWeightsFormat());
            try
            {
                Check(CheckpointFormats.Names.First() == "test-json" && CheckpointFormats.For(folder).Name == "test-json"
                      && CheckpointFormats.For(source).Name == "safetensors", "a new format is asked first, and only claims its folders");
                using var custom = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                using var original = PretrainedModel.Load(source, new PretrainedOptions { Device = device });
                Check(custom.Spec.ToJson().ToJsonString() == original.Spec.ToJson().ToJsonString(), "same spec");
                Check(custom.Notes.Count > 0 && custom.Notes[0] == "weights read from weights.json", $"notes: {string.Join("; ", custom.Notes)}");
                float a = ProbeLoss(custom), b = ProbeLoss(original);
                Check(a == b, $"loss {a} from the JSON weights, {b} from safetensors");
            }
            finally
            {
                Check(CheckpointFormats.Unregister("test-json") && !CheckpointFormats.Unregister("test-json"), "unregistered");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // A GGUF (version 3) file: metadata (long, double, string, bool and arrays of them) and tensors of raw bytes.
    private static void WriteGguf(string path, IEnumerable<KeyValuePair<string, object>> metadata, IReadOnlyList<(string Name, int Type, long[] Dimensions, byte[] Data)> tensors)
    {
        const int Alignment = 32;
        using var stream = new MemoryStream();
        using var w = new BinaryWriter(stream, Encoding.UTF8);
        void Str(string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
        }

        uint TypeOf(object value) => value switch
        {
            long => 11, ulong => 10, double => 12, string => 8, bool => 7, Array => 9,
            _ => throw new NotSupportedException(value.GetType().Name),
        };

        void Value(object value)
        {
            switch (value)
            {
                case long l: w.Write(l); break;
                case ulong u: w.Write(u); break;
                case double d: w.Write(d); break;
                case string s: Str(s); break;
                case bool b: w.Write(b); break;
                case Array a:
                    w.Write(a switch { string[] => 8u, long[] => 11u, double[] => 12u, bool[] => 7u, _ => throw new NotSupportedException(a.GetType().Name) });
                    w.Write((ulong)a.Length);
                    foreach (var item in a)
                    {
                        Value(item!);
                    }

                    break;
            }
        }

        var entries = metadata.ToList();
        w.Write("GGUF"u8);
        w.Write(3u);
        w.Write((ulong)tensors.Count);
        w.Write((ulong)entries.Count);
        foreach (var (key, value) in entries)
        {
            Str(key);
            w.Write(TypeOf(value));
            Value(value);
        }

        long offset = 0;
        foreach (var (name, type, dimensions, data) in tensors)
        {
            Str(name);
            w.Write((uint)dimensions.Length);
            foreach (long d in dimensions)
            {
                w.Write((ulong)d);
            }

            w.Write((uint)type);
            w.Write((ulong)offset);
            offset += (data.Length + Alignment - 1) / Alignment * Alignment;
        }

        foreach (var (_, _, _, data) in tensors)
        {
            w.Write(new byte[(Alignment - stream.Length % Alignment) % Alignment]);
            w.Write(data);
        }

        w.Flush();
        File.WriteAllBytes(path, stream.ToArray());
    }

    // A copy of a GGUF file with another architecture name (general.architecture, and the keys under it).
    private static void RenameGgufArchitecture(string source, string target, string architecture)
    {
        using var file = GgufFile.Open(source);
        string old = file.Get("general.architecture", "");
        var metadata = file.Metadata.Where(p => p.Key != "general.alignment").Select(p => KeyValuePair.Create(
            p.Key.StartsWith(old + ".", StringComparison.Ordinal) ? architecture + p.Key[old.Length..] : p.Key,
            p.Key == "general.architecture" ? architecture : p.Value));
        var tensors = new List<(string, int, long[], byte[])>();
        using var stream = File.OpenRead(source);
        foreach (var info in file.Tensors.Values)
        {
            var (values, bytes) = GgufFile.BlockSize(info.Type);
            var data = new byte[info.Count / values * bytes];
            stream.Position = info.Offset;
            stream.ReadExactly(data);
            tensors.Add((info.Name, info.Type, info.Dimensions, data));
        }

        WriteGguf(target, metadata, tensors);
    }

    private static void CustomGgufTypeAndArchitecture(Device device)
    {
        string folder = TempFolder();
        try
        {
            // Type 200: blocks of 4 values, a float16 scale then 4 signed bytes (Q8_0 with small blocks).
            var expected = new float[] { 1, -2, 3, 0.5f, 10, 20, -30, 40, -1, -1, 0, 127 };
            float[] scales = [0.5f, 10f, 1f];
            var raw = new byte[3 * 6];
            for (int b = 0; b < 3; b++)
            {
                BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(b * 6), (Half)scales[b]);
                for (int j = 0; j < 4; j++)
                {
                    raw[b * 6 + 2 + j] = (byte)(sbyte)(expected[b * 4 + j] / scales[b]);
                }
            }

            string path = Path.Combine(folder, "types.gguf");
            WriteGguf(path, [KeyValuePair.Create<string, object>("general.architecture", "none")],
                [("t", 200, [4, 3], raw), ("f", 0, [12], MemoryMarshal.AsBytes(expected.AsSpan()).ToArray())]);
            using (var file = GgufFile.Open(path))
            {
                try
                {
                    file.Read("t");
                    Check(false, "an unregistered type should fail");
                }
                catch (NotSupportedException ex)
                {
                    Check(ex.Message.Contains("GgufTypes.Register", StringComparison.Ordinal) && ex.Message.Contains("#200", StringComparison.Ordinal), ex.Message);
                }

                GgufTypes.Register(200, GgufType.Blockwise("TEST_Q8_4", 4, 6, (block, values) =>
                {
                    float d = (float)BinaryPrimitives.ReadHalfLittleEndian(block);
                    for (int j = 0; j < 4; j++)
                    {
                        values[j] = (sbyte)block[2 + j] * d;
                    }
                }));
                try
                {
                    Check(GgufFile.TypeName(200) == "TEST_Q8_4" && GgufFile.BlockSize(200) == (4, 6), "registered name and block");
                    AssertClose(expected, file.Read("t"), 0f, "the registered type, whole tensor");
                    AssertClose(file.Read("f"), file.Read("t"), 0f, "the registered type equals the float32 copy");
                    var rows = new float[8];
                    GgufFile.Dequantize(200, raw.AsSpan(6), rows);
                    AssertClose(expected[4..], rows, 0f, "the registered type, blocks 1 and 2");
                }
                finally
                {
                    Check(GgufTypes.Unregister(200), "type unregistered");
                }
            }

            // A family llama.cpp stores like Llama under another name.
            string renamed = Path.Combine(folder, "tiny-testllama.gguf");
            RenameGgufArchitecture(TestData("gguf/tiny-llama.gguf"), renamed, "testllama");
            string cache = Path.Combine(folder, "cache");
            try
            {
                GgufModel.Prepare(renamed, cache);
                Check(false, "an unregistered GGUF architecture should fail");
            }
            catch (NotSupportedException ex)
            {
                Check(ex.Message.Contains("GgufArchitectures.Register", StringComparison.Ordinal), ex.Message);
            }

            GgufArchitectures.Register("testllama", new GgufArchitecture { HuggingFace = "LlamaForCausalLM", InterleavedQueryKeys = true });
            try
            {
                using var custom = PretrainedModel.Load(GgufModel.Prepare(renamed, cache), new PretrainedOptions { Device = device });
                using var original = PretrainedModel.Load(TestData("gguf/tiny-llama-hf"), new PretrainedOptions { Device = device });
                Check(custom.Spec.ToJson().ToJsonString() == original.Spec.ToJson().ToJsonString(), "same spec as the Hugging Face copy");
                float a = ProbeLoss(custom), b = ProbeLoss(original);
                Check(MathF.Abs(a - b) < 1e-4f, $"loss {a} from the renamed GGUF, {b} from the Hugging Face copy");
            }
            finally
            {
                Check(GgufArchitectures.Unregister("testllama"), "architecture unregistered");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private sealed class TestModelSource(string root) : IModelSource
    {
        public string Name => "test";

        public bool CanResolve(string model) => model.StartsWith("test:", StringComparison.Ordinal);

        public string Resolve(string model, ModelSourceOptions options) => Path.Combine(root, model[5..].Replace('/', '-'));
    }

    private static void CustomModelSource(Device device)
    {
        string root = TestData("gguf");
        try
        {
            ModelSource.Resolve("test:tiny/qwen3-q8-hf", download: false);
            Check(false, "without the source, a 'test:' name is looked up as a hub id, and is not in the caches");
        }
        catch (DirectoryNotFoundException)
        {
        }

        ModelSources.Register(new TestModelSource(root));
        try
        {
            Check(ModelSources.Names.First() == "test" && ModelSources.For("test:tiny/qwen3-q8-hf")?.Name == "test", "asked first (before the Hugging Face id rule)");
            string folder = ModelSource.Resolve("test:tiny/qwen3-q8-hf");
            Check(folder == Path.Combine(root, "tiny-qwen3-q8-hf"), folder);
            Check(ModelSource.Resolve(root) == root, "folders still resolve to themselves");
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            Check(model.Spec.QkNorm && model.Tokenizer is not null, "the model loads");
        }
        finally
        {
            Check(ModelSources.Unregister("test") && ModelSources.Names.Count == 4, "source unregistered");
        }
    }

    private sealed class LambdaNormalizer(Func<string, string> normalize) : ITokenizerNormalizer
    {
        public string Normalize(string text) => normalize(text);
    }

    // Splits pieces into single characters (so no merge applies across them), keeping the pieces it is told to.
    private sealed class CharacterSplitter(string keep) : IPreTokenizer
    {
        public IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart) =>
            [.. pieces.SelectMany(p => p == keep ? [p] : p.Select(c => c.ToString()))];
    }

    private sealed class Brackets : ITokenizerDecoder
    {
        public IReadOnlyList<string> Decode(IReadOnlyList<string> tokens) => [.. tokens.Select(t => $"[{t}]")];
    }

    private static void CustomTokenizerComponents(Device device)
    {
        _ = device;
        JsonObject Json(JsonNode? normalizer, JsonNode? preTokenizer, JsonNode? decoder) => new()
        {
            ["added_tokens"] = new JsonArray(),
            ["normalizer"] = normalizer,
            ["pre_tokenizer"] = preTokenizer,
            ["decoder"] = decoder,
            ["model"] = new JsonObject
            {
                ["type"] = "BPE",
                ["vocab"] = new JsonObject { ["a"] = 0, ["b"] = 1, ["c"] = 2, ["-"] = 3, ["A"] = 4, ["B"] = 5, ["ab"] = 6 },
                ["merges"] = new JsonArray("a b"),
            },
        };
        JsonObject Type(string type) => new() { ["type"] = type };

        Check(BpeTokenizer.FromJson(Json(null, null, null)).Encode("ab-c").SequenceEqual([6, 3, 2]), "the plain tokenizer merges a b");
        try
        {
            BpeTokenizer.FromJson(Json(null, Type("TestCharacters"), null));
            Check(false, "an unregistered pre-tokenizer should fail");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("RegisterPreTokenizer", StringComparison.Ordinal), ex.Message);
        }

        TokenizerComponents.RegisterNormalizer("TestNoDashes", n => new LambdaNormalizer(s => s.Replace((string?)n["remove"] ?? "-", "", StringComparison.Ordinal)));
        TokenizerComponents.RegisterPreTokenizer("TestCharacters", p => new CharacterSplitter((string)p["keep"]!));
        TokenizerComponents.RegisterDecoder("TestBrackets", _ => new Brackets());
        try
        {
            Check(TokenizerComponents.NormalizerTypes.Contains("TestNoDashes") && TokenizerComponents.PreTokenizerTypes.Contains("TestCharacters")
                  && TokenizerComponents.DecoderTypes.Contains("TestBrackets"), "registered types listed");
            var normalized = BpeTokenizer.FromJson(Json(new JsonObject { ["type"] = "TestNoDashes", ["remove"] = "-" }, null, null));
            Check(normalized.Encode("a-b-c").SequenceEqual([6, 2]), $"normalizer: [{string.Join(",", normalized.Encode("a-b-c"))}]");

            // In sequences with built-in components: NFC then the registered normalizer; a split, then the registered pre-tokenizer.
            var json = Json(
                new JsonObject { ["type"] = "Sequence", ["normalizers"] = new JsonArray(Type("NFC"), Type("TestNoDashes")) },
                new JsonObject { ["type"] = "Sequence", ["pretokenizers"] = new JsonArray(
                    new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["String"] = "c" }, ["behavior"] = "Isolated" },
                    new JsonObject { ["type"] = "TestCharacters", ["keep"] = "" }) },
                new JsonObject { ["type"] = "Sequence", ["decoders"] = new JsonArray(Type("TestBrackets"), Type("Fuse")) });
            var tokenizer = BpeTokenizer.FromJson(json);
            Check(tokenizer.Encode("ab-c").SequenceEqual([0, 1, 2]), $"pre-tokenizer splits characters: [{string.Join(",", tokenizer.Encode("ab-c"))}]");
            Check(tokenizer.Decode([6, 2, 0]) == "[ab][c][a]" && tokenizer.Decode(new List<int> { 1 }) == "[b]", $"decoder: {tokenizer.Decode([6, 2, 0])}");

            // A registered built-in name replaces the built-in, until it is removed.
            var lower = Json(Type("Lowercase"), null, null);
            Check(BpeTokenizer.FromJson(lower).Encode("AB").SequenceEqual([6]), "built-in Lowercase");
            TokenizerComponents.RegisterNormalizer("Lowercase", _ => new LambdaNormalizer(s => s));
            Check(BpeTokenizer.FromJson(lower).Encode("AB").SequenceEqual([4, 5]), "Lowercase replaced");
            Check(TokenizerComponents.UnregisterNormalizer("Lowercase") && !TokenizerComponents.UnregisterNormalizer("Lowercase")
                  && TokenizerComponents.NormalizerTypes.Contains("Lowercase") && BpeTokenizer.FromJson(lower).Encode("AB").SequenceEqual([6]), "the built-in back");
        }
        finally
        {
            Check(TokenizerComponents.UnregisterNormalizer("TestNoDashes") && TokenizerComponents.UnregisterPreTokenizer("TestCharacters")
                  && TokenizerComponents.UnregisterDecoder("TestBrackets") && !TokenizerComponents.NormalizerTypes.Contains("TestNoDashes"), "components unregistered");
        }
    }

    // A GGUF file's tokenizer.ggml.pre set to another name (the rest copied).
    private static void SetGgufPreTokenizer(string source, string target, string pre)
    {
        using var file = GgufFile.Open(source);
        var metadata = file.Metadata.Where(p => p.Key != "general.alignment")
            .Select(p => KeyValuePair.Create(p.Key, p.Key == "tokenizer.ggml.pre" ? pre : p.Value));
        var tensors = new List<(string, int, long[], byte[])>();
        using var stream = File.OpenRead(source);
        foreach (var info in file.Tensors.Values)
        {
            var (values, bytes) = GgufFile.BlockSize(info.Type);
            var data = new byte[info.Count / values * bytes];
            stream.Position = info.Offset;
            stream.ReadExactly(data);
            tensors.Add((info.Name, info.Type, info.Dimensions, data));
        }

        WriteGguf(target, metadata, tensors);
    }

    private static void CustomGgufPreTokenizer(Device device)
    {
        _ = device;
        foreach (string name in new[] { "llama3", "llama-bpe", "qwen2", "tekken", "gpt2", "default" })
        {
            Check(GgufPreTokenizers.TryGet(name, out _), $"built-in pre-tokenizer {name}");
        }

        bool known = GgufPreTokenizers.TryGet("gpt2", out string? gpt2);
        Check(known && gpt2 is null, "gpt2: GPT-2's own rule (no split pattern)");
        GgufPreTokenizers.TryGet("qwen2", out string? qwen2);
        GgufPreTokenizers.TryGet("llama3", out string? llama3);
        Check(qwen2 is not null && llama3 is not null && qwen2 != llama3, "qwen2 and llama3 split differently");
        try
        {
            GgufPreTokenizers.Register("broken", "(unclosed");
            Check(false, "a malformed pattern should fail when registered");
        }
        catch (ArgumentException)
        {
        }

        string folder = TempFolder();
        try
        {
            static string? SplitPattern(string prepared) =>
                (string?)((JsonNode.Parse(File.ReadAllText(Path.Combine(prepared, "tokenizer.json")))?["pre_tokenizer"]?["pretokenizers"] as JsonArray)?
                    .FirstOrDefault(n => (string?)n?["type"] == "Split")?["pattern"]?["Regex"]);

            string source = TestData("gguf/tiny-qwen3-q8.gguf");
            string original = GgufModel.Prepare(source, Path.Combine(folder, "original"));
            Check(SplitPattern(original) == qwen2, "the qwen2 file splits with qwen2's pattern");
            static IReadOnlyList<string> NotesOf(string prepared)
            {
                using var model = PretrainedModel.Load(prepared, new PretrainedOptions { Device = Device.Cpu });
                return model.Notes;
            }

            Check(!NotesOf(original).Any(n => n.Contains("pre-tokenizer", StringComparison.Ordinal)), "no note for a registered name");

            string renamed = Path.Combine(folder, "testsplit.gguf");
            SetGgufPreTokenizer(source, renamed, "testsplit");
            string unknown = GgufModel.Prepare(renamed, Path.Combine(folder, "unknown"));
            Check(SplitPattern(unknown) == llama3, "an unregistered name splits with Llama 3's pattern");
            Check(NotesOf(unknown).Any(n => n.Contains("'testsplit' is not registered", StringComparison.Ordinal) && n.Contains("GgufPreTokenizers.Register", StringComparison.Ordinal)),
                $"and the model notes it ({string.Join(" | ", NotesOf(unknown))})");

            GgufPreTokenizers.Register("testsplit", qwen2);
            try
            {
                string registered = GgufModel.Prepare(renamed, Path.Combine(folder, "registered"));
                Check(SplitPattern(registered) == qwen2, "a registered name splits with its pattern");
                using var a = PretrainedModel.Load(original, new PretrainedOptions { Device = Device.Cpu });
                using var b = PretrainedModel.Load(registered, new PretrainedOptions { Device = Device.Cpu });
                const string text = "Hello world, 12345 tokens! Qwen's rules: digits one by one.";
                Check(a.Tokenizer!.Encode(text).SequenceEqual(b.Tokenizer!.Encode(text)), "the registered name gives the original tokens");
            }
            finally
            {
                Check(GgufPreTokenizers.Unregister("testsplit"), "pre-tokenizer unregistered");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
