// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json.Nodes;
using System.Text;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cluster;
using Idrak.Data;
using Idrak.Datasets;
using Idrak.Imaging;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Reinforcement;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins.DeepSeek;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Harmony;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;
using Idrak.Text;
using Idrak.Training;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Vision;
using Idrak;

// Helpers that groups of both sides use (plan 34, M2): they were written beside a group of the console runner (a CLI,
// cluster or ONNX group) and are used by groups of this library too, so they live here, where both see them.
internal abstract partial class Tests
{

    // From InventoryTests.cs.
    // These checks do not depend on the device: each runs once, on the first device of the run. The group runs first so
    // the registries hold their built-in names only (later tests register plug-ins of their own).
    internal static readonly HashSet<string> InventoryDone = [];

    internal static bool FirstRun(string check)
    {
        lock (InventoryDone)
        {
            return InventoryDone.Add(check);
        }
    }

    // From BenchText.cs.
    // The fixture corpus as tools/tokenizers/make_fixtures.py --bench takes it: the lines that are not blank (blank as
    // Python's str.strip sees it, so the \x1c-\x1f separators count as blank too), repeated whole until they hold
    // `codePoints` code points (Python's len), so both sides encode the same texts. Shared with "tokenizers: unigram: speed".
    internal static List<string> TokenizerBenchTexts(int codePoints)
    {
        static bool Blank(string line) => line.All(c => char.IsWhiteSpace(c) || c is >= '\u001c' and <= '\u001f');
        static int CodePoints(string text) => text.Length - text.Count(char.IsLowSurrogate);
        var lines = TokenizerCorpusLines().Where(line => !Blank(line)).ToArray();
        long perCopy = lines.Sum(CodePoints);
        var texts = new List<string>();
        for (long total = 0; total < codePoints; total += perCopy)
        {
            texts.AddRange(lines);
        }

        return texts;
    }

    // From ServeImageTests.cs.
    // Records the greedy tokens of every generation while it runs (the default sampler wrapped).
    internal static T WithRecordedTokens<T>(List<int> sampled, Func<T> run)
    {
        var library = TokenSamplers.Default(TokenSamplers.DefaultName)!;
        TokenSamplers.Register(TestKinds.Sampler(TokenSamplers.DefaultName, request => new RecordingSampler(library.Create(request), sampled)));
        try
        {
            return run();
        }
        finally
        {
            TokenSamplers.Unregister(TokenSamplers.DefaultName);
        }
    }

    // From TextFamilyPluginTests.cs.
    // P7's plug-ins, in TextPlugins.All's order (re-registered in it, so the tool-call formats keep the order they are asked in).
    internal static readonly IIdrakPlugin[] TextFamilyPlugins = [LlamaPlugin.Plugin, QwenPlugin.Plugin, MistralPlugin.Plugin, HarmonyPlugin.Plugin, DeepSeekPlugin.Plugin];

    // Runs `body` with P7's families taken away, then registers again, in TextPlugins.All's order, those that were.
    internal static void WithoutTextFamilies(Action body)
    {
        var was = TextFamilyPlugins.Where(p => p.IsRegistered).ToList();
        try
        {
            foreach (var plugin in Enumerable.Reverse(TextFamilyPlugins))
            {
                IdrakPlugins.Unregister(plugin);
            }

            Check(TextFamilyPlugins.All(p => !p.IsRegistered && IdrakPlugins.Find(p.Name) is null), "P7's families taken away");
            body();
        }
        finally
        {
            foreach (var plugin in Enumerable.Reverse(TextFamilyPlugins))
            {
                IdrakPlugins.Unregister(plugin);
            }

            foreach (var plugin in TextFamilyPlugins.Where(was.Contains))
            {
                IdrakPlugins.Register(plugin);
            }
        }
    }

    // From CliImageTests.cs.
    internal static string VlmModel => TestData("vlm/tiny-gemma3");

    internal static bool Fails(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or ArgumentException)
        {
            return true;
        }
    }

    // The default sampler, recording the ids of the first row it hands out (the CLI's generations, seen from outside).
    internal sealed class RecordingSampler(ITokenSampler inner, List<int> ids) : ITokenSampler
    {
        public int Rows => inner.Rows;

        public int Vocabulary => inner.Vocabulary;

        public Tensor Ids => inner.Ids;

        public bool Recordable => inner.Recordable;

        public void Sample(Tensor logits) => inner.Sample(logits);

        public void SetHistory(IReadOnlyList<int> tokens) => inner.SetHistory(tokens);

        public void Reset() => inner.Reset();

        public SampledToken[][] Read(int fromStep, int toStep)
        {
            var steps = inner.Read(fromStep, toStep);
            if (fromStep > 0 || toStep > 1)
            {
                ids.AddRange(steps.Select(s => s[0].Id));                // the prompt's own step (0 to 1) is read again later
            }

            return steps;
        }

        public void Dispose() => inner.Dispose();
    }

    // From VisionContractTests.cs.
    internal static string LlavaData(string name) => TestData($"vlm-llava/{name}");

    /// <summary>Registers the Gemma 3 vision family (Idrak.Vision.Plugins' plug-in, with its text decoder), as an app or <c>-P</c> does.</summary>
    internal static void RegisterGemma3Vision() => Gemma3VisionPlugin.Register();

    /// <summary>
    /// Takes the Gemma 3 vision family away and keeps its text decoder (Idrak.Text.Plugins' Gemma family) registered, as for
    /// an app that registers the text plug-ins only; <see cref="RegisterGemma3Vision"/> puts it back.
    /// </summary>
    internal static void UnregisterGemma3Vision()
    {
        Gemma3VisionPlugin.Unregister();
        GemmaPlugin.Register();
    }

    /// <summary>
    /// Removes the Gemma 3 vision family and its attention rule, and checks that the tiny Gemma 3 vision checkpoint then
    /// fails to load with the "not registered" error naming the registry (no fallback to any family).
    /// </summary>
    internal static void Gemma3VisionNotRegistered(Device device)
    {
        UnregisterGemma3Vision();
        ImageAttentionRules.Unregister(Gemma3VisionPlugin.ImageBlocks);
        Check(VisionFamilies.Find(Gemma3VisionFamily.Architecture) is null && VisionFamilies.Default(Gemma3VisionFamily.Architecture) is null
              && ImageAttentionRules.Origin(Gemma3VisionPlugin.ImageBlocks) == Overrides.Library
              && ImageAttentionRules.Names.SequenceEqual([ImageAttentionRules.Causal, ImageAttentionRules.ImageBlocks]),
            "the library registers no vision family; its image attention rules are causal and image-blocks (the plug-in uses the library's)");
        try
        {
            using var _ = PretrainedModel.Load(TestData("vlm/tiny-gemma3-v5"), new PretrainedOptions { Device = device });
            Check(false, "a vision checkpoint whose family is not registered is refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("Vision family 'Gemma3ForConditionalGeneration' is not registered", StringComparison.Ordinal)
                  && e.Message.Contains("VisionFamilies.Register", StringComparison.Ordinal) && e.Message.Contains("Idrak.Vision.Plugins", StringComparison.Ordinal),
                $"the error names the registry and the plug-in package: {e.Message}");
        }

        Check(VisionFamilies.For("Gemma3ForCausalLM", new System.Text.Json.Nodes.JsonObject()) is null, "a text checkpoint needs no vision family");
    }

    // From OnnxTests.cs.
    internal static float[] TransposeRows(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result[j * rows + i] = values[i * columns + j];
            }
        }

        return result;
    }

    internal static Tensor RandomInput(Device device, Random r, params int[] shape) =>
        Tensor.From([.. Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)(r.NextDouble() * 2 - 1))], shape, device);

    internal static Tensor RandomIds(Device device, Random r, int vocabulary, params int[] shape) =>
        Tensor.From([.. Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)r.Next(vocabulary))], shape, device);

    // From CliDesignTests.cs.
    // A grayscale 8-bit PNG (filter 0 on every row), with real CRCs.
    internal static byte[] GrayPng(byte[] pixels, int width, int height)
    {
        var raw = new MemoryStream();
        for (int y = 0; y < height; y++)
        {
            raw.WriteByte(0);
            raw.Write(pixels, y * width, width);
        }

        var compressed = new MemoryStream();
        using (var z = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
        {
            z.Write(raw.ToArray());
        }

        var png = new MemoryStream();
        png.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        void Chunk(string type, byte[] data)
        {
            var header = new byte[4];
            BinaryPrimitives.WriteInt32BigEndian(header, data.Length);
            png.Write(header);
            byte[] body = [.. Encoding.ASCII.GetBytes(type), .. data];
            png.Write(body);
            BinaryPrimitives.WriteUInt32BigEndian(header, Crc32(body));
            png.Write(header);
        }

        var ihdr = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr, width);
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4), height);
        ihdr[8] = 8;                                                                    // bit depth; colour type 0 (gray)
        Chunk("IHDR", ihdr);
        Chunk("IDAT", compressed.ToArray());
        Chunk("IEND", []);
        return png.ToArray();
    }

    internal static uint Crc32(byte[] data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++)
            {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320 : crc >> 1;
            }
        }

        return ~crc;
    }

    // From CliHealthTests.cs.
    // Sets environment variables for the length of an action, then restores them.
    internal static void WithEnvironment(IReadOnlyDictionary<string, string?> values, Action action)
    {
        var old = values.Keys.ToDictionary(k => k, Environment.GetEnvironmentVariable);
        try
        {
            foreach (var (k, v) in values)
            {
                Environment.SetEnvironmentVariable(k, v);
            }

            action();
        }
        finally
        {
            foreach (var (k, v) in old)
            {
                Environment.SetEnvironmentVariable(k, v);
            }
        }
    }

    // From ClusterTrainingTests.cs.
    // `world` ranks of one node, every one carved from `device`.
    internal static ClusterTopology DpTopology(Device device, int world) => DpTopology([.. Enumerable.Repeat(device.ToString(), world)]);

    // Different devices in the node (cpu beside a GPU) are joined by pcie, measured here; ranks on one device by same-device.
    internal static ClusterTopology DpTopology(IReadOnlyList<string> devices) =>
        ClusterTopology.Create([new ClusterNodeSpec("box", null, [.. devices.Select(d => new ClusterDeviceSpec(d, null))])],
            devices.Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1 ? "pcie" : null, null);

    // Equal splits (no speed measured), so a rerun splits alike.
    internal static ClusterOptions DpOptions(WireType wire = WireType.Float32) => new() { RebalanceSteps = 0, Wire = wire, Timeout = TimeSpan.FromSeconds(300) };

    // Runs `work` on every rank of an in-process cluster and returns each rank's result, failing with every rank's error.
    internal static T[] DpRun<T>(IClusterTopology topology, ClusterOptions options, Func<ClusterRank, T> work)
    {
        var results = new T[topology.WorldSize];
        InProcessCluster.Run(topology, options, rank => results[rank.Rank] = work(rank)).ThrowIfFailed();
        return results;
    }

    internal static float Gap(float[] a, float[] b) => a.Zip(b, (x, y) => MathF.Abs(x - y)).DefaultIfEmpty(0f).Max();

    internal static PretrainingOptions DpPretraining(string tokens, int steps, ICollectives? cluster, ClusterOptions? options) => new()
    {
        Stages = [new PretrainingStage("text") { Sources = [new PretrainingSource(PretrainingSources.Tokens, [tokens])], Steps = steps, LearningRate = 0.2f, Schedule = "constant", WarmupFraction = 0 }],
        MaxLength = 32, BatchTokens = 64, GradientAccumulation = 2, Optimizer = "sgd", WeightDecay = 0f, Momentum = 0f, MaxGradientNorm = 0f, CudaGraphs = false,
        KeepCheckpoints = 0, Seed = 3, Cluster = cluster, ClusterOptions = options,
    };

    // A pretraining run's losses and its last checkpoint's weights (concatenated in name order).
    internal static (float[] Losses, float[] Weights) DpPretrain(Device device, PretrainingOptions options, string folder, Action<string>? trace = null, PretrainingCheckpoint? resume = null,
        string? model = null)
    {
        using var loaded = PretrainedModel.Load(model ?? TinyLlama, new PretrainedOptions { Device = device });
        var result = Pretrainer.Train(loaded, options, folder, resume, trace: trace);
        float[] weights = result.LastCheckpoint is { } last && File.Exists(Path.Combine(last, "model.safetensors"))
            ? [.. ReadTensors(Path.Combine(last, "model.safetensors")).OrderBy(t => t.Key, StringComparer.Ordinal).SelectMany(t => t.Value.Values)]
            : [];
        return ([.. result.Losses], weights);
    }

    // From ClusterOverlapTests.cs.
    // Runs `work` on every rank with the network telemetry recorded.
    internal static (T[] Results, MetricsRecorder Metrics) OverlapRun<T>(IClusterTopology topology, ClusterOptions options, Func<ClusterRank, T> work)
    {
        var metrics = new MetricsRecorder(TelemetryLevel.Network);
        T[] results;
        using (Telemetry.Subscribe(metrics))
        {
            results = DpRun(topology, options, work);
        }

        return (results, metrics);
    }

    // From ReinforcementTests.cs.
    internal static (Space Observations, Space Actions) Spaces(IEnvironmentKind kind)
    {
        using var environment = kind.Create(new EnvironmentRequest());
        return (environment.ObservationSpace, environment.ActionSpace);
    }

    internal static string? SameBits(float[] expected, float[] actual, string what)
    {
        if (expected.Length != actual.Length)
        {
            return $"{what}: {actual.Length} values, {expected.Length} expected";
        }

        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                return FormattableString.Invariant($"{what}: element {i} is {actual[i]:R}, {expected[i]:R} expected");
            }
        }

        return null;
    }

    // Trains until `done` says so (the run is then cancelled before its next iteration); the iteration it was reached at, or 0.
    internal static (ReinforcementResult Result, int Reached) TrainUntil<TAction>(IEnvironmentKind kind, IPolicy<float[], TAction> policy, ReinforcementOptions options,
        Func<IterationSummary, bool> done)
    {
        using var stop = new CancellationTokenSource();
        int reached = 0;
        var trainer = new ReinforcementTrainer<float[], TAction>(kind, policy, options)
        {
            OnIteration = s =>
            {
                if (reached == 0 && done(s))
                {
                    reached = s.Iteration;
                    stop.Cancel();
                }
            },
        };
        return (trainer.Train(stop.Token), reached);
    }

    // From CliVisionTuningTests.cs.
    // The fixture's three records as LlamaFactory writes them (train.json): a JSON array of ShareGPT records.
    internal static string WriteShareGpt(string folder, string family, string name, int records = 3)
    {
        var tuning = JsonNode.Parse(File.ReadAllText(TuningData($"{family}/tuning.json")))!;
        var array = new JsonArray();
        foreach (var record in tuning["records"]!.AsArray().Take(records))
        {
            var sharegpt = record!["sharegpt"]?.DeepClone() ?? ShareGptOf(record);
            array.Add(sharegpt);
        }

        string path = Path.Combine(folder, name);
        File.WriteAllText(path, array.ToJsonString());
        return path;
    }

    // A record's messages in ShareGPT (for a model whose fixture keeps none): <image> where each image part is.
    internal static JsonNode ShareGptOf(JsonNode record)
    {
        var conversations = new JsonArray();
        string? system = null;
        foreach (var m in record["messages"]!.AsArray())
        {
            string role = (string)m!["role"]!;
            string text = m["content"] is JsonValue v ? (string)v! : string.Concat(m["content"]!.AsArray().Select(p => (string?)p!["type"] == "image" ? "<image>" : (string?)p!["text"]));
            if (role == "system")
            {
                system = text;
                continue;
            }

            conversations.Add(new JsonObject { ["from"] = role == "user" ? "human" : "gpt", ["value"] = text });
        }

        var json = new JsonObject { ["conversations"] = conversations, ["images"] = record["images"]!.DeepClone() };
        if (system is not null)
        {
            json["system"] = system;
        }

        return json;
    }

    // The images the records name, in a folder and in a zip.
    internal static (string Folder, string Zip) WriteTuningImages(string folder)
    {
        string images = Path.Combine(folder, "images");
        Directory.CreateDirectory(images);
        foreach (string name in new[] { "image.png", "image-palette.png" })
        {
            File.Copy(TestData($"vlm/{name}"), Path.Combine(images, name));
        }

        string zip = Path.Combine(folder, "images.zip");
        ZipFile.CreateFromDirectory(images, zip);
        return (images, zip);
    }

    // From FontTests.cs.
    internal static string FontData(string name) => TestData($"fonts/{name}");

    internal static IFontFace SubsetFace() => FontFormats.Read(FontData("dejavu-sans-subset.ttf")).Single();

    // From GenericVisionComponentTests.cs.
    // A line of dark marks on a light page: a full stroke for a, the upper half for b, the lower half for c, a gap between.
    internal static ImageData LineImage(string text, int height = 12)
    {
        int width = 2 + text.Length * 5;
        var pixels = new float[width * height];
        Array.Fill(pixels, 0.95f);
        for (int i = 0; i < text.Length; i++)
        {
            int x = 2 + i * 5;
            var (top, bottom) = text[i] switch { 'a' => (1, height - 1), 'b' => (1, height / 2), 'c' => (height / 2, height - 1), _ => (0, 0) };
            for (int y = top; y < bottom; y++)
            {
                pixels[y * width + x] = pixels[y * width + x + 1] = pixels[y * width + x + 2] = 0.1f;
            }
        }

        return new ImageData(pixels, 1, height, width);
    }
}
