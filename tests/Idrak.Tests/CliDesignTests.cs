// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;

// The Design group of the idrak tool (suggest, explain, viz), run in-process: the networks suggest writes for a
// table, an image folder, text and chat rows are read by Network.FromJson, build and train a step on the device;
// --search keeps its best candidate; explain counts the parameters of the built network.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliDesignGroup =
    [
        ("cli design: suggest on a CSV writes a network that builds and trains (regression and classes)", CliSuggestTable),
        ("cli design: suggest on an image folder writes a CNN that builds and trains", CliSuggestImages),
        ("cli design: suggest on chat and text rows writes models that build and train", CliSuggestText),
        ("cli design: suggest --search 2 keeps the better candidate", CliSuggestSearch),
        ("cli design: explain counts the built network's parameters; viz draws it", CliExplainViz),
        ("cli design: base-model setups, usage errors, help and short forms", CliDesignUsage),
    ];

    // Runs idrak in-process; returns the exit code and the captured output and errors.
    private static (int Code, string Output, string Error) RunIdrak(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = global::Idrak.Cli.CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static JsonObject RunIdrakJson(params string[] args)
    {
        var (code, output, error) = RunIdrak([.. args, "--json"]);
        Check(code == 0, $"idrak {string.Join(' ', args)} exited with {code}: {error}");
        return JsonNode.Parse(output)!.AsObject();
    }

    // Builds network.json on the device and takes one AdamW step on the first batch of the prepared data; returns the loss.
    private static float TrainOneStep(string folder, string dataPath, Device device, string? target = null, string? task = null)
    {
        var description = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "network.json")))!;
        var prep = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "prep.json")))!.AsObject();
        var train = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "train.json")))!.AsObject();
        var profile = DataProfile.Read(dataPath, target, task);
        var data = DataPreparation.Apply(prep, profile);
        using var model = Network.FromJson(description).OnDevice(device).Build();
        using var optimizer = new AdamW(model.Parameters(), learningRate: (float)train["learningRate"]!);
        var loss = DesignSearch.Loss((string)train["loss"]!);
        using var scope = new TensorScope();
        using var batch = new DataLoader(data.Train, Math.Min(8, data.Train.Count), device: device).First();
        optimizer.ZeroGrad();
        var value = loss(model.Forward(batch.Features), batch.Targets);
        value.Backward();
        optimizer.Step();
        float result = value.Item();
        Check(float.IsFinite(result), $"the training step's loss is {result}");
        return result;
    }

    private static void CliSuggestTable(Device device)
    {
        string folder = TempFolder();
        try
        {
            // A regression table: a number, a category, an identifier and an empty column.
            var random = new Random(3);
            var csv = new StringBuilder("id,area,rooms,zone,note,price\n");
            for (int i = 0; i < 240; i++)
            {
                double area = 50 + random.NextDouble() * 200;
                int rooms = random.Next(1, 6);
                string zone = "ABC"[i % 3].ToString();
                double price = area * 1000 + rooms * 4000 + (zone == "C" ? 30_000 : 0) + random.NextDouble() * 3000;
                csv.Append(CultureInfo.InvariantCulture, $"{i},{area:F1},{rooms},{zone},,{price:F0}\n");
            }

            string data = Path.Combine(folder, "houses.csv");
            File.WriteAllText(data, csv.ToString());
            string run = Path.Combine(folder, "run");
            var json = RunIdrakJson("suggest", data, "-t", "price", "-o", run, "-d", device.ToString(), "-e");
            Check((string?)json["task"] == "regression", $"task: {json["task"]}");
            Check(json["files"]!.AsArray().Select(f => (string?)f).SequenceEqual(["network.json", "train.json", "prep.json"]), $"files: {json["files"]}");
            var prep = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "prep.json")))!;
            var dropped = prep["dropped"]!.AsArray().Select(d => (string?)d!["column"]).ToList();
            Check(dropped.Contains("id") && dropped.Contains("note"), $"identifier and empty columns dropped: {string.Join(", ", dropped)}");
            Check(json["reasons"]!.AsArray().Count > 5 && json["reasons"]!.AsArray().Any(r => (string?)r!["area"] == "task"), "--explain reasons");
            TrainOneStep(run, data, device, "price");

            // The same rules give the same files.
            string again = Path.Combine(folder, "again");
            RunIdrakJson("sg", data, "--target", "price", "--out", again, "-d", device.ToString());
            Check(File.ReadAllText(Path.Combine(run, "network.json")) == File.ReadAllText(Path.Combine(again, "network.json")), "suggest is deterministic");

            // Text output: the data, task, network and next lines.
            var (code, text, _) = RunIdrak("suggest", data, "-t", "price", "-o", again, "-d", device.ToString(), "-e");
            Check(code == 0 && text.Contains("Task      regression") && text.Contains("Network   MLP") && text.Contains("Next      idrak train") && text.Contains("Why:"), text);

            // Classes: a word target, unbalanced (oversampled), with --budget small.
            var classes = new StringBuilder("x1,x2,kind\n");
            for (int i = 0; i < 200; i++)
            {
                double a = random.NextDouble(), b = random.NextDouble();
                classes.Append(CultureInfo.InvariantCulture, $"{a:F3},{b:F3},{(i % 8 == 0 ? "rare" : a > 0.5 ? "high" : "low")}\n");
            }

            string kinds = Path.Combine(folder, "kinds.csv");
            File.WriteAllText(kinds, classes.ToString());
            string classRun = Path.Combine(folder, "classes");
            json = RunIdrakJson("suggest", kinds, "-t", "kind", "-o", classRun, "-d", device.ToString(), "--budget", "small");
            Check((string?)json["task"] == "classify", $"task: {json["task"]}");
            var classPrep = JsonNode.Parse(File.ReadAllText(Path.Combine(classRun, "prep.json")))!;
            Check(classPrep["target"]!["classes"]!.AsArray().Count == 3 && (string?)classPrep["balance"] == "oversample", $"classes and balance: {classPrep["target"]}");
            Check(json["warnings"]!.AsArray().Any(w => ((string?)w)!.Contains("unbalanced", StringComparison.Ordinal)), "unbalanced classes are reported");
            Check((long)json["parameters"]! <= (long)json["parameterCap"]!, "the network fits its parameter cap");
            TrainOneStep(classRun, kinds, device, "kind");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliSuggestImages(Device device)
    {
        string folder = TempFolder();
        try
        {
            // Two classes of 16x16 grayscale drawings: squares (as PGM) and crosses (as PNG, to read the PNG decoder too).
            string root = Path.Combine(folder, "shapes");
            var random = new Random(5);
            for (int i = 0; i < 24; i++)
            {
                foreach (string shape in new[] { "square", "cross" })
                {
                    var pixels = new byte[16 * 16];
                    int cx = 5 + random.Next(6), cy = 5 + random.Next(6), r = 3 + random.Next(2);
                    for (int y = 0; y < 16; y++)
                    {
                        for (int x = 0; x < 16; x++)
                        {
                            bool on = shape == "square" ? Math.Max(Math.Abs(x - cx), Math.Abs(y - cy)) == r : (x == cx || y == cy) && Math.Max(Math.Abs(x - cx), Math.Abs(y - cy)) <= r;
                            pixels[y * 16 + x] = (byte)(on ? 220 : random.Next(30));
                        }
                    }

                    Directory.CreateDirectory(Path.Combine(root, shape));
                    if (shape == "square")
                    {
                        File.WriteAllBytes(Path.Combine(root, shape, $"{i}.pgm"), [.. Encoding.ASCII.GetBytes("P5\n16 16\n255\n"), .. pixels]);
                    }
                    else
                    {
                        File.WriteAllBytes(Path.Combine(root, shape, $"{i}.png"), GrayPng(pixels, 16, 16));
                    }
                }
            }

            var decoded = ImageFiles.Decode(Path.Combine(root, "cross", "0.png"));
            Check(decoded is { Channels: 1, Height: 16, Width: 16 }, "the PNG decoder reads the test image");
            Check(ImageFiles.ReadHeader(Path.Combine(root, "square", "0.pgm")) is { Width: 16, Height: 16, Channels: 1 }, "the PGM header");

            string run = Path.Combine(folder, "run");
            var json = RunIdrakJson("suggest", root, "-o", run, "-d", device.ToString());
            Check((string?)json["task"] == "image", $"task: {json["task"]}");
            Check(json["data"]!["classes"]!.AsArray().Count == 2, $"classes: {json["data"]!["classes"]}");
            var network = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "network.json")))!;
            Check((string?)network["input"] == "Image" && network["steps"]!.AsArray().Any(s => (string?)s!["op"] == "conv2d"), "a CNN for images");
            TrainOneStep(run, root, device);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliSuggestText(Device device)
    {
        string folder = TempFolder();
        try
        {
            // Chat rows without a base model: a small character-level language model.
            string chats = Path.Combine(folder, "chats.jsonl");
            string[] topics = ["the weather", "a recipe", "the train times", "a good book", "the garden"];
            File.WriteAllLines(chats, Enumerable.Range(0, 40).Select(i => new JsonObject
            {
                ["messages"] = new JsonArray(
                    new JsonObject { ["role"] = "user", ["content"] = $"Tell me about {topics[i % 5]}, please ({i})." },
                    new JsonObject { ["role"] = "assistant", ["content"] = $"Here is something about {topics[i % 5]}: it is fine today." }),
            }.ToJsonString()));
            string run = Path.Combine(folder, "chat");
            var json = RunIdrakJson("sg", chats, "-o", run, "-d", device.ToString());
            Check((string?)json["task"] == "chat" && (string?)json["data"]!["kind"] == "chat", $"task: {json["task"]}, kind {json["data"]!["kind"]}");
            Check(json["warnings"]!.AsArray().Any(w => ((string?)w)!.Contains("--base", StringComparison.Ordinal)), "the warning names --base");
            var network = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "network.json")))!;
            Check(network["steps"]!.AsArray().Any(s => (string?)s!["op"] == "transformer" && (bool?)s["causal"] == true), "a causal transformer");
            TrainOneStep(run, chats, device);

            // A text column with a label: a text classifier (words as tokens).
            string reviews = Path.Combine(folder, "reviews.jsonl");
            string[] good = ["great", "lovely", "excellent", "fine"], bad = ["awful", "poor", "broken", "terrible"];
            File.WriteAllLines(reviews, Enumerable.Range(0, 80).Select(i => new JsonObject
            {
                ["review"] = $"this product is {(i % 2 == 0 ? good[i % 4] : bad[i % 4])} and the delivery was {(i % 3 == 0 ? "quick" : "slow")} overall",
                ["label"] = i % 2 == 0 ? "pos" : "neg",
            }.ToJsonString()));
            string textRun = Path.Combine(folder, "text");
            json = RunIdrakJson("suggest", reviews, "-t", "label", "--text", "review", "-o", textRun, "-d", device.ToString());
            Check((string?)json["task"] == "sequence", $"task: {json["task"]}");
            var prep = JsonNode.Parse(File.ReadAllText(Path.Combine(textRun, "prep.json")))!;
            Check((string?)prep["tokenizer"]!["type"] == "words" && (string?)prep["tokenizer"]!["vocabulary"]![1] == "<unk>", "a word vocabulary");
            TrainOneStep(textRun, reviews, device, "label");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliSuggestSearch(Device device)
    {
        string folder = TempFolder();
        try
        {
            // Two well separated classes: candidates differ, and the best one is kept.
            var random = new Random(7);
            var csv = new StringBuilder("a,b,c,group\n");
            for (int i = 0; i < 160; i++)
            {
                int g = i % 2;
                csv.Append(CultureInfo.InvariantCulture, $"{g * 2 + random.NextDouble():F3},{random.NextDouble():F3},{g - random.NextDouble():F3},g{g}\n");
            }

            string data = Path.Combine(folder, "groups.csv");
            File.WriteAllText(data, csv.ToString());
            string run = Path.Combine(folder, "run");
            var json = RunIdrakJson("suggest", data, "-t", "group", "-n", "2", "-o", run, "-d", device.ToString());
            var search = json["search"]!.AsArray();
            Check(search.Count == 2 && search.All(r => r!["error"] is null), $"two candidates trained: {search.ToJsonString()}");
            Check((string?)search[0]!["metric"] == "accuracy", "classes are scored by accuracy");
            double s0 = (double)search[0]!["score"]!, s1 = (double)search[1]!["score"]!;
            double l0 = (double)search[0]!["validationLoss"]!, l1 = (double)search[1]!["validationLoss"]!;
            int better = s1 > s0 || s1 == s0 && l1 < l0 ? 1 : 0;
            Check((bool)search[better]!["best"]! && !(bool)search[1 - better]!["best"]!, $"the better candidate is marked best: {search.ToJsonString()}");

            // network.json is the best candidate's network, and train.json has its learning rate.
            var bestVariant = search[better]!["variant"]!;
            Check((int)json["variant"]!["width"]! == (int)bestVariant["width"]! && (float)json["variant"]!["dropout"]! == (float)bestVariant["dropout"]!, "the chosen variant is the best");
            using var built = Network.FromJson(JsonNode.Parse(File.ReadAllText(Path.Combine(run, "network.json")))!).OnDevice(device).Build();
            Check(built.ParameterCount == (long)search[better]!["parameters"]!, $"network.json has the best candidate's {search[better]!["parameters"]} parameters, not {built.ParameterCount}");
            var train = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "train.json")))!;
            Check((float)train["learningRate"]! == (float)bestVariant["learningRate"]!, "train.json has the best learning rate");

            // A regression search scores by error, lowest first.
            var numbers = new StringBuilder("x,y\n");
            for (int i = 0; i < 120; i++)
            {
                double x = random.NextDouble() * 4;
                numbers.Append(CultureInfo.InvariantCulture, $"{x:F3},{Math.Sin(x) * 10 + random.NextDouble():F3}\n");
            }

            string wave = Path.Combine(folder, "wave.csv");
            File.WriteAllText(wave, numbers.ToString());
            json = RunIdrakJson("suggest", wave, "-t", "y", "--search", "2", "-o", Path.Combine(folder, "wave"), "-d", device.ToString());
            search = json["search"]!.AsArray();
            int lower = (double)search[1]!["score"]! < (double)search[0]!["score"]! ? 1 : 0;
            Check((string?)search[0]!["metric"] == "rmse" && (bool)search[lower]!["best"]!, $"the lower error wins: {search.ToJsonString()}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliExplainViz(Device device)
    {
        string folder = TempFolder();
        try
        {
            // Networks of every kind of step the analysis counts.
            var networks = new (string Name, NetworkBuilder Builder)[]
            {
                ("mlp", Architectures.Mlp(7, [32, 16], 3, Activation.ReLU, 0.1f)),
                ("cnn", Architectures.Cnn(3, 16, 16, [8, 16], 3, 4)),
                ("transformer", Architectures.TransformerClassifier(50, 12, 32, 4, 2, 64, 0.1f, 2)),
                ("gpt", Architectures.Gpt(40, 16, 32, 2, 2, 128, 0f)),
                ("lstm", Architectures.Rnn(RecurrentCell.LSTM, 30, 10, 8, 12, 2)),
                ("gru", Network.Sequence(6, 5).GRU(7, returnSequences: true).MultiHeadAttention(1).LayerNorm().LastStep().Linear(3, bias: false)),
                ("pool", Network.Image(1, 12, 12).Conv2d(4, 3, stride: 2).BatchNorm().GELU().GlobalAveragePool2d().Tanh().Linear(2).Softmax()),
            };
            foreach (var (name, builder) in networks)
            {
                string path = Path.Combine(folder, name + ".json");
                File.WriteAllText(path, builder.ToJson().ToJsonString());
                using var built = Network.FromJson(builder.ToJson()).OnDevice(device).Build();
                var json = RunIdrakJson("explain", path, "-d", device.ToString());
                Check((long)json["parameters"]! == built.ParameterCount, $"{name}: explain counts {json["parameters"]} parameters, the built network has {built.ParameterCount}");
                Check(json["output"]!.AsArray().Select(v => (int)v!).SequenceEqual(builder.CurrentShape), $"{name}: output shape {json["output"]!.ToJsonString()}");
                Check((long)json["flopsPerSample"]! > 0 && (long)json["memory"]!["training"]! > (long)json["memory"]!["inference1"]!, $"{name}: FLOPs and memory");
            }

            var batched = RunIdrakJson("x", Path.Combine(folder, "mlp.json"), "--batch", "64");
            Check((int)batched["batch"]! == 64, "--batch sets the batch the memory is computed for");
            var (code, text, _) = RunIdrak("x", Path.Combine(folder, "cnn.json"), "-d", device.ToString());
            Check(code == 0 && text.Contains("conv2d 8, 3x3, padding 1") && text.Contains("Parameters") && text.Contains("Training"), text);

            (code, text, _) = RunIdrak("viz", Path.Combine(folder, "mlp.json"));
            Check(code == 0 && text.Contains("linear 32") && text.Contains("  |"), text);
            (code, text, _) = RunIdrak("viz", Path.Combine(folder, "mlp.json"), "--as", "mermaid");
            Check(code == 0 && text.StartsWith("flowchart TD", StringComparison.Ordinal) && text.Contains("n0 --> n1"), text);
            (code, text, _) = RunIdrak("viz", Path.Combine(folder, "gpt.json"), "--as=svg");
            Check(code == 0 && text.StartsWith("<svg", StringComparison.Ordinal) && text.TrimEnd().EndsWith("</svg>", StringComparison.Ordinal), "an SVG document");
            Check(RunIdrak("viz", Path.Combine(folder, "mlp.json"), "--as", "png").Code == 2, "an unknown format is a usage error");
            Check(RunIdrak("explain", Path.Combine(folder, "missing.json")).Code == 2, "a missing file is a usage error");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliDesignUsage(Device device)
    {
        string folder = TempFolder();
        try
        {
            string chats = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(chats, Enumerable.Range(0, 30).Select(i => new JsonObject
            {
                ["messages"] = new JsonArray(
                    new JsonObject { ["role"] = "user", ["content"] = $"What is {i} plus {i}?" },
                    new JsonObject { ["role"] = "assistant", ["content"] = $"{i} plus {i} is {2 * i}." }),
            }.ToJsonString()));

            // With a base model (a local folder): a LoRA setup in tune.json, tokens counted with its tokenizer.
            string model = TestData(Path.Combine("gguf", "tiny-qwen3-q8-hf"));
            string run = Path.Combine(folder, "tune");
            var json = RunIdrakJson("suggest", chats, "-b", model, "-o", run, "-d", device.ToString());
            Check(json["files"]!.AsArray().Select(f => (string?)f).SequenceEqual(["tune.json"]), $"files: {json["files"]}");
            var tune = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "tune.json")))!;
            Check((int)tune["rank"]! == 8 && (int)tune["alpha"]! == 16 && (string?)tune["loss"] == "sft" && (int)tune["max-length"]! == 256, $"tune.json: {tune.ToJsonString()}");
            Check(!((string?)json["lines"]!["data"])!.Contains("estimated", StringComparison.Ordinal), "tokens counted with the model's tokenizer");

            // --assist: a local chat model explains the choices; the files stay as the rules made them.
            string assisted = Path.Combine(folder, "assisted");
            json = RunIdrakJson("suggest", chats, "-o", assisted, "-d", device.ToString(), "--assist", model);
            Check(json["assist"] is not null, "--assist adds the model's explanation");
            string plain = Path.Combine(folder, "plain");
            RunIdrakJson("suggest", chats, "-o", plain, "-d", device.ToString());
            Check(File.ReadAllText(Path.Combine(assisted, "network.json")) == File.ReadAllText(Path.Combine(plain, "network.json")), "--assist does not change the network");

            // Preference pairs: DPO with a base model, a usage error without one.
            string pairs = Path.Combine(folder, "pairs.jsonl");
            File.WriteAllLines(pairs, Enumerable.Range(0, 20).Select(i => new JsonObject
            {
                ["prompt"] = $"Say {i}.", ["chosen"] = $"{i}.", ["rejected"] = $"No.",
            }.ToJsonString()));
            json = RunIdrakJson("suggest", pairs, "-b", model, "-o", Path.Combine(folder, "dpo"), "-d", device.ToString());
            Check((string?)json["task"] == "preference", $"task: {json["task"]}");
            Check((string?)JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "dpo", "tune.json")))!["loss"] == "dpo", "DPO loss");
            Check(RunIdrak("suggest", pairs, "-o", Path.Combine(folder, "x")).Code == 2, "preference pairs without --base are a usage error");

            // Usage errors exit with 2 and name the fix.
            var (code, _, error) = RunIdrak("suggest", chats, "--task", "magic");
            Check(code == 2 && error.Contains("regression, classify", StringComparison.Ordinal), $"unknown task: {error}");
            Check(RunIdrak("suggest").Code == 2, "missing DATA");
            Check(RunIdrak("suggest", Path.Combine(folder, "nothing.csv")).Code == 2, "missing data file");
            Check(RunIdrak("suggest", chats, "--budget", "huge").Code == 2, "unknown budget");
            Check(RunIdrak("suggest", chats, "--max-params", "lots").Code == 2, "bad --max-params");
            Check(global::Idrak.Cli.Commands.Design.SuggestCommand.ParseCount("50k") == 50_000 && global::Idrak.Cli.Commands.Design.SuggestCommand.ParseCount("2M") == 2_000_000, "counts with suffixes");

            // Help: an example per command, aliases, and no short form used twice or for a common option.
            foreach (string command in new[] { "suggest", "explain", "viz" })
            {
                var (helpCode, help, _) = RunIdrak("help", command);
                Check(helpCode == 0 && help.Contains("Examples:", StringComparison.Ordinal) && help.Contains("Environment:", StringComparison.Ordinal), $"help for {command}");
            }

            Check(RunIdrak("sg", "-h").Output.Contains("idrak suggest", StringComparison.Ordinal) && RunIdrak("x", "--help").Output.Contains("idrak explain", StringComparison.Ordinal), "aliases sg and x");
            foreach (var command in global::Idrak.Cli.Commands.DesignCommands.All)
            {
                var shorts = command.ShortForms.Keys.ToList();
                Check(shorts.Distinct().Count() == shorts.Count && !shorts.Any(global::Idrak.Cli.CommandContext.CommonShortForms.ContainsKey), $"{command.Name}: short forms");
                Check(command.ShortForms.Values.All(l => command.ValueOptions.Contains(l) || command.Flags.Contains(l)), $"{command.Name}: every short form names an option");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A grayscale 8-bit PNG (filter 0 on every row), with real CRCs.
    private static byte[] GrayPng(byte[] pixels, int width, int height)
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

    private static uint Crc32(byte[] data)
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
}
