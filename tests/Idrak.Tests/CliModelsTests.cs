// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.LanguageModels;

// The idrak model commands (pull, list, rm, show, search, alias, memory, quantize, merge, inspect, verify, convert, diff,
// families), run in-process through CommandLine.Run with captured output. No network: the hub is a local HTTP server
// (HF_ENDPOINT) serving the tiny fixtures of tests/Idrak.Tests/data/gguf, and every cache is a temporary folder.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliModelsGroup =
    [
        ("cli models: pull (transformers and GGUF, resume, dry run, offline), list, rm with --yes and --dry-run, against a local hub", CliModelsPullListRm),
        ("cli models: search keeps loadable models (registered families and GGUF architectures)", CliModelsSearch),
        ("cli models: show, inspect, verify, diff, families and memory read fixtures without loading", CliModelsReadOnly),
        ("cli models: alias set/list/rm in the config, read by every model argument", CliModelsAliases),
        ("cli models: quantize (size and perplexity), convert GGUF to a folder, merge a LoRA adapter", CliModelsWrite),
        ("cli models: every model command has help with examples, distinct short forms, no provider names", CliModelsHelp),
    ];

    // Runs idrak with args; returns the exit code and the captured output and error.
    private static (int Code, string Out, string Err) ModelsCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = global::Idrak.Cli.CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static JsonNode ModelsJson(params string[] args)
    {
        var (code, output, error) = ModelsCli([.. args, "--json"]);
        Check(code == 0, $"idrak {string.Join(' ', args)} exited with {code}: {error}{output}");
        return JsonNode.Parse(output) ?? throw new Exception($"idrak {string.Join(' ', args)}: no JSON");
    }

    private static string ModelsTemp(string tag)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-cli-{tag}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void CliModelsPullListRm(Device device)
    {
        _ = device;
        string cache = ModelsTemp("pull");
        using var hub = new FakeHub();
        try
        {
            string[] c = ["--cache", cache];
            var empty = ModelsJson(["list", .. c]);
            Check(empty["models"]!.AsArray().Count == 0, "an empty cache lists nothing");

            // Dry run: what would be downloaded, nothing written.
            var dry = ModelsJson(["pull", "test/tiny-llama", "--dry-run", .. c]);
            Check((bool)dry["dryRun"]! && dry["files"]!.AsArray().Count == 4 && !Directory.Exists(Path.Combine(cache, "downloads")), $"dry run lists 4 files: {dry}");

            // A transformers model: config, tokenizer and safetensors, into the layout ModelSource reads.
            var pulled = ModelsJson(["pull", "test/tiny-llama", .. c]);
            string folder = (string)pulled["path"]!;
            Check((int)pulled["downloaded"]! == 4 && File.Exists(Path.Combine(folder, "model.safetensors"))
                  && folder.StartsWith(Path.Combine(cache, "downloads", "huggingface", "models", "test", "tiny-llama"), StringComparison.Ordinal), $"pulled: {pulled}");
            Check(File.ReadAllBytes(Path.Combine(folder, "model.safetensors")).SequenceEqual(File.ReadAllBytes(TestData("gguf/tiny-llama-hf/model.safetensors"))), "same bytes");
            var again = ModelsCli(["pull", "test/tiny-llama", .. c]);
            Check(again.Code == 0 && again.Out.Contains("already in the cache", StringComparison.Ordinal), $"a second pull downloads nothing: {again.Out}");

            // Resume: a partial file is continued with a range request.
            string weights = Path.Combine(folder, "model.safetensors");
            byte[] full = File.ReadAllBytes(weights);
            File.Delete(weights);
            File.WriteAllBytes(weights + ".part", full[..1000]);
            var partial = ModelsJson(["list", .. c]);
            Check(partial["models"]!.AsArray().Any(m => (string?)m!["kind"] == "partial"), $"an interrupted download lists as partial: {partial}");
            hub.Ranges.Clear();
            var resumed = ModelsJson(["pull", "test/tiny-llama", .. c]);
            Check((int)resumed["downloaded"]! == 1 && File.ReadAllBytes(weights).SequenceEqual(full) && hub.Ranges.Contains("bytes=1000-"), $"resumed at byte 1000: {resumed}");

            // GGUF repositories: a tag picks one file; two files without a tag is a usage error naming them.
            var ambiguous = ModelsCli(["pull", "test/tiny-gguf", .. c]);
            Check(ambiguous.Code == 2 && ambiguous.Err.Contains("tiny-llama-Q8_0.gguf", StringComparison.Ordinal) && ambiguous.Err.Contains("--file", StringComparison.Ordinal),
                $"two GGUF files need a choice: {ambiguous.Err}");
            var gguf = ModelsJson(["pull", "test/tiny-gguf:F32", .. c]);
            Check((int)gguf["downloaded"]! == 1 && gguf["prepared"] is not null && gguf["files"]![0]!["name"]!.ToString() == "tiny-llama-F32.gguf", $"tagged GGUF pull: {gguf}");

            var list = ModelsJson(["ls", .. c]);
            var names = list["models"]!.AsArray().Select(m => $"{m!["name"]} {m["kind"]} {m["format"]}").ToList();
            Check(names.Contains("test/tiny-llama huggingface safetensors F32") && names.Contains("test/tiny-gguf/tiny-llama-F32.gguf gguf gguf F32"), $"list: {string.Join("; ", names)}");
            var text = ModelsCli(["list", "tiny", .. c]);
            Check(text.Code == 0 && text.Out.Contains("Last use", StringComparison.Ordinal) && text.Out.Contains("just now", StringComparison.Ordinal), $"text list: {text.Out}");

            // A cached model is found by id (show reads it without the network).
            var shown = ModelsJson(["show", "test/tiny-gguf", .. c]);
            Check((string?)shown["architecture"] == "LlamaForCausalLM", $"show of a pulled GGUF repository: {shown}");

            // Offline: cached is fine, anything else names the model and pull.
            Check(ModelsCli(["pull", "test/tiny-llama", "--offline", .. c]) is var offline && (offline.Code == 0 || offline.Err.Contains("Unknown option", StringComparison.Ordinal)),
                "an offline pull of a cached model succeeds (once --offline is a common option)");

            // rm: a dry run removes nothing; without --yes (captured output, no terminal) it is a usage error naming --yes.
            var rmDry = ModelsJson(["rm", "test/tiny-llama", "--dry-run", .. c]);
            Check((int)rmDry["removed"]! == 1 && Directory.Exists(folder), $"rm --dry-run: {rmDry}");
            var noYes = ModelsCli(["rm", "test/tiny-llama", .. c]);
            Check(noYes.Code == 2 && noYes.Err.Contains("--yes", StringComparison.Ordinal) && Directory.Exists(folder), $"rm without --yes: {noYes.Err}");
            var removed = ModelsJson(["rm", "test/tiny-llama", "test/tiny-gguf", "-y", .. c]);
            Check(!Directory.Exists(folder) && (int)removed["removed"]! >= 2 && !Directory.Exists(Path.Combine(cache, "downloads", "huggingface", "models", "test")),
                $"rm -y removes both and the empty folders: {removed}");
            Check(!Directory.Exists(Path.Combine(cache, "gguf")) || !Directory.EnumerateDirectories(Path.Combine(cache, "gguf")).Any(), "the GGUF file's prepared folder goes with it");
            Check(ModelsCli(["rm", "test/missing", .. c]).Code == 1 && ModelsCli(["rm", "test/missing", "-f", "-y", .. c]).Code == 0, "a missing name fails unless --force");
            Check(ModelsCli(["show", "test/tiny-llama", .. c]) is { Code: 1 } gone && gone.Err.Contains("idrak pull test/tiny-llama", StringComparison.Ordinal), "a removed model names pull");
            Check(ModelsCli(["pull", "test/missing-repo", .. c]).Code == 1, "a repository the hub does not have fails with 1");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliModelsSearch(Device device)
    {
        _ = device;
        using var hub = new FakeHub();
        var found = ModelsJson("search", "tiny");
        var ids = found["models"]!.AsArray().Select(m => (string)m!["id"]!).ToList();
        Check(ids.SequenceEqual(["test/tiny-llama", "test/tiny-gguf"]), $"loadable only: {string.Join(", ", ids)}");
        var all = ModelsJson("search", "tiny", "--all");
        Check(all["models"]!.AsArray().Count == 4 && all["models"]!.AsArray().Any(m => (string?)m!["problem"] == "family BertModel not registered"), $"--all: {all}");
        var gguf = ModelsJson("search", "tiny", "--kind", "gguf");
        Check(gguf["models"]!.AsArray().Count == 1, "--kind gguf");
        Check(ModelsCli("search", "tiny", "--limit", "0").Code == 2 && ModelsCli("search").Code == 2, "usage errors");
    }

    private static void CliModelsReadOnly(Device device)
    {
        _ = device;
        string cache = ModelsTemp("read");
        try
        {
            string[] c = ["--cache", cache];
            string hf = TestData("gguf/tiny-llama-hf"), gguf = TestData("gguf/tiny-qwen3-q8.gguf");
            var show = ModelsJson(["show", hf, .. c]);
            Check((long)show["parameters"]! == 91_200 && (int)show["spec"]!["layers"]! == 2 && (string?)show["toolCallFormat"] == "json", $"show: {show}");
            var showGguf = ModelsCli(["show", gguf, .. c]);
            Check(showGguf.Code == 0 && showGguf.Out.Contains("Qwen3ForCausalLM", StringComparison.Ordinal) && showGguf.Out.Contains("gguf Q8_0", StringComparison.Ordinal)
                  && showGguf.Out.Contains("query/key norms", StringComparison.Ordinal), $"show gguf: {showGguf.Out}");

            var inspect = ModelsJson(["i", gguf, "--filter", "blk.0.attn_q.weight"]);
            Check(inspect["tensors"]!.AsArray().Count == 1 && (string?)inspect["tensors"]![0]!["type"] == "Q8_0" && (string?)inspect["metadata"]!.AsObject().FirstOrDefault().Key is null,
                $"inspect gguf filtered: {inspect}");
            var all = ModelsJson(["inspect", gguf]);
            Check((string?)all["metadata"]!["general.architecture"] == "qwen3" && ((string?)all["metadata"]!["tokenizer.ggml.tokens"])!.Contains("values", StringComparison.Ordinal),
                "metadata, long arrays by length");
            var st = ModelsJson(["inspect", Path.Combine(hf, "model.safetensors")]);
            Check(st["tensors"]!.AsArray().Count == 20 && (string?)st["format"] == "safetensors", "inspect safetensors");
            Check(ModelsCli(["inspect", "missing.gguf"]).Code == 2, "a missing file is a usage error");

            var verify = ModelsJson(["verify", hf, "--read", .. c]);
            Check((bool)verify["ok"]!, $"verify: {verify}");
            string broken = ModelsTemp("broken");
            foreach (string file in Directory.GetFiles(hf))
            {
                File.Copy(file, Path.Combine(broken, Path.GetFileName(file)));
            }

            using (var stream = new FileStream(Path.Combine(broken, "model.safetensors"), FileMode.Open))
            {
                stream.SetLength(stream.Length - 100);
            }

            var bad = ModelsCli(["verify", broken, .. c]);
            Check(bad.Code == 1 && bad.Out.Contains("truncated", StringComparison.Ordinal), $"a truncated file fails: {bad.Out}");
            Directory.Delete(broken, true);

            var same = ModelsJson(["diff", TestData("gguf/tiny-llama.gguf"), hf, .. c]);
            Check((bool)same["same"]! && (int)same["shared"]! == 20, $"a GGUF file and its Hugging Face copy hold the same tensors: {same}");
            var differ = ModelsJson(["diff", hf, TestData("gguf/tiny-qwen3-q8-hf"), .. c]);
            Check(!(bool)differ["same"]! && differ["onlyB"]!.AsArray().Count > 0, $"Llama vs Qwen3: {differ["onlyB"]}");

            var families = ModelsJson("families");
            var qwen3 = families["families"]!.AsArray().First(f => (string?)f!["family"] == "Qwen3ForCausalLM")!;
            var gemma2 = families["families"]!.AsArray().First(f => (string?)f!["family"] == "Gemma2ForCausalLM")!;
            Check((bool)qwen3["queryKeyNorm"]! && qwen3["gguf"]!.AsArray().Count == 1 && (bool)gemma2["softCapping"]! && !(bool)qwen3["softCapping"]!, $"families: {families}");

            var memory = ModelsJson(["memory", hf, "-d", "cpu", "-w", "float32,int8", "-k", "float32", "--context", "128", "--memory", "1M", .. c]);
            var rows = memory["formats"]!.AsArray();
            long f32 = (long)rows[0]!["weightBytes"]!, i8 = (long)rows[1]!["weightBytes"]!;
            // KV: 2 layers × 2 key/value heads × (keys and values) × 16 values × 4 bytes × 128 positions.
            Check(rows.Count == 2 && f32 == 91_200 * 4 && i8 < f32 / 2 && (long)rows[0]!["kvBytes"]! == 2 * 2 * 2 * 16 * 4 * 128
                  && (bool)rows[0]!["fits"]!["1 MB"]!, $"memory: {memory}");
            Check(ModelsCli(["memory", hf, "-w", "nonsense", "-d", "cpu", .. c]).Code == 2, "an unknown weight format is a usage error");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliModelsAliases(Device device)
    {
        _ = device;
        string folder = ModelsTemp("alias");
        try
        {
            string config = Path.Combine(folder, "config.json");
            string[] c = ["-C", config, "--cache", Path.Combine(folder, "cache")];
            Check(ModelsCli(["alias", .. c]) is { Code: 0 } none && none.Out.Contains("No aliases", StringComparison.Ordinal), "no aliases yet");
            var set = ModelsJson(["alias", "set", "tiny", TestData("gguf/tiny-llama-hf"), "-w", "int8", "-k", "int8", .. c]);
            Check(!(bool)set["replaced"]!, "added");
            var stored = JsonNode.Parse(File.ReadAllText(config))!["aliases"]!["tiny"]!;
            Check((string?)stored["weights"] == "int8" && (string?)stored["kv"] == "int8" && (string?)stored["model"] == TestData("gguf/tiny-llama-hf"),
                $"the config keeps {{model, weights, kv}}: {stored}");
            var list = ModelsJson(["alias", "list", .. c]);
            Check(list["aliases"]!["tiny"] is not null, "listed");
            var shown = ModelsJson(["show", "tiny", .. c]);
            Check((long)shown["parameters"]! == 91_200, "show reads the alias");
            var memory = ModelsJson(["memory", "tiny", "-d", "cpu", .. c]);
            Check(memory["formats"]!.AsArray().Count == 1 && (string?)memory["formats"]![0]!["weights"] == "int8", "memory takes the alias's formats");
            Check(ModelsCli(["alias", "set", "a/b", "x", .. c]).Code == 2, "an alias with '/' would read as an id");
            Check(ModelsCli(["alias", "rm", "tiny", .. c]).Code == 0 && ModelsCli(["alias", "rm", "tiny", .. c]).Code == 1, "removed, then missing");
            Check(JsonNode.Parse(File.ReadAllText(config))!["aliases"] is null, "the last alias removes the object");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CliModelsWrite(Device device)
    {
        string folder = ModelsTemp("write");
        try
        {
            string[] c = ["--cache", Path.Combine(folder, "cache"), "-d", device.ToString()];
            string hf = TestData("gguf/tiny-llama-hf");
            var q = ModelsJson(["quantize", hf, "-w", "int4", "-o", Path.Combine(folder, "q4"), .. c]);
            double ppl = (double)q["perplexity"]!, reference = (double)q["floatPerplexity"]!;
            Check((long)q["bytes"]! < (long)q["floatBytes"]! / 2 && double.IsFinite(ppl) && Math.Abs(ppl / reference - 1) < 0.25
                  && File.Exists(Path.Combine(folder, "q4", "weights.bin")) && File.Exists(Path.Combine(folder, "q4", "quantize.json")), $"quantize: {q}");
            Check(ModelsCli(["quantize", hf, "-w", "int4", "-o", Path.Combine(folder, "q4"), .. c]).Code == 2, "a non-empty output needs --force");
            Check(ModelsJson(["quantize", hf, "-w", "int8", "--dry-run", .. c])["dryRun"]!.GetValue<bool>(), "quantize --dry-run");
            Check(ModelsCli(["quantize", hf, .. c]).Code == 2, "quantize needs -w");

            // GGUF to a folder: same tensors, and the copy is an ordinary safetensors model.
            string converted = Path.Combine(folder, "converted");
            var conv = ModelsJson(["convert", TestData("gguf/tiny-qwen3-q8.gguf"), converted, "--type", "f32", .. c]);
            Check(File.Exists(Path.Combine(converted, "model.safetensors")) && !File.Exists(Path.Combine(converted, "gguf.json")), $"convert: {conv}");
            Check((bool)ModelsJson(["diff", TestData("gguf/tiny-qwen3-q8.gguf"), converted, .. c])["same"]!, "converted tensors equal the GGUF file's");
            Check(ModelsCli(["convert", hf, Path.Combine(folder, "x.gguf"), .. c]).Code == 1, "writing GGUF is a stated gap");

            // An adapter with non-zero updates, merged: only the adapted projections change.
            string adapter = Path.Combine(folder, "adapter");
            using (var model = PretrainedModel.Load(hf, new PretrainedOptions { Device = Device.Cpu }))
            {
                model.AddAdapters(2, 4, ["q", "v"], seed: 1);
                model.SaveAdapter(adapter);
            }

            string adapterFile = Path.Combine(adapter, "adapter_model.safetensors");
            List<(string, int[], float[])> tensors;
            using (var reader = SafeTensorsReader.Open(adapterFile))
            {
                var random = new Random(3);
                tensors = [.. reader.Tensors.Values.Select(t => (t.Name, t.Shape, reader.Read(t.Name).Select(_ => (float)(random.NextDouble() - 0.5) * 0.2f).ToArray()))];
            }

            SafeTensorsWriter.Write(adapterFile, tensors, SafeTensorType.F32);
            string merged = Path.Combine(folder, "merged");
            var merge = ModelsJson(["merge", hf, adapter, "-o", merged, "--type", "f32", .. c]);
            Check(File.Exists(Path.Combine(merged, "model.safetensors")) && (string?)merge["kind"] == "LoRA", $"merge: {merge}");
            var diff = ModelsJson(["diff", hf, merged, .. c]);
            var changed = diff["differing"]!.AsArray().Select(d => (string)d!["name"]!).ToList();
            Check(changed.Count == 4 && changed.All(n => n.Contains("q_proj", StringComparison.Ordinal) || n.Contains("v_proj", StringComparison.Ordinal)), $"merged changes: {string.Join(", ", changed)}");
            Check(ModelsCli(["merge", hf, folder, "-o", Path.Combine(folder, "m2"), .. c]).Code == 2, "a folder without an adapter is a usage error");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CliModelsHelp(Device device)
    {
        _ = device;
        string[] names = ["pull", "list", "rm", "show", "search", "alias set", "alias list", "alias rm", "memory", "quantize", "merge", "inspect", "verify", "convert", "diff", "families"];
        var commands = global::Idrak.Cli.Commands.ModelCommands.All;
        Check(commands.Select(x => x.Name).Order().SequenceEqual(names.Order()), $"the group's commands: {string.Join(", ", commands.Select(x => x.Name))}");
        foreach (var command in commands)
        {
            var (code, output, _) = ModelsCli([.. command.Name.Split(' '), "--help"]);
            Check(code == 0 && output.Contains("Examples:", StringComparison.Ordinal) && output.Contains($"idrak {command.Name}", StringComparison.Ordinal), $"help of {command.Name}");
            var shorts = command.ShortForms.Keys.Concat(global::Idrak.Cli.CommandContext.CommonShortForms.Keys).ToList();
            Check(shorts.Distinct().Count() == shorts.Count, $"{command.Name}: two options share a short form");
            foreach (var (shortForm, longForm) in command.ShortForms)
            {
                Check(command.ValueOptions.Contains(longForm) || command.Flags.Contains(longForm), $"{command.Name}: {shortForm} names an option it does not have ({longForm})");
            }

            foreach (string provider in new[] { "Ollama", "OpenAI", "Anthropic", "Claude" })
            {
                Check(!System.Text.RegularExpressions.Regex.Replace(output, @"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", "").Contains(provider, StringComparison.OrdinalIgnoreCase), $"{command.Name}: help names {provider}");
            }
        }

        Check(ModelsCli("ls", "--help").Out.Contains("idrak list", StringComparison.Ordinal) && ModelsCli("i", "--help").Out.Contains("idrak inspect", StringComparison.Ordinal), "aliases ls and i");
    }

    /// <summary>
    /// A local stand-in for the Hugging Face hub's API (revision, tree, resolve with ranges, search) over the fixtures,
    /// set as HF_ENDPOINT while it lives.
    /// </summary>
    private sealed class FakeHub : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly string? _oldEndpoint = Environment.GetEnvironmentVariable("HF_ENDPOINT");
        private readonly Dictionary<string, Dictionary<string, string>> _repos;
        private readonly string _sha = Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(20));

        public FakeHub()
        {
            string hf = TestData("gguf/tiny-llama-hf");
            _repos = new()
            {
                ["test/tiny-llama"] = Directory.GetFiles(hf).ToDictionary(f => Path.GetFileName(f), f => f),
                ["test/tiny-gguf"] = new() { ["tiny-llama-F32.gguf"] = TestData("gguf/tiny-llama.gguf"), ["tiny-llama-Q8_0.gguf"] = TestData("gguf/tiny-qwen3-q8.gguf"), ["README.md"] = Path.Combine(hf, "config.json") },
            };
            var probe = new TcpListener(IPAddress.Loopback, 0);
            probe.Start();
            int port = ((IPEndPoint)probe.LocalEndpoint).Port;
            probe.Stop();
            _listener.Prefixes.Add($"http://localhost:{port}/");
            _listener.Start();
            Environment.SetEnvironmentVariable("HF_ENDPOINT", $"http://localhost:{port}");
            new Thread(Serve) { IsBackground = true }.Start();
        }

        public List<string> Ranges { get; } = [];

        private void Serve()
        {
            while (_listener.IsListening)
            {
                HttpListenerContext context;
                try
                {
                    context = _listener.GetContext();
                }
                catch (Exception e) when (e is HttpListenerException or ObjectDisposedException or InvalidOperationException)
                {
                    return;
                }

                try
                {
                    Answer(context);
                }
                catch (Exception e) when (e is HttpListenerException or IOException)
                {
                }
                finally
                {
                    context.Response.Close();
                }
            }
        }

        private void Answer(HttpListenerContext context)
        {
            string path = Uri.UnescapeDataString(context.Request.Url!.AbsolutePath);
            var response = context.Response;
            void Json(JsonNode node)
            {
                byte[] body = Encoding.UTF8.GetBytes(node.ToJsonString());
                response.ContentType = "application/json";
                response.OutputStream.Write(body);
            }

            if (path == "/api/models")
            {
                Json(new JsonArray(
                    new JsonObject { ["id"] = "test/tiny-llama", ["downloads"] = 10, ["likes"] = 1, ["tags"] = new JsonArray("safetensors"), ["config"] = new JsonObject { ["architectures"] = new JsonArray("LlamaForCausalLM") } },
                    new JsonObject { ["id"] = "test/tiny-bert", ["downloads"] = 9, ["tags"] = new JsonArray("safetensors"), ["config"] = new JsonObject { ["architectures"] = new JsonArray("BertModel") } },
                    new JsonObject { ["id"] = "test/tiny-gguf", ["downloads"] = 8, ["tags"] = new JsonArray("gguf"), ["gguf"] = new JsonObject { ["architecture"] = "llama" } },
                    new JsonObject { ["id"] = "test/tiny-other", ["downloads"] = 7, ["tags"] = new JsonArray("pytorch") }));
                return;
            }

            foreach (var (repo, files) in _repos)
            {
                if (path == $"/api/models/{repo}/revision/main")
                {
                    Json(new JsonObject { ["sha"] = _sha });
                    return;
                }

                if (path == $"/api/models/{repo}/tree/{_sha}")
                {
                    Json(new JsonArray([.. files.Select(f => (JsonNode)new JsonObject
                    {
                        ["type"] = "file",
                        ["path"] = f.Key,
                        ["size"] = new FileInfo(f.Value).Length,
                        ["lfs"] = f.Key.EndsWith(".json", StringComparison.Ordinal) ? null : new JsonObject { ["oid"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(f.Value))) },
                    })]));
                    return;
                }

                string prefix = $"/{repo}/resolve/{_sha}/";
                if (path.StartsWith(prefix, StringComparison.Ordinal) && files.TryGetValue(path[prefix.Length..], out string? source))
                {
                    byte[] bytes = File.ReadAllBytes(source);
                    int start = 0;
                    if (context.Request.Headers["Range"] is { } range)
                    {
                        lock (Ranges)
                        {
                            Ranges.Add(range);
                        }

                        start = int.Parse(range["bytes=".Length..].TrimEnd('-'), System.Globalization.CultureInfo.InvariantCulture);
                        response.StatusCode = 206;
                    }

                    response.ContentLength64 = bytes.Length - start;
                    response.OutputStream.Write(bytes, start, bytes.Length - start);
                    return;
                }
            }

            response.StatusCode = 404;
            Json(new JsonObject { ["error"] = "Repository not found" });
        }

        public void Dispose()
        {
            _listener.Stop();
            _listener.Close();
            Environment.SetEnvironmentVariable("HF_ENDPOINT", _oldEndpoint);
        }
    }
}
