// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak;
using Idrak.Cli;
using Idrak.Cli.Commands.Health;
using Idrak.Cli.Shared;

// The idrak tool's Health group and shared foundation, run in-process through CommandLine.Run: doctor, devices,
// version, report, env, init, cache, config, plugins list, formats, completion, update, login, setup android, help
// topics; the common options (@file, --log, --output, --format, --color, --timeout, --threads); and the checks over
// every command of every group (short forms, provider names, the environment variable table).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliHealthGroup =
    [
        ("cli health: no two options of a command share a short form, and names and aliases are unique", CliShortFormsUnique),
        ("cli health: no command, option or help text names a provider", CliNoProviderNames),
        ("cli health: every environment variable read in src is in the tool's table", CliEnvironmentTableComplete),
        ("cli health: doctor checks the runtime, backends, the device and the cache (text and JSON)", CliDoctor),
        ("cli health: doctor --android --fix --yes adds the safe lines once; --network --offline", CliDoctorAndroidFix),
        ("cli health: devices and version list the device and the drivers (text and JSON)", CliDevicesVersion),
        ("cli health: report writes Markdown and JSON, a zip, README rows and benchmarks", CliReport),
        ("cli health: env lists values, defaults and meanings; secrets only as set; help env", CliEnv),
        ("cli health: env set saves variables every run applies (asked or given), the terminal wins; env unset; idrak test passes runner arguments", CliEnvSet),
        ("cli health: cache info and cache clear with --dry-run, --yes and no terminal", CliCache),
        ("cli health: config get/set/unset/list with profiles and hidden tokens; init --yes", CliConfigInit),
        ("cli health: plugins list, formats, completion scripts and candidates", CliPluginsFormatsCompletion),
        ("cli health: update reads the published versions; login and logout store and remove a token", CliUpdateLogin),
        ("cli health: help topics and pages, group words, setup android", CliHelpTopics),
        ("cli health: @file, --log, --output, --format csv/md, --color, --timeout, --threads, progress", CliFoundation),
    ];

    // Runs the tool in-process: the exit code and what it wrote.
    private static (int Exit, string Out, string Err) Cli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int exit = CommandLine.Run(args, output, error);
        return (exit, output.ToString(), error.ToString());
    }

    private static JsonNode CliJson(params string[] args)
    {
        var (exit, output, error) = Cli(args);
        Check(exit is 0 or 1, $"idrak {string.Join(' ', args)} exited with {exit}: {error}");
        return JsonNode.Parse(output) ?? throw new Exception($"idrak {string.Join(' ', args)}: no JSON in '{output}'");
    }

    private static string CliTemp()
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-cli-health-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    // Sets environment variables for the length of an action, then restores them.
    private static void WithEnvironment(IReadOnlyDictionary<string, string?> values, Action action)
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

    private static void CliShortFormsUnique(Device device)
    {
        _ = device;
        var names = new Dictionary<string, string>();
        foreach (var command in CommandTable.All)
        {
            foreach (string name in command.Aliases.Prepend(command.Name))
            {
                Check(names.TryAdd(name, command.Name), $"'{name}' names both '{names.GetValueOrDefault(name)}' and '{command.Name}'.");
            }

            var known = CommandContext.CommonValueOptions.Concat(CommandContext.CommonFlags).Concat(command.ValueOptions).Concat(command.Flags).ToHashSet();
            foreach (var (shortForm, longForm) in command.ShortForms)
            {
                Check(Regex.IsMatch(shortForm, "^-[A-Za-z]$"), $"{command.Name}: short form '{shortForm}' is not one letter.");
                Check(!CommandContext.CommonShortForms.ContainsKey(shortForm), $"{command.Name}: {shortForm} reuses a common short form ({CommandContext.CommonShortForms.GetValueOrDefault(shortForm)}) for {longForm}.");
                Check(known.Contains(longForm), $"{command.Name}: {shortForm} stands for {longForm}, which the command does not take.");
            }

            var duplicates = command.ShortForms.GroupBy(p => p.Value).Where(g => g.Count() > 1).ToList();
            Check(duplicates.Count == 0, $"{command.Name}: {string.Join(", ", duplicates.Select(g => g.Key))} has two short forms.");
            var both = command.ValueOptions.Intersect(command.Flags).Concat(command.ValueOptions.Concat(command.Flags).Intersect(CommandContext.CommonValueOptions.Concat(CommandContext.CommonFlags))).ToList();
            Check(both.Count == 0, $"{command.Name}: {string.Join(", ", both)} declared twice (or as a common option).");
        }

        // Same letter, same meaning across commands.
        var meanings = CommandTable.All.SelectMany(c => c.ShortForms.Select(p => (c.Name, p.Key, p.Value))).GroupBy(x => x.Key);
        foreach (var letter in meanings)
        {
            // -n is "how many" (plans/idrak-cli.md: --search N, --repeat N, --rows N): one meaning, several option names.
            var longs = letter.Select(x => x.Value).Distinct().ToList();
            Check(longs.Count == 1 || letter.Key == "-n", $"{letter.Key} means {string.Join(" and ", longs)} ({string.Join(", ", letter.Select(x => x.Name))}).");
        }
    }

    private static void CliNoProviderNames(Device device)
    {
        _ = device;
        // Split so this file does not name them either; "OpenAI-style" (the wire format) is allowed.
        string[] banned = ["oll" + "ama", "anth" + "ropic", "cla" + "ude", "chat" + "gpt", "gem" + "ini", "open" + "router", "gr" + "oq", "lm" + "studio", "lm " + "studio", "vl" + "lm"];
        var texts = new List<(string Where, string Text)> { ("help", Help.Overview(CommandTable.All)) };
        texts.AddRange(HelpTopics.Names.Select(t => ($"help {t}", HelpTopics.Page(t)!)));
        foreach (var command in CommandTable.All)
        {
            texts.Add((command.Name, string.Join('\n', command.Aliases.Prepend(command.Name).Append(command.Summary).Append(command.Usage)
                .Concat(command.ValueOptions).Concat(command.Flags).Append(Help.For(command)))));
        }

        // A variable's name is what users type to set it, not a name the tool gives anything.
        var variables = EnvironmentVariables.All.Select(v => v.Name).OrderByDescending(n => n.Length).ToList();
        foreach (var (where, raw) in texts)
        {
            string text = variables.Aggregate(raw, (t, v) => t.Replace(v, "", StringComparison.Ordinal));
            string lower = Regex.Replace(text, "openai-style", "", RegexOptions.IgnoreCase).ToLowerInvariant();
            foreach (string name in banned.Append("open" + "ai"))
            {
                Check(!lower.Contains(name, StringComparison.Ordinal), $"{where}: names a provider ('{name}').");
            }
        }
    }

    private static void CliEnvironmentTableComplete(Device device)
    {
        _ = device;
        string root = RepositoryRoot();
        var read = new SortedSet<string>(StringComparer.Ordinal);
        var patterns = new[]
        {
            new Regex("GetEnvironmentVariable\\(\\s*\"([A-Za-z_][A-Za-z0-9_]*)\""),             // Environment.GetEnvironmentVariable("X")
            new Regex("[Ee]nvironment\\w*\\(\\s*\"([A-Z][A-Z0-9_]+)\"\\s*\\)"),                  // environment("X"), EnvironmentInt("X")
            new Regex("(?:Setting|PositiveSetting|BytesSetting)\\(\\s*\"([A-Z][A-Z0-9_]+)\""),   // the Vulkan backend's setting readers
            new Regex("\"(IDRAK_[A-Z0-9_]+)\""),                                                  // any IDRAK_ name written in full
            new Regex("Environment\\[\"([A-Za-z_][A-Za-z0-9_]*)\"\\]"),                           // set for child processes
        };
        foreach (string file in Directory.EnumerateFiles(Path.Combine(root, "src"), "*.cs", SearchOption.AllDirectories)
                     .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}") && !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}")))
        {
            string text = File.ReadAllText(file);
            foreach (var pattern in patterns)
            {
                foreach (Match m in pattern.Matches(text))
                {
                    read.Add(m.Groups[1].Value);
                }
            }
        }

        var table = EnvironmentVariables.All.Select(v => v.Name).ToHashSet();
        var missing = read.Where(n => !table.Contains(n)).ToList();
        Check(missing.Count == 0, $"read in src but missing from src/Idrak.Cli/Shared/EnvironmentVariables.cs: {string.Join(", ", missing)}");
        Check(EnvironmentVariables.All.Select(v => v.Name).Distinct().Count() == EnvironmentVariables.All.Count, "the environment table lists a variable twice");
        Check(read.Count > 50, $"only {read.Count} variables found: the scan is broken");
    }

    private static void CliDoctor(Device device)
    {
        string cache = CliTemp();
        var (exit, output, error) = Cli("doctor", "-d", device.ToString(), "--cache", cache, "-e");
        Check(exit is 0 or 1, $"doctor exited with {exit}: {error}");
        Check(output.Contains(".NET runtime") && output.Contains("why:") && output.Contains(device.ToString()), $"doctor text: {output}");
        var json = CliJson("doc", "-d", device.ToString(), "--cache", cache, "-j");
        var checks = json["checks"]!.AsArray();
        Check(checks.Any(c => (string?)c!["name"] == device.ToString() && (string?)c["status"] == "ok"), $"doctor: {device} not ok: {json}");
        Check(checks.Any(c => (string?)c!["name"] == "cache" && (string?)c["status"] == "ok"), "doctor: the cache check");
        Check(checks.Any(c => (string?)c!["area"] == "backend"), "doctor: no backend checks");
        Check((int)json["failed"]! == checks.Count(c => (string?)c!["status"] == "fail"), "doctor: the failed count");
        Directory.Delete(cache, true);
    }

    private static void CliDoctorAndroidFix(Device device)
    {
        string folder = CliTemp(), rc = Path.Combine(folder, "bashrc");
        WithEnvironment(new Dictionary<string, string?> { ["DOTNET_GCHeapHardLimit"] = null }, () =>
        {
            var json = CliJson("doctor", "-d", device.ToString(), "--android", "--fix", "--yes", "--rc", rc, "-j");
            var heap = json["checks"]!.AsArray().First(c => (string?)c!["name"] == "DOTNET_GCHeapHardLimit")!;
            Check((string?)heap["status"] == "warn" && ((string?)heap["command"])!.Contains(rc), $"doctor --android: {heap}");
            Check(json["repaired"]!.AsArray().Count >= 1, "doctor --fix --yes repaired nothing");
            Cli("doctor", "-d", device.ToString(), "--android", "--fix", "--yes", "--rc", rc);
            int lines = File.ReadAllLines(rc).Count(l => l.Contains("DOTNET_GCHeapHardLimit=0x100000000"));
            Check(lines == 1, $"the heap line is in {rc} {lines} times");
        });
        var network = CliJson("doctor", "-d", device.ToString(), "--network", "--offline", "-j");
        Check(network["checks"]!.AsArray().Any(c => (string?)c!["name"] == "reachability" && ((string)c["detail"]!).Contains("--offline")), "doctor --network --offline");
        Check(network["checks"]!.AsArray().Any(c => (string?)c!["name"] == "tokens"), "doctor --network: tokens");
        Directory.Delete(folder, true);
    }

    private static void CliDevicesVersion(Device device)
    {
        var (exit, output, _) = Cli("devices");
        Check(exit == 0 && output.Contains("cpu") && output.Contains("Lanes"), $"devices: {output}");
        var json = CliJson("dev", "-j", "-d", device.ToString());
        var listed = json["devices"]!.AsArray();
        Check(listed.Count == 1 && (string?)listed[0]!["device"] == device.ToString() && listed[0]!["error"] is null, $"devices -d: {json}");
        Check(listed[0]!["memoryBytes"] is not null && listed[0]!["subgroupSize"] is not null, $"devices: memory and lanes: {json}");
        Check(json["backends"]!.AsArray().Count >= 1, "devices: no backends");
        var all = CliJson("devices", "-j")["devices"]!.AsArray();
        Check(all.Any(d => (string?)d!["device"] == "cpu"), "devices: no cpu");

        var version = CliJson("version", "-j");
        Check(((string?)version["tool"])?.Length > 0 && version["libraries"]!.AsObject().ContainsKey("Idrak") && version["drivers"] is JsonObject, $"version: {version}");
        if (device.IsGpu)
        {
            Check(version["drivers"]!.AsObject().ContainsKey(device.ToString()), $"version: no driver for {device}");
        }

        var (vexit, vtext, _) = Cli("-V");
        Check(vexit == 0 && vtext.StartsWith("idrak ", StringComparison.Ordinal), $"-V: {vtext}");
    }

    private static void CliReport(Device device)
    {
        string folder = CliTemp();
        string md = Path.Combine(folder, "r.md");
        var (exit, output, error) = Cli("report", "-d", device.ToString(), "-o", md, "--bench");
        Check(exit == 0, $"report: {exit} {error}");
        string text = File.ReadAllText(md);
        Check(text.Contains("## Machine") && text.Contains("## Devices") && text.Contains("## Benchmarks") && text.Contains(device.ToString()), $"report.md: {text}");
        var json = JsonNode.Parse(File.ReadAllText(Path.ChangeExtension(md, ".json")))!;
        var bench = json["benchmarks"]!.AsArray();
        Check(bench.Count == 2 && bench.All(b => (string?)b!["device"] == device.ToString() && (double)b["gflops"]! > 0), $"report benchmarks: {bench}");
        Check(output.Contains("Wrote"), "report: no Wrote line");

        string zip = Path.Combine(folder, "phone-report.zip");
        Check(Cli("report", "-d", device.ToString(), "-o", zip).Exit == 0, "report --zip");
        using (var archive = ZipFile.OpenRead(zip))
        {
            Check(archive.GetEntry("report.md") is not null && archive.GetEntry("report.json") is not null, "report zip entries");
        }

        var (rexit, rows, _) = Cli("report", "-d", device.ToString(), "--readme");
        Check(rexit == 0 && rows.Contains("| Date | Library | GPU |") && rows.Contains("<tr><td>") && rows.Contains("not run (idrak report --tests)"), $"report --readme: {rows}");
        Check(CliJson("report", "-d", device.ToString(), "--readme", "-j")["file"] is null, "report --readme alone wrote files");
        var (texit, _, terror) = Cli("report", "--tests", "--source", folder);
        Check(texit == 2 && terror.Contains("not an Idrak source checkout"), $"report --tests --source: {terror}");
        Directory.Delete(folder, true);
    }

    private static void CliEnv(Device device)
    {
        _ = device;
        WithEnvironment(new Dictionary<string, string?> { ["HF_TOKEN"] = "hf_secret_value_123", ["IDRAK_AUTOTUNE"] = "1" }, () =>
        {
            var (exit, output, _) = Cli("env");
            Check(exit == 0 && output.Contains("IDRAK_AUTOTUNE") && output.Contains("HF_TOKEN") && !output.Contains("hf_secret_value_123"), $"env: {output}");
            var json = CliJson("env", "--all", "-j")["variables"]!.AsArray();
            Check(json.Count == EnvironmentVariables.All.Count, "env --all: not every variable");
            var hf = json.First(v => (string?)v!["name"] == "HF_TOKEN")!;
            Check((bool)hf["set"]! && hf["value"] is null && (bool)hf["secret"]!, $"env: HF_TOKEN shown: {hf}");
            var cache = json.First(v => (string?)v!["name"] == "IDRAK_CACHE")!;
            Check(((string?)cache["default"])!.Contains(".cache/idrak") && ((string?)cache["meaning"])!.Length > 0, $"env: IDRAK_CACHE: {cache}");
            Check(!Cli("env", "--all", "-j").Out.Contains("hf_secret_value_123"), "env --json printed a token");
        });
        var (one, oneText, _) = Cli("env", "IDRAK_TRACE");
        Check(one == 0 && oneText.Contains("IDRAK_TRACE") && !oneText.Contains("IDRAK_CACHE"), $"env NAME: {oneText}");
        Check(Cli("env", "NOT_A_VARIABLE").Exit == 2, "env with an unknown name");
        var (h, help, _) = Cli("help", "env");
        Check(h == 0 && help.Contains("VK_ICD_FILENAMES") && help.Contains("default:"), "help env");
        Check(Cli("help", "environment").Out == help, "help environment differs from help env");
        Check(Cli("help", "doctor").Out.Contains("Environment (idrak help env for all):"), "help doctor: no environment section");
    }

    private static void CliEnvSet(Device device)
    {
        _ = device;
        string folder = CliTemp();
        string config = Path.Combine(folder, "config.json");
        WithEnvironment(new Dictionary<string, string?> { ["IDRAK_CUDA_DEBUG"] = null, ["IDRAK_WINDOW_KERNELS"] = null, ["IDRAK_TRACE"] = null, ["IDRAK_PORT"] = null }, () =>
        {
            // Given: saved in the config, applied to each run, undone after it.
            var set = CliJson("env", "set", "IDRAK_CUDA_DEBUG", "1", "-C", config, "-j");
            Check((string?)set["changes"]![0]!["value"] == "1" && (string?)CliConfig.Load(config).Root["env"]!["IDRAK_CUDA_DEBUG"] == "1", $"env set NAME VALUE: {set}");
            var shown = CliJson("env", "IDRAK_CUDA_DEBUG", "-C", config, "-j")["variables"]![0]!;
            Check((string?)shown["value"] == "1" && (string?)shown["source"] == "saved", $"env: saved value applied: {shown}");
            Check(Environment.GetEnvironmentVariable("IDRAK_CUDA_DEBUG") is null, "a saved variable stayed set after the run");

            // The terminal's value wins over the saved one.
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_CUDA_DEBUG"] = "0" }, () =>
            {
                var terminal = CliJson("env", "IDRAK_CUDA_DEBUG", "-C", config, "-j")["variables"]![0]!;
                Check((string?)terminal["value"] == "0" && (string?)terminal["source"] == "terminal", $"env: the terminal's value: {terminal}");
                Check(Environment.GetEnvironmentVariable("IDRAK_CUDA_DEBUG") == "0", "the terminal's value was undone");
            });

            // A profile's saved values apply when it is in use.
            Check(Cli("env", "set", "IDRAK_TRACE", "1", "--profile", "laptop", "-C", config).Exit == 0, "env set --profile");
            Check(CliJson("env", "IDRAK_TRACE", "-C", config, "-j")["variables"]![0]!["value"] is null, "a profile's variable applied without the profile");
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_PROFILE"] = "laptop" }, () =>
                Check((string?)CliJson("env", "IDRAK_TRACE", "-C", config, "-j")["variables"]![0]!["source"] == "saved", "the profile's variable"));

            // Refused: tokens, variables read before the tool starts, unknown names, too many arguments; no terminal to ask on.
            var (secret, _, secretError) = Cli("env", "set", "HF_TOKEN", "hf_x", "-C", config);
            Check(secret == 2 && secretError.Contains("idrak login") && !File.ReadAllText(config).Contains("hf_x"), $"env set a token: {secretError}");
            Check(Cli("env", "set", "IDRAK_CONFIG", "x", "-C", config).Exit == 2 && Cli("env", "set", "NOT_A_VARIABLE", "1", "-C", config).Exit == 2
                  && Cli("env", "set", "IDRAK_TRACE", "1", "2", "-C", config).Exit == 2, "env set refusals");
            var (asked, _, askedError) = Cli("env", "set", "-C", config);
            Check(asked == 2 && askedError.Contains("NAME VALUE"), $"env set without a terminal: {askedError}");

            // Asked: a number takes the suggestion, part of a name finds the variable, a value typed, "-" removes one.
            Terminal.TestInput = new StringReader("4\n\ny\nport\n8080\ny\ncuda_debug\n-\nn\n");
            try
            {
                var (exit, output, error) = Cli("env", "set", "-C", config);
                Check(exit == 0 && output.Contains("Saved IDRAK_WINDOW_KERNELS=0") && output.Contains("Saved IDRAK_PORT=8080") && output.Contains("Removed IDRAK_CUDA_DEBUG"),
                    $"env set asked: {output} {error}");
                Terminal.TestInput = new StringReader("\n");
                Check(Cli("env", "set", "IDRAK_VULKAN_DEFAULT", "-C", config).Exit == 0, "env set NAME asks for the value");
            }
            finally
            {
                Terminal.TestInput = null;
            }

            var saved = CliConfig.Load(config).Root["env"]!.AsObject();
            Check((string?)saved["IDRAK_WINDOW_KERNELS"] == "0" && (string?)saved["IDRAK_PORT"] == "8080" && (string?)saved["IDRAK_VULKAN_DEFAULT"] == "1"
                  && !saved.ContainsKey("IDRAK_CUDA_DEBUG"), $"env set answers: {saved.ToJsonString()}");

            // Unset; the list no longer has it.
            var unset = CliJson("env", "unset", "IDRAK_PORT", "IDRAK_LANG", "-C", config, "-j");
            Check(unset["removed"]!.AsArray().Count == 1 && !CliConfig.Load(config).Root["env"]!.AsObject().ContainsKey("IDRAK_PORT"), $"env unset: {unset}");
            Check(Cli("env", "unset", "-C", config).Exit == 2, "env unset without a name");
            Check(Cli("help", "env", "set").Out.Contains("Enter takes the suggested one"), "help env set");
        });

        // The test runner gets arguments after --; a stray argument is still refused.
        Check(Cli("test", "window").Exit == 2, "test with a stray argument");
        Directory.Delete(folder, true);
    }

    private static void CliCache(Device device)
    {
        _ = device;
        string cache = CliTemp();
        Directory.CreateDirectory(Path.Combine(cache, "tuning", "cuda"));
        File.WriteAllText(Path.Combine(cache, "tuning", "cuda", "a.tsv"), "x\ty");
        Directory.CreateDirectory(Path.Combine(cache, "vulkan"));
        File.WriteAllText(Path.Combine(cache, "vulkan", "tuning.tsv"), "abc");
        string model = Path.Combine(cache, "downloads", "huggingface", "models", "Owner", "Model");
        Directory.CreateDirectory(model);
        File.WriteAllBytes(Path.Combine(model, "w.bin"), new byte[1000]);

        var info = CliJson("cache", "info", "--cache", cache, "-j");
        var parts = info["parts"]!.AsArray();
        Check((long)parts.First(p => (string?)p!["kind"] == "models")!["bytes"]! == 1000, $"cache info models: {info}");
        Check((long)parts.First(p => (string?)p!["kind"] == "tuning")!["bytes"]! == 6, $"cache info tuning: {info}");
        Check((long)parts.First(p => (string?)p!["kind"] == "downloads")!["bytes"]! == 0, "cache info: downloads counted the models");

        var (dry, dryText, _) = Cli("cache", "clear", "tuning", "--cache", cache, "--dry-run");
        Check(dry == 0 && dryText.Contains("Would delete") && File.Exists(Path.Combine(cache, "vulkan", "tuning.tsv")), $"cache clear --dry-run: {dryText}");
        var (no, _, noError) = Cli("cache", "clear", "tuning", "--cache", cache);
        Check(no == 2 && noError.Contains("--yes"), $"cache clear without a terminal: {no} {noError}");
        var (yes, _, _) = Cli("cache", "clear", "tuning", "--cache", cache, "-y");
        Check(yes == 0 && !Directory.Exists(Path.Combine(cache, "tuning")) && !File.Exists(Path.Combine(cache, "vulkan", "tuning.tsv")) && File.Exists(Path.Combine(model, "w.bin")), "cache clear tuning -y");
        Check(Cli("cache", "clear", "bogus", "--cache", cache).Exit == 2, "cache clear bogus");
        Check(Cli("cache", "clear", "all", "--cache", cache, "--yes").Exit == 0 && !Directory.Exists(cache), "cache clear all");
    }

    private static void CliConfigInit(Device device)
    {
        string folder = CliTemp(), config = Path.Combine(folder, "config.json");
        Check(Cli("config", "set", "device", "cpu", "-C", config).Exit == 0, "config set");
        Check(Cli("config", "set", "sizes", "[1, 2]", "-C", config).Exit == 0, "config set JSON");
        Check(Cli("config", "set", "aliases.qwen.model", "Qwen/Qwen3-0.6B", "-C", config).Exit == 0, "config set dotted");
        Check(Cli("config", "set", "device", device.ToString(), "--profile", "phone", "-C", config).Exit == 0, "config set --profile");
        Check(Cli("config", "set", "tokens.hub", "secret-token-42", "-C", config).Exit == 0, "config set token");
        Check(Cli("config", "get", "device", "-C", config).Out.Trim() == "cpu", "config get");
        Check(Cli("config", "get", "device", "--profile", "phone", "-C", config).Out.Trim() == device.ToString(), "config get --profile");
        Check(Cli("config", "get", "aliases.qwen.model", "-C", config).Out.Trim() == "Qwen/Qwen3-0.6B", "config get dotted");
        Check(Cli("config", "get", "nothing", "-C", config).Exit == 1, "config get of an unset key");
        var sizes = CliJson("config", "get", "sizes", "-C", config, "-j");
        Check(sizes["value"] is JsonArray { Count: 2 }, $"config get sizes: {sizes}");
        var (_, list, _) = Cli("config", "list", "-C", config);
        Check(list.Contains("aliases.qwen.model") && list.Contains("tokens") && !list.Contains("secret-token-42"), $"config list: {list}");
        Check(!Cli("config", "list", "-C", config, "-j").Out.Contains("secret-token-42") && !Cli("config", "get", "tokens", "-C", config).Out.Contains("secret-token-42"), "config printed a token");
        WithEnvironment(new Dictionary<string, string?> { ["IDRAK_PROFILE"] = "phone" }, () =>
            Check(CliJson("devices", "-C", config, "-j", "-d", device.ToString())["devices"] is not null && CliConfig.Load(config).Get("device") == device.ToString(), "profile in use"));
        Check(Cli("config", "unset", "device", "-C", config).Exit == 0 && Cli("config", "get", "device", "-C", config).Exit == 1, "config unset");

        string config2 = Path.Combine(folder, "init.json");
        var dry = CliJson("init", "--yes", "--dry-run", "-C", config2, "-j");
        Check(!(bool)dry["written"]! && !File.Exists(config2), "init --dry-run wrote");
        Check(Cli("init", "-C", config2).Exit == 2, "init without a terminal or --yes");
        var init = CliJson("init", "-y", "-C", config2, "--cache", folder, "-d", device.ToString(), "-j");
        Check((bool)init["written"]! && CliConfig.Load(config2).Get("device") == device.ToString() && CliConfig.Load(config2).Get("cache") == Path.GetFullPath(folder), $"init --yes: {init}");
        Terminal.TestInput = new StringReader($"{device}\n\nq=Org/Model\n\n");
        try
        {
            Check(Cli("init", "-C", config2, "--profile", "desk").Exit == 0, "init with answers");
        }
        finally
        {
            Terminal.TestInput = null;
        }

        var desk = CliConfig.Load(config2).Root["profiles"]!["desk"]!;
        Check((string?)desk["device"] == device.ToString() && (string?)desk["aliases"]!["q"]!["model"] == "Org/Model", $"init answers: {desk}");
        Directory.Delete(folder, true);
    }

    private static void CliPluginsFormatsCompletion(Device device)
    {
        _ = device;
        var plugins = CliJson("plugins", "list", "-j");
        Check(plugins["weights"]!["names"]!.AsArray().Any(n => (string?)n == "int8") && plugins["devices"]!["names"]!.AsArray().Any(n => (string?)n == "cpu"), $"plugins list: {plugins}");
        Check(plugins["weights"]!["added"]!.AsArray().Count == 0, "plugins list: something added without --plugin");
        var (fexit, csv, _) = Cli("formats", "--format", "csv");
        Check(fexit == 0 && csv.StartsWith("Kind,Formats", StringComparison.Ordinal) && csv.Contains("int8"), $"formats csv: {csv}");

        foreach (string shell in (string[])["bash", "zsh", "fish", "pwsh"])
        {
            var (exit, script, _) = Cli("completion", shell);
            Check(exit == 0 && script.Contains("completion --complete --"), $"completion {shell}: {script}");
        }

        Check(Cli("completion", "tcsh").Exit == 2, "completion tcsh");
        string Candidates(params string[] words) => Cli(["completion", "--complete", "--", .. words]).Out;
        Check(Candidates("cache", "").Split('\n').Contains("clear") && Candidates("cache", "").Split('\n').Contains("info"), "complete cache");
        Check(Candidates("do").Contains("doctor"), "complete do");
        Check(Candidates("doctor", "--and").Trim() == "--android", $"complete --and: {Candidates("doctor", "--and")}");
        Check(Candidates("devices", "-d", "").Contains("cpu"), "complete -d");
        Check(Candidates("completion", "").Contains("pwsh"), "complete completion");
        Check(Candidates("help", "exit").Contains("exit-codes"), "complete help topics");
    }

    private static void CliUpdateLogin(Device device)
    {
        _ = device;
        string folder = CliTemp(), index = Path.Combine(folder, "index.json");
        File.WriteAllText(index, "{\"versions\":[\"0.0.1\",\"99.0.0-preview.1\",\"98.1.0\"]}");
        WithEnvironment(new Dictionary<string, string?> { ["IDRAK_UPDATE_INDEX"] = index }, () =>
        {
            var json = CliJson("update", "-j");
            Check((bool)json["newer"]! && (string?)json["latest"] is "98.1.0" or "99.0.0-preview.1", $"update: {json}");
            Check((string?)CliJson("update", "--prerelease", "-j")["latest"] == "99.0.0-preview.1", "update --prerelease");
            Check(Cli("update", "--offline").Exit == 1, "update --offline");
        });
        Check(UpdateCommand.Compare("1.2.0", "1.2.0-dev.5") > 0 && UpdateCommand.Compare("1.10.0", "1.9.9") > 0, "update: version order");

        string token = Path.Combine(folder, "hf-token");
        WithEnvironment(new Dictionary<string, string?> { ["HF_TOKEN_PATH"] = token }, () =>
        {
            Terminal.TestInput = new StringReader("hf_abc_secret\n");
            try
            {
                var (exit, output, _) = Cli("login", "hf");
                Check(exit == 0 && File.ReadAllText(token) == "hf_abc_secret" && !output.Contains("hf_abc_secret"), $"login hf: {output}");
            }
            finally
            {
                Terminal.TestInput = null;
            }

            Check(Idrak.Datasets.HuggingFace.Token() is "hf_abc_secret" || Environment.GetEnvironmentVariable("HF_TOKEN") is not null, "the library does not read the stored token");
            Check(Cli("logout", "hf").Exit == 2 && File.Exists(token), "logout without a terminal or --yes");
            Check(Cli("logout", "hf", "-y").Exit == 0 && !File.Exists(token), "logout hf -y");
        });
        Check(Cli("login", "nowhere").Exit == 2, "login nowhere");
        Directory.Delete(folder, true);
    }

    private static void CliHelpTopics(Device device)
    {
        _ = device;
        var (exit, topics, _) = Cli("help", "topics");
        Check(exit == 0 && topics.Contains("exit-codes") && topics.Contains("precision"), $"help topics: {topics}");
        foreach (string topic in HelpTopics.Names)
        {
            var (e, page, _) = Cli("help", topic);
            Check(e == 0 && page.Length > 100, $"help {topic}: {page}");
        }

        Check(Cli("help", "exit", "codes").Out.Contains("usage error"), "help exit codes");
        Check(Cli("help", "formats").Out.Contains("int8"), "help formats lists the registered formats");
        Check(Cli("help", "devices").Out.Contains("Lanes") || Cli("help", "devices").Out.Contains("idrak devices"), "help devices shows the topic and the command");
        var (g, _, gerror) = Cli("config");
        Check(g == 2 && gerror.Contains("idrak config set"), $"group word: {gerror}");
        foreach (var command in Idrak.Cli.Commands.HealthCommands.All)
        {
            Check(command.Usage.Contains("Example", StringComparison.Ordinal), $"{command.Name}: no example in its help");
        }

        string folder = CliTemp(), rc = Path.Combine(folder, "bashrc");
        var dry = CliJson("setup", "android", "--rc", rc, "-j");
        Check(!(bool)dry["applied"]! && !File.Exists(rc), "setup android without --yes changed the file");
        Check(Cli("setup", "android", "--rc", rc, "-y").Exit == 0 && Cli("setup", "android", "--rc", rc, "-y").Exit == 0, "setup android -y");
        Check(File.ReadAllLines(rc).Count(l => l == "export DOTNET_GCHeapHardLimit=0x100000000") == 1, "setup android: the heap line once");
        Directory.Delete(folder, true);
    }

    private static void CliFoundation(Device device)
    {
        string folder = CliTemp();

        // @file: one argument per line, comments and blank lines skipped, @@ for a literal @.
        string args = Path.Combine(folder, "args.txt");
        File.WriteAllText(args, "# the device\n-d\n" + device + "\n\n--json\n");
        var fromFile = Cli("devices", "@" + args);
        Check(fromFile.Exit == 0 && JsonNode.Parse(fromFile.Out)!["devices"]!.AsArray().Count == 1, $"@file: {fromFile.Out} {fromFile.Err}");
        Check(Cli("devices", "@" + Path.Combine(folder, "missing.txt")).Exit == 2, "@file missing");
        Check(ResponseFiles.Expand(["@@x", "--", "@y"]) is ["@x", "--", "@y"], "@@ and --");

        // --log: every line, the verbose ones too, while the output stays as it is.
        string log = Path.Combine(folder, "logs", "run.log");
        var logged = Cli("env", "--all", "--log", log, "-q");
        Check(logged.Exit == 0 && logged.Out.Length == 0 && File.ReadAllText(log).Contains("IDRAK_CACHE"), "--log with --quiet");

        // --output / -O, --format csv and md, --color, --plain.
        string file = Path.Combine(folder, "out.md");
        var written = Cli("formats", "-O", file, "--format", "md");
        Check(written.Exit == 0 && written.Out.Length == 0 && File.ReadAllText(file).StartsWith("| Kind | Formats |", StringComparison.Ordinal), "-O with --format md");
        Check(Cli("env", "--all", "--format", "json").Out.TrimStart().StartsWith('{'), "--format json");
        Check(Cli("env", "--format", "xml").Exit == 2 && Cli("env", "--color", "sometimes").Exit == 2, "bad --format or --color");
        var coloured = Cli("doctor", "-d", device.ToString(), "--color", "always");
        Check(Environment.GetEnvironmentVariable("NO_COLOR") is { Length: > 0 } || coloured.Out.Contains('\u001b'), "--color always gave no colour");
        Check(!Cli("doctor", "-d", device.ToString(), "--color", "never").Out.Contains('\u001b') && !Cli("doctor", "-d", device.ToString()).Out.Contains('\u001b'), "colour without a terminal or with never");

        // --timeout, --threads, --seed, --offline parse; durations.
        Check(CommandContext.ParseDuration("1h30m", "--timeout") == TimeSpan.FromMinutes(90) && CommandContext.ParseDuration("250ms", "--timeout") == TimeSpan.FromMilliseconds(250)
              && CommandContext.ParseDuration("45", "--timeout") == TimeSpan.FromSeconds(45) && CommandContext.ParseDuration("5m", "--timeout") == TimeSpan.FromMinutes(5), "durations");
        Check(Cli("env", "--timeout", "soon").Exit == 2, "--timeout soon");
        int threads = ComputeResources.MaxCpuThreads;
        try
        {
            Check(Cli("env", "--threads", "1", "--seed", "7", "--offline", "--plain", "--timeout", "30s").Exit == 0 && ComputeResources.MaxCpuThreads == 1, "--threads 1");
            Check(Cli("env", "--threads", "0").Exit == 2, "--threads 0");
        }
        finally
        {
            ComputeResources.MaxCpuThreads = threads;
        }

        // Progress text, byte amounts and questions without a terminal.
        Check(ProgressLine.Format("pull", 512L << 20, 1L << 30, TimeSpan.FromSeconds(4), ProgressUnit.Bytes) == "pull   50%  512 MB / 1.00 GB  128 MB/s  4s left",
            $"progress: {ProgressLine.Format("pull", 512L << 20, 1L << 30, TimeSpan.FromSeconds(4), ProgressUnit.Bytes)}");
        Check(Units.Bytes(1536) == "1.5 KB" && Units.Duration(TimeSpan.FromSeconds(185)) == "3m 05s", "bytes and durations");
        Check(Cli("logout", "github", "--dry-run", "-C", Path.Combine(folder, "c.json")).Exit == 0, "logout --dry-run");
        Directory.Delete(folder, true);
    }
}
