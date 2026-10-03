// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak;
using Idrak.Cli;
using Idrak.Cli.Shared;

// The idrak tool across its groups, after they were merged: one help layout for every command and help grouped as the
// plan groups them, the environment table against the command table, --cache and --offline through model resolution
// (pull, then run offline from the same cache), --timeout on server calls, --format on every table, --seed, and the
// progress line. Run in-process through CommandLine.Run with captured output; the hub is the local stand-in.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliPolishGroup =
    [
        ("cli polish: help lists the commands by plan group with aliases; every command's help has one layout", CliPolishHelpLayout),
        ("cli polish: the environment table names real commands; every help ends with the generated section", CliPolishEnvironment),
        ("cli polish: pull then run from the same --cache offline; list shows the use; --offline names what is missing", CliPolishCacheOffline),
        ("cli polish: --timeout gives up on a server that does not answer (ping, api)", CliPolishTimeout),
        ("cli polish: --format csv and md on the commands that print tables", CliPolishFormats),
        ("cli polish: --seed reaches the data commands and suggest; progress lines draw, erase and clear", CliPolishSeedProgress),
    ];

    private static (int Code, string Out, string Err) PolishCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        // No piped input: a command that reads standard input (run's prompt) gets none instead of waiting for it.
        int code = StandardInput.With(new StringReader(""), () =>
            CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-polish-no-config.json")], output, error));
        return (code, output.ToString(), error.ToString());
    }

    private static string PolishTemp()
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-cli-polish-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void CliPolishHelpLayout(Device device)
    {
        _ = device;
        var helpOutput = new StringWriter();
        Check(CommandLine.Run(["help"], helpOutput, new StringWriter()) == 0, "idrak help");
        string overview = helpOutput.ToString();
        int last = -1;
        foreach (var (title, commands) in CommandTable.Groups)
        {
            int at = overview.IndexOf($"\n{title}:\n", StringComparison.Ordinal);
            Check(at > last, $"idrak help: the group '{title}' is missing or out of the plan's order");
            last = at;
            foreach (var command in commands)
            {
                string name = command.Aliases.Count > 0 ? $"{command.Name} ({string.Join(", ", command.Aliases)})" : command.Name;
                Check(overview.Contains($"\n  {name} ", StringComparison.Ordinal), $"idrak help: {name} is not listed under {title}");
            }
        }

        var problems = new List<string>();
        foreach (var command in CommandTable.All)
        {
            string body = Help.Body(command);
            string[] lines = body.Split('\n');
            int Index(Func<string, bool> match) => Array.FindIndex(lines, l => match(l));
            int options = Index(l => l == "Options:"), arguments = Index(l => l == "Arguments:"), examples = Index(l => l == "Examples:");
            int limits = Index(l => Regex.IsMatch(l, @"^(Limits?|Gaps?)( \([^)]*\))?:"));
            string where = $"idrak {command.Name}";
            if (examples < 0)
            {
                problems.Add($"{where}: no 'Examples:' section");
            }

            if (command.ValueOptions.Count + command.Flags.Count > 0 && options < 0)
            {
                problems.Add($"{where}: options but no 'Options:' section");
            }

            if (arguments >= 0 && options >= 0 && arguments > options || options >= 0 && examples >= 0 && options > examples)
            {
                problems.Add($"{where}: sections out of order (Arguments, Options, Examples)");
            }

            if (limits >= 0 && limits < examples)
            {
                problems.Add($"{where}: limits and gaps come after the examples");
            }

            foreach (int heading in Enumerable.Range(0, lines.Length).Where(i => Regex.IsMatch(lines[i], @"^[A-Z][A-Za-z -]*:$")))
            {
                if (heading > 0 && lines[heading - 1].Length > 0)
                {
                    problems.Add($"{where}: no blank line before '{lines[heading]}'");
                }
            }

            foreach (string option in command.ValueOptions.Concat(command.Flags).Where(o => !body.Contains(o, StringComparison.Ordinal)))
            {
                problems.Add($"{where}: {option} is not described");
            }

            foreach (var (shortForm, longForm) in command.ShortForms.Where(p => !body.Contains($"{p.Key}, {p.Value}", StringComparison.Ordinal)))
            {
                problems.Add($"{where}: the options do not show '{shortForm}, {longForm}'");
            }

            string help = Help.For(command);
            if (!help.StartsWith($"idrak {command.Name}: {command.Summary}\n\nUsage: idrak {command.Name}", StringComparison.Ordinal))
            {
                problems.Add($"{where}: the help's first lines");
            }

            if (help.IndexOf("\nCommon options:\n", StringComparison.Ordinal) < help.IndexOf("\nExamples:\n", StringComparison.Ordinal))
            {
                problems.Add($"{where}: the common options come before the command's own text");
            }

            problems.AddRange(help.TrimEnd().Split('\n').Where(l => l.Length > 120).Select(l => $"{where}: wider than 120 characters: {l[..60]}..."));
        }

        Check(problems.Count == 0, "help layout:\n  " + string.Join("\n  ", problems));
        var (alias, aliasHelp, _) = PolishCli("ls", "--help");
        Check(alias == 0 && aliasHelp.Contains("Aliases: idrak ls", StringComparison.Ordinal), "an alias's help names it");
        var (usage, _, usageError) = PolishCli("rm");
        Check(usage == 2 && usageError.Contains("Usage: idrak rm MODEL", StringComparison.Ordinal) && !usageError.Contains("Examples:", StringComparison.Ordinal),
            $"a usage error shows the usage line only: {usageError}");
    }

    private static void CliPolishEnvironment(Device device)
    {
        _ = device;
        var names = CommandTable.All.Select(c => c.Name).ToHashSet(StringComparer.Ordinal);
        foreach (var variable in EnvironmentVariables.All)
        {
            foreach (string command in variable.Commands.Where(c => c != EnvironmentVariables.Every))
            {
                Check(names.Contains(command), $"{variable.Name} names '{command}', which is not a command");
            }
        }

        foreach (var command in CommandTable.All)
        {
            string help = Help.For(command);
            int section = help.IndexOf("\nEnvironment (idrak help env for all):\n", StringComparison.Ordinal);
            Check(section > 0 && help.IndexOf("Environment:", StringComparison.Ordinal) < 0, $"idrak {command.Name}: one generated environment section and no hand-written one");
            foreach (var variable in EnvironmentVariables.For(command.Name))
            {
                Check(help.IndexOf(variable.Name, section, StringComparison.Ordinal) > 0, $"idrak {command.Name}: {variable.Name} is missing from its environment section");
            }
        }

        Check(Help.For(CommandTable.All.First(c => c.Name == "serve")).Contains("IDRAK_API_KEY", StringComparison.Ordinal), "serve's help names IDRAK_API_KEY");
        Check(Help.For(CommandTable.All.First(c => c.Name == "setup android")).Contains("VK_ICD_FILENAMES", StringComparison.Ordinal), "setup android names VK_ICD_FILENAMES");
        Check(!Help.For(CommandTable.All.First(c => c.Name == "data preview")).Contains("HF_TOKEN", StringComparison.Ordinal), "data preview reads local files only");
    }

    private static void CliPolishCacheOffline(Device device)
    {
        string cache = PolishTemp(), work = PolishTemp();
        try
        {
            string[] c = ["--cache", cache];
            using (new FakeHub())
            {
                var (pulled, _, pullError) = PolishCli(["pull", "test/tiny-llama", .. c]);
                Check(pulled == 0, $"pull into --cache: {pullError}");
            }

            // The hub is gone: run finds the model in the same cache, without the network and with --offline.
            var before = DateTime.UtcNow.AddSeconds(-1);
            var (ran, answer, runError) = PolishCli(["run", "test/tiny-llama", "Hello", "--max-tokens", "3", "--seed", "1", "--offline", "-d", device.ToString(), "-j", .. c]);
            Check(ran == 0 && JsonNode.Parse(answer)?["text"] is not null, $"run from the cache offline: {runError}{answer}");
            var listed = JsonNode.Parse(PolishCli(["list", "-j", .. c]).Out)!["models"]!.AsArray().First(m => (string?)m!["name"] == "test/tiny-llama")!;
            Check(DateTime.Parse((string)listed["lastUse"]!, System.Globalization.CultureInfo.InvariantCulture).ToUniversalTime() >= before, $"list shows run's use: {listed}");

            // A model the cache lacks: offline names it and pull; a GGUF file of a repository is fetched by pull only.
            var (missing, _, missingError) = PolishCli(["run", "test/missing", "Hi", "--offline", .. c]);
            Check(missing == 1 && missingError.Contains("idrak pull test/missing", StringComparison.Ordinal) && missingError.Contains("--offline", StringComparison.Ordinal),
                $"run of a missing model offline: {missingError}");
            var (tagged, _, taggedError) = PolishCli(["run", "test/tiny-gguf:Q8_0", "Hi", .. c]);
            Check(tagged == 1 && taggedError.Contains("idrak pull test/tiny-gguf:Q8_0", StringComparison.Ordinal), $"an unpulled GGUF tag names pull: {taggedError}");

            // A download refused by --offline names the missing file.
            string recipe = Path.Combine(work, "recipe.json");
            File.WriteAllText(recipe, """{"sources": ["https://example.invalid/data/chats.jsonl"]}""");
            var (mixed, _, mixError) = PolishCli(["data", "mix", recipe, "-o", Path.Combine(work, "out.jsonl"), "--offline", .. c]);
            Check(mixed == 1 && mixError.Contains("--offline", StringComparison.Ordinal) && mixError.Contains("chats.jsonl", StringComparison.Ordinal), $"an offline download: {mixError}");

            // A .gguf file is prepared under --cache, not the default cache.
            var (tokens, _, tokensError) = PolishCli(["tokenize", TestData("gguf/tiny-llama.gguf"), "hello", "--count", .. c]);
            Check(tokens == 0 && Directory.Exists(Path.Combine(cache, "gguf")), $"a GGUF file prepared under --cache: {tokensError}");
        }
        finally
        {
            Directory.Delete(cache, true);
            Directory.Delete(work, true);
        }
    }

    private static void CliPolishTimeout(Device device)
    {
        _ = device;
        // A server that accepts connections and never answers.
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        var held = new List<TcpClient>();
        var accepting = Task.Run(async () =>
        {
            try
            {
                while (true)
                {
                    held.Add(await listener.AcceptTcpClientAsync().ConfigureAwait(false));
                }
            }
            catch (Exception e) when (e is SocketException or ObjectDisposedException)
            {
            }
        });

        var watch = Stopwatch.StartNew();
        var (pinged, _, pingError) = PolishCli("ping", $"http://127.0.0.1:{port}", "--timeout", "300ms");
        Check(pinged == 1 && pingError.Contains("no answer", StringComparison.Ordinal) && watch.Elapsed < TimeSpan.FromSeconds(10), $"ping --timeout: {pingError} ({watch.Elapsed})");
        watch.Restart();
        var (called, _, callError) = PolishCli("api", "/api/tags", "-H", "127.0.0.1", "-p", port.ToString(System.Globalization.CultureInfo.InvariantCulture), "--timeout", "300ms");
        Check(called == 1 && callError.Contains("--timeout", StringComparison.Ordinal) && watch.Elapsed < TimeSpan.FromSeconds(10), $"api --timeout: {callError} ({watch.Elapsed})");
        listener.Stop();
        accepting.Wait(TimeSpan.FromSeconds(5));
        held.ForEach(h => h.Dispose());
    }

    private static void CliPolishFormats(Device device)
    {
        string cache = PolishTemp();
        try
        {
            string model = TestData("gguf/tiny-llama-hf");
            string[][] commands =
            [
                ["devices"], ["families"], ["formats"], ["plugins", "list"], ["env", "--all"], ["cache", "info"], ["version"], ["doctor"],
                ["show", model], ["inspect", TestData("gguf/tiny-llama.gguf")], ["memory", model], ["verify", model], ["tokenize", model, "hello"],
            ];
            foreach (string[] command in commands)
            {
                var (csvCode, csv, csvError) = PolishCli([.. command, "--format", "csv", "--cache", cache, "-d", device.ToString()]);
                Check(csvCode is 0 or 1 && csv.Length > 0 && csv.Split('\n')[0].Contains(',', StringComparison.Ordinal) && !csv.StartsWith(' '),
                    $"idrak {string.Join(' ', command)} --format csv: {csv}{csvError}");
                var (mdCode, md, _) = PolishCli([.. command, "--format", "md", "--cache", cache, "-d", device.ToString()]);
                Check(mdCode is 0 or 1 && md.Contains("\n|---", StringComparison.Ordinal) || md.StartsWith("| ", StringComparison.Ordinal), $"idrak {string.Join(' ', command)} --format md: {md}");
            }

            // Every table's headers are in sentence case.
            var (_, devices, _) = PolishCli("devices", "--format", "csv");
            Check(devices.StartsWith("Device,Backend,", StringComparison.Ordinal), $"devices headers: {devices}");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliPolishSeedProgress(Device device)
    {
        string folder = PolishTemp();
        try
        {
            string rows = Path.Combine(folder, "rows.jsonl");
            File.WriteAllLines(rows, Enumerable.Range(0, 50).Select(i => $$"""{"id": {{i}}, "text": "row {{i}}"}"""));
            string Sample(string seed) => JsonNode.Parse(PolishCli("data", "sample", rows, "-n", "5", "--seed", seed, "-j").Out)!["sample"]!.ToJsonString();
            Check(Sample("3") == Sample("3") && Sample("3") != Sample("4"), "data sample follows --seed");

            string csv = Path.Combine(folder, "houses.csv");
            File.WriteAllLines(csv, ["size,rooms,price", .. Enumerable.Range(0, 40).Select(i => $"{50 + i},{1 + i % 4},{100 + 3 * i}")]);
            var (suggested, _, suggestError) = PolishCli("suggest", csv, "-t", "price", "-o", Path.Combine(folder, "design"), "--seed", "7", "-d", device.ToString());
            Check(suggested == 0, $"suggest --seed: {suggestError}");
            var prep = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "design", "prep.json")))!;
            var train = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "design", "train.json")))!;
            Check((int?)prep["split"]?["seed"] == 7 && (int?)train["seed"] == 7, $"suggest writes the seed: {prep["split"]} {train["seed"]}");

            // The progress line: drawn on request, taken off for a line of output, cleared at the end; never with --plain.
            var command = CommandTable.All.First(c => c.Name == "version");
            var error = new StringWriter();
            using (var context = new CommandContext(command, [], [], [], new StringWriter(), error))
            {
                var line = new ProgressLine(context, "work", 10, enabled: true);
                line.Report(5);
                Check(error.ToString().Contains("\rwork   50%  5 / 10", StringComparison.Ordinal), $"progress drawn: {error}");
                line.Erase();
                line.Report(6);
                line.Clear();
                Check(error.ToString().EndsWith('\r') && error.ToString().Contains("60%", StringComparison.Ordinal), $"erased, redrawn and cleared: {error}");
                Check(ProgressLine.Format("loading", 0, null, TimeSpan.FromSeconds(3), ProgressUnit.Elapsed) == "loading  3s", "elapsed progress");
            }

            var plainError = new StringWriter();
            using (var plain = new CommandContext(command, [], [], ["--plain"], new StringWriter(), plainError))
            {
                var line = new ProgressLine(plain, "work", 10);
                line.Report(5);
                line.Finish();
                Check(!line.Enabled && plainError.ToString().Length == 0, "--plain draws no progress");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
