// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Specialized;
using System.Diagnostics;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Idrak.Mobile;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Tests;

// The Terminal page over the app's command runner: the scripted one here, and the app's own (IdrakCommandRunner, compiled
// from apps/Idrak.Mobile) running the idrak commands in this process.
internal static partial class MauiTests
{
    private sealed class ScriptedRunner(Func<IReadOnlyList<string>, TextWriter, TextWriter, CancellationToken, int> run) : ICommandRunner
    {
        public string Name => "idrak";

        public IReadOnlyList<TerminalCommand> Commands { get; } =
        [
            new("models list", ["ls"], "The models in the cache", "Models", true, "", ["--json"]),
            new("models pull", [], "Pulls a model", "Models", true, "", ["--revision"]),
            new("serve", [], "An HTTP server", "Serving", false, "idrak serve is not available in the Idrak app: it opens a network listener (a server).", ["--port"]),
            new("help", [], "Help", "Help", true, "", []),
        ];

        public IReadOnlyList<string> CommonOptions { get; } = ["--device", "--json"];

        public IReadOnlyList<string> HelpTopics { get; } = ["devices", "options"];

        public string? Folder { get; private set; }

        public IReadOnlyList<string> Split(string line) => line.Contains('"') && line.Count(c => c == '"') % 2 == 1
            ? throw new FormatException("The quote \" is not closed.") : line.Split(' ', StringSplitOptions.RemoveEmptyEntries).SkipWhile(w => w == "idrak").ToList();

        public Task<CommandRunResult> RunAsync(IReadOnlyList<string> args, TextWriter output, TextWriter error, string workingFolder, CancellationToken cancellationToken = default)
        {
            Folder = workingFolder;
            return Task.Run(() =>
            {
                var clock = Stopwatch.StartNew();
                int code = run(args, output, error, cancellationToken);
                return new CommandRunResult(code, code == 130, code == 3, clock.Elapsed);
            }, CancellationToken.None);
        }
    }

    private static TerminalViewModel Terminal(ICommandRunner? runner, out string folder, ISettingsStore? store = null, TerminalLabOptions? options = null)
    {
        folder = Temp("terminal");
        return new TerminalViewModel(runner, store ?? new MemorySettingsStore(), folder, options);
    }

    private static void TerminalStreams()
    {
        using var held = new ManualResetEventSlim();
        using var written = new ManualResetEventSlim();
        var runner = new ScriptedRunner((args, output, error, token) =>
        {
            for (int i = 0; i < 1000; i++)
            {
                output.WriteLine($"line {i}");
            }

            output.Write("pull 10%\rpull 50%");
            output.Write("\rpull \u001b[32m100%\u001b[0m\r\n");
            error.WriteLine("a warning");
            output.Write("no newline at the end");
            written.Set();
            held.Wait(token);
            output.Write(" ... and the rest\r\n");
            return 0;
        });
        var vm = Terminal(runner, out string folder);
        int changes = 0, added = 0;
        vm.Lines.CollectionChanged += (_, e) =>
        {
            changes++;
            added += e.Action == NotifyCollectionChangedAction.Add ? e.NewItems!.Count : 0;
        };
        vm.Input = "models list";
        var run = vm.RunAsync();
        Check(written.Wait(TimeSpan.FromSeconds(20)) && vm.IsRunning && !vm.IsIdle && vm.Lines.Count == 1 && vm.Lines[0].IsCommand && vm.Lines[0].Text == "› idrak models list",
            $"running: {vm.Lines.Count} lines before a frame");
        Check(vm.HasPending && vm.Drain() && changes == 2, $"one frame: {changes} changes (the command line, then the frame's batch)");
        Check(vm.Lines.Count == 1 + 1000 + 3 && vm.Lines[1000].Text == "line 999" && vm.Lines[1001].Text == "pull 100%" && vm.Lines[1002] is { IsError: true, Text: "a warning" }
              && vm.Lines[1003].Text == "no newline at the end",
            $"{vm.Lines.Count} lines: '{vm.Lines[1001].Text}', '{vm.Lines[^1].Text}'");
        Check(!vm.Drain() && changes == 2, "nothing new: no change");
        held.Set();
        var result = run.GetAwaiter().GetResult();
        Check(result is { ExitCode: 0, Stopped: false } && vm.Lines[1003].Text == "no newline at the end ... and the rest" && vm.Lines[^1] is { IsInfo: true } done
              && done.Text.StartsWith("done", StringComparison.Ordinal) && !vm.IsRunning && runner.Folder == folder && vm.WorkingFolder == folder,
            $"ended: {vm.Lines[^1].Text}; {vm.Lines[1003].Text}");
        Check(added == vm.Lines.Count, $"{added} lines added in {changes} changes");

        // The most lines kept: the oldest dropped as one change.
        var few = Terminal(new ScriptedRunner((_, o, _, _) =>
        {
            for (int i = 0; i < 25; i++)
            {
                o.WriteLine(i.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            return 2;
        }), out _, options: new TerminalLabOptions { Lines = 10 });
        Check(few.RunAsync("models list").GetAwaiter().GetResult()?.ExitCode == 2 && few.Lines.Count == 10 && few.Lines[^1].Text.StartsWith("exit 2", StringComparison.Ordinal),
            $"{few.Lines.Count} lines kept: {few.Lines[^1].Text}");

        // Copy, clear, a quote not closed, no runner.
        string copied = "";
        few.Copy = text =>
        {
            copied = text;
            return Task.CompletedTask;
        };
        few.CopyCommand.Execute(null);
        Check(copied.Split('\n').Length == 10, $"copied {copied.Split('\n').Length} lines");
        few.Input = "clear";
        few.RunCommand.Execute(null);
        Check(few.Lines.Count == 0 && few.Input.Length == 0, "cleared");
        Check(few.RunAsync("models pull \"Qwen").GetAwaiter().GetResult() is null && few.Lines[^1] is { IsError: true, Text: "The quote \" is not closed." }, "a quote not closed");
        var none = Terminal(null, out _);
        Check(!none.HasRunner && none.RunAsync("help").GetAwaiter().GetResult() is null && none.Lines.Single().IsError && none.NoRunnerText.Contains("ICommandRunner", StringComparison.Ordinal),
            "no runner");
    }

    private static void TerminalStops()
    {
        using var started = new ManualResetEventSlim();
        var vm = Terminal(new ScriptedRunner((_, o, _, token) =>
        {
            started.Set();
            while (!token.IsCancellationRequested)
            {
                o.Write("\rworking");
                Thread.Sleep(1);
            }

            o.WriteLine();
            return 130;
        }), out _);
        var run = vm.RunAsync("models pull Qwen/Qwen3-0.6B");
        Check(started.Wait(TimeSpan.FromSeconds(20)) && vm.Cancel(), "stop asked");
        var result = run.GetAwaiter().GetResult();
        Check(result is { Stopped: true } && vm.Status.StartsWith("stopped", StringComparison.Ordinal) && vm.Lines.Count(l => l.Text == "working") == 1 && !vm.IsRunning,
            $"stopped: {vm.Status}; {string.Join(" | ", vm.Lines.Select(l => l.Text))}");
        Check(!vm.Cancel(), "nothing runs: nothing to stop");
    }

    private static void TerminalHistoryAndSuggestions()
    {
        var store = new MemorySettingsStore();
        var runner = new ScriptedRunner((_, o, _, _) =>
        {
            o.WriteLine("ok");
            return 0;
        });
        var vm = Terminal(runner, out _, store);
        foreach (string line in (string[])["help", "models list", "models list", "models pull x"])
        {
            vm.RunAsync(line).GetAwaiter().GetResult();
        }

        Check(vm.History.SequenceEqual(["models pull x", "models list", "help"]), $"history {string.Join(", ", vm.History)}");
        Check(vm.Recall(older: true) && vm.Input == "models pull x" && vm.Recall(older: true) && vm.Input == "models list" && vm.Recall(older: true) && vm.Input == "help"
              && vm.Recall(older: true) && vm.Input == "help", vm.Input);
        Check(vm.Recall(older: false) && vm.Input == "models list" && vm.Recall(older: false) && !vm.Recall(older: false) && vm.Input.Length == 0, $"forward: '{vm.Input}'");
        var again = Terminal(runner, out _, store);
        Check(again.History.SequenceEqual(vm.History), "kept between runs");
        again.RunAsync("history").GetAwaiter().GetResult();
        Check(again.Lines.Count(l => l.IsInfo) == 4 && again.Lines.Last().Text.Trim().EndsWith("history", StringComparison.Ordinal)
              && again.Lines[^2].Text.Trim().EndsWith("models pull x", StringComparison.Ordinal), string.Join(" | ", again.Lines.Select(l => l.Text)));

        // The suggestions: the runner's commands (one not allowed greyed, saying why), then a command's options.
        vm.Input = "";
        Check(vm.Suggestions.Select(s => s.Text).SequenceEqual(["models list", "models pull", "serve", "help"]) && vm.Suggestions.Single(s => s.Text == "serve").IsUnavailable,
            string.Join(", ", vm.Suggestions.Select(s => s.Text)));
        vm.Input = "mod";
        Check(vm.Suggestions.Select(s => s.Text).SequenceEqual(["models list", "models pull"]), string.Join(", ", vm.Suggestions.Select(s => s.Text)));
        vm.Suggestions[0].TapCommand.Execute(null);
        Check(vm.Input == "models list " && vm.Suggestions.Select(s => s.Text).SequenceEqual(["--json", "--device"]), $"'{vm.Input}': {string.Join(", ", vm.Suggestions.Select(s => s.Text))}");
        vm.Input = "models list --j";
        Check(vm.Suggestions.Single().Text == "--json", "an option by its start");
        vm.Suggestions[0].TapCommand.Execute(null);
        Check(vm.Input == "models list --json ", vm.Input);
        vm.Input = "help ";
        Check(vm.Suggestions.Select(s => s.Text).Take(2).SequenceEqual(["devices", "options"]), "help's topics");
        vm.Input = "";
        vm.Suggestions.Single(s => s.Text == "serve").TapCommand.Execute(null);
        Check(vm.Input.Length == 0 && vm.Status.Contains("network listener", StringComparison.Ordinal), vm.Status);
        Check(vm.Groups.Select(g => g.Title).SequenceEqual(["Models", "Serving", "Help"]) && vm.Groups[1].Single() is { IsUnavailable: true, Reason.Length: > 0 }, "the commands pane");
    }

    private static void TerminalRunsIdrak()
    {
        var runner = new IdrakCommandRunner(allows: () => TerminalAllowance.None);              // a phone's: nothing beyond the library
        var serve = runner.Commands.FirstOrDefault(c => c.Name == "serve");
        Check(runner.Commands.Count > 20 && runner.Commands.Any(c => c.IsAvailable) && serve is { IsAvailable: false } && serve.Unavailable.Contains("the Idrak app", StringComparison.Ordinal),
            $"{runner.Commands.Count} commands; serve: {serve?.Unavailable}");
        Check(runner.Split("idrak models pull \"a b\"").SequenceEqual(["models", "pull", "a b"]), "split as a shell would");
        var vm = Terminal(runner, out string folder);
        var help = vm.RunAsync("help").GetAwaiter().GetResult();
        Check(help is { ExitCode: 0, Refused: false } && vm.Lines.Count > 10 && vm.Lines.Any(l => l.Text.Contains("models", StringComparison.Ordinal)), $"help: {help}; {vm.Lines.Count} lines");
        var refused = vm.RunAsync("serve").GetAwaiter().GetResult();
        Check(refused is { Refused: true } && vm.Lines.Any(l => l.Text.Contains("is not available in the Idrak app", StringComparison.Ordinal))
              && vm.Lines[^1].Text.StartsWith("not available here", StringComparison.Ordinal), $"serve: {refused}; {vm.Lines[^2].Text}");

        // One command at a time.
        using var gate = new ManualResetEventSlim();
        var first = runner.RunAsync(["help"], new BlockingWriter(gate), TextWriter.Null, folder);
        bool second = false;
        try
        {
            runner.RunAsync(["help"], TextWriter.Null, TextWriter.Null, folder).GetAwaiter().GetResult();
        }
        catch (InvalidOperationException)
        {
            second = true;
        }

        gate.Set();
        Check(second && first.GetAwaiter().GetResult().ExitCode == 0, "a second command while one runs was taken");

        // The feature: in this device's group, reached from All labs and Home; its view binds in light and dark, ticking only while a command runs on screen.
        var feature = MobileFeatures.Get(MobileFeatures.Terminal);
        Check(feature is { Title: "Terminal", Group: FeatureGroup.Device, Icon: "terminal" } && new LabsViewModel(new IdrakMauiOptions()).All.Any(l => l.Name == MobileFeatures.Terminal), feature.Title);
        using var services = AppServices();
        foreach (var theme in new[] { AppTheme.Light, AppTheme.Dark })
        {
            var view = new TerminalView(Terminal(new ScriptedRunner((_, _, _, _) => 0), out _), services);
            Themed(view, theme);
            view.Apply(true);
            view.Apply(false);
            Check(CheckLabels(view, theme, "terminal") >= 3 && !view.IsTicking, theme.ToString());
        }
    }

    private static void TerminalSettings()
    {
        var app = new IdrakMauiOptions();
        var store = new MemorySettingsStore();
        var runner = new ScriptedRunner((_, o, _, _) =>
        {
            for (int i = 0; i < 25; i++)
            {
                o.WriteLine(i.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            return 0;
        });
        string folder = Temp("terminal-settings");
        var vm = new TerminalViewModel(runner, store, folder, app: app);
        Check(vm.LineLimit == new TerminalLabOptions().Lines && vm.WorkingFolder == folder, "the app's own limits when nothing is set");
        app.TerminalLines = 5;
        vm.RunAsync("models list").GetAwaiter().GetResult();
        Check(vm.Lines.Count == 5, $"{vm.Lines.Count} lines kept at 5");
        foreach (string line in (string[])["models pull a", "models pull b", "help"])
        {
            vm.RunAsync(line).GetAwaiter().GetResult();
        }

        // A change in Settings is taken when the page is next shown, no restart.
        app.TerminalLines = 3;
        app.TerminalHistory = 2;
        app.TerminalHistoryShown = 1;
        app.TerminalSuggestions = 2;
        string other = Temp("terminal-settings-other");
        app.TerminalFolder = other;
        vm.Refresh();
        vm.Input = "";
        Check(vm.Lines.Count == 3 && vm.History.SequenceEqual(["help"]) && store.Get(TerminalViewModel.HistoryKey)!.Split('\n').Length == 2 && vm.Suggestions.Count == 2
              && vm.WorkingFolder == other, $"{vm.Lines.Count} lines, history {string.Join(", ", vm.History)}, {vm.Suggestions.Count} suggestions, {vm.WorkingFolder}");
        vm.RunAsync("models list").GetAwaiter().GetResult();
        Check(runner.Folder == other, "the commands run in the folder set");

        // The settings page's Terminal part: found by the search, kept as idrak.terminal.*, applied at once.
        var page = new SettingsViewModel(app, new IdrakMauiSettings(store, new MemorySecretStore()), new IdrakStartupReport(ComputeResources.CacheFolder, "auto", [], folder));
        var part = page.Groups.Single(g => g.Title == "Terminal");
        Check(part.Select(e => e.Setting.Key).SequenceEqual(["terminal.lines", "terminal.history", "terminal.history_shown", "terminal.suggestions", "terminal.folder", "terminal.allow"])
              && part.All(e => !e.Setting.NeedsRestart), string.Join(", ", part.Select(e => e.Setting.Key)));
        Check(IdrakMauiSettings.Definitions.All(d => IdrakMauiSettings.Prefix + d.Key != TerminalViewModel.HistoryKey),
              "the saved history lines have a key of their own: no setting is stored under it (saving Settings would write over them)");
        page.Query = "terminal working folder";
        Check(page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).SequenceEqual(["terminal.folder"]), "found by the search");
        page.Query = "terminal";
        Check(part.All(e => e.IsShown) && part.IsShown, "the whole part by its heading");
        page.Query = "";
        part.Single(e => e.Setting.Key == "terminal.lines").Text = "7";
        part.Single(e => e.Setting.Key == "terminal.folder").Text = "relative";
        Check(!page.SaveAsync().GetAwaiter().GetResult(), "a folder that is not a full path refused");
        part.Single(e => e.Setting.Key == "terminal.folder").Text = "";
        Check(page.SaveAsync().GetAwaiter().GetResult() && app.TerminalLines == 7 && vm.LineLimit == 7 && vm.WorkingFolder == folder && store.Get("idrak.terminal.lines") == "7"
              && !page.IsRestartPending, page.Message);
    }

    private static void TerminalAllowed()
    {
        // The platform's default, decided at run time: both on a desktop, none on a phone, which holds any choice to none.
        Check(TerminalAllowances.For(null, desktop: true) == TerminalAllowances.Desktop && TerminalAllowances.For(null, desktop: false) == TerminalAllowance.None
              && TerminalAllowances.For(TerminalAllowance.NetworkListener, desktop: false) == TerminalAllowance.None
              && TerminalAllowances.For(TerminalAllowance.ChildProcesses, desktop: true) == TerminalAllowance.ChildProcesses, "the allowance by platform");
        Check(IdrakPlatform.IsDesktop && TerminalAllowances.Default == TerminalAllowances.Desktop && new IdrakMauiOptions().TerminalAllowInEffect == TerminalAllowances.Desktop,
            "this machine is a desktop: both by default");

        // The setting (the Terminal part, idrak.terminal.allow): each name read back, a bad one refused naming what is taken, kept between runs.
        var setting = IdrakMauiSettings.Definitions.Single(d => d.Key == "terminal.allow");
        var app = new IdrakMauiOptions();
        Check(setting is { Group: "Terminal", Kind: IdrakSettingKind.Choice, NeedsRestart: false, Placeholder: "programs+listener" } && setting.Get(app).Length == 0
              && setting.Choices.SequenceEqual(["default", "programs+listener", "programs", "listener", "none"]) && setting.Summary.Contains("Console input is never allowed", StringComparison.Ordinal),
            $"{setting.Group} {setting.Placeholder} {string.Join(", ", setting.Choices)}");
        foreach (var (text, allow) in new (string, TerminalAllowance?)[]
                 {
                     ("none", TerminalAllowance.None), ("programs", TerminalAllowance.ChildProcesses), ("listener", TerminalAllowance.NetworkListener),
                     ("programs+listener", TerminalAllowances.Desktop), ("all", TerminalAllowances.Desktop), ("server, programs", TerminalAllowances.Desktop), ("default", null), ("", null),
                 })
        {
            setting.Set(app, text);
            Check(app.TerminalAllow == allow && (allow is { } a ? setting.Get(app) == TerminalAllowances.Name(a) : setting.Get(app).Length == 0), $"'{text}': {app.TerminalAllow}, read back '{setting.Get(app)}'");
        }

        Check(Throws<FormatException>(() => setting.Set(app, "console"), "programs+listener"), "console input is not a choice");
        var store = new MemorySettingsStore();
        var saved = new IdrakMauiOptions { TerminalAllow = TerminalAllowance.ChildProcesses };
        new IdrakMauiSettings(store, new MemorySecretStore()).SaveAsync(saved).GetAwaiter().GetResult();
        var read = new IdrakMauiOptions();
        new IdrakMauiSettings(store, new MemorySecretStore()).LoadIntoAsync(read).GetAwaiter().GetResult();
        Check(store.Get("idrak.terminal.allow") == "programs" && read.TerminalAllow == TerminalAllowance.ChildProcesses, $"kept: {store.Get("idrak.terminal.allow")}, read {read.TerminalAllow}");

        // On: serve, cluster start and test are listed as available and run, not refused for what they need; shell and
        // serve --mcp (console input) still refused with why. The same list while the setting stays.
        app.TerminalAllow = null;
        var runner = new IdrakCommandRunner(allows: () => app.TerminalAllowInEffect);
        var on = runner.Commands;
        TerminalCommand Listed(string name) => runner.Commands.Single(c => c.Name == name);
        Check(Listed("serve").IsAvailable && Listed("cluster start").IsAvailable && Listed("test").IsAvailable && Listed("ui").IsAvailable
              && Listed("shell") is { IsAvailable: false, Unavailable: var shell } && shell.Contains("keyboard", StringComparison.Ordinal) && ReferenceEquals(runner.Commands, on),
            $"on: serve {Listed("serve").IsAvailable}, test {Listed("test").IsAvailable}, shell {Listed("shell").Unavailable}");
        string folder = Temp("terminal-allow");
        var vm = new TerminalViewModel(runner, new MemorySettingsStore(), folder, app: app);
        var serve = vm.RunAsync("serve Qwen/Qwen3-0.6B -p 0").GetAwaiter().GetResult();
        string said = string.Join("\n", vm.Lines.Select(l => l.Text));
        Check(serve is not null && !said.Contains("network listener", StringComparison.Ordinal) && said.Contains("no server is registered here", StringComparison.Ordinal),
            $"serve allowed (this process registers no server, as the app's does not yet): {serve}; {said}");
        vm.Clear();
        var mcp = vm.RunAsync("serve Qwen/Qwen3-0.6B --mcp").GetAwaiter().GetResult();
        Check(mcp is { Refused: true } && vm.Lines.Any(l => l.Text.Contains("standard input", StringComparison.Ordinal)), $"serve --mcp: {mcp}");
        bool inCheckout = false;
        for (var up = new DirectoryInfo(folder); up is not null; up = up.Parent)
        {
            inCheckout |= File.Exists(Path.Combine(up.FullName, "Idrak.slnx"));
        }

        if (!inCheckout)                                                         // outside a source checkout, idrak test only says so (it runs no tests)
        {
            vm.Clear();
            var test = vm.RunAsync("test").GetAwaiter().GetResult();
            Check(test is { Refused: false, ExitCode: 1 } && vm.Lines.Any(l => l.Text.Contains("not inside an Idrak source checkout", StringComparison.Ordinal)),
                $"test allowed: {test}; {string.Join(" | ", vm.Lines.Select(l => l.Text))}");
        }

        // Off: refused with the reason, and the commands pane and the suggestions grey them when the page is next shown.
        var serveRow = vm.Groups.SelectMany(g => g).Single(s => s.Text == "serve");
        Check(serveRow.IsAvailable, "the pane: serve available while on");
        app.TerminalAllow = TerminalAllowance.None;
        Check(!ReferenceEquals(runner.Commands, on) && !Listed("serve").IsAvailable && Listed("serve").Unavailable.Contains("network listener", StringComparison.Ordinal)
              && Listed("test").Unavailable.Contains("other programs", StringComparison.Ordinal), $"off: {Listed("serve").Unavailable}; {Listed("test").Unavailable}");
        vm.Refresh();
        vm.Input = "ser";
        Check(vm.Groups.SelectMany(g => g).Single(s => s.Text == "serve") is { IsUnavailable: true, Reason.Length: > 0 } && vm.Suggestions.Single(s => s.Text == "serve").IsUnavailable,
            "the pane and the suggestions follow the setting");
        vm.Clear();
        var refused = vm.RunAsync("serve Qwen/Qwen3-0.6B").GetAwaiter().GetResult();
        Check(refused is { Refused: true } && vm.Lines.Any(l => l.Text.Contains("is not available in the Idrak app: it opens a network listener", StringComparison.Ordinal)),
            $"off: {refused}; {string.Join(" | ", vm.Lines.Select(l => l.Text))}");
        app.TerminalAllow = TerminalAllowance.NetworkListener;
        vm.Refresh();
        Check(vm.Groups.SelectMany(g => g).Single(s => s.Text == "serve").IsAvailable && vm.Groups.SelectMany(g => g).Single(s => s.Text == "test").IsUnavailable,
            "the listener alone: serve on, test off");
    }

    // A writer that holds the command until let go (to keep a run going).
    private sealed class BlockingWriter(ManualResetEventSlim gate) : TextWriter
    {
        public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;

        public override void Write(char value) => gate.Wait();
    }
}
