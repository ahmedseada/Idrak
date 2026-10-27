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
        var runner = new IdrakCommandRunner();
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
        Check(part.Select(e => e.Setting.Key).SequenceEqual(["terminal.lines", "terminal.history", "terminal.history_shown", "terminal.suggestions", "terminal.folder"])
              && part.All(e => !e.Setting.NeedsRestart), string.Join(", ", part.Select(e => e.Setting.Key)));
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

    // A writer that holds the command until let go (to keep a run going).
    private sealed class BlockingWriter(ManualResetEventSlim gate) : TextWriter
    {
        public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;

        public override void Write(char value) => gate.Wait();
    }
}
