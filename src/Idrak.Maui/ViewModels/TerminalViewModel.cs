// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Globalization;
using System.Text;
using System.Windows.Input;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>What a line of the terminal is.</summary>
public enum TerminalLineKind
{
    /// <summary>The command typed.</summary>
    Command,

    /// <summary>What the command wrote.</summary>
    Output,

    /// <summary>What it wrote as an error.</summary>
    Error,

    /// <summary>The terminal's own words: how a command ended, the history.</summary>
    Info,
}

/// <summary>One line of the terminal's output; the last line of a command's output may still change (a progress line redrawn in place).</summary>
public sealed class TerminalLine : ViewModel
{
    private string _text;

    internal TerminalLine(string text, TerminalLineKind kind)
    {
        _text = text;
        Kind = kind;
    }

    /// <summary>Its text.</summary>
    public string Text
    {
        get => _text;
        internal set => SetProperty(ref _text, value);
    }

    /// <summary>What it is.</summary>
    public TerminalLineKind Kind { get; }

    /// <summary>Whether it is the command typed.</summary>
    public bool IsCommand => Kind == TerminalLineKind.Command;

    /// <summary>Whether it is an error.</summary>
    public bool IsError => Kind == TerminalLineKind.Error;

    /// <summary>Whether it is the terminal's own.</summary>
    public bool IsInfo => Kind == TerminalLineKind.Info;
}

/// <summary>The terminal's lines: added and dropped many at a time, each a single change (a list view takes one batch a frame).</summary>
public sealed class TerminalLines : ObservableCollection<TerminalLine>
{
    /// <summary>Adds <paramref name="lines"/> at the end as one change.</summary>
    public void AddRange(IReadOnlyList<TerminalLine> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);
        if (lines.Count == 0)
        {
            return;
        }

        CheckReentrancy();
        int start = Count;
        foreach (var line in lines)
        {
            Items.Add(line);
        }

        OnPropertyChanged(new PropertyChangedEventArgs(nameof(Count)));
        OnPropertyChanged(new PropertyChangedEventArgs("Item[]"));
        OnCollectionChanged(new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, lines.ToList(), start));
    }

    /// <summary>Drops the first <paramref name="count"/> lines as one change.</summary>
    public void RemoveFirst(int count)
    {
        count = Math.Min(count, Count);
        if (count <= 0)
        {
            return;
        }

        CheckReentrancy();
        var removed = new List<TerminalLine>(count);
        for (int i = 0; i < count; i++)
        {
            removed.Add(Items[0]);
            Items.RemoveAt(0);
        }

        OnPropertyChanged(new PropertyChangedEventArgs(nameof(Count)));
        OnPropertyChanged(new PropertyChangedEventArgs("Item[]"));
        OnCollectionChanged(new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Remove, removed, 0));
    }
}

/// <summary>A word the terminal offers to type: a command, an option, a help topic; a command the app does not allow is greyed with why.</summary>
public sealed class TerminalSuggestion(string text, string detail, bool available, string reason, ICommand tap)
{
    /// <summary>The word ("models list", "--json").</summary>
    public string Text { get; } = text;

    /// <summary>One line on it (a command's help).</summary>
    public string Detail { get; } = detail;

    /// <summary>Whether it runs here.</summary>
    public bool IsAvailable { get; } = available;

    /// <summary>Whether it does not.</summary>
    public bool IsUnavailable => !IsAvailable;

    /// <summary>Why it does not, or "".</summary>
    public string Reason { get; } = reason;

    /// <summary>Types it (or, for one that does not run here, says why).</summary>
    public ICommand TapCommand { get; } = tap;
}

/// <summary>The commands of one group ("Models"), for the commands pane.</summary>
public sealed class TerminalCommandGroup(string title, IEnumerable<TerminalSuggestion> commands) : ObservableCollection<TerminalSuggestion>(commands)
{
    /// <summary>Its heading.</summary>
    public string Title { get; } = title;
}

/// <summary>What a command writes, taken from its thread without waiting: the terminal drains it once a display frame.</summary>
internal sealed class TerminalBuffer
{
    private readonly Lock _gate = new();
    private List<(bool Error, StringBuilder Text)> _chunks = [];
    private List<(bool Error, StringBuilder Text)> _spare = [];

    public bool IsEmpty
    {
        get
        {
            lock (_gate)
            {
                return _chunks.Count == 0;
            }
        }
    }

    public void Append(bool error, ReadOnlySpan<char> text)
    {
        if (text.IsEmpty)
        {
            return;
        }

        lock (_gate)
        {
            if (_chunks.Count > 0 && _chunks[^1].Error == error)
            {
                _chunks[^1].Text.Append(text);
            }
            else
            {
                _chunks.Add((error, new StringBuilder().Append(text)));
            }
        }
    }

    // Everything written since the last take, in order; the lists are swapped, so a writer never waits for the reader's work.
    public List<(bool Error, StringBuilder Text)> Take()
    {
        lock (_gate)
        {
            (_chunks, _spare) = (_spare, _chunks);
            _chunks.Clear();
            return _spare;
        }
    }
}

/// <summary>A writer into the terminal's buffer (the command's output or its errors), safe from any thread.</summary>
internal sealed class TerminalWriter(TerminalBuffer buffer, bool error) : TextWriter
{
    public override Encoding Encoding => Encoding.UTF8;

    public override void Write(char value) => buffer.Append(error, [value]);

    public override void Write(string? value) => buffer.Append(error, value);

    public override void Write(char[] buffer1, int index, int count) => buffer.Append(error, buffer1.AsSpan(index, count));

    public override void Write(ReadOnlySpan<char> value) => buffer.Append(error, value);

    public override void WriteLine(string? value)
    {
        buffer.Append(error, value);
        buffer.Append(error, "\n");
    }
}

/// <summary>
/// The Terminal page (the owner's request): the app's command runner (<see cref="ICommandRunner"/>; Idrak.Mobile's runs the
/// idrak commands in the app's own process) on a line typed or picked from suggestions (the runner's commands, options and
/// help topics, never a list of its own; a command the app does not allow greyed with why). One command at a time, run off
/// the UI thread in the app's data folder (so <c>idrak models list</c> lists what Models pulled); what it writes is taken
/// from its thread without waiting and drained into the lines once a display frame (<see cref="Drain"/>), never one update
/// a line; a progress line redrawn with a carriage return is rewritten in place. Stop cancels it as Ctrl+C would; the
/// history recalls earlier lines; copy, clear, <c>help</c>.
/// </summary>
public sealed class TerminalViewModel : ViewModel, IDisposable
{
    /// <summary>What the history's key in the settings store is.</summary>
    public const string HistoryKey = IdrakMauiSettings.Prefix + "terminal.history";

    private readonly ICommandRunner? _runner;
    private readonly ISettingsStore _settings;
    private readonly TerminalBuffer _buffer = new();
    private readonly List<string> _history = [];
    private readonly List<TerminalLine> _added = [];
    private readonly TerminalLabOptions _options;
    private CancellationTokenSource? _stop;
    private TerminalLine? _open;
    private bool _carriage, _running, _showCommands;
    private int _recall = -1;
    private string _input = "", _status = "";

    /// <summary>A terminal over <paramref name="runner"/> (none: the page says the app registers none), working in <paramref name="workingFolder"/>.</summary>
    public TerminalViewModel(ICommandRunner? runner, ISettingsStore settings, string workingFolder, TerminalLabOptions? options = null)
    {
        _runner = runner;
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        ArgumentException.ThrowIfNullOrWhiteSpace(workingFolder);
        _options = options ?? new TerminalLabOptions();
        WorkingFolder = _options.WorkingFolder ?? workingFolder;
        foreach (string line in (settings.Get(HistoryKey) ?? "").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            _history.Add(line);
        }

        RunCommand = new Command(async () => await RunAsync());
        CancelCommand = new Command(() => Cancel());
        ClearCommand = new Command(Clear);
        CopyCommand = new Command(async () => await CopyAsync());
        HelpCommand = new Command(async () => await RunAsync("help"));
        PreviousCommand = new Command(() => Recall(older: true));
        NextCommand = new Command(() => Recall(older: false));
        ToggleCommandsCommand = new Command(() => ShowCommands = !ShowCommands);
        if (runner is not null)
        {
            foreach (var group in runner.Commands.GroupBy(c => c.Group))
            {
                Groups.Add(new TerminalCommandGroup(group.Key, group.Select(Suggest)));
            }
        }

        UpdateHistory();
        Suggest();
    }

    /// <summary>The page's title.</summary>
    public string Title => "Terminal";

    /// <summary>The folder commands run in (relative paths read from it).</summary>
    public string WorkingFolder { get; }

    /// <summary>The tool's name ("idrak"), or "" when no runner is registered.</summary>
    public string ToolName => _runner?.Name ?? "";

    /// <summary>Whether the app registers a runner.</summary>
    public bool HasRunner => _runner is not null;

    /// <summary>Why there is nothing to run, or "".</summary>
    public string NoRunnerText => HasRunner ? "" : "No command runner is registered: the app registers one (ICommandRunner) to run its tool's commands here.";

    /// <summary>The output, a line each (virtualised by the view).</summary>
    public TerminalLines Lines { get; } = [];

    /// <summary>The words offered for what is typed.</summary>
    public ObservableCollection<TerminalSuggestion> Suggestions { get; } = [];

    /// <summary>Every command by its group (the commands pane).</summary>
    public ObservableCollection<TerminalCommandGroup> Groups { get; } = [];

    /// <summary>The lines run before, the latest first.</summary>
    public ObservableCollection<string> History { get; } = [];

    /// <summary>The line being typed.</summary>
    public string Input
    {
        get => _input;
        set
        {
            if (SetProperty(ref _input, value ?? ""))
            {
                Suggest();
            }
        }
    }

    /// <summary>Whether a command runs.</summary>
    public bool IsRunning
    {
        get => _running;
        private set
        {
            if (SetProperty(ref _running, value))
            {
                OnPropertyChanged(nameof(IsIdle));
            }
        }
    }

    /// <summary>Whether none runs.</summary>
    public bool IsIdle => !IsRunning;

    /// <summary>Whether what a command wrote waits to be drawn.</summary>
    public bool HasPending => !_buffer.IsEmpty;

    /// <summary>How the last command ended, or what the terminal says now.</summary>
    public string Status
    {
        get => _status;
        private set => SetProperty(ref _status, value);
    }

    /// <summary>Whether the commands pane shows (a narrow window; a wide one shows it beside the output).</summary>
    public bool ShowCommands
    {
        get => _showCommands;
        set => SetProperty(ref _showCommands, value);
    }

    /// <summary>Copies the output (the view gives the clipboard).</summary>
    public Func<string, Task>? Copy { get; set; }

    /// <summary>Raised on the UI thread after a drain added lines (the view follows the end).</summary>
    public event EventHandler? LinesAdded;

    /// <summary>Runs the line typed.</summary>
    public ICommand RunCommand { get; }

    /// <summary>Stops the command running.</summary>
    public ICommand CancelCommand { get; }

    /// <summary>Clears the output.</summary>
    public ICommand ClearCommand { get; }

    /// <summary>Copies the output.</summary>
    public ICommand CopyCommand { get; }

    /// <summary>Runs <c>help</c>.</summary>
    public ICommand HelpCommand { get; }

    /// <summary>Recalls the line run before the one shown.</summary>
    public ICommand PreviousCommand { get; }

    /// <summary>Recalls the line run after the one shown.</summary>
    public ICommand NextCommand { get; }

    /// <summary>Shows or hides the commands pane.</summary>
    public ICommand ToggleCommandsCommand { get; }

    /// <summary>
    /// Runs <paramref name="line"/> (the line typed when null): "clear" clears and "history" lists the history here; anything
    /// else goes to the runner, off the UI thread, its output drained once a frame. The exit code, or null when nothing ran.
    /// </summary>
    public async Task<CommandRunResult?> RunAsync(string? line = null)
    {
        string text = (line ?? Input).Trim();
        if (text.Length == 0 || IsRunning)
        {
            return null;
        }

        if (line is null)
        {
            Input = "";
        }

        Remember(text);
        Drain();
        if (text is "clear" or "cls")
        {
            Clear();
            return null;
        }

        if (text == "history")
        {
            Append(new TerminalLine($"› {text}", TerminalLineKind.Command));
            Append([.. _history.Select((h, i) => new TerminalLine(string.Create(CultureInfo.InvariantCulture, $"{i + 1,4}  {h}"), TerminalLineKind.Info))]);
            return null;
        }

        if (_runner is not { } runner)
        {
            Append(new TerminalLine(NoRunnerText, TerminalLineKind.Error));
            return null;
        }

        IReadOnlyList<string> args;
        try
        {
            args = runner.Split(text);
        }
        catch (FormatException e)
        {
            Append(new TerminalLine($"› {text}", TerminalLineKind.Command));
            Append(new TerminalLine(e.Message, TerminalLineKind.Error));
            return null;
        }

        Append(new TerminalLine($"› {runner.Name} {string.Join(' ', args)}".TrimEnd(), TerminalLineKind.Command));
        using var stop = new CancellationTokenSource();
        _stop = stop;
        IsRunning = true;
        Status = $"Running {runner.Name} {string.Join(' ', args)}…";
        try
        {
            Directory.CreateDirectory(WorkingFolder);
            var result = await runner.RunAsync(args, new TerminalWriter(_buffer, error: false), new TerminalWriter(_buffer, error: true), WorkingFolder, stop.Token)
                .ConfigureAwait(true);
            Drain();
            Finish();
            string ended = result.Stopped ? "stopped" : result.Refused ? "not available here" : result.ExitCode == 0 ? "done" : $"exit {result.ExitCode}";
            Status = string.Create(CultureInfo.InvariantCulture, $"{ended} · {result.Elapsed.TotalSeconds:0.0} s");
            Append(new TerminalLine(Status, TerminalLineKind.Info));
            return result;
        }
        catch (Exception e) when (e is InvalidOperationException or IOException or UnauthorizedAccessException)
        {
            Drain();
            Finish();
            Append(new TerminalLine(e.Message, TerminalLineKind.Error));
            Status = "";
            return null;
        }
        finally
        {
            _stop = null;
            IsRunning = false;
        }
    }

    /// <summary>Stops the command running (as Ctrl+C would); false when none runs.</summary>
    public bool Cancel()
    {
        try
        {
            _stop?.Cancel();
            return _stop is not null;
        }
        catch (ObjectDisposedException)
        {
            return false;
        }
    }

    /// <summary>
    /// On the UI thread, once a display frame while a command runs: what it wrote since the last frame made into lines, the new
    /// ones added as one change, a carriage return rewriting the line it is on; true when anything changed.
    /// </summary>
    public bool Drain()
    {
        var chunks = _buffer.Take();
        if (chunks.Count == 0)
        {
            return false;
        }

        foreach (var (error, text) in chunks)
        {
            Feed(error ? TerminalLineKind.Error : TerminalLineKind.Output, text);
        }

        chunks.Clear();
        Flush();
        return true;
    }

    /// <summary>Clears the output.</summary>
    public void Clear()
    {
        Drain();
        _open = null;
        _carriage = false;
        Lines.Clear();
        Status = "";
    }

    /// <summary>The output as text, a line each.</summary>
    public string OutputText() => string.Join("\n", Lines.Select(l => l.Text));

    /// <summary>Recalls an earlier (<paramref name="older"/>) or later line into the input; false when there is none.</summary>
    public bool Recall(bool older)
    {
        if (_history.Count == 0)
        {
            return false;
        }

        int next = older ? Math.Min(_recall + 1, _history.Count - 1) : _recall - 1;
        if (next < 0)
        {
            _recall = -1;
            _input = "";
            OnPropertyChanged(nameof(Input));
            Suggest();
            return false;
        }

        _recall = next;
        _input = _history[^(next + 1)];
        OnPropertyChanged(nameof(Input));
        Suggest();
        return true;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Cancel();
        _stop?.Dispose();
    }

    private async Task CopyAsync()
    {
        Drain();
        if (Copy is { } copy && Lines.Count > 0)
        {
            await copy(OutputText()).ConfigureAwait(true);
            Status = $"Copied {Lines.Count} line{(Lines.Count == 1 ? "" : "s")}.";
        }
    }

    // The text a command wrote, into lines: a newline ends the line, a carriage return goes back to its start (what follows
    // replaces it, as a console redraws a progress line), "\r\n" is a newline, and the console's escape sequences are left out.
    private void Feed(TerminalLineKind kind, StringBuilder text)
    {
        foreach (var memory in text.GetChunks())
        {
            var span = memory.Span;
            int i = 0;
            while (i < span.Length)
            {
                int stop = span[i..].IndexOfAny('\n', '\r', '\u001b');
                var piece = stop < 0 ? span[i..] : span.Slice(i, stop);
                Write(kind, piece);
                if (stop < 0)
                {
                    break;
                }

                i += stop;
                char c = span[i];
                if (c == '\n')
                {
                    if (_open is null)
                    {
                        Start(kind);                                             // an empty line
                    }

                    _open = null;
                    _carriage = false;
                    i++;
                }
                else if (c == '\r')
                {
                    if (i + 1 < span.Length && span[i + 1] == '\n')
                    {
                        i++;                                                    // "\r\n": the newline handles it
                    }
                    else
                    {
                        _carriage = true;
                        i++;
                    }
                }
                else
                {
                    i += Escape(span[i..]);
                }
            }
        }
    }

    // An escape sequence's length at the start of `span` (CSI: ESC [ ... a final letter), so it is skipped.
    private static int Escape(ReadOnlySpan<char> span)
    {
        if (span.Length < 2 || span[1] != '[')
        {
            return Math.Min(2, span.Length);
        }

        int at = 2;
        while (at < span.Length && !char.IsAsciiLetter(span[at]) && span[at] != '~')
        {
            at++;
        }

        return Math.Min(at + 1, span.Length);
    }

    private void Write(TerminalLineKind kind, ReadOnlySpan<char> piece)
    {
        if (piece.IsEmpty)
        {
            return;
        }

        if (_open is null || _open.Kind != kind && !_carriage)
        {
            Start(kind);
        }

        if (_carriage)
        {
            _open!.Text = piece.ToString();
            _carriage = false;
        }
        else
        {
            _open!.Text = string.Concat(_open.Text, piece);
        }
    }

    private void Start(TerminalLineKind kind)
    {
        _open = new TerminalLine("", kind);
        _added.Add(_open);
    }

    // The lines made this frame, added as one change; the oldest beyond the setting's most dropped as one.
    private void Flush()
    {
        if (_added.Count > 0)
        {
            Lines.AddRange(_added);
            _added.Clear();
            int over = Lines.Count - _options.Lines;
            if (over > 0)
            {
                Lines.RemoveFirst(over);
            }

            LinesAdded?.Invoke(this, EventArgs.Empty);
        }
    }

    // The command's last line ends with it.
    private void Finish()
    {
        _open = null;
        _carriage = false;
    }

    private void Append(params IReadOnlyList<TerminalLine> lines)
    {
        Finish();
        _added.AddRange(lines);
        Flush();
    }

    private void Remember(string line)
    {
        _recall = -1;
        if (_history.Count == 0 || _history[^1] != line)
        {
            _history.Add(line);
            if (_history.Count > _options.HistoryLimit)
            {
                _history.RemoveRange(0, _history.Count - _options.HistoryLimit);
            }

            _settings.Set(HistoryKey, string.Join('\n', _history));
            UpdateHistory();
        }
    }

    private void UpdateHistory() => Lists.Replace(History, [.. Enumerable.Reverse(_history).Take(_options.HistoryShown)], static (a, b) => a == b);

    // The words offered for the line: the commands that start with what is typed (all of them when nothing is), then, once a
    // command is typed, its options and the common ones not typed yet (the help topics and commands after "help").
    private void Suggest()
    {
        if (_runner is not { } runner)
        {
            return;
        }

        string typed = _input.TrimStart();
        bool spaced = typed.EndsWith(' ');
        string[] words = typed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length > 0 && words[0] == runner.Name)
        {
            words = words[1..];
        }

        string partial = spaced || words.Length == 0 ? "" : words[^1];
        string done = string.Join(' ', spaced ? words : words.SkipLast(words.Length == 0 ? 0 : 1));
        var command = runner.Commands.Where(c => done.StartsWith(c.Name, StringComparison.Ordinal) && (done.Length == c.Name.Length || done[c.Name.Length] == ' '))
            .MaxBy(c => c.Name.Length);
        var found = new List<TerminalSuggestion>();
        if (command is null && !done.StartsWith("help", StringComparison.Ordinal))
        {
            string prefix = string.Join(' ', words);
            foreach (var c in runner.Commands)
            {
                if (prefix.Length == 0 || c.Name.StartsWith(prefix, StringComparison.Ordinal) || c.Aliases.Any(a => a.StartsWith(prefix, StringComparison.Ordinal)))
                {
                    found.Add(Suggest(c));
                }
            }
        }
        else
        {
            var given = words.ToHashSet(StringComparer.Ordinal);
            IEnumerable<string> next = command is not null && command.Name != "help" ? command.Options.Concat(runner.CommonOptions)
                : runner.HelpTopics.Concat(runner.Commands.Select(c => c.Name));
            foreach (string word in next.Distinct(StringComparer.Ordinal))
            {
                if (!given.Contains(word) && word.StartsWith(partial, StringComparison.Ordinal))
                {
                    found.Add(new TerminalSuggestion(word, "", true, "", new Command(() => Type(word))));
                }
            }
        }

        Lists.Replace(Suggestions, [.. found.Take(_options.Suggestions)], static (a, b) => a.Text == b.Text && a.IsAvailable == b.IsAvailable);
    }

    private TerminalSuggestion Suggest(TerminalCommand command) =>
        new(command.Name, command.Summary, command.IsAvailable, command.Unavailable,
            new Command(() =>
            {
                if (command.IsAvailable)
                {
                    TypeCommand(command.Name);
                }
                else
                {
                    Status = command.Unavailable;
                }
            }));

    // A command picked: the line becomes it (and a space for what follows).
    private void TypeCommand(string name)
    {
        Input = name + " ";
        ShowCommands = false;
    }

    // A word picked: it takes the place of the word being typed.
    private void Type(string word)
    {
        string typed = _input;
        int cut = typed.EndsWith(' ') ? typed.Length : typed.LastIndexOf(' ') + 1;
        Input = typed[..cut] + word + " ";
    }
}

/// <summary>The Terminal page's settings (the app gives its own by registering one; null: the library's).</summary>
public sealed class TerminalLabOptions
{
    /// <summary>The folder commands run in; null: the app's storage.</summary>
    public string? WorkingFolder { get; set; }

    /// <summary>The most lines kept (the oldest dropped); 5000 by default.</summary>
    public int Lines { get; set; } = 5000;

    /// <summary>The most lines the history keeps; 200 by default.</summary>
    public int HistoryLimit { get; set; } = 200;

    /// <summary>The latest lines of the history shown as chips; 12 by default.</summary>
    public int HistoryShown { get; set; } = 12;

    /// <summary>The most suggestions shown at once; 48 by default.</summary>
    public int Suggestions { get; set; } = 48;
}
