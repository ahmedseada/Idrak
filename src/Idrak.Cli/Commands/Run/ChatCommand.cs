// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;

namespace Idrak.Cli.Commands.Run;

/// <summary><c>idrak chat MODEL</c> (<c>c</c>): an interactive chat, streamed, with slash commands.</summary>
internal sealed class ChatCommand : Command
{
    public override string Name => "chat";

    public override IReadOnlyCollection<string> Aliases => ["c"];

    public override string Summary => "Interactive chat with a model, streamed with tokens per second";

    public override string Usage => """
        MODEL [options]

        MODEL is a Hugging Face id, a model folder, a .gguf file or an alias. Type a message and press Enter; end a line
        with \ to continue it on the next. An empty input (Ctrl+D) or /exit ends the chat; Ctrl+C stops an answer.

        Options:
          -w, --weights FORMAT   int8, int4, bf16 or a registered packed format (default: as stored)
          -k, --kv FORMAT        KV cache format: float32, int8, bfloat16 or a registered one (default float32)
              --context N        context window in tokens (default 4096, at most the model's)
              --adapter DIR      merge a LoRA adapter into the weights as they are read
              --history FILE     load the conversation from FILE if it exists, and save it there after every answer
              --file FILE        add a text file's content to the first message (repeatable; /file adds more later)
              --mcp SERVER       let the model call an MCP server's tools: an http(s) URL, or a command that starts
                                 the server ("npx -y some-server"); repeatable
        """ + "\n" + GenerationSettings.Help + """

        Slash commands:
        """ + "\n" + ChatSession.CommandHelp + """

        Examples:
          idrak chat Qwen/Qwen3-0.6B
          idrak c qwen -d vulkan:0 -s "Answer briefly." --history talk.json
          idrak chat ./model.gguf -w int8 -k int8 --temperature 0 --think
          idrak chat qwen --file notes.md --file todo.txt
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. Models.ValueOptions, .. GenerationSettings.ValueOptions, "--history", "--file", "--mcp"];

    public override IReadOnlyCollection<string> Flags => GenerationSettings.Flags;

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(Models.ShortForms.Concat(GenerationSettings.ShortForms));

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        if (context.Positional.Count > 1)
        {
            throw new UsageException($"chat takes one model; for a single answer use: idrak run {name} \"{string.Join(' ', context.Positional.Skip(1))}\"");
        }

        var settings = GenerationSettings.From(context);
        var files = context.Options("--file");
        foreach (string file in files.Where(f => !File.Exists(f)))
        {
            throw new UsageException($"File not found: {file}");
        }

        using var loaded = LoadedChat.Load(context, name, settings);
        var session = new ChatSession(context, loaded, settings, context.Option("--history"));
        foreach (string file in files)
        {
            session.Attach(file);
        }

        return session.Run(StandardInput.Reader, interactive: !StandardInput.IsRedirected);
    }
}

/// <summary>A chat: the conversation, the settings and the slash commands, reading lines from any reader (tests drive it).</summary>
internal sealed class ChatSession
{
    /// <summary>The slash commands, for help.</summary>
    public const string CommandHelp =
        "  /help                 these commands\n" +
        "  /system [TEXT]        show or set the system prompt (/system none removes it)\n" +
        "  /reset                start over (the system prompt stays)\n" +
        "  /save FILE            save the conversation (chat JSON); /load FILE: continue a saved one\n" +
        "  /file FILE            add a text file's content to the next message\n" +
        "  /stats                speed, tokens, context used and memory\n" +
        "  /think on|off|default reasoning mode\n" +
        "  /tools                the tools the model may call\n" +
        "  /set [NAME VALUE]     show or change temperature, top-k, top-p, max-tokens, seed, think\n" +
        "  /copy                 copy the last answer (to the clipboard on a terminal that allows it)\n" +
        "  /retry                answer the last message again\n" +
        "  /exit                 end the chat (also /quit, or Ctrl+D)\n";

    private readonly CommandContext _context;
    private readonly LoadedChat _loaded;
    private readonly GenerationSettings _settings;
    private readonly string? _historyPath;
    private readonly ChatResponder _responder;
    private readonly List<ChatMessage> _messages = [];
    private readonly List<ChatAnswer> _answers = [];
    private readonly List<string> _attachments = [];
    private readonly bool _color;
    private CancellationTokenSource? _cancel;

    public ChatSession(CommandContext context, LoadedChat loaded, GenerationSettings settings, string? historyPath)
    {
        _context = context;
        _loaded = loaded;
        _settings = settings;
        _historyPath = historyPath;
        _color = Terminal.UseColour(context);
        var stream = context.Json ? null : context.Output;
        _responder = new ChatResponder(loaded.Chat, loaded.Registry) { Stream = stream, ThinkingStream = context.Quiet ? null : stream, Color = _color };
        if (historyPath is not null && File.Exists(historyPath))
        {
            _messages.AddRange(ChatHistory.Load(historyPath));
            if (_messages.FirstOrDefault(m => m.Role == "system") is { } system && settings.System is null)
            {
                settings.System = system.Content;
            }
        }

        ApplySystem();
    }

    /// <summary>Adds a text file's content to the next message.</summary>
    public void Attach(string path) =>
        _attachments.Add($"File {Path.GetFileName(path)}:\n```\n{File.ReadAllText(path).TrimEnd()}\n```");

    /// <summary>The conversation so far.</summary>
    public IReadOnlyList<ChatMessage> Messages => _messages;

    /// <summary>Reads lines from <paramref name="input"/> until it ends or /exit; returns the exit code.</summary>
    public int Run(TextReader input, bool interactive)
    {
        if (interactive)
        {
            _context.Write($"Chatting with {_loaded.Describe(_context)}. /help lists the commands, /exit ends.");
            if (_messages.Count(m => m.Role != "system") is > 0 and int count)
            {
                _context.Write($"Continuing {count} messages from {_historyPath}.");
            }

            Console.CancelKeyPress += OnCancel;
        }

        try
        {
            while (true)
            {
                if (interactive && !_context.Json)
                {
                    _context.Output.Write(_color ? "\u001b[1m> \u001b[0m" : "> ");
                }

                if (ReadMessage(input) is not { } line)
                {
                    break;
                }

                if (line.Length == 0)
                {
                    continue;
                }

                if (line.StartsWith('/') && !line.StartsWith("//", StringComparison.Ordinal))
                {
                    if (!Command(line))
                    {
                        break;
                    }

                    continue;
                }

                string message = line.StartsWith("//", StringComparison.Ordinal) ? line[1..] : line;
                if (_attachments.Count > 0)
                {
                    message = string.Join("\n\n", _attachments) + "\n\n" + message;
                    _attachments.Clear();
                }

                _messages.Add(new ChatMessage("user", message));
                Answer();
            }
        }
        finally
        {
            if (interactive)
            {
                Console.CancelKeyPress -= OnCancel;
            }
        }

        _context.WriteJson(Json());
        return ExitCodes.Ok;
    }

    // A message: one line, or several joined while each ends with a backslash.
    private static string? ReadMessage(TextReader input)
    {
        if (input.ReadLine() is not { } line)
        {
            return null;
        }

        var text = new StringBuilder();
        while (line.EndsWith('\\'))
        {
            text.Append(line, 0, line.Length - 1).Append('\n');
            if (input.ReadLine() is not { } next)
            {
                return text.ToString().Trim();
            }

            line = next;
        }

        return text.Append(line).ToString().Trim();
    }

    private void OnCancel(object? sender, ConsoleCancelEventArgs e)
    {
        if (_cancel is { IsCancellationRequested: false } cancel)
        {
            e.Cancel = true;                                  // stop the answer, not the chat
            cancel.Cancel();
        }
    }

    // Generates the answer to the conversation as it is.
    private void Answer()
    {
        using var cancel = _cancel = new CancellationTokenSource();
        try
        {
            var answer = _responder.Answer(_messages, _settings, _loaded.Context, cancel.Token);
            _answers.Add(answer);
            _context.Write(Dim(answer.Summary(_loaded.Context)));
            if (answer.Message.ToolCalls is { Count: > 0 } && _loaded.Registry is null)
            {
                _context.Write("(the model asked for tools; give them with --tools FILE.dll)");
            }
        }
        catch (OperationCanceledException)
        {
            // Drop the unanswered turn so the conversation stays well formed.
            int last = _messages.FindLastIndex(m => m.Role == "user");
            _messages.RemoveRange(last, _messages.Count - last);
            _context.Write("\n[stopped]");
        }
        finally
        {
            _cancel = null;
        }

        SaveHistory();
    }

    private void SaveHistory()
    {
        if (_historyPath is not null)
        {
            ChatHistory.Save(_historyPath, _messages, _loaded.Registry?.Definitions, _settings.Think);
        }
    }

    private void ApplySystem()
    {
        _messages.RemoveAll(m => m.Role == "system");
        if (_settings.System is { Length: > 0 } system)
        {
            _messages.Insert(0, new ChatMessage("system", system));
        }
    }

    // Runs a slash command; false ends the chat.
    private bool Command(string line)
    {
        int space = line.IndexOf(' ');
        string command = (space < 0 ? line : line[..space]).ToLowerInvariant(), argument = space < 0 ? "" : line[(space + 1)..].Trim();
        switch (command)
        {
            case "/exit" or "/quit" or "/bye":
                return false;
            case "/help" or "/?":
                Say("Commands:\n" + CommandHelp.TrimEnd() + "\nA message starting with / is sent as typed when it starts with //.");
                break;
            case "/system" when argument.Length == 0:
                Say(_settings.System is { } current ? $"System prompt: {current}" : "No system prompt (set one with /system TEXT).");
                break;
            case "/system":
                _settings.System = argument is "none" or "off" ? null : argument;
                ApplySystem();
                Say(_settings.System is null ? "System prompt removed." : "System prompt set.");
                break;
            case "/reset" or "/clear":
                _messages.Clear();
                _answers.Clear();
                ApplySystem();
                SaveHistory();
                Say("Conversation cleared.");
                break;
            case "/save" when argument.Length > 0:
                ChatHistory.Save(argument, _messages, _loaded.Registry?.Definitions, _settings.Think);
                Say($"Saved {_messages.Count} messages to {argument}.");
                break;
            case "/load" when argument.Length > 0:
                if (!File.Exists(argument))
                {
                    Say($"No file {argument}.");
                    break;
                }

                _messages.Clear();
                _messages.AddRange(ChatHistory.Load(argument));
                _settings.System = _messages.FirstOrDefault(m => m.Role == "system")?.Content;
                Say($"Loaded {_messages.Count} messages from {argument}.");
                break;
            case "/file" when argument.Length > 0:
                if (!File.Exists(argument))
                {
                    Say($"No file {argument}.");
                    break;
                }

                Attach(argument);
                Say($"{argument} will be sent with the next message.");
                break;
            case "/save" or "/load" or "/file":
                Say($"{command} needs a file name, e.g. {command} talk.json");
                break;
            case "/stats":
                Say(Stats());
                break;
            case "/think":
                Say(_settings.Set("think", argument.Length == 0 ? "on" : argument.ToLowerInvariant()) ?? $"Reasoning: {_settings.Describe().First(d => d.Name == "think").Value}.");
                break;
            case "/tools":
                Say(_loaded.Tools.Count == 0 ? "No tools; give an assembly with --tools FILE.dll."
                    : string.Join('\n', _loaded.Tools.Select(t => $"  {t.Definition.Name}: {t.Definition.Description}")));
                break;
            case "/set" when argument.Length == 0:
                Say(string.Join('\n', _settings.Describe().Select(d => $"  {d.Name} = {d.Value}")));
                break;
            case "/set":
                var parts = argument.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                string setting = parts[0].ToLowerInvariant().Replace('_', '-');
                Say(parts.Length < 2 ? $"/set {setting} needs a value, e.g. /set temperature 0.7"
                    : _settings.Set(setting, parts[1]) ?? $"{setting} = {_settings.Describe().FirstOrDefault(d => d.Name == setting).Value}");
                break;
            case "/copy":
                Copy();
                break;
            case "/retry":
                int lastUser = _messages.FindLastIndex(m => m.Role == "user");
                if (lastUser < 0)
                {
                    Say("Nothing to retry yet.");
                    break;
                }

                _messages.RemoveRange(lastUser + 1, _messages.Count - lastUser - 1);
                Answer();
                break;
            default:
                Say($"Unknown command {command}; /help lists them.");
                break;
        }

        return true;
    }

    private void Copy()
    {
        if (_messages.LastOrDefault(m => m.Role == "assistant") is not { } last)
        {
            Say("No answer to copy yet.");
            return;
        }

        if (_color)
        {
            // OSC 52: terminals that allow it put the text on the clipboard (no clipboard library needed).
            _context.Output.Write($"\u001b]52;c;{Convert.ToBase64String(Encoding.UTF8.GetBytes(last.Content))}\u0007");
            Say($"Copied the last answer ({last.Content.Length} characters) to the clipboard.");
        }
        else
        {
            Say(last.Content);
        }
    }

    private string Stats()
    {
        var text = new StringBuilder();
        var invariant = CultureInfo.InvariantCulture;
        int turns = _answers.Count, generated = _answers.Sum(a => a.GeneratedTokens);
        double seconds = _answers.Sum(a => a.GenerationSeconds);
        text.Append(invariant, $"Model     {_loaded.Describe(_context)}\n");
        if (_answers.LastOrDefault() is { } last)
        {
            text.Append(invariant, $"Last      {last.GeneratedTokens} tokens at {last.TokensPerSecond:F1} tokens/s; prompt {last.PromptTokens} tokens in {last.PromptSeconds:F2} s\n");
            text.Append(invariant, $"Context   {Math.Min(last.PromptTokens + last.GeneratedTokens, _loaded.Context)} of {_loaded.Context} tokens used\n");
        }

        text.Append(invariant, $"Session   {turns} answers, {generated} tokens, {(seconds > 0 ? generated / seconds : 0):F1} tokens/s on average, {_messages.Count} messages\n");
        using var process = Process.GetCurrentProcess();
        text.Append(invariant, $"Memory    process {process.WorkingSet64 / (1 << 20)} MiB, managed {GC.GetTotalMemory(false) / (1 << 20)} MiB");
        return text.ToString();
    }

    private JsonObject Json() => new()
    {
        ["model"] = _loaded.Choice.Model,
        ["device"] = _context.Device.ToString(),
        ["context"] = _loaded.Context,
        ["settings"] = new JsonObject(_settings.Describe().Select(d => KeyValuePair.Create(d.Name, (JsonNode?)d.Value))),
        ["messages"] = ChatJson.Messages(_messages),
        ["answers"] = new JsonArray([.. _answers.Select(a => (JsonNode)a.Figures())]),
    };

    // Command replies: on the output, even in quiet mode (they answer what was typed), but not as JSON.
    private void Say(string text)
    {
        if (!_context.Json)
        {
            _context.Output.WriteLine(text);
        }
    }

    private string Dim(string text) => _color ? $"\u001b[2m{text}\u001b[0m" : text;
}
