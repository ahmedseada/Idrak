// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Shared;

/// <summary>Colours for <see cref="Terminal.Paint"/>.</summary>
internal enum Colour
{
    Red = 31,
    Green = 32,
    Yellow = 33,
    Blue = 34,
    Cyan = 36,
    Dim = 2,
    Bold = 1,
}

/// <summary>
/// Terminal helpers every command shares (plans/idrak-cli.md, "Helpers every command shares"): whether output goes to a
/// terminal, colour (only on a terminal, never with <c>NO_COLOR</c> or <c>--json</c>), questions and confirmations
/// (<c>--yes</c> skips them; without a terminal a question is an error that names <c>--yes</c>), <c>--dry-run</c>, and
/// Ctrl+C handling (<see cref="Interrupt"/>).
/// </summary>
internal static class Terminal
{
    /// <summary>The flags of a command that asks before deleting or overwriting: <c>--yes</c> and <c>--dry-run</c>.</summary>
    public static readonly string[] ConfirmFlags = ["--yes", "--dry-run"];

    /// <summary>The short form of <c>--yes</c> (plans/idrak-cli.md: <c>-y</c>, the same in every command).</summary>
    public static readonly IReadOnlyDictionary<string, string> ConfirmShortForms = new Dictionary<string, string> { ["-y"] = "--yes" };

    /// <summary>Tests: answers questions as a terminal would, reading from this reader (null: the console decides).</summary>
    internal static TextReader? TestInput { get; set; }

    /// <summary>Whether <paramref name="writer"/> is the console's output or error stream and that stream is a terminal.</summary>
    public static bool IsTerminal(TextWriter writer) =>
        ReferenceEquals(writer, Console.Out) ? !Console.IsOutputRedirected
        : ReferenceEquals(writer, Console.Error) && !Console.IsErrorRedirected;

    /// <summary>Whether questions can be asked: the command's output is a terminal and the input is the keyboard.</summary>
    public static bool IsInteractive(CommandContext context) => TestInput is not null || IsTerminal(context.ConsoleOutput) && !Console.IsInputRedirected;

    /// <summary>Whether <c>NO_COLOR</c> asks for no colour (set and not empty, as no-color.org defines it).</summary>
    public static bool NoColour => Environment.GetEnvironmentVariable("NO_COLOR") is { Length: > 0 };

    /// <summary>
    /// Whether text on <paramref name="writer"/> may be coloured: never with <c>NO_COLOR</c>, <c>--json</c> or
    /// <c>--color never</c>; always with <c>--color always</c>; otherwise only on a terminal (not <c>TERM=dumb</c>).
    /// </summary>
    public static bool UseColour(CommandContext context, TextWriter? writer = null) =>
        !context.Json && !NoColour && context.ColorMode switch
        {
            "never" => false,
            "always" => true,
            _ => Environment.GetEnvironmentVariable("TERM") != "dumb" && IsTerminal(writer ?? context.Output),
        };

    /// <summary><paramref name="unicode"/>, or <paramref name="plain"/> with <c>--plain</c> (no Unicode box or progress characters).</summary>
    public static string Text(CommandContext context, string unicode, string plain) => context.Plain ? plain : unicode;

    /// <summary><paramref name="text"/> in <paramref name="colour"/> when <see cref="UseColour"/> allows it, else unchanged.</summary>
    public static string Paint(CommandContext context, string text, Colour colour, TextWriter? writer = null) =>
        UseColour(context, writer) ? $"\u001b[{(int)colour}m{text}\u001b[0m" : text;

    /// <summary>Whether the command runs with <c>--dry-run</c> (say what would happen, change nothing).</summary>
    public static bool DryRun(CommandContext context) => context.Flag("--dry-run");

    /// <summary>
    /// Asks <paramref name="question"/> with a yes/no answer (no by default). True at once with <c>--yes</c>; without a
    /// terminal, a usage error that names <c>--yes</c>.
    /// </summary>
    public static bool Confirm(CommandContext context, string question)
    {
        if (context.Flag("--yes"))
        {
            return true;
        }

        string answer = Ask(context, $"{question} [y/N]", "", requireTerminal: "--yes");
        return answer.Equals("y", StringComparison.OrdinalIgnoreCase) || answer.Equals("yes", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Asks <paramref name="question"/> and returns the answer, or <paramref name="fallback"/> for an empty one. Without a
    /// terminal, a usage error naming <paramref name="requireTerminal"/> (the option that answers instead).
    /// </summary>
    public static string Ask(CommandContext context, string question, string fallback, string requireTerminal = "--yes")
    {
        if (!IsInteractive(context))
        {
            throw new UsageException($"'{question}' needs an answer, and there is no terminal to ask on; add {requireTerminal}.");
        }

        context.ConsoleOutput.Write(fallback.Length > 0 ? $"{question} [{fallback}] " : $"{question} ");
        context.ConsoleOutput.Flush();
        string? line = (TestInput ?? Console.In).ReadLine();
        return string.IsNullOrWhiteSpace(line) ? fallback : line.Trim();
    }

    /// <summary>
    /// Reads a secret (a token): the first line of piped input (<c>echo $TOKEN | idrak login hf</c>), or typed at a
    /// prompt without echo. Never printed. Without a terminal or piped input, a usage error saying how to pipe it.
    /// </summary>
    public static string AskSecret(CommandContext context, string prompt)
    {
        string? secret;
        if (TestInput is not null || Console.IsInputRedirected)
        {
            secret = (TestInput ?? Console.In).ReadLine();
        }
        else if (IsTerminal(context.ConsoleOutput))
        {
            context.ConsoleOutput.Write($"{prompt}: ");
            context.ConsoleOutput.Flush();
            var typed = new System.Text.StringBuilder();
            while (Console.ReadKey(intercept: true) is var key && key.Key != ConsoleKey.Enter)
            {
                if (key.Key == ConsoleKey.Backspace)
                {
                    typed.Length = Math.Max(0, typed.Length - 1);
                }
                else if (!char.IsControl(key.KeyChar))
                {
                    typed.Append(key.KeyChar);
                }
            }

            context.ConsoleOutput.WriteLine();
            secret = typed.ToString();
        }
        else
        {
            throw new UsageException($"{prompt}: there is no terminal to ask on; pipe it in instead (e.g. echo \"$TOKEN\" | idrak {context.Command.Name} ...).");
        }

        return string.IsNullOrWhiteSpace(secret) ? throw new UsageException($"{prompt}: nothing was given.") : secret.Trim();
    }
}

/// <summary>
/// Ctrl+C handling for long commands: the first Ctrl+C cancels <see cref="Token"/> (the command stops cleanly after the
/// current token, chunk or request) instead of ending the process; a second one ends the process as usual. Dispose to
/// restore the default.
/// </summary>
/// <example><code>
/// using var interrupt = new Interrupt(context);
/// foreach (var token in generator.Stream(prompt, interrupt.Token)) { ... }
/// if (interrupt.Requested) context.Write("stopped");
/// </code></example>
internal sealed class Interrupt : IDisposable
{
    private readonly CancellationTokenSource _source = new();
    private readonly CommandContext? _context;

    public Interrupt(CommandContext? context = null)
    {
        _context = context;
        Console.CancelKeyPress += OnCancel;
    }

    /// <summary>Cancelled at the first Ctrl+C.</summary>
    public CancellationToken Token => _source.Token;

    /// <summary>Whether Ctrl+C was pressed.</summary>
    public bool Requested => _source.IsCancellationRequested;

    /// <summary>Cancels as Ctrl+C would (tests, or a command that stops itself the same way).</summary>
    public void Request() => _source.Cancel();

    public void Dispose()
    {
        Console.CancelKeyPress -= OnCancel;
        _source.Dispose();
    }

    private void OnCancel(object? sender, ConsoleCancelEventArgs e)
    {
        if (_source.IsCancellationRequested)
        {
            return;                                                              // the second Ctrl+C ends the process
        }

        e.Cancel = true;
        _context?.ErrorOutput.WriteLine("Stopping (Ctrl+C again to end at once)...");
        _source.Cancel();
    }
}
