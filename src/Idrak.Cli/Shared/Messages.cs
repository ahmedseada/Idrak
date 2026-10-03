// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Cli.Shared;

/// <summary>
/// The tool's messages in English (the default) or Arabic (<c>--lang ar</c>, the config's "lang", <c>IDRAK_LANG</c>).
/// The English text is the key, as written at the call (<c>T("No file {0}.", path)</c>); the Arabic catalog
/// (<see cref="Arabic"/>) maps it to its translation, and a message without one stays English. Translated: <c>idrak
/// help</c> (titles, command summaries, common options, the labels of a command's help), the dispatcher's usage errors,
/// <c>doctor</c>, <c>devices</c>, questions and Ctrl+C, and the chat's own lines. JSON, CSV and Markdown output are never
/// translated (the language is English for them), so scripts read the same keys and values in every language. A test
/// checks that every message written at a call and every command summary has its Arabic text, and that the catalog
/// holds nothing unused.
/// </summary>
/// <remarks>
/// Arabic needs a terminal that shows it right to left with joined letters; where the terminal does not, the output
/// goes through <see cref="VisualText"/> (<see cref="Begin"/> decides, <c>--lang-render</c> overrides).
/// </remarks>
internal static partial class Messages
{
    /// <summary>The languages <c>--lang</c> takes.</summary>
    public static readonly string[] Languages = ["en", "ar"];

    private static readonly AsyncLocal<string?> Current = new();

    /// <summary>The language of this run: "en" or "ar".</summary>
    public static string Language => Current.Value ?? "en";

    /// <summary>Whether this run's messages are Arabic.</summary>
    public static bool IsArabic => Language == "ar";

    /// <summary>
    /// <paramref name="english"/> in this run's language, with <paramref name="args"/> put in its {0}, {1}, ...
    /// placeholders (formatted with the invariant culture).
    /// </summary>
    public static string T(string english, params object?[] args)
    {
        if (!IsArabic || !Arabic.TryGetValue(english, out string? arabic))
        {
            return args.Length == 0 ? english : string.Format(CultureInfo.InvariantCulture, english, args);
        }

        string text = MarkedArabic.GetOrAdd(arabic, MarkLatin);
        return args.Length == 0 ? text : string.Format(CultureInfo.InvariantCulture, text, [.. args.Select(Isolate)]);
    }

    private const char Lrm = '‎';

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string> MarkedArabic = new(StringComparer.Ordinal);

    // An English word in Arabic text that starts or ends with punctuation ("--device", ".NET", "/help", "(Markdown")
    // would have that punctuation moved to its other side by the bidirectional algorithm, as the punctuation sits
    // between an Arabic letter and a Latin one; a left-to-right mark beside it keeps it with the word.
    private static string MarkLatin(string text)
    {
        var words = text.Split(' ');
        for (int i = 0; i < words.Length; i++)
        {
            string word = words[i];
            if (word.Length < 2 || !word.Any(char.IsAsciiLetterOrDigit) || word.Any(c => c is >= '؀' and <= 'ۿ'))
            {
                continue;
            }

            bool before = word[0] is '-' or '/' or '~' or '.' or '(' or '[' or '<' or '\'' or '"' or '@' or '$';
            bool after = word[^1] is ')' or ']' or '>' or '\'' or '"' or '/';
            words[i] = (before ? Lrm.ToString() : "") + word + (after ? Lrm.ToString() : "");
        }

        return string.Join(' ', words);
    }

    // A value put into Arabic text (a path, a device name, a size) that holds left-to-right text and no Arabic is kept
    // together left to right, its punctuation included, between left-to-right marks.
    private static object? Isolate(object? value)
    {
        string? text = value as string ?? (value is IFormattable formattable ? null : value?.ToString());
        if (text is null || text.Length == 0 || text.Any(c => c is >= '֐' and <= 'ࣿ') || !text.Any(char.IsAsciiLetterOrDigit))
        {
            return value;
        }

        return Lrm + text + Lrm;
    }

    /// <summary>A command's one-line summary in this run's language.</summary>
    public static string Summary(Command command) => T(command.Summary);

    /// <summary>Parses a language name: en or ar (also english, arabic, and regional forms such as ar-EG).</summary>
    public static string ParseLanguage(string text, string source)
    {
        string name = text.Trim().ToLowerInvariant();
        return name is "en" or "english" || name.StartsWith("en-", StringComparison.Ordinal) || name.StartsWith("en_", StringComparison.Ordinal) ? "en"
            : name is "ar" or "arabic" || name.StartsWith("ar-", StringComparison.Ordinal) || name.StartsWith("ar_", StringComparison.Ordinal) ? "ar"
            : throw new UsageException($"{source} takes en or ar, not '{text}'.");
    }

    /// <summary>
    /// Sets up the language and the rendering of one run of the tool, before anything is printed: takes
    /// <c>--lang</c> and <c>--lang-render</c> out of <paramref name="args"/>, resolves them with the config and the
    /// environment, and with Arabic text output on a terminal that does not reorder text itself, wraps
    /// <paramref name="output"/> and <paramref name="error"/> in <see cref="VisualWriter"/>s. Dispose the result when the
    /// run ends (it renders what is pending and restores the language of an outer run, as for <c>idrak shell</c>).
    /// </summary>
    public static IDisposable Begin(ref IReadOnlyList<string> args, ref TextWriter output, ref TextWriter error)
    {
        string? lang = null, render = null, config = null;
        bool data = false;
        var rest = new List<string>(args.Count);
        for (int i = 0; i < args.Count; i++)
        {
            string arg = args[i];
            if (arg == "--")
            {
                rest.AddRange(args.Skip(i));
                break;
            }

            string name = arg, value = "";
            int equals = arg.IndexOf('=');
            bool inline = equals > 0 && arg.StartsWith('-');
            if (inline)
            {
                (name, value) = (arg[..equals], arg[(equals + 1)..]);
            }

            if (name is "--lang" or "--lang-render")
            {
                if (!inline)
                {
                    value = i + 1 < args.Count ? args[++i] : throw new UsageException($"{name} needs a value.");
                }

                if (name == "--lang")
                {
                    lang = value;
                }
                else
                {
                    render = value;
                }

                continue;
            }

            if (name is "--config" or "-C")
            {
                config = inline ? value : i + 1 < args.Count ? args[i + 1] : null;
            }

            // JSON, CSV and Markdown output are data: never translated.
            data |= name is "--json" or "-j" || name == "--format" && (inline ? value : i + 1 < args.Count ? args[i + 1] : "") is not ("text" or "");
            rest.Add(arg);
        }

        args = rest;
        CliConfig? settings = null;
        try
        {
            settings = CliConfig.Load(config);
        }
        catch (Exception e) when (e is UsageException or IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            // A broken config is reported by the command itself.
        }

        // Variables saved by idrak env set apply before anything reads them (IDRAK_LANG below included).
        var saved = SavedEnvironment.Apply(settings);

        string language;
        (TextRendering Rendering, bool RightAlign) visual;
        try
        {
            language = lang is not null ? ParseLanguage(lang, "--lang")
                : Current.Value ?? (settings?.Get("lang") is { Length: > 0 } fromConfig ? ParseLanguage(fromConfig, "the config's \"lang\"")
                : Environment.GetEnvironmentVariable("IDRAK_LANG") is { Length: > 0 } fromEnvironment ? ParseLanguage(fromEnvironment, "IDRAK_LANG") : "en");
            visual = VisualText.Parse(render ?? settings?.Get("lang-render") ?? Environment.GetEnvironmentVariable("IDRAK_LANG_RENDER") ?? "auto",
                render is not null ? "--lang-render" : "lang-render");
        }
        catch
        {
            saved.Dispose();
            throw;
        }

        var (rendering, right) = visual;
        if (data)
        {
            language = "en";
        }

        var scope = new Scope(Current.Value, saved);
        Current.Value = language;
        if (language == "ar")
        {
            scope.UseUtf8(ref output, ref error);
            output = scope.Wrap(output, rendering, right);
            error = scope.Wrap(error, rendering, right);
        }

        return scope;
    }

    private sealed class Scope(string? previous, IDisposable saved) : IDisposable
    {
        private readonly List<VisualWriter> _writers = [];
        private System.Text.Encoding? _encoding;

        // Windows consoles start in a legacy code page (437, 1252, or 1256 on Arabic systems), none of which has the
        // Arabic presentation forms: the console writes UTF-8 for this run. Setting the encoding replaces Console.Out
        // and Console.Error, so writers that were the console's become the new ones.
        public void UseUtf8(ref TextWriter output, ref TextWriter error)
        {
            bool isOut = ReferenceEquals(output, Console.Out), isError = ReferenceEquals(error, Console.Error);
            if (!OperatingSystem.IsWindows() || !isOut && !isError || Console.OutputEncoding.CodePage == 65001)
            {
                return;
            }

            _encoding = Console.OutputEncoding;
            Console.OutputEncoding = new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
            output = isOut ? Console.Out : output;
            error = isError ? Console.Error : error;
        }

        public TextWriter Wrap(TextWriter writer, TextRendering requested, bool right)
        {
            if (writer is VisualWriter)
            {
                return writer;                                                   // an outer run already renders it
            }

            bool console = ReferenceEquals(writer, Console.Out) || ReferenceEquals(writer, Console.Error);
            bool terminal = Terminal.IsTerminal(writer);
            if (VisualText.Resolve(requested, terminal, Environment.GetEnvironmentVariable) != TextRendering.Visual)
            {
                return writer;
            }

            int width = 0;
            if (console && terminal)
            {
                try
                {
                    width = Console.WindowWidth;
                }
                catch (IOException)
                {
                    // No width known: lines are not wrapped or right-aligned.
                }
            }

            var visual = new VisualWriter(writer, width, right);
            _writers.Add(visual);
            return visual;
        }

        public void Dispose()
        {
            foreach (var writer in _writers)
            {
                writer.RenderPending();
            }

            if (_encoding is not null)
            {
                Console.OutputEncoding = _encoding;
            }

            Current.Value = previous;
            saved.Dispose();
        }
    }
}
