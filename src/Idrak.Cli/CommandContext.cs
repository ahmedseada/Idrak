// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Cli;

/// <summary>
/// What a command runs with: its parsed options and positional arguments, the common options (device, plug-ins, JSON
/// output, verbosity, cache folder), the config file, and output helpers that print text or, with <c>--json</c>, one
/// JSON document.
/// </summary>
internal sealed class CommandContext : IDisposable
{
    // Options every command accepts; the value-taking ones are listed for the parser.
    internal static readonly string[] CommonValueOptions =
        ["--device", "--plugin", "--cache", "--config", "--log", "--threads", "--seed", "--format", "--output", "--color", "--timeout"];
    internal static readonly string[] CommonFlags = ["--json", "--quiet", "--verbose", "--help", "--offline", "--plain"];

    /// <summary>Short forms every command accepts (plans/idrak-cli.md, "Short forms").</summary>
    internal static readonly IReadOnlyDictionary<string, string> CommonShortForms = new Dictionary<string, string>
    {
        ["-h"] = "--help", ["-d"] = "--device", ["-j"] = "--json", ["-q"] = "--quiet", ["-v"] = "--verbose",
        ["-P"] = "--plugin", ["-C"] = "--config", ["-O"] = "--output",
    };

    // Indented, and without escaping characters such as '+' that are safe outside HTML.
    internal static readonly JsonSerializerOptions JsonOutput = new() { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private readonly Dictionary<string, List<string>> _options;
    private readonly HashSet<string> _flags;
    private Device? _device;
    private TextWriter? _log;
    private TextWriter? _outputFile;
    private CancellationTokenSource? _timeoutSource;

    internal CommandContext(Command command, IReadOnlyList<string> positional, Dictionary<string, List<string>> options, HashSet<string> flags,
        TextWriter output, TextWriter error)
    {
        Command = command;
        Positional = positional;
        _options = options;
        _flags = flags;
        Output = output;
        ConsoleOutput = output;
        ErrorOutput = error;
        Config = CliConfig.Load(Option("--config"));
        // A token stored by idrak login github: the library reads GITHUB_TOKEN, so it is passed on for this process.
        if (Environment.GetEnvironmentVariable("GITHUB_TOKEN") is null && Config.Root["tokens"]?["github"] is JsonValue github && github.TryGetValue(out string? token))
        {
            Environment.SetEnvironmentVariable("GITHUB_TOKEN", token);
        }

        Format = ParseFormat(Option("--format") ?? (flags.Contains("--json") ? "json" : "text"));
        Timeout = Option("--timeout") is { } timeout ? ParseDuration(timeout, "--timeout") : null;
        if (Option("--color") is { } color && color is not ("auto" or "always" or "never"))
        {
            throw new UsageException($"--color takes auto, always or never, not '{color}'.");
        }

        if (Option("--threads") is not null)
        {
            int threads = IntOption("--threads", 0);
            ComputeResources.MaxCpuThreads = threads > 0 ? threads : throw new UsageException("--threads needs a number above 0.");
        }

        if (Option("--output") is { } file)
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(file))!);
            _outputFile = new StreamWriter(file, append: false) { AutoFlush = true };
            Output = _outputFile;
        }
        if (Option("--log") is { } log)
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(log))!);
            _log = new StreamWriter(log, append: false) { AutoFlush = true };
        }
    }

    public Command Command { get; }

    /// <summary>The arguments that are not options, in order.</summary>
    public IReadOnlyList<string> Positional { get; }

    /// <summary>Where the command's main output goes: the console, or the file <c>--output</c> (<c>-O</c>) names.</summary>
    public TextWriter Output { get; }

    /// <summary>The output the tool was started with (the console), even with <c>--output</c>: for questions and progress.</summary>
    public TextWriter ConsoleOutput { get; }

    public TextWriter ErrorOutput { get; }

    public CliConfig Config { get; }

    public bool Json => Format == OutputFormat.Json;

    /// <summary>The output format: <c>--format text|json|csv|md</c> (<c>--json</c> is the short way to json).</summary>
    public OutputFormat Format { get; }

    /// <summary><c>--offline</c>: use only what is cached; a download is an error naming the missing file.</summary>
    public bool Offline => Flag("--offline");

    /// <summary><c>--plain</c>: no Unicode box or progress characters (screen readers, old terminals, logs).</summary>
    public bool Plain => Flag("--plain");

    /// <summary><c>--seed N</c>: one seed for sampling, shuffling and initialization; null when not given.</summary>
    public int? Seed => Option("--seed") is null ? null : IntOption("--seed", 0);

    /// <summary><c>--threads N</c> (already applied to <see cref="ComputeResources.MaxCpuThreads"/>); null when not given.</summary>
    public int? Threads => Option("--threads") is null ? null : IntOption("--threads", 0);

    /// <summary><c>--color auto|always|never</c> (default auto; <c>NO_COLOR</c> still wins).</summary>
    public string ColorMode => Option("--color") ?? "auto";

    /// <summary><c>--timeout DURATION</c> ("30s", "5m", "1h30m", "250ms", or seconds); null when not given.</summary>
    public TimeSpan? Timeout { get; }

    /// <summary>Cancelled when <c>--timeout</c> runs out (never without it): pass it to downloads, server calls and long runs.</summary>
    public CancellationToken TimeoutToken => Timeout is { } t ? (_timeoutSource ??= new CancellationTokenSource(t)).Token : CancellationToken.None;

    /// <summary>A duration: "30s", "5m", "1h30m", "250ms", "2d", a number of seconds, or hh:mm:ss.</summary>
    public static TimeSpan ParseDuration(string text, string option)
    {
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds) && seconds >= 0)
        {
            return TimeSpan.FromSeconds(seconds);
        }

        var match = System.Text.RegularExpressions.Regex.Match(text, @"^(?:(\d+(?:\.\d+)?)d)?(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+)ms)?$");
        if (text.Length > 0 && match.Success)
        {
            double Part(int i) => match.Groups[i].Success ? double.Parse(match.Groups[i].Value, CultureInfo.InvariantCulture) : 0;
            return TimeSpan.FromDays(Part(1)) + TimeSpan.FromHours(Part(2)) + TimeSpan.FromMinutes(Part(3)) + TimeSpan.FromSeconds(Part(4)) + TimeSpan.FromMilliseconds(Part(5));
        }

        return TimeSpan.TryParse(text, CultureInfo.InvariantCulture, out var span) && span >= TimeSpan.Zero ? span
            : throw new UsageException($"{option} needs a duration such as 30s, 5m or 1h30m, not '{text}'.");
    }

    private static OutputFormat ParseFormat(string text) => text switch
    {
        "text" => OutputFormat.Text,
        "json" => OutputFormat.Json,
        "csv" => OutputFormat.Csv,
        "md" or "markdown" => OutputFormat.Markdown,
        _ => throw new UsageException($"--format takes text, json, csv or md, not '{text}'."),
    };

    public bool Quiet => Flag("--quiet");

    public bool Verbose => Flag("--verbose");

    /// <summary>The cache folder: <c>--cache</c>, the config, <c>IDRAK_CACHE</c>, or <c>~/.cache/idrak</c>.</summary>
    public string CacheFolder => Option("--cache") ?? Config.Get("cache") ?? Environment.GetEnvironmentVariable("IDRAK_CACHE")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");

    /// <summary>The device from <c>--device</c> or the config, else <see cref="Device.Default"/>.</summary>
    public Device Device => _device ??= (Option("--device") ?? Config.Get("device")) is { } name ? Device.Parse(name) : Device.Default;

    /// <summary>Whether <paramref name="flag"/> (e.g. "--int8") was given.</summary>
    public bool Flag(string flag) => _flags.Contains(flag);

    /// <summary>The last value given for <paramref name="option"/>, or null.</summary>
    public string? Option(string option) => _options.TryGetValue(option, out var values) && values.Count > 0 ? values[^1] : null;

    /// <summary>Every value given for a repeatable option.</summary>
    public IReadOnlyList<string> Options(string option) => _options.TryGetValue(option, out var values) ? values : [];

    /// <summary>An integer option, or <paramref name="fallback"/> when absent.</summary>
    public int IntOption(string option, int fallback) => Option(option) is not { } text ? fallback
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) ? value
        : throw new UsageException($"{option} needs a whole number, not '{text}'.");

    /// <summary>The positional argument at <paramref name="index"/>, or a usage error naming it.</summary>
    public string Argument(int index, string name) => index < Positional.Count ? Positional[index] : throw new UsageException($"Missing {name}.");

    /// <summary>A line of text (not printed with <c>--quiet</c> or <c>--json</c>).</summary>
    public void Write(string line)
    {
        Log(line);
        if (!Quiet && Format is OutputFormat.Text or OutputFormat.Markdown)
        {
            Output.WriteLine(line);
        }
    }

    /// <summary>A line only with <c>--verbose</c>.</summary>
    public void Detail(string line)
    {
        Log(line);
        if (Verbose && !Json)
        {
            Output.WriteLine(line);
        }
    }

    /// <summary>An error line on the error output (always printed).</summary>
    public void Error(string line)
    {
        Log(line);
        ErrorOutput.WriteLine(line);
    }

    /// <summary>
    /// A line for the <c>--log FILE</c> file only (which also receives every <see cref="Write"/>, <see cref="Detail"/>,
    /// <see cref="Error"/>, <see cref="Table"/> and <see cref="WriteJson"/> line, whatever <c>--quiet</c> and
    /// <c>--verbose</c> say); nothing without <c>--log</c>.
    /// </summary>
    public void Log(string line) => _log?.WriteLine(line);

    /// <summary>Closes the <c>--log</c> file.</summary>
    public void Dispose()
    {
        _log?.Dispose();
        _log = null;
        _outputFile?.Dispose();
        _outputFile = null;
        _timeoutSource?.Dispose();
    }

    /// <summary>A table with aligned columns (text mode).</summary>
    public void Table(IReadOnlyList<string> headers, IEnumerable<IReadOnlyList<string>> rows)
    {
        if ((Quiet || Json) && _log is null)
        {
            return;
        }

        var all = rows.ToList();
        if (Format is OutputFormat.Csv or OutputFormat.Markdown)
        {
            string Csv(string cell) => cell.IndexOfAny([',', '"', '\n']) >= 0 ? $"\"{cell.Replace("\"", "\"\"", StringComparison.Ordinal)}\"" : cell;
            string Md(string cell) => cell.Replace("|", "\\|", StringComparison.Ordinal);
            var lines = Format == OutputFormat.Csv
                ? all.Prepend(headers).Select(r => string.Join(',', r.Select(Csv)))
                : all.Select(r => $"| {string.Join(" | ", r.Select(Md))} |").Prepend($"|{string.Concat(headers.Select(_ => "---|"))}").Prepend($"| {string.Join(" | ", headers.Select(Md))} |");
            foreach (string line in lines)
            {
                Log(line);
                if (!Quiet)
                {
                    Output.WriteLine(line);
                }
            }

            return;
        }

        var widths = headers.Select((h, i) => Math.Max(h.Length, all.Count == 0 ? 0 : all.Max(r => i < r.Count ? r[i].Length : 0))).ToArray();
        string Line(IReadOnlyList<string> cells) => string.Join("  ", cells.Select((c, i) => i == cells.Count - 1 ? c : c.PadRight(widths[i]))).TrimEnd();
        foreach (string line in all.Select(Line).Prepend(string.Join("  ", widths.Select(w => new string('-', w)))).Prepend(Line(headers)))
        {
            Log(line);
            if (!Quiet && !Json)
            {
                Output.WriteLine(line);
            }
        }
    }

    /// <summary>The command's JSON result, printed only with <c>--json</c> (one document per command).</summary>
    public void WriteJson(JsonNode node)
    {
        if (Json)
        {
            Output.WriteLine(node.ToJsonString(JsonOutput));
        }

        if (_log is not null)
        {
            Log(node.ToJsonString(JsonOutput));
        }
    }

    /// <summary>
    /// Loads the assemblies given with <c>--plugin</c> (and in the config's "plugins" list): each one's module
    /// initializer runs, then any public static parameterless <c>RegisterIdrakPlugin()</c> method on its types.
    /// </summary>
    internal void LoadPlugins()
    {
        foreach (string path in Options("--plugin").Concat(Config.List("plugins")))
        {
            string full = Path.GetFullPath(path);
            if (!File.Exists(full))
            {
                throw new UsageException($"Plug-in not found: {full}");
            }

            var assembly = Assembly.LoadFrom(full);
            RuntimeHelpers.RunModuleConstructor(assembly.ManifestModule.ModuleHandle);
            foreach (var type in assembly.GetExportedTypes())
            {
                type.GetMethod("RegisterIdrakPlugin", BindingFlags.Public | BindingFlags.Static, Type.EmptyTypes)?.Invoke(null, null);
            }

            Detail($"plug-in loaded: {full}");
        }
    }
}

/// <summary>The output formats of <c>--format</c>.</summary>
internal enum OutputFormat
{
    Text,
    Json,
    Csv,
    Markdown,
}

/// <summary>The config file (<c>~/.idrak/config.json</c>, <c>IDRAK_CONFIG</c> or <c>--config</c>): defaults for the common options.</summary>
internal sealed class CliConfig
{
    private readonly JsonObject _values;

    private CliConfig(string path, JsonObject values)
    {
        Path = path;
        _values = values;
    }

    public string Path { get; }

    public static string DefaultPath => Environment.GetEnvironmentVariable("IDRAK_CONFIG")
        ?? System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".idrak", "config.json");

    public static CliConfig Load(string? path)
    {
        path ??= DefaultPath;
        JsonObject values = [];
        if (File.Exists(path))
        {
            values = JsonNode.Parse(File.ReadAllText(path), documentOptions: ReadOptions) as JsonObject ?? throw new UsageException($"{path} is not a JSON object.");
        }

        return new CliConfig(path, values);
    }

    /// <summary>How config files are read: comments and trailing commas are allowed (a tune.json written by <c>idrak tune init</c> has comments).</summary>
    internal static readonly JsonDocumentOptions ReadOptions = new() { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };

    public IEnumerable<KeyValuePair<string, JsonNode?>> All => _values;

    /// <summary>The whole file as JSON (the "profiles" object included).</summary>
    public JsonObject Root => _values;

    /// <summary>
    /// The active profile (<c>IDRAK_PROFILE</c>, else the file's "profile" value), or null: a named set of values under
    /// "profiles" (e.g. "phone", "desktop") that <see cref="Get"/>, <see cref="Object"/> and <see cref="List"/> read
    /// before the file's top-level values.
    /// </summary>
    public string? Profile => Environment.GetEnvironmentVariable("IDRAK_PROFILE") is { Length: > 0 } name ? name
        : _values["profile"] is JsonValue v && v.TryGetValue(out string? s) ? s : null;

    // A value: the active profile's, else the top-level one.
    private JsonNode? Value(string key) => (Profile is { } p && _values["profiles"]?[p] is JsonObject profile ? profile[key] : null) ?? _values[key];

    public string? Get(string key) => Value(key) is JsonValue v && v.TryGetValue(out string? s) ? s : Value(key)?.ToJsonString();

    /// <summary>A JSON object value (e.g. "aliases"), or null.</summary>
    public JsonObject? Object(string key) => Value(key) as JsonObject;

    /// <summary>Saves the file after changes made through <see cref="Root"/>.</summary>
    public void SaveChanges() => Save();

    /// <summary>Sets a JSON value (null removes it) and saves the file.</summary>
    public void SetNode(string key, JsonNode? value)
    {
        if (value is null)
        {
            _values.Remove(key);
        }
        else
        {
            _values[key] = value;
        }

        Save();
    }

    public IEnumerable<string> List(string key) => Value(key) is JsonArray a ? a.Select(n => n?.GetValue<string>()).OfType<string>() : [];

    public void Set(string key, string? value)
    {
        if (value is null)
        {
            _values.Remove(key);
        }
        else
        {
            _values[key] = value;
        }

        Save();
    }

    private void Save()
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(Path))!);
        File.WriteAllText(Path, _values.ToJsonString(CommandContext.JsonOutput));
    }
}
