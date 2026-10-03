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
internal sealed class CommandContext
{
    // Options every command accepts; the value-taking ones are listed for the parser.
    internal static readonly string[] CommonValueOptions = ["--device", "--plugin", "--cache", "--config"];
    internal static readonly string[] CommonFlags = ["--json", "--quiet", "--verbose", "--help", "-h"];

    // Indented, and without escaping characters such as '+' that are safe outside HTML.
    internal static readonly JsonSerializerOptions JsonOutput = new() { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private readonly Dictionary<string, List<string>> _options;
    private readonly HashSet<string> _flags;
    private Device? _device;

    internal CommandContext(Command command, IReadOnlyList<string> positional, Dictionary<string, List<string>> options, HashSet<string> flags,
        TextWriter output, TextWriter error)
    {
        Command = command;
        Positional = positional;
        _options = options;
        _flags = flags;
        Output = output;
        ErrorOutput = error;
        Config = CliConfig.Load(Option("--config"));
    }

    public Command Command { get; }

    /// <summary>The arguments that are not options, in order.</summary>
    public IReadOnlyList<string> Positional { get; }

    public TextWriter Output { get; }

    public TextWriter ErrorOutput { get; }

    public CliConfig Config { get; }

    public bool Json => Flag("--json");

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
        if (!Quiet && !Json)
        {
            Output.WriteLine(line);
        }
    }

    /// <summary>A line only with <c>--verbose</c>.</summary>
    public void Detail(string line)
    {
        if (Verbose && !Json)
        {
            Output.WriteLine(line);
        }
    }

    /// <summary>An error line on the error output (always printed).</summary>
    public void Error(string line) => ErrorOutput.WriteLine(line);

    /// <summary>A table with aligned columns (text mode).</summary>
    public void Table(IReadOnlyList<string> headers, IEnumerable<IReadOnlyList<string>> rows)
    {
        if (Quiet || Json)
        {
            return;
        }

        var all = rows.ToList();
        var widths = headers.Select((h, i) => Math.Max(h.Length, all.Count == 0 ? 0 : all.Max(r => i < r.Count ? r[i].Length : 0))).ToArray();
        string Line(IReadOnlyList<string> cells) => string.Join("  ", cells.Select((c, i) => i == cells.Count - 1 ? c : c.PadRight(widths[i]))).TrimEnd();
        Output.WriteLine(Line(headers));
        Output.WriteLine(string.Join("  ", widths.Select(w => new string('-', w))));
        foreach (var row in all)
        {
            Output.WriteLine(Line(row));
        }
    }

    /// <summary>The command's JSON result, printed only with <c>--json</c> (one document per command).</summary>
    public void WriteJson(JsonNode node)
    {
        if (Json)
        {
            Output.WriteLine(node.ToJsonString(JsonOutput));
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
            values = JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? throw new UsageException($"{path} is not a JSON object.");
        }

        return new CliConfig(path, values);
    }

    public IEnumerable<KeyValuePair<string, JsonNode?>> All => _values;

    public string? Get(string key) => _values[key] is JsonValue v && v.TryGetValue(out string? s) ? s : _values[key]?.ToJsonString();

    public IEnumerable<string> List(string key) => _values[key] is JsonArray a ? a.Select(n => n?.GetValue<string>()).OfType<string>() : [];

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

        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(Path))!);
        File.WriteAllText(Path, _values.ToJsonString(CommandContext.JsonOutput));
    }
}
