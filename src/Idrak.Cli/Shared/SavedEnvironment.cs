// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Cli.Shared;

/// <summary>
/// Environment variables saved in the config's "env" object (by <c>idrak env set</c>): set for each run of the tool
/// before any command reads them, so they apply the same way in every terminal and shell, and pass on to the processes
/// the tool starts (<c>idrak test</c>'s test runner, the agent's commands). A variable set in the environment itself
/// wins, so a one-off value in one terminal still overrides the saved one. The active profile's "env" values win over
/// the top-level ones.
/// </summary>
internal static class SavedEnvironment
{
    /// <summary>The config key holding the saved variables (NAME: value).</summary>
    public const string Key = "env";

    // Variables that cannot be saved: the config's own path (read before the config), and the ones the terminal, the
    // system or the .NET runtime set or read before the tool starts, where a value set during the run does nothing.
    private static readonly HashSet<string> NotSaveable = new(StringComparer.Ordinal)
    {
        "IDRAK_CONFIG", "PATH", "PATHEXT", "PROCESSOR_IDENTIFIER", "TERM", "TERM_PROGRAM", "VTE_VERSION", "KONSOLE_VERSION", "MLTERM",
        "WT_SESSION", "CI", "DOTNET_GCHeapHardLimit",
    };

    /// <summary>The variables applied from the config in this run (the environment did not set them).</summary>
    public static IReadOnlySet<string> Applied => _applied;

    private static HashSet<string> _applied = new(StringComparer.Ordinal);

    /// <summary>Why <paramref name="variable"/> cannot be saved, or null when it can.</summary>
    public static string? WhyNot(EnvironmentVariables.Variable variable) =>
        variable.Secret ? $"{variable.Name} is a token or key; idrak login stores tokens without printing them"
        : NotSaveable.Contains(variable.Name) ? $"{variable.Name} is read before idrak starts (or set by the terminal), so a saved value would do nothing"
        : variable.Group == EnvironmentVariables.ChildGroup ? $"{variable.Name} is set by the tool itself for the commands the coding tools run"
        : null;

    /// <summary>The saved variables (top level, then the active profile's over them).</summary>
    public static IReadOnlyDictionary<string, string> Read(CliConfig? config)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        if (config is null)
        {
            return values;
        }

        var sources = new List<JsonObject?> { config.Root[Key] as JsonObject };
        if (config.Profile is { } profile)
        {
            sources.Add(config.Root["profiles"]?[profile]?[Key] as JsonObject);
        }

        foreach (var source in sources.OfType<JsonObject>())
        {
            foreach (var (name, node) in source)
            {
                if (node is JsonValue value && value.TryGetValue(out string? text))
                {
                    values[name] = text;
                }
                else if (node is not null)
                {
                    values[name] = node.ToJsonString();
                }
            }
        }

        return values;
    }

    /// <summary>
    /// Sets the saved variables the environment does not set; disposing the result unsets them again (the tests run many
    /// commands in one process).
    /// </summary>
    public static IDisposable Apply(CliConfig? config)
    {
        var applied = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, value) in Read(config))
        {
            if (Environment.GetEnvironmentVariable(name) is not { Length: > 0 } && value.Length > 0 && name != "IDRAK_CONFIG")
            {
                Environment.SetEnvironmentVariable(name, value);
                applied.Add(name);
            }
        }

        var previous = _applied;
        _applied = applied;
        return new Undo(applied, previous);
    }

    private sealed class Undo(HashSet<string> applied, HashSet<string> previous) : IDisposable
    {
        public void Dispose()
        {
            foreach (string name in applied)
            {
                Environment.SetEnvironmentVariable(name, null);
            }

            _applied = previous;
        }
    }
}
