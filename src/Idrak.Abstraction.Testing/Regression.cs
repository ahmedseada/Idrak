// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Testing.Devices;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A case that once failed, as data: its kind ("operation" for a device's operation call, or a contract suite's
/// <see cref="ContractSuite{T}.Name"/>), its name, and everything needed to run it again
/// (<see cref="Regression"/>).
/// </summary>
/// <param name="Kind">"operation", or the name of the contract suite that runs it.</param>
/// <param name="Name">The case's name (the operation and the case it came from).</param>
/// <param name="Data">The case: inputs and, for an operation, the values the CPU wrote.</param>
public sealed record RegressionCase(string Kind, string Name, JsonObject Data);

/// <summary>
/// Saved regression cases: a failing case from a report (<see cref="ConformanceFailure.Repro"/>) written to a small JSON
/// file and replayed later, so the case that once broke an app stays one of its tests (and the library's, once the fix
/// moves there).
/// </summary>
public static class Regression
{
    /// <summary>The kind of a saved operation call of a device.</summary>
    public const string OperationKind = "operation";

    private const string Format = "idrak-regression-case";

    /// <summary>Writes <paramref name="case"/> to <paramref name="path"/> (creating its folder); returns the path.</summary>
    public static string Save(string path, RegressionCase @case)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(@case);
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        var json = new JsonObject
        {
            ["format"] = Format,
            ["version"] = 1,
            ["kind"] = @case.Kind,
            ["name"] = @case.Name,
            ["data"] = @case.Data.DeepClone(),
        };
        File.WriteAllText(path, json.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
        return path;
    }

    /// <summary>Reads a case written by <see cref="Save"/>.</summary>
    /// <exception cref="InvalidDataException">The file is not a saved case.</exception>
    public static RegressionCase Load(string path)
    {
        var json = JsonNode.Parse(File.ReadAllText(path)) as JsonObject;
        if (json is null || (string?)json["format"] != Format || json["data"] is not JsonObject data)
        {
            throw new InvalidDataException($"{path} is not a saved regression case (\"format\": \"{Format}\").");
        }

        return new RegressionCase((string?)json["kind"] ?? "", (string?)json["name"] ?? Path.GetFileNameWithoutExtension(path), (JsonObject)data.DeepClone());
    }

    /// <summary>The case in a file, or every case (*.json) in a folder and its subfolders, in name order.</summary>
    public static IReadOnlyList<RegressionCase> LoadAll(string path)
    {
        if (File.Exists(path))
        {
            return [Load(path)];
        }

        return Directory.Exists(path)
            ? [.. Directory.EnumerateFiles(path, "*.json", SearchOption.AllDirectories).Order(StringComparer.Ordinal).Select(Load)]
            : throw new FileNotFoundException($"No saved regression case at {path}.", path);
    }

    /// <summary>
    /// Runs the saved operation calls at <paramref name="path"/> (a file or a folder) on <paramref name="device"/> and
    /// compares what they write with what the CPU wrote when they were saved. Cases of other kinds are left out.
    /// </summary>
    public static ConformanceReport Replay(string path, Device device, DeviceCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        return Replay(path, device.Backend, $"{device} ({device.Name})", options);
    }

    /// <summary>The same as <see cref="Replay(string, Device, DeviceCheckOptions?)"/> for a backend that is not registered as a device.</summary>
    public static ConformanceReport Replay(string path, Backend backend, DeviceCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return Replay(path, backend, $"{backend.Kind} ({backend.Name})", options);
    }

    /// <summary>
    /// Runs the saved cases of <paramref name="suite"/> at <paramref name="path"/> (a file or a folder) on
    /// <paramref name="implementation"/>, compared with the library default as the suite compares. Cases of other kinds
    /// are left out.
    /// </summary>
    public static ConformanceReport Replay<T>(string path, T implementation, ContractSuite<T> suite)
        where T : class
    {
        ArgumentNullException.ThrowIfNull(suite);
        var cases = LoadAll(path).Where(c => c.Kind == suite.Name).Select(c => new ContractCase(c.Name, c.Data));
        return suite.Check(implementation, cases.ToList(), $"saved cases from {path}");
    }

    // A file name for a case: its kind and name with anything but letters and digits as '-', and a short hash.
    internal static string FileName(RegressionCase @case)
    {
        var name = new StringBuilder();
        foreach (char c in $"{@case.Kind}-{@case.Name}")
        {
            name.Append(char.IsAsciiLetterOrDigit(c) ? char.ToLowerInvariant(c) : '-');
        }

        string text = name.ToString();
        while (text.Contains("--", StringComparison.Ordinal))
        {
            text = text.Replace("--", "-", StringComparison.Ordinal);
        }

        string hash = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(@case.Data.ToJsonString())))[..8];
        return $"{text.Trim('-')[..Math.Min(60, text.Trim('-').Length)]}-{hash}.json";
    }

    private static ConformanceReport Replay(string path, Backend backend, string subject, DeviceCheckOptions? options)
    {
        var calls = LoadAll(path).Where(c => c.Kind == OperationKind).Select(OperationCases.FromCase).ToList();
        return DeviceConformance.Replay(backend, $"{subject}, saved cases from {path}", calls, options ?? new DeviceCheckOptions(), [], calls.Count);
    }
}
