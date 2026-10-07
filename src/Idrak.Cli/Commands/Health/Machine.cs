// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Health;

/// <summary>What the health commands report about the machine: versions, the runtime, devices and their drivers.</summary>
internal static class Machine
{
    /// <summary>An assembly's informational version (without the source-link suffix), or its version.</summary>
    public static string Version(Assembly assembly)
    {
        string text = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? assembly.GetName().Version?.ToString() ?? "?";
        int plus = text.IndexOf('+');
        return plus > 0 ? text[..plus] : text;
    }

    /// <summary>The tool's version.</summary>
    public static string ToolVersion => Version(typeof(Machine).Assembly);

    /// <summary>The Idrak libraries the tool runs with, by assembly name.</summary>
    public static IReadOnlyList<(string Name, string Version)> Libraries() =>
    [
        .. new[] { typeof(Idrak.Layers.Sequential).Assembly, typeof(Tensor).Assembly, typeof(Nlp.ChatTranscriptEncoder).Assembly, typeof(Datasets.Downloader).Assembly }
            .Concat(AppDomain.CurrentDomain.GetAssemblies().Where(a => a.GetName().Name?.StartsWith("Idrak.", StringComparison.Ordinal) == true
                && a != typeof(Machine).Assembly && !a.IsDynamic))
            .DistinctBy(a => a.GetName().Name)
            .Select(a => (a.GetName().Name ?? "?", Version(a))),
    ];

    /// <summary>Memory the process may use (the machine's, or a container's limit).</summary>
    public static long Memory => GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;

    /// <summary>The machine as JSON: OS, architecture, processors, memory, runtime.</summary>
    public static JsonObject Json() => new()
    {
        ["os"] = RuntimeInformation.OSDescription,
        ["architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
        ["processors"] = Environment.ProcessorCount,
        ["memoryBytes"] = Memory,
        ["runtime"] = RuntimeInformation.FrameworkDescription,
    };

    /// <summary>One line: OS, architecture, processors, memory, runtime.</summary>
    public static string Describe() =>
        $"{RuntimeInformation.OSDescription} ({RuntimeInformation.ProcessArchitecture}), {Environment.ProcessorCount} logical processors, " +
        $"{Units.Bytes(Memory)} memory, {RuntimeInformation.FrameworkDescription}";

    /// <summary>A device as JSON.</summary>
    public static JsonObject Json(DeviceInfo d) => new()
    {
        ["device"] = d.Device,
        ["backend"] = d.Backend,
        ["name"] = d.Name,
        ["memoryBytes"] = d.MemoryBytes,
        ["computeUnits"] = d.ComputeUnits,
        ["subgroupSize"] = d.SubgroupSize,
        ["matrixUnits"] = d.MatrixUnits,
        ["kernelWidth"] = d.KernelWidth,
        ["driver"] = d.Driver,
        ["kind"] = d.HardwareKind,
        ["listed"] = d.Listed,
        ["default"] = d.IsDefault,
        ["note"] = d.Note,
        ["error"] = d.Error,
    };

    /// <summary>A backend as JSON.</summary>
    public static JsonObject Json(BackendInfo b) => new()
    {
        ["kind"] = b.Kind,
        ["backend"] = b.Display,
        ["count"] = b.Count,
        ["unavailable"] = b.UnavailableReason,
    };

    /// <summary>"16.0 GB", or "-" when unknown.</summary>
    public static string Bytes(long? value) => value is long v ? Units.Bytes(v) : "-";

    /// <summary>A number, or "-" when unknown.</summary>
    public static string Number(int? value) => value is int v ? v.ToString(CultureInfo.InvariantCulture) : "-";

    /// <summary>The devices the command looks at: all of them, or only <c>--device</c> when given.</summary>
    public static IReadOnlyList<DeviceInfo> Devices(CommandContext context) =>
        context.Option("--device") is not null ? [DeviceListing.Describe(context.Device)] : DeviceListing.All();
}
