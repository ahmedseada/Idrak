// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak.Cli.Commands.Health;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands;

/// <summary><c>idrak version</c> (<c>--version</c>, <c>-V</c>): versions of the tool, the libraries, the runtime and the drivers.</summary>
internal sealed class VersionCommand : Command
{
    public override string Name => "version";

    public override string Summary => "Versions of the tool, the libraries, the .NET runtime and the device drivers";

    public override string Usage => """

        The drivers are those of every device found (each backend that found none says why).

        Examples:
          idrak version
          idrak -V
          idrak --version -j
        """;

    public override int Run(CommandContext context)
    {
        string tool = Machine.ToolVersion;
        var libraries = Machine.Libraries();
        var drivers = Drivers();
        string runtime = $"{RuntimeInformation.FrameworkDescription} on {RuntimeInformation.OSDescription} ({RuntimeInformation.ProcessArchitecture})";
        if (context.Format is OutputFormat.Csv or OutputFormat.Markdown)
        {
            context.Fields([("idrak", tool), .. libraries, (".NET", runtime), .. drivers], "Component", "Version");
        }
        else
        {
            context.Write($"idrak {tool}");
            foreach (var (name, version) in libraries)
            {
                context.Write($"{name} {version}");
            }

            context.Write(runtime);
            foreach (var (device, driver) in drivers)
            {
                context.Write($"{device}: {driver}");
            }
        }

        context.WriteJson(new JsonObject
        {
            ["tool"] = tool,
            ["library"] = Machine.Version(typeof(Tensor).Assembly),
            ["libraries"] = new JsonObject([.. libraries.Select(l => KeyValuePair.Create(l.Name, (JsonNode?)l.Version))]),
            ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["os"] = RuntimeInformation.OSDescription,
            ["architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
            ["drivers"] = new JsonObject([.. drivers.Select(d => KeyValuePair.Create(d.Device, (JsonNode?)d.Driver))]),
        });
        return ExitCodes.Ok;
    }

    /// <summary>Each GPU's driver (its name, which carries the driver, when the backend gives none apart), and each backend with none.</summary>
    internal static List<(string Device, string Driver)> Drivers()
    {
        var drivers = new List<(string, string)>();
        foreach (var d in DeviceListing.All().Where(d => d.Kind != "cpu"))
        {
            drivers.Add((d.Device, d.Error is not null ? $"cannot start: {d.Error}" : d.Driver ?? d.Name));
        }

        foreach (var b in DeviceListing.Backends.Where(b => b.Count == 0))
        {
            drivers.Add(($"{b.Kind}", $"none found: {b.UnavailableReason}"));
        }

        return drivers;
    }
}
