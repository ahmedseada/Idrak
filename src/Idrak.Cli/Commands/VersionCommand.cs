// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands;

/// <summary><c>idrak version</c>: versions of the tool, the library and the runtime.</summary>
internal sealed class VersionCommand : Command
{
    public override string Name => "version";

    public override string Summary => "Versions of the tool, the library and the .NET runtime";

    public override int Run(CommandContext context)
    {
        string Version(Assembly a) => a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? a.GetName().Version?.ToString() ?? "?";
        string tool = Version(typeof(VersionCommand).Assembly), library = Version(typeof(Tensor).Assembly);
        context.Write($"idrak {tool}");
        context.Write($"Idrak library {library}");
        context.Write($"{RuntimeInformation.FrameworkDescription} on {RuntimeInformation.OSDescription} ({RuntimeInformation.ProcessArchitecture})");
        context.WriteJson(new JsonObject
        {
            ["tool"] = tool,
            ["library"] = library,
            ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["os"] = RuntimeInformation.OSDescription,
            ["architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
        });
        return ExitCodes.Ok;
    }
}
