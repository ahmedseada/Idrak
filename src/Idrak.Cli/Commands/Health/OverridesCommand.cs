// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak overrides</c>: every registry entry an app (here, the <c>--plugin</c> assemblies) registered over the library's
/// defaults or beside them: the startup report of the override loop (<see cref="Overrides.Report"/>).
/// </summary>
internal sealed class OverridesCommand : Command
{
    public override string Name => "overrides";

    public override string Summary => "What the plug-ins override: each registry entry, where it comes from and its failure policy";

    public override string Usage => """

        Lists the registry entries the --plugin assemblies (and the config's "plugins") registered: the library default
        each one replaces (or "added" for a new name), the assembly it comes from and its failure policy: "fall back"
        (the default: a call that throws is answered by the library's default and reported), "throw", "shadow N%" (the
        library answers, the plug-in's runs beside it on N% of the calls and is compared), or none for registries whose
        entries cannot fall back. Telemetry of fallbacks and comparisons: idrak trace --levels overrides -- COMMAND.

        Examples:
          idrak overrides -P ./MySampler.dll
          idrak overrides -P ./MySampler.dll -j
        """;

    public override int Run(CommandContext context)
    {
        var report = Overrides.Report();
        if (report.Count == 0)
        {
            context.Write("No overrides: every registry runs the library's defaults.");
        }
        else
        {
            context.Table(["Registry", "Name", "Replaces", "Implementation", "Origin", "Policy"], report.Select(o => (IReadOnlyList<string>)
                [o.Registry, o.Name, o.ReplacesDefault ? "library default" : "added", o.Implementation, o.Origin, o.PolicyText]));
        }

        context.WriteJson(new JsonArray([.. report.Select(o => (JsonNode)new JsonObject
        {
            ["registry"] = o.Registry,
            ["name"] = o.Name,
            ["replaces_default"] = o.ReplacesDefault,
            ["implementation"] = o.Implementation,
            ["origin"] = o.Origin,
            ["guarded"] = o.Guarded,
            ["policy"] = o.Guarded && o.ReplacesDefault ? o.Policy.ToString() : null,
            ["shadow_rate"] = o.Guarded && o.Policy == SlotPolicy.Shadow ? o.ShadowRate : null,
            ["fallbacks"] = o.FallBacks,
            ["compared"] = o.Compared,
            ["differed"] = o.Differed,
        })]));
        return ExitCodes.Ok;
    }
}
