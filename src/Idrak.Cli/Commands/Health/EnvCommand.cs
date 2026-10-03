// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak env</c>: every environment variable Idrak reads that is set, with its value, default and meaning; <c>--all</c>
/// adds the unset ones. <c>idrak help env</c> (or <c>help environment</c>) prints the reference without values.
/// </summary>
internal sealed class EnvCommand : Command
{
    public override string Name => "env";

    public override IReadOnlyCollection<string> Aliases => ["environment"];

    public override string Summary => "Every environment variable Idrak reads: value, default and meaning";

    public override string Usage => $"""
        [--all] [NAME...]

        Arguments:
          NAME  only these variables (e.g. IDRAK_CACHE)

        Options:
              --all  also the variables that are not set

        Tokens and keys are shown as set or not set, never their values. From says where a value comes from: the
        terminal, or saved (idrak env set saves variables for every idrak run, in any terminal; idrak env unset
        removes them).

        Examples:
          idrak env
          idrak env --all -j
          idrak env IDRAK_CACHE VK_ICD_FILENAMES
          idrak env set

        Every variable:
        {EnvironmentVariables.Reference()}
        """;

    public override IReadOnlyCollection<string> Flags => ["--all"];

    public override int Run(CommandContext context)
    {
        var unknown = context.Positional.Where(n => EnvironmentVariables.Find(n) is null).ToList();
        if (unknown.Count > 0)
        {
            throw new UsageException($"Not a variable Idrak reads: {string.Join(", ", unknown)} (idrak help env lists them).");
        }

        bool all = context.Flag("--all") || context.Positional.Count > 0;
        var chosen = EnvironmentVariables.All.Where(v => context.Positional.Count == 0 || context.Positional.Contains(v.Name))
            .Select(v => (Variable: v, Value: EnvironmentVariables.Value(v)))
            .Where(v => all || v.Value is not null).ToList();

        if (chosen.Count == 0)
        {
            context.Write("No variable Idrak reads is set (idrak env --all lists them with their defaults).");
        }
        else
        {
            context.Table(["Variable", "Value", "From", "Default", "Meaning"],
                chosen.Select(v => (IReadOnlyList<string>)[v.Variable.Name, Short(v.Value) ?? (v.Variable.Secret ? "not set" : "-"), Source(v.Variable, v.Value) ?? "-",
                    v.Variable.Default, v.Variable.Meaning]));
        }

        if (!all)
        {
            int unset = EnvironmentVariables.All.Count - chosen.Count;
            context.Write($"{unset} more not set; idrak env --all lists them.");
        }

        context.WriteJson(new JsonObject
        {
            ["variables"] = new JsonArray([.. chosen.Select(v => (JsonNode)new JsonObject
            {
                ["name"] = v.Variable.Name,
                ["group"] = v.Variable.Group,
                ["set"] = v.Value is not null,
                ["source"] = Source(v.Variable, v.Value),
                ["value"] = v.Variable.Secret ? null : v.Value,
                ["secret"] = v.Variable.Secret,
                ["default"] = v.Variable.Default,
                ["meaning"] = v.Variable.Meaning,
            })]),
        });
        return ExitCodes.Ok;
    }

    // Where a set value comes from: saved by idrak env set, or the terminal.
    private static string? Source(EnvironmentVariables.Variable variable, string? value) =>
        value is null ? null : SavedEnvironment.Applied.Contains(variable.Name) ? "saved" : "terminal";

    // Long values (PATH, NO_PROXY) cut in the table; --json has them whole.
    private static string? Short(string? value) => value is { Length: > 60 } ? value[..57] + "..." : value;
}
