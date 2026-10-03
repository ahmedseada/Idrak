// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak alias set NAME MODEL [-w F] [-k F]</c>: a short name for a model and its settings, kept in the config's
/// "aliases" object ({"qwen": {"model": "Qwen/Qwen3-0.6B", "weights": "int8", "kv": "int8"}}) that every model argument
/// reads (Shared/Models.cs).
/// </summary>
internal sealed class AliasSetCommand : Command
{
    public override string Name => "alias set";

    public override string Summary => "Give a model (and its weight and KV formats) a short name, kept in the config";

    public override string Usage => """
        NAME MODEL [-w FORMAT] [-k FORMAT]

          NAME               the short name (letters, digits, '.', '-', '_')
          MODEL              a Hugging Face id, a folder, a .gguf file
          -w, --weights F    the weight format to load it with (int8, int4, bf16 or a registered packed format)
          -k, --kv F         the KV cache format (int8, bfloat16 or a registered one)

        Options given to a command override the alias's (idrak chat qwen -w int4).

        Examples:
          idrak alias set qwen Qwen/Qwen3-0.6B -w int8 -k int8
          idrak c qwen

        Environment: IDRAK_CONFIG (the config file)
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--weights", "--kv"];

    public override IReadOnlyDictionary<string, string> ShortForms => Shared.Models.ShortForms;

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "NAME"), model = context.Argument(1, "MODEL");
        if (context.Positional.Count > 2)
        {
            throw new UsageException($"alias set takes NAME and MODEL; '{context.Positional[2]}' is extra.");
        }

        if (name.Length == 0 || !name.All(c => char.IsAsciiLetterOrDigit(c) || c is '.' or '-' or '_') || name.All(c => c == '.'))
        {
            throw new UsageException($"'{name}' cannot be an alias: use letters, digits, '.', '-' and '_' (a '/' would read as a Hugging Face id).");
        }

        var aliases = (JsonObject?)context.Config.Object("aliases")?.DeepClone() ?? [];
        bool replaced = aliases.ContainsKey(name);
        var entry = new JsonObject { ["model"] = model };
        if (context.Option("--weights") is { } weights)
        {
            entry["weights"] = weights;
        }

        if (context.Option("--kv") is { } kv)
        {
            entry["kv"] = kv;
        }

        aliases[name] = entry;
        context.Config.SetNode("aliases", aliases);
        context.Write($"{(replaced ? "Changed" : "Added")} alias {name} → {AliasListCommand.Describe(entry)} in {context.Config.Path}");
        context.WriteJson(new JsonObject { ["name"] = name, ["alias"] = entry.DeepClone(), ["replaced"] = replaced, ["config"] = context.Config.Path });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak alias list</c>: the aliases in the config.</summary>
internal sealed class AliasListCommand : Command
{
    public override string Name => "alias list";

    public override IReadOnlyCollection<string> Aliases => ["alias", "alias ls"];

    public override string Summary => "The model aliases in the config";

    public override string Usage => """

        Examples:
          idrak alias list
          idrak alias --json

        Environment: IDRAK_CONFIG (the config file)
        """;

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"Unknown alias command '{context.Positional[0]}'; use alias set, alias list or alias rm.");
        }

        var aliases = context.Config.Object("aliases") ?? [];
        if (aliases.Count == 0)
        {
            context.Write($"No aliases in {context.Config.Path}. Add one: idrak alias set qwen Qwen/Qwen3-0.6B -w int8");
        }
        else
        {
            context.Table(["Alias", "Model", "Weights", "KV"], aliases.Select(a => (IReadOnlyList<string>)[a.Key, (string?)a.Value?["model"] ?? "?",
                (string?)a.Value?["weights"] ?? "-", (string?)a.Value?["kv"] ?? "-"]));
        }

        context.WriteJson(new JsonObject { ["config"] = context.Config.Path, ["aliases"] = aliases.DeepClone() });
        return ExitCodes.Ok;
    }

    internal static string Describe(JsonObject alias) =>
        (string?)alias["model"] + ((string?)alias["weights"] is { } w ? $" -w {w}" : "") + ((string?)alias["kv"] is { } k ? $" -k {k}" : "");
}

/// <summary><c>idrak alias rm NAME</c>: removes an alias from the config.</summary>
internal sealed class AliasRmCommand : Command
{
    public override string Name => "alias rm";

    public override string Summary => "Remove a model alias from the config";

    public override string Usage => """
        NAME...

        Examples:
          idrak alias rm qwen

        Environment: IDRAK_CONFIG (the config file)
        """;

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Missing NAME (an alias; idrak alias list shows them).");
        }

        var aliases = (JsonObject?)context.Config.Object("aliases")?.DeepClone() ?? [];
        var missing = context.Positional.Where(n => !aliases.ContainsKey(n)).ToList();
        if (missing.Count > 0)
        {
            throw new InvalidOperationException($"No alias named {string.Join(", ", missing)}; idrak alias list shows them.");
        }

        foreach (string name in context.Positional)
        {
            aliases.Remove(name);
        }

        context.Config.SetNode("aliases", aliases.Count == 0 ? null : aliases);
        context.Write($"Removed {string.Join(", ", context.Positional)} from {context.Config.Path}");
        context.WriteJson(new JsonObject { ["removed"] = new JsonArray([.. context.Positional.Select(n => (JsonNode?)JsonValue.Create(n))]), ["config"] = context.Config.Path });
        return ExitCodes.Ok;
    }
}
