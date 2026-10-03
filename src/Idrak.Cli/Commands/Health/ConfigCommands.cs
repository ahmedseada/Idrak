// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// The config commands (<c>config get/set/unset/list</c>): keys are dotted paths ("device", "aliases.qwen.model"); with
/// <c>--profile NAME</c> they read and write the named profile under "profiles" (the profile in use is
/// <c>IDRAK_PROFILE</c> or the file's "profile" value).
/// </summary>
internal abstract class ConfigCommand : Command
{
    public override IReadOnlyCollection<string> ValueOptions => ["--profile"];

    /// <summary>The object keys are read from and written to: the file, or the profile <c>--profile</c> names.</summary>
    protected static JsonObject Scope(CommandContext context, bool create)
    {
        var root = context.Config.Root;
        if (context.Option("--profile") is not { } profile)
        {
            return root;
        }

        if (root["profiles"] is not JsonObject profiles)
        {
            profiles = [];
            if (create)
            {
                root["profiles"] = profiles;
            }
        }

        if (profiles[profile] is not JsonObject scope)
        {
            scope = [];
            if (create)
            {
                profiles[profile] = scope;
            }
        }

        return scope;
    }

    /// <summary>The node at a dotted <paramref name="key"/> under <paramref name="scope"/>, or null.</summary>
    protected static JsonNode? Get(JsonObject scope, string key)
    {
        JsonNode? node = scope;
        foreach (string part in key.Split('.'))
        {
            node = node is JsonObject o ? o[part] : null;
        }

        return node;
    }

    /// <summary>The object holding the last part of <paramref name="key"/> (created along the way when asked), and that part.</summary>
    protected static (JsonObject? Parent, string Last) Parent(JsonObject scope, string key, bool create)
    {
        var parts = key.Split('.');
        if (parts.Any(p => p.Length == 0))
        {
            throw new UsageException($"'{key}' is not a key: use names joined by dots, such as aliases.qwen.model.");
        }

        JsonObject? current = scope;
        foreach (string part in parts[..^1])
        {
            if (current![part] is JsonObject next)
            {
                current = next;
            }
            else if (create)
            {
                current[part] = next = [];
                current = next;
            }
            else
            {
                return (null, parts[^1]);
            }
        }

        return (current, parts[^1]);
    }

    /// <summary>A value as text: strings as they are, everything else as JSON.</summary>
    protected static string Text(JsonNode? node) => node is JsonValue v && v.TryGetValue(out string? s) ? s : node?.ToJsonString() ?? "";

    /// <summary>Whether a key holds a token or key (shown only as "set", never printed).</summary>
    internal static bool IsSecret(string key) => key.Split('.').Any(p => p.Contains("token", StringComparison.OrdinalIgnoreCase)
        || p.Contains("key", StringComparison.OrdinalIgnoreCase) || p.Contains("password", StringComparison.OrdinalIgnoreCase));

    /// <summary>The value as listings show it: "set" for a secret.</summary>
    protected static string Shown(string key, JsonNode? node) => IsSecret(key) && node is not null ? "set" : Text(node);

    /// <summary>The value as JSON output shows it: "set" for a secret (objects of secrets too).</summary>
    protected static JsonNode? ShownJson(string key, JsonNode? node) => IsSecret(key) && node is not null ? "set"
        : node is JsonObject o ? new JsonObject([.. o.Select(p => KeyValuePair.Create(p.Key, ShownJson($"{key}.{p.Key}", p.Value)))])
        : node?.DeepClone();

    protected static string Where(CommandContext context) =>
        context.Option("--profile") is { } p ? $"profile {p} in {context.Config.Path}" : context.Config.Path;
}

/// <summary><c>idrak config get KEY</c>.</summary>
internal sealed class ConfigGetCommand : ConfigCommand
{
    public override string Name => "config get";

    public override string Summary => "Prints a config value (dotted keys, e.g. aliases.qwen.model)";

    public override string Usage => """
        KEY [--profile NAME]

        Options:
              --profile NAME  read the named profile (e.g. phone, desktop)

        Examples:
          idrak config get device
          idrak config get aliases.qwen --profile phone
        """;

    public override int Run(CommandContext context)
    {
        string key = context.Argument(0, "KEY (e.g. device)");
        var node = Get(Scope(context, create: false), key);
        if (node is null)
        {
            context.Error($"{key} is not set in {Where(context)} (idrak config set {key} VALUE sets it).");
            return ExitCodes.Failed;
        }

        if (!context.Json && !context.Quiet)
        {
            context.Output.WriteLine(Shown(key, node));
        }

        context.WriteJson(new JsonObject { ["key"] = key, ["value"] = ShownJson(key, node) });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak config set KEY VALUE</c>.</summary>
internal sealed class ConfigSetCommand : ConfigCommand
{
    public override string Name => "config set";

    public override string Summary => "Sets a config value (JSON values such as [\"a.dll\"] or 4 are kept as JSON)";

    public override string Usage => """
        KEY VALUE [--profile NAME]

        Arguments:
          VALUE  text, or JSON when it is a number, true/false, null, an array or an object

        Options:
              --profile NAME  write to the named profile (idrak config set profile NAME makes it the one in use)

        Keys the tool reads: device, cache, plugins (a list of paths), aliases (NAME: {model, weights, kv}),
        profile, serve.port (the port of serve and ui, and of the server the client commands call), and any key
        another command documents.

        Examples:
          idrak config set device vulkan:0
          idrak config set plugins '["./MyFormat.dll"]'
          idrak config set device cpu --profile phone
          idrak config set aliases.qwen.model Qwen/Qwen3-0.6B
        """;

    public override int Run(CommandContext context)
    {
        string key = context.Argument(0, "KEY (e.g. device)");
        string text = context.Argument(1, "VALUE");
        if (context.Positional.Count > 2)
        {
            throw new UsageException("config set takes one KEY and one VALUE; quote a value with spaces.");
        }

        JsonNode? value;
        try
        {
            value = text is ['[', ..] or ['{', ..] or ['"', ..] or "true" or "false" or "null" || double.TryParse(text, System.Globalization.CultureInfo.InvariantCulture, out _)
                ? JsonNode.Parse(text) : JsonValue.Create(text);
        }
        catch (JsonException e)
        {
            throw new UsageException($"{text} is not valid JSON ({e.Message}); quote it for plain text.");
        }

        var (parent, last) = Parent(Scope(context, create: true), key, create: true);
        parent![last] = value;
        context.Config.SaveChanges();
        context.Write($"{key} = {Shown(key, value)} ({Where(context)})");
        context.WriteJson(new JsonObject { ["key"] = key, ["value"] = ShownJson(key, value), ["file"] = context.Config.Path, ["profile"] = context.Option("--profile") });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak config unset KEY</c>.</summary>
internal sealed class ConfigUnsetCommand : ConfigCommand
{
    public override string Name => "config unset";

    public override string Summary => "Removes a config value";

    public override string Usage => """
        KEY [--profile NAME]

        Options:
              --profile NAME  remove it from the named profile instead of the top level

        Examples:
          idrak config unset device
          idrak config unset aliases.qwen
          idrak config unset device --profile phone
        """;

    public override int Run(CommandContext context)
    {
        string key = context.Argument(0, "KEY (e.g. device)");
        var (parent, last) = Parent(Scope(context, create: false), key, create: false);
        bool removed = parent?.Remove(last) == true;
        if (removed)
        {
            context.Config.SaveChanges();
        }

        context.Write(removed ? $"Removed {key} ({Where(context)})." : $"{key} was not set ({Where(context)}).");
        context.WriteJson(new JsonObject { ["key"] = key, ["removed"] = removed });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak config list</c>.</summary>
internal sealed class ConfigListCommand : ConfigCommand
{
    public override string Name => "config list";

    public override string Summary => "Every config value (or a profile's), and the file's path";

    public override string Usage => """
        [--profile NAME]

        Options:
              --profile NAME  only the named profile's values

        Examples:
          idrak config list
          idrak config list --profile phone
          idrak config list -j
        """;

    public override int Run(CommandContext context)
    {
        var scope = Scope(context, create: false);
        context.Write($"{Where(context)}{(File.Exists(context.Config.Path) ? "" : " (no file yet)")}{(context.Config.Profile is { } p && context.Option("--profile") is null ? $", profile in use: {p}" : "")}");
        var rows = new List<IReadOnlyList<string>>();
        void Walk(JsonObject o, string prefix)
        {
            foreach (var (key, value) in o)
            {
                if (value is JsonObject child && child.Count > 0 && !IsSecret(prefix + key))
                {
                    Walk(child, prefix + key + ".");
                }
                else
                {
                    rows.Add([prefix + key, Shown(prefix + key, value)]);
                }
            }
        }

        Walk(scope, "");
        if (rows.Count > 0)
        {
            context.Table(["Key", "Value"], rows);
        }

        context.WriteJson(new JsonObject { ["file"] = context.Config.Path, ["profile"] = context.Option("--profile"), ["values"] = ShownJson("", scope) });
        return ExitCodes.Ok;
    }
}
