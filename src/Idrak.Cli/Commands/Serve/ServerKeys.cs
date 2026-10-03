// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Serve;

/// <summary>
/// API keys for <c>idrak serve</c>, kept in the config file under "server-keys" as SHA-256 hashes (never the keys):
/// {"server-keys": {"laptop": {"sha256": "...", "created": "..."}}}.
/// </summary>
internal static class ServerKeys
{
    public const string ConfigKey = "server-keys";

    public static string Hash(string key) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(key)));

    /// <summary>The stored hashes.</summary>
    public static IEnumerable<string> Hashes(CommandContext context) =>
        context.Config.Object(ConfigKey)?.Select(p => (string?)p.Value?["sha256"]).OfType<string>() ?? [];

    /// <summary>Whether <paramref name="key"/> hashes to one of <paramref name="hashes"/> (compared in constant time).</summary>
    public static bool Matches(string key, IReadOnlyList<string> hashes)
    {
        byte[] given = Encoding.ASCII.GetBytes(Hash(key));
        bool found = false;
        foreach (string hash in hashes)
        {
            found |= CryptographicOperations.FixedTimeEquals(given, Encoding.ASCII.GetBytes(hash));
        }

        return found;
    }
}

/// <summary><c>idrak server keys add NAME</c>: makes a key, prints it once and stores its hash.</summary>
internal sealed class ServerKeysAddCommand : Command
{
    public override string Name => "server keys add";

    public override string Summary => "Make an API key for serve (printed once; the config keeps only its hash)";

    public override string Usage => """
        NAME [options]

        Makes a random key named NAME, prints it once and stores its SHA-256 hash in the config file; 'idrak serve'
        then accepts it (as "Authorization: Bearer KEY"), along with --api-key and the other stored keys.

        Options:
              --key KEY  store this key instead of a random one

        Examples:
          idrak server keys add laptop
          idrak server keys add ci --json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--key"];

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "NAME (a name for the key, e.g. laptop)");
        var keys = (context.Config.Object(ServerKeys.ConfigKey)?.DeepClone() as JsonObject) ?? [];
        if (keys.ContainsKey(name))
        {
            throw new UsageException($"A key named '{name}' exists; remove it first with 'idrak server keys rm {name}'.");
        }

        string key = context.Option("--key") ?? "idrak-" + Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(24));
        if (key.Length < 8)
        {
            throw new UsageException("--key must have at least 8 characters.");
        }

        string created = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        keys[name] = new JsonObject { ["sha256"] = ServerKeys.Hash(key), ["created"] = created };
        context.Config.SetNode(ServerKeys.ConfigKey, keys);
        if (context.Json)
        {
            context.WriteJson(new JsonObject { ["name"] = name, ["key"] = key, ["created"] = created, ["config"] = context.Config.Path });
        }
        else
        {
            context.Output.WriteLine(key);                  // even with --quiet: it is the command's result
            context.Write($"Key '{name}' stored (as its hash) in {context.Config.Path}; it is not shown again.");
        }

        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak server keys list</c>.</summary>
internal sealed class ServerKeysListCommand : Command
{
    public override string Name => "server keys list";

    public override string Summary => "The API keys serve accepts (names and dates; the keys are not stored)";

    public override string Usage => """
        [options]

        Examples:
          idrak server keys list
          idrak server keys list --json
        """;

    public override int Run(CommandContext context)
    {
        var keys = context.Config.Object(ServerKeys.ConfigKey) ?? [];
        context.Table(["Name", "Created", "Hash"], keys.Select(p => (IReadOnlyList<string>)[p.Key, (string?)p.Value?["created"] ?? "", ((string?)p.Value?["sha256"] ?? "")[..Math.Min(12, ((string?)p.Value?["sha256"] ?? "").Length)] + "..."]));
        if (keys.Count == 0)
        {
            context.Write("No keys; add one with 'idrak server keys add NAME'.");
        }

        context.WriteJson(new JsonObject
        {
            ["keys"] = new JsonArray([.. keys.Select(p => (JsonNode)new JsonObject { ["name"] = p.Key, ["created"] = p.Value?["created"]?.DeepClone() })]),
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak server keys rm NAME</c>.</summary>
internal sealed class ServerKeysRemoveCommand : Command
{
    public override string Name => "server keys rm";

    public override string Summary => "Remove an API key (servers started afterwards no longer accept it)";

    public override string Usage => """
        NAME [options]

        Examples:
          idrak server keys rm laptop
        """;

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "NAME (see 'idrak server keys list')");
        var keys = (context.Config.Object(ServerKeys.ConfigKey)?.DeepClone() as JsonObject) ?? [];
        if (!keys.Remove(name))
        {
            throw new InvalidOperationException($"no key named '{name}'; 'idrak server keys list' shows the names.");
        }

        context.Config.SetNode(ServerKeys.ConfigKey, keys.Count == 0 ? null : keys);
        context.Write($"Key '{name}' removed; restart running servers for it to stop working.");
        context.WriteJson(new JsonObject { ["removed"] = name });
        return ExitCodes.Ok;
    }
}
