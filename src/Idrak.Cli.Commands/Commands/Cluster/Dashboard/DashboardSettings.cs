// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Cli.Commands.Serve;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>The dashboard's settings (plan 23, phase C9): where it listens, who may call it, and what the page starts with.</summary>
/// <param name="Host">The address it listens on (<c>--host</c>, <c>IDRAK_CLUSTER_UI_HOST</c>, 127.0.0.1).</param>
/// <param name="Port">The port (<c>--port</c>, <c>IDRAK_CLUSTER_UI_PORT</c>; 0: one the system picks).</param>
/// <param name="ApiKey">A key requests from other machines may give (<c>IDRAK_API_KEY</c>), or null.</param>
/// <param name="KeyHashes">The keys <c>idrak server keys add</c> stored (their hashes).</param>
/// <param name="Topology">The topology file the page opens and saves (<c>--topology</c>), or null.</param>
internal sealed record DashboardSettings(string Host, int Port, string? ApiKey, IReadOnlyList<string> KeyHashes, string? Topology)
{
    /// <summary>The variable choosing the port.</summary>
    public const string PortVariable = "IDRAK_CLUSTER_UI_PORT";

    /// <summary>The variable choosing the address.</summary>
    public const string HostVariable = "IDRAK_CLUSTER_UI_HOST";

    /// <summary>The records kept per run for a page that connects late (<c>IDRAK_CLUSTER_UI_HISTORY</c>).</summary>
    public int History { get; init; } = 20_000;

    /// <summary>How often a listener reads a run folder's new lines (<c>IDRAK_CLUSTER_UI_POLL</c>, seconds).</summary>
    public TimeSpan Poll { get; init; } = TimeSpan.FromMilliseconds(500);

    /// <summary>The unit the page shows sizes in (B, KB, MB, GB, TB, powers of 1,024), or null: the largest each fills.</summary>
    public string? Unit { get; init; }

    /// <summary>The page's language: the tool's ("en" or "ar", right to left).</summary>
    public string Language { get; init; } = "en";

    /// <summary>A run folder to follow from the start (<c>--follow</c>), or null.</summary>
    public string? Follow { get; init; }

    /// <summary>Whether a key is configured for requests from other machines.</summary>
    public bool NeedsKey => ApiKey is not null || KeyHashes.Count > 0;

    /// <summary>The settings from a command's options and the environment.</summary>
    /// <exception cref="UsageException">A bad port, unit or setting; the message names its source.</exception>
    public static DashboardSettings Read(CommandContext context, string? topology)
    {
        string host = context.Option("--host") ?? (Environment.GetEnvironmentVariable(HostVariable) is { Length: > 0 } named ? named : "127.0.0.1");
        int port = Whole(context.Option("--port"), "--port", 0, 65535) ?? Whole(Environment.GetEnvironmentVariable(PortVariable), PortVariable, 0, 65535) ?? 0;
        string? unit = context.Option("--unit");
        if (unit is not null && unit.Replace("i", "", StringComparison.OrdinalIgnoreCase).ToUpperInvariant() is not ("B" or "KB" or "MB" or "GB" or "TB"))
        {
            throw new UsageException($"--unit: '{unit}' is not a unit; use B, KB, MB, GB or TB (powers of 1,024).");
        }

        double poll = Environment.GetEnvironmentVariable("IDRAK_CLUSTER_UI_POLL") is { Length: > 0 } pollText
            ? double.TryParse(pollText, NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds) && seconds > 0 && double.IsFinite(seconds)
                ? seconds : throw new UsageException($"IDRAK_CLUSTER_UI_POLL={pollText} is not a number of seconds above 0.")
            : 0.5;
        return new DashboardSettings(host, port, Environment.GetEnvironmentVariable("IDRAK_API_KEY") is { Length: > 0 } key ? key : null, [.. ServerKeys.Hashes(context)], topology)
        {
            History = Whole(Environment.GetEnvironmentVariable("IDRAK_CLUSTER_UI_HISTORY"), "IDRAK_CLUSTER_UI_HISTORY", 1, int.MaxValue) ?? 20_000,
            Poll = TimeSpan.FromSeconds(poll),
            Unit = unit?.Replace("i", "", StringComparison.OrdinalIgnoreCase).ToUpperInvariant(),
            Language = Messages.Language,
            Follow = context.Option("--follow"),
        };
    }

    private static int? Whole(string? text, string source, int min, int max) => text is not { Length: > 0 } ? null
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= min && value <= max
            ? value : throw new UsageException($"{source} must be a whole number from {min} to {max}, not '{text}'.");
}
