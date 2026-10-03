// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak update</c>: whether a newer version of the tool is published, and the command that installs it (the tool
/// never replaces itself).
/// </summary>
internal sealed class UpdateCommand : Command
{
    /// <summary>The package feed's list of the tool's versions.</summary>
    internal const string DefaultIndex = "https://api.nuget.org/v3-flatcontainer/idrak.cli/index.json";

    public override string Name => "update";

    public override string Summary => "Whether a newer version of the tool exists, and the command to install it";

    public override string Usage => """
        [--prerelease]

        Options:
              --prerelease  also count preview versions

        Reads the versions published on the package feed (IDRAK_UPDATE_INDEX names another list: a URL or a file);
        --offline and --timeout apply. Installing stays with dotnet tool update.

        Examples:
          idrak update
          idrak update --prerelease -j
        """;

    public override IReadOnlyCollection<string> Flags => ["--prerelease"];

    public override int Run(CommandContext context)
    {
        if (context.Offline)
        {
            throw new InvalidOperationException("--offline: the published versions cannot be read without the network.");
        }

        string index = Environment.GetEnvironmentVariable("IDRAK_UPDATE_INDEX") is { Length: > 0 } custom ? custom : DefaultIndex;
        string text;
        if (File.Exists(index))
        {
            text = File.ReadAllText(index);
        }
        else
        {
            using var http = new HttpClient { Timeout = context.Timeout ?? TimeSpan.FromSeconds(20) };
            try
            {
                text = http.GetStringAsync(index, context.TimeoutToken).GetAwaiter().GetResult();
            }
            catch (Exception e) when (e is HttpRequestException or TaskCanceledException)
            {
                throw new InvalidOperationException($"could not read {index}: {e.Message} (check the network or the proxy; idrak doctor --network)");
            }
        }

        var versions = (JsonNode.Parse(text)?["versions"] as JsonArray ?? throw new InvalidOperationException($"{index} has no \"versions\" list."))
            .Select(v => (string?)v).OfType<string>().ToList();
        string current = Health.Machine.ToolVersion;
        bool prerelease = context.Flag("--prerelease") || current.Contains('-');
        string? latest = versions.Where(v => prerelease || !v.Contains('-')).OrderBy(v => v, Comparer<string>.Create(Compare)).LastOrDefault();
        bool newer = latest is not null && Compare(latest, current) > 0;
        const string Command = "dotnet tool update -g Idrak.Cli";
        context.Write(latest is null ? $"No published version found; this is idrak {current}."
            : newer ? $"idrak {latest} is available (this is {current}). Install it with:\n  {Command}{(latest.Contains('-') ? $" --version {latest}" : "")}"
            : $"idrak {current} is the newest version{(prerelease ? "" : " (--prerelease counts previews)")}.");
        context.WriteJson(new JsonObject { ["current"] = current, ["latest"] = latest, ["newer"] = newer, ["command"] = newer ? Command : null });
        return ExitCodes.Ok;
    }

    /// <summary>Orders versions: numeric parts first, a release after its previews, previews by their labels.</summary>
    internal static int Compare(string a, string b)
    {
        static (int[] Numbers, string? Label) Split(string v)
        {
            int dash = v.IndexOf('-');
            string core = dash < 0 ? v : v[..dash];
            return ([.. core.Split('.').Select(p => int.TryParse(p, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : 0)], dash < 0 ? null : v[(dash + 1)..]);
        }

        var (x, xl) = Split(a);
        var (y, yl) = Split(b);
        for (int i = 0; i < Math.Max(x.Length, y.Length); i++)
        {
            int c = (i < x.Length ? x[i] : 0).CompareTo(i < y.Length ? y[i] : 0);
            if (c != 0)
            {
                return c;
            }
        }

        return (xl, yl) switch
        {
            (null, null) => 0,
            (null, _) => 1,
            (_, null) => -1,
            _ => string.CompareOrdinal(xl, yl),
        };
    }
}
