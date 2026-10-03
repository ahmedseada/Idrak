// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak new console|webapi|rag|plugin NAME</c>: writes a ready-to-build project from the templates embedded in the
/// tool (Templates/KIND/...), referencing the Idrak packages of the tool's own version, or with <c>--source DIR</c>
/// the projects of a source checkout.
/// </summary>
internal sealed partial class NewCommand : Command
{
    /// <summary>The templates, with what each writes.</summary>
    public static readonly IReadOnlyDictionary<string, string> Kinds = new Dictionary<string, string>
    {
        ["console"] = "a console app that trains a small network, saves a model package and predicts from it",
        ["webapi"] = "a Web API serving a model package through the inference engine (Idrak.AspNetCore)",
        ["rag"] = "retrieval over a folder of text files, answered with citations by a chat model (Idrak.LanguageModels)",
        ["plugin"] = "a plug-in registering a packed weight format and a network step, with tests on the public API only",
    };

    public override string Name => "new";

    public override string Summary => "Writes a new project: console, webapi, rag or plugin";

    public override string Usage =>
        "console|webapi|rag|plugin NAME [options]\n\n" +
        string.Concat(Kinds.Select(k => $"  {k.Key,-8} {k.Value}\n")) + "\n" +
        "Options:\n" +
        "  -o, --out DIR       where to write it (default ./NAME)\n" +
        "  -f, --force         write into a folder that is not empty\n" +
        "      --source DIR    reference the projects of an Idrak source checkout instead of the packages\n\n" +
        "Examples:\n" +
        "  idrak new plugin MyFormat\n" +
        "  idrak new webapi PriceApi -o ./services/price\n" +
        "  idrak new console Demo --source ~/src/Idrak\n\n" +
        "Environment: none of Idrak's; building the project uses dotnet's own (DOTNET_CLI_TELEMETRY_OPTOUT, DOTNET_NOLOGO).";

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--source"];

    public override IReadOnlyCollection<string> Flags => ["--force"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string kind = context.Argument(0, "the template (console, webapi, rag or plugin)").ToLowerInvariant();
        if (!Kinds.ContainsKey(kind))
        {
            throw new UsageException($"Unknown template '{kind}'; choose one of {string.Join(", ", Kinds.Keys)}.");
        }

        string name = context.Argument(1, "the project NAME");
        if (!ProjectName().IsMatch(name))
        {
            throw new UsageException($"'{name}' is not a usable project name: start with a letter, then letters, digits, '.', '_' or '-'.");
        }

        if (context.Positional.Count > 2)
        {
            throw new UsageException($"Unexpected argument '{context.Positional[2]}'.");
        }

        string folder = Path.GetFullPath(context.Option("--out") ?? name);
        if (Directory.Exists(folder) && Directory.EnumerateFileSystemEntries(folder).Any() && !context.Flag("--force"))
        {
            context.Error($"{folder} is not empty. Choose another folder with --out, or pass --force to write into it.");
            return ExitCodes.Failed;
        }

        string? source = context.Option("--source") is { } given ? Path.GetFullPath(given) : null;
        if (source is not null && !File.Exists(Path.Combine(source, "src", "Idrak", "Idrak.csproj")))
        {
            context.Error($"{source} is not an Idrak source checkout (no src/Idrak/Idrak.csproj). Pass the folder that holds Idrak.slnx.");
            return ExitCodes.Failed;
        }

        string version = PackageVersion();
        var values = new Dictionary<string, string>
        {
            ["__NAME__"] = name,
            ["__NAMESPACE__"] = Namespace(name),
            ["__FORMAT__"] = FormatName(name),
        };
        var written = new List<string>();
        foreach (var (path, text) in Template(kind))
        {
            string target = Path.Combine(folder, Fill(path, values));
            string content = References().Replace(Fill(text, values), m => Reference(m.Groups[1].Value, source, Path.GetDirectoryName(target)!, version));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.WriteAllText(target, content);
            written.Add(Path.GetRelativePath(folder, target).Replace('\\', '/'));
        }

        string build = kind == "plugin" ? $"cd {Quote(folder)} && dotnet build && dotnet run --project tests/{name}.Tests" : $"cd {Quote(folder)} && dotnet run";
        context.Write($"Wrote the {kind} project {name} to {folder} ({written.Count} files, referencing {(source is null ? $"the Idrak packages {version}" : $"the projects in {source}")})");
        foreach (string file in written)
        {
            context.Detail($"  {file}");
        }

        context.Write($"Next: {build}");
        context.WriteJson(new JsonObject
        {
            ["template"] = kind,
            ["name"] = name,
            ["folder"] = folder,
            ["references"] = source is null ? $"packages {version}" : $"source {source}",
            ["files"] = new JsonArray([.. written.Select(f => (JsonNode)f)]),
            ["next"] = build,
        });
        return ExitCodes.Ok;
    }

    /// <summary>The files of a template (relative path with placeholders, text), from the tool's embedded resources.</summary>
    internal static IEnumerable<(string Path, string Text)> Template(string kind)
    {
        var assembly = typeof(NewCommand).Assembly;
        string prefix = $"Templates/{kind}/";
        foreach (string resource in assembly.GetManifestResourceNames().Select(r => (Name: r, Path: r.Replace('\\', '/'))).Where(r => r.Path.StartsWith(prefix, StringComparison.Ordinal))
                     .OrderBy(r => r.Path, StringComparer.Ordinal).Select(r => r.Name))
        {
            using var stream = assembly.GetManifestResourceStream(resource)!;
            using var reader = new StreamReader(stream, Encoding.UTF8);
            yield return (resource.Replace('\\', '/')[prefix.Length..], reader.ReadToEnd());
        }
    }

    /// <summary>The version the package references ask for: the tool's own, or every prerelease of it for a local build.</summary>
    internal static string PackageVersion()
    {
        var assembly = typeof(Tensor).Assembly;
        string version = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? assembly.GetName().Version?.ToString(3) ?? "0.1.0";
        version = version.Split('+')[0];
        int dash = version.IndexOf('-', StringComparison.Ordinal);
        return dash < 0 ? version : version[..dash] + "-*";
    }

    // A C# namespace from a project name: dashes become underscores, a segment starting with a digit gets one in front.
    internal static string Namespace(string name) =>
        string.Join('.', name.Replace('-', '_').Split('.', StringSplitOptions.RemoveEmptyEntries).Select(s => char.IsDigit(s[0]) ? "_" + s : s));

    // The registered name of the plug-in's format: lower case, words joined by dashes.
    internal static string FormatName(string name) => NonWord().Replace(name.ToLowerInvariant(), "-").Trim('-');

    private static string Fill(string text, Dictionary<string, string> values) =>
        values.Aggregate(text, (t, p) => t.Replace(p.Key, p.Value, StringComparison.Ordinal));

    // A package reference, or a project reference into the checkout: relative when the two folders are close, else absolute.
    private static string Reference(string package, string? source, string projectFolder, string version)
    {
        if (source is null)
        {
            return $"<PackageReference Include=\"{package}\" Version=\"{version}\" />";
        }

        string project = Path.Combine(source, "src", package, package + ".csproj");
        string relative = Path.GetRelativePath(projectFolder, project);
        return $"<ProjectReference Include=\"{(relative.Split(Path.DirectorySeparatorChar).Count(p => p == "..") > 3 ? project : relative)}\" />";
    }

    private static string Quote(string path) => path.Contains(' ', StringComparison.Ordinal) ? $"\"{path}\"" : path;

    [GeneratedRegex(@"^[A-Za-z][A-Za-z0-9._-]*$")]
    private static partial Regex ProjectName();

    [GeneratedRegex(@"__REF\(([A-Za-z.]+)\)__")]
    private static partial Regex References();

    [GeneratedRegex(@"[^a-z0-9]+")]
    private static partial Regex NonWord();
}
