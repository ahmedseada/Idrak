// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Reinforcement.Plugins.Coding.Abstractions;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins.CSharp;

namespace Idrak.Reinforcement.Plugins.CSharp;

/// <summary>
/// The <c>csharp</c> environment kind (registered by <see cref="CSharpCodingPlugin"/>): the <c>coding</c> environment on C#
/// projects, with the <c>csharp</c> coding language's defaults (<see cref="CSharpLanguagePlugin"/>: its programs and prompt
/// lines; a task that names no commands restores, builds and tests with dotnet), and rewards read from machine-readable
/// output: every verification <c>dotnet build</c> writes its errors and warnings to logs (<see cref="ErrorLog"/>,
/// <see cref="WarningLog"/>; <c>-fileLoggerParameters</c> errorsonly and warningsonly) and every <c>dotnet test</c> a TRX
/// results file (<see cref="ResultsFolder"/>), which the episode's info carries (<see cref="CSharpInfo"/>) for
/// <c>csharp-tests</c> and <c>csharp-build</c>; stale ones the agent left are removed before verification.
/// <list type="bullet">
/// <item><b>Options</b> beside the coding environment's: <c>packages</c> (one NuGet package folder every episode restores
/// from, <c>NUGET_PACKAGES</c>; <see cref="PackagesVariable"/>, else the machine's own), <c>node-reuse</c> (false: build
/// nodes end with their build, <c>MSBUILDDISABLENODEREUSE=1</c>, so parallel episodes leave none running), <c>build-server</c>
/// (true: the compiler server is shared between builds; false: each build compiles in its own process,
/// <c>UseSharedCompilation=false</c>).</item>
/// <item><b>Defaults</b> for the coding options: <c>reward</c> <c>csharp-tests</c>, <c>languages</c> <c>csharp</c>, and
/// <c>protect</c> (<see cref="ProtectVariable"/>, else <see cref="DefaultProtect"/>): edits to the test projects and the
/// build's shared files count as test edits (reported; refused with <c>refuse-test-edits</c>).</item>
/// </list>
/// </summary>
public sealed class CSharpEnvironmentVariant : CodingEnvironmentVariant
{
    /// <summary>The kind's name: "csharp".</summary>
    public const string KindName = "csharp";

    /// <summary>The variable of the option <c>packages</c>: "IDRAK_CSHARP_PACKAGES" (a folder).</summary>
    public const string PackagesVariable = "IDRAK_CSHARP_PACKAGES";

    /// <summary>The variable of the default <c>protect</c> globs: "IDRAK_CSHARP_PROTECT" (separated by commas or semicolons).</summary>
    public const string ProtectVariable = "IDRAK_CSHARP_PROTECT";

    /// <summary>The folder (in the workspace) verification's <c>dotnet test</c> writes its TRX results to: ".idrak-results".</summary>
    public const string ResultsFolder = ".idrak-results";

    /// <summary>The log of verification's <c>dotnet build</c> errors (in the workspace): "msbuild1.log" (<c>-fl1 -flp1:errorsonly</c>).</summary>
    public const string ErrorLog = "msbuild1.log";

    /// <summary>The log of verification's <c>dotnet build</c> warnings (in the workspace): "msbuild2.log" (<c>-fl2 -flp2:warningsonly</c>).</summary>
    public const string WarningLog = "msbuild2.log";

    // What verification's builds and test runs get, so their results are files (no ';': the commands are not run by a shell
    // and the coding tools refuse one).
    private const string BuildLogging = " -fl1 -flp1:errorsonly -fl2 -flp2:warningsonly", TestLogging = " --logger trx --results-directory " + ResultsFolder;

    private static readonly string[] OwnKeys = ["packages", "node-reuse", "build-server"];

    /// <summary>
    /// The files whose edits count as test edits by default: the build's shared files (<c>Directory.Build.props</c>,
    /// <c>Directory.Build.targets</c>, <c>Directory.Packages.props</c>, <c>global.json</c>) and the test projects (folders
    /// and projects named <c>*Tests</c> or <c>*Test</c>).
    /// </summary>
    public static IReadOnlyList<string> DefaultProtect { get; } =
        ["**/Directory.Build.props", "**/Directory.Build.targets", "**/Directory.Packages.props", "**/global.json", "**/*Tests/**", "**/*Test/**", "**/*Tests.csproj", "**/*Test.csproj"];

    /// <inheritdoc />
    public override string Name => KindName;

    /// <inheritdoc />
    public override string Summary =>
        "C# coding tasks (suite=DIR): the coding environment with dotnet's defaults; rewards from the build's error log and the tests' TRX results (reward=csharp-tests)";

    /// <inheritdoc />
    public override string Language => CSharpLanguagePlugin.LanguageName;

    /// <inheritdoc />
    public override IReadOnlyCollection<string> Keys => OwnKeys;

    /// <summary>The protected globs: <see cref="ProtectVariable"/>'s, else <see cref="DefaultProtect"/>.</summary>
    public static IReadOnlyList<string> Protect =>
        Environment.GetEnvironmentVariable("IDRAK_CSHARP_PROTECT") is { } text && text.Trim().Length > 0
            ? [.. text.Split([',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)]
            : DefaultProtect;

    /// <inheritdoc />
    public override IReadOnlyDictionary<string, string> Defaults() => new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["reward"] = CSharpRewards.Tests, ["languages"] = Language, ["protect"] = string.Join(",", Protect),
    };

    /// <inheritdoc />
    public override void Check(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _ = Flag(options, "node-reuse", false);
        _ = Flag(options, "build-server", true);
        if (options.TryGetValue("packages", out var packages) && packages.Trim().Length == 0)
        {
            throw new ArgumentException($"The {KindName} environment's packages= is empty; give a folder.", nameof(options));
        }
    }

    /// <summary>The package folder every episode restores from: the option <c>packages</c>, else <see cref="PackagesVariable"/>, else null (the machine's own NuGet folder).</summary>
    public static string? PackageFolder(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        string? folder = options.TryGetValue("packages", out var given) ? given : Environment.GetEnvironmentVariable("IDRAK_CSHARP_PACKAGES");
        return string.IsNullOrWhiteSpace(folder) ? null : Path.GetFullPath(folder.Trim());
    }

    /// <inheritdoc />
    public override IReadOnlyDictionary<string, string> Variables(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var variables = new Dictionary<string, string>(StringComparer.Ordinal);
        if (PackageFolder(options) is { } packages)
        {
            Directory.CreateDirectory(packages);
            variables["NUGET_PACKAGES"] = packages;
        }

        if (!Flag(options, "node-reuse", false))
        {
            variables["MSBUILDDISABLENODEREUSE"] = "1";
        }

        if (!Flag(options, "build-server", true))
        {
            variables["UseSharedCompilation"] = "false";
        }

        return variables;
    }

    /// <summary>The task as the environment runs it: the language's commands for what it does not name, and every verification <c>dotnet build</c> and <c>dotnet test</c> writing its results to files.</summary>
    public override CodingTask Task(CodingTask task, ICodingLanguage language, IReadOnlyDictionary<string, string> options)
    {
        var withDefaults = base.Task(task, language, options);
        return withDefaults with { Verify = [.. (withDefaults.Verify ?? []).Select(Logged)] };
    }

    /// <summary>Removes the logs and results a verification writes (and any the agent left) from the workspace.</summary>
    public override void BeforeVerify(string workspace, IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        foreach (var log in new[] { ErrorLog, WarningLog })
        {
            string path = Path.Combine(workspace, log);
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }

        string results = Path.Combine(workspace, ResultsFolder);
        if (Directory.Exists(results))
        {
            CodingWorkspace.Delete(results);
        }
    }

    /// <summary>Adds the tests' results (TRX) and the build's errors and warnings (its logs) to the info (<see cref="CSharpInfo"/>).</summary>
    public override void AfterVerify(CodingTask task, string workspace, CodingVerification verification, IDictionary<string, string> info, IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(workspace);
        ArgumentNullException.ThrowIfNull(verification);
        ArgumentNullException.ThrowIfNull(info);
        var (total, passed) = CSharpResults.Tests(Path.Combine(workspace, ResultsFolder));
        info[CSharpInfo.TestsTotal] = total.ToString(CultureInfo.InvariantCulture);
        info[CSharpInfo.TestsPassed] = passed.ToString(CultureInfo.InvariantCulture);

        // The build passed when its command ran (every command before it passed) and passed itself.
        var verify = task.Verify ?? [];
        int build = -1;
        for (int i = 0; i < verify.Count && build < 0; i++)
        {
            build = IsDotnet(verify[i], "build") ? i : -1;
        }

        info[CSharpInfo.BuildPassed] = build >= 0 && build < verification.PassedCommands ? "true" : "false";
        info[CSharpInfo.BuildErrors] = CSharpResults.Diagnostics(Path.Combine(workspace, ErrorLog), "error").ToString(CultureInfo.InvariantCulture);
        info[CSharpInfo.BuildWarnings] = CSharpResults.Diagnostics(Path.Combine(workspace, WarningLog), "warning").ToString(CultureInfo.InvariantCulture);
    }

    // A verification command with its results written to files: a dotnet build gets the error and warning logs, a dotnet test the TRX logger.
    private static string Logged(string command) =>
        IsDotnet(command, "build") ? command + BuildLogging : IsDotnet(command, "test") ? command + TestLogging : command;

    private static bool IsDotnet(string command, string verb)
    {
        var words = command.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return words.Length >= 2 && Path.GetFileNameWithoutExtension(words[0]).Equals("dotnet", StringComparison.OrdinalIgnoreCase) && words[1].Equals(verb, StringComparison.OrdinalIgnoreCase);
    }

    private static bool Flag(IReadOnlyDictionary<string, string> options, string key, bool fallback) =>
        !options.TryGetValue(key, out var value) ? fallback
        : value.Trim().ToLowerInvariant() switch
        {
            "true" or "1" or "on" or "yes" => true,
            "false" or "0" or "off" or "no" => false,
            _ => throw new ArgumentException($"The {KindName} environment's {key}={value} is true or false."),
        };
}
