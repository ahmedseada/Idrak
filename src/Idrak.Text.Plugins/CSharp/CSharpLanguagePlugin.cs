// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Plugins.CSharp;

/// <summary>
/// C# as a coding language (<see cref="CodingLanguages"/>: "csharp") for the coding agent and its tools: a workspace
/// with a solution or a project (<c>*.sln</c>, <c>*.slnx</c>, <c>*.csproj</c>) is C#; <c>run_command</c> may start
/// <c>dotnet</c>; the agent checks its work with <c>dotnet build</c> and <c>dotnet test</c>; <c>bin</c>, <c>obj</c> and
/// <c>.vs</c> are skipped; a coding task that names no commands restores, builds and tests with dotnet.
/// </summary>
public static class CSharpLanguagePlugin
{
    /// <summary>The language's name in <see cref="CodingLanguages"/>: "csharp".</summary>
    public const string LanguageName = "csharp";

    /// <summary>The profile the plug-in registers.</summary>
    public static ICodingLanguage Language { get; } = new CodingLanguage(LanguageName, "C# (.NET): dotnet restores, builds and tests a solution or project")
    {
        Markers = ["*.sln", "*.slnx", "*.csproj"],
        Commands = ["dotnet"],
        PromptLines = ["C#: dotnet build, dotnet test"],
        ToolExamples = new CodingToolExamples
        {
            Folders = ["bin", "obj"], FileGlobs = ["*.cs"], SearchPatterns = [@"class \w+Service"], Commands = ["dotnet build", "dotnet test"], CommandLines = ["dotnet test --no-restore"],
        },
        IgnoredDirectories = ["bin", "obj", ".vs"],
        GitExcludes = ["bin/", "obj/"],
        Environment = new Dictionary<string, string>(StringComparer.Ordinal) { ["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1", ["DOTNET_NOLOGO"] = "1", ["DOTNET_SKIP_FIRST_TIME_EXPERIENCE"] = "1" },
        DefaultSetup = ["dotnet restore"],
        DefaultVerify = ["dotnet build --no-restore", "dotnet test --no-build"],
    };

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("csharp-language"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new CSharpLanguageRegistration();

    /// <summary>Whether the language is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the language (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the language, leaving <see cref="CodingLanguages"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class CSharpLanguageRegistration() : IdrakPlugin("csharp-language")
    {
        protected override void RegisterParts() => Part(() => CodingLanguages.Register(Language), () => CodingLanguages.Unregister(LanguageName));
    }
}
