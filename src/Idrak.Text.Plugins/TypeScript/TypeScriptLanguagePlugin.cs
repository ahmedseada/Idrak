// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Plugins.TypeScript;

/// <summary>
/// TypeScript and Angular as coding languages (<see cref="CodingLanguages"/>: "typescript", then "angular") for the
/// coding agent and its tools. <b>typescript</b>: a workspace with <c>package.json</c> or <c>tsconfig.json</c>; its
/// build and test runners <c>npm</c> and <c>tsc</c>; checked with <c>npm exec -- tsc --noEmit</c> (the package's own
/// compiler, as <c>npx tsc</c> runs it) and <c>npm test</c>; <c>node_modules</c> (linked into a task's workspace, not
/// copied) and <c>dist</c> skipped. <b>angular</b>: a workspace with <c>angular.json</c>; the same programs; checked with
/// the workspace's scripts <c>npm run build</c> and <c>npm test -- --watch=false</c>; <c>.angular</c> skipped too.
/// Programs that run any code they are given (<c>node</c>, <c>npx</c>, <c>ng</c>) are in neither's defaults: the coding
/// tools allow them only when their options name them (<c>CodingToolOptions.Commands</c>, IDRAK_AGENT_COMMANDS).
/// </summary>
public static class TypeScriptLanguagePlugin
{
    /// <summary>The TypeScript language's name in <see cref="CodingLanguages"/>: "typescript".</summary>
    public const string TypeScriptName = "typescript";

    /// <summary>The Angular language's name in <see cref="CodingLanguages"/>: "angular".</summary>
    public const string AngularName = "angular";

    private static readonly Dictionary<string, string> Npm = new(StringComparer.Ordinal) { ["npm_config_yes"] = "true", ["npm_config_fund"] = "false", ["npm_config_audit"] = "false" };

    /// <summary>The TypeScript profile.</summary>
    public static ICodingLanguage TypeScript { get; } = new CodingLanguage(TypeScriptName, "TypeScript and JavaScript (npm): installs, type-checks and tests a package")
    {
        Markers = ["package.json", "tsconfig.json"],
        Commands = ["npm", "tsc"],
        PromptLines = ["TypeScript: npm exec -- tsc --noEmit, npm test"],
        ToolExamples = new CodingToolExamples { Folders = ["node_modules", "dist"], SearchGlobs = ["*.ts"], Commands = ["npm test", "npm exec -- tsc --noEmit"] },
        IgnoredDirectories = ["node_modules", "dist"],
        LinkedDirectories = ["node_modules"],
        GitExcludes = ["node_modules", "dist/"],
        Environment = Npm,
        DefaultSetup = ["npm install"],
        DefaultVerify = ["npm exec -- tsc --noEmit", "npm test"],
    };

    /// <summary>The Angular profile (an Angular workspace is an npm package too: its programs include TypeScript's).</summary>
    public static ICodingLanguage Angular { get; } = new CodingLanguage(AngularName, "Angular (the ng command line on npm): builds and tests an Angular workspace")
    {
        Markers = ["angular.json"],
        Commands = ["npm", "tsc"],
        PromptLines = ["Angular: npm run build, npm test -- --watch=false"],
        ToolExamples = new CodingToolExamples { FileGlobs = ["*.component.ts"], SearchPatterns = ["ngOnInit"], Commands = ["npm run build"] },
        IgnoredDirectories = ["node_modules", "dist", ".angular"],
        LinkedDirectories = ["node_modules"],
        GitExcludes = ["node_modules", "dist/", ".angular/"],
        Environment = new Dictionary<string, string>(Npm, StringComparer.Ordinal) { ["NG_CLI_ANALYTICS"] = "false" },
        DefaultSetup = ["npm install"],
        DefaultVerify = ["npm run build", "npm test -- --watch=false"],
    };

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("typescript-language"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new TypeScriptLanguageRegistration();

    /// <summary>Whether the languages are registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers TypeScript, then Angular (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters them, leaving <see cref="CodingLanguages"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class TypeScriptLanguageRegistration() : IdrakPlugin("typescript-language")
    {
        protected override void RegisterParts()
        {
            Part(() => CodingLanguages.Register(TypeScript), () => CodingLanguages.Unregister(TypeScriptName));
            Part(() => CodingLanguages.Register(Angular), () => CodingLanguages.Unregister(AngularName));
        }
    }
}
