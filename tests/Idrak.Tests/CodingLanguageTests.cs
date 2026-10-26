// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Tar;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using Idrak;
using Idrak.Abstraction.Plugins;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;
using Idrak.Text.Plugins;
using Idrak.Text.Plugins.CSharp;
using Idrak.Text.Plugins.Python;
using Idrak.Text.Plugins.TypeScript;

// Plan 27, R4b part 1: no programming language in the library. The coding languages are a registry (CodingLanguages,
// Idrak.Text.Abstractions) the library fills with none; the coding agent and tools compose their defaults from the
// registered ones; C#, TypeScript and Angular, and Python are Idrak.Text.Plugins' profiles. Idrak.Data skips only .git and
// the editors' folders unless told; the tool's data commands pass the languages' folders.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingLanguageGroup =
    [
        ("coding languages: with Idrak.Text.Plugins' languages and all, the agent's system prompt, the programs allowed and the tools' descriptions are the ones the library had byte for byte, the ignored folders those plus python's __pycache__ and .venv, the commands' variables the same", d => { if (d == Device.Cpu) CodingLanguagesUnchanged(); }),
        ("coding languages: the library registers none; with none registered the prompt, the programs, the folders, the tools' descriptions and the variables name no language (every profile's programs, folders, examples and labels scanned)", d => { if (d == Device.Cpu) CodingLanguagesNone(); }),
        ("coding languages: detected picks csharp for a .csproj workspace, typescript for a package.json one, typescript and angular for an Angular one; names choose in their order and one not registered is refused naming Idrak.Text.Plugins; a workspace no chosen language detects gets a notice naming the plug-in, a task's own commands still run; IDRAK_CODING_LANGUAGES chooses; the merge keeps each list's order", d => { if (d == Device.Cpu) CodingLanguagesChoice(); }),
        ("plugins: csharp-language: CSharpLanguagePlugin is the plug-in \"csharp-language\" of TextPlugins.All, in Idrak.Text.Plugins.CSharp; Register twice registers csharp once; Unregister leaves CodingLanguages as found", d => { if (d == Device.Cpu) LanguagePluginRegisters(CSharpLanguagePlugin.Plugin, "csharp-language", "Idrak.Text.Plugins.CSharp", ["csharp"]); }),
        ("plugins: typescript-language: TypeScriptLanguagePlugin is the plug-in \"typescript-language\" of TextPlugins.All, in Idrak.Text.Plugins.TypeScript; Register twice registers typescript and angular once; Unregister leaves CodingLanguages as found", d => { if (d == Device.Cpu) LanguagePluginRegisters(TypeScriptLanguagePlugin.Plugin, "typescript-language", "Idrak.Text.Plugins.TypeScript", ["typescript", "angular"]); }),
        ("plugins: python-language: PythonLanguagePlugin is the plug-in \"python-language\" of TextPlugins.All, in Idrak.Text.Plugins.Python; Register twice registers python once (its folders only: no programs, prompt lines or commands); Unregister leaves CodingLanguages as found", d => { if (d == Device.Cpu) LanguagePluginRegisters(PythonLanguagePlugin.Plugin, "python-language", "Idrak.Text.Plugins.Python", ["python"]); }),
        ("data files: Idrak.Data alone skips only .git, .vs and .idea in folders and archives (ReadOptions.SkippedFolders adds more); the tool's data commands skip the folders they always did, from the registered coding languages, in folders and archives", d => { if (d == Device.Cpu) DataFilesSkippedFolders(); }),
    ];

    // The language plug-ins, in TextPlugins.All's order.
    private static readonly IIdrakPlugin[] LanguagePlugins = [CSharpLanguagePlugin.Plugin, TypeScriptLanguagePlugin.Plugin, PythonLanguagePlugin.Plugin];

    // Runs `body` with no coding language registered, then registers again, in TextPlugins.All's order, those that were.
    private static void WithoutLanguages(Action body)
    {
        var was = LanguagePlugins.Where(p => p.IsRegistered).ToList();
        try
        {
            foreach (var plugin in Enumerable.Reverse(LanguagePlugins))
            {
                IdrakPlugins.Unregister(plugin);
            }

            body();
        }
        finally
        {
            foreach (var plugin in was)
            {
                IdrakPlugins.Register(plugin);
            }
        }
    }

    // Runs `body` with exactly Idrak.Text.Plugins' languages registered, in their order.
    private static void WithTextLanguages(Action body) => WithoutLanguages(() =>
    {
        foreach (var plugin in LanguagePlugins)
        {
            IdrakPlugins.Register(plugin);
        }

        try
        {
            body();
        }
        finally
        {
            foreach (var plugin in Enumerable.Reverse(LanguagePlugins))
            {
                IdrakPlugins.Unregister(plugin);
            }
        }
    });

    // The descriptions of the tools (the tool's, then each parameter's), "name: text" and "name.parameter: text".
    private static Dictionary<string, string> ToolTexts(CodingTools tools)
    {
        var texts = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var tool in tools.All)
        {
            texts[tool.Definition.Name] = tool.Definition.Description ?? "";
            foreach (var (parameter, schema) in tool.Definition.Parameters!["properties"]!.AsObject())
            {
                texts[$"{tool.Definition.Name}.{parameter}"] = (string?)schema!["description"] ?? "";
            }
        }

        return texts;
    }

    private static string LanguageTemp(string name)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-languages-{name}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void CodingLanguagesUnchanged()
    {
        // The library's defaults before R4b, kept here only (the library has no copy of them).
        const string OldPrompt = """
            You are a coding agent working in a software project on the user's machine. You have tools to list, read and
            search the project's files, edit and write files, and run commands such as builds and tests.

            How to work:
            - Understand before changing: find the relevant files (list_files, search) and read them before editing.
            - Make focused changes with edit_file; copy old_text exactly as the file has it (read_file shows line numbers
              before a "| " separator: they are not part of the file). Create new files with write_file.
            - Follow the project's conventions: its naming, formatting, file layout, frameworks and library versions.
            - Check your work: build and run the tests after changing code (C#: dotnet build, dotnet test; TypeScript:
              npx tsc --noEmit, npm test; Angular: npx ng build, npx ng test --watch=false). Read the errors, fix them and
              run the checks again until they pass.
            - Do not weaken or delete tests to make them pass, unless the user asks for it.
            - Do only what was asked; mention anything else worth doing instead of doing it.
            - When the task is done, answer without a tool call: say briefly what you changed and how you checked it.
            """;
        string[] oldCommands = ["dotnet", "npm", "npx", "ng", "node", "tsc", "git"];
        string[] oldIgnored = [".git", "bin", "obj", "node_modules", "dist", ".angular", ".vs", ".idea", "coverage", "out"];
        var oldVariables = new Dictionary<string, string>
        {
            ["CI"] = "1", ["NO_COLOR"] = "1", ["FORCE_COLOR"] = "0", ["NG_CLI_ANALYTICS"] = "false", ["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1", ["DOTNET_NOLOGO"] = "1",
            ["DOTNET_SKIP_FIRST_TIME_EXPERIENCE"] = "1", ["npm_config_yes"] = "true", ["npm_config_fund"] = "false", ["npm_config_audit"] = "false",
        };
        var oldTexts = new Dictionary<string, string>
        {
            ["list_files"] = "List the files of the workspace, or of a folder in it, as paths relative to the workspace. Build output and dependency folders (bin, obj, node_modules, dist, .git …) are skipped.",
            ["list_files.path"] = "Folder to list, relative to the workspace (default: the whole workspace).",
            ["list_files.pattern"] = "Only file names matching this glob, e.g. *.cs or *.component.ts.",
            ["read_file"] = "Read a text file, with line numbers. Long files are returned in parts: pass start_line and end_line for the rest.",
            ["read_file.path"] = "File path, relative to the workspace.",
            ["read_file.start_line"] = "First line to return (1-based, default 1).",
            ["read_file.end_line"] = "Last line to return (default: as many as fit).",
            ["search"] = "Search the files of the workspace for a regular expression; returns path:line: text for each match.",
            ["search.pattern"] = "Regular expression (.NET syntax), e.g. class \\w+Service or ngOnInit.",
            ["search.path"] = "Folder or file to search, relative to the workspace (default: everything).",
            ["search.glob"] = "Only files whose names match this glob, e.g. *.ts.",
            ["search.ignore_case"] = "Match case-insensitively.",
            ["edit_file"] = "Replace an exact piece of a file with new text. old_text must match the file exactly (whitespace and indentation included) and occur once, unless replace_all is set; include enough surrounding lines to make it unique.",
            ["edit_file.path"] = "File path, relative to the workspace.",
            ["edit_file.old_text"] = "The exact text to replace.",
            ["edit_file.new_text"] = "The replacement.",
            ["edit_file.replace_all"] = "Replace every occurrence.",
            ["write_file"] = "Create a file, or replace a file's whole content. Prefer edit_file for changes to existing files.",
            ["write_file.path"] = "File path, relative to the workspace; missing folders are created.",
            ["write_file.content"] = "The complete file content.",
            ["run_command"] = "Run a command in the workspace and return its exit code and output. Allowed programs: dotnet, npm, npx, ng, node, tsc, git (git only status, diff, log, show, ls-files, grep, blame). The command is not run by a shell: no pipes, redirection, && or variables. Use it to build, test, lint and inspect, e.g. dotnet build, dotnet test, npm test, npx tsc --noEmit, ng build.",
            ["run_command.command"] = "The command line, e.g. dotnet test --no-restore.",
            ["run_command.directory"] = "Folder to run in, relative to the workspace (default: the workspace).",
            ["run_command.timeout_seconds"] = "Stop the command after this many seconds (at most 300).",
        };

        string root = LanguageTemp("unchanged");
        try
        {
            WithTextLanguages(() =>
            {
                Check(CodingLanguages.Names.SequenceEqual(["csharp", "typescript", "angular", "python"]), $"the registered languages, in order: {string.Join(", ", CodingLanguages.Names)}");
                var tools = new CodingTools(root, new CodingToolOptions { Languages = CodingLanguages.All });
                string prompt = CodingAgent.SystemPrompt(tools.Languages);
                string old = OldPrompt.ReplaceLineEndings("\n");
                int at = Enumerable.Range(0, Math.Min(old.Length, prompt.Length)).FirstOrDefault(i => old[i] != prompt[i], Math.Min(old.Length, prompt.Length));
                Check(prompt == old, $"the system prompt is the old one byte for byte; first difference at {at}: \"{prompt[Math.Max(0, at - 20)..Math.Min(prompt.Length, at + 40)]}\"");
                Check(CodingAgent.DefaultSystemPrompt == old, "DefaultSystemPrompt: every registered language, the old prompt");
                Check(tools.Commands.SequenceEqual(oldCommands), $"the programs allowed, in the old order: {string.Join(", ", tools.Commands)}");
                Check(tools.Limits.Programs!.SequenceEqual(oldCommands), "the sandbox gets the same programs");
                Check(tools.IgnoredDirectories.Order(StringComparer.Ordinal).SequenceEqual(oldIgnored.Append("__pycache__").Append(".venv").Order(StringComparer.Ordinal)),
                    $"the ignored folders: the old ones and python's two ({string.Join(", ", tools.IgnoredDirectories)})");
                Check(tools.Limits.Environment!.Count == oldVariables.Count && oldVariables.All(v => tools.Limits.Environment[v.Key] == v.Value),
                    $"the commands' variables: {string.Join(", ", tools.Limits.Environment.Select(v => $"{v.Key}={v.Value}"))}");
                var texts = ToolTexts(tools);
                Check(texts.Count == oldTexts.Count, $"{texts.Count} tool texts, the old {oldTexts.Count}");
                foreach (var (key, text) in oldTexts)
                {
                    Check(texts.TryGetValue(key, out var now) && now == text, $"{key}: the old text byte for byte:\n{text}\nnow:\n{now}");
                }

                // The default choice (IDRAK_CODING_LANGUAGES unset) and the language names in order give the same tools.
                if (Environment.GetEnvironmentVariable(CodingLanguages.EnvironmentVariable) is null)
                {
                    Check(ToolTexts(new CodingTools(root)).SequenceEqual(texts), "the default choice is all");
                }

                var named = new CodingTools(root, new CodingToolOptions { Languages = "csharp, typescript, angular, python" });
                Check(ToolTexts(named).SequenceEqual(texts) && named.Commands.SequenceEqual(oldCommands), "the names in the registry's order give the same tools");
            });
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void CodingLanguagesNone()
    {
        string root = LanguageTemp("none");
        try
        {
            WithoutLanguages(() =>
            {
                Check(CodingLanguages.Names.Count == 0, $"with Idrak.Text.Plugins' languages unregistered none is left: the library registers no language ({string.Join(", ", CodingLanguages.Names)})");
                var tools = new CodingTools(root);
                Check(tools.Languages.Count == 0 && tools.Commands.SequenceEqual(["git"]), $"git alone: {string.Join(", ", tools.Commands)}");
                Check(tools.IgnoredDirectories.SequenceEqual(CodingToolOptions.LanguageFreeIgnoredDirectories), $"the language-free folders: {string.Join(", ", tools.IgnoredDirectories)}");
                string prompt = CodingAgent.SystemPrompt(tools.Languages);
                Check(prompt == CodingAgent.DefaultSystemPrompt && prompt.Contains("- Check your work: build and run the tests after changing code. Read the errors", StringComparison.Ordinal),
                    $"the prompt's line on checking the work names no command:\n{prompt}");
                Check(prompt.Split('\n').All(l => l.Length <= 110), "the prompt's lines stay within its width");
                var texts = ToolTexts(tools);
                Check(texts["list_files"].EndsWith("Build output and dependency folders (.git …) are skipped.", StringComparison.Ordinal)
                      && texts["list_files.pattern"] == "Only file names matching this glob." && texts["search.pattern"] == "Regular expression (.NET syntax)."
                      && texts["search.glob"] == "Only files whose names match this glob." && texts["run_command.command"] == "The command line."
                      && texts["run_command"].EndsWith("Use it to build, test, lint and inspect.", StringComparison.Ordinal), "the descriptions without examples");

                // Everything the profiles name: their programs, folders, examples, prompt labels, variables and names.
                ICodingLanguage[] profiles = [CSharpLanguagePlugin.Language, TypeScriptLanguagePlugin.TypeScript, TypeScriptLanguagePlugin.Angular, PythonLanguagePlugin.Language];
                var words = profiles.SelectMany(p => p.Commands.Concat(p.IgnoredDirectories).Concat(p.LinkedDirectories).Concat(p.ToolExamples.Folders).Concat(p.ToolExamples.FileGlobs)
                        .Concat(p.ToolExamples.SearchGlobs).Concat(p.ToolExamples.SearchPatterns).Concat(p.Environment.Keys).Concat(p.PromptLines.Select(l => l[..l.IndexOf(':')])).Append(p.Name))
                    .Concat(["C#", "TypeScript", "Angular", "Python"]).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
                var everything = new List<(string What, string Text)> { ("the prompt", prompt) };
                everything.AddRange(texts.Select(t => (t.Key, t.Value)));
                everything.AddRange(tools.Commands.Select(c => ("a program", c)));
                everything.AddRange(tools.IgnoredDirectories.Select(f => ("an ignored folder", f)));
                everything.AddRange(tools.Limits.Environment!.Select(v => ("a variable", $"{v.Key}={v.Value}")));
                foreach (var word in words)
                {
                    var pattern = new Regex($@"(?<![\w.]){Regex.Escape(word)}(?![\w])", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
                    foreach (var (what, text) in everything)
                    {
                        Check(!pattern.IsMatch(text), $"{what} names '{word}' with no language registered: {text}");
                    }
                }

                Check(words.Count > 25, $"only {words.Count} words scanned: the scan is broken");
                Check(tools.LanguageNotice is null, "an empty workspace: no notice");
                File.WriteAllText(Path.Combine(root, "App.csproj"), "<Project />");
                var later = new CodingTools(root);
                Check(later.LanguageNotice is { } notice && notice.Contains("no coding language", StringComparison.Ordinal) && notice.Contains("Idrak.Text.Plugins", StringComparison.Ordinal),
                    $"a project with no language registered: a notice naming the package: {later.LanguageNotice}");
            });
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void CodingLanguagesChoice()
    {
        string root = LanguageTemp("choice");
        string dotnet = Path.Combine(root, "dotnet"), node = Path.Combine(root, "node"), ng = Path.Combine(root, "ng"), go = Path.Combine(root, "go"), deep = Path.Combine(root, "deep");
        foreach (var folder in new[] { dotnet, node, ng, go, Path.Combine(deep, "src", "App") })
        {
            Directory.CreateDirectory(folder);
        }

        File.WriteAllText(Path.Combine(dotnet, "App.csproj"), "<Project />");
        File.WriteAllText(Path.Combine(dotnet, "Program.cs"), "return 0;");
        File.WriteAllText(Path.Combine(node, "package.json"), "{}");
        File.WriteAllText(Path.Combine(ng, "package.json"), "{}");
        File.WriteAllText(Path.Combine(ng, "angular.json"), "{}");
        File.WriteAllText(Path.Combine(go, "main.go"), "package main");
        File.WriteAllText(Path.Combine(deep, "src", "App", "App.csproj"), "<Project />");
        try
        {
            WithTextLanguages(() =>
            {
                string Names(string workspace, string choice) => string.Join(",", new CodingTools(workspace, new CodingToolOptions { Languages = choice }).Languages.Select(l => l.Name));
                Check(Names(dotnet, CodingLanguages.Detected) == "csharp", $"a .csproj workspace: {Names(dotnet, CodingLanguages.Detected)}");
                Check(Names(node, CodingLanguages.Detected) == "typescript", $"a package.json workspace: {Names(node, CodingLanguages.Detected)}");
                Check(Names(ng, CodingLanguages.Detected) == "typescript,angular", $"an Angular workspace: {Names(ng, CodingLanguages.Detected)}");
                Check(Names(deep, CodingLanguages.Detected) == "csharp", "a project two folders down (the default depth)");
                Check(new CodingTools(deep, new CodingToolOptions { Languages = CodingLanguages.Detected, DetectionDepth = 1 }).Languages.Count == 0, "not within a depth of 1");
                Check(Names(go, CodingLanguages.None) == "" && Names(dotnet, "python,csharp") == "python,csharp", "none, and names in their order");
                var detected = new CodingTools(dotnet, new CodingToolOptions { Languages = CodingLanguages.Detected });
                Check(detected.Commands.SequenceEqual(["dotnet", "git"]) && detected.IgnoredDirectories.Contains("bin") && !detected.IgnoredDirectories.Contains("node_modules")
                      && CodingAgent.SystemPrompt(detected.Languages).Contains("(C#: dotnet build, dotnet test).", StringComparison.Ordinal), "a detected language's defaults alone");
                Check(detected.LanguageNotice is null, "a detected workspace gets no notice");
                ThrowsNaming<NotSupportedException>(() => _ = new CodingTools(dotnet, new CodingToolOptions { Languages = "csharp,cobol" }), "a language not registered", "'cobol'", "not registered",
                    "Idrak.Text.Plugins");

                // A workspace no chosen language detects: a notice naming the plug-in (its files' language from Idrak.Data's table); the task's own commands still run.
                var pythonOnly = new CodingTools(dotnet, new CodingToolOptions { Languages = "python" });
                Check(pythonOnly.LanguageNotice is { } notice && notice.Contains("csharp", StringComparison.Ordinal) && notice.Contains("Idrak.Text.Plugins", StringComparison.Ordinal)
                      && notice.Contains("-language", StringComparison.Ordinal), $"a C# workspace with python alone chosen: {pythonOnly.LanguageNotice}");
                var goTools = new CodingTools(go);
                Check(goTools.LanguageNotice is { } goNotice && goNotice.Contains("'go'", StringComparison.Ordinal), $"a Go workspace, every language registered: {goTools.LanguageNotice}");
                Check(new CodingTools(go, new CodingToolOptions { Languages = CodingLanguages.None }).LanguageNotice is null, "none chosen: no notice");
                var own = new CodingTools(dotnet, new CodingToolOptions { Languages = "python", Commands = ["dotnet"] });
                Check(own.RunCommandAsync("dotnet --version").GetAwaiter().GetResult().Contains("exit code 0", StringComparison.Ordinal), "the given programs still run whatever the languages");

                WithEnvironment(new Dictionary<string, string?> { [CodingLanguages.EnvironmentVariable] = "detected" }, () =>
                    Check(new CodingTools(node).Languages.Select(l => l.Name).SequenceEqual(["typescript"]) && CodingLanguages.DefaultChoice == "detected", "IDRAK_CODING_LANGUAGES=detected"));
                WithEnvironment(new Dictionary<string, string?> { [CodingLanguages.EnvironmentVariable] = null }, () => Check(CodingLanguages.DefaultChoice == CodingLanguages.All, "unset: all"));

                Check(CodingLanguages.Merge([["a", "b", "d"], ["b", "c", "d", "e"], ["x"]]).SequenceEqual(["a", "b", "c", "d", "e", "x"]), "the merge puts a new item after the one before it in its list");
                Check(CodingLanguages.Merge([["dotnet"], ["npm", "npx", "node", "tsc"], ["npm", "npx", "ng", "node", "tsc"], ["git"]]).SequenceEqual(["dotnet", "npm", "npx", "ng", "node", "tsc", "git"]),
                    "the programs' merge");
                ThrowsNaming<ArgumentException>(() => CodingLanguages.Register(new CodingLanguage("all", "x")), "a language named all", "all");
            });
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void LanguagePluginRegisters(IIdrakPlugin plugin, string name, string ns, string[] languages)
    {
        Check(plugin.Name == name && plugin.GetType().Namespace == ns && TextPlugins.All.Contains(plugin) && plugin.GetType().Assembly.GetName().Name == TextPlugins.Package.Name,
            $"{name}: the plug-in \"{name}\" of TextPlugins.All, in {ns} ({plugin.Name}, {plugin.GetType().Namespace})");
        WithoutLanguages(() =>
        {
            string found = string.Join(",", CodingLanguages.Names);
            IdrakPlugins.Register(plugin);
            IdrakPlugins.Register(plugin);
            Check(CodingLanguages.Names.SequenceEqual(languages) && languages.All(l => CodingLanguages.Origin(l) == TextPlugins.Package.Name),
                $"{name} registers {string.Join(", ", languages)} once, from Idrak.Text.Plugins: {string.Join(", ", CodingLanguages.Names)}");
            if (name == "python-language")
            {
                var python = CodingLanguages.Get("python");
                Check(python.Commands.Count == 0 && python.PromptLines.Count == 0 && python.DefaultSetup.Count == 0 && python.DefaultVerify.Count == 0
                      && python.IgnoredDirectories.SequenceEqual(["__pycache__", ".venv"]), "python: its two folders only");
            }

            IdrakPlugins.Unregister(plugin);
            Check(string.Join(",", CodingLanguages.Names) == found, $"Unregister leaves CodingLanguages as found ({found}): {string.Join(",", CodingLanguages.Names)}");
        });
    }

    private static void DataFilesSkippedFolders()
    {
        string root = LanguageTemp("data");
        try
        {
            string[] files =
            [
                "src/App.cs", "web/app.ts", "bin/Debug/gen.cs", "obj/x.cs", "node_modules/lib/index.js", "dist/main.js", ".angular/cache.js", "__pycache__/m.py", ".venv/lib/site.py",
                ".git/hooks/pre.py", ".vs/state.json", ".idea/workspace.xml", "coverage/report.js", "out/o.js",
            ];
            foreach (var file in files)
            {
                string path = Path.Combine(root, "repo", file);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                File.WriteAllText(path, "x = 1");
            }

            string folder = Path.Combine(root, "repo");
            List<string> Paths(IEnumerable<System.Text.Json.Nodes.JsonObject> rows) => [.. rows.Select(r => ((string)r["path"]!).Replace('\\', '/')).Order(StringComparer.Ordinal)];
            var documents = new ReadOptions { Documents = true, Text = TextRows.Document };
            var alone = Paths(DatasetRows.FromFolder(folder, options: documents));
            Check(ReadOptions.DefaultSkippedFolders.SequenceEqual([".git", ".vs", ".idea"]) && ReadOptions.Default.SkippedFolders.SequenceEqual([".git", ".vs", ".idea"]), "the library's default");
            Check(alone.SequenceEqual(files.Where(f => !f.StartsWith(".git/", StringComparison.Ordinal) && !f.StartsWith(".vs/", StringComparison.Ordinal) && !f.StartsWith(".idea/", StringComparison.Ordinal))
                    .Order(StringComparer.Ordinal)), $"Idrak.Data alone skips only .git, .vs and .idea: {string.Join(", ", alone)}");
            var some = Paths(DatasetRows.FromFolder(folder, options: documents with { SkippedFolders = [.. ReadOptions.DefaultSkippedFolders, "BIN"] }));
            Check(!some.Any(p => p.StartsWith("bin/", StringComparison.Ordinal)) && some.Count == alone.Count - 1, "a folder added to SkippedFolders is skipped (ignoring case)");

            // The tool's data commands: the folders they always skipped (Idrak.Data's old list), from the registered languages.
            WithTextLanguages(() =>
            {
                string[] oldSkipped = ["bin", "obj", "node_modules", ".git", ".vs", ".idea", "dist", ".angular", "__pycache__", ".venv"];
                Check(CodeFolders.Skipped.Order(StringComparer.Ordinal).SequenceEqual(oldSkipped.Order(StringComparer.Ordinal)), $"the old folders: {string.Join(", ", CodeFolders.Skipped)}");
                var cli = Paths(DatasetRows.FromFolder(folder, options: CodeFolders.Read with { Documents = true, Text = TextRows.Document }));
                Check(cli.SequenceEqual(["coverage/report.js", "out/o.js", "src/App.cs", "web/app.ts"]), $"the tool's reading: {string.Join(", ", cli)}");
                var (exit, output, error) = Cli("data", "count", folder + "?documents=true");
                Check(exit == 0 && output.Contains(": 4 rows", StringComparison.Ordinal), $"idrak data count on the folder as documents: {exit} {output} {error}");

                string archive = Path.Combine(root, "repo.tar.gz");
                using (var file = File.Create(archive))
                using (var gzip = new GZipStream(file, CompressionLevel.Fastest))
                using (var tar = new TarWriter(gzip))
                {
                    foreach (var name in files)
                    {
                        tar.WriteEntry(new PaxTarEntry(TarEntryType.RegularFile, "repo-abc/" + name) { DataStream = new MemoryStream(Encoding.UTF8.GetBytes("x = 1")) });
                    }
                }

                var inArchive = DatasetRows.FromFile(archive, documents).Count();
                var cliArchive = DatasetRows.FromFile(archive, CodeFolders.Read with { Documents = true, Text = TextRows.Document }).Count();
                Check(inArchive == alone.Count && cliArchive == 4, $"in an archive the same: Idrak.Data alone {inArchive}, the tool's {cliArchive}");
            });
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
