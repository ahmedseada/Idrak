// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.CodeAnalysis.MSBuild;

// The pre-build check: a project and every project it references, loaded as MSBuild loads them (the Release configuration
// by default), each compiled in memory with its own options, source generators and analyzers (nothing emitted, no file
// written), in dependency order. Every error is printed (a warning the project treats as an error is one: it fails the
// build), grouped by project, then a count; other warnings only as a count. A project whose dependencies have errors is
// still checked, against what they declare, so one pass finds what a build would show one failing project at a time.
// Exit code 0 when there is no error (whatever the warnings), 1 when there are, 2 when a project cannot load.
//
//   dotnet run -c Release --project tools/check -- tests/Idrak.Tests/Idrak.Tests.csproj [-c Debug]

string? path = null;
string configuration = "Release";
for (int a = 0; a < args.Length; a++)
{
    if (args[a] is "-c" or "--configuration" && a + 1 < args.Length)
    {
        configuration = args[++a];
    }
    else if (args[a] is "-h" or "--help")
    {
        Console.WriteLine("Usage: dotnet run -c Release --project tools/check -- [project.csproj] [-c Configuration]");
        Console.WriteLine("Checks the project and every project it references for errors without building (default: tests/Idrak.Tests/Idrak.Tests.csproj, Release).");
        return 0;
    }
    else
    {
        path = args[a];
    }
}

path ??= DefaultProject();
if (path is null || !File.Exists(path))
{
    Console.Error.WriteLine($"No project at '{path ?? "tests/Idrak.Tests/Idrak.Tests.csproj"}': give a .csproj path.");
    return 2;
}

var instance = MSBuildLocator.RegisterDefaults();                     // before any MSBuild type loads (Checker below)
Console.WriteLine($"MSBuild {instance.Version} ({instance.MSBuildPath}), configuration {configuration}");
return await Checker.Run(Path.GetFullPath(path), configuration);

// tests/Idrak.Tests/Idrak.Tests.csproj under the repository (the folder holding Idrak.slnx) the current folder is in.
static string? DefaultProject()
{
    for (var folder = new DirectoryInfo(Environment.CurrentDirectory); folder is not null; folder = folder.Parent)
    {
        if (File.Exists(Path.Combine(folder.FullName, "Idrak.slnx")))
        {
            return Path.Combine(folder.FullName, "tests", "Idrak.Tests", "Idrak.Tests.csproj");
        }
    }

    return null;
}

internal static class Checker
{
    public static async Task<int> Run(string path, string configuration)
    {
        var total = Stopwatch.StartNew();
        using var workspace = MSBuildWorkspace.Create(new Dictionary<string, string> { ["Configuration"] = configuration });
        var progress = new Progress<ProjectLoadProgress>(p =>
        {
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"load {p.Operation.ToString().ToLowerInvariant(),-8} {Path.GetFileNameWithoutExtension(p.FilePath)}{(p.TargetFramework is { } tfm ? $" ({tfm})" : "")}  {p.ElapsedTime.TotalSeconds:0.0} s"));
        });

        Project root;
        try
        {
            root = await workspace.OpenProjectAsync(path, progress);
        }
        catch (Exception e) when (e is InvalidOperationException or IOException)
        {
            Console.Error.WriteLine($"Could not load {path}: {e.Message}");
            return 2;
        }

        bool loadFailed = false;
        foreach (var diagnostic in workspace.Diagnostics)
        {
            loadFailed |= diagnostic.Kind == WorkspaceDiagnosticKind.Failure;
            Console.WriteLine($"load {diagnostic.Kind.ToString().ToLowerInvariant()}: {diagnostic.Message}");
        }

        var solution = root.Solution;
        var graph = solution.GetProjectDependencyGraph();
        var order = graph.GetTopologicallySortedProjects().Select(id => solution.GetProject(id)!).ToList();
        var loadTime = total.Elapsed;
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"loaded {order.Count} projects in {loadTime.TotalSeconds:0.0} s; checking each once its dependencies are, {Environment.ProcessorCount} at a time"));

        // Each project starts when the projects it references are checked (their metadata images made), as many at a time
        // as there are processors; the results are listed in dependency order.
        var images = new System.Collections.Concurrent.ConcurrentDictionary<string, MetadataReference>(StringComparer.Ordinal);
        var results = new (List<Diagnostic> Errors, int Warnings)[order.Count];
        var tasks = new Dictionary<ProjectId, Task>();
        using var slots = new SemaphoreSlim(Environment.ProcessorCount);
        int started = 0, done = 0;
        for (int i = 0; i < order.Count; i++)
        {
            int index = i;
            var project = order[i];
            var dependencies = graph.GetProjectsThatThisProjectDirectlyDependsOn(project.Id).Where(tasks.ContainsKey).Select(id => tasks[id]).ToArray();
            tasks[project.Id] = Task.Run(async () =>
            {
                await Task.WhenAll(dependencies);
                await slots.WaitAsync();
                try
                {
                    var watch = Stopwatch.StartNew();
                    Console.WriteLine($"[{Interlocked.Increment(ref started)}/{order.Count} started] {project.Name}");
                    var result = await Check(project, images);
                    results[index] = (result.Errors, result.Warnings);
                    images[project.AssemblyName] = result.Image;
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"[{Interlocked.Increment(ref done)}/{order.Count} done] {project.Name}: {result.Errors.Count} error{(result.Errors.Count == 1 ? "" : "s")}, {result.Warnings} warning{(result.Warnings == 1 ? "" : "s")}, {watch.Elapsed.TotalSeconds:0.0} s"));
                }
                finally
                {
                    slots.Release();
                }
            });
        }

        await Task.WhenAll(tasks.Values);
        var errors = new List<(string Project, Diagnostic Diagnostic)>();
        int failing = 0, warnings = 0;
        for (int i = 0; i < order.Count; i++)
        {
            failing += results[i].Errors.Count > 0 ? 1 : 0;
            warnings += results[i].Warnings;
            errors.AddRange(results[i].Errors.Select(d => (order[i].Name, d)));
        }

        foreach (var group in errors.GroupBy(e => e.Project))
        {
            Console.WriteLine();
            Console.WriteLine($"{group.Key}:");
            foreach (var (_, diagnostic) in group.OrderBy(e => e.Diagnostic.Location.GetLineSpan().Path, StringComparer.Ordinal)
                         .ThenBy(e => e.Diagnostic.Location.GetLineSpan().StartLinePosition.Line))
            {
                Console.WriteLine($"  {Format(diagnostic)}");
            }
        }

        Console.WriteLine();
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{errors.Count} error{(errors.Count == 1 ? "" : "s")} in {failing} of {order.Count} projects, {warnings} warning{(warnings == 1 ? "" : "s")} (not listed; they do not fail the build); loaded in {loadTime.TotalSeconds:0.0} s, checked in {(total.Elapsed - loadTime).TotalSeconds:0.0} s, {total.Elapsed.TotalSeconds:0.0} s in all"));
        return loadFailed ? 2 : errors.Count > 0 ? 1 : 0;
    }

    // A project's errors as its build reports them: its source generators run on its sources as the compiler runs them
    // (their own errors kept), the compilation's diagnostics, then its analyzers' (with its options: .editorconfig,
    // the build's global config), each at the severity the project's options give it (warnings as errors included).
    //
    // The compilation is the project's own sources, its metadata references and options, and, for each project it
    // references, that project's metadata image (made in memory once it is checked), as the build references a project's
    // assembly: so what another assembly keeps internal stays out of reach, as in the build. A project whose declarations
    // have errors has no image; its compilation stands in for it, so the projects using it are still checked.
    private static async Task<(List<Diagnostic> Errors, int Warnings, MetadataReference Image)> Check(Project project,
        IReadOnlyDictionary<string, MetadataReference> images)
    {
        var found = new List<Diagnostic>();
        int warnings = 0;
        var trees = new List<SyntaxTree>();
        foreach (var document in project.Documents)
        {
            if (await document.GetSyntaxTreeAsync() is { } tree)
            {
                trees.Add(tree);
            }
        }

        var references = new List<MetadataReference>(project.MetadataReferences);
        foreach (var reference in project.ProjectReferences)
        {
            if (project.Solution.GetProject(reference.ProjectId) is { } dependency && images.TryGetValue(dependency.AssemblyName, out var image))
            {
                references.Add(WithProperties(image, new MetadataReferenceProperties(MetadataImageKind.Assembly, reference.Aliases, reference.EmbedInteropTypes)));
            }
        }

        Compilation compilation = CSharpCompilation.Create(project.AssemblyName, trees, references,
            ((CSharpCompilationOptions)project.CompilationOptions!).WithConcurrentBuild(true));
        var generators = project.AnalyzerReferences.SelectMany(r => r.GetGenerators(project.Language)).ToImmutableArray();
        if (generators.Length > 0 && project.ParseOptions is CSharpParseOptions parse)
        {
            var driver = CSharpGeneratorDriver.Create(generators, project.AnalyzerOptions.AdditionalFiles, parse, project.AnalyzerOptions.AnalyzerConfigOptionsProvider);
            driver.RunGeneratorsAndUpdateCompilation(compilation, out var output, out var generated);
            compilation = output;
            foreach (var diagnostic in generated)
            {
                switch (Effective(diagnostic, output.Options))
                {
                    case DiagnosticSeverity.Error:
                        found.Add(diagnostic);
                        break;
                    case DiagnosticSeverity.Warning:
                        warnings++;
                        break;
                }
            }
        }

        // The compiler's diagnostics and the analyzers' from one binding of the sources (the analyzers' compilation reports both).
        var analyzers = project.AnalyzerReferences.SelectMany(r => r.GetAnalyzers(project.Language)).ToImmutableArray();
        var options = new CompilationWithAnalyzersOptions(project.AnalyzerOptions, onAnalyzerException: null, concurrentAnalysis: true, logAnalyzerExecutionTime: false);
        var all = analyzers.Length > 0 ? await compilation.WithAnalyzers(analyzers, options).GetAllDiagnosticsAsync() : compilation.GetDiagnostics();

        foreach (var diagnostic in all)
        {
            if (diagnostic.Severity == DiagnosticSeverity.Error)
            {
                found.Add(diagnostic);
            }
            else if (diagnostic.Severity == DiagnosticSeverity.Warning)
            {
                warnings++;
            }
        }

        // Metadata only, in memory: the declarations its dependents bind against (no method body is compiled).
        using var stream = new MemoryStream();
        var emitted = compilation.Emit(stream, options: new EmitOptions(metadataOnly: true));
        MetadataReference own = emitted.Success ? MetadataReference.CreateFromImage(stream.ToArray()) : compilation.ToMetadataReference();
        return (found, warnings, own);
    }

    // A dependency's image under the project's reference to it (its aliases).
    private static MetadataReference WithProperties(MetadataReference image, MetadataReferenceProperties properties) => image switch
    {
        PortableExecutableReference metadata => metadata.WithProperties(properties),
        CompilationReference compilation => compilation.WithProperties(properties),
        _ => image,
    };

    // A generator's diagnostic at the severity the compilation's options give it (the compiler applies them the same way):
    // an error, a warning the project treats as an error, a warning, or nothing to report.
    private static DiagnosticSeverity? Effective(Diagnostic diagnostic, CompilationOptions options)
    {
        if (diagnostic.Severity is DiagnosticSeverity.Error or DiagnosticSeverity.Hidden or DiagnosticSeverity.Info)
        {
            return diagnostic.Severity == DiagnosticSeverity.Error ? DiagnosticSeverity.Error : null;
        }

        var report = options.SpecificDiagnosticOptions.TryGetValue(diagnostic.Id, out var specific) ? specific : ReportDiagnostic.Default;
        return report switch
        {
            ReportDiagnostic.Error => DiagnosticSeverity.Error,
            ReportDiagnostic.Suppress or ReportDiagnostic.Hidden or ReportDiagnostic.Info => null,
            _ => options.GeneralDiagnosticOption == ReportDiagnostic.Error ? DiagnosticSeverity.Error
                : options.GeneralDiagnosticOption == ReportDiagnostic.Suppress ? null : DiagnosticSeverity.Warning,
        };
    }

    // "path(line,col): error ID: message", as the build prints it.
    private static string Format(Diagnostic diagnostic)
    {
        var span = diagnostic.Location.GetLineSpan();
        string where = span.Path is { Length: > 0 } file
            ? string.Create(CultureInfo.InvariantCulture, $"{file}({span.StartLinePosition.Line + 1},{span.StartLinePosition.Character + 1}): ")
            : "";
        return $"{where}error {diagnostic.Id}: {diagnostic.GetMessage(CultureInfo.InvariantCulture)}{(diagnostic.IsWarningAsError ? " (warning as error)" : "")}";
    }
}
