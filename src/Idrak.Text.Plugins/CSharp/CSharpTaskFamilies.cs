// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Data.Code;

namespace Idrak.Text.Plugins.CSharp;

/// <summary>
/// How the C# task families check a task: the commands that run a project's tests, the variable a named-entry test runner
/// reads its filter from and the variables it runs with, how many tests check a method, and the smallest body worth a task.
/// </summary>
public sealed record CSharpTaskOptions
{
    /// <summary>
    /// The command that runs a project's named test entries, <c>{project}</c> its path: "dotnet run --project {project}"
    /// (a test list run as a program, its filter in <see cref="FilterVariable"/>).
    /// </summary>
    public string NamedTestCommand { get; init; } = "dotnet run --project {project}";

    /// <summary>The command that runs a project's attribute tests, <c>{project}</c> its path and <c>{filter}</c> the tests: "dotnet test {project} --filter {filter}".</summary>
    public string AttributeTestCommand { get; init; } = "dotnet test {project} --filter \"{filter}\"";

    /// <summary>The variable a named-entry runner reads the names it runs from ("IDRAK_FILTER": texts separated by '|', each a name or part of one).</summary>
    public string FilterVariable { get; init; } = "IDRAK_FILTER";

    /// <summary>Other variables a named-entry test runs with: <c>IDRAK_DEVICES=cpu</c> (a check runs on the CPU, the same on every machine).</summary>
    public IReadOnlyDictionary<string, string> TestVariables { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal) { ["IDRAK_DEVICES"] = "cpu" };

    /// <summary>The most tests that check one method (the first that reach it, in file order): 8.</summary>
    public int MaxTests { get; init; } = 8;

    /// <summary>The smallest body worth a task, in characters of its tokens (white space, comments, braces left out): 24.</summary>
    public int MinBodySize { get; init; } = 24;
}

/// <summary>
/// The C# task families of the plug-in (plan 36), registered into <see cref="TaskFamilies"/> with the language
/// (<see cref="CSharpLanguagePlugin"/>): "method-body" (<see cref="MethodBody"/>) and "test-subject" (<see cref="TestSubject"/>).
/// Both read C# as Roslyn syntax trees (<c>Microsoft.CodeAnalysis.CSharp</c>, each file parsed once for both families;
/// syntax only, no compilation or semantic model) and take the tests that reach a method from the tests' own code: a test
/// reaches a method it calls by name, directly or through a helper of its test files, when no other method of the source
/// has that name.
/// </summary>
public static class CSharpTaskFamilies
{
    /// <summary>"method-body": a method's body removed (its signature and documentation kept); the check is the project's tests that reach it.</summary>
    public const string MethodBodyName = "method-body";

    /// <summary>"test-subject": the code a named test reaches removed; the check is that test, run by its name.</summary>
    public const string TestSubjectName = "test-subject";

    /// <summary>The method-body family with the default <see cref="CSharpTaskOptions"/>.</summary>
    public static ITaskFamily MethodBody { get; } = new MethodBodyFamily(new CSharpTaskOptions());

    /// <summary>The test-subject family with the default <see cref="CSharpTaskOptions"/>.</summary>
    public static ITaskFamily TestSubject { get; } = new TestSubjectFamily(new CSharpTaskOptions());

    /// <summary>The method-body family checking with <paramref name="options"/> (register it over the plug-in's to use them).</summary>
    public static ITaskFamily CreateMethodBody(CSharpTaskOptions options) => new MethodBodyFamily(options ?? throw new ArgumentNullException(nameof(options)));

    /// <summary>The test-subject family checking with <paramref name="options"/> (register it over the plug-in's to use them).</summary>
    public static ITaskFamily CreateTestSubject(CSharpTaskOptions options) => new TestSubjectFamily(options ?? throw new ArgumentNullException(nameof(options)));

    private sealed class MethodBodyFamily(CSharpTaskOptions options) : ITaskFamily
    {
        public string Name => MethodBodyName;

        public string Summary => "a C# method's body removed (its signature and documentation kept); checked by the project's tests that reach it";

        public IReadOnlyList<string> Languages { get; } = [CSharpLanguagePlugin.LanguageName];

        public async IAsyncEnumerable<TaskDraft> DraftAsync(CodeSource source, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            var repository = CSharpRepository.For(source);
            foreach (var (file, method) in repository.SourceMethods)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!repository.IsSubject(file, method, options))
                {
                    continue;
                }

                var tests = repository.TestsReaching(method.Name).Take(options.MaxTests).ToList();
                if (tests.Count == 0)
                {
                    continue;
                }

                string documentation = CSharpSyntax.Documentation(method);
                var prompt = new StringBuilder()
                    .Append(CultureInfo($"In {file.Path}, the body of {method.Type}.{method.Name} was removed (it throws NotImplementedException). "))
                    .Append("Write it again so the method does what its signature and documentation say; the project's tests that use it must pass.\n\n")
                    .Append(documentation.Length > 0 ? documentation + "\n" : "")
                    .Append(method.Header);
                yield return repository.Draft(MethodBodyName, $"{file.Path}#{method.Type}.{method.Name}", prompt.ToString(), file, method, tests, options);
            }
        }
    }

    private sealed class TestSubjectFamily(CSharpTaskOptions options) : ITaskFamily
    {
        public string Name => TestSubjectName;

        public string Summary => "the C# code a named test reaches removed; checked by that test, run by its name";

        public IReadOnlyList<string> Languages { get; } = [CSharpLanguagePlugin.LanguageName];

        public async IAsyncEnumerable<TaskDraft> DraftAsync(CodeSource source, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            var repository = CSharpRepository.For(source);
            foreach (var test in repository.Tests.Where(t => t.Entry.Kind == CSharpTestKind.Named))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var subject = test.Reached.Select(repository.Unique).FirstOrDefault(s => s is { } found && repository.IsSubject(found.File, found.Method, options));
                if (subject is not { } chosen)
                {
                    continue;
                }

                string prompt = CultureInfo($"The test \"{test.Entry.Name}\" ({test.RunProject}) fails: code it relies on was removed. Find that code and write it again so the test passes; do not change the test.");
                yield return repository.Draft(TestSubjectName, test.Entry.Name, prompt, chosen.File, chosen.Method, [test], options);
            }
        }
    }

    private static string CultureInfo(FormattableString text) => FormattableString.Invariant(text);
}

// A C# repository as the families see it: its files read and parsed once per walk (shared by both families through the
// list of files the extractor gives them), each file's project, the tests and the methods they reach.
internal sealed class CSharpRepository
{
    private static readonly ConditionalWeakTable<IReadOnlyList<CodeSourceFile>, CSharpRepository> Cache = new();

    internal sealed record SourceFile(string Path, CSharpSyntax Syntax, Project? Project)
    {
        public bool IsTest => Project?.IsTest ?? false;
    }

    internal sealed record Project(string FullPath, string Path, bool IsTest, bool IsExe, IReadOnlyList<string> References);

    internal sealed record Test(SourceFile File, CSharpTestEntry Entry, string RunProject, IReadOnlyList<string> Reached);

    private readonly string _root;
    private readonly Dictionary<string, (SourceFile File, CSharpMethod Method)?> _unique = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<Test>> _reaching = new(StringComparer.Ordinal);

    private CSharpRepository(CodeSource source)
    {
        _root = Path.GetFullPath(source.Root);
        var files = source.Files.Count > 0 ? source.Files : CodeExtractor.Walk(source);
        var projects = new Dictionary<string, Project?>(StringComparer.OrdinalIgnoreCase);
        var parsed = new List<SourceFile>();
        foreach (var file in files.Where(f => f.Language == CSharpLanguagePlugin.LanguageName && f.Path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase)))
        {
            string full = Path.Combine(_root, file.Path);
            parsed.Add(new SourceFile(file.Path, CSharpSyntax.Read(File.ReadAllBytes(full)), ProjectOf(Path.GetDirectoryName(full)!, projects)));
        }

        Files = parsed;
        SourceMethods = [.. parsed.Where(f => !f.IsTest).SelectMany(f => f.Syntax.Methods.Select(m => (f, m)))];
        foreach (var group in SourceMethods.GroupBy(p => p.Method.Name, StringComparer.Ordinal))
        {
            _unique[group.Key] = group.Count() == 1 ? group.First() : null;
        }

        // Test helpers by name: a named entry that runs a method group, and the helpers a test calls, are read through them.
        var helpers = parsed.Where(f => f.IsTest).SelectMany(f => f.Syntax.Methods.Select(m => (File: f, Method: m)))
            .GroupBy(p => p.Method.Name, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.ToList(), StringComparer.Ordinal);
        var tests = new List<Test>();
        foreach (var file in parsed.Where(f => f.IsTest))
        {
            foreach (var entry in file.Syntax.Tests)
            {
                var reached = Reached(file, entry, helpers);
                string run = RunProject(file.Project, projects);
                var test = new Test(file, entry, run, reached);
                tests.Add(test);
                foreach (string name in reached)
                {
                    if (!_reaching.TryGetValue(name, out var list))
                    {
                        _reaching[name] = list = [];
                    }

                    list.Add(test);
                }
            }
        }

        Tests = tests;
    }

    public IReadOnlyList<SourceFile> Files { get; }

    public IReadOnlyList<(SourceFile File, CSharpMethod Method)> SourceMethods { get; }

    public IReadOnlyList<Test> Tests { get; }

    public static CSharpRepository For(CodeSource source) =>
        source.Files.Count > 0 ? Cache.GetValue(source.Files, _ => new CSharpRepository(source)) : new CSharpRepository(source);

    // The source method of that name when it is the only one; null otherwise.
    public (SourceFile File, CSharpMethod Method)? Unique(string name) => _unique.TryGetValue(name, out var found) ? found : null;

    public IEnumerable<Test> TestsReaching(string name) => _reaching.TryGetValue(name, out var tests) ? tests : [];

    // A method worth a task: the only one of its name, with a body that does something and does not already throw.
    public bool IsSubject(SourceFile file, CSharpMethod method, CSharpTaskOptions options) =>
        Unique(method.Name) is { } only && ReferenceEquals(only.Method, method) && ReferenceEquals(only.File, file)
        && CSharpSyntax.BodySize(method) >= options.MinBodySize && !CSharpSyntax.Unimplemented(method);

    public TaskDraft Draft(string family, string id, string prompt, SourceFile file, CSharpMethod method, IReadOnlyList<Test> tests, CSharpTaskOptions options)
    {
        var commands = new List<string>();
        var environment = new Dictionary<string, string>(StringComparer.Ordinal);
        var named = tests.Where(t => t.Entry.Kind == CSharpTestKind.Named && !t.Entry.Name.Contains('|', StringComparison.Ordinal)).ToList();
        foreach (string project in named.Select(t => t.RunProject).Distinct(StringComparer.Ordinal))
        {
            commands.Add(options.NamedTestCommand.Replace("{project}", project, StringComparison.Ordinal));
        }

        if (named.Count > 0)
        {
            environment[options.FilterVariable] = string.Join("|", named.Select(t => t.Entry.Name));
            foreach (var (name, value) in options.TestVariables)
            {
                environment[name] = value;
            }
        }

        foreach (var group in tests.Where(t => t.Entry.Kind == CSharpTestKind.Attribute).GroupBy(t => t.File.Project?.Path ?? "", StringComparer.Ordinal))
        {
            string filter = string.Join("|", group.Select(t => "FullyQualifiedName~" + t.Entry.Name));
            commands.Add(options.AttributeTestCommand.Replace("{project}", group.Key, StringComparison.Ordinal).Replace("{filter}", filter, StringComparison.Ordinal));
        }

        var first = tests[0];
        string group0 = first.Entry.Kind == CSharpTestKind.Named && first.Entry.Name.IndexOf(':', StringComparison.Ordinal) is var colon and > 0
            ? first.Entry.Name[..colon] : first.File.Project?.Path ?? first.File.Path;
        return new TaskDraft(id, family, prompt)
        {
            Language = CSharpLanguagePlugin.LanguageName,
            Workspace = [new CodeFile(file.Path, file.Syntax.WithoutBody(method))],
            Solution = [new CodeFile(file.Path, file.Syntax.Bytes)],
            VerifyCommands = commands,
            Environment = environment,
            Tags = [CSharpLanguagePlugin.LanguageName, family],
            Provenance = new TaskProvenance { Path = file.Path, Test = first.Entry.Name, Group = group0 },
        };
    }

    // The names a test calls: its own code's, then (one level) those of the test helpers it calls or runs by name.
    private static List<string> Reached(SourceFile file, CSharpTestEntry entry, Dictionary<string, List<(SourceFile File, CSharpMethod Method)>> helpers)
    {
        var names = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        void Add(IEnumerable<string> found)
        {
            foreach (string name in found)
            {
                if (seen.Add(name))
                {
                    names.Add(name);
                }
            }
        }

        var direct = entry.Method is { } runs && entry.Kind == CSharpTestKind.Named && helpers.TryGetValue(runs, out var bodies)
            ? bodies.OrderBy(b => b.File == file ? 0 : 1).Take(1).SelectMany(b => CSharpSyntax.CalledNames(BodyOf(b.Method))).ToList()
            : [.. CSharpSyntax.CalledNames(entry.Body)];
        Add(direct);
        foreach (string name in direct)
        {
            if (helpers.TryGetValue(name, out var helper))
            {
                foreach (var (_, method) in helper.Take(4))
                {
                    Add(CSharpSyntax.CalledNames(BodyOf(method)));
                }
            }
        }

        return names;
    }

    private static Microsoft.CodeAnalysis.SyntaxNode BodyOf(CSharpMethod method) => (Microsoft.CodeAnalysis.SyntaxNode?)method.Node.Body ?? method.Node.ExpressionBody!;

    // The nearest project above a folder (its .csproj), read once: a test project by its name or its test SDK, a program by
    // its output type, and the projects it references.
    private Project? ProjectOf(string folder, Dictionary<string, Project?> projects)
    {
        if (projects.TryGetValue(folder, out var known))
        {
            return known;
        }

        Project? project = null;
        string? file = Directory.Exists(folder) ? Directory.GetFiles(folder, "*.csproj").Order(StringComparer.Ordinal).FirstOrDefault() : null;
        if (file is not null)
        {
            byte[] bytes = File.ReadAllBytes(file);
            string text = Encoding.UTF8.GetString(bytes);
            bool test = Path.GetFileNameWithoutExtension(file).Contains("test", StringComparison.OrdinalIgnoreCase)
                        || text.Contains("Microsoft.NET.Test.Sdk", StringComparison.OrdinalIgnoreCase);
            bool exe = text.Contains("<OutputType>Exe</OutputType>", StringComparison.OrdinalIgnoreCase);
            var references = new List<string>();
            const string Marker = "<ProjectReference Include=\"";
            for (int at = text.IndexOf(Marker, StringComparison.OrdinalIgnoreCase); at >= 0; at = text.IndexOf(Marker, at + 1, StringComparison.OrdinalIgnoreCase))
            {
                int start = at + Marker.Length, end = text.IndexOf('"', start);
                if (end > start)
                {
                    references.Add(Path.GetFullPath(Path.Combine(folder, text[start..end].Replace('\\', Path.DirectorySeparatorChar))));
                }
            }

            project = new Project(Path.GetFullPath(file), Path.GetRelativePath(_root, file).Replace('\\', '/'), test, exe, references);
        }
        else if (!Path.GetFullPath(folder).Equals(_root, StringComparison.OrdinalIgnoreCase) && Path.GetDirectoryName(folder) is { } parent
                 && Path.GetFullPath(parent).StartsWith(_root, StringComparison.OrdinalIgnoreCase))
        {
            project = ProjectOf(parent, projects);
        }

        projects[folder] = project;
        return project;
    }

    // The project that runs a test project's named entries: itself when it is a program, else a program that references it.
    private static string RunProject(Project? project, Dictionary<string, Project?> projects)
    {
        if (project is null)
        {
            return "";
        }

        if (project.IsExe)
        {
            return project.Path;
        }

        var runner = projects.Values.OfType<Project>().Where(p => p.IsExe && p.References.Contains(project.FullPath, StringComparer.OrdinalIgnoreCase))
            .OrderBy(p => p.Path, StringComparer.Ordinal).FirstOrDefault();
        return runner?.Path ?? project.Path;
    }
}
