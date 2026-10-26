// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Tests.Runner;

// Where the tests find the repository's files (plan 34, M2): one root (the folder holding Idrak.slnx) and one data
// folder (tests/Idrak.Tests/data, or a copy a phone carries), set by the runner's options and else found above the
// program's folder. Where there is no repository (an app on a phone), a lookup throws TestNeedsRepositoryException and
// the runner reports the test as not run here, naming what it needed.
internal static class TestPaths
{
    internal static readonly Lazy<string?> Found = new(() =>
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Idrak.slnx")))
            {
                return dir.FullName;
            }
        }

        return null;
    });

    internal static volatile string? _root, _data;
    internal static volatile bool _rootGiven;

    /// <summary>The run's settings: a root of its own (or null to find it above the program's folder) and a data folder of its own (or null for the root's).</summary>
    internal static void Configure(string? root, string? data)
    {
        _root = root;
        _rootGiven = root is not null;
        _data = data;
    }

    /// <summary>The repository's root, or null when there is none here.</summary>
    internal static string? Root => _rootGiven ? (File.Exists(Path.Combine(_root!, "Idrak.slnx")) ? Path.GetFullPath(_root!) : null) : Found.Value;

    /// <summary>The run's own root as it was given, or null when it is found above the program's folder.</summary>
    internal static string? GivenRoot => _rootGiven ? _root : null;

    /// <summary>Where the run's own data folder is, or null when it is the repository's.</summary>
    internal static string? GivenData => _data;

    /// <summary>Where the root was looked for, for the message.</summary>
    internal static string Searched => _rootGiven ? $"no Idrak.slnx in {_root}" : $"no Idrak.slnx above {AppContext.BaseDirectory}";
}

internal abstract partial class Tests
{
    /// <summary>The repository's root (the folder holding Idrak.slnx); where there is none, the test is not run here.</summary>
    /// <exception cref="TestNeedsRepositoryException">There is no repository here: the message names the test that asked.</exception>
    internal static string RepositoryRoot([CallerMemberName] string test = "") =>
        TestPaths.Root ?? throw new TestNeedsRepositoryException($"the checkout, read by {test}; {TestPaths.Searched}");

    /// <summary>A path under the repository's root (<paramref name="parts"/> joined); where there is none, the test is not run here, naming the path.</summary>
    internal static string RepositoryPath(params string[] parts) =>
        Path.Combine([TestPaths.Root ?? throw new TestNeedsRepositoryException(string.Join('/', parts)), .. parts]);

    /// <summary>The test data: the run's data folder, else the repository's tests/Idrak.Tests/data; with neither, the test is not run here.</summary>
    internal static string DataFolder() =>
        TestPaths.GivenData ?? Path.Combine(TestPaths.Root ?? throw new TestNeedsRepositoryException("tests/Idrak.Tests/data"), "tests", "Idrak.Tests", "data");

    /// <summary>
    /// A file or folder of the test data (<paramref name="relative"/> under <see cref="DataFolder"/>). One the repository's
    /// data folder lacks fails the test; one a data folder of the run's own lacks (a phone's copy holds what it carries)
    /// is not run here, naming it.
    /// </summary>
    internal static string TestData(string relative)
    {
        string folder = DataFolder(), candidate = Path.Combine(folder, relative);
        if (File.Exists(candidate) || Directory.Exists(candidate))
        {
            return candidate;
        }

        throw TestPaths.GivenData is not null
            ? new TestNeedsRepositoryException($"tests/Idrak.Tests/data/{relative.Replace('\\', '/')}, not in the data folder {folder}")
            : new FileNotFoundException($"Test data '{relative}' not found in {folder}.");
    }
}
