// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using Idrak.Tests.Runner;

// Plan 34, M2: this program's list is the test library's (TestSuite.All, what an app runs) with this program's own
// groups (those that need the desktop) between, so both run the library's tests in the same order.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] DesktopRunnerGroup =
    [
        ("test runner: the library's list (TestSuite.All) is this program's list without the groups it declares, in the same order; every test is in one of the two", DesktopListMatchesLibrary),
    ];

    private static void DesktopListMatchesLibrary(Device device)
    {
        _ = device;
        var own = typeof(DesktopTests).GetFields(BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
            .Where(f => f.FieldType == typeof((string Name, Action<Device> Run)[]))
            .SelectMany(f => ((string Name, Action<Device> Run)[])f.GetValue(null)!).Select(t => t.Name).ToHashSet(StringComparer.Ordinal);
        var library = TestSuite.All.Select(t => t.Name).ToList();
        var all = All.Select(t => t.Name).ToList();
        var libraryNames = library.ToHashSet(StringComparer.Ordinal);
        Check(own.Count > 0 && !own.Overlaps(libraryNames), $"names in both: {string.Join("; ", own.Intersect(libraryNames).Take(5))}");
        Check(all.Count == own.Count + library.Count && all.All(n => own.Contains(n) || libraryNames.Contains(n)),
            $"{all.Count} tests here, {own.Count} of this program's groups and {library.Count} of the library's; not in either: {string.Join("; ", all.Where(n => !own.Contains(n) && !libraryNames.Contains(n)).Take(5))}");
        var order = all.Where(libraryNames.Contains).ToList();
        int at = Enumerable.Range(0, library.Count).FirstOrDefault(i => order[i] != library[i], -1);
        Check(at < 0, $"the library's order differs at {at}: here \"{(at >= 0 ? order[at] : "")}\", the library's \"{(at >= 0 ? library[at] : "")}\"");
    }
}
