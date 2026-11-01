// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.Loader;

namespace Idrak.Abstraction;

/// <summary>
/// The assemblies the program has as files, by simple name: those the host lists as the program's own (its trusted
/// platform assemblies, from the program's dependency list) and those beside Idrak.Abstraction (a plug-in's folder when
/// a host loads it from there). Read once a process (rule 75). Asked before loading a library by name, so a library the
/// program does not ship is skipped without a load that throws (a first-chance <see cref="FileNotFoundException"/> in
/// every debugger and trail); the same on every platform (rule 70): a host that gives no files (Idrak.Abstraction not a
/// file of its own: a single-file bundle, a phone's assembly store) tells nothing, and every name may then be loaded.
/// </summary>
internal static class LibraryFiles
{
    // The host's list of the program's assemblies, separated by Path.PathSeparator (none on hosts without one).
    private const string TrustedAssemblies = "TRUSTED_PLATFORM_ASSEMBLIES";

    private static readonly Lazy<Listing?> Files = new(Scan);

    /// <summary>The program's assembly files: simple name to path (the first listed wins), and the paths outside the runtime's shared frameworks.</summary>
    /// <param name="ByName">Every file, by simple name (ignoring case, as assembly names are).</param>
    /// <param name="Program">The files outside the shared frameworks' folder (the program's and its packages'), in listing order.</param>
    internal sealed record Listing(Dictionary<string, string> ByName, string[] Program);

    /// <summary>The program's files; null when the host gives none (see the class).</summary>
    public static Listing? Present => Files.Value;

    /// <summary>
    /// Whether an assembly of this simple name may be loaded from Idrak.Abstraction's load context: it is one of the
    /// program's files, or loaded already (in that context or the default one, where a context falls back), or the host
    /// gives no files to tell.
    /// </summary>
    public static bool MayLoad(string name) => Files.Value is not { } files || files.ByName.ContainsKey(name) || Loaded(name);

    private static bool Loaded(string name)
    {
        var own = AssemblyLoadContext.GetLoadContext(typeof(LibraryFiles).Assembly);
        return Contains(AssemblyLoadContext.Default.Assemblies, name) || own is not null && own != AssemblyLoadContext.Default && Contains(own.Assemblies, name);

        static bool Contains(IEnumerable<Assembly> assemblies, string name)
        {
            foreach (var assembly in assemblies)
            {
                if (string.Equals(assembly.GetName().Name, name, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }
    }

    [UnconditionalSuppressMessage("SingleFile", "IL3000", Justification = "An empty location (a single-file bundle) is read as a host that gives no files.")]
    private static Listing? Scan()
    {
        string location = typeof(LibraryFiles).Assembly.Location;
        if (location.Length == 0)
        {
            return null;                                                            // no files to tell from
        }

        var byName = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var program = new List<string>();
        string? frameworks = SharedFrameworks();
        if (AppContext.GetData(TrustedAssemblies) is string listed)
        {
            ReadOnlySpan<char> all = listed;
            foreach (Range range in all.Split(Path.PathSeparator))
            {
                Add(all[range].Trim().ToString());
            }
        }

        if (Path.GetDirectoryName(location) is { Length: > 0 } folder)
        {
            try
            {
                foreach (string path in Directory.EnumerateFiles(folder, "*.dll"))
                {
                    Add(path);
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // A folder that cannot be listed adds nothing: its assemblies are not loaded from it either.
            }
        }

        return new Listing(byName, [.. program]);

        void Add(string path)
        {
            if (path.Length == 0 || !byName.TryAdd(Path.GetFileNameWithoutExtension(path), path))
            {
                return;
            }

            if (frameworks is null || !path.StartsWith(frameworks, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            {
                program.Add(path);
            }
        }
    }

    // The folder holding the runtime's shared frameworks (…/shared, whose children are …/<framework>/<version>/), when
    // the runtime is not the program's own (a self-contained program has its runtime beside it: nothing is left out).
    [UnconditionalSuppressMessage("SingleFile", "IL3000", Justification = "An empty location (a single-file bundle) is read as a host that gives no files.")]
    private static string? SharedFrameworks()
    {
        string core = typeof(object).Assembly.Location;
        if (core.Length == 0 || Path.GetDirectoryName(core) is not { Length: > 0 } runtime
            || string.Equals(Path.TrimEndingDirectorySeparator(runtime), Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory), StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return Path.GetDirectoryName(Path.GetDirectoryName(runtime)) is { Length: > 0 } shared ? Path.TrimEndingDirectorySeparator(shared) + Path.DirectorySeparatorChar : null;
    }
}
