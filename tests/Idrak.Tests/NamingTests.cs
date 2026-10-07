// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;

// The library's name: nothing in the repository still uses the one it had before (sources, projects, docs, tools); and
// no public name is a provider's.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] NamingGroup =
    [
        ("naming: no file in the repository uses the library's former name or its tool names", FormerNamesGone),
        ("naming: no public type, member or parameter of an Idrak assembly is named after a provider", NoProviderNamesInPublicApi),
    ];

    // Names of model servers and model providers; public names describe what an API does, not whose it resembles.
    private static readonly string[] ProviderNames = ["ollama", "openai", "anthropic", "claude", "llamacpp", "lmstudio"];

    private static void NoProviderNamesInPublicApi(Device device)
    {
        _ = device;
        const System.Reflection.BindingFlags Declared = System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic
            | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.DeclaredOnly;
        var assemblies = Directory.EnumerateFiles(AppContext.BaseDirectory, "Idrak*.dll")
            .Select(Path.GetFileNameWithoutExtension)
            .Where(n => n is not ("Idrak.Tests" or "Idrak.PluginTests"))
            .Select(n => System.Reflection.Assembly.Load(n!))
            .ToList();
        Check(assemblies.Any(a => a.GetName().Name == "Idrak.AspNetCore") && assemblies.Any(a => a.GetName().Name == "Idrak.Nlp"),
            $"assemblies scanned: {string.Join(", ", assemblies.Select(a => a.GetName().Name))}");

        static bool Obsolete(System.Reflection.MemberInfo member) => member.IsDefined(typeof(ObsoleteAttribute), inherit: false);
        static bool Provider(string name) => ProviderNames.Any(p => name.Replace("_", "").Replace(".", "").Contains(p, StringComparison.OrdinalIgnoreCase));
        static bool Visible(System.Reflection.MethodBase? m) => m is { IsPublic: true } or { IsFamily: true } or { IsFamilyOrAssembly: true };

        var found = new List<string>();
        foreach (var assembly in assemblies)
        {
            foreach (var type in assembly.GetExportedTypes())
            {
                // The former names kept for one release are marked obsolete (the type, or the member on a current type).
                if (Obsolete(type) || type.DeclaringType is { } outer && Obsolete(outer))
                {
                    continue;
                }

                string typeName = type.FullName ?? type.Name;
                if (Provider(typeName))
                {
                    found.Add(typeName);
                }

                foreach (var member in type.GetMembers(Declared))
                {
                    bool visible = member switch
                    {
                        System.Reflection.MethodBase m => Visible(m),
                        System.Reflection.PropertyInfo p => Visible(p.GetMethod) || Visible(p.SetMethod),
                        System.Reflection.EventInfo e => Visible(e.AddMethod),
                        System.Reflection.FieldInfo f => f.IsPublic || f.IsFamily || f.IsFamilyOrAssembly,
                        _ => false,                                  // nested types are checked as exported types
                    };
                    if (!visible || Obsolete(member))
                    {
                        continue;
                    }

                    if (Provider(member.Name))
                    {
                        found.Add($"{typeName}.{member.Name}");
                    }

                    if (member is System.Reflection.MethodBase method)
                    {
                        found.AddRange(method.GetParameters().Where(p => p.Name is { } n && Provider(n)).Select(p => $"{typeName}.{member.Name}({p.Name})"));
                    }
                }
            }
        }

        Check(found.Count == 0, $"provider names in public API: {string.Join(", ", found.Distinct().Take(20))}");
    }

    private static void FormerNamesGone(Device device)
    {
        _ = device;
        string root = RepositoryRoot();
        string[] former = ["Neural" + "Sharp", "ns" + "tune", "ns" + "data"];   // split so this file does not match
        string[] skipped = ["bin", "obj", ".git", ".vs", ".idea", "__pycache__"];
        string[] binary = [".gguf", ".safetensors", ".parquet", ".onnx", ".png", ".jpg", ".bin", ".gz", ".zip", ".ikm", ".ikw"];
        var found = new List<string>();
        foreach (string path in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(root, path);
            if (relative.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Any(skipped.Contains)
                || binary.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
            {
                continue;
            }

            // Shared with writers: a file another process is still writing (test output redirected into the
            // repository, e.g. `> tests.txt`) is read as far as it has got.
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var reader = new StreamReader(stream);
            string text = relative + "\n" + reader.ReadToEnd();
            found.AddRange(former.Where(n => text.Contains(n, StringComparison.OrdinalIgnoreCase)).Select(n => $"{relative}: {n}"));
        }

        Check(found.Count == 0, $"former names left: {string.Join(", ", found.Take(10))}");
    }

    // The folder holding Idrak.slnx, found from the build output folder.
    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Idrak.slnx")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException($"Idrak.slnx not found above {AppContext.BaseDirectory}.");
    }
}
