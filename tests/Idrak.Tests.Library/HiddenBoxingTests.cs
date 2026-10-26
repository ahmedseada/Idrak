// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.RegularExpressions;

// docs/optimization.md rules 88 and 89: calls that box in a method's code before it is optimized, which a machine whose
// JIT is busy elsewhere runs for thousands of calls (a warm BPE encode allocated 96 bytes a call there).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] HiddenBoxingGroup =
    [
        ("hidden boxing: nothing in src/ calls the generic range searches (ContainsAnyInRange, IndexOfAnyInRange, IndexOfAnyExceptInRange, …) or Enum.HasFlag, which box until a method is optimized (optimization rules 88 and 89: SearchValues, a bitwise test)", HiddenBoxingNoCalls),
    ];

    internal static void HiddenBoxingNoCalls(Device device)
    {
        _ = device;
        var boxing = new Regex(@"\.(?:Contains|IndexOf|LastIndexOf)Any(?:Except)?InRange\(|\.HasFlag\(", RegexOptions.CultureInvariant);
        string src = RepositoryPath("src");
        var found = new List<string>();
        int files = 0;
        foreach (string file in Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            {
                continue;
            }

            files++;
            int number = 0;
            foreach (string line in File.ReadLines(file))
            {
                number++;
                string code = line.TrimStart();
                if (!code.StartsWith("//", StringComparison.Ordinal) && boxing.IsMatch(code))
                {
                    found.Add($"{Path.GetRelativePath(src, file)}:{number}: {code}");
                }
            }
        }

        Check(files > 100, $"the scan read src/ ({files} files)");
        Check(found.Count == 0, "calls that box until a method is optimized, in src/ (a static readonly SearchValues<char> for a range, "
            + $"(flags & Flag) != 0 for a flag; docs/optimization.md rules 88 and 89):\n{string.Join("\n", found)}");
    }
}
