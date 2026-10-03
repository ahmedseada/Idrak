// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Shared;

/// <summary>
/// <c>@file</c> arguments (plans/idrak-cli.md, "Helpers every command shares"): an argument <c>@args.txt</c> is replaced
/// by the file's lines, one argument per line (long option lists in scripts). Blank lines and lines starting with '#'
/// are skipped, spaces around a line are trimmed, a file may name further <c>@file</c>s, and <c>@@text</c> stands for
/// the literal argument <c>@text</c>. Arguments after <c>--</c> are left as they are.
/// </summary>
internal static class ResponseFiles
{
    // Response files naming response files: deep enough for any script, shallow enough to stop a file naming itself.
    private const int MaxDepth = 8;

    /// <summary><paramref name="args"/> with every <c>@file</c> replaced by its arguments.</summary>
    /// <exception cref="UsageException">A file is missing or the files name each other too deeply.</exception>
    public static IReadOnlyList<string> Expand(IReadOnlyList<string> args) => args.Any(a => a.StartsWith('@')) ? Expand(args, 0) : args;

    private static List<string> Expand(IReadOnlyList<string> args, int depth)
    {
        var result = new List<string>(args.Count);
        bool optionsEnded = false;
        foreach (string arg in args)
        {
            if (optionsEnded || arg.Length < 2 || arg[0] != '@')
            {
                optionsEnded |= arg == "--";
                result.Add(arg);
            }
            else if (arg[1] == '@')
            {
                result.Add(arg[1..]);
            }
            else
            {
                string path = arg[1..];
                if (!File.Exists(path))
                {
                    throw new UsageException($"Argument file not found: {Path.GetFullPath(path)} (write @@{path} for a literal argument starting with @).");
                }

                if (depth >= MaxDepth)
                {
                    throw new UsageException($"Argument files nest more than {MaxDepth} deep at {path}; does a file name itself?");
                }

                var lines = File.ReadAllLines(path).Select(l => l.Trim()).Where(l => l.Length > 0 && l[0] != '#').ToList();
                result.AddRange(Expand(lines, depth + 1));
            }
        }

        return result;
    }
}
