// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.RegularExpressions;
using System.Xml;

namespace Idrak.Reinforcement.Plugins.CSharp;

/// <summary>
/// Reads what a C# build and test run wrote: the tests' TRX results files (<see cref="Tests"/>, through the base
/// library's <see cref="XmlReader"/>) and a build's error or warning log (<see cref="Diagnostics"/>: MSBuild's
/// <c>file(line,col): error CSnnnn: text</c> lines).
/// </summary>
public static partial class CSharpResults
{
    /// <summary>
    /// The tests every TRX file in <paramref name="folder"/> counts (their <c>Counters</c>' <c>total</c>, tests that did
    /// not run included) and those that passed, summed over the files; (0, 0) when there is none (the tests did not run).
    /// A file that is not readable XML counts nothing.
    /// </summary>
    public static (int Total, int Passed) Tests(string folder)
    {
        ArgumentNullException.ThrowIfNull(folder);
        if (!Directory.Exists(folder))
        {
            return (0, 0);
        }

        int total = 0, passed = 0;
        var files = Directory.GetFiles(folder, "*.trx", SearchOption.AllDirectories);
        Array.Sort(files, StringComparer.Ordinal);
        var settings = new XmlReaderSettings { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, IgnoreComments = true, IgnoreWhitespace = true };
        foreach (var file in files)
        {
            try
            {
                using var reader = XmlReader.Create(file, settings);
                while (reader.Read())
                {
                    if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "Counters")
                    {
                        total += Count(reader.GetAttribute("total"));
                        passed += Count(reader.GetAttribute("passed"));
                        break;
                    }
                }
            }
            catch (XmlException)
            {
            }
        }

        return (total, Math.Min(passed, total));
    }

    /// <summary>
    /// The distinct diagnostics of <paramref name="category"/> ("error" or "warning") in the build log
    /// <paramref name="log"/> (MSBuild's file logger with <c>errorsonly</c> or <c>warningsonly</c>); 0 when there is no log.
    /// </summary>
    public static int Diagnostics(string log, string category)
    {
        ArgumentNullException.ThrowIfNull(log);
        ArgumentException.ThrowIfNullOrWhiteSpace(category);
        if (!File.Exists(log))
        {
            return 0;
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var raw in File.ReadLines(log))
        {
            // A line of a multi-process build starts with the node and project ("  2:3>"): the same diagnostic either way.
            string line = NodePrefix().Replace(raw, "").Trim();
            var match = Diagnostic().Match(line);
            if (match.Success && match.Groups["category"].Value.Equals(category, StringComparison.OrdinalIgnoreCase))
            {
                seen.Add(line);
            }
        }

        return seen.Count;
    }

    private static int Count(string? text) => int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0 ? n : 0;

    [GeneratedRegex(@"^\s*\d+(:\d+)?>")]
    private static partial Regex NodePrefix();

    // MSBuild's canonical form: "origin: [subcategory] category code: text", the origin and the code optional.
    [GeneratedRegex(@"(^|:\s*)(?<category>error|warning)(\s+[A-Za-z]+[A-Za-z0-9]*)?\s*:", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex Diagnostic();
}
