// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.RegularExpressions;
using Idrak.Samples.ReRanker;

namespace Idrak.Samples.Rag;

/// <summary>The short answer a passage gives to a question of its aspect ("about 40 thousand people"), read from its text.</summary>
public static partial class Answers
{
    public const string Unknown = "i do not know";

    public static string Of(Passage passage)
    {
        string t = passage.Text;
        Match m;
        return passage.Aspect switch
        {
            "population" => $"about {Number().Match(t).Value} thousand people",
            "founded" => $"it was founded in {Number().Match(t).Value}",
            "food" => $"it is known for its {Regex.Match(t, "(?:famous for its|try the local) (.+) \\.$").Groups[1].Value}",
            "river" => $"the {Regex.Match(t, "the (\\w+) river flows|banks of the (\\w+) \\.").Groups.Values.Skip(1).First(g => g.Success).Value} river",
            "climate" => (m = Regex.Match(t, "summers in \\w+ are (\\w+) and winters are (\\w+)")).Success
                ? $"{m.Groups[1].Value} summers and {m.Groups[2].Value} winters"
                : Regex.Replace(t, "^.* has a (\\w+) climate with (\\w+) winters \\.$", "a $1 climate with $2 winters"),
            "sport" => Regex.Match(t, "(?:sport in \\w+ is|the local) (\\w+)").Groups[1].Value,
            "mayor" => $"mayor {Regex.Match(t, "mayor (\\w+ \\w+) \\.|^(\\w+ \\w+) was elected").Groups.Values.Skip(1).First(g => g.Success).Value}",
            "transport" => t.Contains("tram") ? "a tram network" : $"a {Regex.Match(t, "has a (\\w+) service").Groups[1].Value} service",
            _ => throw new ArgumentException($"Passage {passage.Id} answers no question."),
        };
    }

    /// <summary>The reply the chat model is trained to give: the answer and the citation, or that it does not know.</summary>
    public static string Reply(string? answer, int citation) => answer is null ? Unknown + " ." : $"{answer} [{citation}] .";

    /// <summary>The reply without its citations, spaces normalized, for comparing with an answer.</summary>
    public static string Strip(string reply) => Regex.Replace(Citation().Replace(reply, ""), "\\s+", " ").Trim().TrimEnd('.').Trim();

    [GeneratedRegex("\\d+")]
    private static partial Regex Number();

    [GeneratedRegex("\\[\\s*\\d+\\s*\\]")]
    private static partial Regex Citation();
}
