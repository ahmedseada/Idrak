// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A data file a tuning data format is checked on (<see cref="TuningDataFormatSuite"/>): one the format reads, with the
/// conversations it must give when known, or one it must refuse.
/// </summary>
/// <param name="Path">The data file.</param>
/// <param name="Expected">The messages of each conversation the file holds, in order; null to check the format's own consistency only.</param>
/// <param name="Error">
/// When set, the file is not valid: reading it must fail with an <see cref="InvalidDataException"/> (or a
/// <see cref="FileNotFoundException"/> for a missing image) whose message names the file and contains this text.
/// </param>
public sealed record TuningDataSample(string Path, IReadOnlyList<IReadOnlyList<ChatMessage>>? Expected = null, string? Error = null);

/// <summary>
/// The checks of a fine-tuning data format (Idrak.Nlp's <c>ITuningDataFormat</c>) on data files of its own. The contract
/// lives with the package that uses it and this kit depends on Idrak.Abstraction alone, so a test passes the format's
/// reading as a delegate: a file's path in, each conversation's messages out (<c>format.Read(path, options).Select(t =&gt;
/// t.Messages)</c>). For each sample:
/// <list type="bullet">
/// <item>a valid file reads without failing; every conversation has messages, with roles system, user, assistant or tool,
/// and its image parts carry their bytes;</item>
/// <item>reading again gives the same conversations (images equal by their bytes), and so do two readings at once;</item>
/// <item>taking the first few conversations gives the first few of a whole reading (reading is lazy, and stopping early
/// is clean);</item>
/// <item>with <see cref="TuningDataSample.Expected"/>, exactly those conversations;</item>
/// <item>an invalid file (<see cref="TuningDataSample.Error"/>) fails with an error naming the file and saying what is
/// wrong.</item>
/// </list>
/// </summary>
/// <param name="samples">The data files, in the format checked.</param>
public sealed class TuningDataFormatSuite(IReadOnlyList<TuningDataSample> samples) : ContractSuite<Func<string, IEnumerable<IReadOnlyList<ChatMessage>>>>
{
    private static readonly HashSet<string> Roles = new(StringComparer.Ordinal) { "system", "user", "assistant", "tool" };

    private readonly IReadOnlyList<TuningDataSample> _samples = samples is { Count: > 0 } ? samples
        : throw new ArgumentException("A tuning data format is checked on data files of its own: give at least one sample.", nameof(samples));

    /// <inheritdoc />
    public override string Name => "tuning-data-format";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases() => _samples.Select((s, i) => Case(i, take: s.Error is null ? 1 : 0));

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int index = random.Next(_samples.Count);
        return Case(index, take: random.Next(0, large ? 100 : 5));
    }

    /// <inheritdoc />
    public override void Run(Func<string, IEnumerable<IReadOnlyList<ChatMessage>>> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var sample = _samples[(int)@case.Data["sample"]!];
        int take = (int)@case.Data["take"]!;
        string file = System.IO.Path.GetFileName(sample.Path);
        if (sample.Error is { } error)
        {
            try
            {
                _ = implementation(sample.Path).ToList();
                checks.Fail("refuses an invalid file", $"read {file} without an error (expected one about \"{error}\")");
            }
            catch (Exception e) when (e is InvalidDataException or FileNotFoundException)
            {
                checks.Check("an invalid file's error names the file", e.Message.Contains(file, StringComparison.Ordinal), $"\"{e.Message}\" does not name {file}");
                checks.Check("an invalid file's error says what is wrong", e.Message.Contains(error, StringComparison.Ordinal), $"\"{e.Message}\" does not say \"{error}\"");
            }

            return;
        }

        var all = implementation(sample.Path).ToList();
        checks.Check("every conversation has messages", all.All(m => m.Count > 0), "a conversation has no message");
        var roles = all.SelectMany(m => m).Select(m => m.Role).Where(r => !Roles.Contains(r)).Distinct().ToList();
        checks.Check("roles are system, user, assistant or tool", roles.Count == 0, $"roles {string.Join(", ", roles)}");
        var images = all.SelectMany(m => m).SelectMany(m => m.Parts).OfType<ChatImage>().ToList();
        checks.Check("image parts carry their bytes", images.All(i => i.Data.Length > 0 && i.Hash.Length == 64), "an image part without bytes");

        checks.Compare("the same conversations on a second reading", Different(all, implementation(sample.Path).ToList()));
        var parallel = new List<IReadOnlyList<ChatMessage>>[2];
        Parallel.For(0, 2, i => parallel[i] = implementation(sample.Path).ToList());
        checks.Compare("the same conversations from two readings at once", Different(all, parallel[0]) ?? Different(all, parallel[1]));
        var first = implementation(sample.Path).Take(take).ToList();
        checks.Compare($"the first {take} conversations of a reading stopped early", Different([.. all.Take(take)], first));
        if (sample.Expected is { } expected)
        {
            checks.Compare("the expected conversations", Different(expected, all));
        }
        else
        {
            checks.Skip("the expected conversations", "the sample gives none");
        }
    }

    // Where two lists of conversations differ, or null.
    private static string? Different(IReadOnlyList<IReadOnlyList<ChatMessage>> expected, IReadOnlyList<IReadOnlyList<ChatMessage>> actual)
    {
        if (expected.Count != actual.Count)
        {
            return $"{actual.Count} conversations, expected {expected.Count}";
        }

        for (int i = 0; i < expected.Count; i++)
        {
            if (expected[i].Count != actual[i].Count)
            {
                return $"conversation {i + 1}: {actual[i].Count} messages, expected {expected[i].Count}";
            }

            for (int j = 0; j < expected[i].Count; j++)
            {
                var (e, a) = (expected[i][j], actual[i][j]);
                if (e.Role != a.Role || e.Thinking != a.Thinking || e.ToolName != a.ToolName || !e.Parts.SequenceEqual(a.Parts) || (e.ToolCalls?.Count ?? 0) != (a.ToolCalls?.Count ?? 0))
                {
                    return $"conversation {i + 1}, message {j + 1}: {Show(a)}, expected {Show(e)}";
                }
            }
        }

        return null;
    }

    private static string Show(ChatMessage message) =>
        $"{message.Role} [{string.Join(", ", message.Parts.Select(p => p is ChatText t ? $"\"{(t.Text.Length > 40 ? t.Text[..40] + "…" : t.Text)}\"" : p.ToString()))}]";

    private ContractCase Case(int sample, int take) =>
        new($"{System.IO.Path.GetFileName(_samples[sample].Path)}{(_samples[sample].Error is null ? $", the first {take}" : ", refused")}",
            new JsonObject { ["sample"] = sample, ["take"] = take });
}
