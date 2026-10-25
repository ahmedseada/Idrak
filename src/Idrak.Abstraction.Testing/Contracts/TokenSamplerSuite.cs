// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a token sampler kind (<see cref="ITokenSamplerKind"/>, whose samplers are called as generation calls them)
/// against a reference kind (the library's default sampler), each sampler made from the same request, on the same logits
/// and history:
/// <list type="bullet">
/// <item>the sampler has the asked rows and vocabulary, and its ids are [rows] tokens in [0, vocabulary);</item>
/// <item>greedy settings (top-k 1) choose the reference's tokens, penalties included;</item>
/// <item>without penalties, every sampled token is one the settings allow (among the k largest distinct scores, inside
/// the top-p nucleus, at least min-p as likely as the best);</item>
/// <item>a reset with the same history samples the same tokens again;</item>
/// <item>the statistics report, per step and row, the token sampled with a probability in [0, 1];</item>
/// <item>with <see cref="Exact"/>, the very tokens and probabilities of the reference (same seed);</item>
/// <item>logits that are not finite follow the library's rule (<c>Backend.SampleRowsKernel</c>): a NaN or -∞ logit is
/// never sampled (greedy takes the best finite token), a row with +∞ logits takes one of them, with probability one over
/// their count, and a row with nothing finite is reported as an error (by <c>Sample</c> or <c>Read</c>).</item>
/// </list>
/// </summary>
/// <param name="reference">The kind the samplers are compared with (the library's <see cref="TokenSamplers.DefaultName"/> when null).</param>
/// <param name="device">Where the logits and the samplers live (the CPU when null).</param>
public sealed class TokenSamplerSuite(ITokenSamplerKind? reference = null, Device? device = null) : ContractSuite<ITokenSamplerKind>
{
    private readonly ITokenSamplerKind _reference = reference ?? TokenSamplers.Default(TokenSamplers.DefaultName)!;

    /// <summary>Where the logits and the samplers live.</summary>
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>Whether every token must be the reference's (the same random stream), not only the greedy ones.</summary>
    public bool Exact { get; init; }

    /// <inheritdoc />
    public override string Name => "token-sampler";

    /// <inheritdoc />
    protected override string Describe(ITokenSamplerKind implementation) => implementation.Name;

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        var random = new Random(7);
        yield return Case("greedy, one row", random, rows: 1, vocabulary: 50, steps: 4, topK: 1);
        yield return Case("greedy with penalties", random, rows: 2, vocabulary: 30, steps: 6, topK: 1, repeat: 1.3f, presence: 0.5f, frequency: 0.2f);
        yield return Case("two tokens", random, rows: 3, vocabulary: 2, steps: 5, topK: 0);
        yield return Case("near ties", random, rows: 2, vocabulary: 16, steps: 3, topK: 1, scale: 1e-4f);
        yield return Case("nucleus 0.5", random, rows: 2, vocabulary: 100, steps: 4, topK: 0, topP: 0.5f);
        yield return Case("min-p 0.1", random, rows: 2, vocabulary: 100, steps: 4, topK: 0, minP: 0.1f);
        yield return Case("top-k 5 at temperature 1.5", random, rows: 4, vocabulary: 64, steps: 4, topK: 5, temperature: 1.5f);
        yield return Case("large logits", random, rows: 2, vocabulary: 40, steps: 3, topK: 40, scale: 50f);
        yield return NonFinite("one NaN, greedy", [[0.5f, float.NaN, 1f, 4f, 2f]], topK: 1, NonFiniteExpectation.BestFinite);
        yield return NonFinite("NaN first, greedy", [[float.NaN, 1f, 3f, 2f]], topK: 1, NonFiniteExpectation.BestFinite);
        yield return NonFinite("NaN and -∞, nucleus", [[float.NegativeInfinity, 2f, float.NaN, 1f, 0.5f], [1f, float.NaN, float.NaN, 3f, float.NegativeInfinity]],
            topK: 0, NonFiniteExpectation.Finite, topP: 0.9f);
        yield return NonFinite("NaN with min-p and top-k", [[float.NaN, 2f, 1.5f, float.NaN, 1f, 0f]], topK: 3, NonFiniteExpectation.Finite, minP: 0.05f);
        yield return NonFinite("+∞ wins", [[1f, float.PositiveInfinity, 3f, float.NaN, float.PositiveInfinity]], topK: 0, NonFiniteExpectation.Infinite);
        yield return NonFinite("+∞, greedy", [[float.PositiveInfinity, 9f, float.NaN]], topK: 1, NonFiniteExpectation.Infinite);
        yield return NonFinite("nothing finite", [[float.NaN, float.NegativeInfinity, float.NaN]], topK: 0, NonFiniteExpectation.Error);

        // Wide rows, where devices sample top-k in two stages (candidates per slice first).
        float[] wide = [.. Enumerable.Range(0, 9000).Select(i => MathF.Sin(i) * 3f)];
        wide[123] = wide[5000] = float.NaN;
        wide[8000] = 7f;
        yield return NonFinite("NaN in a wide row, top-k 1", [wide], topK: 1, NonFiniteExpectation.BestFinite);
        float[] wideInfinite = [.. wide];
        wideInfinite[4500] = wideInfinite[8999] = float.PositiveInfinity;
        yield return NonFinite("+∞ in a wide row, top-k 5", [wideInfinite], topK: 5, NonFiniteExpectation.Infinite);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (random.Next(4) == 0)
        {
            return RandomNonFinite(random, large);
        }

        int[] vocabularies = large ? [2, 7, 50, 1000, 32_000, 151_936] : [2, 3, 7, 50, 257, 1000];
        bool penalties = random.Next(3) == 0;
        return Case("random", random, rows: random.Next(1, large ? 9 : 5), vocabulary: vocabularies[random.Next(vocabularies.Length)],
            steps: random.Next(1, large ? 33 : 7), topK: new[] { 0, 1, 5, 40 }[random.Next(4)], topP: new[] { 1f, 0.95f, 0.5f }[random.Next(3)],
            minP: random.Next(3) == 0 ? 0.05f : 0f, temperature: new[] { 0.5f, 0.8f, 1f, 1.5f }[random.Next(4)],
            repeat: penalties ? 1.1f + random.NextSingle() : 1f, presence: penalties ? random.NextSingle() : 0f,
            frequency: penalties ? random.NextSingle() * 0.5f : 0f, scale: random.Next(2) == 0 ? 3f : 10f);
    }

    /// <inheritdoc />
    public override void Run(ITokenSamplerKind implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int rows = (int)d["rows"]!, vocabulary = (int)d["vocabulary"]!, steps = (int)d["steps"]!;
        var settings = new GenerationOptions
        {
            Temperature = (float)d["temperature"]!, TopK = (int)d["topK"]!, TopP = (float)d["topP"]!, MinP = (float)d["minP"]!,
            RepeatPenalty = (float)d["repeatPenalty"]!, RepeatLastN = (int)d["repeatLastN"]!, PresencePenalty = (float)d["presencePenalty"]!,
            FrequencyPenalty = (float)d["frequencyPenalty"]!, Seed = (int)d["seed"]!,
        };
        var request = new SamplerRequest(Device, rows, vocabulary, steps, 64, settings);
        int[] history = [.. d["history"]!.AsArray().Select(n => (int)n!)];
        float[] logits = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String((string)d["logits"]!)).ToArray();
        if (d["expect"] is { } expect)
        {
            RunNonFinite(implementation.Create(request), Enum.Parse<NonFiniteExpectation>((string)expect!), logits, rows, vocabulary, checks);
            return;
        }

        using var mine = implementation.Create(request);
        using var library = _reference.Create(request);
        checks.Check("the asked rows and vocabulary", mine.Rows == rows && mine.Vocabulary == vocabulary,
            $"{mine.Rows} rows and {mine.Vocabulary} tokens, {rows} and {vocabulary} asked for");
        var ids = Sample(mine, history, logits, rows, vocabulary, steps, checks);
        var expected = Sample(library, history, logits, rows, vocabulary, steps, null);

        var flat = ids.SelectMany(s => s).ToList();
        int outside = flat.FindIndex(id => id < 0 || id >= vocabulary);
        checks.Check("ids in range", outside < 0, outside < 0 ? "" : $"token {flat[outside]} at step {outside / rows}, row {outside % rows}");

        bool penalties = settings.RepeatPenalty != 1f || settings.PresencePenalty != 0f || settings.FrequencyPenalty != 0f;
        if (settings.TopK == 1 || Exact)
        {
            checks.Compare(settings.TopK == 1 ? "greedy tokens are the reference's" : "the reference's tokens", Comparisons.Difference(expected.SelectMany(s => s).ToList(), flat));
        }
        else if (penalties)
        {
            checks.Skip("tokens the settings allow", "penalties change the logits");
        }
        else
        {
            checks.Compare("tokens the settings allow", Allowed(logits, ids, rows, vocabulary, settings));
        }

        mine.Reset();
        checks.Compare("a reset samples the same tokens", Comparisons.Difference(flat, Sample(mine, history, logits, rows, vocabulary, steps, null).SelectMany(s => s).ToList()));

        var statistics = mine.Read(0, steps);
        string? problem = statistics.Length != steps || statistics.Any(s => s.Length != rows) ? $"{statistics.Length} steps of statistics, {steps} expected"
            : Enumerable.Range(0, steps * rows).Select(i => (Step: i / rows, Row: i % rows)).Select(p => statistics[p.Step][p.Row] is var token
                && token.Id != ids[p.Step][p.Row] ? $"step {p.Step}, row {p.Row}: token {token.Id}, {ids[p.Step][p.Row]} was sampled"
                : !(token.Probability >= 0f && token.Probability <= 1f + 1e-5f) ? $"step {p.Step}, row {p.Row}: probability {token.Probability}"
                : !(token.Entropy >= -1e-4f && float.IsFinite(token.Entropy)) ? $"step {p.Step}, row {p.Row}: entropy {token.Entropy}" : null)
                .FirstOrDefault(m => m is not null);
        checks.Compare("statistics of the tokens sampled", problem);
        if (Exact)
        {
            var expectedStatistics = library.Read(0, steps);
            checks.Compare("the reference's probabilities", Comparisons.Difference(
                [.. expectedStatistics.SelectMany(s => s).Select(t => t.Probability)], [.. statistics.SelectMany(s => s).Select(t => t.Probability)], 1e-4f));
        }
    }

    // One step over logits that are not all finite, checked against the rule (no reference needed: the rule says the answer).
    private void RunNonFinite(ITokenSampler sampler, NonFiniteExpectation expect, float[] logits, int rows, int vocabulary, CaseChecks checks)
    {
        using var mine = sampler;
        int[]? ids = null;
        SampledToken[][]? statistics = null;
        Exception? error = null;
        try
        {
            ids = Sample(mine, [], logits, rows, vocabulary, 1, null)[0];
            statistics = mine.Read(0, 1);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            error = e;
        }

        if (expect == NonFiniteExpectation.Error)
        {
            checks.Check("a row with nothing finite is reported", error is not null, ids is null ? "" : $"sampled token {ids[0]} and no error");
            return;
        }

        checks.Check("non-finite logits do not break sampling", error is null, error is null ? "" : $"{error.GetType().Name}: {error.Message}");
        if (ids is null || statistics is null)
        {
            return;
        }

        for (int r = 0; r < rows; r++)
        {
            var row = logits.AsSpan(r * vocabulary, vocabulary);
            int id = ids[r];
            string where = $"row {r}: token {id}";
            if (id < 0 || id >= vocabulary)
            {
                checks.Check("ids in range", false, where);
                continue;
            }

            float value = row[id];
            switch (expect)
            {
                case NonFiniteExpectation.BestFinite:
                    int best = -1;
                    for (int i = 0; i < vocabulary; i++)
                    {
                        best = float.IsFinite(row[i]) && (best < 0 || row[i] > row[best]) ? i : best;
                    }

                    checks.Check("greedy takes the best finite token", id == best, $"{where} ({value}), the best finite is {best} ({row[best]})");
                    break;
                case NonFiniteExpectation.Finite:
                    checks.Check("NaN and -∞ logits are never sampled", float.IsFinite(value), $"{where} has logit {value}");
                    break;
                default:
                    int count = 0;
                    foreach (float v in row)
                    {
                        count += float.IsPositiveInfinity(v) ? 1 : 0;
                    }

                    checks.Check("a +∞ logit wins", float.IsPositiveInfinity(value), $"{where} has logit {value}");
                    checks.Check("the +∞ tokens share the probability", Math.Abs(statistics[0][r].Probability - 1f / count) < 1e-5f,
                        $"{where}: probability {statistics[0][r].Probability}, 1/{count} expected");
                    break;
            }
        }
    }

    // The ids sampled at each step (after the history is set), [steps][rows].
    private int[][] Sample(ITokenSampler sampler, int[] history, float[] logits, int rows, int vocabulary, int steps, CaseChecks? checks)
    {
        sampler.SetHistory(history);
        var ids = new int[steps][];
        for (int s = 0; s < steps; s++)
        {
            using var step = Tensor.From(logits.AsSpan(s * rows * vocabulary, rows * vocabulary), [rows, vocabulary], Device);
            sampler.Sample(step);
            var values = sampler.Ids.ToArray();
            checks?.Check("ids are [rows]", values.Length == rows, $"{values.Length} ids for {rows} rows");
            ids[s] = [.. values.Take(rows).Select(v => (int)v)];
        }

        return ids;
    }

    // Null when every sampled token is among the top k, in the top-p nucleus and at least min-p as likely as the best
    // token (each with a little slack for rounding); else the first that is not.
    private static string? Allowed(float[] logits, int[][] ids, int rows, int vocabulary, GenerationOptions settings)
    {
        for (int s = 0; s < ids.Length; s++)
        {
            for (int r = 0; r < rows; r++)
            {
                var row = logits.AsSpan((s * rows + r) * vocabulary, vocabulary);
                int id = ids[s][r];
                if (id < 0 || id >= vocabulary)
                {
                    continue;
                }

                double max = double.NegativeInfinity, sum = 0;
                foreach (float v in row)
                {
                    max = Math.Max(max, v);
                }

                var p = new double[vocabulary];
                for (int i = 0; i < vocabulary; i++)
                {
                    sum += p[i] = Math.Exp((row[i] - max) / settings.Temperature);
                }

                double mine = p[id] / sum, above = 0, best = 1 / sum;
                // Top-k keeps the k largest distinct scores (ties at the cut-off kept): count distinct likelier scores.
                var higherScores = new HashSet<float>();
                for (int i = 0; i < vocabulary; i++)
                {
                    if (p[i] / sum > mine * (1 + 1e-4))
                    {
                        higherScores.Add(row[i]);
                        above += p[i] / sum;
                    }
                }

                int higher = higherScores.Count;

                string where = $"step {s}, row {r}: token {id} (probability {mine:G4})";
                if (settings.TopK > 0 && higher >= settings.TopK)
                {
                    return $"{where} is not among the top {settings.TopK} ({higher} distinct scores are higher)";
                }

                if (settings.TopP < 1f && above >= settings.TopP + 1e-4)
                {
                    return $"{where} is outside the top-p {settings.TopP} nucleus (the likelier tokens hold {above:G4})";
                }

                if (settings.MinP > 0f && mine < settings.MinP * best * (1 - 1e-4))
                {
                    return $"{where} is below min-p {settings.MinP} of the best token's {best:G4}";
                }
            }
        }

        return null;
    }

    // Logits with NaN and -∞ sprinkled over each row and a finite best token in it; greedy.
    private static ContractCase RandomNonFinite(Random random, bool large)
    {
        int rows = random.Next(1, 4), vocabulary = random.Next(2, large ? 5000 : 60);
        var logits = new float[rows][];
        for (int r = 0; r < rows; r++)
        {
            logits[r] = [.. Enumerable.Range(0, vocabulary).Select(_ => random.NextSingle() * 10f - 5f)];
            for (int i = 0, bad = random.Next(1, Math.Max(2, vocabulary / 4)); i < bad; i++)
            {
                logits[r][random.Next(vocabulary)] = random.Next(3) == 0 ? float.NegativeInfinity : float.NaN;
            }

            logits[r][random.Next(vocabulary)] = 6f;
        }

        return NonFinite($"random NaN and -∞ ({rows} rows, {vocabulary} tokens)", logits, topK: 1, NonFiniteExpectation.BestFinite);
    }

    private static ContractCase NonFinite(string name, float[][] logits, int topK, NonFiniteExpectation expect, float topP = 1f, float minP = 0f)
    {
        float[] flat = [.. logits.SelectMany(row => row)];
        return new ContractCase($"non-finite logits, {name}", new JsonObject
        {
            ["rows"] = logits.Length, ["vocabulary"] = logits[0].Length, ["steps"] = 1, ["temperature"] = 1f, ["topK"] = topK, ["topP"] = topP,
            ["minP"] = minP, ["repeatPenalty"] = 1f, ["repeatLastN"] = 64, ["presencePenalty"] = 0f, ["frequencyPenalty"] = 0f, ["seed"] = 1,
            ["history"] = new JsonArray(), ["expect"] = expect.ToString(),
            ["logits"] = Convert.ToBase64String(MemoryMarshal.AsBytes(flat.AsSpan())),
        });
    }

    private enum NonFiniteExpectation
    {
        BestFinite,
        Finite,
        Infinite,
        Error,
    }

    private static ContractCase Case(string name, Random random, int rows, int vocabulary, int steps, int topK, float topP = 1f, float minP = 0f,
        float temperature = 1f, float repeat = 1f, float presence = 0f, float frequency = 0f, float scale = 3f)
    {
        var logits = new float[steps * rows * vocabulary];
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = (random.NextSingle() * 2f - 1f) * scale;
        }

        int seed = random.Next();
        return new ContractCase($"{name}: {rows} rows, {vocabulary} tokens, {steps} steps, top-k {topK}, top-p {topP}, min-p {minP}, temperature {temperature}"
                                + (repeat != 1f || presence != 0f || frequency != 0f ? $", penalties {repeat:G3}/{presence:G3}/{frequency:G3}" : "") + $", seed {seed}",
            new JsonObject
            {
                ["rows"] = rows, ["vocabulary"] = vocabulary, ["steps"] = steps, ["temperature"] = temperature, ["topK"] = topK, ["topP"] = topP,
                ["minP"] = minP, ["repeatPenalty"] = repeat, ["repeatLastN"] = 64, ["presencePenalty"] = presence, ["frequencyPenalty"] = frequency,
                ["seed"] = seed,
                ["history"] = new JsonArray([.. Enumerable.Range(0, random.Next(0, 20)).Select(_ => (JsonNode)random.Next(vocabulary))]),
                ["logits"] = Convert.ToBase64String(MemoryMarshal.AsBytes(logits.AsSpan())),
            });
    }
}
