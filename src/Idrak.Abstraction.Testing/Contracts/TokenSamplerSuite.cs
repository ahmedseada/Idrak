// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a token sampler, given as the factory a generation calls (<c>TokenSampler.Create</c> is the library's),
/// against the library's <see cref="TokenSampler"/> on the same requests, logits and history:
/// <list type="bullet">
/// <item>the sampler has the request's rows and vocabulary, and its ids are [rows] tokens in [0, vocabulary);</item>
/// <item>greedy settings (top-k 1) choose the library's tokens, penalties included;</item>
/// <item>without penalties, every sampled token is one the settings allow (among the top k, inside the top-p nucleus,
/// at least min-p as likely as the best);</item>
/// <item><see cref="ITokenSampler.Reset"/> with the same history samples the same tokens again;</item>
/// <item><see cref="ITokenSampler.Read"/> reports, per step and row, the token sampled with a probability in [0, 1];</item>
/// <item>with <see cref="Exact"/>, the very tokens and probabilities of the library's sampler (same seed).</item>
/// </list>
/// </summary>
/// <param name="device">Where the logits and the samplers live (the CPU when null).</param>
public sealed class TokenSamplerSuite(Device? device = null) : ContractSuite<Func<SamplerRequest, ITokenSampler>>
{
    /// <summary>Where the logits and the samplers live.</summary>
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>Whether every token must be the library sampler's (the same random stream), not only the greedy ones.</summary>
    public bool Exact { get; init; }

    /// <inheritdoc />
    public override string Name => "token-sampler";

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
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int[] vocabularies = large ? [2, 7, 50, 1000, 32_000, 151_936] : [2, 3, 7, 50, 257, 1000];
        bool penalties = random.Next(3) == 0;
        return Case("random", random, rows: random.Next(1, large ? 9 : 5), vocabulary: vocabularies[random.Next(vocabularies.Length)],
            steps: random.Next(1, large ? 33 : 7), topK: new[] { 0, 1, 5, 40 }[random.Next(4)], topP: new[] { 1f, 0.95f, 0.5f }[random.Next(3)],
            minP: random.Next(3) == 0 ? 0.05f : 0f, temperature: new[] { 0.5f, 0.8f, 1f, 1.5f }[random.Next(4)],
            repeat: penalties ? 1.1f + random.NextSingle() : 1f, presence: penalties ? random.NextSingle() : 0f,
            frequency: penalties ? random.NextSingle() * 0.5f : 0f, scale: random.Next(2) == 0 ? 3f : 10f);
    }

    /// <inheritdoc />
    public override void Run(Func<SamplerRequest, ITokenSampler> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int rows = (int)d["rows"]!, vocabulary = (int)d["vocabulary"]!, steps = (int)d["steps"]!;
        var options = new GenerationOptions
        {
            Temperature = (float)d["temperature"]!, TopK = (int)d["topK"]!, TopP = (float)d["topP"]!, MinP = (float)d["minP"]!,
            RepeatPenalty = (float)d["repeatPenalty"]!, RepeatLastN = (int)d["repeatLastN"]!, PresencePenalty = (float)d["presencePenalty"]!,
            FrequencyPenalty = (float)d["frequencyPenalty"]!, Seed = (int)d["seed"]!,
        };
        int[] history = [.. d["history"]!.AsArray().Select(n => (int)n!)];
        float[] logits = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String((string)d["logits"]!)).ToArray();
        var request = new SamplerRequest(Device, rows, vocabulary, steps, 64, options);

        using var mine = implementation(request);
        using var library = TokenSampler.Create(request);
        checks.Check("the request's rows and vocabulary", mine.Rows == rows && mine.Vocabulary == vocabulary,
            $"{mine.Rows} rows and {mine.Vocabulary} tokens, {rows} and {vocabulary} asked for");
        var ids = Sample(mine, history, logits, rows, vocabulary, steps, checks);
        var expected = Sample(library, history, logits, rows, vocabulary, steps, null);

        var flat = ids.SelectMany(s => s).ToList();
        int outside = flat.FindIndex(id => id < 0 || id >= vocabulary);
        checks.Check("ids in range", outside < 0, outside < 0 ? "" : $"token {flat[outside]} at step {outside / rows}, row {outside % rows}");

        bool penalties = options.RepeatPenalty != 1f || options.PresencePenalty != 0f || options.FrequencyPenalty != 0f;
        if (options.TopK == 1 || Exact)
        {
            checks.Compare(options.TopK == 1 ? "greedy tokens are the library's" : "the library's tokens", Comparisons.Difference(expected.SelectMany(s => s).ToList(), flat));
        }
        else if (penalties)
        {
            checks.Skip("tokens the settings allow", "penalties change the logits");
        }
        else
        {
            checks.Compare("tokens the settings allow", Allowed(logits, ids, rows, vocabulary, options));
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
            var reference = library.Read(0, steps);
            checks.Compare("the library's probabilities", Comparisons.Difference(
                [.. reference.SelectMany(s => s).Select(t => t.Probability)], [.. statistics.SelectMany(s => s).Select(t => t.Probability)], 1e-4f));
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
    private static string? Allowed(float[] logits, int[][] ids, int rows, int vocabulary, GenerationOptions options)
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
                    sum += p[i] = Math.Exp((row[i] - max) / options.Temperature);
                }

                double mine = p[id] / sum, above = 0, best = 1 / sum;
                int higher = 0;
                for (int i = 0; i < vocabulary; i++)
                {
                    if (p[i] / sum > mine * (1 + 1e-4))
                    {
                        higher++;
                        above += p[i] / sum;
                    }
                }

                string where = $"step {s}, row {r}: token {id} (probability {mine:G4})";
                if (options.TopK > 0 && higher >= options.TopK)
                {
                    return $"{where} is not among the top {options.TopK} ({higher} tokens are more likely)";
                }

                if (options.TopP < 1f && above >= options.TopP + 1e-4)
                {
                    return $"{where} is outside the top-p {options.TopP} nucleus (the likelier tokens hold {above:G4})";
                }

                if (options.MinP > 0f && mine < options.MinP * best * (1 - 1e-4))
                {
                    return $"{where} is below min-p {options.MinP} of the best token's {best:G4}";
                }
            }
        }

        return null;
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
