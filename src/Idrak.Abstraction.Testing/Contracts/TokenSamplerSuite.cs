// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>What a sampler check asks a token sampler for: where it runs, its shape and its settings (as a generation's request carries them).</summary>
/// <param name="Device">Where the logits live.</param>
/// <param name="Rows">Sequences sampled per step.</param>
/// <param name="Vocabulary">Vocabulary size.</param>
/// <param name="Steps">Steps whose statistics are kept.</param>
/// <param name="HistoryCapacity">Recent tokens per row the penalties may look at.</param>
/// <param name="Temperature">Softmax temperature.</param>
/// <param name="TopK">Sample among the k most likely tokens (0: all).</param>
/// <param name="TopP">Nucleus sampling (1: off).</param>
/// <param name="MinP">Drop tokens less than this fraction as likely as the best (0: off).</param>
/// <param name="RepeatPenalty">Repetition penalty (1: off).</param>
/// <param name="RepeatLastN">Recent tokens the penalties look at.</param>
/// <param name="PresencePenalty">Subtracted from every token in the window.</param>
/// <param name="FrequencyPenalty">Subtracted once per occurrence in the window.</param>
/// <param name="Seed">The random seed.</param>
public sealed record SamplerSettings(Device Device, int Rows, int Vocabulary, int Steps, int HistoryCapacity, float Temperature, int TopK, float TopP,
    float MinP, float RepeatPenalty, int RepeatLastN, float PresencePenalty, float FrequencyPenalty, int Seed);

/// <summary>One sampled token's statistics, as a sampler reports them.</summary>
/// <param name="Id">The token.</param>
/// <param name="Probability">Its probability.</param>
/// <param name="Entropy">The entropy of the distribution sampled from, in bits.</param>
public readonly record struct SampledStatistics(int Id, float Probability, float Entropy);

/// <summary>
/// A token sampler as the sampler checks drive it. The token-sampler contract lives with the package that uses it
/// (<c>ITokenSampler</c> in Idrak.Nlp) and this kit depends on Idrak.Abstraction alone, so a test wraps the sampler it
/// checks, and the library's one it compares with, in these few delegates.
/// </summary>
/// <param name="rows">Sequences sampled per step.</param>
/// <param name="vocabulary">Vocabulary size.</param>
/// <param name="setHistory">Sets the recent tokens of every row (<c>SetHistory</c>).</param>
/// <param name="sample">Samples one step from [rows, vocabulary] logits and returns the ids, [rows].</param>
/// <param name="reset">Restarts from step 0 (<c>Reset</c>).</param>
/// <param name="read">The statistics of steps [from, to) as [step][row] (<c>Read</c>).</param>
/// <param name="owner">Disposed with this (the sampler itself), or null.</param>
public sealed class SamplerUnderTest(int rows, int vocabulary, Action<IReadOnlyList<int>> setHistory, Func<Tensor, float[]> sample, Action reset,
    Func<int, int, SampledStatistics[][]> read, IDisposable? owner = null) : IDisposable
{
    /// <summary>Sequences sampled per step.</summary>
    public int Rows { get; } = rows;

    /// <summary>Vocabulary size.</summary>
    public int Vocabulary { get; } = vocabulary;

    /// <summary>Sets the recent tokens of every row.</summary>
    public void SetHistory(IReadOnlyList<int> tokens) => setHistory(tokens);

    /// <summary>Samples one step; returns the ids, [rows].</summary>
    public float[] Sample(Tensor logits) => sample(logits);

    /// <summary>Restarts from step 0.</summary>
    public void Reset() => reset();

    /// <summary>The statistics of steps [<paramref name="from"/>, <paramref name="to"/>).</summary>
    public SampledStatistics[][] Read(int from, int to) => read(from, to);

    /// <summary>Disposes the sampler.</summary>
    public void Dispose() => owner?.Dispose();
}

/// <summary>
/// The checks of a token sampler against a reference sampler (the library's <c>TokenSampler</c>), both wrapped as
/// <see cref="SamplerUnderTest"/>, on the same settings, logits and history:
/// <list type="bullet">
/// <item>the sampler has the asked rows and vocabulary, and its ids are [rows] tokens in [0, vocabulary);</item>
/// <item>greedy settings (top-k 1) choose the reference's tokens, penalties included;</item>
/// <item>without penalties, every sampled token is one the settings allow (among the k largest distinct scores, inside
/// the top-p nucleus, at least min-p as likely as the best);</item>
/// <item>a reset with the same history samples the same tokens again;</item>
/// <item>the statistics report, per step and row, the token sampled with a probability in [0, 1];</item>
/// <item>with <see cref="Exact"/>, the very tokens and probabilities of the reference (same seed).</item>
/// </list>
/// </summary>
/// <param name="reference">Makes the reference sampler for a case's settings.</param>
/// <param name="device">Where the logits and the samplers live (the CPU when null).</param>
public sealed class TokenSamplerSuite(Func<SamplerSettings, SamplerUnderTest> reference, Device? device = null)
    : ContractSuite<Func<SamplerSettings, SamplerUnderTest>>
{
    /// <summary>Where the logits and the samplers live.</summary>
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>Whether every token must be the reference's (the same random stream), not only the greedy ones.</summary>
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
    public override void Run(Func<SamplerSettings, SamplerUnderTest> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int rows = (int)d["rows"]!, vocabulary = (int)d["vocabulary"]!, steps = (int)d["steps"]!;
        var settings = new SamplerSettings(Device, rows, vocabulary, steps, 64, (float)d["temperature"]!, (int)d["topK"]!, (float)d["topP"]!,
            (float)d["minP"]!, (float)d["repeatPenalty"]!, (int)d["repeatLastN"]!, (float)d["presencePenalty"]!, (float)d["frequencyPenalty"]!, (int)d["seed"]!);
        int[] history = [.. d["history"]!.AsArray().Select(n => (int)n!)];
        float[] logits = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String((string)d["logits"]!)).ToArray();

        using var mine = implementation(settings);
        using var library = reference(settings);
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

    // The ids sampled at each step (after the history is set), [steps][rows].
    private int[][] Sample(SamplerUnderTest sampler, int[] history, float[] logits, int rows, int vocabulary, int steps, CaseChecks? checks)
    {
        sampler.SetHistory(history);
        var ids = new int[steps][];
        for (int s = 0; s < steps; s++)
        {
            using var step = Tensor.From(logits.AsSpan(s * rows * vocabulary, rows * vocabulary), [rows, vocabulary], Device);
            var values = sampler.Sample(step);
            checks?.Check("ids are [rows]", values.Length == rows, $"{values.Length} ids for {rows} rows");
            ids[s] = [.. values.Take(rows).Select(v => (int)v)];
        }

        return ids;
    }

    // Null when every sampled token is among the top k, in the top-p nucleus and at least min-p as likely as the best
    // token (each with a little slack for rounding); else the first that is not.
    private static string? Allowed(float[] logits, int[][] ids, int rows, int vocabulary, SamplerSettings settings)
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
