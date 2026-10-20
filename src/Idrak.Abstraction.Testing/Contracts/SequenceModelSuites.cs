// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a sequence encoder (<see cref="ISequenceEncoder"/>, an encoder family's or the library's) on padded
/// batches of random token ids:
/// <list type="bullet">
/// <item>the states are [batch, time, width], every value finite, the same when encoded again;</item>
/// <item>a sequence's states are the same alone as in the padded batch, within <see cref="Tolerance"/> (no position
/// attends to padding);</item>
/// <item>other ids at the padded positions change no sequence's states;</item>
/// <item>for an encoder with token types: no types given equals every token of type 0, and other types change the
/// states; for one without, types are refused;</item>
/// <item>lengths that do not fit the batch are refused.</item>
/// </list>
/// </summary>
/// <param name="device">Where the ids are made: the encoder's device (the CPU when null).</param>
public sealed class SequenceEncoderSuite(Device? device = null) : ContractSuite<ISequenceEncoder>
{
    /// <inheritdoc />
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>The agreement wanted between a sequence alone and in the batch.</summary>
    public float Tolerance { get; init; } = 1e-4f;

    /// <inheritdoc />
    public override string Name => "sequence-encoder";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one sequence", 1, [5], 5);
        yield return Case("padded batch", 2, [6, 3, 1], 6);
        yield return Case("nothing padded", 3, [4, 4], 4);
        yield return Case("long and short", 4, [16, 2, 9, 16], 16);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int n = random.Next(1, large ? 9 : 5), t = random.Next(1, large ? 65 : 13);
        return Case("random", random.Next(), [.. Enumerable.Range(0, n).Select(_ => random.Next(1, t + 1))], t);
    }

    /// <inheritdoc />
    public override void Run(ISequenceEncoder implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int seed = (int)d["seed"]!, t = Math.Min((int)d["time"]!, implementation.MaxPositions);
        int[] lengths = [.. d["lengths"]!.AsArray().Select(l => Math.Min((int)l!, t))];
        int n = lengths.Length, width = implementation.Width, vocabulary = implementation.Vocabulary;
        var random = new Random(seed);
        float[] ids = [.. Enumerable.Range(0, n * t).Select(_ => (float)random.Next(vocabulary))];
        using var scope = new TensorScope();
        using var noGrad = Autograd.NoGrad();
        var states = implementation.Encode(Tensor.From(ids, [n, t], Device), lengths);
        bool shaped = states.Rank == 3 && states.Shape[0] == n && states.Shape[1] == t && states.Shape[2] == width;
        checks.Check("the states are [batch, time, width]", shaped, $"{Tensor.FormatShape(states.Shape)}, [{n}, {t}, {width}] expected");
        if (!shaped)
        {
            return;
        }

        float[] batch = states.ToArray();
        float[] real = Real(batch, lengths, t, width);
        checks.Check("every state finite", real.All(float.IsFinite), $"a state of {real.FirstOrDefault(v => !float.IsFinite(v))}");
        checks.Compare("the same states when encoded again", Comparisons.Difference(real, Real(implementation.Encode(Tensor.From(ids, [n, t], Device), lengths).ToArray(), lengths, t, width), 0f));

        string? alone = null;
        for (int b = 0; b < n && alone is null; b++)
        {
            int length = lengths[b];
            float[] single = implementation.Encode(Tensor.From(ids.AsSpan(b * t, length), [1, length], Device)).ToArray();
            alone = Comparisons.Difference(single, batch.AsSpan(b * t * width, length * width), Tolerance) is { } difference
                ? $"sequence {b} (length {length} of {t}): {difference}" : null;
        }

        checks.Compare("a sequence alone equals its states in the padded batch", alone);

        float[] otherIds = [.. ids];
        for (int b = 0; b < n; b++)
        {
            for (int p = lengths[b]; p < t; p++)
            {
                otherIds[b * t + p] = random.Next(vocabulary);
            }
        }

        checks.Compare("other ids at the padding change nothing", Comparisons.Difference(real,
            Real(implementation.Encode(Tensor.From(otherIds, [n, t], Device), lengths).ToArray(), lengths, t, width), Tolerance));

        if (implementation.TypeVocabulary > 0)
        {
            float[] zeros = Real(implementation.Encode(Tensor.From(ids, [n, t], Device), lengths, Tensor.Zeros([n, t], Device)).ToArray(), lengths, t, width);
            checks.Compare("no token types equal type 0 everywhere", Comparisons.Difference(real, zeros, Tolerance));
            if (implementation.TypeVocabulary > 1)
            {
                float[] types = [.. Enumerable.Repeat(1f, n * t)];                  // every token of type 1
                float[] typed = Real(implementation.Encode(Tensor.From(ids, [n, t], Device), lengths, Tensor.From(types, [n, t], Device)).ToArray(), lengths, t, width);
                checks.Check("other token types change the states", Comparisons.Difference(real, typed, Tolerance) is not null, "type 1 gave the states of type 0");
            }
        }
        else
        {
            checks.Check("token types refused by an encoder without them",
                Throws<ArgumentException>(() => implementation.Encode(Tensor.From(ids, [n, t], Device), lengths, Tensor.Zeros([n, t], Device))), "token types were taken");
        }

        checks.Check("lengths that do not fit the batch refused",
            Throws<ArgumentException>(() => implementation.Encode(Tensor.From(ids, [n, t], Device), [.. lengths, 1])), $"{n + 1} lengths for {n} sequences were taken");
    }

    // The states of the real positions only, sequence after sequence.
    private static float[] Real(float[] states, int[] lengths, int t, int width)
    {
        var real = new List<float>();
        for (int b = 0; b < lengths.Length; b++)
        {
            real.AddRange(states.AsSpan(b * t * width, lengths[b] * width));
        }

        return [.. real];
    }

    internal static bool Throws<TException>(Action action) where TException : Exception
    {
        try
        {
            action();
            return false;
        }
        catch (TException)
        {
            return true;
        }
    }

    private static ContractCase Case(string name, int seed, int[] lengths, int time) =>
        new($"{name}: lengths {string.Join(", ", lengths)} of {time}", new JsonObject
        {
            ["seed"] = seed, ["time"] = time, ["lengths"] = new JsonArray([.. lengths.Select(l => (JsonNode)l)]),
        });
}

/// <summary>
/// The checks of a text encoder-decoder (<see cref="ISeq2SeqModel"/>, a family's or the library's) on random sources and
/// targets:
/// <list type="bullet">
/// <item>the memory is [batch, sourceTime, width] and the logits [batch, targetTime, vocabulary], finite;</item>
/// <item>cached decoding equals the whole target at once: the logits of each step through the context (the memory set
/// once) equal those of <see cref="ISeq2SeqModel.Decode"/> at that position, and so do a prefill of several positions
/// followed by single steps, within <see cref="Tolerance"/>;</item>
/// <item>a padded batch of sources and targets gives each sequence the logits it has alone;</item>
/// <item>a memory of one sequence serves a context of two rows: both rows decode as the sequence alone;</item>
/// <item>greedy decoding through the cache picks the tokens greedy decoding of the whole target at each step picks.</item>
/// </list>
/// </summary>
/// <param name="device">Where the ids are made: the model's device (the CPU when null).</param>
public sealed class Seq2SeqModelSuite(Device? device = null) : ContractSuite<ISeq2SeqModel>
{
    /// <inheritdoc />
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>The agreement wanted between the cached and the whole decoding, and between the batch and one by one.</summary>
    public float Tolerance { get; init; } = 1e-4f;

    /// <inheritdoc />
    public override string Name => "seq2seq-model";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one step", 1, [3], 1);
        yield return Case("short source, longer target", 2, [2, 5], 6);
        yield return Case("padded sources", 3, [7, 2, 4], 5);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int n = random.Next(1, large ? 6 : 4), s = random.Next(1, large ? 33 : 10);
        return Case("random", random.Next(), [.. Enumerable.Range(0, n).Select(_ => random.Next(1, s + 1))], random.Next(1, large ? 17 : 8));
    }

    /// <inheritdoc />
    public override void Run(ISeq2SeqModel implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int seed = (int)d["seed"]!;
        int[] sources = [.. d["sources"]!.AsArray().Select(l => Math.Min((int)l!, implementation.Encoder.MaxPositions))];
        int steps = Math.Min((int)d["steps"]!, implementation.MaxTargetPositions);
        int n = sources.Length, s = sources.Max(), v = implementation.Vocabulary, sourceVocabulary = implementation.Encoder.Vocabulary;
        var random = new Random(seed);
        float[] source = [.. Enumerable.Range(0, n * s).Select(_ => (float)random.Next(sourceVocabulary))];
        float[] target = [.. Enumerable.Range(0, n * steps).Select(i => (float)(i % steps == 0 ? implementation.DecoderStartToken : random.Next(v)))];
        using var scope = new TensorScope();
        using var noGrad = Autograd.NoGrad();
        implementation.Network.Eval();
        var memory = implementation.Encoder.Encode(Tensor.From(source, [n, s], Device), sources);
        checks.Check("the memory is [batch, sourceTime, width]", memory.Rank == 3 && memory.Shape[0] == n && memory.Shape[1] == s && memory.Shape[2] == implementation.Encoder.Width,
            $"{Tensor.FormatShape(memory.Shape)}, [{n}, {s}, {implementation.Encoder.Width}] expected");
        var logits = implementation.Decode(Tensor.From(target, [n, steps], Device), memory, sources);
        bool shaped = logits.Rank == 3 && logits.Shape[0] == n && logits.Shape[1] == steps && logits.Shape[2] == v;
        checks.Check("the logits are [batch, targetTime, vocabulary]", shaped, $"{Tensor.FormatShape(logits.Shape)}, [{n}, {steps}, {v}] expected");
        if (!shaped)
        {
            return;
        }

        float[] whole = logits.ToArray();
        checks.Check("every logit finite", whole.All(float.IsFinite), $"a logit of {whole.FirstOrDefault(x => !float.IsFinite(x))}");

        // Sequence by sequence: alone (its own source length, no padding), cached step by step, a prefill then steps, and
        // through a context of two rows over its one memory.
        string? alone = null, cached = null, prefilled = null, rows = null;
        for (int b = 0; b < n; b++)
        {
            int length = sources[b];
            var own = implementation.Encoder.Encode(Tensor.From(source.AsSpan(b * s, length), [1, length], Device));
            var ownTarget = target.AsSpan(b * steps, steps).ToArray();
            float[] single = implementation.Decode(Tensor.From(ownTarget, [1, steps], Device), own).ToArray();
            alone ??= Comparisons.Difference(single, whole.AsSpan(b * steps * v, steps * v), Tolerance) is { } a ? $"sequence {b}: {a}" : null;
            cached ??= Cached(implementation, own, ownTarget, 1, 1, v) is var c && Comparisons.Difference(single, c, Tolerance) is { } cd ? $"sequence {b}: {cd}" : null;
            int prefill = Math.Max(1, steps / 2);
            prefilled ??= Cached(implementation, own, ownTarget, prefill, 1, v) is var p && Comparisons.Difference(single, p, Tolerance) is { } pd ? $"sequence {b}: {pd}" : null;
            rows ??= Cached(implementation, own, ownTarget, 1, 2, v) is var r && Comparisons.Difference([.. single, .. single], r, Tolerance) is { } rd ? $"sequence {b}: {rd}" : null;
        }

        checks.Compare("a padded batch gives each sequence its logits alone", alone);
        checks.Compare("cached steps equal the whole target at once", cached);
        checks.Compare("a cached prefill and steps equal the whole target", prefilled);
        checks.Compare("one memory serves two rows", rows);
        var first = implementation.Encoder.Encode(Tensor.From(source.AsSpan(0, sources[0]), [1, sources[0]], Device));
        checks.Compare("greedy through the cache picks the whole target's tokens", Greedy(implementation, first, steps));
    }

    // The logits [rows, steps, v] of the target through a context of `rows` rows over the memory: the first `prefill`
    // positions at once, then one at a time (every position's logits kept).
    private float[] Cached(ISeq2SeqModel model, Tensor memory, float[] target, int prefill, int rows, int v)
    {
        int steps = target.Length;
        using var context = new DecodingContext(Device, rows, steps);
        model.SetMemory(memory, context);
        var result = new float[rows * steps * v];
        int at = 0;
        while (at < steps)
        {
            int count = at == 0 ? Math.Min(prefill, steps) : 1;
            var ids = new float[rows * count];
            for (int r = 0; r < rows; r++)
            {
                target.AsSpan(at, count).CopyTo(ids.AsSpan(r * count));
            }

            float[] step = model.DecodeCached(Tensor.From(ids, [rows, count], Device), context).ToArray();
            for (int r = 0; r < rows; r++)
            {
                step.AsSpan(r * count * v, count * v).CopyTo(result.AsSpan((r * steps + at) * v));
            }

            at += count;
        }

        return result;
    }

    // Null when greedy decoding through the cache picks, at every step, the token greedy decoding of the whole target so
    // far picks; else where they part.
    private string? Greedy(ISeq2SeqModel model, Tensor memory, int steps)
    {
        var tokens = new List<float> { model.DecoderStartToken };
        using var context = new DecodingContext(Device, 1, steps);
        model.SetMemory(memory, context);
        int v = model.Vocabulary;
        for (int i = 0; i < steps - 1; i++)
        {
            float[] step = model.DecodeCached(Tensor.From([tokens[^1]], [1, 1], Device), context).ToArray();
            float[] all = model.Decode(Tensor.From([.. tokens], [1, tokens.Count], Device), memory).ToArray();
            int cachedPick = ArgMax(step.AsSpan(0, v)), wholePick = ArgMax(all.AsSpan((tokens.Count - 1) * v, v));
            if (cachedPick != wholePick)
            {
                return $"step {i}: the cache picks {cachedPick}, the whole target {wholePick}";
            }

            tokens.Add(cachedPick);
        }

        return null;
    }

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
        {
            best = row[i] > row[best] ? i : best;
        }

        return best;
    }

    private static ContractCase Case(string name, int seed, int[] sources, int steps) =>
        new($"{name}: sources {string.Join(", ", sources)}, {steps} target steps", new JsonObject
        {
            ["seed"] = seed, ["steps"] = steps, ["sources"] = new JsonArray([.. sources.Select(l => (JsonNode)l)]),
        });
}
