// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Inference.Abstractions;

/// <summary>One reading of a sequence: its labels (repeats collapsed, blanks dropped) and their log-probability.</summary>
/// <param name="Labels">The labels, in order.</param>
/// <param name="LogProbability">The decoder's score: the log-probability of the best path for "greedy", of the labels
/// summed over the paths the beam kept for "beam".</param>
public sealed record CtcHypothesis(int[] Labels, float LogProbability);

/// <summary>What a CTC decoder is asked for.</summary>
public sealed record CtcDecodeOptions
{
    /// <summary>The blank class (default 0).</summary>
    public int Blank { get; init; }

    /// <summary>How many readings to return, the most likely first (default 1).</summary>
    public int Results { get; init; } = 1;

    /// <summary>The prefixes a beam search keeps after each step (default 10).</summary>
    public int BeamWidth { get; init; } = 10;

    /// <summary>
    /// Per-step pruning for a beam search: only the <see cref="TopClasses"/> most likely classes of a step extend the
    /// prefixes (null: every class). With many classes and few likely ones per step it is much faster for the same result.
    /// </summary>
    public int? TopClasses { get; init; }

    /// <summary>
    /// Per-step pruning for a beam search: classes whose log-probability at a step is below this extend no prefix
    /// (null: none skipped). The step's most likely class is always kept.
    /// </summary>
    public float? MinLogProbability { get; init; }
}

/// <summary>
/// Reads one sequence's log-probabilities, <paramref name="steps"/> rows of <paramref name="classes"/> (a log-softmax per
/// step, as a network trained with CTC gives them), as label sequences, the most likely first.
/// </summary>
public delegate IReadOnlyList<CtcHypothesis> CtcDecoder(ReadOnlyMemory<float> logProbs, int steps, int classes, CtcDecodeOptions options);

/// <summary>
/// The CTC decoders, by name: the library registers "greedy" (best path: the most likely class at each step, repeats
/// collapsed and blanks dropped) and "beam" (prefix beam search, Hannun et al. 2014, "First-Pass Large Vocabulary
/// Continuous Speech Recognition using Bi-Directional Recurrent DNNs", with <see cref="CtcDecodeOptions.BeamWidth"/> and
/// per-step pruning). Register a decoder of your own (a lexicon, a language model) under a new name, or under a built-in
/// name to replace it: the built-in stays behind it until <see cref="Unregister"/>.
/// </summary>
public static class CtcDecoders
{
    /// <summary>"greedy": the best path.</summary>
    public const string Greedy = "greedy";

    /// <summary>"beam": prefix beam search.</summary>
    public const string Beam = "beam";

    private static readonly SlotTable<string, CtcDecoder> Registry = BuiltIn();

    private static SlotTable<string, CtcDecoder> BuiltIn()
    {
        var table = new SlotTable<string, CtcDecoder>(nameof(CtcDecoders), Guard, StringComparer.Ordinal);
        table.RegisterDefault(Greedy, CtcDecoding.Greedy);
        table.RegisterDefault(Beam, CtcDecoding.BeamSearch);
        return table;
    }

    /// <summary>Registers the decoder <paramref name="name"/>; under a built-in name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, CtcDecoder decoder)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(decoder);
        Registry.Register(name, decoder, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's decoder <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered decoder names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The decoder <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    public static CtcDecoder Get(string name) =>
        Registry.TryGet(name, out var decoder) ? decoder
            : throw new NotSupportedException($"No CTC decoder '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with CtcDecoders.Register.");

    /// <summary>The library's decoder <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static CtcDecoder? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the decoder <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's decoder <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error
    /// reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the
    /// library's (<see cref="SlotPolicy.Shadow"/>: the library's answers, the readings are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Decodes one sequence's [steps, classes] log-probabilities with the decoder <paramref name="name"/>.</summary>
    public static IReadOnlyList<CtcHypothesis> Decode(ReadOnlyMemory<float> logProbs, int steps, int classes, string name = Greedy, CtcDecodeOptions? options = null)
    {
        Check(logProbs.Length, steps, classes, options ??= new());
        return Get(name)(logProbs, steps, classes, options);
    }

    /// <summary>
    /// Decodes a batch: <paramref name="logProbs"/> is [steps, batch, classes] or, with <paramref name="batchFirst"/>,
    /// [batch, steps, classes] (as <see cref="Tensor.CtcLoss"/> reads it); sequence n reads its first
    /// <paramref name="lengths"/>[n] steps (all of them when null). One list of readings per sequence.
    /// </summary>
    public static IReadOnlyList<CtcHypothesis>[] Decode(Tensor logProbs, IReadOnlyList<int>? lengths = null, string name = Greedy, CtcDecodeOptions? options = null,
        bool batchFirst = false)
    {
        ArgumentNullException.ThrowIfNull(logProbs);
        if (logProbs.Rank != 3)
        {
            throw new ArgumentException($"CTC decoding takes log-probabilities [{(batchFirst ? "batch, steps" : "steps, batch")}, classes], not {Tensor.FormatShape(logProbs.Shape)}.");
        }

        int steps = logProbs.Shape[batchFirst ? 1 : 0], batch = logProbs.Shape[batchFirst ? 0 : 1], classes = logProbs.Shape[2];
        if (lengths is not null && (lengths.Count != batch || lengths.Any(l => (uint)l > (uint)steps)))
        {
            throw new ArgumentException($"CTC decoding needs a length in [0, {steps}] for each of the {batch} sequences.");
        }

        var values = logProbs.ToArray();
        var decoder = Get(name);
        options ??= new();
        var results = new IReadOnlyList<CtcHypothesis>[batch];
        for (int n = 0; n < batch; n++)
        {
            int length = lengths?[n] ?? steps;
            float[] rows;
            if (batchFirst)
            {
                rows = values.AsSpan(n * steps * classes, length * classes).ToArray();
            }
            else
            {
                rows = new float[length * classes];
                for (int t = 0; t < length; t++)
                {
                    values.AsSpan((t * batch + n) * classes, classes).CopyTo(rows.AsSpan(t * classes));
                }
            }

            Check(rows.Length, length, classes, options);
            results[n] = decoder(rows, length, classes, options);
        }

        return results;
    }

    private static void Check(int values, int steps, int classes, CtcDecodeOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (steps < 0 || classes <= 0 || values != steps * classes)
        {
            throw new ArgumentException($"{values} log-probabilities are not {steps} steps of {classes} classes.");
        }

        if ((uint)options.Blank >= (uint)classes || options.Results < 1 || options.BeamWidth < 1 || options.TopClasses is < 1)
        {
            throw new ArgumentException($"CTC decoding needs a blank among the {classes} classes and at least one result, beam entry and class a step; got {options}.");
        }
    }

    // An app's decoder under its policy: Throw and FallBack per call; Shadow answers with the library's and compares the labels.
    private static CtcDecoder Guard(Slot slot, CtcDecoder app, CtcDecoder library) => (logProbs, steps, classes, options) =>
        slot.Call(() => app(logProbs, steps, classes, options), () => library(logProbs, steps, classes, options), (a, b) =>
            a.Count == b.Count && a.Zip(b).All(p => p.First.Labels.SequenceEqual(p.Second.Labels)) ? null
                : $"labels [{string.Join(" | ", a.Select(h => string.Join(",", h.Labels)))}] and [{string.Join(" | ", b.Select(h => string.Join(",", h.Labels)))}]");
}
