// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Vision.Abstractions;

/// <summary>The settings of a box matcher (<see cref="BoxMatchers"/>); each matcher reads those it has.</summary>
public sealed record BoxMatchOptions
{
    /// <summary>Threshold matching: a prediction whose best quality is at least this is matched to that truth (default 0.5).</summary>
    public float PositiveThreshold { get; init; } = 0.5f;

    /// <summary>
    /// Threshold matching: a prediction whose best quality is below this is a negative (default 0.4); between the two
    /// thresholds it is ignored (<see cref="BoxMatchers.Ignored"/>), neither trained as an object nor as background.
    /// </summary>
    public float NegativeThreshold { get; init; } = 0.4f;

    /// <summary>
    /// Threshold matching: every truth also keeps the predictions of its own best quality, even below the thresholds
    /// (torchvision's <c>allow_low_quality_matches</c>), so no object goes without a prediction (default false).
    /// </summary>
    public bool AllowLowQuality { get; init; }
}

/// <summary>
/// Assigns predictions (anchors, or a set predictor's outputs) to the true objects from a quality matrix
/// [predictions, truths] (row-major, higher is better: an IoU, or a negated cost), registered in <see cref="BoxMatchers"/>
/// under its <see cref="IKind.Name"/>. The result holds, for each prediction, the index of its truth,
/// <see cref="BoxMatchers.Negative"/> or <see cref="BoxMatchers.Ignored"/>. A quality of -∞ forbids the pair.
/// </summary>
public interface IBoxMatcher : IKind
{
    /// <summary>The assignment of each of <paramref name="predictions"/> predictions to one of <paramref name="truths"/> truths (or none).</summary>
    int[] Match(ReadOnlyMemory<float> quality, int predictions, int truths, BoxMatchOptions options);
}

/// <summary>
/// The box matchers, by name. The library's: "iou-threshold" (each prediction to its best truth when the quality passes
/// <see cref="BoxMatchOptions.PositiveThreshold"/>, a negative below <see cref="BoxMatchOptions.NegativeThreshold"/>,
/// ignored between; torchvision's <c>Matcher</c>) and "hungarian" (one to one, the largest total quality, by the
/// Hungarian algorithm in O(n³): set prediction). Build quality matrices with <c>BoxMatching</c> (Idrak.Vision). Register
/// another under a new name, or under a library name to replace it: the library's stays behind it until <see cref="Unregister"/>.
/// </summary>
public static class BoxMatchers
{
    /// <summary>A prediction matched to no object: background.</summary>
    public const int Negative = -1;

    /// <summary>A prediction left out of training (between the thresholds).</summary>
    public const int Ignored = -2;

    /// <summary>"iou-threshold": threshold matching.</summary>
    public const string Threshold = "iou-threshold";

    /// <summary>"hungarian": optimal one-to-one matching.</summary>
    public const string Hungarian = "hungarian";

    private static readonly KindTable<IBoxMatcher> Registry = BuiltIn();

    private static KindTable<IBoxMatcher> BuiltIn()
    {
        var table = new KindTable<IBoxMatcher>(nameof(BoxMatchers), "Box matcher", "add it with BoxMatchers.Register", Guard, StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(new Matcher(Threshold, "each prediction to its best truth above the positive threshold, a negative below the negative one, ignored between",
            BoxMatching.MatchByThreshold));
        table.RegisterDefault(new Matcher(Hungarian, "one to one, the largest total quality (the Hungarian algorithm)", BoxMatching.MatchOneToOne));
        return table;
    }

    /// <summary>Registers <paramref name="matcher"/> under its name; under a library name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IBoxMatcher matcher) => Registry.Register(matcher, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's matcher <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered matcher names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The matcher <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names those that are.</exception>
    public static IBoxMatcher Get(ReadOnlySpan<char> name) => Registry.TryGet(name, out var matcher) ? matcher
        : throw new NotSupportedException($"No box matcher '{name}' is registered ({string.Join(", ", Registry.Names)}); add it with BoxMatchers.Register.");

    /// <summary>The library's matcher <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IBoxMatcher? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the matcher <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's matcher <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's answers, the assignments are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Matches with the matcher <paramref name="name"/>, checking the matrix's size and the answer's.</summary>
    public static int[] Match(string name, ReadOnlyMemory<float> quality, int predictions, int truths, BoxMatchOptions? options = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(predictions);
        ArgumentOutOfRangeException.ThrowIfNegative(truths);
        if (quality.Length != (long)predictions * truths)
        {
            throw new ArgumentException($"{quality.Length} qualities are not {predictions} predictions x {truths} truths.", nameof(quality));
        }

        var matches = Get(name).Match(quality, predictions, truths, options ?? new());
        if (matches.Length != predictions || matches.Any(m => m < Ignored || m >= truths))
        {
            throw new InvalidOperationException($"The box matcher '{name}' gave {matches.Length} assignments for {predictions} predictions, or one outside [{Ignored}, {truths}).");
        }

        return matches;
    }

    private static IBoxMatcher Guard(Slot slot, IBoxMatcher app, IBoxMatcher library) => new Matcher(app.Name, app.Summary, (quality, predictions, truths, options) =>
        slot.Call(() => app.Match(quality, predictions, truths, options), () => library.Match(quality, predictions, truths, options), (a, b) => Comparisons.Difference(a, b)));

    // A matcher as a function.
    private sealed class Matcher(string name, string summary, Func<ReadOnlyMemory<float>, int, int, BoxMatchOptions, int[]> match) : IBoxMatcher
    {
        public string Name => name;

        public string Summary => summary;

        public int[] Match(ReadOnlyMemory<float> quality, int predictions, int truths, BoxMatchOptions options) => match(quality, predictions, truths, options);
    }
}
