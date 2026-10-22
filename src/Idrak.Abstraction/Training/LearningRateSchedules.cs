// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Training;

/// <summary>
/// What a learning-rate schedule is made with, by name (<see cref="LearningRateSchedules"/>). A schedule reads what it uses
/// and leaves the rest.
/// </summary>
/// <param name="WarmupFraction">The share of the steps the rate rises over, linearly to the optimizer's own (step i of w uses (i + 1) / (w + 1) of it).</param>
/// <param name="MinLearningRate">The rate a decaying schedule ends at.</param>
/// <param name="DecayFraction">The share of the steps a warm-up, stable, decay schedule decays over (from 0 to 1).</param>
public sealed record LearningRateScheduleSettings(float WarmupFraction = 0.03f, float MinLearningRate = 0f, float DecayFraction = 0.2f);

/// <summary>
/// A kind of learning-rate schedule, made by name: from <see cref="LearningRateScheduleSettings"/>, what makes the schedule
/// of an optimizer over a run's number of optimizer steps (the form <c>FineTuningOptions.Scheduler</c> takes). The schedule
/// is stepped once per optimizer step and works from the optimizer's own rate, its peak. Registered in
/// <see cref="LearningRateSchedules"/> under <see cref="Name"/>. Thread-safe: one kind makes the schedules of several runs.
/// </summary>
public interface ILearningRateSchedule
{
    /// <summary>The schedule's name in <see cref="LearningRateSchedules"/> ("cosine", "wsd").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>What makes the schedule from the optimizer and the number of optimizer steps, with <paramref name="settings"/>.</summary>
    /// <exception cref="ArgumentException">A setting the schedule cannot use (a decay share outside [0, 1], say), refused here, before a run starts.</exception>
    Func<Optimizer, int, LearningRateScheduler> Create(LearningRateScheduleSettings settings);
}

/// <summary>
/// The learning-rate schedules trainers make by name (ignoring case), each with a warm-up over
/// <see cref="LearningRateScheduleSettings.WarmupFraction"/> of the steps. The library's:
/// <list type="bullet">
/// <item>"cosine": cosine decay to <see cref="LearningRateScheduleSettings.MinLearningRate"/> at the last step;</item>
/// <item>"linear": linear decay to the minimum at the last step;</item>
/// <item>"constant": the optimizer's rate unchanged;</item>
/// <item>"wsd": warm-up, stable, decay (Hu et al. 2024, "MiniCPM", arXiv:2404.06395): the optimizer's rate until the last
/// <see cref="LearningRateScheduleSettings.DecayFraction"/> of the steps, then linear decay to the minimum.</item>
/// </list>
/// Register another with <see cref="Register"/>; one under a library name takes the library's place, which
/// <see cref="Unregister"/> brings back. A schedule sets the optimizer's rate at every step of a run from where it is, so
/// another cannot take over half-way: the registry has no fallback.
/// </summary>
public static class LearningRateSchedules
{
    /// <summary>Warm-up, then cosine decay: "cosine" (the fine-tuner's default).</summary>
    public const string Cosine = "cosine";

    /// <summary>Warm-up, then linear decay: "linear".</summary>
    public const string Linear = "linear";

    /// <summary>Warm-up, then the optimizer's rate: "constant".</summary>
    public const string Constant = "constant";

    /// <summary>Warm-up, stable, decay: "wsd".</summary>
    public const string WarmupStableDecay = "wsd";

    private static readonly SlotTable<string, ILearningRateSchedule> Table = BuiltIn();

    private static SlotTable<string, ILearningRateSchedule> BuiltIn()
    {
        var table = new SlotTable<string, ILearningRateSchedule>(nameof(LearningRateSchedules), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a schedule sets the optimizer's rate at every step of a run from where it is, so another cannot take over half-way");
        table.RegisterDefault(Cosine, new BuiltInSchedule(Cosine, "warm-up, then cosine decay to the minimum rate at the last step",
            s => CosineDecay(s.WarmupFraction, s.MinLearningRate)));
        table.RegisterDefault(Linear, new BuiltInSchedule(Linear, "warm-up, then linear decay to the minimum rate at the last step",
            s => LinearDecay(s.WarmupFraction, s.MinLearningRate)));
        table.RegisterDefault(Constant, new BuiltInSchedule(Constant, "warm-up, then the optimizer's rate unchanged", s => Unchanged(s.WarmupFraction)));
        table.RegisterDefault(WarmupStableDecay, new BuiltInSchedule(WarmupStableDecay,
            "warm-up, the optimizer's rate, then linear decay to the minimum over the last decay share of the steps (MiniCPM)",
            s => StableThenDecay(s.WarmupFraction, s.DecayFraction, s.MinLearningRate)));
        return table;
    }

    /// <summary>Registers <paramref name="schedule"/> under its <see cref="ILearningRateSchedule.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ILearningRateSchedule schedule)
    {
        ArgumentNullException.ThrowIfNull(schedule);
        ArgumentException.ThrowIfNullOrWhiteSpace(schedule.Name);
        Table.Register(schedule.Name, schedule, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's schedule <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The schedule registered as <paramref name="name"/>, or null.</summary>
    public static ILearningRateSchedule? Find(string name) => Table.Find(name);

    /// <summary>The schedule registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ILearningRateSchedule Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Learning-rate schedule '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with LearningRateSchedules.Register.");

    /// <summary>The library's schedule <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ILearningRateSchedule? Default(string name) => Table.Default(name);

    /// <summary>Who registered the schedule <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered schedule as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));

    /// <summary>
    /// What makes the schedule <paramref name="name"/> with <paramref name="settings"/> (its defaults when null) from the
    /// optimizer and the number of optimizer steps. For command lines and settings files.
    /// </summary>
    /// <exception cref="NotSupportedException">The schedule is not registered.</exception>
    /// <exception cref="ArgumentException">A setting the schedule cannot use.</exception>
    public static Func<Optimizer, int, LearningRateScheduler> Factory(string name, LearningRateScheduleSettings? settings = null)
    {
        var schedule = Get(name);
        return schedule.Create(settings ?? new LearningRateScheduleSettings())
            ?? throw new InvalidOperationException($"The learning-rate schedule '{schedule.Name}' made no factory.");
    }

    // Warm-up, then cosine decay to minLearningRate at the last step.
    private static Func<Optimizer, int, LearningRateScheduler> CosineDecay(float warmupFraction, float minLearningRate) =>
        (optimizer, steps) => new CosineAnnealing(optimizer, Math.Max(1, steps), minLearningRate, Warmup(warmupFraction, steps));

    // Warm-up, then linear decay to minLearningRate at the last step.
    private static Func<Optimizer, int, LearningRateScheduler> LinearDecay(float warmupFraction, float minLearningRate) =>
        (optimizer, steps) =>
        {
            int warmup = Warmup(warmupFraction, steps);
            return new LambdaSchedule(optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1)
                : rate + (minLearningRate - rate) * (float)Math.Min(1.0, (step - warmup) / (double)Math.Max(1, steps - warmup)));
        };

    // Warm-up, then the optimizer's rate unchanged.
    private static Func<Optimizer, int, LearningRateScheduler> Unchanged(float warmupFraction) =>
        (optimizer, steps) =>
        {
            int warmup = Warmup(warmupFraction, steps);
            return new LambdaSchedule(optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1) : rate);
        };

    // Warm-up, then the optimizer's rate until the last decayFraction of the steps, then linear decay to minLearningRate.
    private static Func<Optimizer, int, LearningRateScheduler> StableThenDecay(float warmupFraction, float decayFraction, float minLearningRate)
    {
        if (decayFraction is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(decayFraction), "The decay fraction is in [0, 1].");
        }

        return (optimizer, steps) =>
        {
            int warmup = Warmup(warmupFraction, steps);
            int decay = Math.Max(0, Math.Min(steps - warmup, (int)Math.Round(decayFraction * steps)));
            int stable = steps - decay;
            return new LambdaSchedule(optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1)
                : step < stable ? rate
                : rate + (minLearningRate - rate) * (float)Math.Min(1.0, (step - stable + 1) / (double)Math.Max(1, decay)));
        };
    }

    private static int Warmup(float fraction, int steps) => (int)Math.Round(fraction * steps);

    private sealed class BuiltInSchedule(string name, string summary, Func<LearningRateScheduleSettings, Func<Optimizer, int, LearningRateScheduler>> create)
        : ILearningRateSchedule
    {
        public string Name => name;

        public string Summary => summary;

        public Func<Optimizer, int, LearningRateScheduler> Create(LearningRateScheduleSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            return create(settings);
        }

        public override string ToString() => name;
    }
}
