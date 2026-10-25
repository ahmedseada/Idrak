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

/// <summary>What a learning-rate schedule is made for (<see cref="ILearningRateSchedule.Create"/>).</summary>
/// <param name="Optimizer">The optimizer it schedules; its current rate is the peak the schedule works from.</param>
/// <param name="Steps">The run's number of optimizer steps (the scheduler is stepped once per step).</param>
/// <param name="Settings">The warm-up, minimum and decay the schedule reads.</param>
public sealed record LearningRateScheduleRequest(IOptimizer Optimizer, int Steps, LearningRateScheduleSettings Settings);

/// <summary>
/// A kind of learning-rate schedule, made by name: the scheduler of an optimizer over a run's number of optimizer steps,
/// with <see cref="LearningRateScheduleSettings"/>. The scheduler is stepped once per optimizer step and works from the
/// optimizer's own rate, its peak. Registered in <see cref="LearningRateSchedules"/> under its <see cref="IKind.Name"/>
/// ("cosine", "wsd"). Thread-safe: one kind makes the schedulers of several runs.
/// </summary>
public interface ILearningRateSchedule : IKind
{
    /// <summary>
    /// Refuses settings the schedule cannot use before a run starts (<see cref="LearningRateSchedules.Factory"/> asks it as
    /// the schedule is chosen); nothing by default, for a schedule that takes any.
    /// </summary>
    /// <exception cref="ArgumentException">A setting the schedule cannot use (a decay share outside [0, 1], say).</exception>
    void Check(LearningRateScheduleSettings settings)
    {
    }

    /// <summary>The scheduler of the request's optimizer over its steps, with its settings; it sets the first step's rate.</summary>
    /// <exception cref="ArgumentException">A setting the schedule cannot use.</exception>
    ILearningRateScheduler Create(LearningRateScheduleRequest request);
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

    private static readonly KindTable<ILearningRateSchedule> Table = BuiltIn();

    private static KindTable<ILearningRateSchedule> BuiltIn()
    {
        var table = new KindTable<ILearningRateSchedule>(nameof(LearningRateSchedules), "Learning-rate schedule", "add it with LearningRateSchedules.Register",
            comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a schedule sets the optimizer's rate at every step of a run from where it is, so another cannot take over half-way");
        table.RegisterDefault(new BuiltInSchedule(Cosine, "warm-up, then cosine decay to the minimum rate at the last step", CosineDecay));
        table.RegisterDefault(new BuiltInSchedule(Linear, "warm-up, then linear decay to the minimum rate at the last step", LinearDecay));
        table.RegisterDefault(new BuiltInSchedule(Constant, "warm-up, then the optimizer's rate unchanged", Unchanged));
        table.RegisterDefault(new BuiltInSchedule(WarmupStableDecay,
            "warm-up, the optimizer's rate, then linear decay to the minimum over the last decay share of the steps (MiniCPM)",
            StableThenDecay, CheckDecay));
        return table;
    }

    /// <summary>Registers <paramref name="schedule"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ILearningRateSchedule schedule) => Table.Register(schedule, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's schedule <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The schedule registered as <paramref name="name"/>, or null.</summary>
    public static ILearningRateSchedule? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The schedule registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ILearningRateSchedule Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's schedule <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ILearningRateSchedule? Default(string name) => Table.Default(name);

    /// <summary>Who registered the schedule <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered schedule as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>
    /// What makes the schedule <paramref name="name"/> with <paramref name="settings"/> (its defaults when null) from the
    /// optimizer and the number of optimizer steps. For command lines and settings files: the schedule is looked up and
    /// the settings checked now, so a bad name or setting fails here, before a run starts.
    /// </summary>
    /// <exception cref="NotSupportedException">The schedule is not registered.</exception>
    /// <exception cref="ArgumentException">A setting the schedule cannot use.</exception>
    public static Func<IOptimizer, int, ILearningRateScheduler> Factory(ReadOnlySpan<char> name, LearningRateScheduleSettings? settings = null)
    {
        var schedule = Get(name);
        var chosen = settings ?? new LearningRateScheduleSettings();
        schedule.Check(chosen);
        return (optimizer, steps) => schedule.Create(new LearningRateScheduleRequest(optimizer, steps, chosen))
            ?? throw new InvalidOperationException($"The learning-rate schedule '{schedule.Name}' made no scheduler.");
    }

    // Warm-up, then cosine decay to the minimum rate at the last step.
    private static LearningRateScheduler CosineDecay(LearningRateScheduleRequest r) =>
        new CosineAnnealing(r.Optimizer, Math.Max(1, r.Steps), r.Settings.MinLearningRate, Warmup(r.Settings.WarmupFraction, r.Steps));

    // Warm-up, then linear decay to the minimum rate at the last step.
    private static LearningRateScheduler LinearDecay(LearningRateScheduleRequest r)
    {
        int steps = r.Steps, warmup = Warmup(r.Settings.WarmupFraction, steps);
        float minLearningRate = r.Settings.MinLearningRate;
        return new LambdaSchedule(r.Optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1)
            : rate + (minLearningRate - rate) * (float)Math.Min(1.0, (step - warmup) / (double)Math.Max(1, steps - warmup)));
    }

    // Warm-up, then the optimizer's rate unchanged.
    private static LearningRateScheduler Unchanged(LearningRateScheduleRequest r)
    {
        int warmup = Warmup(r.Settings.WarmupFraction, r.Steps);
        return new LambdaSchedule(r.Optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1) : rate);
    }

    // Warm-up, then the optimizer's rate until the last decay share of the steps, then linear decay to the minimum rate.
    private static LearningRateScheduler StableThenDecay(LearningRateScheduleRequest r)
    {
        int steps = r.Steps, warmup = Warmup(r.Settings.WarmupFraction, steps);
        float minLearningRate = r.Settings.MinLearningRate;
        int decay = Math.Max(0, Math.Min(steps - warmup, (int)Math.Round(r.Settings.DecayFraction * steps)));
        int stable = steps - decay;
        return new LambdaSchedule(r.Optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1)
            : step < stable ? rate
            : rate + (minLearningRate - rate) * (float)Math.Min(1.0, (step - stable + 1) / (double)Math.Max(1, decay)));
    }

    private static void CheckDecay(LearningRateScheduleSettings settings)
    {
        if (settings.DecayFraction is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), settings.DecayFraction, "The decay fraction is in [0, 1].");
        }
    }

    private static int Warmup(float fraction, int steps) => (int)Math.Round(fraction * steps);

    private sealed class BuiltInSchedule(string name, string summary, Func<LearningRateScheduleRequest, LearningRateScheduler> create,
        Action<LearningRateScheduleSettings>? check = null) : ILearningRateSchedule
    {
        public string Name => name;

        public string Summary => summary;

        public void Check(LearningRateScheduleSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            check?.Invoke(settings);
        }

        public ILearningRateScheduler Create(LearningRateScheduleRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            ArgumentNullException.ThrowIfNull(request.Optimizer);
            Check(request.Settings);
            return create(request);
        }

        public override string ToString() => name;
    }
}
