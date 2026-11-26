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
/// A kind of learning-rate schedule, made by name: the scheduler of an optimizer over a run's number of optimizer
/// steps,
/// with <see cref="LearningRateScheduleSettings"/>. The scheduler is stepped once per optimizer step and works from the
/// optimizer's own rate, its peak. Registered in <see cref="LearningRateSchedules"/> under its <c>Name</c>
/// ("cosine", "wsd"). Thread-safe: one kind makes the schedulers of several runs.
/// </summary>
public interface ILearningRateSchedule
{
    /// <summary>
    /// Refuses settings the schedule cannot use before a run starts (<c>Factory</c> asks
    /// it as the schedule is chosen); nothing by default, for a schedule that takes any.
    /// </summary>
    /// <exception cref="ArgumentException">A setting the schedule cannot use (a decay share outside [0, 1],
    /// say).</exception>
    void Check(LearningRateScheduleSettings settings)
    {
    }

    /// <summary>The scheduler of the request's optimizer over its steps, with its settings; it sets the first step's rate.</summary>
    /// <exception cref="ArgumentException">A setting the schedule cannot use.</exception>
    ILearningRateScheduler Create(LearningRateScheduleRequest request);
}

/// <summary>
/// The learning-rate schedules trainers make by name (ignoring case), each with a warm-up over <see cref="LearningRateScheduleSettings.WarmupFraction"/> of the steps. The library's: A schedule sets the optimizer's
/// rate at every step of a run from where it is, so another cannot take over half-way: the registry has no fallback.
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
