// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of an optimizer kind (<see cref="IOptimizerKind"/>, registered in <see cref="OptimizerKinds"/>), each case a
/// few parameters of random values pulled towards targets at least 0.2 away by a squared loss for a few steps (rates up to
/// 0.03, momentum up to 0.5, so a descent method moves towards them):
/// <list type="bullet">
/// <item>the optimizer updates the parameters it was given, in their order, and starts at the rate asked for;</item>
/// <item>its first step lowers the loss, and so do its steps together; every value stays finite;</item>
/// <item>a parameter the loss does not reach keeps its values (cases without weight decay);</item>
/// <item>the same case gives the same values again (no state shared between the optimizers it makes);</item>
/// <item>given a <c>reference</c> (the library's kind of that name, or another implementation), the same values within
/// <see cref="Tolerance"/>.</item>
/// </list>
/// </summary>
/// <param name="reference">The kind to compare with, or null.</param>
/// <param name="device">Where the parameters live (the CPU when null).</param>
public sealed class OptimizerKindSuite(IOptimizerKind? reference = null, Device? device = null) : ContractSuite<IOptimizerKind>
{
    /// <inheritdoc />
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>The agreement wanted with the reference (see <see cref="Comparisons.Close"/>).</summary>
    public float Tolerance { get; init; } = 1e-5f;

    /// <inheritdoc />
    public override string Name => "optimizer-kind";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (var (parameters, size, steps, rate, decay, momentum) in new[]
                 {
                     (1, 1, 1, 0.01, 0.0, 0.0), (2, 5, 5, 0.01, 0.0, 0.5), (3, 17, 10, 0.003, 0.01, 0.0), (2, 64, 20, 0.02, 0.0, 0.25),
                     (1, 300, 8, 0.03, 0.1, 0.5), (4, 3, 12, 0.001, 0.0, 0.0),
                 })
        {
            yield return Case(parameters, size, steps, rate, decay, momentum, seed: parameters * 31 + size);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case(random.Next(1, 5), random.Next(1, large ? 5000 : 65), random.Next(1, large ? 31 : 13), 0.001 + random.NextDouble() * 0.029,
            random.Next(2) == 0 ? 0 : random.NextDouble() * 0.1, random.NextDouble() * 0.5, random.Next());
    }

    /// <inheritdoc />
    public override void Run(IOptimizerKind implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var trained = Train(implementation, @case.Data, checks);
        if (trained is not { } result)
        {
            return;
        }

        checks.Check("every value stays finite", result.Values.All(v => v.All(float.IsFinite)), "a parameter holds a value that is not finite");
        checks.Check("the first step lowers the loss", result.Losses[1] < result.Losses[0], $"loss {result.Losses[0]} before it, {result.Losses[1]} after");
        checks.Check("the steps lower the loss", result.Losses[^1] < result.Losses[0], $"loss {result.Losses[0]} at the start, {result.Losses[^1]} after the steps");
        if (result.Still is { } still)
        {
            checks.Compare("a parameter the loss does not reach keeps its values", Comparisons.Difference(still.Before, still.After, 0f));
        }

        if (Train(implementation, @case.Data, null) is { } again)
        {
            checks.Compare("the same case gives the same values", Comparisons.Difference(Flat(result.Values), Flat(again.Values), 0f));
        }

        if (reference is not null && Train(reference, @case.Data, null) is { } expected)
        {
            checks.Compare("the reference's values", Comparisons.Difference(Flat(expected.Values), Flat(result.Values), Tolerance));
        }
    }

    /// <inheritdoc />
    protected override string Describe(IOptimizerKind implementation) => $"{implementation.GetType().Name} '{implementation.Name}'";

    // The case's parameters trained by `kind`: their values after the steps, the loss before each step and after the
    // last, and the untouched parameter's values (cases without weight decay); construction checked when `checks` is given.
    private (float[][] Values, float[] Losses, (float[] Before, float[] After)? Still)? Train(IOptimizerKind kind, JsonObject data, CaseChecks? checks)
    {
        int count = (int)data["parameters"]!, size = (int)data["size"]!, steps = (int)data["steps"]!, seed = (int)data["seed"]!;
        var settings = new OptimizerSettings((float)(double)data["rate"]!, (float)(double)data["weight_decay"]!, (float)(double)data["momentum"]!);
        var random = new Random(seed);
        var parameters = new List<Tensor>();
        var targets = new List<Tensor>();
        try
        {
            for (int i = 0; i < count; i++)
            {
                float[] values = [.. Enumerable.Range(0, size).Select(_ => random.NextSingle() * 2f - 1f)];
                float[] goal = [.. values.Select(v => v + (random.Next(2) == 0 ? -1f : 1f) * (0.2f + random.NextSingle() * 0.8f))];
                parameters.Add(Tensor.Persistent(values, [size], Device, requiresGrad: true));
                targets.Add(Tensor.Persistent(goal, [size], Device));
            }

            float[] untouched = [.. Enumerable.Range(0, 3).Select(_ => random.NextSingle() * 2f - 1f)];
            bool keepsStill = settings.WeightDecay == 0f;
            if (keepsStill)
            {
                parameters.Add(Tensor.Persistent(untouched, [untouched.Length], Device, requiresGrad: true));
            }

            using var optimizer = kind.Create(new OptimizerRequest(parameters, settings));
            if (checks is not null)
            {
                checks.Check("it updates the parameters it was given, in order",
                    optimizer.Parameters.Count == parameters.Count && optimizer.Parameters.Select((p, i) => ReferenceEquals(p, parameters[i])).All(same => same),
                    $"{optimizer.Parameters.Count} parameters, {parameters.Count} given (or others in their place)");
                checks.Check("it starts at the rate asked for", optimizer.LearningRate == settings.LearningRate, $"rate {optimizer.LearningRate}, {settings.LearningRate} asked for");
            }

            var losses = new float[steps + 1];
            for (int step = 0; step <= steps; step++)
            {
                optimizer.ZeroGrad();
                using var scope = new TensorScope();
                Tensor? loss = null;
                for (int i = 0; i < count; i++)
                {
                    var term = (parameters[i] - targets[i]).Square().Sum();
                    loss = loss is null ? term : loss + term;
                }

                losses[step] = loss!.Item();
                if (step == steps)
                {
                    break;                                                       // the loss after the last step
                }

                loss.Backward();
                optimizer.Step();
            }

            float[][] trained = [.. parameters.Take(count).Select(p => p.ToArray())];
            return (trained, losses, keepsStill ? (untouched, parameters[^1].ToArray()) : null);
        }
        finally
        {
            foreach (var t in parameters.Concat(targets))
            {
                t.Dispose();
            }
        }
    }

    private static float[] Flat(float[][] values) => [.. values.SelectMany(v => v)];

    private static ContractCase Case(int parameters, int size, int steps, double rate, double decay, double momentum, int seed) =>
        new(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"{parameters} parameters of {size}, {steps} steps at rate {rate:G3}, weight decay {decay:G3}, momentum {momentum:G3}"),
            new JsonObject
            {
                ["parameters"] = parameters, ["size"] = size, ["steps"] = steps, ["rate"] = rate, ["weight_decay"] = decay, ["momentum"] = momentum, ["seed"] = seed,
            });
}

/// <summary>
/// The checks of a learning-rate schedule kind (<see cref="ILearningRateSchedule"/>, registered in
/// <see cref="LearningRateSchedules"/>), each case a schedule of an optimizer over a number of steps:
/// <list type="bullet">
/// <item>the schedule works from the optimizer it was given and its rate (its base);</item>
/// <item>every rate is finite, not negative, and not above the optimizer's own rate (the peak a schedule works from);</item>
/// <item>made again from the same settings, it gives the same rates (no state shared between schedules);</item>
/// <item>given a <c>reference</c> (the library's schedule of that name, or another implementation), the same rates within
/// <see cref="Tolerance"/> of the base rate.</item>
/// </list>
/// </summary>
/// <param name="reference">The schedule to compare with, or null.</param>
public sealed class LearningRateScheduleSuite(ILearningRateSchedule? reference = null) : ContractSuite<ILearningRateSchedule>
{
    /// <summary>The agreement wanted with the reference, relative to the base rate.</summary>
    public float Tolerance { get; init; } = 1e-6f;

    /// <inheritdoc />
    public override string Name => "learning-rate-schedule";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (var (steps, warmup, minimum, decay, rate) in new[]
                 {
                     (1, 0.0, 0.0, 0.2, 1.0), (2, 0.5, 0.0, 0.5, 1.0), (10, 0.2, 0.0, 0.3, 1.0), (10, 0.03, 0.1, 0.2, 3e-4), (100, 0.1, 0.05, 1.0, 2e-5),
                     (7, 0.0, 0.0, 0.0, 0.5), (1000, 0.03, 0.0, 0.2, 1e-3),
                 })
        {
            yield return Case(steps, warmup, minimum, decay, rate);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case(random.Next(1, large ? 100_001 : 501), random.NextDouble() * 0.5, random.NextDouble() * 0.5, random.NextDouble(), 1e-5 + random.NextDouble());
    }

    /// <inheritdoc />
    public override void Run(ILearningRateSchedule implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        int steps = (int)data["steps"]!;
        float rate = (float)(double)data["rate"]!;
        var settings = new LearningRateScheduleSettings((float)(double)data["warmup"]!, rate * (float)(double)data["minimum"]!, (float)(double)data["decay"]!);
        float[] rates = Rates(implementation, settings, steps, rate, checks);
        checks.Check("every rate is finite, not negative and not above the optimizer's", rates.All(r => float.IsFinite(r) && r >= 0f && r <= rate * (1f + 1e-6f)),
            $"a rate of {rates.FirstOrDefault(r => !(float.IsFinite(r) && r >= 0f && r <= rate * (1f + 1e-6f)))} with the optimizer's {rate}");
        checks.Compare("made again, the same rates", Comparisons.Difference(rates, Rates(implementation, settings, steps, rate, null), 0f));
        if (reference is not null)
        {
            float[] expected = Rates(reference, settings, steps, rate, null);
            checks.Compare("the reference's rates", Comparisons.Difference([.. expected.Select(r => r / rate)], [.. rates.Select(r => r / rate)], Tolerance));
        }
    }

    /// <inheritdoc />
    protected override string Describe(ILearningRateSchedule implementation) => $"{implementation.GetType().Name} '{implementation.Name}'";

    // The rate of each step of a schedule of an SGD over one parameter at `rate`; what it works from checked when `checks` is given.
    private static float[] Rates(ILearningRateSchedule kind, LearningRateScheduleSettings settings, int steps, float rate, CaseChecks? checks)
    {
        using var parameter = Tensor.Persistent([0f], [1], Device.Cpu, requiresGrad: true);
        using var optimizer = new Sgd([parameter], rate);
        var schedule = kind.Create(new LearningRateScheduleRequest(optimizer, steps, settings));
        checks?.Check("it works from the optimizer and its rate", ReferenceEquals(schedule.Optimizer, optimizer) && schedule.BaseLearningRate == rate,
            $"base rate {schedule.BaseLearningRate}, the optimizer's {rate}");
        var rates = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            rates[i] = optimizer.LearningRate;
            schedule.Step();
        }

        return rates;
    }

    private static ContractCase Case(int steps, double warmup, double minimum, double decay, double rate) =>
        new(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"{steps} steps at rate {rate:G3}, warm-up {warmup:G3}, minimum {minimum:G3} of the rate, decay {decay:G3}"),
            new JsonObject { ["steps"] = steps, ["warmup"] = warmup, ["minimum"] = minimum, ["decay"] = decay, ["rate"] = rate });
}
