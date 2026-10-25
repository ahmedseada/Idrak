// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Training;

/// <summary>
/// Changes an optimizer's learning rate over epochs (or optimizer steps, as a fine-tuner steps it): the rate of the first
/// is set when the scheduler is made, and each <see cref="Step"/> sets the next one's. The <c>Trainer</c> steps it once
/// per epoch when given one; <see cref="LearningRateSchedules"/> makes them by name.
/// </summary>
public interface ILearningRateScheduler
{
    /// <summary>The optimizer being scheduled.</summary>
    IOptimizer Optimizer { get; }

    /// <summary>The learning rate the optimizer had when the scheduler was made: the rate a schedule works from.</summary>
    float BaseLearningRate { get; }

    /// <summary>Completed epochs (or steps).</summary>
    int Epoch { get; }

    /// <summary>Advances one epoch and sets the optimizer's learning rate for the next.</summary>
    void Step();
}

/// <summary>
/// A schedule given as the rate of each (0-based) epoch from the base rate: the base of the library's schedulers. The
/// constructor sets the first epoch's rate from the function it is given, so a derived class has nothing to call once
/// its own fields are set.
/// </summary>
public abstract class LearningRateScheduler : ILearningRateScheduler
{
    private readonly Func<int, float, float> _rate;

    /// <summary>
    /// Attaches to <paramref name="optimizer"/>: its current learning rate becomes the base rate, and the rate of epoch 0
    /// is set at once.
    /// </summary>
    /// <param name="optimizer">The optimizer to schedule.</param>
    /// <param name="rate">The learning rate of a (0-based) epoch, from the epoch and the base rate.</param>
    protected LearningRateScheduler(IOptimizer optimizer, Func<int, float, float> rate)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(rate);
        Optimizer = optimizer;
        BaseLearningRate = optimizer.LearningRate;
        _rate = rate;
        optimizer.LearningRate = rate(0, BaseLearningRate);
    }

    /// <inheritdoc />
    public IOptimizer Optimizer { get; }

    /// <inheritdoc />
    public float BaseLearningRate { get; }

    /// <inheritdoc />
    public int Epoch { get; private set; }

    /// <inheritdoc />
    public void Step()
    {
        Epoch++;
        Optimizer.LearningRate = _rate(Epoch, BaseLearningRate);
    }
}

/// <summary>Multiplies the rate by gamma every stepSize epochs.</summary>
/// <param name="optimizer">The optimizer to schedule.</param>
/// <param name="stepSize">Epochs between two multiplications.</param>
/// <param name="gamma">The factor.</param>
public sealed class StepDecay(IOptimizer optimizer, int stepSize, float gamma = 0.1f) : LearningRateScheduler(optimizer, Rate(stepSize, gamma))
{
    private static Func<int, float, float> Rate(int stepSize, float gamma)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stepSize);
        return (epoch, rate) => rate * MathF.Pow(gamma, epoch / stepSize);
    }
}

/// <summary>Multiplies the rate by gamma every epoch.</summary>
/// <param name="optimizer">The optimizer to schedule.</param>
/// <param name="gamma">The factor.</param>
public sealed class ExponentialDecay(IOptimizer optimizer, float gamma) : LearningRateScheduler(optimizer, (epoch, rate) => rate * MathF.Pow(gamma, epoch));

/// <summary>
/// Optional linear warm-up from near zero, then cosine annealing from the base rate down to
/// minLearningRate at totalEpochs. A strong default for most training runs.
/// </summary>
/// <param name="optimizer">The optimizer to schedule.</param>
/// <param name="totalEpochs">The epoch the rate reaches the minimum at.</param>
/// <param name="minLearningRate">The rate at the end.</param>
/// <param name="warmupEpochs">Epochs of linear warm-up first.</param>
public sealed class CosineAnnealing(IOptimizer optimizer, int totalEpochs, float minLearningRate = 0f, int warmupEpochs = 0)
    : LearningRateScheduler(optimizer, Rate(totalEpochs, minLearningRate, warmupEpochs))
{
    private static Func<int, float, float> Rate(int totalEpochs, float minLearningRate, int warmupEpochs)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalEpochs);
        ArgumentOutOfRangeException.ThrowIfNegative(warmupEpochs);
        return (epoch, rate) =>
        {
            if (epoch < warmupEpochs)
            {
                return rate * (epoch + 1) / (warmupEpochs + 1);
            }

            double progress = Math.Min(1.0, (epoch - warmupEpochs) / (double)Math.Max(1, totalEpochs - warmupEpochs));
            return (float)(minLearningRate + (rate - minLearningRate) * 0.5 * (1 + Math.Cos(Math.PI * progress)));
        };
    }
}

/// <summary>Any schedule as a function of (epoch, base rate) → rate.</summary>
/// <param name="optimizer">The optimizer to schedule.</param>
/// <param name="schedule">The rate of a (0-based) epoch from the epoch and the base rate.</param>
public sealed class LambdaSchedule(IOptimizer optimizer, Func<int, float, float> schedule) : LearningRateScheduler(optimizer, schedule);
