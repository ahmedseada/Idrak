// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Training;

/// <summary>
/// Changes an optimizer's learning rate over epochs. Call <see cref="Step"/> once per epoch (the
/// <c>Trainer</c> does this when given a scheduler).
/// </summary>
public abstract class LearningRateScheduler
{
    /// <summary>Attaches to <paramref name="optimizer"/>; its current learning rate becomes the base rate.</summary>
    protected LearningRateScheduler(Optimizer optimizer)
    {
        Optimizer = optimizer;
        BaseLearningRate = optimizer.LearningRate;
    }

    /// <summary>The optimizer being scheduled.</summary>
    public Optimizer Optimizer { get; }

    /// <summary>The learning rate the optimizer had when the scheduler was created.</summary>
    public float BaseLearningRate { get; }

    /// <summary>Completed epochs.</summary>
    public int Epoch { get; private set; }

    /// <summary>Sets the rate for the first epoch; call from derived constructors once their fields are set.</summary>
    protected void Initialize() => Optimizer.LearningRate = Compute(0);

    /// <summary>Advances one epoch and updates the optimizer's learning rate.</summary>
    public void Step()
    {
        Epoch++;
        Optimizer.LearningRate = Compute(Epoch);
    }

    /// <summary>The learning rate for the (0-based) epoch about to start.</summary>
    protected abstract float Compute(int epoch);
}

/// <summary>Multiplies the rate by gamma every stepSize epochs.</summary>
public sealed class StepDecay : LearningRateScheduler
{
    private readonly int _stepSize;
    private readonly float _gamma;

    /// <summary>Creates the schedule.</summary>
    public StepDecay(Optimizer optimizer, int stepSize, float gamma = 0.1f) : base(optimizer)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stepSize);
        _stepSize = stepSize;
        _gamma = gamma;
        Initialize();
    }

    /// <inheritdoc />
    protected override float Compute(int epoch) => BaseLearningRate * MathF.Pow(_gamma, epoch / _stepSize);
}

/// <summary>Multiplies the rate by gamma every epoch.</summary>
public sealed class ExponentialDecay : LearningRateScheduler
{
    private readonly float _gamma;

    /// <summary>Creates the schedule.</summary>
    public ExponentialDecay(Optimizer optimizer, float gamma) : base(optimizer)
    {
        _gamma = gamma;
        Initialize();
    }

    /// <inheritdoc />
    protected override float Compute(int epoch) => BaseLearningRate * MathF.Pow(_gamma, epoch);
}

/// <summary>
/// Optional linear warm-up from near zero, then cosine annealing from the base rate down to
/// minLearningRate at totalEpochs. A strong default for most training runs.
/// </summary>
public sealed class CosineAnnealing : LearningRateScheduler
{
    private readonly int _totalEpochs;
    private readonly int _warmupEpochs;
    private readonly float _minLearningRate;

    /// <summary>Creates the schedule.</summary>
    public CosineAnnealing(Optimizer optimizer, int totalEpochs, float minLearningRate = 0f, int warmupEpochs = 0) : base(optimizer)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalEpochs);
        ArgumentOutOfRangeException.ThrowIfNegative(warmupEpochs);
        _totalEpochs = totalEpochs;
        _warmupEpochs = warmupEpochs;
        _minLearningRate = minLearningRate;
        Initialize();
    }

    /// <inheritdoc />
    protected override float Compute(int epoch)
    {
        if (epoch < _warmupEpochs)
        {
            return BaseLearningRate * (epoch + 1) / (_warmupEpochs + 1);
        }

        double progress = Math.Min(1.0, (epoch - _warmupEpochs) / (double)Math.Max(1, _totalEpochs - _warmupEpochs));
        return (float)(_minLearningRate + (BaseLearningRate - _minLearningRate) * 0.5 * (1 + Math.Cos(Math.PI * progress)));
    }
}

/// <summary>Any schedule as a function of (epoch, base rate) → rate.</summary>
public sealed class LambdaSchedule : LearningRateScheduler
{
    private readonly Func<int, float, float> _schedule;

    /// <summary>Creates the schedule.</summary>
    public LambdaSchedule(Optimizer optimizer, Func<int, float, float> schedule) : base(optimizer)
    {
        _schedule = schedule;
        Initialize();
    }

    /// <inheritdoc />
    protected override float Compute(int epoch) => _schedule(epoch, BaseLearningRate);
}
