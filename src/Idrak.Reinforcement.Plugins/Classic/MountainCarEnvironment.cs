// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Classic;

/// <summary>
/// The settings of a <see cref="MountainCarEnvironment"/>, by default Gymnasium's MountainCar-v0; as the kind's options
/// (<see cref="FromOptions"/>): <c>max-steps</c>, <c>goal-velocity</c>, <c>reset-low</c>, <c>reset-high</c>.
/// </summary>
public sealed record MountainCarOptions
{
    /// <summary>The step that reaches this many steps is truncated (Gymnasium's TimeLimit of MountainCar-v0: 200); 0 for no limit. Option <c>max-steps</c>.</summary>
    public int MaxSteps { get; init; } = 200;

    /// <summary>The least velocity at the goal that ends the episode (Gymnasium's goal_velocity, 0). Option <c>goal-velocity</c>.</summary>
    public double GoalVelocity { get; init; }

    /// <summary>The low bound of the start position's uniform draw (Gymnasium's −0.6). Option <c>reset-low</c>.</summary>
    public double ResetLow { get; init; } = -0.6;

    /// <summary>The high bound of the start position's uniform draw (Gymnasium's −0.4). Option <c>reset-high</c>.</summary>
    public double ResetHigh { get; init; } = -0.4;

    private static readonly OptionSpec<int> MaxStepsOption = OptionSpec.Whole("max-steps", 200, min: 0, summary: "the step that reaches it is truncated; 0 for no limit");
    private static readonly OptionSpec<double> Goal = OptionSpec.Real("goal-velocity", 0.0, summary: "the least velocity at the flag");
    private static readonly OptionSpec<double> Low = OptionSpec.Real("reset-low", -0.6, MountainCarEnvironment.MinPosition, MountainCarEnvironment.MaxPosition);
    private static readonly OptionSpec<double> High = OptionSpec.Real("reset-high", -0.4, MountainCarEnvironment.MinPosition, MountainCarEnvironment.MaxPosition);

    /// <summary>The kind's options: their keys, types, ranges and defaults (a value of blanks is the default).</summary>
    public static OptionSchema Schema { get; } = new($"{MountainCarEnvironment.KindName} environment", blankIsDefault: true, MaxStepsOption, Goal, Low, High);

    /// <summary>The kind's option keys.</summary>
    public static OptionKeys Keys => Schema.Keys;

    /// <summary>The settings the kind's <paramref name="options"/> give (the defaults for those not given).</summary>
    /// <exception cref="ArgumentException">An option not in <see cref="Keys"/>, a bad value, or a low bound above the high one.</exception>
    public static MountainCarOptions FromOptions(IReadOnlyDictionary<string, string>? options)
    {
        var read = Schema.Parse(options);
        var settings = new MountainCarOptions
        {
            MaxSteps = read.Get(MaxStepsOption),
            GoalVelocity = read.Get(Goal),
            ResetLow = read.Get(Low),
            ResetHigh = read.Get(High),
        };
        return settings.ResetLow <= settings.ResetHigh
            ? settings
            : throw new ArgumentException($"The {MountainCarEnvironment.KindName} environment's reset-low ({settings.ResetLow}) is above its reset-high ({settings.ResetHigh}).", nameof(options));
    }
}

/// <summary>
/// Mountain car (Moore 1990) as Gymnasium's MountainCar-v0 has it: an under-powered car in a valley must rock back and
/// forth to reach the flag at position 0.5; it pushes left (action 0), not at all (1) or right (2), earns −1 a step, and
/// the episode terminates at the goal (position ≥ 0.5 with velocity ≥ the goal velocity) or is truncated after 200 steps.
/// The dynamics (force 0.001, gravity 0.0025 along cos 3x), the clips (velocity ±0.07, position [−1.2, 0.6], the velocity
/// zeroed against the left wall) and the ends are Gymnasium's (gymnasium/envs/classic_control/mountain_car.py), matched step
/// by step against Gymnasium from the same start states and actions (tests/Idrak.Tests/data/reinforcement).
/// <list type="bullet">
/// <item><b>Precision:</b> position and velocity in double precision (Gymnasium's float64), in its order of operations;
/// observations rounded once to float32.</item>
/// <item><b>Start:</b> the position uniform in [−0.6, −0.4), the velocity 0 (Gymnasium's distribution), from the episode's
/// seed through <see cref="Random"/>; <see cref="ResetTo"/> starts from a given state.</item>
/// </list>
/// One environment runs one episode at a time; it allocates only each observation it returns.
/// </summary>
public sealed class MountainCarEnvironment : IEnvironment<float[], int>
{
    /// <summary>The kind's name in <see cref="EnvironmentKinds"/>: "mountain-car".</summary>
    public const string KindName = "mountain-car";

    internal const double MinPosition = -1.2, MaxPosition = 0.6;
    private const double MaxSpeed = 0.07, GoalPosition = 0.5, Force = 0.001, Gravity = 0.0025;

    private double _position, _velocity;
    private EpisodeClock _clock;

    /// <summary>A mountain car of <paramref name="options"/> (Gymnasium's MountainCar-v0 when null).</summary>
    public MountainCarEnvironment(MountainCarOptions? options = null)
    {
        Options = options ?? new MountainCarOptions();
    }

    /// <summary>The settings it was made with.</summary>
    public MountainCarOptions Options { get; }

    /// <inheritdoc />
    public Space ObservationSpace { get; } = Space.Box([2], [(float)MinPosition, -(float)MaxSpeed], [(float)MaxPosition, (float)MaxSpeed]);

    /// <inheritdoc />
    public Space ActionSpace { get; } = Space.Discrete(3);

    /// <summary>The state in double precision (position, velocity), a copy.</summary>
    public double[] State => [_position, _velocity];

    /// <summary>The steps of the episode under way.</summary>
    public int Steps => _clock.Steps;

    /// <inheritdoc />
    public float[] Reset(int seed)
    {
        var random = new Random(seed);
        _position = Options.ResetLow + (Options.ResetHigh - Options.ResetLow) * random.NextDouble();
        _velocity = 0;
        _clock.Start();
        return Observe();
    }

    /// <summary>Starts an episode from <paramref name="state"/> (position, velocity) and returns its observation: how a test starts from the state a reference started from.</summary>
    /// <exception cref="ArgumentException">Not 2 finite values.</exception>
    public float[] ResetTo(ReadOnlySpan<double> state)
    {
        if (state.Length != 2 || !CartPoleEnvironment.AllFinite(state))
        {
            throw new ArgumentException($"A {KindName} state is 2 finite values (position, velocity), not {state.Length}.", nameof(state));
        }

        (_position, _velocity) = (state[0], state[1]);
        _clock.Start();
        return Observe();
    }

    /// <inheritdoc />
    public ValueTask<EnvironmentStep<float[]>> StepAsync(int action, CancellationToken cancellationToken = default) => ValueTask.FromResult(Step(action, cancellationToken));

    /// <summary>The step <see cref="StepAsync"/> takes, synchronously.</summary>
    /// <exception cref="InvalidOperationException">No episode is under way.</exception>
    /// <exception cref="ArgumentException">An action other than 0, 1 or 2.</exception>
    public EnvironmentStep<float[]> Step(int action, CancellationToken cancellationToken = default)
    {
        if ((uint)action > 2)
        {
            throw new ArgumentException($"A {KindName} action is 0 (push left), 1 (no push) or 2 (push right), not {action}.", nameof(action));
        }

        _clock.Tick(KindName, cancellationToken);

        // Gymnasium's step, in its order of operations.
        _velocity += (action - 1) * Force + Math.Cos(3 * _position) * -Gravity;
        _velocity = Math.Clamp(_velocity, -MaxSpeed, MaxSpeed);
        _position += _velocity;
        _position = Math.Clamp(_position, MinPosition, MaxPosition);
        if (_position == MinPosition && _velocity < 0)
        {
            _velocity = 0;
        }

        bool terminated = _position >= GoalPosition && _velocity >= Options.GoalVelocity;
        bool truncated = _clock.Truncates(Options.MaxSteps);
        _clock.End(terminated, truncated);
        return new EnvironmentStep<float[]>(Observe(), -1f, terminated, truncated);
    }

    /// <summary>Ends the episode under way; more than once is harmless.</summary>
    public void Dispose() => _clock.Running = false;

    private float[] Observe() => [(float)_position, (float)_velocity];
}
