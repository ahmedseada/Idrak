// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Classic;

/// <summary>
/// The settings of a <see cref="PendulumEnvironment"/>, by default Gymnasium's Pendulum-v1; as the kind's options
/// (<see cref="FromOptions"/>): <c>max-steps</c>, <c>g</c>, <c>reset-angle</c>, <c>reset-speed</c>.
/// </summary>
public sealed record PendulumOptions
{
    /// <summary>The step that reaches this many steps is truncated (Gymnasium's TimeLimit of Pendulum-v1: 200); 0 for no limit. Option <c>max-steps</c>.</summary>
    public int MaxSteps { get; init; } = 200;

    /// <summary>The acceleration of gravity (Gymnasium's g, 10). Option <c>g</c>.</summary>
    public double Gravity { get; init; } = 10.0;

    /// <summary>The start angle is drawn uniformly from [−this, this) (Gymnasium's π, its x_init). Option <c>reset-angle</c>.</summary>
    public double ResetAngle { get; init; } = Math.PI;

    /// <summary>The start angular velocity is drawn uniformly from [−this, this) (Gymnasium's 1, its y_init). Option <c>reset-speed</c>.</summary>
    public double ResetSpeed { get; init; } = 1.0;

    private static readonly OptionSpec<int> MaxStepsOption = OptionSpec.Whole("max-steps", 200, min: 0, summary: "the step that reaches it is truncated; 0 for no limit");
    private static readonly OptionSpec<double> G = OptionSpec.Real("g", 10.0, summary: "the gravity");
    private static readonly OptionSpec<double> Angle = OptionSpec.Real("reset-angle", Math.PI, min: 0, summary: "the start angle's bound");
    private static readonly OptionSpec<double> Speed = OptionSpec.Real("reset-speed", 1.0, min: 0, summary: "the start speed's bound");

    /// <summary>The kind's options: their keys, types, ranges and defaults (a value of blanks is the default).</summary>
    public static OptionSchema Schema { get; } = new($"{PendulumEnvironment.KindName} environment", blankIsDefault: true, MaxStepsOption, G, Angle, Speed);

    /// <summary>The kind's option keys.</summary>
    public static OptionKeys Keys => Schema.Keys;

    /// <summary>The settings the kind's <paramref name="options"/> give (the defaults for those not given).</summary>
    /// <exception cref="ArgumentException">An option not in <see cref="Keys"/>, or a bad value.</exception>
    public static PendulumOptions FromOptions(IReadOnlyDictionary<string, string>? options)
    {
        var read = Schema.Parse(options);
        return new PendulumOptions
        {
            MaxSteps = read.Get(MaxStepsOption),
            Gravity = read.Get(G),
            ResetAngle = read.Get(Angle),
            ResetSpeed = read.Get(Speed),
        };
    }
}

/// <summary>
/// The inverted pendulum swing-up as Gymnasium's Pendulum-v1 has it: a pendulum of mass 1 and length 1 under gravity g,
/// driven by a torque in [−2, 2] (the action, a box of one float); the reward is −(θ² + 0.1 θ̇² + 0.001 u²) with θ normalized
/// to [−π, π), so 0 is upright and still; the angular velocity is clipped to ±8, a step is 0.05 s, and the 200-step
/// truncation is the only end. The equations, constants and clips are Gymnasium's (gymnasium/envs/classic_control/pendulum.py),
/// matched step by step against Gymnasium from the same start states and actions (tests/Idrak.Tests/data/reinforcement).
/// <list type="bullet">
/// <item><b>Precision:</b> θ and θ̇ are kept in double precision (Gymnasium's float64); the torque u is the action's float32,
/// clipped in float32, and the two terms Gymnasium computes from it in float32 (numpy 2 keeps a Python float times a float32
/// in float32: 3u / (m l²) and 0.001 u²) are rounded to float32 the same way, then added in double; observations
/// [cos θ, sin θ, θ̇] are rounded once to float32.</item>
/// <item><b>Actions out of the box</b> are clipped to [−2, 2], as Gymnasium clips them; a NaN or infinite torque, or an action
/// of another length than 1, is refused.</item>
/// <item><b>Start:</b> θ uniform in [−π, π), θ̇ uniform in [−1, 1) (Gymnasium's distribution) from the episode's seed through
/// <see cref="Random"/>; <see cref="ResetTo"/> starts from a given state.</item>
/// </list>
/// One environment runs one episode at a time; it allocates only each observation it returns.
/// </summary>
public sealed class PendulumEnvironment : IEnvironment<float[], float[]>
{
    /// <summary>The kind's name in <see cref="EnvironmentKinds"/>: "pendulum".</summary>
    public const string KindName = "pendulum";

    private const double MaxSpeed = 8, Dt = 0.05, Mass = 1.0, Length = 1.0;
    private const float MaxTorque = 2.0f;

    private readonly float _torqueGain, _costGain;
    private double _theta, _thetaDot;
    private EpisodeClock _clock;

    /// <summary>A pendulum of <paramref name="options"/> (Gymnasium's Pendulum-v1 when null).</summary>
    public PendulumEnvironment(PendulumOptions? options = null)
    {
        Options = options ?? new PendulumOptions();
        _torqueGain = (float)(3.0 / (Mass * (Length * Length)));                  // a Python float times the float32 torque: float32
        _costGain = (float)0.001;
    }

    /// <summary>The settings it was made with.</summary>
    public PendulumOptions Options { get; }

    /// <inheritdoc />
    public Space ObservationSpace { get; } = Space.Box([3], [-1f, -1f, -(float)MaxSpeed], [1f, 1f, (float)MaxSpeed]);

    /// <inheritdoc />
    public Space ActionSpace { get; } = Space.Box([1], -MaxTorque, MaxTorque);

    /// <summary>The state in double precision (θ, θ̇), a copy.</summary>
    public double[] State => [_theta, _thetaDot];

    /// <summary>The steps of the episode under way.</summary>
    public int Steps => _clock.Steps;

    /// <inheritdoc />
    public float[] Reset(int seed)
    {
        var random = new Random(seed);
        _theta = -Options.ResetAngle + 2 * Options.ResetAngle * random.NextDouble();
        _thetaDot = -Options.ResetSpeed + 2 * Options.ResetSpeed * random.NextDouble();
        _clock.Start();
        return Observe();
    }

    /// <summary>Starts an episode from <paramref name="state"/> (θ, θ̇) and returns its observation: how a test starts from the state a reference started from.</summary>
    /// <exception cref="ArgumentException">Not 2 finite values.</exception>
    public float[] ResetTo(ReadOnlySpan<double> state)
    {
        if (state.Length != 2 || !CartPoleEnvironment.AllFinite(state))
        {
            throw new ArgumentException($"A {KindName} state is 2 finite values (angle, angular velocity), not {state.Length}.", nameof(state));
        }

        (_theta, _thetaDot) = (state[0], state[1]);
        _clock.Start();
        return Observe();
    }

    /// <inheritdoc />
    public ValueTask<EnvironmentStep<float[]>> StepAsync(float[] action, CancellationToken cancellationToken = default) => ValueTask.FromResult(Step(action, cancellationToken));

    /// <summary>The step <see cref="StepAsync"/> takes, synchronously.</summary>
    /// <exception cref="InvalidOperationException">No episode is under way.</exception>
    /// <exception cref="ArgumentException">An action that is not one finite torque.</exception>
    public EnvironmentStep<float[]> Step(ReadOnlySpan<float> action, CancellationToken cancellationToken = default)
    {
        if (action.Length != 1 || !float.IsFinite(action[0]))
        {
            throw new ArgumentException($"A {KindName} action is one finite torque (clipped to ±{MaxTorque}), not {action.Length} values.", nameof(action));
        }

        _clock.Tick(KindName, cancellationToken);

        // Gymnasium's step, in its order of operations and precisions.
        float u = Math.Clamp(action[0], -MaxTorque, MaxTorque);
        double normalized = AngleNormalize(_theta);
        double costs = normalized * normalized + 0.1 * (_thetaDot * _thetaDot) + (double)(_costGain * (u * u));
        double newThetaDot = _thetaDot + (3 * Options.Gravity / (2 * Length) * Math.Sin(_theta) + (double)(_torqueGain * u)) * Dt;
        newThetaDot = Math.Clamp(newThetaDot, -MaxSpeed, MaxSpeed);
        _theta += newThetaDot * Dt;
        _thetaDot = newThetaDot;

        bool truncated = _clock.Truncates(Options.MaxSteps);
        _clock.End(false, truncated);
        return new EnvironmentStep<float[]>(Observe(), (float)-costs, false, truncated);
    }

    /// <summary>Ends the episode under way; more than once is harmless.</summary>
    public void Dispose() => _clock.Running = false;

    private float[] Observe() => [(float)Math.Cos(_theta), (float)Math.Sin(_theta), (float)_thetaDot];

    // Gymnasium's angle_normalize: ((x + π) mod 2π) − π with Python's floor modulo (numpy's remainder: the sign of the divisor).
    private static double AngleNormalize(double x)
    {
        double a = x + Math.PI, b = 2 * Math.PI;
        double mod = a % b;
        if (mod != 0)
        {
            if (b < 0 != mod < 0)
            {
                mod += b;
            }
        }
        else
        {
            mod = Math.CopySign(0.0, b);
        }

        return mod - Math.PI;
    }
}
