// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Classic;

/// <summary>
/// The settings of a <see cref="CartPoleEnvironment"/>, by default Gymnasium's CartPole-v1; as the kind's options
/// (<see cref="FromOptions"/>): <c>max-steps</c>, <c>sutton-barto</c>, <c>reset-low</c>, <c>reset-high</c>.
/// </summary>
public sealed record CartPoleOptions
{
    /// <summary>The step that reaches this many steps is truncated (Gymnasium's TimeLimit of CartPole-v1: 500); 0 for no limit. Option <c>max-steps</c>.</summary>
    public int MaxSteps { get; init; } = 500;

    /// <summary>Sutton and Barto's reward (0 a step, −1 on the step that fails) instead of 1 a step. Option <c>sutton-barto</c>.</summary>
    public bool SuttonBartoReward { get; init; }

    /// <summary>The low bound of each start element's uniform draw (Gymnasium's −0.05). Option <c>reset-low</c>.</summary>
    public double ResetLow { get; init; } = -0.05;

    /// <summary>The high bound of each start element's uniform draw (Gymnasium's 0.05). Option <c>reset-high</c>.</summary>
    public double ResetHigh { get; init; } = 0.05;

    private static readonly OptionSpec<int> MaxStepsOption = OptionSpec.Whole("max-steps", 500, min: 0, summary: "the step that reaches it is truncated; 0 for no limit");
    private static readonly OptionSpec<bool> SuttonBarto = OptionSpec.Flag("sutton-barto", summary: "0 a step, −1 on the step that fails");
    private static readonly OptionSpec<double> Low = OptionSpec.Real("reset-low", -0.05, summary: "the low bound of each start element's draw");
    private static readonly OptionSpec<double> High = OptionSpec.Real("reset-high", 0.05, summary: "the high bound of each start element's draw");

    /// <summary>The kind's options: their keys, types, ranges and defaults (a value of blanks is the default).</summary>
    public static OptionSchema Schema { get; } = new($"{CartPoleEnvironment.KindName} environment", blankIsDefault: true, MaxStepsOption, SuttonBarto, Low, High);

    /// <summary>The kind's option keys.</summary>
    public static OptionKeys Keys => Schema.Keys;

    /// <summary>The settings the kind's <paramref name="options"/> give (the defaults for those not given).</summary>
    /// <exception cref="ArgumentException">An option not in <see cref="Keys"/>, a bad value, or a low bound above the high one.</exception>
    public static CartPoleOptions FromOptions(IReadOnlyDictionary<string, string>? options)
    {
        var read = Schema.Parse(options);
        var settings = new CartPoleOptions
        {
            MaxSteps = read.Get(MaxStepsOption),
            SuttonBartoReward = read.Get(SuttonBarto),
            ResetLow = read.Get(Low),
            ResetHigh = read.Get(High),
        };
        return settings.ResetLow <= settings.ResetHigh
            ? settings
            : throw new ArgumentException($"The {CartPoleEnvironment.KindName} environment's reset-low ({settings.ResetLow}) is above its reset-high ({settings.ResetHigh}).", nameof(options));
    }
}

/// <summary>
/// CartPole (Barto, Sutton and Anderson 1983) as Gymnasium's CartPole-v1 has it: a pole on a cart that moves along a
/// track; the agent pushes the cart left (action 0) or right (1) with a force of 10, and earns 1 a step while the pole is
/// within 12° of upright and the cart within 2.4 of the centre. The equations, constants, Euler integration (τ = 0.02),
/// termination thresholds, rewards and the 500-step truncation are Gymnasium's (gymnasium/envs/classic_control/cartpole.py),
/// matched step by step against Gymnasium from the same start states and actions (tests/Idrak.Tests/data/reinforcement).
/// <list type="bullet">
/// <item><b>Precision:</b> the state is kept and integrated in double precision, in Gymnasium's order of operations (it
/// computes in float64); observations are the state rounded once to float32, as Gymnasium returns them.</item>
/// <item><b>Start:</b> each element drawn uniformly from [−0.05, 0.05) (Gymnasium's distribution) by the episode's seed,
/// through <see cref="Random"/> (Gymnasium's numpy generator is not imitated); <see cref="ResetTo"/> starts from a given
/// state.</item>
/// <item><b>Spaces:</b> observations a box of 4 (x, ẋ, θ, θ̇) bounded by ±4.8, ±∞, ±0.4189 (twice the thresholds), actions
/// discrete(2).</item>
/// </list>
/// One environment runs one episode at a time; it allocates only each observation it returns.
/// </summary>
public sealed class CartPoleEnvironment : IEnvironment<float[], int>
{
    /// <summary>The kind's name in <see cref="EnvironmentKinds"/>: "cartpole".</summary>
    public const string KindName = "cartpole";

    private const double Gravity = 9.8, MassCart = 1.0, MassPole = 0.1, Length = 0.5, ForceMagnitude = 10.0, Tau = 0.02, XThreshold = 2.4;
    private const double TotalMass = MassPole + MassCart, PoleMassLength = MassPole * Length;
    private static readonly double ThetaThreshold = 12 * 2 * Math.PI / 360;

    private readonly double[] _state = new double[4];
    private EpisodeClock _clock;

    /// <summary>A CartPole of <paramref name="options"/> (Gymnasium's CartPole-v1 when null).</summary>
    public CartPoleEnvironment(CartPoleOptions? options = null)
    {
        Options = options ?? new CartPoleOptions();
        ObservationSpace = Space.Box([4], [-(float)(XThreshold * 2), float.NegativeInfinity, -(float)(ThetaThreshold * 2), float.NegativeInfinity],
            [(float)(XThreshold * 2), float.PositiveInfinity, (float)(ThetaThreshold * 2), float.PositiveInfinity]);
    }

    /// <summary>The settings it was made with.</summary>
    public CartPoleOptions Options { get; }

    /// <inheritdoc />
    public Space ObservationSpace { get; }

    /// <inheritdoc />
    public Space ActionSpace { get; } = Space.Discrete(2);

    /// <summary>The state in double precision (x, ẋ, θ, θ̇), a copy.</summary>
    public double[] State => [.. _state];

    /// <summary>The steps of the episode under way.</summary>
    public int Steps => _clock.Steps;

    /// <inheritdoc />
    public float[] Reset(int seed)
    {
        var random = new Random(seed);
        for (int i = 0; i < 4; i++)
        {
            _state[i] = Options.ResetLow + (Options.ResetHigh - Options.ResetLow) * random.NextDouble();
        }

        _clock.Start();
        return Observe();
    }

    /// <summary>Starts an episode from <paramref name="state"/> (x, ẋ, θ, θ̇) and returns its observation: how a test starts from the state a reference started from.</summary>
    /// <exception cref="ArgumentException">Not 4 finite values.</exception>
    public float[] ResetTo(ReadOnlySpan<double> state)
    {
        if (state.Length != 4 || !AllFinite(state))
        {
            throw new ArgumentException($"A {KindName} state is 4 finite values (x, x velocity, angle, angular velocity), not {state.Length}.", nameof(state));
        }

        state.CopyTo(_state);
        _clock.Start();
        return Observe();
    }

    /// <inheritdoc />
    public ValueTask<EnvironmentStep<float[]>> StepAsync(int action, CancellationToken cancellationToken = default) => ValueTask.FromResult(Step(action, cancellationToken));

    /// <summary>The step <see cref="StepAsync"/> takes, synchronously.</summary>
    /// <exception cref="InvalidOperationException">No episode is under way.</exception>
    /// <exception cref="ArgumentException">An action other than 0 or 1.</exception>
    public EnvironmentStep<float[]> Step(int action, CancellationToken cancellationToken = default)
    {
        if ((uint)action > 1)
        {
            throw new ArgumentException($"A {KindName} action is 0 (push left) or 1 (push right), not {action}.", nameof(action));
        }

        _clock.Tick(KindName, cancellationToken);

        // Gymnasium's step, in its order of operations.
        double x = _state[0], xDot = _state[1], theta = _state[2], thetaDot = _state[3];
        double force = action == 1 ? ForceMagnitude : -ForceMagnitude;
        double cos = Math.Cos(theta), sin = Math.Sin(theta);
        double temp = (force + PoleMassLength * (thetaDot * thetaDot) * sin) / TotalMass;
        double thetaAcc = (Gravity * sin - cos * temp) / (Length * (4.0 / 3.0 - MassPole * (cos * cos) / TotalMass));
        double xAcc = temp - PoleMassLength * thetaAcc * cos / TotalMass;
        x += Tau * xDot;
        xDot += Tau * xAcc;
        theta += Tau * thetaDot;
        thetaDot += Tau * thetaAcc;
        _state[0] = x;
        _state[1] = xDot;
        _state[2] = theta;
        _state[3] = thetaDot;

        bool terminated = x < -XThreshold || x > XThreshold || theta < -ThetaThreshold || theta > ThetaThreshold;
        float reward = Options.SuttonBartoReward ? (terminated ? -1f : 0f) : 1f;
        bool truncated = _clock.Truncates(Options.MaxSteps);
        _clock.End(terminated, truncated);
        return new EnvironmentStep<float[]>(Observe(), reward, terminated, truncated);
    }

    /// <summary>Ends the episode under way; more than once is harmless.</summary>
    public void Dispose() => _clock.Running = false;

    private float[] Observe() => [(float)_state[0], (float)_state[1], (float)_state[2], (float)_state[3]];

    internal static bool AllFinite(ReadOnlySpan<double> values)
    {
        foreach (double value in values)
        {
            if (!double.IsFinite(value))
            {
                return false;
            }
        }

        return true;
    }
}
