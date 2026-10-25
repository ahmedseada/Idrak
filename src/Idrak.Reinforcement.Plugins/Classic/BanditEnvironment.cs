// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Classic;

/// <summary>
/// The settings of a <see cref="BanditEnvironment"/>; as the kind's options (<see cref="FromOptions"/>): <c>arms</c>,
/// <c>means</c>, <c>spreads</c>, <c>means-seed</c>, <c>pulls</c>. By default Sutton and Barto's 10-armed testbed (2018,
/// section 2.3): 10 arms whose means are drawn once from a standard normal (by <see cref="MeansSeed"/>), each pull paying its
/// arm's mean plus standard normal noise.
/// </summary>
public sealed record BanditOptions
{
    /// <summary>The arms (actions). Option <c>arms</c>; implied by <see cref="Means"/> when those are given.</summary>
    public int Arms { get; init; } = 10;

    /// <summary>Each arm's mean reward; null to draw them from a standard normal by <see cref="MeansSeed"/>. Option <c>means</c> (numbers separated by commas).</summary>
    public IReadOnlyList<double>? Means { get; init; }

    /// <summary>
    /// The standard deviation of each arm's reward around its mean: one value for every arm, or one per arm (0: the mean
    /// exactly). Option <c>spreads</c> (one number, or one per arm separated by commas).
    /// </summary>
    public IReadOnlyList<double> Spreads { get; init; } = [1.0];

    /// <summary>The seed the means are drawn from when <see cref="Means"/> is null: every environment of the kind has the same arms. Option <c>means-seed</c>.</summary>
    public int MeansSeed { get; init; }

    /// <summary>The pulls of an episode (each a step; the last one terminates it). Option <c>pulls</c>.</summary>
    public int Pulls { get; init; } = 1;

    /// <summary>The kind's option keys.</summary>
    public static OptionKeys Keys { get; } = new("arms", "means", "spreads", "means-seed", "pulls");

    /// <summary>The settings the kind's <paramref name="options"/> give (the defaults for those not given).</summary>
    /// <exception cref="ArgumentException">An option not in <see cref="Keys"/>, a bad value, or means and spreads of other counts than the arms.</exception>
    public static BanditOptions FromOptions(IReadOnlyDictionary<string, string>? options)
    {
        var read = new ClassicOptions(BanditEnvironment.KindName, Keys, options);
        var means = read.Numbers("means");
        var settings = new BanditOptions
        {
            Arms = read.Int("arms", means?.Length ?? 10, min: 1),
            Means = means,
            Spreads = read.Numbers("spreads", min: 0) ?? [1.0],
            MeansSeed = read.Int("means-seed", 0),
            Pulls = read.Int("pulls", 1, min: 1),
        };
        settings.Check();
        return settings;
    }

    // The arms' means and spreads, checked: as many as the arms (one spread may stand for all).
    internal (double[] Means, double[] Spreads) Resolve()
    {
        Check();
        double[] means;
        if (Means is { } given)
        {
            means = [.. given];
        }
        else
        {
            var random = new Random(MeansSeed);
            means = new double[Arms];
            for (int a = 0; a < Arms; a++)
            {
                means[a] = random.NextGaussian();
            }
        }

        double[] spreads = Spreads.Count == 1 ? Enumerable.Repeat(Spreads[0], Arms).ToArray() : [.. Spreads];
        return (means, spreads);
    }

    private void Check()
    {
        string kind = BanditEnvironment.KindName;
        if (Arms < 1 || Pulls < 1)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"A {kind} has at least one arm and one pull an episode, not {Arms} and {Pulls}."));
        }

        if (Means is { } means && (means.Count != Arms || means.Any(m => !double.IsFinite(m))))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The {kind}'s means are {means.Count} finite numbers for its {Arms} arms (option 'means')."));
        }

        if ((Spreads.Count != 1 && Spreads.Count != Arms) || Spreads.Any(s => !(s >= 0) || !double.IsFinite(s)))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The {kind}'s spreads are one number of at least 0, or one per arm ({Arms}), not {Spreads.Count} (option 'spreads')."));
        }
    }
}

/// <summary>
/// A multi-armed bandit (Sutton and Barto 2018, chapter 2): each step the agent pulls one of the arms (a discrete action)
/// and is paid that arm's mean plus Gaussian noise of the arm's spread; an episode is <see cref="BanditOptions.Pulls"/>
/// pulls, the last one terminating it. There is nothing to observe: the observation is a box of one 0. Gymnasium has no
/// bandit, so there is no reference trajectory: the rewards' means and spreads are checked by their statistics.
/// <list type="bullet">
/// <item><b>Draws:</b> the noise of an episode comes from its seed through <see cref="Random"/> and
/// <see cref="RandomDraws.NextGaussian"/> (one draw a pull, in order), so the same seed and pulls replay exactly; the means
/// are drawn once per environment from <see cref="BanditOptions.MeansSeed"/>, so every environment of a kind has the same
/// arms.</item>
/// <item><b>Precision:</b> rewards are computed in double precision and rounded once to float32.</item>
/// </list>
/// </summary>
public sealed class BanditEnvironment : IEnvironment<float[], int>
{
    /// <summary>The kind's name in <see cref="EnvironmentKinds"/>: "bandit".</summary>
    public const string KindName = "bandit";

    private readonly double[] _means, _spreads;
    private Random _random = new(0);
    private EpisodeClock _clock;

    /// <summary>A bandit of <paramref name="options"/> (the 10-armed testbed when null).</summary>
    /// <exception cref="ArgumentException">Means or spreads of other counts than the arms.</exception>
    public BanditEnvironment(BanditOptions? options = null)
    {
        Options = options ?? new BanditOptions();
        (_means, _spreads) = Options.Resolve();
        ActionSpace = Space.Discrete(_means.Length);
        BestArm = _means.AsSpan().IndexOf(_means.Max());
    }

    /// <summary>The settings it was made with.</summary>
    public BanditOptions Options { get; }

    /// <inheritdoc />
    public Space ObservationSpace { get; } = Space.Box([1], 0f, 0f);

    /// <inheritdoc />
    public Space ActionSpace { get; }

    /// <summary>Each arm's mean reward.</summary>
    public IReadOnlyList<double> Means => _means;

    /// <summary>Each arm's spread.</summary>
    public IReadOnlyList<double> Spreads => _spreads;

    /// <summary>The arm of the highest mean (the first of them on a tie).</summary>
    public int BestArm { get; }

    /// <summary>The pulls of the episode under way.</summary>
    public int Steps => _clock.Steps;

    /// <inheritdoc />
    public float[] Reset(int seed)
    {
        _random = new Random(seed);
        _clock.Start();
        return [0f];
    }

    /// <inheritdoc />
    public ValueTask<EnvironmentStep<float[]>> StepAsync(int action, CancellationToken cancellationToken = default) => ValueTask.FromResult(Step(action, cancellationToken));

    /// <summary>The step <see cref="StepAsync"/> takes, synchronously.</summary>
    /// <exception cref="InvalidOperationException">No episode is under way.</exception>
    /// <exception cref="ArgumentException">An arm that does not exist.</exception>
    public EnvironmentStep<float[]> Step(int action, CancellationToken cancellationToken = default)
    {
        if ((uint)action >= (uint)_means.Length)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"A {KindName} action is an arm from 0 to {_means.Length - 1}, not {action}."), nameof(action));
        }

        _clock.Tick(KindName, cancellationToken);
        double noise = _random.NextGaussian();                                    // one draw a pull, whichever arm
        float reward = (float)(_means[action] + _spreads[action] * noise);
        bool terminated = _clock.Steps >= Options.Pulls;
        _clock.End(terminated, false);
        return new EnvironmentStep<float[]>([0f], reward, terminated, false);
    }

    /// <summary>Ends the episode under way; more than once is harmless.</summary>
    public void Dispose() => _clock.Running = false;
}
