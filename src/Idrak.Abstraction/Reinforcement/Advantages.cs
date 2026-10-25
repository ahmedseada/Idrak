// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>
/// What an <see cref="IAdvantageEstimator"/> reads: the rewards of a batch's steps, episode after episode, how long each
/// episode is and how it ended, the value estimates when the policy has a critic, and which episodes form a group
/// (<see cref="RolloutBatch{TObservation, TAction}.Input"/> makes one; a test makes one of its own arrays). Checked when
/// made: every length agrees.
/// </summary>
public readonly ref struct AdvantageInput
{
    /// <summary>The input of <paramref name="rewards"/> divided into episodes of <paramref name="lengths"/> steps.</summary>
    /// <param name="rewards">The reward of each step [steps], episodes one after another.</param>
    /// <param name="values">The value estimate of each step's observation [steps]; empty when the policy has no critic.</param>
    /// <param name="lengths">The steps of each episode [episodes], each at least 1.</param>
    /// <param name="truncated">Whether each episode was cut by a limit [episodes] (bootstrapped from its final value); empty when none was.</param>
    /// <param name="finalValues">The value estimate of each episode's last observation [episodes], used for a truncated one; empty exactly when <paramref name="values"/> is.</param>
    /// <param name="groups">The group of each episode [episodes]; empty puts each episode in a group of its own.</param>
    /// <exception cref="ArgumentException">Lengths that do not add up to the rewards, an episode without steps, or a span of another length than it should have.</exception>
    public AdvantageInput(ReadOnlySpan<float> rewards, ReadOnlySpan<float> values, ReadOnlySpan<int> lengths, ReadOnlySpan<bool> truncated,
        ReadOnlySpan<float> finalValues, ReadOnlySpan<int> groups)
    {
        long steps = 0;
        foreach (int length in lengths)
        {
            if (length <= 0)
            {
                throw new ArgumentException("Every episode has at least one step.", nameof(lengths));
            }

            steps += length;
        }

        int episodes = lengths.Length;
        if (episodes == 0 || steps != rewards.Length)
        {
            throw new ArgumentException($"The episodes' lengths add up to {steps} steps in {episodes} episodes, for {rewards.Length} rewards (at least one episode).", nameof(lengths));
        }

        if (!values.IsEmpty && values.Length != rewards.Length)
        {
            throw new ArgumentException($"A value per step: {values.Length} values for {rewards.Length} steps.", nameof(values));
        }

        if (values.IsEmpty ? !finalValues.IsEmpty : finalValues.Length != episodes)
        {
            throw new ArgumentException($"A final value per episode with values, none without: {finalValues.Length} for {episodes} episodes, {values.Length} values.", nameof(finalValues));
        }

        if (!truncated.IsEmpty && truncated.Length != episodes)
        {
            throw new ArgumentException($"A truncation flag per episode: {truncated.Length} for {episodes} episodes.", nameof(truncated));
        }

        if (!groups.IsEmpty && groups.Length != episodes)
        {
            throw new ArgumentException($"A group per episode: {groups.Length} for {episodes} episodes.", nameof(groups));
        }

        Rewards = rewards;
        Values = values;
        Lengths = lengths;
        Truncated = truncated;
        FinalValues = finalValues;
        Groups = groups;
    }

    /// <summary>The reward of each step, episodes one after another.</summary>
    public ReadOnlySpan<float> Rewards { get; }

    /// <summary>The value estimate of each step's observation; empty without a critic.</summary>
    public ReadOnlySpan<float> Values { get; }

    /// <summary>The steps of each episode.</summary>
    public ReadOnlySpan<int> Lengths { get; }

    /// <summary>Whether each episode was cut by a limit; empty when none was.</summary>
    public ReadOnlySpan<bool> Truncated { get; }

    /// <summary>The value estimate of each episode's last observation; empty without a critic.</summary>
    public ReadOnlySpan<float> FinalValues { get; }

    /// <summary>The group of each episode; empty when each is a group of its own.</summary>
    public ReadOnlySpan<int> Groups { get; }

    /// <summary>The steps of every episode together.</summary>
    public int Steps => Rewards.Length;

    /// <summary>The episodes.</summary>
    public int Episodes => Lengths.Length;

    /// <summary>Whether the input has value estimates.</summary>
    public bool HasValues => !Values.IsEmpty;

    /// <summary>The group of episode <paramref name="episode"/> (the episode's own index when no groups are given).</summary>
    public int Group(int episode) => Groups.IsEmpty ? episode : Groups[episode];

    /// <summary>Whether episode <paramref name="episode"/> was cut by a limit.</summary>
    public bool IsTruncated(int episode) => !Truncated.IsEmpty && Truncated[episode];
}

/// <summary>
/// Turns a batch's rewards (and value estimates) into each step's advantage (how much better the action was than
/// expected: what a policy loss weights its log-probability by) and return (a value head's target), registered in
/// <see cref="AdvantageEstimators"/> under its <see cref="IKind.Name"/>: GAE, a group's centred rewards (GRPO), leave-one-out
/// (RLOO), discounted returns (REINFORCE). Every sum runs in a fixed order, episode by episode and step by step (a group's
/// mean and spread over its episodes in their order), never by thread timing, so the same input gives the same output bit
/// for bit, on every machine. Its option keys (<see cref="IConfigurableKind.Keys"/>) are the ones <see cref="Estimate"/>
/// takes: γ, λ, normalization. Thread-safe: one estimator serves many trainers at once.
/// </summary>
public interface IAdvantageEstimator : IConfigurableKind
{
    /// <summary>Whether it needs value estimates (a critic): an input without them is then refused.</summary>
    bool UsesValues { get; }

    /// <summary>Writes the advantage and the return of every step of <paramref name="input"/>.</summary>
    /// <param name="input">The rewards, lengths, ends, values and groups.</param>
    /// <param name="advantages">Receives each step's advantage [<see cref="AdvantageInput.Steps"/>].</param>
    /// <param name="returns">Receives each step's return [<see cref="AdvantageInput.Steps"/>].</param>
    /// <param name="options">The estimator's options by key (<see cref="IConfigurableKind.Keys"/>), values as text; null for its defaults.</param>
    /// <exception cref="ArgumentException">
    /// Outputs of another length than the steps, an input without values when <see cref="UsesValues"/>, an option the
    /// estimator does not take (the message names it and <see cref="IConfigurableKind.Keys"/>), or a bad value.
    /// </exception>
    void Estimate(AdvantageInput input, Span<float> advantages, Span<float> returns, IReadOnlyDictionary<string, string>? options = null);
}

/// <summary>
/// The advantage estimators a trainer can use, by name (ignoring case). The library's (GAE, group-relative,
/// leave-one-out, returns) come with Idrak.Reinforcement, registered when it ships with the application; Idrak.Abstraction
/// holds the registry so that any package's policy and trainer reach them. Register another with <see cref="Register"/>;
/// one under a library name shadows the library's, which <see cref="Unregister"/> brings back, guarded by its
/// <see cref="SetPolicy">policy</see>.
/// </summary>
public static class AdvantageEstimators
{
    private static readonly KindTable<IAdvantageEstimator> Table = new(nameof(AdvantageEstimators), "Advantage estimator",
        "add it with AdvantageEstimators.Register, or reference Idrak.Reinforcement (its estimators register themselves when it ships with the application)",
        (slot, app, library) => new GuardedEstimator(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static AdvantageEstimators() => LibraryDefaults.Ensure(typeof(AdvantageEstimators));

    /// <summary>Registers <paramref name="estimator"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAdvantageEstimator estimator) => Table.Register(estimator, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's estimator <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The estimator registered as <paramref name="name"/>, or null.</summary>
    public static IAdvantageEstimator? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The estimator registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones and the package to reference.</exception>
    public static IAdvantageEstimator Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's estimator <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IAdvantageEstimator? Default(string name) => Table.Default(name);

    /// <summary>Who registered the estimator <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's estimator <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> estimates with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's
    /// answer and compares the app's on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered estimator as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    // An app's estimator over the library's: each call falls back on its own, or is compared with the library's (written
    // into pooled arrays of its own, so the caller's outputs hold the library's answer).
    private sealed class GuardedEstimator(Slot slot, IAdvantageEstimator app, IAdvantageEstimator library) : IAdvantageEstimator
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public bool UsesValues => app.UsesValues;

        public void Estimate(AdvantageInput input, Span<float> advantages, Span<float> returns, IReadOnlyDictionary<string, string>? options = null)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                library.Estimate(input, advantages, returns, options);
                if (run is not null)
                {
                    run.Answered();
                    float[] rented = ArrayPool<float>.Shared.Rent(advantages.Length + returns.Length);
                    try
                    {
                        var otherAdvantages = rented.AsSpan(0, advantages.Length);
                        var otherReturns = rented.AsSpan(advantages.Length, returns.Length);
                        app.Estimate(input, otherAdvantages, otherReturns, options);
                        run.Done(Difference("advantages", advantages, otherAdvantages) ?? Difference("returns", returns, otherReturns));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(rented);
                    }
                }

                return;
            }

            try
            {
                app.Estimate(input, advantages, returns, options);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                library.Estimate(input, advantages, returns, options);
            }
        }

        private static string? Difference(string what, ReadOnlySpan<float> expected, ReadOnlySpan<float> actual) =>
            Comparisons.Difference(expected, actual, 1e-5f) is { } difference ? $"{what}: {difference}" : null;
    }
}
