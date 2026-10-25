// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>What a reward function scores: an answer to a prompt, with the expected answer and what the environment reported.</summary>
/// <param name="Prompt">The prompt (or the task) the answer is to.</param>
/// <param name="Answer">The policy's answer: a reply's text, the end of an episode's transcript.</param>
/// <param name="Reference">The expected answer, when the task has one; null otherwise.</param>
/// <param name="Info">What the environment reported (tests passed, rounds used, files changed), as text by key; null for nothing.</param>
public sealed record RewardInput(string Prompt, string Answer, string? Reference = null, IReadOnlyDictionary<string, string>? Info = null);

/// <summary>
/// Scores answers for an environment that rewards text (a prompt answered, a coding task solved), registered in
/// <see cref="RewardFunctions"/> under <see cref="Name"/>: a metric against the reference, a pattern, valid JSON, a length
/// budget, a judge model, the tests passed. Each answer's score depends on that answer alone, so a batch scores what its
/// inputs score one by one (a judge batches them for speed, not to compare them); the same input scores the same every
/// time. Thread-safe: one reward function scores many episodes at once.
/// </summary>
public interface IRewardFunction
{
    /// <summary>The reward's name in <see cref="RewardFunctions"/> ("exact", "tests-passed").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>The option keys <see cref="ScoreAsync"/> takes; any other is refused, naming these.</summary>
    IReadOnlyCollection<string> Keys { get; }

    /// <summary>
    /// The score of each of <paramref name="inputs"/>, in order, finite (higher is better). Completes synchronously when
    /// nothing is waited for (a metric); awaits when something is (a judge model): await the result once.
    /// </summary>
    /// <param name="inputs">The answers to score.</param>
    /// <param name="options">The reward's options by key (<see cref="Keys"/>), values as text; null for its defaults.</param>
    /// <param name="cancellationToken">Stops the call.</param>
    /// <exception cref="ArgumentException">An option the reward does not take (the message names it and <see cref="Keys"/>), or a bad value.</exception>
    ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default);
}

/// <summary>
/// The reward functions environments score text with, by name (ignoring case). The library's text rewards come with
/// Idrak.Text, and an environment family's own with its plug-in (the coding family's tests passed, in
/// <c>Idrak.Reinforcement.Plugins</c>), each registered when it ships with the application. Register another with
/// <see cref="Register"/>; one under a library name shadows the library's, which <see cref="Unregister"/> brings back,
/// guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class RewardFunctions
{
    private static readonly SlotTable<string, IRewardFunction> Table =
        new(nameof(RewardFunctions), (slot, app, library) => new GuardedReward(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static RewardFunctions() => LibraryDefaults.Ensure(typeof(RewardFunctions));

    /// <summary>Registers <paramref name="reward"/> under its <see cref="IRewardFunction.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IRewardFunction reward)
    {
        ArgumentNullException.ThrowIfNull(reward);
        ArgumentException.ThrowIfNullOrWhiteSpace(reward.Name);
        Table.Register(reward.Name, reward, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's reward <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The reward registered as <paramref name="name"/>, or null.</summary>
    public static IRewardFunction? Find(string name) => Table.Find(name);

    /// <summary>The reward registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones and the packages to reference.</exception>
    public static IRewardFunction Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(RewardFunctions), name, Table.Keys,
            $"Reward function '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); "
            + "add it with RewardFunctions.Register, or reference Idrak.Text (the text rewards) or Idrak.Reinforcement.Plugins "
            + "and call ReinforcementPlugins.RegisterAll() (the environment families' rewards).");

    /// <summary>The library's reward <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IRewardFunction? Default(string name) => Table.Default(name);

    /// <summary>Who registered the reward <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's reward <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> scores with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's score
    /// and compares the app's on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered reward as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(r => $"{r.Name}: {r.Summary}"));

    // An app's reward over the library's: each call falls back on its own, or is compared with the library's. Across an
    // await a compared call may move threads, so its allocation counts are those of the threads it ended on.
    private sealed class GuardedReward(Slot slot, IRewardFunction app, IRewardFunction library) : IRewardFunction
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlyCollection<string> Keys => app.Keys;

        public async ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null,
            CancellationToken cancellationToken = default)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = await library.ScoreAsync(inputs, options, cancellationToken).ConfigureAwait(false);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        var other = await app.ScoreAsync(inputs, options, cancellationToken).ConfigureAwait(false);
                        run.Done(Comparisons.Difference(answer, other, 1e-5f));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return await app.ScoreAsync(inputs, options, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return await library.ScoreAsync(inputs, options, cancellationToken).ConfigureAwait(false);
            }
        }
    }
}
