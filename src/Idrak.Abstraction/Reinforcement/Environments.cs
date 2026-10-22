// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>
/// What one step of an environment gives back (<see cref="IEnvironment{TObservation, TAction}.StepAsync"/>), as
/// Gymnasium's <c>step</c> does: the next observation, the reward for the action, and whether the episode ended.
/// </summary>
/// <param name="Observation">What the agent sees after the action: the next state, or the last one when the episode ended.</param>
/// <param name="Reward">The reward for the action, finite.</param>
/// <param name="Terminated">
/// The episode reached an end of its own (a goal, a failure, an answer given): nothing follows, so no value is bootstrapped
/// from <paramref name="Observation"/>.
/// </param>
/// <param name="Truncated">
/// The episode was cut by a limit outside the task (steps, rounds, time): the state still had a future, so an estimator
/// bootstraps from the value of <paramref name="Observation"/>.
/// </param>
/// <param name="Info">What else the environment reports for the step (tests passed, files changed), as text by key; null for nothing.</param>
public readonly record struct EnvironmentStep<TObservation>(TObservation Observation, float Reward, bool Terminated, bool Truncated,
    IReadOnlyDictionary<string, string>? Info = null)
{
    /// <summary>Whether the episode ended (<see cref="Terminated"/> or <see cref="Truncated"/>): it must be reset before the next step.</summary>
    public bool Done => Terminated || Truncated;
}

/// <summary>
/// What every environment has whatever its observations and actions are: the spaces they come from, and its resources (a
/// workspace, a process), released by <see cref="IDisposable.Dispose"/> (more than once is harmless). An
/// <see cref="IEnvironmentKind"/> creates environments as this, and the generic
/// <see cref="IEnvironment{TObservation, TAction}"/> is what an episode runs on.
/// </summary>
public interface IEnvironment : IDisposable
{
    /// <summary>The values the observations take.</summary>
    Space ObservationSpace { get; }

    /// <summary>The values the actions take.</summary>
    Space ActionSpace { get; }
}

/// <summary>
/// An environment an agent acts in, an episode at a time, as Gymnasium's <c>Env</c>: <see cref="Reset"/> starts an
/// episode from a seed and gives the first observation, <see cref="StepAsync"/> takes an action and gives the next
/// observation, the reward and whether the episode ended. The rules a contract suite checks
/// (<c>Idrak.Abstraction.Testing</c>'s <c>EnvironmentSuite</c>):
/// <list type="bullet">
/// <item>a step before the first <see cref="Reset"/>, or after a step that ended the episode
/// (<see cref="EnvironmentStep{TObservation}.Done"/>), is refused with an <see cref="InvalidOperationException"/>: a
/// finished episode must be reset;</item>
/// <item>the same seed and the same actions give the same observations, rewards, ends and info, in this environment
/// reset again and in another one of the same kind: everything random is drawn from the episode's seed, nothing from the
/// clock or shared state;</item>
/// <item>environments of one kind are independent: several run at once (one episode each, on as many threads) give the
/// steps each gives alone;</item>
/// <item>rewards are finite, and observations lie in <see cref="IEnvironment.ObservationSpace"/>;</item>
/// <item>a step whose token is cancelled throws an <see cref="OperationCanceledException"/>.</item>
/// </list>
/// One environment is not thread-safe: it runs one episode at a time. A rollout runs many episodes at once with an
/// environment each (<see cref="IEnvironmentKind.Create"/>).
/// </summary>
/// <typeparam name="TObservation">What the agent sees: a float array for a box, a conversation for a chat task.</typeparam>
/// <typeparam name="TAction">What the agent does: a choice, a float array, a chat message.</typeparam>
public interface IEnvironment<TObservation, TAction> : IEnvironment
{
    /// <summary>Starts an episode drawn from <paramref name="seed"/> and returns its first observation; any episode under way is abandoned.</summary>
    /// <param name="seed">The episode's seed: the same seed starts the same episode.</param>
    TObservation Reset(int seed);

    /// <summary>
    /// Takes <paramref name="action"/> in the episode under way and returns what follows. Completes synchronously when
    /// the environment does no waiting (a simulation); awaits when it does (a build, a test run): await the result once.
    /// </summary>
    /// <param name="action">The agent's action, a value of <see cref="IEnvironment.ActionSpace"/>.</param>
    /// <param name="cancellationToken">Stops the step; the episode must then be reset.</param>
    /// <exception cref="InvalidOperationException">No episode is under way: reset first (also after an episode ended).</exception>
    /// <exception cref="ArgumentException">The action is not a value of the action space.</exception>
    ValueTask<EnvironmentStep<TObservation>> StepAsync(TAction action, CancellationToken cancellationToken = default);
}

/// <summary>
/// A kind of environment, registered in <see cref="EnvironmentKinds"/> under <see cref="Name"/>: what a trainer and
/// <c>idrak reinforce --env</c> create environments from, with options of the kind's own (<see cref="Keys"/>). Creating is
/// cheap enough to make an environment per episode slot; what is costly and shared (a prepared workspace, a loaded task
/// list) the kind keeps and its environments reuse. Thread-safe: <see cref="Create"/> is called from many threads.
/// </summary>
public interface IEnvironmentKind
{
    /// <summary>The kind's name in <see cref="EnvironmentKinds"/> ("prompts", "coding").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>The option keys <see cref="Create"/> takes; any other is refused, naming these.</summary>
    IReadOnlyCollection<string> Keys { get; }

    /// <summary>The type of the observations of the environments it creates (the TObservation of <see cref="IEnvironment{TObservation, TAction}"/>).</summary>
    Type ObservationType { get; }

    /// <summary>The type of their actions (the TAction of <see cref="IEnvironment{TObservation, TAction}"/>).</summary>
    Type ActionType { get; }

    /// <summary>
    /// A new environment of the kind, an <see cref="IEnvironment{TObservation, TAction}"/> of <see cref="ObservationType"/>
    /// and <see cref="ActionType"/>; the caller disposes it.
    /// </summary>
    /// <param name="options">The kind's options by key (<see cref="Keys"/>), values as text; null for its defaults.</param>
    /// <param name="seed">What the environment draws once when it is made (an order of tasks); each episode has its own seed (<see cref="IEnvironment{TObservation, TAction}.Reset"/>).</param>
    /// <exception cref="ArgumentException">An option the kind does not take (the message names it and <see cref="Keys"/>), or a bad value.</exception>
    IEnvironment Create(IReadOnlyDictionary<string, string>? options = null, int seed = 0);
}

/// <summary>
/// The environment kinds by name (ignoring case): what a trainer creates its environments from. The library registers
/// none: environments live in plug-ins (<c>Idrak.Reinforcement.Plugins</c>, one namespace per family) and apps, and the
/// packages that add one of their own register it (Idrak.Text's chat prompts). A kind that is not registered is refused
/// with "not registered", naming the plug-in package; nothing falls back to another kind. Register another with
/// <see cref="Register"/>; one under a library name takes the library's place, which <see cref="Unregister"/> brings back.
/// </summary>
public static class EnvironmentKinds
{
    private static readonly SlotTable<string, IEnvironmentKind> Table = new(nameof(EnvironmentKinds), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "an environment kind is not called per use: it creates environments that keep an episode's state from step to step, so no single call can fall back");

    static EnvironmentKinds() => LibraryDefaults.Ensure(typeof(EnvironmentKinds));

    /// <summary>Registers <paramref name="kind"/> under its <see cref="IEnvironmentKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IEnvironmentKind kind)
    {
        ArgumentNullException.ThrowIfNull(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(kind.Name);
        Table.Register(kind.Name, kind, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's kind <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The kind registered as <paramref name="name"/>, or null.</summary>
    public static IEnvironmentKind? Find(string name) => Table.Find(name);

    /// <summary>The kind registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones and the plug-in package to reference.</exception>
    public static IEnvironmentKind Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Environment kind '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); "
                                           + "add it with EnvironmentKinds.Register, or reference Idrak.Reinforcement.Plugins and call ReinforcementPlugins.RegisterAll().");

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IEnvironmentKind? Default(string name) => Table.Default(name);

    /// <summary>Who registered the kind <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered kind as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(k => $"{k.Name}: {k.Summary}"));

    /// <summary>A new environment of the kind <paramref name="name"/> (<see cref="IEnvironmentKind.Create"/>), as the observation and action types asked for.</summary>
    /// <exception cref="NotSupportedException">No kind of that name is registered.</exception>
    /// <exception cref="ArgumentException">The kind's environments observe or act with other types (the message names both), or an option it does not take.</exception>
    public static IEnvironment<TObservation, TAction> Create<TObservation, TAction>(string name, IReadOnlyDictionary<string, string>? options = null, int seed = 0)
    {
        var kind = Get(name);
        if (kind.ObservationType != typeof(TObservation) || kind.ActionType != typeof(TAction))
        {
            throw new ArgumentException($"Environment kind '{kind.Name}' observes {kind.ObservationType.Name} and acts with {kind.ActionType.Name}, "
                                        + $"not {typeof(TObservation).Name} and {typeof(TAction).Name}.", nameof(name));
        }

        var environment = kind.Create(options, seed);
        if (environment is IEnvironment<TObservation, TAction> typed)
        {
            return typed;
        }

        environment.Dispose();
        throw new ArgumentException($"Environment kind '{kind.Name}' says it observes {kind.ObservationType.Name} and acts with {kind.ActionType.Name}, "
                                    + $"but created a {environment.GetType().Name}, which is not an IEnvironment<{typeof(TObservation).Name}, {typeof(TAction).Name}>.", nameof(name));
    }
}
