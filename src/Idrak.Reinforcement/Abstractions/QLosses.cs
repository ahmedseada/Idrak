// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Reinforcement.Abstractions;

/// <summary>
/// What a Q-learning loss (<see cref="IQLoss"/>) is given for one mini-batch of transitions drawn from a replay buffer
/// (<see cref="ReplayBuffer"/>): the online network's action values at the states, recorded by autograd, and on the host
/// what the targets are made of: the actions taken, the rewards, each transition's discount (γ, or 0 where the episode
/// terminated, so nothing is bootstrapped), and the target network's action values at the next states (and the online
/// network's there, for double Q-learning). Checked when made.
/// </summary>
public sealed class QLossInput
{
    /// <summary>A mini-batch of <paramref name="values"/>' rows.</summary>
    /// <param name="values">Q(s, ·) of the online network [batch, actions], recorded: the loss's gradient flows into it.</param>
    /// <param name="actions">The action taken in each transition [batch].</param>
    /// <param name="rewards">The reward of each transition [batch], finite.</param>
    /// <param name="discounts">What each transition bootstraps with [batch]: γ (γⁿ for an n-step return), 0 where the episode terminated.</param>
    /// <param name="nextTargetValues">Q'(s', ·) of the target network at the next states [batch · actions, row-major], not recorded.</param>
    /// <param name="nextValues">Q(s', ·) of the online network at the next states [batch · actions], for double Q-learning; empty when not computed.</param>
    /// <param name="weights">Each transition's weight [batch] (importance weights of a prioritized replay); empty for 1 each.</param>
    /// <exception cref="ArgumentException">Values that are not [batch, actions], or arrays of other lengths, an action outside the actions, or a reward or discount that is not finite.</exception>
    public QLossInput(Tensor values, ReadOnlyMemory<int> actions, ReadOnlyMemory<float> rewards, ReadOnlyMemory<float> discounts, ReadOnlyMemory<float> nextTargetValues,
        ReadOnlyMemory<float> nextValues = default, ReadOnlyMemory<float> weights = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Shape.Length != 2 || values.Shape[0] == 0 || values.Shape[1] == 0)
        {
            throw new ArgumentException($"A Q-learning loss reads action values [batch, actions], not [{string.Join(", ", values.Shape.ToArray())}].", nameof(values));
        }

        int batch = values.Shape[0], count = values.Shape[1];
        if (actions.Length != batch || rewards.Length != batch || discounts.Length != batch || nextTargetValues.Length != batch * count
            || (!nextValues.IsEmpty && nextValues.Length != batch * count) || (!weights.IsEmpty && weights.Length != batch))
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"A batch of {batch} transitions over {count} actions has {batch} actions, rewards, discounts (and weights) and {batch * count} next values: "
                + $"{actions.Length}, {rewards.Length}, {discounts.Length}, {weights.Length} and {nextTargetValues.Length} (online {nextValues.Length}) given."));
        }

        var a = actions.Span;
        var r = rewards.Span;
        var d = discounts.Span;
        for (int i = 0; i < batch; i++)
        {
            if ((uint)a[i] >= (uint)count || !float.IsFinite(r[i]) || !float.IsFinite(d[i]))
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                    $"Transition {i} has the action {a[i]} (of {count}), the reward {r[i]} and the discount {d[i]}: an action of the space and finite numbers."));
            }
        }

        Values = values;
        Actions = actions;
        Rewards = rewards;
        Discounts = discounts;
        NextTargetValues = nextTargetValues;
        NextValues = nextValues;
        Weights = weights;
    }

    /// <summary>Q(s, ·) of the online network [batch, actions], recorded.</summary>
    public Tensor Values { get; }

    /// <summary>The transitions of the mini-batch.</summary>
    public int Batch => Values.Shape[0];

    /// <summary>The actions of the space.</summary>
    public int ActionCount => Values.Shape[1];

    /// <summary>The action taken in each transition.</summary>
    public ReadOnlyMemory<int> Actions { get; }

    /// <summary>The reward of each transition.</summary>
    public ReadOnlyMemory<float> Rewards { get; }

    /// <summary>What each transition bootstraps with: γ, or 0 where its episode terminated.</summary>
    public ReadOnlyMemory<float> Discounts { get; }

    /// <summary>Q'(s', ·) of the target network, row-major [batch · actions].</summary>
    public ReadOnlyMemory<float> NextTargetValues { get; }

    /// <summary>Q(s', ·) of the online network, row-major; empty when not computed.</summary>
    public ReadOnlyMemory<float> NextValues { get; }

    /// <summary>Each transition's weight; empty for 1 each.</summary>
    public ReadOnlyMemory<float> Weights { get; }
}

/// <summary>
/// A Q-learning loss: from the online network's action values at a mini-batch's states and the target network's at
/// the next states, a one-element loss whose gradient trains the online network (<c>OffPolicyTrainer</c>), registered in
/// <see cref="QLosses"/> under <see cref="Name"/>. The library's is DQN's ("dqn": Mnih et al. 2015, with double Q-learning
/// as an option). Thread-safe: it keeps no state.
/// </summary>
public interface IQLoss
{
    /// <summary>The loss's name in <see cref="QLosses"/> ("dqn").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>The option keys <see cref="Loss"/> takes; any other is refused, naming these.</summary>
    IReadOnlyCollection<string> Keys { get; }

    /// <summary>The mini-batch's loss, one element, finite; its gradient flows to <see cref="QLossInput.Values"/>.</summary>
    /// <param name="input">The online values, the targets' parts.</param>
    /// <param name="options">The loss's options by key (<see cref="Keys"/>), values as text; null for its defaults.</param>
    /// <exception cref="ArgumentException">An option the loss does not take (the message names it and <see cref="Keys"/>), a bad value, or an input without what the loss needs (the online next values).</exception>
    Tensor Loss(QLossInput input, IReadOnlyDictionary<string, string>? options = null);

    /// <summary>
    /// Whether <see cref="Loss"/> with <paramref name="options"/> reads <see cref="QLossInput.NextValues"/> (double
    /// Q-learning chooses the next action by the online network), so a trainer runs the online network over the next states
    /// only for a loss that reads it. True unless the loss says otherwise, so a loss that does not answer is always given them.
    /// </summary>
    /// <param name="options">The options <see cref="Loss"/> will be called with; null for its defaults.</param>
    bool NeedsNextValues(IReadOnlyDictionary<string, string>? options = null) => true;
}

/// <summary>
/// The Q-learning losses an off-policy trainer can use, by name (ignoring case): its algorithm. The library's:
/// <list type="bullet">
/// <item><see cref="Dqn"/> ("dqn"): the deep Q-network's temporal-difference loss (Mnih et al. 2015, "Human-level control
/// through deep reinforcement learning", Nature 518): y = r + d · max_a Q'(s', a) with the target network Q' and the
/// discount d (0 where the episode terminated), a Huber loss of Q(s, a) − y (<see cref="HuberKey"/>, δ 1: the paper's error
/// clipped to [−1, 1]; 0 for the squared error ½(Q − y)²), averaged over the batch (weighted when weights are given). With
/// <see cref="DoubleKey"/> true, double Q-learning (van Hasselt, Guez and Silver 2016, "Deep Reinforcement Learning with
/// Double Q-learning", AAAI): y = r + d · Q'(s', argmax_a Q(s', a)), the action chosen by the online network and valued by
/// the target one.</item>
/// </list>
/// No R1 registry fits an off-policy algorithm (<c>PolicyLosses</c> reads the log-probabilities of a policy that acted,
/// which a Q-network has not), so this one lives with the trainer that uses it, in Idrak.Reinforcement. Register another
/// with <see cref="Register"/>; one under a library name shadows the library's, which <see cref="Unregister"/> brings back,
/// guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class QLosses
{
    /// <summary>The deep Q-network's loss: "dqn".</summary>
    public const string Dqn = "dqn";

    /// <summary>The Huber loss's δ (0: the squared error): "huber".</summary>
    public const string HuberKey = "huber";

    /// <summary>Double Q-learning (true or false): "double".</summary>
    public const string DoubleKey = "double";

    private static readonly SlotTable<string, IQLoss> Table = BuiltIn();

    private static SlotTable<string, IQLoss> BuiltIn()
    {
        var table = new SlotTable<string, IQLoss>(nameof(QLosses), (slot, app, library) => new GuardedLoss(slot, app, library), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(Dqn, new DqnLoss());
        return table;
    }

    /// <summary>Registers <paramref name="loss"/> under its <see cref="IQLoss.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IQLoss loss)
    {
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentException.ThrowIfNullOrWhiteSpace(loss.Name);
        Table.Register(loss.Name, loss, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's loss <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The loss registered as <paramref name="name"/>, or null.</summary>
    public static IQLoss? Find(string name) => Table.Find(name);

    /// <summary>The loss registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IQLoss Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(QLosses), name, Table.Keys,
            $"Q-learning loss '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with QLosses.Register.");

    /// <summary>The library's loss <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IQLoss? Default(string name) => Table.Default(name);

    /// <summary>Who registered the loss <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's loss <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> computes the library's; <see cref="SlotPolicy.Shadow"/> lets the library's answer
    /// and compares the app's value on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered loss as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(l => $"{l.Name}: {l.Summary}"));

    // An app's loss over the library's: each call falls back on its own, or is compared with the library's. A compared
    // call computes the app's loss without recording it (the library's is the one trained on).
    private sealed class GuardedLoss(Slot slot, IQLoss app, IQLoss library) : IQLoss
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlyCollection<string> Keys => app.Keys;

        // The library's loss may answer in the app's place (a fall-back, a shadow), so the next values are given when either reads them.
        public bool NeedsNextValues(IReadOnlyDictionary<string, string>? options = null) => app.NeedsNextValues(options) || library.NeedsNextValues(options);

        public Tensor Loss(QLossInput input, IReadOnlyDictionary<string, string>? options = null)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Loss(input, options);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        Tensor other;
                        using (Autograd.NoGrad())
                        {
                            other = app.Loss(input, options);
                        }

                        run.Done(Comparisons.Difference(answer.ToArray(), other.ToArray(), 1e-5f));
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
                return app.Loss(input, options);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                return library.Loss(input, options);
            }
        }
    }
}
