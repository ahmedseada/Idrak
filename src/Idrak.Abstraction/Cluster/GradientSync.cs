// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>How a data-parallel run makes every rank start from the same values (<see cref="ClusterOptions.StartingState"/>).</summary>
public enum StartingState
{
    /// <summary>Rank 0's values are sent to every rank (what each rank loaded or drew before does not matter).</summary>
    Broadcast,

    /// <summary>Every rank's values are hashed and compared; a rank whose values differ stops the run, naming it and the parameter.</summary>
    Check,
}

/// <summary>What a step's gradient reduction gave every rank: the same two numbers on each.</summary>
/// <param name="Tokens">The tokens (or samples) of the step over every rank: 0 when no rank had any, so a loop over ranks of unequal lengths ends together.</param>
/// <param name="Loss">The step's loss over every rank's rows: the loss one process would report for the whole step.</param>
public readonly record struct GradientSyncResult(double Tokens, double Loss);

/// <summary>
/// One rank's gradient synchronisation in data-parallel training (plan 23, phase C2): the trainers (<c>Trainer</c>,
/// <c>FineTuner</c>, <c>Pretrainer</c>) call it after the last micro-batch's backward pass and before the optimizer's
/// update, so every rank updates with the gradient one process would compute on every rank's rows. The gradients are
/// addressed as (parameter, offset, count) pieces in parameter order, never packed whole on the device; every sum is made
/// in the order the collective algorithm fixes (rule 86 of docs/optimization.md), so every rank holds identical values.
/// </summary>
public interface IGradientSync : IDisposable
{
    /// <summary>The synchronisation's name in <see cref="GradientSyncs"/> ("all-reduce").</summary>
    string Name { get; }

    /// <summary>The collectives it reduces over (this rank's).</summary>
    ICollectives Collectives { get; }

    /// <summary>This rank.</summary>
    int Rank => Collectives.Rank;

    /// <summary>The number of ranks.</summary>
    int WorldSize => Collectives.WorldSize;

    /// <summary>The parameters whose gradients it reduces (the optimizer's, in its order).</summary>
    IReadOnlyList<Tensor> Parameters { get; }

    /// <summary>
    /// Each rank's share of a step's rows, the same list on every rank: equal until the ranks' speeds are measured
    /// (<see cref="ClusterOptions.RebalanceSteps"/>), then in proportion to each rank's measured tokens per second.
    /// <see cref="DrawShares.Owners"/> turns it into the rank of each of a step's draws.
    /// </summary>
    IReadOnlyList<double> Shares { get; }

    /// <summary>Makes every rank start from the same parameter values (<see cref="ClusterOptions.StartingState"/>). Called once, before the first step.</summary>
    /// <exception cref="InvalidOperationException">With <see cref="StartingState.Check"/>: a rank's values differ from rank 0's; the message names the rank and the parameter.</exception>
    void Start();

    /// <summary>
    /// Reduces the gradients of this step across the ranks, in place: afterwards every rank's parameters hold the same
    /// gradient, the one one process would compute on every rank's rows. Every rank calls it once per step, also a rank
    /// that had no rows this step (<paramref name="tokens"/> 0: it adds nothing).
    /// </summary>
    /// <param name="tokens">This rank's tokens (or samples) in the step, the count its loss is a mean over; 0 when it had none.</param>
    /// <param name="loss">This rank's loss for the step.</param>
    /// <param name="normalized">
    /// False (the usual): each rank's loss and gradients are means over its own <paramref name="tokens"/>, and are weighted
    /// by them (the sum is token-weighted). True: each rank's loss is already its share of the whole step's loss (divided
    /// by the step's count over every rank), so the gradients and losses are summed as they are.
    /// </param>
    /// <returns>The step's tokens over every rank and its loss over every rank's rows.</returns>
    GradientSyncResult Reduce(double tokens, double loss, bool normalized = false);

    /// <summary>
    /// Tells the synchronisation that step <paramref name="step"/> (from 1) finished its update: it sends rank 0's weights
    /// every <see cref="ClusterOptions.SyncWeightsEvery"/> steps and re-measures the ranks' speeds when due.
    /// </summary>
    void Stepped(long step);

    /// <summary>Sums <paramref name="values"/> across the ranks, in place, in float32 (losses, counts).</summary>
    void Sum(Span<float> values);

    /// <summary>Copies rank <paramref name="root"/>'s values of <paramref name="tensors"/> (on this rank's device) to every rank's.</summary>
    void Broadcast(IReadOnlyList<Tensor> tensors, int root = 0);

    /// <summary>Returns once every rank has called it (after rank 0 wrote a checkpoint, say).</summary>
    void Barrier();
}

/// <summary>One kind of gradient synchronisation, registered by name in <see cref="GradientSyncs"/>.</summary>
public interface IGradientSyncProvider
{
    /// <summary>The name ("all-reduce").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>The synchronisation of <paramref name="optimizer"/>'s parameters over <paramref name="collectives"/>.</summary>
    IGradientSync Create(ICollectives collectives, Optimizer optimizer, ClusterOptions options);
}

/// <summary>
/// An optimizer that brings the gradients to the host itself (<c>HostOptimizer</c>): a gradient synchronisation reduces
/// them there, on the pieces it downloaded, instead of moving them down and up again; the optimizer then clips and
/// updates from the reduced host values.
/// </summary>
public interface IHostGradients
{
    /// <summary>Brings every parameter's gradient to the host (a parameter without one gets none).</summary>
    void DownloadGradients();

    /// <summary>Parameter <paramref name="parameter"/>'s gradient on the host after <see cref="DownloadGradients"/>, or empty when it has none.</summary>
    Span<float> HostGradient(int parameter);

    /// <summary>Gives parameter <paramref name="parameter"/> a host gradient when it has none (its values to be written), and returns it.</summary>
    Span<float> CreateHostGradient(int parameter);

    /// <summary>The parameters' values changed on their device (rank 0's were sent to this rank): the host copies are read again.</summary>
    void ReloadParameters();
}

/// <summary>
/// The gradient synchronisations, by name (plan 23). Idrak.Network registers "all-reduce" (the gradients summed after the
/// backward pass, every step); none is built into Idrak.Abstraction. <see cref="ClusterOptions.GradientSync"/> names the one a run uses.
/// </summary>
public static class GradientSyncs
{
    private static readonly SlotTable<string, IGradientSyncProvider> Table = new(nameof(GradientSyncs), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a synchronisation holds its rank's place in the group's message order, so another cannot take over a step");

    static GradientSyncs() => LibraryDefaults.Ensure(typeof(GradientSyncs));

    /// <summary>The registered synchronisations, in registration order.</summary>
    public static IReadOnlyList<IGradientSyncProvider> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="provider"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IGradientSyncProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(provider.Name);
        Table.Register(provider.Name, provider, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's synchronisation of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The synchronisation of that name, or null.</summary>
    public static IGradientSyncProvider? Find(string name) => Table.Find(name);

    /// <summary>The synchronisation of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name (Idrak.Network registers "all-reduce": ship it with the app).</exception>
    public static IGradientSyncProvider Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Gradient synchronisation '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with GradientSyncs.Register, or reference Idrak.Network.");

    /// <summary>The library's synchronisation of that name, or null.</summary>
    public static IGradientSyncProvider? Default(string name) => Table.Default(name);

    /// <summary>Who registered the synchronisation of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each synchronisation as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));

    /// <summary>
    /// The synchronisation <paramref name="options"/> names (<see cref="ClusterOptions.GradientSync"/>) of
    /// <paramref name="optimizer"/>'s parameters over <paramref name="collectives"/>.
    /// </summary>
    public static IGradientSync Create(ICollectives collectives, Optimizer optimizer, ClusterOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(collectives);
        ArgumentNullException.ThrowIfNull(optimizer);
        options ??= ClusterOptions.FromEnvironment();
        return Get(options.GradientSync).Create(collectives, optimizer, options);
    }
}

/// <summary>
/// Which rank takes each of a step's draws (plan 23): data is sharded by draw position, so with equal shares rank r takes
/// draws k·W + r; with shares measured from the ranks' speeds each draw goes to the rank furthest behind its share of the
/// step's tokens. The same on every rank for the same shares, so every rank knows every rank's rows.
/// </summary>
public static class DrawShares
{
    /// <summary>The rank of each draw, from each draw's size (tokens) and each rank's share (the same list on every rank).</summary>
    /// <param name="sizes">Each draw's tokens (or samples), in draw order.</param>
    /// <param name="shares">Each rank's share, summing to 1 (<see cref="IGradientSync.Shares"/>).</param>
    public static int[] Owners(ReadOnlySpan<long> sizes, IReadOnlyList<double> shares)
    {
        ArgumentNullException.ThrowIfNull(shares);
        int world = shares.Count;
        ArgumentOutOfRangeException.ThrowIfZero(world);
        var owners = new int[sizes.Length];
        if (Equal(shares))
        {
            for (int i = 0; i < owners.Length; i++)
            {
                owners[i] = i % world;                                   // draws k·W + r
            }

            return owners;
        }

        var taken = new double[world];
        double total = 0;
        for (int i = 0; i < sizes.Length; i++)
        {
            total += Math.Max(1, sizes[i]);
            int best = 0;
            double deficit = double.NegativeInfinity;
            for (int r = 0; r < world; r++)
            {
                double behind = (shares[r] * total) - taken[r];          // ties go to the lower rank: the same on every rank
                if (behind > deficit)
                {
                    (best, deficit) = (r, behind);
                }
            }

            owners[i] = best;
            taken[best] += Math.Max(1, sizes[i]);
        }

        return owners;
    }

    /// <summary>Equal shares for <paramref name="world"/> ranks.</summary>
    public static IReadOnlyList<double> Even(int world)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(world);
        return [.. Enumerable.Repeat(1.0 / world, world)];
    }

    private static bool Equal(IReadOnlyList<double> shares)
    {
        for (int r = 1; r < shares.Count; r++)
        {
            if (shares[r] != shares[0])
            {
                return false;
            }
        }

        return true;
    }
}
