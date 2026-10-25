// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// One rank's part in the ranks' agreement on a stop (Plan 27, R7). A rank that wants the run to stop (its token cancelled
/// by Ctrl+C or a timeout, a stop request) does not leave on its own: it asks (<see cref="Request"/>), and the ranks agree,
/// at the same step boundary, that the run stops there (<see cref="Agree"/>, which every rank calls at every boundary and
/// which gives every rank the same answer); every rank then leaves at that boundary, with no collective left half-done,
/// and the trainer ends the run as a stopped run ends (one checkpoint, written by rank 0). The flag travels in a collective
/// the ranks make anyway where the agreement takes one (<see cref="ICarriedStopAgreement"/>: a gradient reduction adds its
/// contribution to its header and hands the sum back), else in a one-value all-reduce at the boundary. A run in one
/// process has no agreement: its token stops it as before. <see cref="StopAgreements.For"/> gives a rank's agreement.
/// </summary>
public interface IStopAgreement
{
    /// <summary>The agreement's name in <see cref="StopAgreements"/> ("all-reduce", "shared").</summary>
    string Name { get; }

    /// <summary>This rank.</summary>
    int Rank { get; }

    /// <summary>The number of ranks.</summary>
    int WorldSize { get; }

    /// <summary>Whether this rank asked for a stop since its run began (<see cref="Reset"/>).</summary>
    bool Requested { get; }

    /// <summary>Why this rank asked (the first reason given), or null.</summary>
    string? Reason { get; }

    /// <summary>
    /// Asks the ranks to stop the run at a coming boundary: from any thread, at any time (a cancelled token, a timeout, a
    /// stop request); a second request keeps the first reason. The run goes on until the ranks agree.
    /// </summary>
    /// <param name="reason">Why, in a few words ("cancelled", "a stop request from the dashboard").</param>
    void Request(string reason);

    /// <summary>
    /// A run begins (every rank calls it at the same point, before the run's first boundary): what was asked and agreed
    /// before is forgotten, so a run after a stopped one on the same ranks starts afresh.
    /// </summary>
    void Reset();

    /// <summary>
    /// The step boundary before <paramref name="step"/>: every rank calls it at each boundary of its run, at the same point
    /// (a collective). True when the ranks agree that the run stops here, the same answer on every rank; once true, true
    /// until <see cref="Reset"/>.
    /// </summary>
    /// <param name="step">The step about to begin (for the messages; the boundaries are counted by the calls).</param>
    bool Agree(long step);

    /// <summary>Whether the ranks agreed to stop.</summary>
    bool Agreed { get; }

    /// <summary>The step the ranks agreed to stop before (no rank began it), or null.</summary>
    long? Step { get; }

    /// <summary>How many ranks had asked when the ranks agreed (0 before).</summary>
    int Askers { get; }

    /// <summary>
    /// The token for the work inside a step on a cluster (a rollout, a step's batches): cancelled only when this rank gave
    /// up waiting for the agreement (<see cref="ClusterOptions.StopWait"/> after it asked), so a rank leaves a step on its
    /// own only then, loudly (<see cref="GaveUp"/>).
    /// </summary>
    CancellationToken GiveUpToken { get; }

    /// <summary>Why this rank gave up waiting for the agreement (it leaves on its own), or null.</summary>
    string? GaveUp { get; }

    /// <summary>The exception every rank's trainer leaves with at the agreed boundary (a cancellation).</summary>
    /// <param name="step">The step no rank began.</param>
    ClusterRunStoppedException Stopped(long step);
}

/// <summary>
/// A stop agreement whose flag a collective the ranks make anyway may carry (a gradient reduction's header): the
/// collective adds every rank's <see cref="Contribution"/> and hands the sum to <see cref="Carried"/>, so the next boundary
/// decides without a message of its own. Asked <c>is ICarriedStopAgreement</c> once, when the collective is set up.
/// </summary>
public interface ICarriedStopAgreement : IStopAgreement
{
    /// <summary>What this rank adds to a sum every rank makes at the same point: 1 when it asked, else 0.</summary>
    float Contribution { get; }

    /// <summary>
    /// The sum over the ranks of every rank's <see cref="Contribution"/>, from a collective every rank made at the same
    /// point since the last boundary; the next boundary decides from it without a message of its own.
    /// </summary>
    void Carried(float sum);
}

/// <summary>One way of agreeing on a stop, registered by name in <see cref="StopAgreements"/>.</summary>
public interface IStopAgreementProvider
{
    /// <summary>The name ("all-reduce").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>The agreement of <paramref name="collectives"/>' rank.</summary>
    /// <exception cref="NotSupportedException">The agreement cannot serve these collectives; the message names one that can.</exception>
    IStopAgreement Create(ICollectives collectives, ClusterOptions options);
}

/// <summary>
/// What every agreement shares (Plan 27, R7): the request and its reason, the agreed state, the give-up timer
/// (<see cref="ClusterOptions.StopWait"/>) and the messages. An agreement says, at each boundary, how many ranks asked by
/// then (<see cref="Decide"/>), the same count on every rank.
/// </summary>
public abstract class StopAgreement : IStopAgreement
{
    private readonly Lock _gate = new();
    private int _asked;
    private string? _reason;
    private CancellationTokenSource _giveUp = new();
    private Timer? _timer;
    private string? _gaveUp;
    private volatile bool _agreed;

    /// <summary>The agreement of rank <paramref name="rank"/> of <paramref name="worldSize"/>.</summary>
    /// <param name="rank">This rank.</param>
    /// <param name="worldSize">The number of ranks.</param>
    /// <param name="wait">How long a rank that asked waits for the agreement before it gives up (zero: as long as it takes).</param>
    protected StopAgreement(int rank, int worldSize, TimeSpan wait)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rank);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(worldSize, rank);
        ArgumentOutOfRangeException.ThrowIfLessThan(wait, TimeSpan.Zero);
        (Rank, WorldSize, Wait) = (rank, worldSize, wait);
    }

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public int Rank { get; }

    /// <inheritdoc/>
    public int WorldSize { get; }

    /// <summary>How long a rank that asked waits for the agreement before it gives up (zero: as long as it takes).</summary>
    public TimeSpan Wait { get; }

    /// <inheritdoc/>
    public bool Requested => Volatile.Read(ref _asked) != 0;

    /// <inheritdoc/>
    public string? Reason => Volatile.Read(ref _reason);

    /// <inheritdoc/>
    public bool Agreed => _agreed;

    /// <inheritdoc/>
    public long? Step { get; private set; }

    /// <inheritdoc/>
    public int Askers { get; private set; }

    /// <inheritdoc/>
    public CancellationToken GiveUpToken
    {
        get
        {
            lock (_gate)
            {
                return _giveUp.Token;
            }
        }
    }

    /// <inheritdoc/>
    public string? GaveUp => Volatile.Read(ref _gaveUp);

    /// <inheritdoc/>
    public void Request(string reason)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reason);
        lock (_gate)
        {
            if (_asked != 0 || _agreed)
            {
                return;
            }

            _reason = reason;
            Volatile.Write(ref _asked, 1);
            if (Wait > TimeSpan.Zero)
            {
                _timer = new Timer(static state => ((StopAgreement)state!).GiveUpNow(), this, Wait, System.Threading.Timeout.InfiniteTimeSpan);
            }
        }

        Asked(reason);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        lock (_gate)
        {
            _timer?.Dispose();
            _timer = null;
            if (_giveUp.IsCancellationRequested)
            {
                _giveUp.Dispose();
                _giveUp = new CancellationTokenSource();
            }

            _asked = 0;
            _reason = null;
            _gaveUp = null;
            _agreed = false;
            Step = null;
            Askers = 0;
        }

        Began();
    }

    /// <inheritdoc/>
    public bool Agree(long step)
    {
        if (_agreed)
        {
            return true;
        }

        int askers = Decide(step);
        if (askers <= 0)
        {
            return false;
        }

        lock (_gate)
        {
            _timer?.Dispose();
            _timer = null;
            Step = step;
            Askers = askers;
            _agreed = true;
        }

        return true;
    }

    /// <inheritdoc/>
    public ClusterRunStoppedException Stopped(long step)
    {
        var inv = CultureInfo.InvariantCulture;
        string who = Askers == 1 ? "1 rank" : $"{Askers.ToString(inv)} ranks";
        string mine = Reason is { } reason ? $"; rank {Rank.ToString(inv)} asked: {reason}" : "";
        return new ClusterRunStoppedException(step, $"the {WorldSize.ToString(inv)} ranks agreed to stop before step {step.ToString(inv)} ({who} of {WorldSize.ToString(inv)} asked{mine})")
        {
            Agreed = true,
        };
    }

    /// <summary>
    /// The boundary before <paramref name="step"/> (every rank calls it at the same point): how many ranks had asked by then,
    /// the same count on every rank; 0 to go on.
    /// </summary>
    protected abstract int Decide(long step);

    /// <summary>This rank asked (once a run), with <paramref name="reason"/>; nothing by default.</summary>
    protected virtual void Asked(string reason)
    {
    }

    /// <summary>A run began on this rank (<see cref="Reset"/>); nothing by default.</summary>
    protected virtual void Began()
    {
    }

    // The wait ran out before the ranks agreed: the work inside the step is cancelled and the rank leaves on its own, saying why.
    private void GiveUpNow()
    {
        CancellationTokenSource source;
        lock (_gate)
        {
            if (_agreed || _asked == 0 || _gaveUp is not null)
            {
                return;
            }

            var inv = CultureInfo.InvariantCulture;
            _gaveUp = $"rank {Rank.ToString(inv)} asked to stop ({_reason}) and the ranks did not reach a step boundary to agree on it within "
                      + $"{Wait.TotalSeconds.ToString("0.###", inv)} s (ClusterOptions.StopWait, IDRAK_CLUSTER_STOP_WAIT): it leaves on its own, inside its step; "
                      + "the other ranks wait for it until their collectives time out (IDRAK_CLUSTER_TIMEOUT)";
            source = _giveUp;
        }

        source.Cancel();
    }
}

/// <summary>
/// The stop agreements, by name (Plan 27, R7). Idrak.Network registers "all-reduce" (the flag carried in the gradient
/// reduction's header, else a one-value all-reduce at the boundary: for ranks that are processes of their own, on one
/// machine or many); Idrak.Cluster registers "shared" (an in-process simulated cluster's ranks agree through the run's
/// memory, no message, whatever ranks take part in a step). <see cref="ClusterOptions.StopAgreement"/> names the one a run
/// uses; "auto" takes "shared" for collectives whose group can change from step to step (<see cref="IStepGroup"/>) when it
/// is registered, else "all-reduce".
/// </summary>
public static class StopAgreements
{
    /// <summary>The name that lets the collectives choose (<see cref="NameFor"/>).</summary>
    public const string Automatic = "auto";

    private static readonly SlotTable<string, IStopAgreementProvider> Table = new(nameof(StopAgreements), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a rank's agreement holds its place in the ranks' order of boundaries, so another cannot take over a run");

    // One agreement per rank's collectives, shared by every trainer and synchronisation over them.
    private static readonly ConditionalWeakTable<ICollectives, IStopAgreement> Made = new();

    static StopAgreements() => LibraryDefaults.Ensure(typeof(StopAgreements));

    /// <summary>The registered agreements, in registration order.</summary>
    public static IReadOnlyList<IStopAgreementProvider> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="provider"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IStopAgreementProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(provider.Name);
        Table.Register(provider.Name, provider, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's agreement of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The agreement of that name, or null.</summary>
    public static IStopAgreementProvider? Find(string name) => Table.Find(name);

    /// <summary>The agreement of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IStopAgreementProvider Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(StopAgreements), name, Table.Keys,
            $"Stop agreement '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with StopAgreements.Register, or reference Idrak.Network.");

    /// <summary>The library's agreement of that name, or null.</summary>
    public static IStopAgreementProvider? Default(string name) => Table.Default(name);

    /// <summary>Who registered the agreement of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each agreement as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));

    /// <summary>
    /// The agreement a run over <paramref name="collectives"/> uses: <see cref="ClusterOptions.StopAgreement"/>, or for
    /// "auto" "shared" when the collectives' group can change from step to step (<see cref="IStepGroup"/>) and "shared" is
    /// registered, else "all-reduce".
    /// </summary>
    public static string NameFor(ICollectives collectives, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(collectives);
        ArgumentNullException.ThrowIfNull(options);
        return !string.Equals(options.StopAgreement, Automatic, StringComparison.OrdinalIgnoreCase) ? options.StopAgreement
            : collectives is IStepGroup && Table.Find("shared") is not null ? "shared" : "all-reduce";
    }

    /// <summary>
    /// This rank's agreement over <paramref name="collectives"/>: made on the first call (by <see cref="NameFor"/>, with
    /// <paramref name="options"/>, else the collectives' own, else <see cref="ClusterOptions.FromEnvironment"/>), then the same
    /// one for every trainer and synchronisation over these collectives.
    /// </summary>
    public static IStopAgreement For(ICollectives collectives, ClusterOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(collectives);
        if (Made.TryGetValue(collectives, out var made))
        {
            return made;
        }

        var o = options ?? collectives.Context?.Options ?? ClusterOptions.FromEnvironment();
        var agreement = Get(NameFor(collectives, o)).Create(collectives, o);
        return Made.GetValue(collectives, _ => agreement);
    }

    /// <summary>
    /// Ties <paramref name="token"/> to <paramref name="agreement"/>: a cancellation asks for a stop (<paramref name="reason"/>)
    /// instead of leaving; disposing unties it. Already cancelled: it asks at once.
    /// </summary>
    public static CancellationTokenRegistration Watch(this IStopAgreement agreement, CancellationToken token, string reason = "cancelled (Ctrl+C or a timeout)")
    {
        ArgumentNullException.ThrowIfNull(agreement);
        return token.Register(static state =>
        {
            var (a, r) = ((IStopAgreement, string))state!;
            a.Request(r);
        }, (agreement, reason));
    }

    /// <summary>
    /// When <paramref name="agreement"/> gave up waiting (<see cref="IStopAgreement.GaveUp"/>): the exception a rank leaves
    /// with, naming why (a <see cref="TimeoutException"/> around <paramref name="cancelled"/>); else null.
    /// </summary>
    public static TimeoutException? GaveUpError(this IStopAgreement? agreement, OperationCanceledException cancelled) =>
        agreement?.GaveUp is { } why ? new TimeoutException(why, cancelled) : null;
}
