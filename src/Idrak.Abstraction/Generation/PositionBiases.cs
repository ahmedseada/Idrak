// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A bias an attention layer adds to its scores by the positions of the query and the key (after the score scale, before
/// the softmax): [heads, queries, keys]. One provider serves one layer, or every layer that shares it (a bias the first
/// layer owns and the others read). Idrak's attention layers take one through their <c>PositionBias</c> setting
/// (<c>MultiHeadAttention</c>, <c>CausalSelfAttention</c>); the kinds are registered by name in
/// <see cref="AttentionPositionBiases"/>. <see cref="AttentionPositionBias"/> is a base that computes a bias once per shape
/// and keeps it while nothing is trained.
/// </summary>
public interface IAttentionPositionBias
{
    /// <summary>The heads it gives a bias for: the attention layer's query heads.</summary>
    int Heads { get; }

    /// <summary>
    /// The bias of the queries at positions [<paramref name="queryStart"/>, <paramref name="queryStart"/> +
    /// <paramref name="queries"/>) over the keys at positions [0, <paramref name="keys"/>): [heads, queries, keys] on
    /// <paramref name="device"/>. A decoding step's bias is the rows of the whole sequence's bias for its positions (so
    /// cached decoding equals the pass over the whole sequence). The caller reads it and never disposes it: a provider
    /// that keeps it owns it, one that makes it for the call leaves it to the current <see cref="TensorScope"/>.
    /// </summary>
    /// <param name="queries">Query positions (at least 1).</param>
    /// <param name="keys">Key positions (at least 1).</param>
    /// <param name="queryStart">The first query's position (0 for a whole sequence; the cached length for a decoding step).</param>
    /// <param name="device">Where the scores are.</param>
    Tensor For(int queries, int keys, int queryStart, Device device);
}

/// <summary>
/// A position bias that is a module (its parameters train and load with the attention layer that owns it) and computes
/// its bias once per shape: while nothing records gradients for its parameters, the bias of the last
/// (queries, keys, start, device) asked is kept, so the layers sharing one provider, and the batches of one length,
/// compute it once; while its parameters train, it is computed for every pass (and the kept one dropped). A kind derives
/// from it and implements <see cref="Compute"/>.
/// </summary>
public abstract class AttentionPositionBias : Module, IAttentionPositionBias
{
    private (int Queries, int Keys, int Start, Device Device, Tensor Bias)? _kept;

    /// <summary>A bias for <paramref name="heads"/> heads.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The heads are not positive.</exception>
    protected AttentionPositionBias(int heads)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        Heads = heads;
    }

    /// <inheritdoc />
    public int Heads { get; }

    /// <inheritdoc />
    /// <exception cref="ArgumentOutOfRangeException">A count is not positive or the start is negative.</exception>
    /// <exception cref="InvalidOperationException">The kind's <see cref="Compute"/> gave another shape than [heads, queries, keys].</exception>
    public Tensor For(int queries, int keys, int queryStart, Device device)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queries);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keys);
        ArgumentOutOfRangeException.ThrowIfNegative(queryStart);
        ArgumentNullException.ThrowIfNull(device);
        if (Autograd.IsEnabled && Trains())
        {
            Invalidate();                                                   // the weights change: no kept bias outlives this pass
            return Checked(Compute(queries, keys, queryStart, device), queries, keys);
        }

        if (_kept is { } kept && kept.Queries == queries && kept.Keys == keys && kept.Start == queryStart && kept.Device == device)
        {
            return kept.Bias;
        }

        DropKept();
        Tensor bias;
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            bias = TensorScope.Untrack(Checked(Compute(queries, keys, queryStart, device), queries, keys));
        }

        _kept = (queries, keys, queryStart, device, bias);
        return bias;
    }

    /// <summary>
    /// The bias of queries at [<paramref name="queryStart"/>, +<paramref name="queries"/>) over keys at [0,
    /// <paramref name="keys"/>), [<see cref="Heads"/>, queries, keys], made with differentiable operations when the
    /// parameters train.
    /// </summary>
    protected abstract Tensor Compute(int queries, int keys, int queryStart, Device device);

    /// <summary>
    /// Drops the kept bias, and whatever else the kind keeps from its parameters (<see cref="OnInvalidated"/>): after the
    /// parameters were changed outside a training pass, for example loaded.
    /// </summary>
    public void Invalidate()
    {
        DropKept();
        OnInvalidated();
    }

    /// <summary>
    /// Called by <see cref="Invalidate"/> (a training pass, a move, a load, the end): a kind that keeps more than the bias
    /// of the last shape computed from its parameters drops it here. Nothing by default.
    /// </summary>
    protected virtual void OnInvalidated()
    {
    }

    private void DropKept()
    {
        _kept?.Bias.Dispose();
        _kept = null;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) =>
        throw new NotSupportedException($"{GetType().Name} is read through For(queries, keys, queryStart, device), not Forward.");

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        Invalidate();
        base.MoveTo(device);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        Invalidate();
        base.Dispose();
    }

    private bool Trains()
    {
        foreach (var parameter in Parameters())
        {
            if (parameter.RequiresGrad)
            {
                return true;
            }
        }

        return false;
    }

    private Tensor Checked(Tensor bias, int queries, int keys) =>
        bias.Rank == 3 && bias.Shape[0] == Heads && bias.Shape[1] == queries && bias.Shape[2] == keys ? bias
            : throw new InvalidOperationException($"{GetType().Name} gave a bias of {Tensor.FormatShape(bias.Shape)}; [{Heads}, {queries}, {keys}] was asked for.");
}

/// <summary>
/// A position bias that follows the distance alone, key position minus query position: a learned value per head in each
/// row of <see cref="Table"/> [rows, heads], the row of a distance given by <see cref="Bucket"/>. A pass over several
/// queries looks the table up through an index of the shape's distances, once per shape (<see cref="AttentionPositionBias"/>).
/// A decoding step (one query, nothing training) reads a slice of the values by distance instead, [heads, distances],
/// which is made once on the bias's device for the distances asked so far and widened, twice as wide, when a step asks
/// for more: a step builds no index on the host and uploads nothing. The values by distance follow the table: they are
/// dropped by <see cref="AttentionPositionBias.Invalidate"/> (every training pass, a move, a load).
/// </summary>
public abstract class DistancePositionBias : AttentionPositionBias
{
    private (Device Device, int Back, Tensor Values)? _byDistance;     // [heads, back + ahead + 1]: column c is distance c - Back

    /// <summary>A bias for <paramref name="heads"/> heads.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The heads are not positive.</exception>
    protected DistancePositionBias(int heads)
        : base(heads)
    {
    }

    /// <summary>The values, [rows, heads] ("table"): row <see cref="Bucket"/> of a distance holds each head's value.</summary>
    public abstract Tensor Table { get; }

    /// <summary>The table row of key position minus query position <paramref name="relative"/>.</summary>
    public abstract int Bucket(int relative);

    /// <inheritdoc />
    protected sealed override Tensor Compute(int queries, int keys, int queryStart, Device device)
    {
        if (Table.Device != device)
        {
            throw new ArgumentException($"The position bias's table is on {Table.Device}; its bias was asked for on {device}.", nameof(device));
        }

        if (queries == 1 && !Autograd.IsEnabled)
        {
            return Step(keys, queryStart, device);
        }

        var index = new float[queries * keys];
        for (int i = 0; i < queries; i++)
        {
            int query = queryStart + i;
            var row = index.AsSpan(i * keys, keys);
            for (int j = 0; j < keys; j++)
            {
                row[j] = Bucket(j - query);
            }
        }

        var indices = Tensor.From(index, [queries * keys], device);
        return Table.EmbeddingLookup(indices).Reshape(queries, keys, Heads).Permute(2, 0, 1);   // [heads, queries, keys]
    }

    /// <inheritdoc />
    protected override void OnInvalidated()
    {
        _byDistance?.Values.Dispose();
        _byDistance = null;
    }

    // The bias of the query at `start` over keys [0, keys): distances -start .. keys - 1 - start, a slice of the values by
    // distance, [heads, 1, keys]. The values cover the distances asked so far, at least 16 back and none ahead, each side a
    // power of two; they are made again, wider, only when a step asks past them.
    private Tensor Step(int keys, int start, Device device)
    {
        int back = start, ahead = keys - 1 - start;
        if (_byDistance is not { } held || held.Device != device || back > held.Back || ahead > held.Values.Shape[1] - 1 - held.Back)
        {
            int heldBack = _byDistance is { } b && b.Device == device ? b.Back : 0;
            int heldAhead = _byDistance is { } a && a.Device == device ? a.Values.Shape[1] - 1 - a.Back : 0;
            int newBack = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(16, Math.Max(back, heldBack)));
            int newAhead = Math.Max(ahead, heldAhead) <= 0 ? 0 : (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(ahead, heldAhead));
            var index = new float[newBack + newAhead + 1];
            for (int c = 0; c < index.Length; c++)
            {
                index[c] = Bucket(c - newBack);
            }

            OnInvalidated();
            Tensor values;
            using (new TensorScope())
            {
                values = TensorScope.Untrack(Table.EmbeddingLookup(Tensor.From(index, [index.Length], device)).Permute(1, 0));   // [heads, distances]
            }

            _byDistance = (device, newBack, values);
            held = _byDistance.Value;
        }

        return held.Values.Narrow(1, held.Back - start, keys).Reshape(Heads, 1, keys);
    }
}

/// <summary>
/// The library's position bias "relative": a learned value per head for each distance from the query to the key,
/// key position minus query position clipped to [-<see cref="MaxDistance"/>, <see cref="MaxDistance"/>] (to
/// [-<see cref="MaxDistance"/>, 0] when not <see cref="Bidirectional"/>: a causal layer's keys are never after its
/// query). Its table is [buckets, heads] ("table"), buckets = 2 · max distance + 1 (max distance + 1 one-way).
/// </summary>
public sealed class RelativePositionBias : DistancePositionBias
{
    private Tensor _table;

    /// <summary>Its name in <see cref="AttentionPositionBiases"/>: "relative".</summary>
    public const string KindName = "relative";

    /// <summary>A bias of random values (uniform in ±0.02) drawn from <paramref name="random"/>.</summary>
    /// <param name="heads">The attention layer's query heads.</param>
    /// <param name="maxDistance">The farthest distance with a value of its own (farther keys share it).</param>
    /// <param name="bidirectional">Whether keys after the query have values of their own (an encoder's layer).</param>
    /// <param name="device">Where the table lives (the default device when null).</param>
    /// <param name="random">The random source (seed 0 when null).</param>
    /// <exception cref="ArgumentOutOfRangeException">The heads or the max distance are not positive.</exception>
    public RelativePositionBias(int heads, int maxDistance, bool bidirectional, Device? device = null, Random? random = null)
        : base(heads)
    {
        if (maxDistance <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDistance), maxDistance, "The position bias setting max_distance is positive.");
        }

        MaxDistance = maxDistance;
        Bidirectional = bidirectional;
        Buckets = bidirectional ? 2 * maxDistance + 1 : maxDistance + 1;
        random ??= new Random(0);
        _table = CreateParameter(UniformValues(Buckets * heads, 0.02f, random), [Buckets, heads], device ?? Device.Default);
    }

    /// <summary>The farthest distance with a value of its own.</summary>
    public int MaxDistance { get; }

    /// <summary>Whether keys after the query have values of their own.</summary>
    public bool Bidirectional { get; }

    /// <summary>The table's rows: 2 · max distance + 1, or max distance + 1 one-way.</summary>
    public int Buckets { get; }

    /// <summary>The values, [buckets, heads] ("table"): row <see cref="Bucket"/> of a distance holds each head's value.</summary>
    public override Tensor Table => _table;

    /// <summary>The table row of key position minus query position <paramref name="relative"/>.</summary>
    public override int Bucket(int relative) => Math.Clamp(relative, -MaxDistance, Bidirectional ? MaxDistance : 0) + MaxDistance;

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [_table];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        _table = MoveTensor(_table, device);
    }

    /// <inheritdoc />
    public override string ToString() => $"RelativePositionBias({Heads} heads, max distance {MaxDistance}{(Bidirectional ? "" : ", one-way")})";
}

/// <summary>
/// The library's position bias "bucketed": a learned value per head for each bucket of distances, key position minus
/// query position, the buckets exact for near keys and log-spaced for far ones up to <see cref="MaxDistance"/> (farther
/// keys share the last bucket), as Mesh TensorFlow's relative attention has them (several encoder-decoder families use
/// it). <see cref="Bidirectional"/> (an encoder's layer) gives keys after the query half of the buckets; one way (a causal
/// layer), every bucket is for keys at or before the query. Its table is [buckets, heads] ("table"). The buckets are
/// those transformers' <c>_relative_position_bucket</c> gives, its float32 logarithm included, worked out once per bias for
/// the distances up to <see cref="MaxDistance"/>; a bias of a shape is then an index into them, computed once per shape
/// (<see cref="AttentionPositionBias"/>).
/// </summary>
public sealed class BucketedPositionBias : DistancePositionBias
{
    private Tensor _table;

    /// <summary>Its name in <see cref="AttentionPositionBiases"/>: "bucketed".</summary>
    public const string KindName = "bucketed";

    private readonly int[] _buckets;                                    // the bucket of key - query = d at [d + MaxDistance], |d| at most MaxDistance

    /// <summary>A bias of random values (uniform in ±0.02) drawn from <paramref name="random"/>.</summary>
    /// <param name="heads">The attention layer's query heads.</param>
    /// <param name="buckets">The table's rows (num_buckets): at least 2 one way; at least 4, and even, both ways.</param>
    /// <param name="maxDistance">The distance from which keys share the last bucket of their side (max_distance): beyond the exact buckets' reach.</param>
    /// <param name="bidirectional">Whether keys after the query have buckets of their own (an encoder's layer).</param>
    /// <param name="device">Where the table lives (the default device when null).</param>
    /// <param name="random">The random source (seed 0 when null).</param>
    /// <exception cref="ArgumentOutOfRangeException">A count does not fit; the message names the setting.</exception>
    public BucketedPositionBias(int heads, int buckets, int maxDistance, bool bidirectional, Device? device = null, Random? random = null)
        : base(heads)
    {
        if (buckets < (bidirectional ? 4 : 2) || bidirectional && buckets % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(buckets), buckets,
                $"The position bias setting num_buckets is {(bidirectional ? "even and at least 4 both ways" : "at least 2 one way")}.");
        }

        int exact = (bidirectional ? buckets / 2 : buckets) / 2;
        if (maxDistance <= exact)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDistance), maxDistance,
                $"The position bias setting max_distance is above {exact}, the distances the exact buckets of {buckets} cover.");
        }

        Buckets = buckets;
        MaxDistance = maxDistance;
        Bidirectional = bidirectional;
        _buckets = new int[2 * maxDistance + 1];
        for (int d = -maxDistance; d <= maxDistance; d++)
        {
            _buckets[d + maxDistance] = BucketOf(d, buckets, maxDistance, bidirectional);
        }

        random ??= new Random(0);
        _table = CreateParameter(UniformValues(buckets * heads, 0.02f, random), [buckets, heads], device ?? Device.Default);
    }

    /// <summary>The table's rows.</summary>
    public int Buckets { get; }

    /// <summary>The distance from which keys share the last bucket of their side.</summary>
    public int MaxDistance { get; }

    /// <summary>Whether keys after the query have buckets of their own.</summary>
    public bool Bidirectional { get; }

    /// <summary>The values, [buckets, heads] ("table"): row <see cref="Bucket"/> of a distance holds each head's value.</summary>
    public override Tensor Table => _table;

    /// <summary>The table row of key position minus query position <paramref name="relative"/>.</summary>
    public override int Bucket(int relative) => _buckets[Math.Clamp(relative, -MaxDistance, MaxDistance) + MaxDistance];

    // transformers' _relative_position_bucket of one distance, in its arithmetic: the far buckets from a float32 logarithm of
    // distance / exact, divided by the float32 of the double log(max distance / exact), times the far buckets, truncated.
    // From max_distance on a side's distances are all in its last bucket (the scaled value reaches the far buckets' count
    // there, within a rounding, and is clamped).
    private static int BucketOf(int relative, int buckets, int maxDistance, bool bidirectional)
    {
        int result = 0;
        if (bidirectional)
        {
            buckets /= 2;
            if (relative > 0)
            {
                result = buckets;
            }

            relative = Math.Abs(relative);
        }
        else
        {
            relative = Math.Max(-relative, 0);
        }

        int exact = buckets / 2;
        if (relative < exact)
        {
            return result + relative;
        }

        float scaled = MathF.Log((float)relative / exact) / (float)Math.Log((double)maxDistance / exact) * (buckets - exact);
        return result + Math.Min(exact + (int)scaled, buckets - 1);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [_table];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        _table = MoveTensor(_table, device);
    }

    /// <inheritdoc />
    public override string ToString() => $"BucketedPositionBias({Heads} heads, {Buckets} buckets, max distance {MaxDistance}{(Bidirectional ? "" : ", one-way")})";
}

/// <summary>What <see cref="AttentionPositionBiases.Create"/> gives a kind to make a position bias from.</summary>
/// <param name="Heads">The attention layer's query heads.</param>
/// <param name="Settings">The kind's own settings (for "relative": <c>max_distance</c>, <c>bidirectional</c>; for "bucketed" also <c>num_buckets</c>).</param>
/// <param name="Device">Where its parameters live (the default device when null).</param>
/// <param name="Seed">The seed of random initial values.</param>
/// <param name="Weights">
/// Reads the kind's trained tensors by their names in the kind (for "relative" and "bucketed": "table", [buckets, heads]) and shapes, or
/// null for random values; it returns null for a tensor it does not hold.
/// </param>
public sealed record PositionBiasRequest(int Heads, JsonObject Settings, Device? Device = null, int Seed = 0,
    Func<string, IReadOnlyList<int>, float[]?>? Weights = null);

/// <summary>
/// The kinds of attention position bias, by name. The library registers "relative" (<see cref="RelativePositionBias"/>:
/// settings <c>max_distance</c>, default 128, and <c>bidirectional</c>, default true) and "bucketed"
/// (<see cref="BucketedPositionBias"/>: settings <c>num_buckets</c>, default 32, <c>max_distance</c>, default 128, and
/// <c>bidirectional</c>, default true); a family that needs another registers its own from its plug-in. A kind is made
/// once per model and its tensors trained with it, so an app's kind and the library's are never mixed in one model: the
/// registry has no fallback.
/// </summary>
public static class AttentionPositionBiases
{
    private static readonly SlotTable<string, Func<PositionBiasRequest, IAttentionPositionBias>> Registry = BuiltIn();

    private static SlotTable<string, Func<PositionBiasRequest, IAttentionPositionBias>> BuiltIn()
    {
        var table = new SlotTable<string, Func<PositionBiasRequest, IAttentionPositionBias>>(nameof(AttentionPositionBiases), comparer: StringComparer.Ordinal,
            unguarded: "a bias is made once per model and its tensors trained with it, so an app's kind and the library's cannot be mixed in one model");
        table.RegisterDefault(RelativePositionBias.KindName, Relative);
        table.RegisterDefault(BucketedPositionBias.KindName, Bucketed);
        return table;
    }

    /// <summary>Registers the kind <paramref name="name"/>; under a library name it takes the library's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, Func<PositionBiasRequest, IAttentionPositionBias> create)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(create);
        Registry.Register(name, create, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's kind <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered kinds.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>What makes the kind <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name; the message lists the registered ones.</exception>
    public static Func<PositionBiasRequest, IAttentionPositionBias> Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"Attention position bias '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); "
                                           + "register it with AttentionPositionBiases.Register (the plug-in that brings the kind does).");

    /// <summary>A bias of the kind <paramref name="name"/> for <paramref name="request"/>.</summary>
    /// <exception cref="NotSupportedException">The kind is not registered.</exception>
    /// <exception cref="ArgumentException">A setting is invalid; the message names it.</exception>
    public static IAttentionPositionBias Create(string name, PositionBiasRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var bias = Get(name)(request);
        return bias.Heads == request.Heads ? bias
            : throw new InvalidOperationException($"Attention position bias '{name}' made a bias of {bias.Heads} heads; {request.Heads} were asked for.");
    }

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static Func<PositionBiasRequest, IAttentionPositionBias>? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the kind <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    // "relative": max_distance (a positive whole number, default 128) and bidirectional (default true).
    private static RelativePositionBias Relative(PositionBiasRequest request)
    {
        var settings = request.Settings ?? [];
        int distance = 128;
        if (settings["max_distance"] is { } node)
        {
            distance = node.GetValueKind() == System.Text.Json.JsonValueKind.Number && node.AsValue().TryGetValue(out int whole) && whole > 0 ? whole
                : throw new ArgumentException($"The position bias setting max_distance is a positive whole number; got {node.ToJsonString()}.", nameof(request));
        }

        bool bidirectional = true;
        if (settings["bidirectional"] is { } flag)
        {
            bidirectional = flag.GetValueKind() is System.Text.Json.JsonValueKind.True or System.Text.Json.JsonValueKind.False ? flag.GetValue<bool>()
                : throw new ArgumentException($"The position bias setting bidirectional is true or false; got {flag.ToJsonString()}.", nameof(request));
        }

        var bias = new RelativePositionBias(request.Heads, distance, bidirectional, request.Device, new Random(request.Seed));
        if (request.Weights?.Invoke("table", [bias.Buckets, request.Heads]) is { } table)
        {
            if (table.Length != bias.Table.Size)
            {
                bias.Dispose();
                throw new InvalidDataException($"The position bias's table has {table.Length} values; [{bias.Buckets}, {request.Heads}] is {bias.Table.Size}.");
            }

            bias.Table.CopyFrom(table);
        }

        return bias;
    }

    // "bucketed": num_buckets (default 32), max_distance (default 128), both positive whole numbers, and bidirectional
    // (default true); the counts that do not fit together are refused by the bias, naming the setting.
    private static BucketedPositionBias Bucketed(PositionBiasRequest request)
    {
        var settings = request.Settings ?? [];
        int buckets = Whole(settings, "num_buckets", 32), distance = Whole(settings, "max_distance", 128);
        bool bidirectional = true;
        if (settings["bidirectional"] is { } flag)
        {
            bidirectional = flag.GetValueKind() is System.Text.Json.JsonValueKind.True or System.Text.Json.JsonValueKind.False ? flag.GetValue<bool>()
                : throw new ArgumentException($"The position bias setting bidirectional is true or false; got {flag.ToJsonString()}.", nameof(request));
        }

        var bias = new BucketedPositionBias(request.Heads, buckets, distance, bidirectional, request.Device, new Random(request.Seed));
        if (request.Weights?.Invoke("table", [bias.Buckets, request.Heads]) is { } table)
        {
            if (table.Length != bias.Table.Size)
            {
                bias.Dispose();
                throw new InvalidDataException($"The position bias's table has {table.Length} values; [{bias.Buckets}, {request.Heads}] is {bias.Table.Size}.");
            }

            bias.Table.CopyFrom(table);
        }

        return bias;
    }

    private static int Whole(JsonObject settings, string name, int fallback) => settings[name] is not { } node ? fallback
        : node.GetValueKind() == System.Text.Json.JsonValueKind.Number && node.AsValue().TryGetValue(out int whole) && whole > 0 ? whole
        : throw new ArgumentException($"The position bias setting {name} is a positive whole number; got {node.ToJsonString()}.", nameof(settings));
}
