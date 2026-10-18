// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Cluster;
using Idrak.Layers;

namespace Idrak.Training;

/// <summary>How a pipeline stage runs one training step (<see cref="PipelineStage.Step"/>).</summary>
public sealed record PipelineStepOptions
{
    /// <summary>
    /// Whether the last stage's loss runs the stage's last module (the output head) itself, on the rows it needs: the loss is
    /// then given the hidden state before the head (a chunked token loss never holds the whole logits).
    /// </summary>
    public bool HeadInLoss { get; init; }

    /// <summary>Whether decoder blocks keep only their output and run again in the backward pass (activation checkpointing).</summary>
    public bool Checkpointing { get; init; }

    /// <summary>A scope entered around each forward and each backward pass of micro-batch m (a packed batch's layout), or null.</summary>
    public Func<int, IDisposable?>? Around { get; init; }

    /// <summary>The training step, for the telemetry; -1 when none.</summary>
    public long Step { get; init; } = -1;
}

/// <summary>What one pipeline step gave this rank.</summary>
/// <param name="Loss">The step's loss: the sum of the micro-batches' losses, the same on every stage of the pipeline.</param>
/// <param name="Wall">The step's time on this rank.</param>
/// <param name="Blocked">The time this stage waited for its neighbours' activations and gradients.</param>
/// <param name="Bubble">Blocked over Wall: the share of the step this stage stood idle.</param>
/// <param name="IdealBubble">The schedule's arithmetic for stages of equal cost.</param>
/// <param name="BytesSent">Bytes of activations and gradients sent.</param>
/// <param name="BytesReceived">Bytes received.</param>
public readonly record struct PipelineStepResult(double Loss, TimeSpan Wall, TimeSpan Blocked, double Bubble, double IdealBubble, long BytesSent, long BytesReceived);

/// <summary>
/// One rank's stage of a pipeline (plan 23, phase C7): the model's top-level modules the plan gives this stage, on this
/// rank's device, joined to the neighbouring stages by <see cref="PipelineBoundary"/> modules. <see cref="Place"/> cuts any
/// <see cref="Sequential"/> model the same way on every rank (the modules' tensor bytes, <see cref="PipelinePlan.For"/>):
/// this rank keeps its stage (and a table its tied head reads) and drops the rest, so it holds only its stage's tensors;
/// load the model on the host (the CPU) first, so no device ever holds it whole. Training steps run the micro-batches in
/// the order of the <see cref="ClusterOptions.Schedule"/>; generation runs on the first stage (<see cref="Network"/>) while
/// the others <see cref="Serve"/>. With more ranks than stages, each stage's replicas reduce its gradients over
/// <see cref="Replicas"/>. Nothing here names a model family: any registered family's network is cut by its modules.
/// </summary>
public sealed class PipelineStage : IDisposable
{
    private const int LossTag = 4 << 20, SharedTag = 5 << 20, SumTag = 6 << 20;
    private readonly PipelineBoundary? _receive;
    private readonly PipelineBoundary? _send;
    private readonly List<(int Global, int[] Holders, Tensor Tensor)> _shared;
    private readonly HashSet<Tensor> _notOwned;
    private DecodingContext? _serving;
    private bool _stopped;

    private readonly int[] _holders;

    private PipelineStage(PipelinePlan plan, PipelineLayout layout, int rank, ITransport transport, Device device, IReadOnlyList<Module> modules,
        IReadOnlyList<Tensor> parameters, IReadOnlyList<int> indices, List<(int, int[], Tensor)> shared, IPipelineSchedule schedule, ICollectives? replicas,
        int[] holders, bool deviceActivations)
    {
        _holders = holders;
        Plan = plan;
        Layout = layout;
        Rank = rank;
        Transport = transport;
        Device = device;
        Modules = modules;
        Parameters = parameters;
        ParameterIndices = indices;
        _shared = shared;
        _notOwned = new HashSet<Tensor>(shared.Where(s => s.Item2[0] != layout.StageOf(rank)).Select(s => s.Item3), ReferenceEqualityComparer.Instance);
        Schedule = schedule;
        Replicas = replicas;
        if (!IsFirst)
        {
            _receive = new PipelineBoundary(transport, layout.RankOf(Stage - 1, Replica), sends: false, device, deviceActivations: deviceActivations);
        }

        if (!IsLast)
        {
            _send = new PipelineBoundary(transport, layout.RankOf(Stage + 1, Replica), sends: true, device,
                answerer: IsFirst ? layout.RankOf(layout.Stages - 1, Replica) : null, deviceActivations: deviceActivations);
        }

        Network = new Sequential(IsFirst && _send is not null ? [.. modules, _send] : modules) { Name = $"pipeline stage {Stage}" };
    }

    /// <summary>The plan every rank cut the model by.</summary>
    public PipelinePlan Plan { get; }

    /// <summary>Which ranks run which stage.</summary>
    public PipelineLayout Layout { get; }

    /// <summary>This rank.</summary>
    public int Rank { get; }

    /// <summary>This rank's stage.</summary>
    public int Stage => Layout.StageOf(Rank);

    /// <summary>This rank's pipeline (replica) when several run side by side; else 0.</summary>
    public int Replica => Layout.ReplicaOf(Rank);

    /// <summary>Whether this stage takes the model's input (the token embedding).</summary>
    public bool IsFirst => Stage == 0;

    /// <summary>Whether this stage gives the model's output (the head).</summary>
    public bool IsLast => Stage == Layout.Stages - 1;

    /// <summary>The transport joining the stages.</summary>
    public ITransport Transport { get; }

    /// <summary>This rank's device, where its stage lives.</summary>
    public Device Device { get; }

    /// <summary>This stage's modules, in the model's order.</summary>
    public IReadOnlyList<Module> Modules { get; }

    /// <summary>
    /// The stage as a network: on the first stage its modules and the boundary to the next stage, so a generator over it
    /// (a <c>TextGenerator</c>) runs the whole pipeline and gets the logits back; on the others its modules alone.
    /// </summary>
    public Sequential Network { get; }

    /// <summary>The stage's parameters (a table a tied head reads included), each once, in the model's order: what its optimizer trains.</summary>
    public IReadOnlyList<Tensor> Parameters { get; }

    /// <summary>Each of <see cref="Parameters"/>' index among the whole model's distinct parameters (the same on every rank).</summary>
    public IReadOnlyList<int> ParameterIndices { get; }

    /// <summary>
    /// Whether this stage is the first of those holding parameter <paramref name="parameter"/> (an index into
    /// <see cref="Parameters"/>): a tensor several stages hold (a tied table) is counted, and written, by the first alone.
    /// </summary>
    public bool Owns(int parameter) => !_notOwned.Contains(Parameters[parameter]);

    /// <summary>
    /// The first stage holding the whole model's distinct parameter <paramref name="globalParameter"/> (its index in the
    /// model's distinct parameters, as <see cref="ParameterIndices"/> counts them): the stage that writes it into a checkpoint.
    /// </summary>
    public int HolderOf(int globalParameter) => _holders[globalParameter];

    /// <summary>How this stage's activations cross to its neighbours: "device" or "host" (<see cref="ClusterOptions.DeviceActivations"/>).</summary>
    public string Staging => (_send ?? _receive)?.Staging ?? "host";

    /// <summary>This stage's bytes, as the plan measured them.</summary>
    public long Bytes => Plan.Stages[Stage].Bytes;

    /// <summary>The schedule its training steps follow.</summary>
    public IPipelineSchedule Schedule { get; }

    /// <summary>The collectives of this stage's data-parallel replicas (the ranks holding the same stage), or null with one replica.</summary>
    public ICollectives? Replicas { get; }

    /// <summary>
    /// Cuts <paramref name="network"/> by the plan <paramref name="options"/> ask for (the same on every rank of
    /// <paramref name="cluster"/>), moves this rank's stage to its device and releases the other modules' tensors (unless
    /// <paramref name="keepOthers"/>: a rank that writes whole checkpoints keeps them where they were loaded).
    /// </summary>
    /// <param name="network">The whole model, every rank's alike (loaded on the host, so no device holds it whole).</param>
    /// <param name="cluster">This rank's collectives over every rank; their <see cref="ICollectives.Transport"/> carries the activations.</param>
    /// <param name="options">The cluster settings, or null: the collectives' own (<see cref="ICollectives.Context"/>), else the environment's.</param>
    /// <param name="keepOthers">Whether the modules of the other stages stay where they are (for a whole checkpoint), instead of being released.</param>
    /// <exception cref="NotSupportedException">The collectives give no transport, or a stage of several replicas has no topology to group them by.</exception>
    /// <exception cref="ArgumentException">The stages do not divide the ranks, or are more than the model's modules, or the cuts are wrong.</exception>
    public static PipelineStage Place(Sequential network, ICollectives cluster, ClusterOptions? options = null, bool keepOthers = false)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(cluster);
        options ??= cluster.Context?.Options ?? ClusterOptions.FromEnvironment();
        if (options.Shard != ShardMode.None)
        {
            // Plan 23, C6 against C7: a stage clips by the norm over every stage and rank 0's checkpoint gathers each stage's whole
            // weights and optimizer state, where a sharded synchronisation keeps each replica's share and clips over the replicas
            // alone; and the parameters shard's gathers would run between the stage-to-stage sends. Refused before anything moves.
            throw new NotSupportedException($"Pipeline placement (IDRAK_CLUSTER_PLACEMENT=pipeline) does not combine with sharding (IDRAK_CLUSTER_SHARD="
                + $"{options.Shard.ToString().ToLowerInvariant()}): a pipeline already splits the parameters by stage, and its clipping and checkpoints need each stage's whole state "
                + "on its replicas. Use one or the other (IDRAK_CLUSTER_SHARD=none with the pipeline).");
        }

        var transport = cluster.Transport ?? throw new NotSupportedException(
            $"The collectives '{cluster.Name}' give no transport (ICollectives.Transport): a pipeline sends activations from stage to stage, point to point.");
        var layout = PipelineLayout.For(cluster.WorldSize, options);
        var schedule = PipelineSchedules.Get(options.Schedule);
        var top = network.ToList();
        var plan = PlanFor(network, layout, options, cluster.Context?.Topology);
        int rank = cluster.Rank, stage = layout.StageOf(rank);
        var range = plan.Stages[stage];

        // The modules each stage holds: its range, and the tables its tied layers read from modules outside it.
        HashSet<Module> Held(PipelineStageRange r)
        {
            var held = new HashSet<Module>(ReferenceEqualityComparer.Instance);
            for (int i = r.Start; i < r.End; i++)
            {
                held.Add(top[i]);
                foreach (var tied in top[i].Descendants().OfType<Linear>().Select(l => l.TiedTo).OfType<Module>())
                {
                    held.Add(tied);
                }
            }

            return held;
        }

        // The whole model's distinct parameters (their order is the model's, the same on every rank) and the stages holding each.
        var all = network.Parameters().Distinct(ReferenceEqualityComparer.Instance).Cast<Tensor>().ToList();
        var index = new Dictionary<Tensor, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < all.Count; i++)
        {
            index[all[i]] = i;
        }

        var holders = new List<int>[all.Count];
        var mine = new HashSet<int>();
        for (int s = 0; s < plan.Stages.Count; s++)
        {
            foreach (var p in Held(plan.Stages[s]).SelectMany(m => m.Parameters()).Distinct(ReferenceEqualityComparer.Instance).Cast<Tensor>())
            {
                (holders[index[p]] ??= []).Add(s);
                if (s == stage)
                {
                    mine.Add(index[p]);
                }
            }
        }

        // This rank's stage goes to its device; a block whose next norm is another stage's stops computing it ahead.
        var kept = Held(range);
        foreach (var module in kept)
        {
            module.To(cluster.Device);
        }

        var inRange = new HashSet<Module>(top.Skip(range.Start).Take(range.Count).SelectMany(m => m.Descendants()), ReferenceEqualityComparer.Instance);
        foreach (var block in inRange.OfType<DecoderBlock>())
        {
            if (block.NextNorm is { } next && !inRange.Contains(next))
            {
                block.NextNorm = null;
            }
        }

        // The other stages' modules release their tensors (but never one this stage reads), unless the rank keeps them.
        if (!keepOthers)
        {
            var reads = new HashSet<Storage>(kept.SelectMany(Reads).Select(t => t.Storage), ReferenceEqualityComparer.Instance);
            foreach (var module in top.Where(m => !kept.Contains(m)))
            {
                if (Reads(module).All(t => !reads.Contains(t.Storage)))
                {
                    module.Dispose();
                    continue;
                }

                foreach (var tensor in module.Parameters().Concat(module.Buffers()).Where(t => !reads.Contains(t.Storage)))
                {
                    tensor.Dispose();
                }
            }
        }

        // The tensors after the move (a moved module holds new ones), at the same places in the model's order.
        var moved = network.Parameters().Distinct(ReferenceEqualityComparer.Instance).Cast<Tensor>().ToList();
        int[] indices = [.. mine.Order()];
        var parameters = indices.Select(i => moved[i]).ToList();
        var shared = indices.Where(i => holders[i].Count > 1).Select(i => (i, holders[i].ToArray(), moved[i])).ToList();

        ICollectives? replicas = null;
        if (layout.Replicas > 1)
        {
            var context = cluster.Context ?? throw new NotSupportedException(
                $"Stage {stage} is held by {layout.Replicas} replicas, which reduce its gradients over a group of their own; the collectives '{cluster.Name}' give no topology to group them by (ICollectives.Context).");
            int[] members = layout.StageRanks(stage);
            var topology = ClusterGroups.Topology(context.Topology, members);
            replicas = Collectives.Get(options.CollectiveFor(topology)).Create(ClusterGroups.Transport(transport, members), cluster.Device,
                new ClusterContext(topology, layout.ReplicaOf(rank), options));
        }

        return new PipelineStage(plan, layout, rank, transport, cluster.Device, [.. top.Skip(range.Start).Take(range.Count)], parameters, indices, shared, schedule, replicas,
            [.. holders.Select(h => h?.Min() ?? -1)], options.DeviceActivations);
    }

    /// <summary>
    /// The plan <see cref="Place"/> cuts <paramref name="network"/> by over <paramref name="layout"/>: its top-level modules
    /// (embeddings first, the head last) and the bytes of the tensors each holds or reads, measured from the tensors the
    /// model was built with (a tied head counts the table it reads), cut as <paramref name="options"/> ask
    /// (<see cref="PipelinePlan.For"/>).
    /// </summary>
    public static PipelinePlan PlanFor(Sequential network, PipelineLayout layout, ClusterOptions? options = null, IClusterTopology? topology = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        options ??= ClusterOptions.FromEnvironment();
        var tensorIndex = new Dictionary<Storage, int>(ReferenceEqualityComparer.Instance);
        var tensorBytes = new List<long>();
        var modules = new List<PipelineModule>(network.Count);
        foreach (var module in network)
        {
            var read = new List<int>();
            foreach (var tensor in Reads(module))
            {
                if (!tensorIndex.TryGetValue(tensor.Storage, out int at))
                {
                    at = tensorBytes.Count;
                    tensorIndex[tensor.Storage] = at;
                    tensorBytes.Add((long)tensor.Storage.Length * sizeof(float));
                }

                if (!read.Contains(at))
                {
                    read.Add(at);
                }
            }

            modules.Add(new PipelineModule(module.Name ?? module.GetType().Name, read));
        }

        return PipelinePlan.For(modules, tensorBytes, layout, options, topology);
    }

    // The tensors a module holds or reads: its parameters and buffers, and the tables its tied layers read.
    private static IEnumerable<Tensor> Reads(Module module) =>
        module.Parameters().Concat(module.Buffers())
            .Concat(module.Descendants().OfType<Linear>().Select(l => l.TiedTo).OfType<Module>().SelectMany(t => t.Parameters().Concat(t.Buffers())))
            .Distinct(ReferenceEqualityComparer.Instance).Cast<Tensor>();

    /// <summary>
    /// One training step of <paramref name="microBatches"/> micro-batches, each forward and backward pass in the schedule's
    /// order; afterwards every parameter of the stage holds the step's gradient (a tied table's summed over the stages that
    /// read it, so each holds the same values). The caller zeroes the gradients before and updates after (with a pipeline's
    /// replicas, after their reduction).
    /// </summary>
    /// <param name="microBatches">The micro-batches.</param>
    /// <param name="input">The first stage's input of micro-batch m (token ids); the other stages receive theirs.</param>
    /// <param name="loss">The last stage's loss of micro-batch m (a single value, weighted so the micro-batches' sum is the step's) from the stage's output.</param>
    /// <param name="options">How the step runs.</param>
    /// <returns>The step's loss (on every stage of the pipeline) and its timing.</returns>
    public PipelineStepResult Step(int microBatches, Func<int, Tensor>? input, Func<int, Tensor, Tensor>? loss, PipelineStepOptions? options = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(microBatches);
        Required(input, loss);
        options ??= new PipelineStepOptions();
        var order = Schedule.Order(Stage, Layout.Stages, microBatches);
        bool nested = Nested(order, microBatches);
        long started = Stopwatch.GetTimestamp();
        _receive?.ResetCounters();
        _send?.ResetCounters();
        var roots = new Tensor?[microBatches];
        var scopes = new TensorScope?[microBatches];
        var stepScope = nested ? null : new TensorScope();
        double total = 0;
        try
        {
            foreach (var work in order)
            {
                int m = work.MicroBatch;
                using var around = options.Around?.Invoke(m);
                if (work.Kind == PipelineWorkKind.Forward)
                {
                    if (nested)
                    {
                        scopes[m] = new TensorScope();
                    }

                    roots[m] = ForwardOne(m, input, loss, options);
                }
                else
                {
                    var root = roots[m] ?? throw new InvalidOperationException($"The schedule '{Schedule.Name}' runs micro-batch {m}'s backward pass before its forward pass.");
                    root.Backward();
                    if (IsLast)
                    {
                        total += root.Item();
                    }

                    roots[m] = null;
                    if (nested)
                    {
                        scopes[m]!.Dispose();
                        scopes[m] = null;
                    }
                }
            }
        }
        finally
        {
            for (int m = microBatches - 1; m >= 0; m--)
            {
                scopes[m]?.Dispose();
            }

            stepScope?.Dispose();
        }

        _send?.Flush();                                                    // every value handed over device to device was taken
        _receive?.Flush();
        ExchangeShared();
        double stepLoss = ShareLoss(total);
        return Report(started, microBatches, options.Step, stepLoss);
    }

    /// <summary>
    /// The micro-batches' forward passes without gradients (an evaluation): the sum of the last stage's losses, on every
    /// stage of the pipeline.
    /// </summary>
    public double Evaluate(int microBatches, Func<int, Tensor>? input, Func<int, Tensor, Tensor>? loss, Func<int, IDisposable?>? around = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(microBatches);
        Required(input, loss);
        double total = 0;
        using (Autograd.NoGrad())
        {
            for (int m = 0; m < microBatches; m++)
            {
                using var scope = new TensorScope();
                using var inside = around?.Invoke(m);
                var x = IsFirst ? input!(m) : _receive!.ReceiveTensor(_receive.Peer, PipelineBoundary.ActivationTag + (2 * m), Device);
                var y = Run(x, Modules.Count, checkpointing: false);
                if (IsLast)
                {
                    total += loss!(m, y).Item();
                }
                else
                {
                    _send!.SendTensor(y, _send.Peer, PipelineBoundary.ActivationTag + (2 * m));
                }
            }

            _send?.Flush();
        }

        return ShareLoss(total);
    }

    /// <summary>
    /// The stage's last module (the output head on the last stage), for a loss that runs it itself (<see cref="PipelineStepOptions.HeadInLoss"/>).
    /// </summary>
    public Module Head => Modules[^1];

    /// <summary>Sums <paramref name="values"/> over this pipeline's stages in place, in stage order, so every stage holds the same bits.</summary>
    public void SumStages(Span<float> values)
    {
        if (Layout.Stages == 1)
        {
            return;
        }

        var bytes = MemoryMarshal.AsBytes(values);
        int first = Layout.RankOf(0, Replica);
        if (!IsFirst)
        {
            Transport.Send(first, bytes, SumTag);
            Transport.Receive(first, bytes, SumTag);
            return;
        }

        var other = new float[values.Length];
        for (int s = 1; s < Layout.Stages; s++)
        {
            Transport.Receive(Layout.RankOf(s, Replica), MemoryMarshal.AsBytes(other.AsSpan()), SumTag);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] += other[i];
            }
        }

        for (int s = 1; s < Layout.Stages; s++)
        {
            Transport.Send(Layout.RankOf(s, Replica), bytes, SumTag);
        }
    }

    /// <summary>
    /// Clips the step's gradients to <paramref name="maxNorm"/> by their norm over the whole pipeline (every stage's, a tied
    /// table counted once; 0: no clipping) and updates <paramref name="optimizer"/> (this stage's parameters).
    /// </summary>
    public void ClipAndStep(Optimizer optimizer, float maxNorm)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        if (maxNorm > 0f)
        {
            var withGrad = Parameters.Where(p => p.Grad is not null).ToList();
            Span<float> squares = [0f];
            if (withGrad.Count > 0)
            {
                using var sum = Tensor.Zeros([1], Device);
                foreach (var p in withGrad.Where(p => !_notOwned.Contains(p)))
                {
                    sum.Backend.SumSquares(p.Grad!.Storage, sum.Storage, p.Size);
                }

                squares[0] = sum.Item();
            }

            SumStages(squares);
            double norm = Math.Sqrt(squares[0]);
            if (norm > maxNorm && norm > 0)
            {
                float factor = (float)(maxNorm / norm);
                foreach (var p in withGrad)
                {
                    p.Backend.Affine(p.Grad!.Storage, p.Grad.Storage, p.Size, factor, 0f);
                }
            }
        }

        optimizer.Step();
    }

    /// <summary>
    /// Serves the first stage's generation (a stage other than the first): each step's hidden state arrives with the
    /// decoding context's position, runs through this stage with a context of its own kept in step (its own key/value
    /// cache), and goes to the next stage, or, from the last, its logits back to the first. Returns when the first stage
    /// stops (<see cref="StopServing"/>).
    /// </summary>
    public void Serve(CancellationToken cancellationToken = default)
    {
        if (IsFirst)
        {
            throw new InvalidOperationException("The first stage runs the generation (a generator over Network); the other stages serve it.");
        }

        Network.Eval();
        int first = Layout.RankOf(0, Replica);
        try
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var header = _receive!.ReceiveGeneration(_receive.Peer);
                if (header.Command == PipelineBoundary.CommandStop)
                {
                    _send?.SendGeneration(header, null, _send.Peer);
                    return;
                }

                using var noGrad = Autograd.NoGrad();
                using var scope = new TensorScope();
                var x = _receive.ReceiveTensor(_receive.Peer, PipelineBoundary.InferenceTag + 4, Device);
                Tensor y;
                if (header.Command == PipelineBoundary.CommandCached)
                {
                    var context = Mirror(header);
                    context.LastPositionOnly = IsLast && header.LastPositionOnly;     // a middle stage passes every position on
                    y = Network.ForwardCached(x, context);
                }
                else
                {
                    y = Network.Forward(x);
                }

                if (IsLast)
                {
                    _receive.SendTensor(y, first, PipelineBoundary.ReturnTag);
                }
                else
                {
                    _send!.SendGeneration(header, y, _send.Peer);
                }
            }
        }
        finally
        {
            _serving?.Dispose();
            _serving = null;
        }
    }

    /// <summary>Tells the other stages to stop serving (the first stage, once its generation is done). Disposing the first stage does it too.</summary>
    public void StopServing()
    {
        if (!IsFirst || _send is null || _stopped)
        {
            return;
        }

        _stopped = true;
        _send.SendGeneration(new GenerationHeader(PipelineBoundary.CommandStop, 0, 0, 0, false, 0, 0, null), null, _send.Peer);
    }

    /// <summary>Releases the boundaries and the replicas' collectives (the modules stay the model's).</summary>
    public void Dispose()
    {
        if (_send is { Generated: true })
        {
            StopServing();
        }

        _receive?.Dispose();
        _send?.Dispose();
        Replicas?.Dispose();
        _serving?.Dispose();
    }

    // The serving stage's own context, in step with the first stage's: made for its shape, reset or cut back with it.
    private DecodingContext Mirror(in GenerationHeader header)
    {
        if (_serving is null || _serving.Batch != header.Batch || _serving.Capacity != header.Capacity || (int)_serving.Format != header.Format
            || _serving.SlidingWindowSteps != header.SlidingWindowSteps)
        {
            _serving?.Dispose();
            _serving = new DecodingContext(Device, header.Batch, header.Capacity, (KeyValueFormat)header.Format) { SlidingWindowSteps = header.SlidingWindowSteps };
        }

        if (header.Length != _serving.Length)
        {
            if (header.Length == 0)
            {
                _serving.Reset();
            }
            else if (header.Length < _serving.Length)
            {
                _serving.Truncate(header.Length);
            }
            else
            {
                throw new InvalidOperationException($"Stage {Stage} holds {_serving.Length} positions and the first stage {header.Length}: the stages' caches are out of step.");
            }
        }

        if (!(header.RowStarts ?? []).SequenceEqual(_serving.RowStarts ?? []))
        {
            _serving.SetRowStarts(header.RowStarts);
        }

        return _serving;
    }

    private void Required(Func<int, Tensor>? input, Func<int, Tensor, Tensor>? loss)
    {
        if (IsFirst && input is null)
        {
            throw new ArgumentNullException(nameof(input), "The first stage takes the micro-batches' input.");
        }

        if (IsLast && loss is null)
        {
            throw new ArgumentNullException(nameof(loss), "The last stage computes the micro-batches' loss.");
        }
    }

    // Micro-batch m's forward pass on this stage: its root (the loss on the last stage, the boundary's send elsewhere).
    private Tensor ForwardOne(int m, Func<int, Tensor>? input, Func<int, Tensor, Tensor>? loss, PipelineStepOptions options)
    {
        Tensor x;
        if (IsFirst)
        {
            x = input!(m);
        }
        else
        {
            _receive!.MicroBatch = m;
            x = _receive.Forward(_receive.Token);
        }

        bool head = IsLast && options.HeadInLoss;
        var y = Run(x, head ? Modules.Count - 1 : Modules.Count, options.Checkpointing);
        if (IsLast)
        {
            return loss!(m, y);
        }

        _send!.MicroBatch = m;
        return _send.Forward(y);
    }

    private Tensor Run(Tensor x, int count, bool checkpointing)
    {
        for (int i = 0; i < count; i++)
        {
            x = checkpointing && Autograd.IsEnabled && Modules[i] is DecoderBlock ? Modules[i].ForwardCheckpointed(x) : Modules[i].Forward(x);
        }

        return x;
    }

    // Whether the schedule runs every forward before any backward and the backward passes in reverse: each micro-batch's
    // scope then closes inside the ones opened before it, so its activations go as soon as its backward pass ran.
    private static bool Nested(IReadOnlyList<PipelineWork> order, int microBatches)
    {
        if (order.Count != 2 * microBatches)
        {
            return false;
        }

        for (int m = 0; m < microBatches; m++)
        {
            if (order[m] != new PipelineWork(PipelineWorkKind.Forward, m) || order[microBatches + m] != new PipelineWork(PipelineWorkKind.Backward, microBatches - 1 - m))
            {
                return false;
            }
        }

        return true;
    }

    // Every stage holding a tensor (a tied table) sums its gradient with the others' in stage order: the same bits on each.
    private void ExchangeShared()
    {
        foreach (var (global, holders, tensor) in _shared)
        {
            int count = tensor.Size;
            var mine = ArrayPool<float>.Shared.Rent(Math.Max(1, count));
            var sum = ArrayPool<float>.Shared.Rent(Math.Max(1, count));
            var other = ArrayPool<float>.Shared.Rent(Math.Max(1, count));
            try
            {
                var grad = tensor.GradStorage();
                tensor.Backend.Download(grad, mine.AsSpan(0, count));
                foreach (int s in holders.Where(s => s != Stage))
                {
                    Transport.Send(Layout.RankOf(s, Replica), MemoryMarshal.AsBytes(mine.AsSpan(0, count)), SharedTag + global);
                }

                var total = sum.AsSpan(0, count);
                total.Clear();
                foreach (int s in holders)
                {
                    ReadOnlySpan<float> part = mine.AsSpan(0, count);
                    if (s != Stage)
                    {
                        Transport.Receive(Layout.RankOf(s, Replica), MemoryMarshal.AsBytes(other.AsSpan(0, count)), SharedTag + global);
                        part = other.AsSpan(0, count);
                    }

                    if (s == holders[0])
                    {
                        part.CopyTo(total);
                    }
                    else
                    {
                        for (int i = 0; i < count; i++)
                        {
                            total[i] += part[i];
                        }
                    }
                }

                tensor.Backend.Upload(total, grad);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(mine);
                ArrayPool<float>.Shared.Return(sum);
                ArrayPool<float>.Shared.Return(other);
            }
        }
    }

    // The last stage's loss to every stage of its pipeline.
    private double ShareLoss(double loss)
    {
        if (Layout.Stages == 1)
        {
            return loss;
        }

        Span<byte> bytes = stackalloc byte[sizeof(double)];
        int last = Layout.RankOf(Layout.Stages - 1, Replica);
        if (IsLast)
        {
            BinaryPrimitives.WriteDoubleLittleEndian(bytes, loss);
            for (int s = 0; s < Layout.Stages - 1; s++)
            {
                Transport.Send(Layout.RankOf(s, Replica), bytes, LossTag);
            }

            return loss;
        }

        Transport.Receive(last, bytes, LossTag);
        return BinaryPrimitives.ReadDoubleLittleEndian(bytes);
    }

    private PipelineStepResult Report(long started, int microBatches, long step, double loss)
    {
        var wall = Stopwatch.GetElapsedTime(started);
        var blocked = (_receive?.Blocked ?? TimeSpan.Zero) + (_send?.Blocked ?? TimeSpan.Zero);
        long sent = (_receive?.BytesSent ?? 0) + (_send?.BytesSent ?? 0), received = (_receive?.BytesReceived ?? 0) + (_send?.BytesReceived ?? 0);
        double bubble = wall > TimeSpan.Zero ? blocked / wall : 0;
        double ideal = Schedule.IdealBubble(Layout.Stages, microBatches);
        if (Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.PipelineStep(new PipelineStepped(Schedule.Name, Rank, Stage, Layout.Stages, Replica, microBatches, step, wall, blocked, bubble, ideal, sent, received,
                Layout.Stages > 1 ? Staging : null));
        }

        return new PipelineStepResult(loss, wall, blocked, bubble, ideal, sent, received);
    }
}
