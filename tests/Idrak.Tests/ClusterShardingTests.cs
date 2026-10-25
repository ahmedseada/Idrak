// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cluster;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Training;
using Idrak.Text;
using Idrak.Training;
using Idrak.Training.Abstractions;

// Plan 23, phase C6: sharding (ZeRO-style: the optimizer's state, the gradients, the weights), and the items handed to it:
// background staging of a finished bucket (C4's open item), receive flow control on the real transports and device
// collectives over transports that move device memory (C5's), and the shared-memory doorbell (the owner's measurements).
// Every run is an in-process cluster of devices carved from the device under test, or real transports in one process, or
// child processes of this program.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterShardingGroup =
    [
        ("cluster: sharded optimizer state equals the unsharded weights after 3 steps: optimizer, gradients and parameters shards bit for bit with naive (3 ranks, Adam, buckets sent during the backward pass), within float rounding with ring and clipping; each rank's optimizer holds its own parameters' moments only", ClusterShardedEqualsUnsharded),
        ("cluster: a model whose training state is over the cap of one fake GPU trains across four with Shard = parameters, each rank holding its share (InUse per rank against the share, not growing over steps), the weights as the unsharded run's", ClusterShardedOverCap),
        ("cluster: a sharded run resumes at the same world size: the pretrainer's checkpoint holds every rank's share of the optimizer state, and the resumed run's losses and weights are the uninterrupted run's", ClusterShardedResume),
        ("cluster: a late joiner in a sharded run takes its share (the weights and momentum moved from the ranks that held them): the run equals the run without the fault", ClusterShardedLateJoin),
        ("cluster: a bucket finished during the backward pass is staged to the host in the background, waited for by the reducing thread: the gradients bit-identical to the synchronous copy, the training thread not waiting for the copies", ClusterAsyncStaging),
        ("cluster: receive flow control bounds a slow receiver's memory on tcp, shared-memory and pcie-host (credits per peer and tag), an awaited send beyond the credit waits without a thread, and every rank sending before it receives still completes", ClusterReceiveCredit),
        ("cluster: collectives over pcie-host keep each piece on the device and sum it there, bit-identical to staging through the host; the telemetry says which", ClusterDeviceCollectives),
        ("cluster: pipeline placement with a shard is refused before anything moves, naming both settings", ClusterShardedPipelineRefused),
        ("cluster: shared memory between two processes wakes in far less than a timer tick (a doorbell, not a sleep) and moves 16 MB messages within a sane fraction of a memory copy", ClusterSharedMemorySpeed),
    ];

    // A classifier of several layers for the sharding tests.
    private static Sequential ShardModel(Device device, int width = 32, int layers = 3)
    {
        var init = new Random(11);
        var model = new Sequential { new Linear(2, width, device: device, random: init), new ReLU() };
        for (int i = 1; i < layers; i++)
        {
            model.Add(new Linear(width, width, device: device, random: init));
            model.Add(new ReLU());
        }

        model.Add(new Linear(width, 3, device: device, random: init));
        return model;
    }

    private static void ClusterShardedEqualsUnsharded(Device device)
    {
        var data = Spirals(30, 3, 1);                                        // 90 samples: 3 steps of 30 an epoch
        (float[] Weights, float[] Curve, string[] Moments) Fit(ClusterRank rank, ClusterOptions options, float? clip, bool adamW)
        {
            using var model = ShardModel(rank.Device);
            using Optimizer optimizer = adamW ? new AdamW(model.Parameters(), 0.01f) : new Adam(model.Parameters(), 0.01f);
            var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options, MaxGradientNorm = clip };
            var history = trainer.Fit(new DataLoader(data, 30 / rank.Collectives.WorldSize, shuffle: true, device: rank.Device, seed: 7), 1);
            var moments = optimizer.RequireState().SaveState().Keys.Where(k => k.StartsWith("m.", StringComparison.Ordinal)).Order(StringComparer.Ordinal).ToArray();
            return ([.. model.Parameters().SelectMany(p => p.ToArray())], [.. history.Epochs.Select(e => (float)e.Loss)], moments);
        }

        var naive = DpOptions() with { Collective = "naive", BucketBytes = 512 };   // about 10 buckets of the model's values
        var topology = DpTopology(device, 3);
        var plain = DpRun(topology, naive, rank => Fit(rank, naive, null, adamW: false));
        using (var probe = ShardModel(device))
        {
            var owners = ShardOwners.Assign([.. probe.Parameters()], 3);
            foreach (var mode in new[] { ShardMode.Optimizer, ShardMode.Gradients, ShardMode.Parameters })
            {
                var sharded = naive with { Shard = mode };
                var results = DpRun(topology, sharded, rank => Fit(rank, sharded, null, adamW: false));
                for (int r = 0; r < 3; r++)
                {
                    Check(results[r].Weights.SequenceEqual(plain[r].Weights) && results[r].Curve.SequenceEqual(plain[r].Curve),
                        $"Shard = {mode}, naive, rank {r}: the weights and losses after 3 steps bit for bit the unsharded run's (largest gap {Gap(results[r].Weights, plain[r].Weights):G3})");
                    var own = Enumerable.Range(0, owners.Length).Where(i => owners[i] == r).Select(i => $"m.{i}").Order(StringComparer.Ordinal);
                    Check(results[r].Moments.SequenceEqual(own), $"Shard = {mode}: rank {r}'s optimizer holds its own parameters' moments only: [{string.Join(", ", results[r].Moments)}], its own [{string.Join(", ", own)}]");
                }
            }

            Check(plain[0].Moments.Length == owners.Length, $"unsharded, every rank holds every parameter's moments ({plain[0].Moments.Length} of {owners.Length})");
        }

        // Ring sums each piece in another order than the unsharded buckets' slices, and the global norm is the ranks' squares
        // summed: within float rounding, with clipping and AdamW.
        var ring = DpOptions() with { Collective = "ring", BucketBytes = 512 };
        var ringPlain = DpRun(DpTopology(device, 2), ring, rank => Fit(rank, ring, 0.5f, adamW: true));
        var ringSharded = DpRun(DpTopology(device, 2), ring with { Shard = ShardMode.Optimizer }, rank => Fit(rank, ring with { Shard = ShardMode.Optimizer }, 0.5f, adamW: true));
        Check(ringSharded[0].Weights.SequenceEqual(ringSharded[1].Weights), "ring, sharded: both ranks end with the same weights, bit for bit");
        AssertClose(ringPlain[0].Weights, ringSharded[0].Weights, 1e-5f, "ring, AdamW clipped to 0.5: the sharded weights against the unsharded");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    ring with clipping: the sharded weights {Gap(ringPlain[0].Weights, ringSharded[0].Weights):G3} from the unsharded after 3 steps"));

        // What a sharded run refuses, naming the setting.
        DpRun(DpTopology(device, 2), DpOptions(), rank =>
        {
            using var p = Tensor.Persistent(new float[4], [4], rank.Device, requiresGrad: true);
            using var sgd = new Sgd([p], 0.1f);
            ThrowsNaming<ArgumentException>(() => GradientSyncs.Create(rank.Collectives, sgd, DpOptions() with { Shard = ShardMode.Optimizer, Compression = "top-k", TopK = 0.1 }), "top-k with a shard", "IDRAK_CLUSTER_COMPRESSION");
            ThrowsNaming<ArgumentException>(() => GradientSyncs.Create(rank.Collectives, sgd, DpOptions() with { Shard = ShardMode.Optimizer, GradientSync = "all-reduce" }), "all-reduce with a shard", "sharded");
            ThrowsNaming<ArgumentException>(() => GradientSyncs.Create(rank.Collectives, sgd, DpOptions() with { Shard = ShardMode.Optimizer, LocalSgdEvery = 2, GradientSync = "local-sgd" }), "local SGD with a shard", "IDRAK_CLUSTER_SHARD");
            using (var sharded = GradientSyncs.Create(rank.Collectives, sgd, DpOptions() with { Shard = ShardMode.Optimizer }))
            using (var whole = GradientSyncs.Create(rank.Collectives, sgd, DpOptions()))
            {
                Check(sharded is IShardedGradientSync && whole is not IShardedGradientSync && rank.Collectives is IShardedHostValueCollectives,
                    "the sharded synchronisation makes the update (IShardedGradientSync), the all-reduce leaves it to the optimizer; the ranks reduce uneven pieces on the host");
            }

            return 0;
        });
        Check(GradientSyncs.Find("sharded") is not null && GradientSyncs.Origin("sharded") == Overrides.Library, $"Idrak.Network registers sharded: {GradientSyncs.Describe()}");
    }

    // Each rank's memory as each epoch ends (the step's tensors released), and, while storages are tracked, what it holds:
    // the device's live storages by size and owner (a parameter's name where Name knows it, else the allocating method).
    private sealed class MemoryProbe(Device device) : ITrainerCallback
    {
        public List<long> InUse { get; } = [];

        public List<string> Held { get; } = [];

        public Func<Storage, string?>? Name { get; set; }

        public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
        {
            InUse.Add(device.Backend.GetMemoryUsage().InUse);
            if (StorageTracking.Enabled)
            {
                Held.Add(StorageTracking.Describe(device.Backend.LiveStorages(Name)));
            }
        }
    }

    // Names a model's parameter storages and their gradients' ("3.weight [256, 256]"), for the memory listings.
    private static Func<Storage, string?> ParameterNames(Module model)
    {
        var names = new Dictionary<Storage, string>(ReferenceEqualityComparer.Instance);
        var parameters = model.Parameters().ToList();
        return storage =>
        {
            for (int i = 0; i < parameters.Count; i++)
            {
                var p = parameters[i];
                if (ReferenceEquals(p.Storage, storage))
                {
                    return $"parameter {i} {Tensor.FormatShape(p.Shape)}";
                }

                if (p.Grad is { } g && ReferenceEquals(g.Storage, storage))
                {
                    return $"gradient of parameter {i} {Tensor.FormatShape(p.Shape)}";
                }
            }

            return names.GetValueOrDefault(storage);
        };
    }

    private static Sequential WideModel(Device device)
    {
        var init = new Random(13);
        var model = new Sequential { new Linear(2, 256, device: device, random: init), new ReLU() };
        for (int i = 0; i < 7; i++)
        {
            model.Add(new Linear(256, 256, device: device, random: init));
            model.Add(new ReLU());
        }

        model.Add(new Linear(256, 3, device: device, random: init));
        return model;
    }

    private static void ClusterShardedOverCap(Device device)
    {
        var data = Spirals(64, 3, 2);                                        // 192 samples: 6 steps of 32 an epoch over 4 ranks
        long values;
        using (var probe = WideModel(Device.Cpu))
        {
            values = probe.Parameters().Sum(p => (long)p.Size);
        }

        long training = 4 * values * sizeof(float);                          // weights, gradients and Adam's two moments
        long cap = 4L << 20;
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {values:N0} values: {values * 4 / 1048576.0:F2} MiB of weights, {training / 1048576.0:F2} MiB of training state with Adam; each fake GPU's cap {cap / 1048576.0:F0} MiB"));
        Check(training > cap, "the training state is over one fake GPU's cap");

        (float[] Weights, List<long> InUse, long Peak, long Owned, List<string> Held) Fit(ClusterRank rank, ClusterOptions options, int epochs)
        {
            using var model = WideModel(rank.Device);
            using var optimizer = new Adam(model.Parameters(), 0.002f);
            var memory = new MemoryProbe(rank.Device) { Name = ParameterNames(model) };
            var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options, Callbacks = { memory } };
            trainer.Fit(new DataLoader(data, 8, shuffle: true, device: rank.Device, seed: 5), epochs);
            var owners = ShardOwners.Assign([.. model.Parameters()], rank.Collectives.WorldSize);
            long owned = model.Parameters().Where((_, i) => owners[i] == rank.Rank).Sum(p => (long)p.Size);
            return ([.. model.Parameters().SelectMany(p => p.ToArray())], memory.InUse, rank.Device.Backend.GetMemoryUsage().Peak, owned, memory.Held);
        }

        // One fake GPU alone cannot train it (a hard cap: CPU, Vulkan, HIP; a CUDA carve spills to host memory instead).
        if (device.Type == DeviceType.Cpu)
        {
            var alone = SimulatedDevices.Carve(device, cap, slot: 63);
            bool failed = false;
            try
            {
                using var model = WideModel(alone);
                using var optimizer = new Adam(model.Parameters(), 0.002f);
                new Trainer(model, optimizer, Losses.CrossEntropy).Fit(new DataLoader(data, 32, shuffle: true, device: alone, seed: 5), 1);
            }
            catch (Exception e) when (e is ResourceLimitExceededException || e.InnerException is ResourceLimitExceededException)
            {
                failed = true;
                Console.WriteLine($"    unsharded on one fake GPU: {e.Message}");
            }

            Check(failed, "unsharded, the model does not train on one fake GPU of that cap");
        }

        var capped = ClusterTopology.Create([new ClusterNodeSpec("box", null, [.. Enumerable.Repeat(new ClusterDeviceSpec(device.ToString(), cap), 4)])], null, null);
        var options = DpOptions() with { Collective = "naive", Shard = ShardMode.Parameters, BucketBytes = 256L << 10 };
        // Every storage tracked (plan 23, C8b): a bound that fails names what each rank's device held at each epoch's end.
        (float[] Weights, List<long> InUse, long Peak, long Owned, List<string> Held)[] sharded;
        using (StorageTracking.Track())
        {
            sharded = DpRun(capped, options, rank => Fit(rank, options, 3));
        }

        var plain = DpRun(DpTopology(device, 4), DpOptions() with { Collective = "naive", BucketBytes = 256L << 10 }, rank => Fit(rank, DpOptions() with { Collective = "naive", BucketBytes = 256L << 10 }, 3));
        for (int r = 0; r < 4; r++)
        {
            var (_, inUse, peak, owned, held) = sharded[r];
            long share = 3 * owned * sizeof(float);                          // its master copies and their two moments
            long slack = 256L << 10;                                         // the loader's and the epoch's sums, the scratch of a bucket
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    rank {r}: owns {owned:N0} values ({owned * 100.0 / values:F1}%), share {share / 1048576.0:F2} MiB, in use at each epoch's end [{string.Join(", ", inUse.Select(b => (b / 1048576.0).ToString("F2", CultureInfo.InvariantCulture)))}] MiB, peak {peak / 1048576.0:F2} MiB"));
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    rank {r} holds at the last epoch's end: {held.LastOrDefault()}"));
            Check(inUse.Count == 3 && inUse.All(b => b >= share && b <= share + slack),
                $"rank {r} holds its share: {string.Join(", ", inUse)} B against {share} B (+{slack} B); the device's live storages at each epoch's end: "
                + string.Join(" | ", held.Select((h, e) => $"epoch {e + 1}: {h}")));
            Check(inUse.Skip(1).All(b => b == inUse[0]), $"rank {r}'s device memory does not grow over the epochs (rule 77): {string.Join(", ", inUse)}; held: {string.Join(" | ", held)}");
            Check(peak <= cap, $"rank {r}'s peak {peak} B within its cap {cap} B");
            Check(sharded[r].Weights.SequenceEqual(plain[r].Weights), $"rank {r}: the weights after 3 epochs bit for bit the unsharded run's (largest gap {Gap(sharded[r].Weights, plain[r].Weights):G3})");
        }
    }

    private static void ClusterShardedResume(Device device)
    {
        string folder = TempFolder("cluster-sharded-resume");
        try
        {
            string corpus = PretrainData("tokens.bin");
            foreach (var mode in new[] { ShardMode.Optimizer, ShardMode.Parameters })
            {
                var options = DpOptions() with { Collective = "ring", Shard = mode };
                PretrainingOptions Options(int steps, ICollectives cluster) =>
                    DpPretraining(corpus, steps, cluster, options) with { Optimizer = "adamw", BatchTokens = 32, GradientAccumulation = 2, SaveEvery = 2 };
                string whole = Path.Combine(folder, $"{mode}-whole"), split = Path.Combine(folder, $"{mode}-split");
                var straight = DpRun(DpTopology(device, 2), options, rank => DpPretrain(rank.Device, Options(4, rank.Collectives), rank.Rank == 0 ? whole : Path.Combine(folder, $"{mode}-whole-1")));
                DpRun(DpTopology(device, 2), options, rank => DpPretrain(rank.Device, Options(2, rank.Collectives), rank.Rank == 0 ? split : Path.Combine(folder, $"{mode}-split-1")));
                string latest = Pretrainer.LatestCheckpoint(split)!;
                var checkpoint = Pretrainer.ReadCheckpoint(latest);
                var resumed = DpRun(DpTopology(device, 2), options, rank => DpPretrain(rank.Device, Options(4, rank.Collectives), rank.Rank == 0 ? split : Path.Combine(folder, $"{mode}-split-1"),
                    resume: checkpoint, model: latest));
                AssertClose(straight[0].Losses, resumed[0].Losses, 1e-6f, $"Shard = {mode}: the resumed run's losses against the uninterrupted run's");
                AssertClose(straight[0].Weights, resumed[0].Weights, 1e-6f, $"Shard = {mode}: the resumed run's last checkpoint against the uninterrupted run's");

                // The checkpoint's optimizer state is whole: every trained parameter's moments, from both ranks' shares.
                var state = ReadTensors(Directory.GetFiles(latest, "optimizer*.safetensors").Single());
                int moments = state.Keys.Count(k => k.StartsWith("m.", StringComparison.Ordinal));
                int trained;
                using (var loaded = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
                {
                    trained = loaded.Network.Parameters().Count(p => p.RequiresGrad);
                }

                Check(moments == trained && state.ContainsKey("step"), $"Shard = {mode}: the checkpoint holds {moments} parameters' moments of {trained}, and the step count");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterShardedLateJoin(Device device)
    {
        // 150 samples, 3 ranks of batch 10: 5 steps an epoch; rank 2 joins at step 7, in the second epoch.
        using var late = Scripted("test-late-sharded", new Fault(FaultKind.LateJoin, 0, [2], TimeSpan.Zero, 7));
        var data = Spirals(50, 3, 1);
        float[] Fit(ClusterRank rank, ClusterOptions options)
        {
            using var model = ShardModel(rank.Device, 16, 2);
            using var optimizer = new Sgd(model.Parameters(), 0.1f, momentum: 0.9f);
            new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options }.Fit(new DataLoader(data, 10, shuffle: true, device: rank.Device, seed: 7), 3);
            return [.. model.Parameters().SelectMany(p => p.ToArray())];
        }

        foreach (var mode in new[] { ShardMode.Optimizer, ShardMode.Parameters })
        {
            var none = DpOptions() with { Collective = "ring", Shard = mode, Simulation = new SimulationOptions { Failures = "none", Seed = 5 } };
            var faulty = none with { Simulation = none.Simulation with { Failures = "test-late-sharded" } };
            var whole = DpRun(DpTopology(device, 3), none, rank => Fit(rank, none));
            var joined = DpRun(DpTopology(device, 3), faulty, rank => Fit(rank, faulty));
            Check(joined.All(w => w.SequenceEqual(joined[0])), $"Shard = {mode}: every rank ends with the same weights after the late join");
            AssertClose(whole[0], joined[0], 1e-5f, $"Shard = {mode}: the late-join run's weights against the run without the fault");
        }
    }

    // Pinned slots as a device has them, here host arrays whose copies complete only after a delay (a device's copy queued
    // behind its work), so the background path runs without a GPU.
    private sealed class SlowStaging(int slots, int floats, TimeSpan delay) : ISlotStaging
    {
        private readonly float[][] _slots = [.. Enumerable.Range(0, slots).Select(_ => new float[floats])];
        private readonly long[] _ready = new long[slots];

        public int Downloads;

        public int SlotFloats => floats;

        public Span<float> Slot(int slot) => _slots[slot];

        public void Download(Storage source, int offset, int slot, int count) => Download(source, offset, slot, 0, count);

        public void Download(Storage source, int offset, int slot, int slotOffset, int count)
        {
            source.Backend.DownloadRange(source, offset, _slots[slot].AsSpan(slotOffset, count));   // the values as they are now (stream order)
            Volatile.Write(ref _ready[slot], Stopwatch.GetTimestamp() + (long)(delay.TotalSeconds * Stopwatch.Frequency));
            Interlocked.Increment(ref Downloads);
        }

        public void Upload(int slot, Storage destination, int offset, int count) => throw new NotSupportedException("the test's staging only downloads");

        public void Wait(int slot)
        {
            while (Stopwatch.GetTimestamp() < Volatile.Read(ref _ready[slot]))
            {
                Thread.Sleep(0);
            }
        }

        public void Dispose()
        {
        }
    }

    private static void ClusterAsyncStaging(Device device)
    {
        var data = Spirals(40, 3, 3);
        var delay = TimeSpan.FromMilliseconds(2);
        var stagings = new List<SlowStaging>();
        (float[] Weights, double Launch, double Waited, bool Staged, int Buckets) Run(ClusterRank rank, ClusterOptions options)
        {
            using var model = ShardModel(rank.Device, 64, 3);
            using var optimizer = new Sgd(model.Parameters(), 0.05f, momentum: 0.9f);
            using var sync = GradientSyncs.Create(rank.Collectives, optimizer, options);
            sync.Start();
            var loader = new DataLoader(data, 20, shuffle: true, device: rank.Device, seed: 9);
            var reduction = (GradientReduction)sync;
            foreach (var batch in loader.Shard(rank.Rank, rank.Collectives.WorldSize).Take(4))
            {
                using var _ = batch;
                using var scope = new TensorScope();
                var loss = Losses.CrossEntropy(model.Forward(batch.Features), batch.Targets);
                optimizer.ZeroGrad();
                using (sync.Overlap(batch.Size))
                {
                    loss.Backward();
                }

                sync.Reduce(batch.Size, loss.Item());
                optimizer.Step();
            }

            return ([.. model.Parameters().SelectMany(p => p.ToArray())], reduction.LaunchTime.TotalMilliseconds, reduction.StagingWait.TotalMilliseconds, reduction.Staged, reduction.Layout.Count);
        }

        var previous = GradientReduction.CreateStaging;
        try
        {
            GradientReduction.CreateStaging = (backend, slots, floats) =>
            {
                var staging = new SlowStaging(slots, floats, delay);
                lock (stagings)
                {
                    stagings.Add(staging);
                }

                return staging;
            };
            var options = DpOptions() with { Collective = "ring", BucketBytes = 2048 };
            var staged = DpRun(DpTopology(device, 2), options, rank => Run(rank, options));
            var direct = DpRun(DpTopology(device, 2), options with { AsyncStaging = false }, rank => Run(rank, options with { AsyncStaging = false }));
            for (int r = 0; r < 2; r++)
            {
                Check(staged[r].Staged && !direct[r].Staged, $"rank {r}: buckets staged in the background with it on, on the training thread with it off");
                Check(staged[r].Weights.SequenceEqual(direct[r].Weights), $"rank {r}: the weights after 4 steps bit for bit the synchronous copy's (largest gap {Gap(staged[r].Weights, direct[r].Weights):G3})");
            }

            Check(stagings.Sum(s => s.Downloads) > 0, "the staging's copies were queued");
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {staged[0].Buckets} buckets a step, a copy completing {delay.TotalMilliseconds:F0} ms after it is queued: the training thread spent {staged[0].Launch:F2} ms queuing them over 4 steps, "
                + $"the reducing thread waited {staged[0].Waited:F2} ms for the copies (time a copy on the training thread would have held the backward pass)"));
            // The training thread must not sit out the copies: its queuing (a backend's submit included: Vulkan flushes
            // its batched commands to hand a bucket over) stays well under what the copies themselves take.
            double copies;
            lock (stagings)
            {
                copies = stagings.Sum(s => s.Downloads) / 2.0 * delay.TotalMilliseconds;   // two ranks' stagings: rank 0's share
            }

            Check(staged[0].Waited > 0 && staged[0].Launch < copies / 2,
                $"the copies' time is the reducing thread's, not the training thread's: the training thread spent {staged[0].Launch:F2} ms queuing against {copies:F0} ms of copies (the reducing thread waited {staged[0].Waited:F2} ms)");
        }
        finally
        {
            GradientReduction.CreateStaging = previous;
        }
    }

    // `world` ranks of one machine joined by `transport`, every rank connecting on a thread of its own, with `options`.
    private static ITransport[] CreditGroup(string transport, int world, ClusterOptions options)
    {
        var topology = RealTopology([world]);
        var provider = Transports.Get(transport);
        return ConnectAll(world, r => provider.Connect(new ClusterContext(topology, r, options)));
    }

    private static void ClusterReceiveCredit(Device device)
    {
        if (!FirstRun(nameof(ClusterReceiveCredit)))
        {
            return;
        }

        const int Size = 16 << 10, Count = 120;
        foreach (string name in new[] { "tcp", "shared-memory", "pcie-host" })
        {
            long Peak(int inFlight, long? budget)
            {
                var group = CreditGroup(name, 2, RealOptions(TimeSpan.FromSeconds(30)) with { InFlight = inFlight, ReceiveBytes = budget });
                try
                {
                    var sending = Task.Run(() =>
                    {
                        for (int i = 0; i < Count; i++)
                        {
                            group[0].Send(1, Enumerable.Repeat((byte)i, Size).ToArray(), 5);
                        }
                    });
                    var buffer = new byte[Size];
                    for (int i = 0; i < Count; i++)
                    {
                        Thread.Sleep(1);                                     // a slow receiver
                        int n = group[1].Receive(0, buffer, 5);
                        Check(n == Size && buffer[0] == (byte)i && buffer[^1] == (byte)i, $"{name}: message {i} arrived whole and in order");
                    }

                    Check(sending.Wait(TimeSpan.FromSeconds(30)), $"{name}: the sender finished");
                    return ((FramedTransport)group[1]).PeakQueuedBytes;
                }
                finally
                {
                    foreach (var t in group)
                    {
                        t.Dispose();
                    }
                }
            }

            long bounded = Peak(2, 24 << 10), loose = Peak(10_000, null);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {name}: {Count} messages of {Size / 1024} KiB to a slow receiver: at most {bounded / 1024.0:F0} KiB held with 2 in flight and a 24 KiB budget, {loose / 1024.0:F0} KiB without a bound"));
            Check(bounded <= Size * 2 && bounded <= (24 << 10) + Size, $"{name}: the receiver held at most the credit: {bounded} B");
            Check(loose > bounded * 4, $"{name}: without the bound a fast sender fills the receiver: {loose} B against {bounded} B");
        }

        // Every rank sends a message far larger than the budget to every other rank, under two tags, before it receives
        // anything: a message is always let go when none of its peer and tag is outstanding, so nobody waits in a circle.
        foreach (string name in new[] { "tcp", "shared-memory", "pcie-host" })
        {
            var group = CreditGroup(name, 3, RealOptions(TimeSpan.FromSeconds(30)) with { InFlight = 1, ReceiveBytes = 4 << 10, SharedMemoryBytes = 64 << 10 });
            try
            {
                const int Large = 1 << 20;
                var ranks = Enumerable.Range(0, 3).Select(me => Task.Factory.StartNew(() =>
                {
                    var t = group[me];
                    for (int to = 0; to < 3; to++)
                    {
                        if (to != me)
                        {
                            t.Send(to, Enumerable.Repeat((byte)(me + 1), Large).ToArray(), 6);
                            t.Send(to, [(byte)me], 7);
                        }
                    }

                    var buffer = new byte[Large];
                    for (int from = 0; from < 3; from++)
                    {
                        if (from != me)
                        {
                            bool small = t.Receive(from, buffer, 7) == 1 && buffer[0] == from;
                            bool large = t.Receive(from, buffer, 6) == Large && buffer[Large - 1] == from + 1;
                            if (!small || !large)
                            {
                                throw new InvalidDataException($"rank {me} got the wrong bytes from rank {from}");
                            }
                        }
                    }
                }, TaskCreationOptions.LongRunning)).ToArray();
                Check(Task.WaitAll(ranks, TimeSpan.FromSeconds(60)), $"{name}: every rank sent 1 MiB to every other before receiving (a 4 KiB budget, 1 in flight) and all arrived");
                foreach (var task in ranks.Where(t => t.IsFaulted))
                {
                    Check(false, $"{name}: {task.Exception!.InnerException!.Message}");
                }

                // An awaited send beyond the credit waits without a thread, and goes once the peer takes the one before.
                var first = group[0].SendAsync(1, new byte[8], 8);
                var second = group[0].SendAsync(1, new byte[8], 8);
                Check(first.IsCompleted && !second.IsCompleted, $"{name}: the second awaited send waits for its credit ({first.IsCompleted}, {second.IsCompleted})");
                group[1].Receive(0, new byte[8], 8);
                Check(second.AsTask().Wait(TimeSpan.FromSeconds(10)), $"{name}: it went once the first was taken");
                group[1].Receive(0, new byte[8], 8);
            }
            finally
            {
                foreach (var t in group)
                {
                    t.Dispose();
                }
            }
        }
    }

    // Every collective on device buffers, as one array of this rank's results.
    private static float[] DeviceWork(ICollectives c)
    {
        var backend = c.Device.Backend;
        int world = c.WorldSize, me = c.Rank;
        var random = new Random(500 + me);
        float[] Values(int count) => [.. Enumerable.Range(0, count).Select(_ => (float)((random.NextDouble() - 0.5) * (1 + me) * Math.Pow(10, random.Next(-3, 4))))];
        var all = new List<float>();
        void Run(float[] input, int outputLength, Action<Storage, Storage> op)
        {
            var buffer = backend.Allocate(Math.Max(1, input.Length), zeroed: false);
            var output = outputLength == 0 ? buffer : backend.Allocate(outputLength, zeroed: true);
            try
            {
                backend.Upload(input, buffer);
                op(buffer, output);
                var result = new float[outputLength == 0 ? input.Length : outputLength];
                backend.Download(output, result);
                all.AddRange(result);
            }
            finally
            {
                buffer.Release();
                if (!ReferenceEquals(output, buffer))
                {
                    output.Release();
                }
            }
        }

        var values = Values(1_031);
        Run(values, 0, (b, _) => c.AllReduce(b, values.Length, ReduceOp.Sum));
        Run(Values(7), 0, (b, _) => c.AllReduce(b, 7, ReduceOp.Max));                 // a maximum: through the host on every path
        Run(Values(5), world * 5, (b, o) => c.AllGather(b, 5, o));
        Run(Values(world * 9), 9, (b, o) => c.ReduceScatter(b, 9, o, ReduceOp.Sum));
        Run(Values(11), 0, (b, _) => c.Broadcast(b, 11, world - 1));
        c.Barrier();
        return [.. all];
    }

    private static void ClusterDeviceCollectives(Device device)
    {
        (float[][] Results, List<CollectiveCompleted> Seen) Run(string transport, bool onDevice)
        {
            var options = RealOptions(TimeSpan.FromSeconds(30)) with { DeviceCollectives = onDevice };
            var topology = RealTopology([3]);
            var group = CreditGroup(transport, 3, options);
            var metrics = new MetricsRecorder(TelemetryLevel.Network);
            var results = new float[3][];
            using (Telemetry.Subscribe(metrics))
            {
                var ranks = Enumerable.Range(0, 3).Select(r => Task.Factory.StartNew(() =>
                {
                    using var collectives = Collectives.Get("ring").Create(group[r], SimulatedDevices.Carve(device, null, 40 + r), new ClusterContext(topology, r, options));
                    results[r] = DeviceWork(collectives);
                }, TaskCreationOptions.LongRunning)).ToArray();
                try
                {
                    Task.WaitAll(ranks, TimeSpan.FromSeconds(60));
                }
                finally
                {
                    foreach (var t in group)
                    {
                        t.Dispose();
                    }
                }
            }

            return (results, [.. metrics.Collectives]);
        }

        var direct = Run("pcie-host", onDevice: true);
        var staged = Run("pcie-host", onDevice: false);
        var tcp = Run("tcp", onDevice: true);
        for (int r = 0; r < 3; r++)
        {
            Check(direct.Results[r].SequenceEqual(staged.Results[r]) && direct.Results[r].SequenceEqual(tcp.Results[r]),
                $"rank {r}: the device path's results bit for bit the host-staged ones (pcie-host staged and tcp)");
        }

        string Paths(List<CollectiveCompleted> seen) => string.Join(", ", seen.Where(e => e.Rank == 0 && e.Operation != "barrier").Select(e => $"{e.Operation}:{e.Staging}"));
        var sums = direct.Seen.Where(e => e.Operation is "all-reduce" or "all-gather" or "reduce-scatter" or "broadcast").ToList();
        Check(sums.Count(e => e.Staging == "device") == 3 * 4 && sums.Count(e => e.Staging == "host") == 3,
            $"pcie-host with device collectives: the sum, gather, scatter and broadcast on the device, the maximum through the host: {Paths(direct.Seen)}");
        Check(staged.Seen.Where(e => e.Operation != "barrier").All(e => e.Staging == "host") && tcp.Seen.Where(e => e.Operation != "barrier").All(e => e.Staging == "host"),
            $"turned off, and over tcp, every call is staged through the host: {Paths(staged.Seen)}; {Paths(tcp.Seen)}");
    }

    private static void ClusterShardedPipelineRefused(Device device)
    {
        var options = DpOptions() with { Placement = "pipeline", Shard = ShardMode.Optimizer };
        DpRun(DpTopology(device, 2), options, rank =>
        {
            using var model = ShardModel(rank.Device);
            ThrowsNaming<NotSupportedException>(() => PipelineStage.Place(model, rank.Collectives, options).Dispose(), "a pipeline with a shard",
                "IDRAK_CLUSTER_PLACEMENT=pipeline", "IDRAK_CLUSTER_SHARD=optimizer");
            return 0;
        });
    }

    // The child of the shared-memory speed test: rank 0 times empty round trips after an idle gap (each side's reader has
    // gone to its doorbell) and 16 MB messages with an empty answer, and writes the medians.
    internal static int SharedMemorySpeedChild(string folder)
    {
        using var joined = LocalhostCluster.Join();
        var transport = joined.Rank.Transport;
        int me = transport.Rank;
        var latencies = new List<double>();
        for (int i = 0; i < 25; i++)
        {
            if (me == 0)
            {
                Thread.Sleep(20);                                            // both readers idle: they block on their doorbells
                long start = Stopwatch.GetTimestamp();
                transport.Send(1, [], 7);
                transport.Receive(1, [], 7);
                latencies.Add(Stopwatch.GetElapsedTime(start).TotalSeconds / 2);
            }
            else
            {
                transport.Receive(0, [], 7);
                transport.Send(0, [], 7);
            }
        }

        const int Bytes = 16 << 20;
        var buffer = new byte[Bytes];
        var rates = new List<double>();
        for (int i = 0; i < 6; i++)
        {
            if (me == 0)
            {
                long start = Stopwatch.GetTimestamp();
                transport.Send(1, buffer, 8);
                transport.Receive(1, [], 9);
                rates.Add(Bytes / Stopwatch.GetElapsedTime(start).TotalSeconds);
            }
            else
            {
                transport.Receive(0, buffer, 8);
                transport.Send(0, [], 9);
            }
        }

        transport.Barrier();
        if (me == 0)
        {
            latencies.Sort();
            var warm = rates.Skip(1).Order().ToList();
            File.WriteAllText(Path.Combine(folder, "speed.txt"), string.Create(CultureInfo.InvariantCulture,
                $"{joined.Plan.Used.Single()} {latencies[latencies.Count / 2]} {latencies[^1]} {rates[0]} {warm[warm.Count / 2]}"));
        }

        return 0;
    }

    private static void ClusterSharedMemorySpeed(Device device)
    {
        if (!FirstRun(nameof(ClusterSharedMemorySpeed)))
        {
            return;
        }

        // One timer tick as this machine has it (what a sleep of a millisecond waits), and a memory copy's rate.
        var ticks = new double[15];
        for (int i = 0; i < ticks.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            Thread.Sleep(1);
            ticks[i] = Stopwatch.GetElapsedTime(start).TotalSeconds;
        }

        Array.Sort(ticks);
        double tick = ticks[ticks.Length / 2];
        var from = new byte[16 << 20];
        var to = new byte[16 << 20];
        new Random(1).NextBytes(from);
        var copies = new double[7];
        for (int i = 0; i < copies.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            from.AsSpan().CopyTo(to);
            copies[i] = from.Length / Stopwatch.GetElapsedTime(start).TotalSeconds;
        }

        Array.Sort(copies);
        double memcpy = copies[copies.Length / 2];

        string folder = TempFolder("cluster-shm-speed");
        try
        {
            string path = Path.Combine(folder, "topology.json");
            File.WriteAllText(path, RealTopology([2]).ToJson());
            var (file, prefix) = LocalhostCluster.ProgramOf(typeof(Tests).Assembly);
            var launch = new LocalhostLaunch(file, [.. prefix, "--cluster-child", "shm-speed", folder])
            {
                Environment = new Dictionary<string, string> { ["IDRAK_CLUSTER_TIMEOUT"] = "60" }, Limit = TimeSpan.FromSeconds(120),
            };
            var result = LocalhostCluster.Run(path, new ClusterOptions { Timeout = TimeSpan.FromSeconds(60) }, launch);
            Check(result.Succeeded, $"the two processes ran: {result.Failure} " + string.Join("; ", result.Ranks.Select(r => $"rank {r.Rank} exit {r.ExitCode}: {string.Join(" | ", r.Errors.TakeLast(3))}")));
            var parts = File.ReadAllText(Path.Combine(folder, "speed.txt")).Split(' ');
            double latency = double.Parse(parts[1], CultureInfo.InvariantCulture), worst = double.Parse(parts[2], CultureInfo.InvariantCulture);
            double cold = double.Parse(parts[3], CultureInfo.InvariantCulture), warm = double.Parse(parts[4], CultureInfo.InvariantCulture);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {parts[0]} between two processes: latency after an idle gap {latency * 1e6:F0} us median ({worst * 1e6:F0} us worst) against a timer tick of {tick * 1e6:F0} us; "
                + $"16 MB messages {warm / 1048576:F0} MiB/s warm, {cold / 1048576:F0} MiB/s cold, a memory copy {memcpy / 1048576:F0} MiB/s"));
            Check(parts[0] == "shared-memory", $"the two ranks of one machine joined over shared memory: {parts[0]}");
            Check(latency < tick / 2, $"a message after an idle gap arrives in far less than a timer tick: {latency * 1e6:F0} us against {tick * 1e6:F0} us");
            Check(warm >= memcpy / 20, $"16 MB messages move within a sane fraction of a memory copy: {warm / 1048576:F0} MiB/s against {memcpy / 1048576:F0} MiB/s");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
