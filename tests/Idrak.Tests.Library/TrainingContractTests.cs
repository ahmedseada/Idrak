// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Datasets;
using Idrak.Datasets.Abstractions;
using Idrak.Inference;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak.Training.Abstractions;

// Plan 33, C3: training and serving contracts as interfaces. An optimizer's saved state its own interface (no exception
// caught to learn an optimizer keeps none), Adam's fused path held by a sealed update instead of a type comparison,
// schedulers that set their first rate themselves, schedule kinds that make the scheduler, the trainer's loss, metrics
// and shardable batch sources, the engine's model kinds and the module's tensors.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TrainingContractGroup =
    [
        ("training contracts: an optimizer that keeps state is an IStatefulOptimizer (SGD, Adam, AdamW, 8-bit AdamW, grouped, host) and one that keeps none is not, RequireState naming it; a state entry's parameter is read from its name as a span (\"m.1\", a group's \"group1.velocity.0\", a host optimizer's \"1.m.0\"), allocating nothing", OptimizerStateContract),
        ("training contracts: Adam's fused clipping and update holds for a subclass (Step is sealed: its update is Adam's), its values Adam's and AdamW's bit for bit; 8-bit AdamW scales the gradients as its step reads them (IScaledGradientStep), so clipping leaves them as they are", AdamFusedForSubclass),
        ("training contracts: a scheduler sets the first rate when made, with nothing for a derived class to call (ILearningRateScheduler); a schedule kind makes one from a request over the optimizer it is given (ILearningRateSchedule.Create) and refuses a bad setting before (Check)", SchedulerContract),
        ("training contracts: the trainer's loss and metrics are ILoss and IMetric, and one's own give the losses and metrics the function forms give, bit for bit; a DataLoader is an IShardableBatchSource (seeded unless it shuffles without a seed; steps shared over a sample source, not over a stream; each rank's share of an epoch by position)", TrainerContracts),
        ("training contracts: the engine hosts any IEngineModel (one written without the EngineModel base) and disposes it; a module is an ITensorOwner whose MoveTo moves what Parameters and Buffers list", EngineAndTensorOwner),
    ];

    internal static void OptimizerStateContract(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(OptimizerStateContract)))
        {
            return;
        }

        using var a = Tensor.Persistent(new float[4], [4], Device.Cpu, requiresGrad: true);
        using var b = Tensor.Persistent(new float[3], [3], Device.Cpu, requiresGrad: true);
        using var adam = new Adam([a, b], 0.01f);
        using var adamW = new AdamW([a, b], 0.01f);
        using var sgd = new Sgd([a, b], 0.1f, momentum: 0.9f);
        using var eightBit = new AdamW8Bit([a, b]);
        using var grouped = new GroupedOptimizer(new Adam([a], 0.01f), new Sgd([b], 0.1f));
        using var host = new HostOptimizer([a, b], ps => new AdamW(ps, 0.01f));
        using var plain = new PlainStep([a, b]);
        Check(new IOptimizer[] { adam, adamW, sgd, eightBit, grouped, host }.All(o => o is IStatefulOptimizer) && (IOptimizer)plain is not IStatefulOptimizer,
            "the library's optimizers keep a saved state; one that keeps none does not say it does");
        var refused = Failure<NotSupportedException>(() => plain.RequireState());
        Check(refused.Message.Contains(nameof(PlainStep), StringComparison.Ordinal) && refused.Message.Contains(nameof(IStatefulOptimizer), StringComparison.Ordinal),
            $"RequireState names the optimizer and the contract it lacks: {refused.Message}");

        IStatefulOptimizer single = adam, groups = grouped, copies = host;
        Check(single.StateParameter("m.1") == 1 && single.StateParameter("v.0") == 0 && single.StateParameter("step") is null && single.StateParameter("m.2") is null
              && single.StateParameter("m.x") is null, "the library's names: the parameter after the last dot, below the parameters' count");
        Check(groups.StateParameter("group0.m.0") == 0 && groups.StateParameter("group1.velocity.0") == 1 && groups.StateParameter("group0.step") is null
              && groups.StateParameter("group01.m.0") is null && groups.StateParameter("group2.m.0") is null && groups.StateParameter("m.0") is null,
            "a group's entry: its group's parameter, counted after the groups before it");
        Check(copies.StateParameter("1.m.0") == 1 && copies.StateParameter("0.step") == 0 && copies.StateParameter("2.m.0") is null && copies.StateParameter("step") is null,
            "a host optimizer's entry: the parameter whose optimizer it is");

        int found = 0;
        void Lookups()
        {
            for (int i = 0; i < 5; i++)
            {
                found += single.StateParameter("m.1") ?? 0;
                found += groups.StateParameter("group1.velocity.0") ?? 0;
                found += copies.StateParameter("1.m.0") ?? 0;
            }
        }

        for (int i = 0; i < 2_000; i++)
        {
            Lookups();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1_000; i++)
        {
            Lookups();
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0 && found > 0, $"15,000 state entries read from their names allocated {allocated} bytes once warm{(allocated == 0 ? "" : $" ({AllocatedTypes(Lookups)})")}");
    }

    internal static void AdamFusedForSubclass(Device device)
    {
        // Five clipped steps of a squared loss: the values each optimizer leaves.
        float[] Train(Func<IReadOnlyList<Tensor>, Optimizer> create)
        {
            var random = new Random(3);
            using var p = Tensor.Persistent([.. Enumerable.Range(0, 300).Select(_ => random.NextSingle() * 2f - 1f)], [300], device, requiresGrad: true);
            using var target = Tensor.Persistent(new float[300], [300], device);
            using var optimizer = create([p]);
            for (int step = 0; step < 5; step++)
            {
                optimizer.ZeroGrad();
                using var scope = new TensorScope();
                (p - target).Square().Sum().Backward();
                optimizer.ClipAndStep(0.5f);
            }

            return p.ToArray();
        }

        Check(typeof(Adam).GetMethod(nameof(Adam.Step))!.IsFinal, "Adam's update cannot be changed by a subclass (Step is sealed)");
        Check(Train(ps => new CountingAdam(ps, 0.01f)).SequenceEqual(Train(ps => new Adam(ps, 0.01f))), "a subclass of Adam: Adam's values, bit for bit");
        Check(Train(ps => new CountingAdam(ps, 0.01f, weightDecay: 0.01f, decoupled: true)).SequenceEqual(Train(ps => new AdamW(ps, 0.01f, weightDecay: 0.01f))),
            "a subclass with decoupled decay: AdamW's values, bit for bit");

        // 8-bit AdamW scales the gradients as its step reads them: clipping leaves them and sets the factor.
        using var w = Tensor.Persistent([.. Enumerable.Range(0, 64).Select(i => i / 8f)], [64], device, requiresGrad: true);
        using var eightBit = new AdamW8Bit([w]);
        using (var scope = new TensorScope())
        {
            w.Square().Sum().Backward();
        }

        float[] gradients = w.Grad!.ToArray();
        double norm = eightBit.ClipGradientNorm(0.1f);
        float scale = ((IScaledGradientStep)eightBit).GradientScale;
        Check(norm > 0.1 && w.Grad!.ToArray().SequenceEqual(gradients) && scale == (float)(0.1 / norm),
            $"clipping 8-bit AdamW's gradients (norm {norm:G4}) leaves them and gives its step the factor {scale:G4}");
        eightBit.Step();
        Check(((IScaledGradientStep)eightBit).GradientScale == 1f, "the step takes the factor and resets it to 1");
    }

    internal static void SchedulerContract(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(SchedulerContract)))
        {
            return;
        }

        using var p = Tensor.Persistent([0f], [1], Device.Cpu, requiresGrad: true);
        using var optimizer = new Sgd([p], 1f);
        ILearningRateScheduler cosine = new CosineAnnealing(optimizer, 10, warmupEpochs: 3);
        Check(optimizer.LearningRate == 1f / 4 && cosine.BaseLearningRate == 1f && cosine.Epoch == 0 && ReferenceEquals(cosine.Optimizer, optimizer),
            $"made, the scheduler set the first epoch's rate: {optimizer.LearningRate}");
        cosine.Step();
        Check(optimizer.LearningRate == 2f / 4 && cosine.Epoch == 1, $"a step sets the next epoch's rate: {optimizer.LearningRate}");

        optimizer.LearningRate = 3f;
        var halving = new Halving(optimizer);
        Check(optimizer.LearningRate == 3f && halving.BaseLearningRate == 3f, "a scheduler of one's own sets its first rate with nothing to call from its constructor");
        halving.Step();
        halving.Step();
        Check(optimizer.LearningRate == 0.75f, $"and steps from it: {optimizer.LearningRate}");

        optimizer.LearningRate = 2f;
        var linear = LearningRateSchedules.Get(LearningRateSchedules.Linear).Create(new LearningRateScheduleRequest(optimizer, 10, new LearningRateScheduleSettings(0.2f)));
        Check(ReferenceEquals(linear.Optimizer, optimizer) && linear.BaseLearningRate == 2f && optimizer.LearningRate == 2f / 3,
            $"a kind makes the scheduler of the request's optimizer, from its rate: {optimizer.LearningRate}");
        var refused = Failure<ArgumentOutOfRangeException>(() => LearningRateSchedules.Get(LearningRateSchedules.WarmupStableDecay).Check(new LearningRateScheduleSettings(DecayFraction: -0.5f)));
        Check(refused.Message.Contains("[0, 1]", StringComparison.Ordinal), "wsd refuses a decay share below 0 before a run");
    }

    internal static void TrainerContracts(Device device)
    {
        // Three epochs of the callbacks' regression: the losses and metrics a trainer reports.
        (double[] Losses, double[] Metrics, double?[] Validation) Fit(bool own)
        {
            var (train, validation) = CallbackData(device);
            using var model = BuiltMlpFor3(device);
            using var trainer = own ? new Trainer(model, new SquaredError(), p => new Adam(p, 0.01f)) { Metrics = { new RootMeanSquare() } }
                : new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.01f)) { Metrics = { Metric.RootMeanSquaredError } };
            var history = trainer.Fit(train, 3, validation);
            return ([.. history.Epochs.Select(e => e.Loss)], [.. history.Epochs.Select(e => e.Metrics["rmse"])], [.. history.Epochs.Select(e => e.ValidationLoss)]);
        }

        var functions = Fit(own: false);
        var interfaces = Fit(own: true);
        Check(functions.Losses.SequenceEqual(interfaces.Losses) && functions.Metrics.SequenceEqual(interfaces.Metrics) && functions.Validation.SequenceEqual(interfaces.Validation),
            $"an ILoss and IMetric of one's own: the function forms' losses ({string.Join(", ", functions.Losses)}) and metrics, bit for bit");
        Check(functions.Metrics.Zip(functions.Losses).All(m => m.First > 0 && Math.Abs(m.First * m.First - m.Second) < 1e-4 * Math.Max(1, m.Second)),
            "the metric's Finish takes the epoch's mean (RMSE: the root of the MSE)");

        var data = SmallRegression(40, 3);
        IBatchSource shuffled = new DataLoader(data, 8, shuffle: true, device: device), seeded = new DataLoader(data, 8, shuffle: true, device: device, seed: 2);
        IBatchSource ordered = new DataLoader(data, 8, device: device), streamed = new DataLoader(new DatasetStream(data), 8, device: device);
        Check(shuffled is IShardableBatchSource { Seeded: false } && seeded is IShardableBatchSource { Seeded: true } && ordered is IShardableBatchSource { Seeded: true },
            "a loader is seeded unless it shuffles without a seed");
        Check(ordered is IShardableBatchSource { StepShares: not null } && streamed is IShardableBatchSource { StepShares: null } && ((IShardableBatchSource)ordered).Device == device,
            "steps are shared over a sample source, not over a stream (its length is not known ahead); the batches' device");
        var shares = Enumerable.Range(0, 2).Select(r => ((IShardableBatchSource)ordered).Shard(r, 2).Select(batch =>
        {
            using (batch)
            {
                return batch.Features.ToArray();
            }
        }).SelectMany(f => f).ToArray()).ToArray();
        float[] whole = [.. ordered.SelectMany(batch =>
        {
            using (batch)
            {
                return batch.Features.ToArray();
            }
        })];
        Check(shares[0].Length == 20 * 3 && shares[1].Length == 20 * 3 && shares[0].AsSpan(0, 3).SequenceEqual(whole.AsSpan(0, 3)) && shares[1].AsSpan(0, 3).SequenceEqual(whole.AsSpan(3, 3)),
            "rank r of 2 takes the samples at positions 2k + r of the epoch");
    }

    internal static void EngineAndTensorOwner(Device device)
    {
        if (FirstRun(nameof(EngineAndTensorOwner)))
        {
            var direct = new DirectModel();
            var engine = InferenceEngine.Create().Add("direct", direct).BuildAsync().GetAwaiter().GetResult();
            try
            {
                Check(engine.Model<DirectModel>("direct").AskAsync("hi").Result == "copy 1:hi" && direct.Loads == 1 && engine.Models.Single().Kind == "direct",
                    "a model kind written on IEngineModel alone: loaded, lent to a request, its kind in the status");
            }
            finally
            {
                engine.DisposeAsync().AsTask().Wait();
            }

            Check(direct.Disposed, "the engine disposes it");
        }

        using var model = BuiltMlpFor3(Device.Cpu);
        ITensorOwner owner = model;
        owner.MoveTo(device);
        Check(owner.Parameters().Count() == 4 && owner.Parameters().Concat(owner.Buffers()).All(t => t.Device == device),
            $"MoveTo through the contract moves every parameter to {device}");
    }

    // An optimizer that keeps no state: plain steps.
    internal sealed class PlainStep(IEnumerable<Tensor> parameters) : Optimizer(parameters, 0.1f)
    {
        public override void Step()
        {
        }
    }

    // A subclass of Adam that counts its steps: its update stays Adam's.
    internal sealed class CountingAdam(IEnumerable<Tensor> parameters, float learningRate, float weightDecay = 0f, bool decoupled = false)
        : Adam(parameters, learningRate, weightDecay: weightDecay)
    {
        public int Steps { get; private set; }

        protected override bool DecoupledWeightDecay => decoupled;

        public override void ClipAndStep(float maxNorm)
        {
            Steps++;
            base.ClipAndStep(maxNorm);
        }
    }

    // A schedule of one's own: half the rate every epoch.
    internal sealed class Halving(IOptimizer optimizer) : LearningRateScheduler(optimizer, (epoch, rate) => rate / (1 << epoch));

    internal sealed class SquaredError : ILoss
    {
        public Tensor Compute(Tensor predictions, Tensor targets) => (predictions - targets).Square().Mean();
    }

    internal sealed class RootMeanSquare : IMetric
    {
        public string Name => "rmse";

        public Tensor BatchMean(Tensor predictions, Tensor targets) => (predictions - targets).Square().Mean();

        public double Finish(double mean) => Math.Sqrt(mean);
    }

    // A model kind on the contract alone: its copies are strings.
    internal sealed class DirectModel : IEngineModel<string>
    {
        private IEngineHost<string>? _host;

        public string Kind => "direct";

        public EngineHosting Hosting { get; } = new() { SharedCopies = true };

        public int Loads { get; private set; }

        public bool Disposed { get; private set; }

        public void Attach(IEngineHost<string> host) => _host = host;

        public string LoadCopy() => $"copy {++Loads}";

        public void UnloadCopy(string copy)
        {
        }

        public ModelDescription Describe(string copy) => new(_host!.Name, Kind, 0, Device.Cpu, null);

        public async Task<string> AskAsync(string text)
        {
            using var lease = await _host!.AcquireAsync(CancellationToken.None);
            lease.Completed(1, text.Length, TimeSpan.Zero);
            return $"{lease.Copy}:{text}";
        }

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            return ValueTask.CompletedTask;
        }
    }
}
