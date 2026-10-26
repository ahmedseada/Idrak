// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Cluster;
using Idrak.Cluster;
using Idrak.Cluster.Abstractions;
using Idrak.Datasets;
using Idrak.Datasets.Abstractions;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

// Plan 33, C6: the contracts that replace type checks on the library's layers, adapters, scalers and cluster parts. The
// library's own types declare them (so the layers packed, offloaded, checkpointed, adapted, saved and measured are the
// same as before), and a layer or cell of one's own opts in through them.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] LayerContractGroup =
    [
        ("layer contracts: the library's types declare the contracts its type checks became (linear layers adapt, pack, tie, name and stage their weights; embeddings, norms and position tables name theirs; attention layers say what their caches hold and which batching they run; position encodings, routed and boundary layers, decoder blocks, position limits and owned position biases; LoRA and DoRA in peft's layout; the scalers a package stores; the topology file, measured links and live failures), and a tied layer names no weights", LayerContractsDeclared),
        ("layer contracts: a recurrent cell of one's own (IRecurrentCell) runs in RecurrentModule's loop: its gates size the weights, its initial bias and its steps give the hand-computed outputs; the LSTM's and GRU's cells are fused, and a GRU keeps its candidate bias apart as its cell asks (IScaledHiddenBiasCell)", RecurrentCellOfOnesOwn),
        ("layer contracts: a linear layer of one's own (IAdaptableLinear) takes LoRA from AddLora by its name, trains through the adapter, merges it with MergeLora and lifts it off with DisableAdapters; the adapter's peft view (IPeftAdapter) keeps its tensors when rescaled", AdaptableLinearOfOnesOwn),
    ];

    internal static void LayerContractsDeclared(Device device)
    {
        var linear = new Linear(4, 3, device: device, random: new Random(1));
        Check(linear is IAdaptableLinear and IPackableLinear and ITiedWeights and INamedWeights and IOffloadableLayer, "Linear adapts, packs, ties, names and stages its weights");
        Check(((INamedWeights)linear).NamedWeights().Select(w => w.Name).SequenceEqual(["weight", "bias"]), "a Linear names its weight and bias");
        var embedding = new Embedding(10, 4, device, new Random(2));
        var tied = Linear.Tied(embedding);
        Check(!((INamedWeights)tied).NamedWeights().Any() && ReferenceEquals(((ITiedWeights)tied).WeightsOf, embedding), "a tied head names no weights and says whose table it reads");
        Check(embedding is INamedWeights && embedding is IOffloadableLayer staged && !staged.StagedParameters().Any(), "an embedding names its table and stages none of it");
        Check(new RMSNorm(4, device: device) is INamedWeights && new LayerNorm(4, device: device) is INamedWeights, "the norms name their weights");
        Check(new PositionEmbedding(Tensor.Zeros([8, 4], device)) is INamedWeights and IPositionLimit { MaxPositions: 8 }, "a position table names its weights and limits the positions");

        var attention = new CausalSelfAttention(8, 2, 2, 4, null, 8, device: device, random: new Random(4)) { SlidingWindow = 4 };
        Check(attention is IAttentionCacheInfo { KvHeads: 2, HeadDim: 4, SlidingWindow: 4 } and IPositionLimit { MaxPositions: 8 }, "causal attention says what its cache holds and its position limit");
        var heads = new MultiHeadAttention(8, 2, device: device, random: new Random(5));
        var info = (IAttentionCacheInfo)heads;
        Check(info is { KvHeads: 2, HeadDim: 4, SlidingWindow: null } && !info.SupportsRows(device.Backend) && !info.SupportsPacking() && !info.SupportsSpans(),
            "multi-head attention keeps every head's keys and runs no packed or row batches");
        Check(new PositionalEncoding(8, 8, device) is IPositionAware, "a sinusoidal encoding numbers positions itself");
        Check(!PackedSequences.Supports(new Sequential(new PositionalEncoding(8, 8, device), attention)), "a position encoding keeps a model from packing");
        var experts = new MixtureOfExperts(8, 16, 4, 2, device: device, random: new Random(6));
        Check(experts is IRoutedLayer and IHostRoundTrip && ((IRoutedLayer)experts).PartsPerPosition().Single(p => p.Times == 2).Part is FeedForward,
            "a mixture runs its router and two experts per position and reads its routing on the host");
        Check(typeof(IHostRoundTrip).IsAssignableFrom(typeof(PipelineBoundary)), "a pipeline boundary goes beyond the device");
        Check(typeof(IDecoderBlock).IsAssignableFrom(typeof(DecoderBlock)) && typeof(ICheckpointable).IsAssignableFrom(typeof(IDecoderBlock)), "a decoder block is checkpointed");
        Check(typeof(IOwnedPositionBias).IsAssignableFrom(typeof(AttentionPositionBias)), "a position bias with parameters is owned through its module");

        Check(typeof(IPeftAdapter).IsAssignableFrom(typeof(LoraAdapter)) && typeof(IPeftAdapter).IsAssignableFrom(typeof(DoraAdapter)), "LoRA and DoRA are in peft's layout");
        float[] rows = [1f, 2f, 3f, 5f];
        Check(StandardScaler.Fit(rows, 2) is ISavableScaler { EntryKind: Idrak.Inference.PackageEntryKind.StandardScaler }
              && MinMaxScaler.Fit(rows, 2) is ISavableScaler { EntryKind: Idrak.Inference.PackageEntryKind.MinMaxScaler }, "the scalers store themselves under their entry kinds");
        Check(typeof(IOverridingTopology).IsAssignableFrom(typeof(ClusterTopology)) && typeof(ISavableTopology).IsAssignableFrom(typeof(ClusterTopology)), "the topology file has overrides and writes itself");
        Check(LinkProfiles.Find(ClusterTopology.SameDevice) is IDeviceMeasuredLink && LinkProfiles.Find("ethernet-1g") is not IDeviceMeasuredLink, "only the measured links are measured on a device");
        Check(new LiveFailures() is ILiveFailureModel { Pending: 0 }, "the live failures are a live model");
    }

    // A plain recurrent cell: h_t = tanh(x_t·W + b + h_{t-1}·U), one gate and one state, its bias starting at 0.5.
    internal sealed class TanhCell(int hiddenSize) : IRecurrentCell
    {
        public int Steps { get; private set; }

        public int Gates => 1;

        public int States => 1;

        public float[] InitialBias(int hidden) => Enumerable.Repeat(0.5f, Gates * hidden).ToArray();

        public Tensor[] Step(Tensor projected, IReadOnlyList<Tensor> state, RecurrentWeights weights)
        {
            Steps++;
            return [(projected + state[0].MatMul(weights.HiddenWeight)).Tanh().Reshape(projected.Shape[0], hiddenSize)];
        }
    }

    internal sealed class TanhLayer(TanhCell cell, int inputSize, int hiddenSize, Device device)
        : RecurrentModule(cell, inputSize, hiddenSize, true, device, new Random(7));

    internal static void RecurrentCellOfOnesOwn(Device device)
    {
        const int inputs = 3, hidden = 4, steps = 5;
        var cell = new TanhCell(hidden);
        using var layer = new TanhLayer(cell, inputs, hidden, device);
        Check(layer.Gates == 1 && layer.Weights[0].InputWeight.Shape.SequenceEqual([inputs, hidden]) && layer.Weights[0].HiddenBias is null && !layer.ScalesLastHiddenBias,
            "the cell's gates size the weights; no recurrent bias kept apart");
        float[] w = layer.Weights[0].InputWeight.ToArray(), u = layer.Weights[0].HiddenWeight.ToArray(), b = layer.Weights[0].Bias.ToArray();
        Check(b.All(v => v == 0.5f), "the cell's initial bias");
        var x = RandomArray(new Random(8), steps * inputs);
        float[] output;
        using (var input = Tensor.From(x, [1, steps, inputs], device))
        using (var y = layer.Forward(input))
        {
            output = y.ToArray();
        }

        var h = new float[hidden];
        float worst = 0f;
        for (int t = 0; t < steps; t++)
        {
            var next = new float[hidden];
            for (int j = 0; j < hidden; j++)
            {
                float z = b[j];
                for (int i = 0; i < inputs; i++)
                {
                    z += x[t * inputs + i] * w[i * hidden + j];
                }

                for (int k = 0; k < hidden; k++)
                {
                    z += h[k] * u[k * hidden + j];
                }

                next[j] = MathF.Tanh(z);
                worst = Math.Max(worst, Math.Abs(next[j] - output[t * hidden + j]));
            }

            h = next;
        }

        Check(cell.Steps == steps && worst < 1e-5f, $"the cell's steps give the hand-computed outputs (largest difference {worst})");
        using var lstm = new LSTM(inputs, hidden, device: device);
        using var gru = new GRU(inputs, hidden, true, false, candidateBias: true, device: device);
        using var plainGru = new GRU(inputs, hidden, device: device);
        Check(lstm.Cell is IFusedRecurrentCell { Gates: 4, States: 2 } && !lstm.ScalesLastHiddenBias && lstm.Weights[0].HiddenBias is null, "the LSTM's cell is fused, four gates, two states");
        Check(gru.Cell is IFusedRecurrentCell and IScaledHiddenBiasCell { KeepsHiddenBias: true, Gates: 3 } && gru.ScalesLastHiddenBias && gru.CandidateBias,
            "a GRU with its candidate bias keeps it apart, as its cell asks");
        Check(plainGru.Cell is IScaledHiddenBiasCell { KeepsHiddenBias: false } && plainGru.ScalesLastHiddenBias && !plainGru.CandidateBias, "a GRU without it still scales that bias");
    }

    // A linear layer of one's own: y = x·W (+ the adapter's term), W [in, out].
    internal sealed class OwnLinear : Module, IAdaptableLinear
    {
        private readonly Tensor _weight;

        public OwnLinear(int inFeatures, int outFeatures, Device device)
        {
            (InFeatures, OutFeatures) = (inFeatures, outFeatures);
            _weight = CreateParameter(RandomArray(new Random(9), inFeatures * outFeatures, 0.5f), [inFeatures, outFeatures], device);
        }

        public int InFeatures { get; }

        public int OutFeatures { get; }

        public Device Device => _weight.Device;

        public ILinearAdapter? Adapter { get; set; }

        public float[] WeightValues() => _weight.ToArray();

        public Tensor BaseProduct(Tensor input) => input.MatMul(_weight);

        public void MergeAdapter()
        {
            if (Adapter is not { } adapter)
            {
                return;
            }

            using (Autograd.NoGrad())
            using (new TensorScope())
            {
                _weight.Load(adapter.Merge(this, _weight).ToArray());
            }

            Adapter = null;
        }

        public override IEnumerable<Tensor> Parameters() => Adapter is { } a ? [_weight, .. a.Parameters] : [_weight];

        protected override Tensor ForwardCore(Tensor input)
        {
            var product = BaseProduct(input);
            return Adapter is { } a ? a.Forward(this, input, product) : product;
        }
    }

    internal static void AdaptableLinearOfOnesOwn(Device device)
    {
        var own = new OwnLinear(4, 3, device) { Name = "q" };
        var other = new OwnLinear(3, 2, device) { Name = "o" };
        using var model = new Sequential(own, other);
        int added = model.AddLora(rank: 2, alpha: 4, targets: l => l.Name is "q", freezeBase: true, random: new Random(10));
        Check(added == 1 && own.Adapter is LoraAdapter { Rank: 2 } && other.Adapter is null, "AddLora adapts the layer of one's own its target names");
        var x = RandomArray(new Random(11), 2 * 4);
        float[] Run()
        {
            using var input = Tensor.From(x, [2, 4], device);
            using var y = model.Forward(input);
            return y.ToArray();
        }

        var before = Run();
        var lora = (LoraAdapter)own.Adapter!;
        lora.B.Load(RandomArray(new Random(12), 2 * 3, 0.3f));
        var adapted = Run();
        Check(adapted.Zip(before).Any(p => Math.Abs(p.First - p.Second) > 1e-4f), "the adapter's term changes the output");
        using (model.DisableAdapters())
        {
            Check(own.Adapter is null && Run().Zip(before).All(p => Math.Abs(p.First - p.Second) < 1e-6f), "DisableAdapters lifts the adapter off");
        }

        Check(ReferenceEquals(own.Adapter, lora), "and puts it back");
        var peft = (IPeftAdapter)lora;
        var rescaled = peft.WithScale(peft.Scale * 2f);
        Check(rescaled is LoraAdapter && ReferenceEquals(rescaled.A, lora.A) && ReferenceEquals(rescaled.B, lora.B) && rescaled.Scale == lora.Scale * 2f && rescaled.Magnitude is null
              && peft.PeftType == "LORA", "the peft view keeps the tensors when rescaled");
        Check(model.MergeLora() == 1 && own.Adapter is null, "MergeLora folds the adapter into the layer of one's own");
        var merged = Run();
        float worst = adapted.Zip(merged).Max(p => Math.Abs(p.First - p.Second));
        Check(worst < 1e-4f, $"the merged layer gives the adapted outputs (largest difference {worst})");
    }
}
