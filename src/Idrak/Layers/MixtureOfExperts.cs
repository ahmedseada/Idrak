// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;
using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// A sparse mixture-of-experts feed-forward block (Mixtral, Qwen2-MoE, Qwen3-MoE): a router ("router", [dim, experts])
/// scores each token, the softmax over its scores picks the <see cref="ExpertsPerToken"/> most probable experts (ties go to
/// the lowest index, as torch.topk does), and the token's output is the sum of those experts' outputs weighted by their
/// probabilities, renormalized to sum to one when <see cref="NormalizeTopK"/>. Each expert is a <see cref="FeedForward"/>
/// ("experts.j") that runs only on the tokens routed to it. Optionally a shared expert ("shared") runs on every token and
/// is added, scaled by sigmoid of a gate ("shared_gate", [dim, 1]) when there is one (Qwen2-MoE).
/// <para>
/// At inference (no gradients recorded), on a device that runs the routing operations itself (<see cref="Backend.RoutesExperts"/>:
/// the CPU), with float32 gated experts and no forward hooks, the routing stays where it is: <see cref="Backend.RouteTopK"/>
/// writes each token's weights and the rows grouped by expert, and <see cref="Backend.ExpertGatedPair"/> and
/// <see cref="Backend.ExpertMatMul"/> run every expert over its rows as their offsets say, with the same values as the
/// experts' own layers (nothing is read back to the host). Otherwise (training, packed or adapted experts, a device
/// without those kernels, or <see cref="RoutesOnDevice"/> off) the routing is read back to the host on every call (a few
/// values per token) and each expert runs on its own rows, on a device without the kernels counted as a host fallback of
/// <see cref="Backend.RouteTopK"/>. A recorded graph (<see cref="ComputeGraph"/>) cannot read results back: recording a
/// step through the host routing fails cleanly and the step runs as it is on every replay.
/// </para>
/// <para>
/// The indices the routing writes are floats, as every gather of the library reads them (integer index tensors are a
/// later plan's).
/// </para>
/// </summary>
public sealed class MixtureOfExperts : Module, IRoutedLayer, IHostRoundTrip
{
    /// <summary>Creates the block with random weights.</summary>
    /// <param name="dim">Model width.</param>
    /// <param name="hidden">Hidden size of each expert.</param>
    /// <param name="experts">Number of experts.</param>
    /// <param name="expertsPerToken">Experts each token is routed to.</param>
    /// <param name="normalizeTopK">Renormalize the chosen experts' probabilities to sum to one.</param>
    /// <param name="sharedHidden">Hidden size of a shared expert with a sigmoid gate, or 0 for none.</param>
    /// <param name="activation">The experts' activation (gated blocks).</param>
    /// <param name="bias">Biases in the experts' projections.</param>
    /// <param name="device">Device.</param>
    /// <param name="random">Initialization.</param>
    public MixtureOfExperts(int dim, int hidden, int experts, int expertsPerToken, bool normalizeTopK = true, int sharedHidden = 0,
        FeedForwardActivation activation = FeedForwardActivation.Silu, bool bias = false, Device? device = null, Random? random = null)
        : this(new Linear(dim, experts, bias: false, device, random),
            [.. Enumerable.Range(0, experts).Select(_ => new FeedForward(dim, hidden, gated: true, activation, bias, device, random))],
            expertsPerToken, normalizeTopK,
            sharedHidden > 0 ? new FeedForward(dim, sharedHidden, gated: true, activation, bias, device, random) : null,
            sharedHidden > 0 ? new Linear(dim, 1, bias: false, device, random) : null)
    {
    }

    /// <summary>Creates the block from existing layers (for example loaded weights); the block takes ownership.</summary>
    /// <param name="router">The router, [dim, experts].</param>
    /// <param name="experts">The experts, one per router output.</param>
    /// <param name="expertsPerToken">Experts each token is routed to.</param>
    /// <param name="normalizeTopK">Renormalize the chosen experts' probabilities to sum to one.</param>
    /// <param name="sharedExpert">An expert every token goes through, or null.</param>
    /// <param name="sharedExpertGate">A [dim, 1] gate scaling the shared expert's output by its sigmoid, or null for no scaling.</param>
    public MixtureOfExperts(Linear router, IReadOnlyList<FeedForward> experts, int expertsPerToken, bool normalizeTopK,
        FeedForward? sharedExpert = null, Linear? sharedExpertGate = null)
    {
        ArgumentNullException.ThrowIfNull(router);
        ArgumentNullException.ThrowIfNull(experts);
        if (router.OutFeatures != experts.Count || experts.Count == 0)
        {
            throw new ArgumentException($"The router scores {router.OutFeatures} experts; {experts.Count} were given.");
        }

        if (expertsPerToken < 1 || expertsPerToken > experts.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(expertsPerToken), $"Each token goes to between 1 and {experts.Count} experts.");
        }

        if (experts.Any(e => e.Up.InFeatures != router.InFeatures || e.Down.OutFeatures != router.InFeatures)
            || sharedExpert is not null && (sharedExpert.Up.InFeatures != router.InFeatures || sharedExpert.Down.OutFeatures != router.InFeatures))
        {
            throw new ArgumentException($"Every expert must map width {router.InFeatures} to itself.");
        }

        if (sharedExpertGate is not null && (sharedExpert is null || sharedExpertGate.InFeatures != router.InFeatures || sharedExpertGate.OutFeatures != 1))
        {
            throw new ArgumentException($"The shared expert's gate must be [{router.InFeatures}, 1], with a shared expert.");
        }

        (Router, Experts, ExpertsPerToken, NormalizeTopK, SharedExpert, SharedExpertGate) =
            (router, experts, expertsPerToken, normalizeTopK, sharedExpert, sharedExpertGate);
        router.Name = "router";
        for (int j = 0; j < experts.Count; j++)
        {
            experts[j].Name = $"experts.{j}";
        }

        if (sharedExpert is not null)
        {
            sharedExpert.Name = "shared";
        }

        if (sharedExpertGate is not null)
        {
            sharedExpertGate.Name = "shared_gate";
        }
    }

    /// <summary>The router ("router"), [dim, experts]: one score per expert and token.</summary>
    public Linear Router { get; }

    /// <summary>The experts ("experts.j").</summary>
    public IReadOnlyList<FeedForward> Experts { get; }

    /// <summary>Experts each token is routed to.</summary>
    public int ExpertsPerToken { get; }

    /// <summary>Whether the chosen experts' probabilities are renormalized to sum to one (Mixtral always; Qwen per norm_topk_prob).</summary>
    public bool NormalizeTopK { get; }

    /// <summary>The expert every token goes through ("shared"), or null.</summary>
    public FeedForward? SharedExpert { get; }

    /// <summary>The shared expert's gate ("shared_gate", [dim, 1]; its sigmoid scales the shared expert's output), or null.</summary>
    public Linear? SharedExpertGate { get; }

    /// <summary>
    /// The load-balancing loss of the last forward pass in training mode with gradients recorded (null otherwise): the
    /// Switch Transformer's experts · Σₑ fₑ · Pₑ, with fₑ the share of the routed slots that went to expert e and Pₑ its mean
    /// router probability (<see cref="ExpertsPerToken"/> when the routing is perfectly even; transformers' definition, so
    /// its router_aux_loss_coef applies). Add it, scaled, to a training loss to keep the experts evenly used; it is valid as
    /// long as the pass's tensors are.
    /// </summary>
    public Tensor? LoadBalancingLoss { get; private set; }

    /// <inheritdoc />
    public override IEnumerable<Module> Children()
    {
        yield return Router;
        foreach (var expert in Experts)
        {
            yield return expert;
        }

        if (SharedExpert is not null)
        {
            yield return SharedExpert;
        }

        if (SharedExpertGate is not null)
        {
            yield return SharedExpertGate;
        }
    }

    /// <summary>
    /// Whether a mixture-of-experts block routes on the device where it can (see the class remarks): on unless the
    /// environment variable <c>IDRAK_MOE_ROUTING</c> is <c>host</c> when the process starts. Off, every block reads its
    /// routing back to the host (the path a device without the routing kernels takes), with the same results: to compare
    /// the two or to isolate one.
    /// </summary>
    public static bool RoutesOnDevice { get; set; } =
        Environment.GetEnvironmentVariable("IDRAK_MOE_ROUTING") is not { } routing || !routing.Trim().Equals("host", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        int dim = Router.InFeatures, tokens = input.Size / dim, experts = Experts.Count, k = ExpertsPerToken;
        var device = input.Device;
        var backend = input.Backend;
        bool kernels = backend.RoutesExperts;
        bool onDevice = kernels && RoutesOnDevice && !Autograd.IsEnabled && ExpertsOnDevice(input);
        if (!onDevice && ComputeGraph.IsCapturing)
        {
            throw new NotSupportedException("Mixture-of-experts routing reads the router's choices back to the host here (the device has no routing kernels, or the experts or the pass need the host routing), which a recorded graph cannot; the step runs as it is.");
        }

        var x = input.Reshape(tokens, dim);
        var probabilities = Router.Forward(x).Softmax();                             // [tokens, experts]
        if (onDevice)
        {
            return RouteOnDevice(input, x, probabilities);
        }

        if (!kernels)
        {
            backend.CountHostFallback(Ops.RouteTopK);                                 // the routing this device has no kernel for
        }

        long readStart = Telemetry.Start(TelemetryLevel.Operations);
        var routed = probabilities.ToArray();
        Tensor.Traced("route_on_host", probabilities, readStart);                    // the host read, for a census of them
        var (choices, weights) = Route(routed, tokens, experts, k, NormalizeTopK);

        // The rows routed to each expert, expert by expert (each token's experts are distinct, so an expert given every token
        // has them in order and reads the input as it is); slot (token i, its s-th expert) is row slotRow[i·k + s] of the
        // experts' outputs stacked in that order.
        var counts = new int[experts];
        foreach (int e in choices)
        {
            counts[e]++;
        }

        var starts = new int[experts];
        for (int e = 1; e < experts; e++)
        {
            starts[e] = starts[e - 1] + counts[e - 1];
        }

        var cursor = (int[])starts.Clone();
        var tokenOfRow = new int[tokens * k];                                         // indices as integers (plan 37, T1)
        var slotRow = new int[tokens * k];
        bool inOrder = true;
        for (int slot = 0; slot < choices.Length; slot++)
        {
            int row = cursor[choices[slot]]++;
            tokenOfRow[row] = slot / k;
            slotRow[slot] = row;
            inOrder &= row == slot;
        }

        var outputs = new List<Tensor>(experts);
        for (int e = 0; e < experts; e++)
        {
            if (counts[e] > 0)
            {
                var rows = counts[e] == tokens ? x : x.EmbeddingLookup(Tensor.Indices(tokenOfRow.AsSpan(starts[e], counts[e]), device));
                outputs.Add(Experts[e].Forward(rows));
            }
        }

        var stacked = outputs.Count == 1 ? outputs[0] : Tensor.Concat(outputs);       // [tokens·k, dim], expert by expert
        var slots = inOrder ? stacked : stacked.EmbeddingLookup(Tensor.Indices(slotRow, device));
        var routing = Autograd.IsEnabled && probabilities.RequiresGrad
            ? RecordedWeights(probabilities, choices, tokens, experts, k)
            : Tensor.From(weights, [tokens, 1, k], device);
        var output = routing.MatMul(slots.Reshape(tokens, k, dim));                    // Σ weight · expert output, [tokens, 1, dim]

        if (SharedExpert is not null)
        {
            var shared = SharedExpert.Forward(x).Reshape(tokens, 1, dim);
            output += SharedExpertGate is null ? shared : SharedExpertGate.Forward(x).Sigmoid().Reshape(tokens, 1, 1).MatMul(shared);
        }

        LoadBalancingLoss = null;
        if (Autograd.IsEnabled && IsTraining && probabilities.RequiresGrad)
        {
            var share = new float[experts];
            for (int e = 0; e < experts; e++)
            {
                share[e] = (float)counts[e] / tokens;
            }

            LoadBalancingLoss = (probabilities.Mean(0) * Tensor.From(share, device)).Sum() * experts;
        }

        return output.Reshape(input.Shape);
    }

    // Whether every expert is a gated block of float32 layers of its own (no adapter, tie or soft-cap) of one hidden width on
    // the input's device, with nothing listening to the forward passes (a sharded run's gathers, offloaded weights): what
    // ExpertGatedPair and ExpertMatMul compute as the experts' own layers do. A plain loop: it runs on every forward.
    private bool ExpertsOnDevice(Tensor input)
    {
        if (ForwardHooks.Listening || input.Backend.Offload is not null)
        {
            return false;
        }

        int hidden = Experts[0].Up.OutFeatures;
        var activation = Experts[0].Activation;
        foreach (var expert in Experts)
        {
            if (expert.Gate is not { } gate || expert.Activation != activation || !Linear.PlainFloat(gate) || !Linear.PlainFloat(expert.Up) || !Linear.PlainFloat(expert.Down)
                || gate.OutFeatures != hidden || expert.Up.OutFeatures != hidden || expert.Down.InFeatures != hidden
                || gate.Weight.Device != input.Device || expert.Up.Weight.Device != input.Device || expert.Down.Weight.Device != input.Device
                || gate.Bias is { } gb && gb.Device != input.Device || expert.Up.Bias is { } ub && ub.Device != input.Device
                || expert.Down.Bias is { } db && db.Device != input.Device)
            {
                return false;
            }
        }

        return true;
    }

    // The routing and the experts where the tensors are (inference): RouteTopK writes the weights and the rows grouped by
    // expert; the rows are gathered once, every expert's gate and up projections and gated activation run over its rows
    // (ExpertGatedPair), then its down projection (ExpertMatMul), each reading its rows from the offsets where they are;
    // the outputs go back to slot order and are weighted as the host routing does. The same values as that path: the same
    // gathers, products and activation per expert, and the same weighting product.
    private Tensor RouteOnDevice(Tensor input, Tensor x, Tensor probabilities)
    {
        int dim = Router.InFeatures, tokens = x.Shape[0], experts = Experts.Count, k = ExpertsPerToken, slots = tokens * k;
        int hidden = Experts[0].Up.OutFeatures;
        var device = input.Device;
        var backend = input.Backend;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var weights = Tensor.Empty([tokens, 1, k], device, track: false);
        var indexType = backend.IndexTypeOf(Ops.RouteTopK);                       // what the routing writes (plan 37, T1)
        var offsets = Tensor.Empty([experts + 1], device, indexType, track: false);
        var tokenOfRow = Tensor.Empty([slots], device, indexType, track: false);
        var slotRow = Tensor.Empty([slots], device, indexType, track: false);
        Tensor? rows = null, gated = null, outputs = null, slotOrder = null;
        try
        {
            backend.RouteTopK(probabilities.Storage, weights.Storage, offsets.Storage, tokenOfRow.Storage, slotRow.Storage, tokens, experts, k, NormalizeTopK);
            Tensor.Traced("route_top_k", weights, start);

            rows = x.EmbeddingLookup(tokenOfRow);                                     // [slots, dim], expert by expert
            var pairs = new (Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)[experts];
            var downs = new (Storage Weight, Storage? Bias)[experts];
            for (int e = 0; e < experts; e++)
            {
                var expert = Experts[e];
                pairs[e] = (expert.Gate!.Weight.Storage, expert.Gate.Bias?.Storage, expert.Up.Weight.Storage, expert.Up.Bias?.Storage);
                downs[e] = (expert.Down.Weight.Storage, expert.Down.Bias?.Storage);
            }

            start = Telemetry.Start(TelemetryLevel.Operations);
            gated = Tensor.Empty([slots, hidden], device, track: false);
            backend.ExpertGatedPair(rows.Storage, offsets.Storage, pairs, gated.Storage, slots, dim, hidden, (int)Experts[0].Activation);
            Tensor.Traced("expert_gated_pair", gated, start);

            start = Telemetry.Start(TelemetryLevel.Operations);
            outputs = Tensor.Empty([slots, dim], device, track: false);
            backend.ExpertMatMul(gated.Storage, offsets.Storage, downs, outputs.Storage, slots, hidden, dim);
            Tensor.Traced("expert_matmul", outputs, start);

            slotOrder = outputs.EmbeddingLookup(slotRow);                             // [slots, dim], token by token
            var output = weights.MatMul(slotOrder.Reshape(tokens, k, dim));           // Σ weight · expert output, [tokens, 1, dim]
            if (SharedExpert is not null)
            {
                var shared = SharedExpert.Forward(x).Reshape(tokens, 1, dim);
                output += SharedExpertGate is null ? shared : SharedExpertGate.Forward(x).Sigmoid().Reshape(tokens, 1, 1).MatMul(shared);
            }

            LoadBalancingLoss = null;
            return output.Reshape(input.Shape);
        }
        finally
        {
            // Read by the operations above only (on a device with a queue, freed after them in its order).
            weights.Dispose();
            offsets.Dispose();
            tokenOfRow.Dispose();
            slotRow.Dispose();
            rows?.Dispose();
            gated?.Dispose();
            outputs?.Dispose();
            slotOrder?.Dispose();
        }
    }

    // The chosen experts' probabilities as [tokens, 1, k], with gradients to the router: gathered from the softmax and,
    // when normalized, divided by their sum.
    private Tensor RecordedWeights(Tensor probabilities, int[] choices, int tokens, int experts, int k)
    {
        var index = new int[choices.Length];                                       // exact past 2^24 (tokens · experts)
        for (int slot = 0; slot < choices.Length; slot++)
        {
            index[slot] = slot / k * experts + choices[slot];
        }

        var chosen = probabilities.Reshape(tokens * experts, 1).EmbeddingLookup(Tensor.Indices(index, probabilities.Device)).Reshape(tokens, 1, k);
        return NormalizeTopK ? chosen.Sum(2, keepDim: true).Pow(-1f).MatMul(chosen) : chosen;
    }

    /// <summary>
    /// The routing of <paramref name="tokens"/> rows of router probabilities [tokens, experts]: each token's
    /// <paramref name="k"/> most probable experts (ties to the lowest index), listed in increasing expert order, and their
    /// weights (the probabilities, renormalized to sum to one when <paramref name="normalize"/>).
    /// </summary>
    internal static (int[] Choices, float[] Weights) Route(float[] probabilities, int tokens, int experts, int k, bool normalize)
    {
        var choices = new int[tokens * k];
        var weights = new float[tokens * k];
        // k comes from the model: on the stack only while small (rule 45). Each row's choices are sorted in place, as
        // the next row reads only the entries it has written.
        const int StackExperts = 64;
        Span<int> best = k <= StackExperts ? stackalloc int[StackExperts] : new int[k];
        best = best[..k];
        for (int i = 0; i < tokens; i++)
        {
            var row = probabilities.AsSpan(i * experts, experts);
            int count = 0;
            for (int e = 0; e < experts; e++)
            {
                // Insert e after every chosen expert at least as probable (earlier indices win ties).
                int at = count;
                while (at > 0 && row[e] > row[best[at - 1]])
                {
                    at--;
                }

                if (at >= k)
                {
                    continue;
                }

                for (int j = Math.Min(count, k - 1); j > at; j--)
                {
                    best[j] = best[j - 1];
                }

                best[at] = e;
                count = Math.Min(count + 1, k);
            }

            best.Sort();
            ReadOnlySpan<int> chosen = best;
            float sum = 0f;
            foreach (int e in chosen)
            {
                sum += row[e];
            }

            for (int s = 0; s < k; s++)
            {
                choices[i * k + s] = chosen[s];
                weights[i * k + s] = normalize ? row[chosen[s]] / sum : row[chosen[s]];
            }
        }

        return (choices, weights);
    }

    /// <inheritdoc />
    IEnumerable<(Module Part, int Times)> IRoutedLayer.PartsPerPosition()
    {
        yield return (Router, 1);
        if (Experts.Count > 0)
        {
            yield return (Experts[0], ExpertsPerToken);
        }

        if (SharedExpert is { } shared)
        {
            yield return (shared, 1);
        }

        if (SharedExpertGate is { } gate)
        {
            yield return (gate, 1);
        }
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"MixtureOfExperts({Experts.Count} experts, {ExpertsPerToken} per token{(NormalizeTopK ? ", normalized" : "")}{(SharedExpert is null ? "" : ", shared expert")})";
}
