// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Devices;

// The mixture-of-experts operations (plan 36, L4): the routing of tokens to their experts and the experts' products over
// rows grouped by expert, each expert reading its row range from the offsets the routing wrote where they are. A layer
// that runs them reads nothing back to the host: the row counts never reach it. Their default bodies are the host
// fallback (every operand mirrored, the experts' weights included), so a mixture-of-experts layer on a device without
// kernels of its own for them (RoutesExperts false) keeps its host routing instead and counts it as a host fallback of
// RouteTopK (CountHostFallback). tools/operations/generate.py reads this file after Backend.cs: these operations come
// after Backend.cs's in the slot order.
public abstract partial class Backend
{
    /// <summary>
    /// The mixture-of-experts routing of <paramref name="tokens"/> rows of router probabilities [tokens, experts]: each
    /// token's <paramref name="k"/> most probable experts (ties to the lowest index, as torch.topk), listed in increasing
    /// expert order, token i's s-th one at slot i·k + s. Writes <paramref name="weights"/>[slot] (its probability, divided
    /// by the sum of the token's k chosen probabilities added in increasing expert order when <paramref name="normalize"/>),
    /// <paramref name="offsets"/>[e] (the first row of expert e when the slots are listed expert by expert, each expert's in
    /// slot order; <paramref name="offsets"/>[experts] = tokens·k), <paramref name="tokenOfRow"/>[row] (the token of that
    /// row) and <paramref name="slotRow"/>[slot] (the row of that slot); indices and offsets are written in the device's
    /// index type (<see cref="BackendCapabilities.IndexType"/>: int32 on the CPU, exact past 2²⁴ rows).
    /// </summary>
    public virtual void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
        bool normalize)
    {
        using var h = new HostCall(this, nameof(RouteTopK));
        CpuBackend.Instance.RouteTopK(h[probabilities], h[weights], h[offsets], h[tokenOfRow], h[slotRow], tokens, experts, k, normalize);
    }

    /// <summary>
    /// The gate and up projections of a mixture's experts over rows grouped by expert, and their gated activation: for
    /// each expert e, rows offsets[e] to offsets[e + 1] of x [rows, k] (the offsets as <see cref="RouteTopK"/> writes them,
    /// read where they are) give those rows of hidden [rows, n] = act(x·gate_e + gateBias_e) · (x·up_e + upBias_e), with
    /// gate_e and up_e [k, n] and the biases [n] or null, act as <see cref="GatedActivation"/>'s
    /// <paramref name="activation"/>. Each expert's rows hold what its own projections and <see cref="GatedActivation"/>
    /// over just those rows give, bit for bit.
    /// </summary>
    public virtual void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts, Storage hidden,
        int rows, int k, int n, int activation)
    {
        using var h = new HostCall(this, nameof(ExpertGatedPair));
        var mirrored = new (Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)[experts.Length];
        for (int e = 0; e < experts.Length; e++)
        {
            var (gate, gateBias, up, upBias) = experts[e];
            mirrored[e] = (h[gate], gateBias is null ? null : h[gateBias], h[up], upBias is null ? null : h[upBias]);
        }

        CpuBackend.Instance.ExpertGatedPair(h[x], h[offsets], mirrored, h[hidden], rows, k, n, activation);
    }

    /// <summary>
    /// The products of a mixture's experts over rows grouped by expert: for each expert e, rows offsets[e] to
    /// offsets[e + 1] of x [rows, k] (the offsets as <see cref="RouteTopK"/> writes them, read where they are) give those
    /// rows of y [rows, n] = x·weight_e + bias_e, with weight_e [k, n] and bias_e [n] or null. Each expert's rows hold what
    /// its own product (<see cref="MatMul"/>, then the bias added) over just those rows gives, bit for bit.
    /// </summary>
    public virtual void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n)
    {
        using var h = new HostCall(this, nameof(ExpertMatMul));
        var mirrored = new (Storage Weight, Storage? Bias)[experts.Length];
        for (int e = 0; e < experts.Length; e++)
        {
            var (weight, bias) = experts[e];
            mirrored[e] = (h[weight], bias is null ? null : h[bias]);
        }

        CpuBackend.Instance.ExpertMatMul(h[x], h[offsets], mirrored, h[y], rows, k, n);
    }

    /// <summary>
    /// Whether this device runs the mixture-of-experts operations (<see cref="RouteTopK"/>, <see cref="ExpertGatedPair"/>
    /// and <see cref="ExpertMatMul"/>) with kernels of its own or registered ones (<see cref="Kernels.Register"/>). Where
    /// it does not, a mixture-of-experts layer reads its routing back and routes on the host, counted as a host fallback
    /// of <see cref="RouteTopK"/> (<see cref="CountHostFallback"/>), instead of mirroring every expert's weights through the
    /// operations' host fallbacks.
    /// </summary>
    public bool RoutesExperts =>
        RunsKernel(OperationIndex.RouteTopK) && RunsKernel(OperationIndex.ExpertGatedPair) && RunsKernel(OperationIndex.ExpertMatMul);

    /// <summary>
    /// Counts one host fallback of <paramref name="operation"/> on this device, as its host fallback counts itself
    /// (<see cref="Kernels.HostCalls"/>, a <see cref="Kernels.Trace"/>, and telemetry's <see cref="HostFallback"/> naming
    /// the device and the operation once): for a layer that runs an operation's work on the host itself because the device
    /// has no kernel for it (a mixture-of-experts layer routing on the host, <see cref="RoutesExperts"/>).
    /// </summary>
    public void CountHostFallback(Operation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        HostCalled(operation.Name);
    }

    // Whether a library operation runs a kernel of the device's own or a registered one here (not its default body).
    // The slots already resolved are read as they are (null: nothing registered, the guarded slots: none either).
    private bool RunsKernel(int operation)
    {
        if (OwnsKernel(operation))
        {
            return true;
        }

        var slots = Volatile.Read(ref _kernels);
        return slots is not null && (slots.Length != 0 ? slots[operation] : Operations.Kernels.Resolve(this)[operation]) is not null;
    }
}
