// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

// A linear layer over a weight as its checkpoint stores it: [out, in] of any type (IElementType), never widened, multiplied
// by the device's product of stored weights (Backend.StoredMatMul), which reads the type through its description.
public sealed partial class Linear
{
    // The weight as stored, [out, in] of its own type, or null.
    private Tensor? _stored;

    /// <summary>
    /// A layer y = x · Wᵀ (+ b) around <paramref name="stored"/>, W as its checkpoint stores it: [out, in] of any type
    /// (<see cref="Tensor.Stored(IElementType, ReadOnlySpan{byte}, ReadOnlySpan{int}, Device?)"/>), never widened; the device's
    /// product of stored weights computes it. For inference: the stored weight takes no gradient. The layer takes ownership.
    /// </summary>
    /// <param name="stored">W [out, in].</param>
    /// <param name="bias">The bias [out], or null.</param>
    /// <exception cref="ArgumentException">W is not [out, in], or the bias is not [out].</exception>
    public static Linear FromStored(Tensor stored, Tensor? bias = null)
    {
        ArgumentNullException.ThrowIfNull(stored);
        if (stored.Rank != 2 || (bias is not null && (bias.Rank != 1 || bias.Shape[0] != stored.Shape[0])))
        {
            throw new ArgumentException($"A stored weight is [out, in] with a bias [out]; got {Tensor.FormatShape(stored.Shape)} and {(bias is null ? "no bias" : Tensor.FormatShape(bias.Shape))}.");
        }

        return new Linear(stored.Shape[1], stored.Shape[0], null, bias) { _stored = stored };
    }

    /// <summary>The weight as its checkpoint stores it ([out, in] of its type: <see cref="FromStored"/>), or null for a float32 weight [in, out].</summary>
    public Tensor? Stored => _stored;

    // x · Wᵀ for a stored W [n, k] (a layer's own, or a tied table of another type): x as [m, k] rows, y [.., n].
    private static Tensor StoredProduct(Tensor input, Tensor stored)
    {
        int n = stored.Shape[0], k = stored.Shape[1];
        if (input.Shape[^1] != k)
        {
            throw new ArgumentException($"The input's rows are {input.Shape[^1]} values; the stored weight's are {k}.", nameof(input));
        }

        var x = input.Contiguous();
        int m = x.Size / Math.Max(1, k);
        var y = Tensor.Empty([.. input.Shape[..^1], n], input.Device);
        input.Backend.StoredMatMul(x.Storage, stored.Storage, y.Storage, m, n, k);
        return y;
    }

    /// <summary>
    /// The products of 2 or 3 layers over weights as stored, of one type, no bias, adapter or cap, reading the same input
    /// (the queries, keys and values), in one device pass (Backend.StoredMatMulMany); null where the layers do not fit or
    /// the device has no such pass for them (the caller runs each on its own).
    /// </summary>
    internal static Tensor[]? StoredMany(Tensor input, Linear[] layers)
    {
        int k = input.Shape[^1], m = input.Size / Math.Max(1, k);
        var description = layers[0]._stored?.Storage.Description;
        foreach (var layer in layers)
        {
            if (layer._stored is not { } w || layer.Bias is not null || layer.Adapter is not null || layer._softcap is not null || layer._tiedTo is not null
                || w.Shape[1] != k || w.Device != input.Device || description is null || !Equals(w.Storage.Description, description))
            {
                return null;
            }
        }

        if (m == 0)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var x = input.Contiguous();
        var outputs = new Tensor[layers.Length];
        for (int j = 0; j < layers.Length; j++)
        {
            outputs[j] = Tensor.Empty([.. input.Shape[..^1], layers[j]._stored!.Shape[0]], input.Device);
        }

        var (a, b) = (layers[0]._stored!, layers[1]._stored!);
        bool done = layers.Length == 2
            ? input.Backend.StoredMatMulMany(x.Storage, m, k, [(a.Storage, outputs[0].Storage, a.Shape[0]), (b.Storage, outputs[1].Storage, b.Shape[0])])
            : input.Backend.StoredMatMulMany(x.Storage, m, k, [(a.Storage, outputs[0].Storage, a.Shape[0]), (b.Storage, outputs[1].Storage, b.Shape[0]),
                (layers[2]._stored!.Storage, outputs[2].Storage, layers[2]._stored!.Shape[0])]);
        if (!done)
        {
            foreach (var output in outputs)
            {
                output.Dispose();
            }

            return null;
        }

        for (int j = 0; j < outputs.Length; j++)
        {
            outputs[j] = Tensor.Traced("matmul_stored_many", outputs[j], start);
        }

        return outputs;
    }

    /// <summary>
    /// act(gate) · up of a feed-forward block's gate and up projections (<paramref name="gateUp"/>: the two layers, each over
    /// a weight as stored, of one type and shape, no bias, adapter or cap), their products and the activation in one device
    /// pass (Backend.StoredMatMulGatedPair) where the device has one for the type and shape; null otherwise and while
    /// gradients are recorded (the caller composes them: the products, then <see cref="Tensor.GatedActivation"/>).
    /// </summary>
    internal static Tensor? StoredGatedPair(Tensor input, Linear[] gateUp, FeedForwardActivation activation)
    {
        if (Autograd.IsEnabled || gateUp is not [var gate, var up] || gate._stored is not { } g || up._stored is not { } u || !Plain(gate) || !Plain(up)
            || g.Storage.Description is not { } description || !Equals(description, u.Storage.Description) || g.Shape[0] != u.Shape[0] || g.Shape[1] != u.Shape[1]
            || input.Shape[^1] != g.Shape[1] || g.Device != input.Device || u.Device != input.Device)
        {
            return null;
        }

        int n = g.Shape[0], k = g.Shape[1], m = input.Size / Math.Max(1, k);
        if (m == 0 || input.Backend.StoredEpilogueFused(0, (int)activation, description, m, n, k) == false)
        {
            return null;
        }

        using var gathered = ForwardHooks.Listening ? Offloading.Gather(gateUp) : null;   // a sharded run's weights of both layers (plan 23, C6)
        using var offload = Offloading.EnterMany(gateUp, input);         // offloaded weights of both staged together
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var x = input.Contiguous();
        var gateOut = Tensor.Empty([m * n], input.Device, track: false);
        var upOut = Tensor.Empty([m * n], input.Device, track: false);
        var hidden = Tensor.Empty([.. input.Shape[..^1], n], input.Device);
        bool done = input.Backend.StoredMatMulGatedPair((int)activation, x.Storage, m, k, [(g.Storage, gateOut.Storage, n), (u.Storage, upOut.Storage, n)], hidden.Storage);
        gateOut.Dispose();                                               // the products' room: freed once the pass read it
        upOut.Dispose();
        if (!done)
        {
            hidden.Dispose();
            return null;
        }

        return Tensor.Traced("matmul_gated_pair_stored", hidden, start);

        static bool Plain(Linear layer) => layer.Bias is null && layer.Adapter is null && layer._softcap is null && layer._tiedTo is null;
    }

    // Whether a tied table is stored in a type of its own (the product then reads it as stored).
    private static bool StoredTable(Tensor table) => table.Storage.Typed && table.Type.Native != NativeKind.Float32;
}
