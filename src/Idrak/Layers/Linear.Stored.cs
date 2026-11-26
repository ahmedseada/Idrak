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

    // Whether a tied table is stored in a type of its own (the product then reads it as stored).
    private static bool StoredTable(Tensor table) => table.Storage.Typed && table.Type.Native != NativeKind.Float32;
}
