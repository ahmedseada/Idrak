// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// The graph operations this assembly ships (the element-wise and tensor operations ONNX names), registered in
/// <see cref="GraphOps"/> (Idrak.Abstraction) by <see cref="LibraryRegistrations"/>.
/// </summary>
internal static class LibraryGraphOps
{
    public static void RegisterAll()
    {
        foreach (var (name, op) in BuiltIns())
        {
            GraphOps.Register(name, op);
        }
    }

    // The element-wise and tensor operations, following the ONNX operators of the same names.
    private static Dictionary<string, GraphOp> BuiltIns() => new()
    {
        ["matmul"] = c => c.Input(0).MatMul(c.Input(1)),
        ["relu"] = c => c.Input(0).Relu(),
        ["tanh"] = c => c.Input(0).Tanh(),
        ["sigmoid"] = c => c.Input(0).Sigmoid(),
        ["exp"] = c => c.Input(0).Exp(),
        ["log"] = c => c.Input(0).Log(),
        ["abs"] = c => c.Input(0).Abs(),
        ["neg"] = c => -c.Input(0),
        ["sqrt"] = c => (c.Input(0).Log() * 0.5f).Exp(),
        ["pow"] = c => Pow(c.Input(0), Scalar(c, 1, "pow", "exponent")),
        ["clip"] = c => Clip(c.Input(0), c.Has(1) ? Scalar(c, 1, "clip", "minimum") : c.Node.Attributes?["min"] is not null ? c.Float("min", 0f) : null,
            c.Has(2) ? Scalar(c, 2, "clip", "maximum") : c.Node.Attributes?["max"] is not null ? c.Float("max", 0f) : null),
        ["leaky_relu"] = c => c.Input(0).Relu() - (-c.Input(0)).Relu() * c.Float("alpha", 0.01f),
        ["elu"] = c => c.Input(0).Relu() + ((-(-c.Input(0)).Relu()).Exp() - 1f) * c.Float("alpha", 1f),
        ["hard_sigmoid"] = c => Clip(c.Input(0) * c.Float("alpha", 0.2f) + c.Float("beta", 0.5f), 0f, 1f),
        ["hard_swish"] = c => c.Input(0) * Clip(c.Input(0) * (1f / 6f) + 0.5f, 0f, 1f),
        ["max"] = c => Fold(c, (a, b) => GraphModule.Binary("add", b, Relu(GraphModule.Binary("sub", a, b)))),     // b + relu(a − b)
        ["min"] = c => Fold(c, (a, b) => GraphModule.Binary("sub", a, Relu(GraphModule.Binary("sub", a, b)))),     // a − relu(a − b)
        ["softmax"] = c =>
        {
            var s = c.Input(0);
            long axis = c.Int("axis", -1);
            return axis == -1 || axis == s.Rank - 1 ? s.Softmax() : throw new NotSupportedException("softmax over an axis other than the last");
        },
        ["flatten"] = c =>
        {
            var f = c.Input(0);
            int at = GraphModule.Normalize((int)c.Int("axis", 1), f.Rank);
            int outer = 1;
            for (int i = 0; i < at; i++)
            {
                outer *= f.Shape[i];
            }

            return f.Reshape(outer, f.Size / Math.Max(outer, 1));
        },
        ["transpose"] = c =>
        {
            var x = c.Input(0);
            int[] perm = c.Ints("perm") is { } p ? [.. p.Select(v => (int)v)] : [.. Enumerable.Range(0, x.Rank).Reverse()];
            return x.Permute(perm);
        },
        ["reduce_mean"] = c =>
        {
            var r = c.Input(0);
            var axes = c.Ints("axes") ?? (c.Has(1) ? c.Integers(1) : null) ?? [.. Enumerable.Range(0, r.Rank).Select(i => (long)i)];
            bool keep = c.Int("keepdims", 1) == 1;
            foreach (int reduce in axes.Select(v => GraphModule.Normalize((int)v, r.Rank)).OrderDescending())
            {
                r = r.Mean(reduce, keep);
            }

            return r;
        },
        ["global_average_pool"] = c =>
        {
            var g = c.Input(0);
            return g.Reshape(g.Shape[0], g.Shape[1], -1).Mean(2).Reshape(g.Shape[0], g.Shape[1], 1, 1);
        },
    };

    private static float Scalar(GraphOpContext c, int index, string op, string what)
    {
        var t = c.Input(index);
        return t.Size == 1 ? t.ToArray()[0] : throw new NotSupportedException($"{op} with a {what} that is not a single value");
    }

    // x^e from multiplications for small whole exponents (any sign of x), otherwise exp(e · log x) (x must be positive).
    private static Tensor Pow(Tensor x, float e)
    {
        if (e == MathF.Floor(e) && e is >= 0f and <= 8f)
        {
            if (e == 0f)
            {
                return x * 0f + 1f;
            }

            var result = x;
            for (int i = 1; i < (int)e; i++)
            {
                result *= x;
            }

            return result;
        }

        return (x.Log() * e).Exp();
    }

    // min(max(x, lo), hi) from two ReLUs: lo + relu(x − lo) − relu(x − hi).
    private static Tensor Clip(Tensor x, float? lo, float? hi) => (lo, hi) switch
    {
        ({ } l, { } h) => (x - l).Relu() - (x - h).Relu() + l,
        ({ } l, null) => (x - l).Relu() + l,
        (null, { } h) => h - (h - x).Relu(),
        _ => x,
    };

    private static object Relu(object value) => ((Tensor)value).Relu();

    private static Tensor Fold(GraphOpContext c, Func<object, object, object> combine)
    {
        object result = c.Input(0);
        for (int i = 1; i < c.Count; i++)
        {
            result = combine(result, c.Input(i));
        }

        return result as Tensor ?? throw new InvalidOperationException("expected a tensor");
    }
}
