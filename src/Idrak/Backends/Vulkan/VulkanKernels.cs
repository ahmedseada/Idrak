// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Backends.Vulkan;

/// <summary>
/// The SPIR-V kernels of the Vulkan backend, generated in C# with <see cref="KernelBuilder"/> (no shader compiler, no
/// precompiled binaries), each matching its <see cref="Backend"/> operation and the CPU's math. A kernel is built on its
/// first use. Dispatch shapes:
/// <list type="bullet">
/// <item>element-wise kernels (local size 256) loop over their n elements with a grid stride: any number of groups works;
/// ⌈n / 256⌉ capped at 65535 is the natural choice;</item>
/// <item>row kernels (one workgroup of 256 per row) loop over the rows with a stride of the group count: min(rows, 65535)
/// groups;</item>
/// <item>single-group kernels (<c>sum</c>, <c>sum_squares</c>, <c>axpy_at</c>, <c>clip_factor</c>) take exactly one group;</item>
/// <item><c>batched_matmul</c> (16 × 16) takes ⌈n / 16⌉ × ⌈m / 16⌉ × min(batch, 65535) groups.</item>
/// </list>
/// Integers that tensors hold as values (indices, positions) are read as floats and converted, as the CPU does; packed
/// words (int8, int4, bfloat16) are read as bits.
/// </summary>
internal static partial class VulkanKernels
{
    /// <summary>Invocations per workgroup of the 1-D kernels (the reductions are unrolled for it).</summary>
    public const int Block = 256;

    /// <summary>Tile edge of the matrix product: workgroups of Tile × Tile, each computing one Tile × Tile block of c.</summary>
    public const int Tile = 16;

    /// <summary>Largest head size <c>attention_decode</c> takes (one invocation per output dimension).</summary>
    public const int AttentionMaxDim = Block;

    // GELU, tanh approximation: 0.5 x (1 + tanh(k (x + 0.044715 x³))), k = sqrt(2/π) (the CPU's constants).
    private const float GeluK = 0.7978845608f;
    private const float GeluC = 0.044715f;

    private static readonly ConcurrentDictionary<string, SpirvKernel> Built = new();

    private static readonly Lazy<Dictionary<string, Func<SpirvKernel>>> LazyFactories = new(() =>
    {
        var all = new Dictionary<string, Func<SpirvKernel>>();
        foreach (var (name, build) in ElementwiseKernels().Concat(RowKernels()).Concat(ShapeKernels()).Concat(MatMulKernels()).Concat(DecodingKernels())
            .Concat(SamplingKernels()))
        {
            all.Add(name, build);
        }

        return all;
    });

    /// <summary>Every kernel's name.</summary>
    public static IEnumerable<string> Names => LazyFactories.Value.Keys;

    /// <summary>The kernel with this name, built on first use.</summary>
    public static SpirvKernel Get(string name) => Built.GetOrAdd(name, static n => LazyFactories.Value.TryGetValue(n, out var build)
        ? build()
        : throw new KeyNotFoundException($"No Vulkan kernel named '{n}'."));

    /// <summary>The kernel for y = op(x).</summary>
    public static SpirvKernel Unary(UnaryOp op) => Get("unary_" + Snake(op));

    /// <summary>The kernel for dx += dy · op'(x).</summary>
    public static SpirvKernel UnaryBackward(UnaryOp op) => Get("unary_backward_" + Snake(op));

    /// <summary>The kernel for c = a op b.</summary>
    public static SpirvKernel Binary(BinaryOp op) => Get("binary_" + op.ToString().ToLowerInvariant());

    private static string Snake(UnaryOp op) => op.ToString().ToLowerInvariant();

    // Grid-stride loop over [0, n): every invocation of the dispatch takes the indices i ≡ its global id.
    private static void Grid(KernelBuilder k, Val n, Action<Val> body) => k.For(k.GlobalX, n, body, k.GridStrideX);

    // One workgroup per row: group g takes rows g, g + groups, … (uniform within the group, so barriers are allowed).
    private static void EachRow(KernelBuilder k, Val rows, Action<Val> body) => k.For(k.GroupX, rows, body, k.GroupsX);

    // tanh(u) = 1 - 2 / (e^(2u) + 1): saturates cleanly to ±1 where e^(2u) overflows or underflows (as the CPU's vector path).
    private static Val Tanh(KernelBuilder k, Val u) => 1f - 2f / (k.Exp(u + u) + 1f);

    private static Val Sigmoid(KernelBuilder k, Val x) => 1f / (1f + k.Exp(-x));

    private static Val Gelu(KernelBuilder k, Val x) => 0.5f * x * (1f + Tanh(k, GeluK * k.Fma(GeluC * x * x, x, x)));

    // d gelu(x) / dx.
    private static Val GeluSlope(KernelBuilder k, Val x)
    {
        var x2 = x * x;
        var t = Tanh(k, GeluK * k.Fma(GeluC * x2, x, x));
        return 0.5f * (1f + t) + 0.5f * x * (1f - t * t) * GeluK * k.Fma(k.Float(3f * GeluC), x2, k.Float(1f));
    }

    private static IEnumerable<(string, Func<SpirvKernel>)> ElementwiseKernels()
    {
        yield return ("fill", () =>
        {
            var k = new KernelBuilder("fill", Block);
            var y = k.Buffer("y");
            var n = k.PushInt("n");
            var value = k.PushFloat("value");
            Grid(k, n, i => y[i] = value);
            return k.Build();
        });

        yield return ("copy", () =>
        {
            var k = new KernelBuilder("copy", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var n = k.PushInt("n");
            Grid(k, n, i => y[i] = x[i]);
            return k.Build();
        });

        foreach (var op in Enum.GetValues<UnaryOp>())
        {
            string name = "unary_" + Snake(op);
            yield return (name, () =>
            {
                var k = new KernelBuilder(name, Block);
                var (x, y) = (k.Buffer("x"), k.Buffer("y"));
                var n = k.PushInt("n");
                Grid(k, n, i =>
                {
                    var v = x[i];
                    y[i] = op switch
                    {
                        UnaryOp.Sigmoid => Sigmoid(k, v),
                        UnaryOp.Tanh => Tanh(k, v),
                        UnaryOp.Relu => k.Max(v, k.Float(0f)),
                        UnaryOp.Square => v * v,
                        UnaryOp.Abs => k.Abs(v),
                        UnaryOp.Exp => k.Exp(v),
                        UnaryOp.Log => k.Log(v),
                        _ => Gelu(k, v),
                    };
                });
                return k.Build();
            });
        }

        // Every backward kernel takes the same four buffers (x, y = op(x), dy, dx), whichever of x and y it reads.
        foreach (var op in Enum.GetValues<UnaryOp>())
        {
            string name = "unary_backward_" + Snake(op);
            yield return (name, () =>
            {
                var k = new KernelBuilder(name, Block);
                var (x, y, dy, dx) = (k.Buffer("x"), k.Buffer("y"), k.Buffer("dy"), k.Buffer("dx"));
                var n = k.PushInt("n");
                Grid(k, n, i =>
                {
                    var g = dy[i];
                    Val d = op switch
                    {
                        UnaryOp.Sigmoid => k.Fma(g, y[i] * (1f - y[i]), dx[i]),
                        UnaryOp.Tanh => k.Fma(g, 1f - y[i] * y[i], dx[i]),
                        UnaryOp.Relu => dx[i] + k.Select(x[i] > 0f, g, k.Float(0f)),
                        UnaryOp.Square => k.Fma(2f * x[i], g, dx[i]),
                        UnaryOp.Abs => dx[i] + k.Select(x[i] > 0f, g, k.Select(x[i] < 0f, -g, k.Float(0f))),
                        UnaryOp.Exp => k.Fma(g, y[i], dx[i]),
                        UnaryOp.Log => dx[i] + g / x[i],
                        _ => k.Fma(g, GeluSlope(k, x[i]), dx[i]),
                    };
                    dx[i] = d;
                });
                return k.Build();
            });
        }

        foreach (var op in Enum.GetValues<BinaryOp>())
        {
            string name = "binary_" + op.ToString().ToLowerInvariant();
            yield return (name, () =>
            {
                var k = new KernelBuilder(name, Block);
                var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
                var n = k.PushInt("n");
                Grid(k, n, i => c[i] = op switch
                {
                    BinaryOp.Add => a[i] + b[i],
                    BinaryOp.Sub => a[i] - b[i],
                    _ => a[i] * b[i],
                });
                return k.Build();
            });
        }

        yield return ("affine", () =>
        {
            var k = new KernelBuilder("affine", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (n, alpha, beta) = (k.PushInt("n"), k.PushFloat("alpha"), k.PushFloat("beta"));
            Grid(k, n, i => y[i] = k.Fma(x[i], alpha, beta));
            return k.Build();
        });

        yield return ("axpy", () =>
        {
            var k = new KernelBuilder("axpy", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (n, alpha) = (k.PushInt("n"), k.PushFloat("alpha"));
            Grid(k, n, i => y[i] = k.Fma(x[i], alpha, y[i]));
            return k.Build();
        });

        yield return ("mul_add", () =>
        {
            var k = new KernelBuilder("mul_add", Block);
            var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
            var n = k.PushInt("n");
            Grid(k, n, i => c[i] = k.Fma(a[i], b[i], c[i]));
            return k.Build();
        });

        // c[r, j] = a[r, j] + v[j] over n = rows · cols elements.
        yield return ("add_row_vector", () =>
        {
            var k = new KernelBuilder("add_row_vector", Block);
            var (a, v, c) = (k.Buffer("a"), k.Buffer("v"), k.Buffer("c"));
            var (n, cols) = (k.PushInt("n"), k.PushInt("cols"));
            Grid(k, n, i => c[i] = a[i] + v[i % cols]);
            return k.Build();
        });

        // y[j] += Σ_r x[r, j]: one invocation per column. TODO(tuning): split the rows across a workgroup for few columns.
        yield return ("sum_rows", () =>
        {
            var k = new KernelBuilder("sum_rows", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (rows, cols) = (k.PushInt("rows"), k.PushInt("cols"));
            Grid(k, cols, j =>
            {
                var acc = k.Local(y[j]);
                k.For(k.Int(0), rows, 1, r => acc.V = acc.V + x[r * cols + j]);
                y[j] = acc.V;
            });
            return k.Build();
        });

        // y[i] += scale · s[0].
        yield return ("add_broadcast_scalar", () =>
        {
            var k = new KernelBuilder("add_broadcast_scalar", Block);
            var (s, y) = (k.Buffer("s"), k.Buffer("y"));
            var (n, scale) = (k.PushInt("n"), k.PushFloat("scale"));
            Grid(k, n, i => y[i] = y[i] + s[k.Int(0)] * scale);
            return k.Build();
        });

        // y[offset] += alpha · x[0] (one group; one invocation does it).
        yield return ("axpy_at", () =>
        {
            var k = new KernelBuilder("axpy_at", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (offset, alpha) = (k.PushInt("offset"), k.PushFloat("alpha"));
            k.If(k.GlobalX.Eq(0), () => y[offset] = y[offset] + alpha * x[k.Int(0)]);
            return k.Build();
        });

        yield return ("inv_sqrt", () =>
        {
            var k = new KernelBuilder("inv_sqrt", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (n, eps) = (k.PushInt("n"), k.PushFloat("eps"));
            Grid(k, n, i => y[i] = 1f / k.Sqrt(x[i] + eps));
            return k.Build();
        });

        // y[i] = gelu(x[i] + bias[i % cols]).
        yield return ("bias_gelu", () =>
        {
            var k = new KernelBuilder("bias_gelu", Block);
            var (x, bias, y) = (k.Buffer("x"), k.Buffer("bias"), k.Buffer("y"));
            var (n, cols) = (k.PushInt("n"), k.PushInt("cols"));
            Grid(k, n, i => y[i] = Gelu(k, x[i] + bias[i % cols]));
            return k.Build();
        });

        // y = act(gate) · up; kind 0 = SiLU, 1 = GELU (tanh), anything else = ReLU (uniform branches on the push constant).
        yield return ("gated_activation", () =>
        {
            var k = new KernelBuilder("gated_activation", Block);
            var (gate, up, y) = (k.Buffer("gate"), k.Buffer("up"), k.Buffer("y"));
            var (n, kind) = (k.PushInt("n"), k.PushInt("kind"));
            Grid(k, n, i =>
            {
                var (value, _) = Activation(k, gate[i], kind, slope: false);
                y[i] = value.V * up[i];
            });
            return k.Build();
        });

        // dgate (+)= dy · up · act'(gate) with flags bit 0 (bit 2: =), dup (+)= dy · act(gate) with bit 1 (bit 3: =).
        yield return ("gated_activation_backward", () =>
        {
            var k = new KernelBuilder("gated_activation_backward", Block);
            var (gate, up, dy, dgate, dup) = (k.Buffer("gate"), k.Buffer("up"), k.Buffer("dy"), k.Buffer("dgate"), k.Buffer("dup"));
            var (n, kind, flags) = (k.PushInt("n"), k.PushInt("kind"), k.PushInt("flags"));
            Grid(k, n, i =>
            {
                var (value, slope) = Activation(k, gate[i], kind, slope: true);
                k.If((flags & 1).Ne(0), () =>
                {
                    var keep = k.Select((flags & 4).Ne(0), k.Float(0f), dgate[i]);
                    dgate[i] = keep + dy[i] * up[i] * slope!.V;
                });
                k.If((flags & 2).Ne(0), () =>
                {
                    var keep = k.Select((flags & 8).Ne(0), k.Float(0f), dup[i]);
                    dup[i] = keep + dy[i] * value.V;
                });
            });
            return k.Build();
        });

        // SGD with momentum: v = momentum · v + g; p -= lr · v. (Without momentum the backend runs axpy with -lr.)
        yield return ("sgd_momentum", () =>
        {
            var k = new KernelBuilder("sgd_momentum", Block);
            var (p, g, v) = (k.Buffer("p"), k.Buffer("g"), k.Buffer("v"));
            var (n, lr, momentum) = (k.PushInt("n"), k.PushFloat("lr"), k.PushFloat("momentum"));
            Grid(k, n, i =>
            {
                var velocity = k.Fma(momentum, v[i], g[i]);
                v[i] = velocity;
                p[i] = k.Fma(-lr, velocity, p[i]);
            });
            return k.Build();
        });

        // Adam with a bias-corrected lr: m, v updated in place; p -= lr · m / (sqrt(v) + eps).
        yield return ("adam", () =>
        {
            var k = new KernelBuilder("adam", Block);
            var (p, g, m, v) = (k.Buffer("p"), k.Buffer("g"), k.Buffer("m"), k.Buffer("v"));
            var (n, lr, beta1, beta2, eps) = (k.PushInt("n"), k.PushFloat("lr"), k.PushFloat("beta1"), k.PushFloat("beta2"), k.PushFloat("eps"));
            Grid(k, n, i =>
            {
                var grad = g[i];
                var mom = k.Fma(beta1, m[i], (1f - beta1) * grad);
                var vel = k.Fma(beta2, v[i], (1f - beta2) * grad * grad);
                m[i] = mom;
                v[i] = vel;
                p[i] = p[i] - lr * mom / (k.Sqrt(vel) + eps);
            });
            return k.Build();
        });

        // result[0] = scale · Σ x: one workgroup. TODO(tuning): partial sums over many groups, then a second pass.
        yield return ("sum", () =>
        {
            var k = new KernelBuilder("sum", Block);
            var (x, result) = (k.Buffer("x"), k.Buffer("result"));
            var (n, scale) = (k.PushInt("n"), k.PushFloat("scale"));
            var scratch = k.Shared("scratch", Block);
            var acc = k.Local(0f);
            k.For(k.LocalX, n, Block, i => acc.V = acc.V + x[i]);
            var total = k.ReduceSum(scratch, acc.V);
            k.If(k.LocalX.Eq(0), () => result[k.Int(0)] = total * scale);
            return k.Build();
        });

        // total[0] += Σ x²: one workgroup.
        yield return ("sum_squares", () =>
        {
            var k = new KernelBuilder("sum_squares", Block);
            var (x, total) = (k.Buffer("x"), k.Buffer("total"));
            var n = k.PushInt("n");
            var scratch = k.Shared("scratch", Block);
            var acc = k.Local(0f);
            k.For(k.LocalX, n, Block, i => acc.V = k.Fma(x[i], x[i], acc.V));
            var sum = k.ReduceSum(scratch, acc.V);
            k.If(k.LocalX.Eq(0), () => total[k.Int(0)] = total[k.Int(0)] + sum);
            return k.Build();
        });
    }

    // act(g) (and act'(g) when asked) for the gated activations, chosen at run time by kind (0 SiLU, 1 GELU, else ReLU).
    private static (Var Value, Var? Slope) Activation(KernelBuilder k, Val g, Val kind, bool slope)
    {
        var value = k.Local(ScalarKind.Float);
        var d = slope ? k.Local(ScalarKind.Float) : null;
        k.If(kind.Eq(0), () =>
        {
            value.V = g / (1f + k.Exp(-g));                              // as the CPU's forward pass
            if (d is not null)
            {
                var s = Sigmoid(k, g);
                d.V = s * (1f + g * (1f - s));
            }
        }, () => k.If(kind.Eq(1), () =>
        {
            value.V = Gelu(k, g);
            if (d is not null)
            {
                d.V = GeluSlope(k, g);
            }
        }, () =>
        {
            value.V = k.Max(g, k.Float(0f));
            if (d is not null)
            {
                d.V = k.Select(g > 0f, k.Float(1f), k.Float(0f));
            }
        }));
        return (value, d);
    }
}
