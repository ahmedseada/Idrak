// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Backends.Vulkan;

/// <summary>
/// The SPIR-V kernels of the Vulkan backend, generated in C# with <see cref="KernelBuilder"/> (no shader compiler, no
/// precompiled binaries), each matching its <see cref="Backend"/> operation and the CPU's math. A kernel is built on its
/// first use. Dispatch shapes:
/// <list type="bullet">
/// <item>element-wise kernels (local size: the device's width, see <see cref="WidthFor"/>) loop over their n elements with
/// a grid stride: any number of groups works; ⌈n / width⌉ capped at the device's group count is the natural choice;</item>
/// <item>row kernels (one workgroup per row) loop over the rows with a stride of the group count: min(rows, the device's
/// group count) groups;</item>
/// <item>single-group kernels (<c>sum</c>, <c>sum_squares</c>, <c>axpy_at</c>, <c>clip_factor</c>) take exactly one group;</item>
/// <item>the tiled products (<c>batched_matmul</c>, <c>batched_matmul_tile</c>) take one group per block of c along x and
/// y and the batches along z;</item>
/// <item>the packed products (<c>int8_gemv_R</c>, …) take column blocks × splits of k × row blocks, and the attention
/// kernels query rows × splits of the positions (see their comments); their splits are added by a second pass.</item>
/// </list>
/// Integers that tensors hold as values (indices, positions) are read as floats and converted, as the CPU does; packed
/// words (int8, int4, bfloat16) are read as bits.
/// </summary>
internal static partial class VulkanKernels
{
    /// <summary>
    /// Invocations per workgroup of the kernel being built (its 1-D width; the reductions are unrolled for it): the
    /// width the device was given (<see cref="WidthFor"/>), set while <see cref="Get(string, int, bool)"/> builds a
    /// kernel, and <see cref="MaxWidth"/> otherwise.
    /// </summary>
    public static int Block => t_width != 0 ? t_width : MaxWidth;

    /// <summary>
    /// Widest workgroup the kernels are built for: the test list builds and validates every kernel at each power of two
    /// up to it (devices reporting more get this width).
    /// </summary>
    public const int MaxWidth = 1024;

    /// <summary>Narrowest workgroup the kernels are built for (32 packed words × 2 slices of k).</summary>
    public const int MinWidth = 64;

    /// <summary>Largest head size the attention kernels take: the library's decoding limit (the CPU backend's
    /// DecodeAttentionHeadDim), independent of the width.</summary>
    public const int AttentionMaxDim = 256;

    // GELU, tanh approximation: 0.5 x (1 + tanh(k (x + 0.044715 x³))), k = sqrt(2/π) (the CPU's constants).
    private const float GeluK = 0.7978845608f;
    private const float GeluC = 0.044715f;

    [ThreadStatic]
    private static int t_width;

    private static readonly ConcurrentDictionary<(string Name, int Width, bool Subgroups), SpirvKernel> Built = new();

    private static readonly Lazy<Dictionary<string, Func<SpirvKernel>>> LazyFactories = new(() =>
    {
        var all = new Dictionary<string, Func<SpirvKernel>>();
        foreach (var (name, build) in ElementwiseKernels().Concat(RowKernels()).Concat(ShapeKernels()).Concat(MatMulKernels()).Concat(DecodingKernels())
            .Concat(SamplingKernels()).Concat(ConvKernels()).Concat(PromptKernels()).Concat(TrainingKernels()).Concat(FusedKernels()))
        {
            all.Add(name, build);
        }

        return all;
    });

    /// <summary>Every kernel's name.</summary>
    public static IEnumerable<string> Names => LazyFactories.Value.Keys;

    /// <summary>The kernel with this name for workgroups of <see cref="MaxWidth"/>, built on first use.</summary>
    public static SpirvKernel Get(string name) => Get(name, MaxWidth);

    /// <summary>The kernel with this name for workgroups of <paramref name="width"/> (a power of two in
    /// [<see cref="MinWidth"/>, <see cref="MaxWidth"/>]), its reductions through subgroup arithmetic when
    /// <paramref name="subgroups"/> (for devices that report it), built on first use.</summary>
    public static SpirvKernel Get(string name, int width, bool subgroups = false)
    {
        if (width is < MinWidth or > MaxWidth || (width & (width - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), width, $"A power of two from {MinWidth} to {MaxWidth}.");
        }

        return Built.GetOrAdd((name, width, subgroups), static key =>
        {
            if (!LazyFactories.Value.TryGetValue(key.Name, out var build))
            {
                throw new KeyNotFoundException($"No Vulkan kernel named '{key.Name}'.");
            }

            var (savedWidth, savedSubgroups) = (t_width, KernelBuilder.t_subgroups);
            (t_width, KernelBuilder.t_subgroups) = (key.Width, key.Subgroups);
            try
            {
                return build();
            }
            finally
            {
                (t_width, KernelBuilder.t_subgroups) = (savedWidth, savedSubgroups);
            }
        });
    }

    /// <summary>
    /// Workgroup memory the widest kernel declares at a width, at most (bytes): the packed products' partial sums (8
    /// floats per invocation), or the sampler's 2,048-score slice with three arrays of the width. The test list checks
    /// every kernel against it.
    /// </summary>
    public static int SharedBytesBound(int width) => Math.Max(32 * width, 12 * width + 8448);

    /// <summary>Subgroups per workgroup the width aims for: enough for a workgroup to keep loads in flight while some of
    /// its subgroups wait, few enough that a barrier waits for few of them.</summary>
    public const int SubgroupsPerWorkgroup = 8;

    /// <summary>
    /// The workgroup width for a device, from what it reports: <see cref="SubgroupsPerWorkgroup"/> subgroups (a power of
    /// two, at least <see cref="MinWidth"/>), halved while it exceeds the device's invocations per workgroup, its
    /// workgroup width or (with <see cref="SharedBytesBound"/>) its workgroup memory. The tuned operations also try half
    /// and twice the width.
    /// </summary>
    public static int WidthFor(int maxInvocations, int maxSizeX, int sharedBytes, int subgroupSize)
    {
        int width = (int)Math.Clamp(System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, subgroupSize * SubgroupsPerWorkgroup)), MinWidth, MaxWidth);
        while (width > MinWidth && (width > maxInvocations || width > maxSizeX || SharedBytesBound(width) > sharedBytes))
        {
            width /= 2;
        }

        return width;
    }

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
