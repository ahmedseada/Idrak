// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Element-wise kernels with parameters or several gradients: powers, clamping, selection by a mask and the gradient of
// element-wise extremes. A gradient buffer that is not needed is bound to another storage and never written (flags).
internal static partial class VulkanKernels
{
    // x^p from pow(|x|, p), which GLSL leaves undefined for x < 0 and for 0^p with p ≤ 0: negative x take the sign the
    // host worked out for p (1 for even integers, -1 for odd ones, NaN otherwise) and zero takes p's value at zero.
    private static Val PowOf(KernelBuilder k, Val x, Val p, Val negative, Val atZero)
    {
        var r = k.Pow(k.Abs(x), p);
        return k.Select(x.Eq(k.Float(0f)), atZero, k.Select(x < 0f, r * negative, r));
    }

    private static IEnumerable<(string, Func<SpirvKernel>)> PointwiseKernels()
    {
        yield return ("pow", () =>
        {
            var k = new KernelBuilder("pow", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (n, p, negative, atZero) = (k.PushInt("n"), k.PushFloat("p"), k.PushFloat("negative"), k.PushFloat("atZero"));
            Grid(k, n, i => y[i] = PowOf(k, x[i], p, negative, atZero));
            return k.Build();
        });

        // dx += dy · (p + 1) · x^p, with p the exponent less one (and the power's parts for it).
        yield return ("pow_backward", () =>
        {
            var k = new KernelBuilder("pow_backward", Block);
            var (x, dy, dx) = (k.Buffer("x"), k.Buffer("dy"), k.Buffer("dx"));
            var (n, p, negative, atZero) = (k.PushInt("n"), k.PushFloat("p"), k.PushFloat("negative"), k.PushFloat("atZero"));
            Grid(k, n, i => dx[i] = k.Fma(dy[i] * (p + 1f), PowOf(k, x[i], p, negative, atZero), dx[i]));
            return k.Build();
        });

        yield return ("clamp", () =>
        {
            var k = new KernelBuilder("clamp", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (n, min, max) = (k.PushInt("n"), k.PushFloat("min"), k.PushFloat("max"));
            Grid(k, n, i => y[i] = k.Min(k.Max(x[i], min), max));
            return k.Build();
        });

        yield return ("clamp_backward", () =>
        {
            var k = new KernelBuilder("clamp_backward", Block);
            var (x, dy, dx) = (k.Buffer("x"), k.Buffer("dy"), k.Buffer("dx"));
            var (n, min, max) = (k.PushInt("n"), k.PushFloat("min"), k.PushFloat("max"));
            Grid(k, n, i => dx[i] = dx[i] + k.Select((x[i] >= min) & (x[i] <= max), dy[i], k.Float(0f)));
            return k.Build();
        });

        yield return ("where", () =>
        {
            var k = new KernelBuilder("where", Block);
            var (condition, a, b, y) = (k.Buffer("condition"), k.Buffer("a"), k.Buffer("b"), k.Buffer("y"));
            var n = k.PushInt("n");
            Grid(k, n, i => y[i] = k.Select(!condition[i].Eq(k.Float(0f)), a[i], b[i]));
            return k.Build();
        });

        // da += dy where the condition is non-zero (flags & 1), db += dy elsewhere (flags & 2).
        yield return ("where_backward", () =>
        {
            var k = new KernelBuilder("where_backward", Block);
            var (condition, dy, da, db) = (k.Buffer("condition"), k.Buffer("dy"), k.Buffer("da"), k.Buffer("db"));
            var (n, flags) = (k.PushInt("n"), k.PushInt("flags"));
            var (first, second) = ((flags & 1).Ne(0), (flags & 2).Ne(0));
            Grid(k, n, i =>
            {
                var chosen = !condition[i].Eq(k.Float(0f));
                k.If(first & chosen, () => da[i] = da[i] + dy[i]);
                k.If(second & !chosen, () => db[i] = db[i] + dy[i]);
            });
            return k.Build();
        });

        // The gradient of max(a, b) (flags & 4 clear) or min(a, b): to a where it won (ties to a), else to b; flags as above.
        yield return ("extremum_backward", () =>
        {
            var k = new KernelBuilder("extremum_backward", Block);
            var (a, b, dy, da, db) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("dy"), k.Buffer("da"), k.Buffer("db"));
            var (n, flags) = (k.PushInt("n"), k.PushInt("flags"));
            var (first, second, minimum) = ((flags & 1).Ne(0), (flags & 2).Ne(0), (flags & 4).Ne(0));
            Grid(k, n, i =>
            {
                var chosen = (minimum & (a[i] <= b[i])) | (!minimum & (a[i] >= b[i]));
                k.If(first & chosen, () => da[i] = da[i] + dy[i]);
                k.If(second & !chosen, () => db[i] = db[i] + dy[i]);
            });
            return k.Build();
        });
    }
}
