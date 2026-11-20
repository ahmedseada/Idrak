// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Adad.Abstractions;
using Idrak.Adad.Measure;
using Idrak.Adad.Types;
using static Idrak.Adad.Devices.Cuda.NativeKernels;

namespace Idrak.Adad.Devices.Cuda;

/// <summary>A lane's scratch for one call (packed or rounded x, widened W, split partials), from the device's pool.</summary>
internal readonly record struct LaneScratch(DeviceBuffer A, DeviceBuffer B, DeviceBuffer Parts);

/// <summary>One kernel of a lane's run, for the kernel bench and its profile: its name, the bytes it moves, its flops.</summary>
public readonly record struct LanePiece(string Kernel, long Bytes, double Flops);

/// <summary>
/// A lane of the prompt product for super-block weights on CUDA: its kernels for one description, its variants (a kernel
/// shape and a k split, as code shape · 16 + splits), and its run. Scratch memory comes from the device's pool on every
/// call (a block of a size is reused by the next call of that size); a call formats no string and builds no list.
/// </summary>
internal abstract class SuperLane(CudaDevice device, Tuner tuner) : IProductLane<SuperBlock>
{
    protected CudaDevice Device { get; } = device;

    protected Tuner Tuner { get; } = tuner;

    protected SuperBlock Description { get; private set; } = null!;

    private CudaKernel _sum = null!;

    // The shapes the driver accepted: their index in the lane's list, the kernel.
    protected List<(int Index, CudaKernel Kernel)> Accepted { get; } = [];

    /// <summary>The shape a lane runs unmeasured: the fastest of the accepted on a 2048³ product, measured once.</summary>
    protected int DeviceShape { get; private set; } = -1;

    /// <summary>Whether the device's shape is measured when the lane is prepared (else the first accepted).</summary>
    public bool MeasureShape { get; set; } = true;

    /// <summary>What the driver's compiler reported for each accepted shape (registers, spilled bytes).</summary>
    public List<string> Compiled { get; } = [];

    public abstract string Name { get; }

    /// <summary>The lane's shapes, in their list's order.</summary>
    public abstract IReadOnlyList<NativeMmaConfig> Configs { get; }

    public abstract bool Fits(SuperBlock description);

    /// <summary>Values of k a block of the shape's pipeline takes (the unit its k splits count in).</summary>
    protected abstract int BlockValues(NativeMmaConfig c);

    /// <summary>Shared bytes the shape takes.</summary>
    protected abstract int SharedBytes(NativeMmaConfig c);

    /// <summary>The shape's module and entry.</summary>
    protected abstract (string Module, string Entry) Kernel(NativeMmaConfig c);

    /// <summary>Loads the lane's own kernels besides the shapes'; false with the reason when refused.</summary>
    protected abstract bool PrepareOwn(out string reason);

    /// <summary>Runs shape <paramref name="shape"/> of the accepted over the zero operands of a 2048³ product (the device shape's measurement).</summary>
    protected abstract void RunSquare(int shape, DeviceBuffer a, DeviceBuffer b, DeviceBuffer c);

    public bool Prepare(SuperBlock description, out string reason)
    {
        Description = description;
        if (!Fits(description))
        {
            reason = $"{Name}: does not read this description";
            return false;
        }

        var sum = Device.LoadModule(Lf(TypedSumModule()), [TypedSumName], static _ => 0, out reason, out _);
        if (sum is null || !PrepareOwn(out reason))
        {
            return false;
        }

        _sum = sum[0];
        string refused = "";
        for (int i = 0; i < Configs.Count; i++)
        {
            int shared = SharedBytes(Configs[i]);
            if (shared > Device.SharedPerBlockOptin)
            {
                refused = $"{Configs[i]} needs {shared} bytes of shared memory, the device allows {Device.SharedPerBlockOptin}";
                continue;
            }

            var (module, entry) = Kernel(Configs[i]);
            var kernels = Device.LoadModule(module, [entry], _ => shared, out string why, out _);
            if (kernels is null)
            {
                refused = why;
                continue;
            }

            Accepted.Add((i, kernels[0]));
            Compiled.Add(string.Create(CultureInfo.InvariantCulture, $"{entry}: {kernels[0].Registers} registers, {kernels[0].SpilledBytes} bytes spilled a thread, {shared} bytes shared"));
        }

        if (Accepted.Count == 0)
        {
            reason = $"{Name}: the driver accepts none of its shapes ({refused})";
            return false;
        }

        // The device's shape: of the accepted, the fastest on a 2048³ product of zeros, measured once (the proof's way).
        reason = "";
        if (!MeasureShape)
        {
            DeviceShape = 0;
            return true;
        }

        const int Size = 2048;
        var a = Device.Allocate(SquareBytes(Size, a: true));
        var b = Device.Allocate(SquareBytes(Size, a: false));
        var c = Device.Allocate((long)Size * Size * 4);
        try
        {
            Device.Fill(a, 0, SquareBytes(Size, a: true) / 4);
            Device.Fill(b, 0, SquareBytes(Size, a: false) / 4);
            int[] candidates = [.. Enumerable.Range(0, Accepted.Count)];
            DeviceShape = Tuner.Measure($"{Name}: device shape, 2048³", candidates, 0, i => RunSquare(i, a, b, c), cold: false);
        }
        finally
        {
            Device.Release(a);
            Device.Release(b);
            Device.Release(c);
        }

        reason = "";
        return true;
    }

    /// <summary>Bytes of the square measurement's x (<paramref name="a"/>) or W operand.</summary>
    protected abstract long SquareBytes(int size, bool a);

    /// <summary>
    /// The k splits a shape may take: 1, 2, 4 and 8, while each split keeps two blocks of k at least (the proof's rule) and
    /// every split starts inside k (the proof's rule alone admits a last split past the end, as 8 splits of 20 blocks: its
    /// kernel then reads past the row; no product measured here has such a k).
    /// </summary>
    internal static bool SplitFits(int blocks, int s) => s == 1 || blocks >= 2 * s && (s - 1) * ((blocks + s - 1) / s) < blocks;

    private static readonly int[] SplitCounts = [1, 2, 4, 8];

    /// <summary>k padded as the lane's x rows are (values a row).</summary>
    protected virtual int Padded(int k) => k;

    public int[] Variants(int m, int n, int k, out int fallback)
    {
        var variants = new List<int>();
        fallback = -1;
        int kp = Padded(k);
        for (int i = 0; i < Accepted.Count; i++)
        {
            int blocks = kp / BlockValues(Configs[Accepted[i].Index]);
            foreach (int s in SplitCounts)
            {
                if (SplitFits(blocks, s))
                {
                    variants.Add(i * 16 + s);
                }
            }
        }

        fallback = DeviceShape * 16 + 1;
        return [.. variants];
    }

    /// <summary>A variant's shape and k split.</summary>
    public (NativeMmaConfig Config, int Splits) Describe(int variant) => (Configs[Accepted[variant / 16].Index], variant % 16);

    public void Run(int variant, in ProductOperands op)
    {
        var scratch = Rent(variant, op);
        int pieces = PieceCount(variant);
        for (int p = 0; p < pieces; p++)
        {
            Launch(p, variant, op, scratch);
        }

        Return(scratch);
    }

    /// <summary>The call's scratch, from the device's pool.</summary>
    public abstract LaneScratch Rent(int variant, in ProductOperands op);

    /// <summary>Gives the call's scratch back to the pool (after its last kernel is enqueued).</summary>
    public void Return(in LaneScratch scratch)
    {
        Device.Release(scratch.A);
        Device.Release(scratch.B);
        Device.Release(scratch.Parts);
    }

    /// <summary>The kernels a run of <paramref name="variant"/> enqueues (its sum only when k is split).</summary>
    public abstract int PieceCount(int variant);

    /// <summary>Enqueues the run's kernel <paramref name="piece"/> (each reads only what the pieces before it wrote, so a piece may run again).</summary>
    public abstract void Launch(int piece, int variant, in ProductOperands op, in LaneScratch scratch);

    /// <summary>Kernel <paramref name="piece"/>'s name, bytes moved and flops (the kernel bench's rates).</summary>
    public abstract LanePiece Piece(int piece, int variant, in ProductOperands op);

    /// <summary>The k split of a variant.</summary>
    protected static int SplitsOf(int variant) => variant % 16;

    /// <summary>The product kernel's launch: grid (n / BN, m / 128, splits), 256 threads, k's blocks split evenly.</summary>
    protected (uint X, uint Y, uint Z, uint PerSplit) Grid(int variant, int m, int n, int kp)
    {
        var (cfg, splits) = Describe(variant);
        int blocks = kp / BlockValues(cfg), per = (blocks + splits - 1) / splits;
        return ((uint)((n + cfg.Bn - 1) / cfg.Bn), (uint)((m + 127) / 128), (uint)splits, (uint)per);
    }

    /// <summary>The kernel of a variant.</summary>
    protected CudaKernel KernelOf(int variant) => Accepted[variant / 16].Kernel;

    /// <summary>The split sums: y = part 0 + part 1 + ... in order.</summary>
    protected void LaunchSum(ulong parts, ulong y, int m, int n, int splits) =>
        Device.Launch(_sum, (uint)((m * n + 255) / 256), 1, 1, 256, [parts, y, (ulong)(m * n), (ulong)splits]);

    /// <summary>A launch with no work (the splits' sum over no values): the host's cost of a launch, for the bench.</summary>
    internal void LaunchEmpty() => Device.Launch(_sum, 1, 1, 1, 256, [0UL, 0UL, 0UL, 1UL]);

    /// <summary>The lane's modules as generated (entry, PTX), its shapes' for every accepted one.</summary>
    internal IEnumerable<(string Entry, string Module)> Modules()
    {
        foreach (var (index, _) in Accepted)
        {
            var (module, entry) = Kernel(Configs[index]);
            yield return (entry, module);
        }
    }

    /// <summary>The split sums' bytes moved.</summary>
    protected static LanePiece SumInfo(int m, int n, int splits) => new(TypedSumName, (long)(splits + 1) * m * n * 4, 0);

    /// <summary>Where the product kernel writes: y unsplit, else the partials.</summary>
    protected static ulong Output(int variant, in ProductOperands op, in LaneScratch scratch) => SplitsOf(variant) > 1 ? scratch.Parts.Pointer : op.Y;

    private DeviceBuffer _counters;
    private long _counterSlots;

    /// <summary>
    /// The tiles' counters of a fused split sum (A and B): one a tile of the grid, kept zero between calls (each tile's
    /// last block sets its counter back), grown (and zeroed once) when a grid needs more.
    /// </summary>
    protected ulong Counters(int variant, int m, int n, int kp)
    {
        var (gx, gy, _, _) = Grid(variant, m, n, kp);
        long slots = (long)gx * gy;
        if (slots > _counterSlots)
        {
            Device.Synchronize();
            Device.Release(_counters);
            _counters = Device.Allocate(slots * 4);
            Device.Fill(_counters, 0, slots);
            _counterSlots = slots;
        }

        return _counters.Pointer;
    }

    /// <summary>The partials of a split variant (none unsplit).</summary>
    protected DeviceBuffer PartsFor(int variant, int m, int n) => SplitsOf(variant) > 1 ? Device.Allocate((long)SplitsOf(variant) * m * n * 4) : DeviceBuffer.None;
}

/// <summary>
/// The control: today's winning path, copied exactly (decision 4): x packed into bfloat16 rows of kp (native_pack_x), the
/// weight's rows widened from the description into bfloat16 rows of kp (super_widen), the native kernel (its shape and k
/// split measured per shape), the splits summed in order (typed_sum_parts). The default of the product.
/// </summary>
internal sealed class ControlLane(CudaDevice device, Tuner tuner) : SuperLane(device, tuner)
{
    private CudaKernel _pack = null!, _widen = null!;

    public override string Name => "control";

    public override IReadOnlyList<NativeMmaConfig> Configs => NativeMmaConfigs;

    public override bool Fits(SuperBlock description) => SuperKernels.SuperFits(description);

    protected override int BlockValues(NativeMmaConfig c) => c.RowBytes * 8 / 16;

    protected override int SharedBytes(NativeMmaConfig c) => NativeMmaShared(c);

    protected override int Padded(int k) => NativePadded(k);

    protected override (string Module, string Entry) Kernel(NativeMmaConfig c)
    {
        string module = NativeMma(NativeKind.BF16, c, out string entry);
        return (Lf(module), entry);
    }

    protected override bool PrepareOwn(out string reason)
    {
        var pack = Device.LoadModule(Lf(NativePackX(NativeKind.BF16)), [NativePackXName(NativeKind.BF16)], static _ => 0, out reason, out _);
        var widen = Device.LoadModule(Lf(SuperKernels.WidenModule(Description)), [SuperKernels.WidenBf16Name(Description)], static _ => 0, out string why, out _);
        if (pack is null || widen is null)
        {
            reason = pack is null ? reason : why;
            return false;
        }

        (_pack, _widen) = (pack[0], widen[0]);
        return true;
    }

    protected override long SquareBytes(int size, bool a) => (long)size * size * 2;

    protected override void RunSquare(int shape, DeviceBuffer a, DeviceBuffer b, DeviceBuffer c)
    {
        int blocks = 2048 / BlockValues(Configs[Accepted[shape].Index]);
        var cfg = Configs[Accepted[shape].Index];
        Device.Launch(Accepted[shape].Kernel, (uint)(2048 / cfg.Bn), 2048 / 128, 1, NativeMmaThreads,
            [a.Pointer, b.Pointer, c.Pointer, 2048, 2048, 2048, 2048, (ulong)blocks]);
    }

    public override LaneScratch Rent(int variant, in ProductOperands op)
    {
        int kp = NativePadded(op.K);
        return new LaneScratch(Device.Allocate((long)op.M * kp * 2), Device.Allocate((long)op.N * kp * 2), PartsFor(variant, op.M, op.N));
    }

    public override int PieceCount(int variant) => SplitsOf(variant) > 1 ? 4 : 3;

    public override void Launch(int piece, int variant, in ProductOperands op, in LaneScratch scratch)
    {
        int m = op.M, n = op.N, k = op.K, kp = NativePadded(k);
        switch (piece)
        {
            case 0:
                var (packX, packY) = NativePackGrid(m, kp);
                Device.Launch(_pack, packX, packY, 1, 256, [op.X, scratch.A.Pointer, (ulong)m, (ulong)k, (ulong)kp]);
                break;
            case 1:
                Device.Launch(_widen, (uint)(((long)n * kp / 8 + 255) / 256), 1, 1, 256, [op.W, scratch.B.Pointer, (ulong)n, (ulong)k, (ulong)kp]);
                break;
            case 2:
                var (gx, gy, gz, per) = Grid(variant, m, n, kp);
                Device.Launch(KernelOf(variant), gx, gy, gz, NativeMmaThreads, [scratch.A.Pointer, scratch.B.Pointer, Output(variant, op, scratch), (ulong)m, (ulong)n, (ulong)kp, (ulong)kp, per]);
                break;
            default:
                LaunchSum(scratch.Parts.Pointer, op.Y, m, n, SplitsOf(variant));
                break;
        }
    }

    public override LanePiece Piece(int piece, int variant, in ProductOperands op)
    {
        int m = op.M, n = op.N, k = op.K, kp = NativePadded(k);
        var s = Description;
        return piece switch
        {
            0 => new(_pack.Name, (long)m * k * 4 + (long)m * kp * 2, 0),
            1 => new(_widen.Name, (long)n * (k / s.Values) * s.Bytes + (long)n * kp * 2, 0),
            2 => new(KernelOf(variant).Name, 0, 2.0 * m * n * k),
            _ => SumInfo(m, n, SplitsOf(variant)),
        };
    }
}

/// <summary>
/// A: x packed as the control packs it, the native kernel reading the stored super-blocks (cp.async into shared memory)
/// and decoding each fragment's values in registers with the widening's exact float32 steps (SuperMmaKernels.DecodeMma).
/// </summary>
internal sealed class DecodeLane(CudaDevice device, Tuner tuner) : SuperLane(device, tuner)
{
    private CudaKernel _pack = null!;

    public override string Name => "A";

    public override IReadOnlyList<NativeMmaConfig> Configs => SuperMmaKernels.DecodeConfigs;

    public override bool Fits(SuperBlock description) => SuperMmaKernels.Fits(description);

    protected override int BlockValues(NativeMmaConfig c) => SuperMmaKernels.DecodeStageValues(c);

    protected override int SharedBytes(NativeMmaConfig c) => SuperMmaKernels.DecodeShared(c);

    protected override (string Module, string Entry) Kernel(NativeMmaConfig c) => (SuperMmaKernels.DecodeModule(Description, c), SuperMmaKernels.DecodeName(Description, c));

    protected override bool PrepareOwn(out string reason)
    {
        var pack = Device.LoadModule(Lf(NativePackX(NativeKind.BF16)), [NativePackXName(NativeKind.BF16)], static _ => 0, out reason, out _);
        if (pack is null)
        {
            return false;
        }

        _pack = pack[0];
        return true;
    }

    protected override long SquareBytes(int size, bool a) => a ? (long)size * size * 2 : (long)size * (size / Description.Values) * Description.Bytes;

    protected override void RunSquare(int shape, DeviceBuffer a, DeviceBuffer b, DeviceBuffer c)
    {
        var cfg = Configs[Accepted[shape].Index];
        int blocks = 2048 / BlockValues(cfg);
        Device.Launch(Accepted[shape].Kernel, (uint)(2048 / cfg.Bn), 2048 / 128, 1, SuperMmaKernels.Threads,
            [a.Pointer, b.Pointer, c.Pointer, 2048, 2048, 2048, 2048, (ulong)blocks, (ulong)(2048 / Description.Values * Description.Bytes), c.Pointer, 0UL]);
    }

    public override LaneScratch Rent(int variant, in ProductOperands op) =>
        new(Device.Allocate((long)op.M * NativePadded(op.K) * 2), DeviceBuffer.None, PartsFor(variant, op.M, op.N));

    public override int PieceCount(int variant) => 2;                              // the split sum is in the kernel

    public override void Launch(int piece, int variant, in ProductOperands op, in LaneScratch scratch)
    {
        int m = op.M, n = op.N, k = op.K, kp = NativePadded(k);
        switch (piece)
        {
            case 0:
                var (packX, packY) = NativePackGrid(m, kp);
                Device.Launch(_pack, packX, packY, 1, 256, [op.X, scratch.A.Pointer, (ulong)m, (ulong)k, (ulong)kp]);
                break;
            case 1:
                var (gx, gy, gz, per) = Grid(variant, m, n, k);
                ulong wrow = (ulong)(k / Description.Values * Description.Bytes);
                Device.Launch(KernelOf(variant), gx, gy, gz, SuperMmaKernels.Threads,
                    [scratch.A.Pointer, op.W, Output(variant, op, scratch), (ulong)m, (ulong)n, (ulong)k, (ulong)kp, per, wrow, op.Y, Counters(variant, m, n, k)]);
                break;
        }
    }

    public override LanePiece Piece(int piece, int variant, in ProductOperands op)
    {
        int m = op.M, n = op.N, k = op.K, kp = NativePadded(k);
        return piece switch
        {
            0 => new(_pack.Name, (long)m * k * 4 + (long)m * kp * 2, 0),
            _ => new(KernelOf(variant).Name, 0, 2.0 * m * n * k),
        };
    }
}

/// <summary>
/// B: x rounded once per call into int8 blocks of 32 with their scales and block sums, int8 tensor cores over the stored
/// codes expanded in registers, a float32 step per 32-value sub-block (SuperMmaKernels.Int8Mma).
/// </summary>
internal sealed class Int8Lane(CudaDevice device, Tuner tuner) : SuperLane(device, tuner)
{
    private CudaKernel _quantize = null!;

    public override string Name => "B";

    public override IReadOnlyList<NativeMmaConfig> Configs => SuperMmaKernels.Int8Configs;

    public override bool Fits(SuperBlock description) => SuperMmaKernels.FitsInt8(description);

    protected override int BlockValues(NativeMmaConfig c) => SuperMmaKernels.Int8StageValues(c);

    protected override int SharedBytes(NativeMmaConfig c) => SuperMmaKernels.Int8Shared(c);

    protected override (string Module, string Entry) Kernel(NativeMmaConfig c) => (SuperMmaKernels.Int8Module(Description, c), SuperMmaKernels.Int8Name(Description, c));

    protected override bool PrepareOwn(out string reason)
    {
        var quantize = Device.LoadModule(SuperMmaKernels.QuantizeModule(), [SuperMmaKernels.QuantizeName], static _ => 0, out reason, out _);
        if (quantize is null)
        {
            return false;
        }

        _quantize = quantize[0];
        return true;
    }

    protected override long SquareBytes(int size, bool a) => a ? (long)size * size + (long)size * (size / 32) * 8 : (long)size * (size / Description.Values) * Description.Bytes;

    protected override void RunSquare(int shape, DeviceBuffer a, DeviceBuffer b, DeviceBuffer c)
    {
        var cfg = Configs[Accepted[shape].Index];
        int blocks = 2048 / BlockValues(cfg);
        Device.Launch(Accepted[shape].Kernel, (uint)(2048 / cfg.Bn), 2048 / 128, 1, SuperMmaKernels.Threads,
            [a.Pointer, a.Pointer + 2048UL * 2048, b.Pointer, c.Pointer, 2048, 2048, 2048, (ulong)blocks, (ulong)(2048 / Description.Values * Description.Bytes), c.Pointer, 0UL]);
    }

    public override LaneScratch Rent(int variant, in ProductOperands op) =>
        new(Device.Allocate((long)op.M * op.K), Device.Allocate((long)op.M * (op.K / 32) * 8), PartsFor(variant, op.M, op.N));

    public override int PieceCount(int variant) => 2;                              // the split sum is in the kernel

    public override void Launch(int piece, int variant, in ProductOperands op, in LaneScratch scratch)
    {
        int m = op.M, n = op.N, k = op.K;
        switch (piece)
        {
            case 0:
                var (qx, qy) = NativePackGrid(m, k);
                Device.Launch(_quantize, qx, qy, 1, 256, [op.X, scratch.A.Pointer, scratch.B.Pointer, (ulong)m, (ulong)k]);
                break;
            case 1:
                var (gx, gy, gz, per) = Grid(variant, m, n, k);
                ulong wrow = (ulong)(k / Description.Values * Description.Bytes);
                Device.Launch(KernelOf(variant), gx, gy, gz, SuperMmaKernels.Threads,
                    [scratch.A.Pointer, scratch.B.Pointer, op.W, Output(variant, op, scratch), (ulong)m, (ulong)n, (ulong)k, per, wrow, op.Y, Counters(variant, m, n, k)]);
                break;
        }
    }

    public override LanePiece Piece(int piece, int variant, in ProductOperands op)
    {
        int m = op.M, n = op.N, k = op.K;
        return piece switch
        {
            0 => new(_quantize.Name, (long)m * k * 4 + (long)m * k + (long)m * (k / 32) * 8, 0),
            _ => new(KernelOf(variant).Name, 0, 2.0 * m * n * k),
        };
    }
}

