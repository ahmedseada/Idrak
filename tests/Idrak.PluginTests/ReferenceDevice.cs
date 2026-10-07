// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Testing;

namespace Idrak.PluginTests;

/// <summary>
/// A device written outside the library on the public device API alone (plan 10, phase 5): memory is a plain array per
/// storage, a few operations are plain loops, and every other operation takes the host fallback the device contract
/// gives (<see cref="Backend"/>'s default kernels). It is found by name only ("reference:0"), and it passes the
/// conformance kit (<see cref="Conformance.Check(Device, DeviceCheckOptions?)"/>).
/// </summary>
public sealed class ReferenceDevice : Backend
{
    /// <summary>The one device.</summary>
    public static ReferenceDevice Instance { get; } = new();

    private readonly MemoryAccountant _memory = new(() => ComputeResources.GpuMemoryLimit, "reference:0");

    private ReferenceDevice()
    {
    }

    /// <inheritdoc />
    public override string Kind => ReferenceProvider.Name;

    /// <inheritdoc />
    public override string Name => "reference device (plain loops, host fallbacks)";

    /// <summary>A small device: few-row products up to 4 rows, attention heads up to 256.</summary>
    public override BackendCapabilities Capabilities { get; } = new()
    {
        FewRows = 4,
        DecodeAttentionHeadDim = 256,
        TiledAttentionHeadDim = 256,
        MatrixUnits = false,
        MatrixUnitAttentionHeadDim = static _ => false,
        FusedKernels = false,
        Profiling = false,
    };

    /// <summary>The operations run by this device's own loops so far.</summary>
    public long OwnCalls => Interlocked.Read(ref _ownCalls);

    private long _ownCalls;

    /// <inheritdoc />
    public override Storage Allocate(int length, bool zeroed)
    {
        _memory.MustReleaseCacheFor(length * 4L);                            // throws over the limit
        _memory.Allocated(length * 4L);
        return new ArrayStorage(this, new float[length]);
    }

    /// <inheritdoc />
    public override void Return(Storage storage)
    {
        _memory.Returned(storage.Length * 4L);
        _memory.Freed(storage.Length * 4L);                                  // no pool: freed at once
    }

    /// <inheritdoc />
    protected override void Detach(Storage storage) => ((ArrayStorage)storage).Data = null!;

    /// <inheritdoc />
    protected override void Attach(Storage storage, Storage fresh) => ((ArrayStorage)storage).Data = ((ArrayStorage)fresh).Data;

    /// <inheritdoc />
    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    /// <inheritdoc />
    public override void ReleaseCachedMemory()
    {
    }

    /// <inheritdoc />
    public override void Upload(ReadOnlySpan<float> source, Storage destination) => source.CopyTo(D(destination));

    /// <inheritdoc />
    public override void Download(Storage source, Span<float> destination) => D(source).AsSpan(0, destination.Length).CopyTo(destination);

    /// <inheritdoc />
    public override void DownloadRange(Storage source, int offset, Span<float> destination) => D(source).AsSpan(offset, destination.Length).CopyTo(destination);

    /// <inheritdoc />
    public override void Synchronize()
    {
    }

    // ------------------------------------------------------------------ the operations run by plain loops

    /// <inheritdoc />
    public override void FillKernel(Storage y, int n, float value)
    {
        Own();
        Array.Fill(D(y), value, 0, n);
    }

    /// <inheritdoc />
    public override void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        Func<float, float>? f = op switch
        {
            UnaryOp.Relu => v => MathF.Max(v, 0f),
            UnaryOp.Square => v => v * v,
            UnaryOp.Abs => MathF.Abs,
            UnaryOp.Tanh => MathF.Tanh,
            UnaryOp.Sigmoid => v => 1f / (1f + MathF.Exp(-v)),
            _ => null,
        };
        if (f is null)
        {
            base.UnaryKernel(op, x, y, n);                                   // the host fallback for the others
            return;
        }

        Own();
        float[] xv = D(x), yv = D(y);
        for (int i = 0; i < n; i++)
        {
            yv[i] = f(xv[i]);
        }
    }

    /// <inheritdoc />
    public override void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        Own();
        float[] av = D(a), bv = D(b), cv = D(c);
        for (int i = 0; i < n; i++)
        {
            cv[i] = op switch
            {
                BinaryOp.Add => av[i] + bv[i],
                BinaryOp.Sub => av[i] - bv[i],
                BinaryOp.Mul => av[i] * bv[i],
                BinaryOp.Maximum => MathF.Max(av[i], bv[i]),
                _ => MathF.Min(av[i], bv[i]),
            };
        }
    }

    /// <inheritdoc />
    public override void AffineKernel(Storage x, Storage y, int n, float alpha, float beta)
    {
        Own();
        float[] xv = D(x), yv = D(y);
        for (int i = 0; i < n; i++)
        {
            yv[i] = alpha * xv[i] + beta;
        }
    }

    /// <inheritdoc />
    public override void AxpyKernel(Storage x, Storage y, int n, float alpha)
    {
        Own();
        float[] xv = D(x), yv = D(y);
        for (int i = 0; i < n; i++)
        {
            yv[i] += alpha * xv[i];
        }
    }

    /// <inheritdoc />
    public override void SumKernel(Storage x, Storage result, int n, float scale)
    {
        Own();
        double sum = 0;
        foreach (float v in D(x).AsSpan(0, n))
        {
            sum += v;
        }

        D(result)[0] = (float)(scale * sum);
    }

    /// <inheritdoc />
    public override void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        Own();
        float[] av = D(a), bv = D(b), cv = D(c);
        for (int z = 0; z < batch; z++)
        {
            int ao = z * m * k, bo = z * k * n, co = z * m * n;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        sum += (double)av[ao + (transA ? p * m + i : i * k + p)] * bv[bo + (transB ? j * k + p : p * n + j)];
                    }

                    cv[co + i * n + j] = (float)(sum + (beta == 0f ? 0.0 : beta * cv[co + i * n + j]));
                }
            }
        }
    }

    /// <inheritdoc />
    public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        Own();
        float[] xv = D(x), yv = D(y);
        for (int r = 0; r < rows; r++)
        {
            int o = r * cols;
            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {
                max = MathF.Max(max, xv[o + j]);
            }

            double sum = 0;
            for (int j = 0; j < cols; j++)
            {
                sum += Math.Exp(xv[o + j] - max);
            }

            for (int j = 0; j < cols; j++)
            {
                yv[o + j] = log ? (float)(xv[o + j] - max - Math.Log(sum)) : (float)(Math.Exp(xv[o + j] - max) / sum);
            }
        }
    }

    private void Own() => Interlocked.Increment(ref _ownCalls);

    private static float[] D(Storage storage) => ((ArrayStorage)storage).Data;

    // A storage holding its values in a managed array.
    private sealed class ArrayStorage(Backend backend, float[] data) : Storage(backend, data.Length)
    {
        public float[] Data = data;
    }
}

/// <summary>Starts <see cref="ReferenceDevice"/>: one device, listed only when asked for by name.</summary>
public sealed class ReferenceProvider : DeviceProvider
{
    /// <summary>The device kind, "reference".</summary>
    public const string Name = "reference";

    /// <inheritdoc />
    public override string Kind => Name;

    /// <inheritdoc />
    public override DeviceType Type => DeviceType.Other;

    /// <inheritdoc />
    public override int Count => 1;

    /// <inheritdoc />
    public override Backend Create(int ordinal) => ReferenceDevice.Instance;

    /// <inheritdoc />
    public override bool IsStarted(int ordinal) => true;

    /// <inheritdoc />
    public override bool Listed(int ordinal) => false;

    /// <inheritdoc />
    public override string? Note(int ordinal) => "a device written outside the library (tests/Idrak.PluginTests): by name only";
}
