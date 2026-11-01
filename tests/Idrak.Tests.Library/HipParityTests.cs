// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;

// Plan 35, HIP parity, on HIP devices (the owner's hip:0 run; nothing runs elsewhere): every method CUDA overrides is
// HIP's own too, and each gives the CPU's results: the partial upload, the pinned host staging, the packed product's
// epilogue, the token cross-entropy rows, the packed gated activation, the convolution's three paths (the composed one,
// the 16 × 16 and the 64 × 64 implicit products, these two the same bits), the measured attention path, and no quantized
// operand to reuse.
internal abstract partial class Tests
{
    internal static void HipDeviceParity(Device device)
    {
        if (device.Type != DeviceType.Hip)
        {
            return;
        }

        var hip = (HipBackend)device.Backend;
        var cpu = CpuBackend.Instance;
        var random = new Random(91);
        var owned = new List<Storage>();
        float[] Values(int n, float low = -2f, float high = 2f) => [.. Enumerable.Range(0, n).Select(_ => low + random.NextSingle() * (high - low))];
        Storage On(Backend backend, float[] values)
        {
            var s = backend.Allocate(values.Length, zeroed: false);
            owned.Add(s);
            backend.Upload(values, s);
            return s;
        }

        float[] Read(Backend backend, Storage s)
        {
            var values = new float[s.Length];
            backend.Download(s, values);
            return values;
        }

        try
        {
            long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);

            // UploadRange: straight to the offset, the rest untouched.
            float[] base0 = Values(300), piece = Values(100);
            var target = On(hip, base0);
            hip.UploadRange(piece, target, 50);
            float[] expected = [.. base0];
            piece.CopyTo(expected, 50);
            AssertClose(expected, Read(hip, target), 0f, "UploadRange");

            // Host staging: pinned slots copied on the work stream, at a slot's start and anywhere in it.
            using (var staging = hip.CreateHostStaging(2, 1000))
            {
                Check(staging is ISlotStaging, "HIP's host staging is a slot staging (the runtime's pinned memory)");
                float[] pinned = Values(500);
                pinned.CopyTo(staging!.Slot(0));
                var destination = On(hip, new float[600]);
                staging.Upload(0, destination, 10, 500);
                staging.Wait(0);
                AssertClose(pinned, Read(hip, destination)[10..510], 0f, "staging upload");
                ((ISlotStaging)staging).Download(destination, 10, 1, 200, 500);
                staging.Wait(1);
                AssertClose(pinned, staging.Slot(1)[200..700].ToArray(), 0f, "staging download at an offset in the slot");
            }

            // The packed product's epilogue (bfloat16 weights, prompt-sized): with the bias, the residual, and GELU.
            const int M = 20, N = 70, K = 48;
            float[] x = Values(M * K), w = Values(K * N, -0.3f, 0.3f), bias = Values(N), residual = Values(M * N);
            var packedCpu = cpu.Allocate((K * N + 1) / 2, zeroed: false);
            owned.Add(packedCpu);
            cpu.PackBFloat16(On(cpu, w), packedCpu, K * N);
            float[] packedWords = Read(cpu, packedCpu);
            foreach (var (useResidual, gelu) in new[] { (false, false), (true, false), (false, true) })
            {
                var yCpu = cpu.Allocate(M * N, zeroed: false);
                var yHip = hip.Allocate(M * N, zeroed: false);
                owned.Add(yCpu);
                owned.Add(yHip);
                cpu.PackedMatMulEpilogue(PackedFormat.BFloat16, On(cpu, x), packedCpu, null, yCpu, M, N, K, On(cpu, bias), useResidual ? On(cpu, residual) : null, gelu);
                hip.PackedMatMulEpilogue(PackedFormat.BFloat16, On(hip, x), On(hip, packedWords), null, yHip, M, N, K, On(hip, bias), useResidual ? On(hip, residual) : null, gelu);
                AssertClose(Read(cpu, yCpu), Read(hip, yHip), 1e-4f, $"PackedMatMulEpilogue{(useResidual ? ", the residual" : "")}{(gelu ? ", GELU" : "")}");
            }

            // The token cross-entropy rows and the packed gated activation.
            const int Rows = 3, Vocabulary = 1500;
            float[] logits = Values(Rows * Vocabulary, -4, 4), targets = [5, 1499, 700], weights = [1f, 0f, 0.5f];
            var (lc, lh) = (On(cpu, logits), On(hip, logits));
            var (lossC, lossH) = (On(cpu, new float[Rows]), On(hip, new float[Rows]));
            cpu.SoftmaxCrossEntropyRows(lc, On(cpu, targets), On(cpu, weights), lossC, Rows, Vocabulary, 0.5f);
            hip.SoftmaxCrossEntropyRows(lh, On(hip, targets), On(hip, weights), lossH, Rows, Vocabulary, 0.5f);
            AssertClose(Read(cpu, lossC), Read(hip, lossH), 1e-4f, "SoftmaxCrossEntropyRows losses");
            AssertClose(Read(cpu, lc), Read(hip, lh), 1e-4f, "SoftmaxCrossEntropyRows gradient");
            const int G = 1001;
            float[] gate = Values(G), up = Values(G);
            var outputs = new (Storage Y, Storage Py, Storage Pg, Storage Pu)[2];
            int at = 0;
            foreach (Backend b in new Backend[] { cpu, hip })
            {
                outputs[at] = (On(b, new float[G]), On(b, new float[(G + 1) / 2]), On(b, new float[(G + 1) / 2]), On(b, new float[(G + 1) / 2]));
                b.GatedActivationPacked(On(b, gate), On(b, up), outputs[at].Pg, outputs[at].Pu, outputs[at].Y, outputs[at].Py, G, 0, 14);
                at++;
            }

            AssertClose(Read(cpu, outputs[0].Y), Read(hip, outputs[1].Y), 1e-5f, "GatedActivationPacked y");
            Check(Read(cpu, outputs[0].Pg).SequenceEqual(Read(hip, outputs[1].Pg)), "GatedActivationPacked: gate packed as the CPU packs it");

            // The convolution's paths: composed, 16 × 16 and 64 × 64 implicit products; the two implicit ones the same bits.
            var g = new ConvGeometry(2, 8, 13, 11, 3, 3, 1, 1, 1, 1);
            const int Filters = 40;
            float[] cx = Values(g.N * g.C * g.H * g.W), cw = Values(Filters * g.PatchSize, -0.2f, 0.2f), cb = Values(Filters);
            var cy = cpu.Allocate(g.N * Filters * g.OH * g.OW, zeroed: false);
            owned.Add(cy);
            cpu.Convolution(On(cpu, cx), On(cpu, cw), On(cpu, cb), cy, g, Filters, 1, ConvActivation.Relu);
            float[]? tile = null;
            foreach (int path in new[] { 0, 1, 2 })
            {
                HipBackend.ConvolutionPath = path;
                try
                {
                    var hy = hip.Allocate(g.N * Filters * g.OH * g.OW, zeroed: false);
                    owned.Add(hy);
                    hip.Convolution(On(hip, cx), On(hip, cw), On(hip, cb), hy, g, Filters, 1, ConvActivation.Relu);
                    float[] got = Read(hip, hy);
                    AssertClose(Read(cpu, cy), got, 1e-4f, $"convolution, path {path}");
                    if (path == 1)
                    {
                        tile = got;
                    }
                    else if (path == 2)
                    {
                        Check(got.Select(BitConverter.SingleToUInt32Bits).SequenceEqual(tile!.Select(BitConverter.SingleToUInt32Bits)),
                            "the 64 x 64 implicit product gives the 16 x 16 one's bits");
                    }
                }
                finally
                {
                    HipBackend.ConvolutionPath = null;
                }
            }

            // The attention path: answered (measured once, then kept), and no quantized operand to reuse.
            const int Heads = 2, AttentionRows = 64, Dim = 32;
            var q = On(hip, Values(Heads * AttentionRows * Dim));
            var keys = On(hip, Values(Heads * AttentionRows * Dim));
            var values = On(hip, Values(Heads * AttentionRows * Dim));
            var starts = On(hip, new float[AttentionRows]);
            var ends = On(hip, [.. Enumerable.Range(1, AttentionRows).Select(r => (float)r)]);
            bool first = hip.PrefersComposedAttention(q, keys, values, starts, ends, null, Heads, AttentionRows, AttentionRows, Dim, 0.125f);
            Check(hip.PrefersComposedAttention(q, keys, values, starts, ends, null, Heads, AttentionRows, AttentionRows, Dim, 0.125f) == first,
                "PrefersComposedAttention keeps its choice for the shape");
            Check(hip.ReuseQuantizedOperands() is null, "HIP's products quantize no operand: nothing to reuse");
            Check(Idrak.Abstraction.Operations.Kernels.HostCalls(hip) == before, "none of it took the host fallback");
        }
        finally
        {
            foreach (var s in owned)
            {
                s.Release();
            }
        }
    }
}
