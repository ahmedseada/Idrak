// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Adad.Devices.Cuda;
using Idrak.Adad.Types;
using static Idrak.Adad.Devices.Cuda.NativeKernels;

namespace Idrak.Adad.Tests;

internal static class PtxTests
{
    public static void ControlKernelsAreTheSource(Checks t)
    {
        t.Check(NativeMmaConfigs.Length == Recorded.Native.Length, $"{NativeMmaConfigs.Length} native shapes, the source's {Recorded.Native.Length}");
        for (int i = 0; i < Math.Min(NativeMmaConfigs.Length, Recorded.Native.Length); i++)
        {
            string text = NativeMma(NativeKind.BF16, NativeMmaConfigs[i], out string entry);
            string sha = Data.ShaText(text);
            t.Check(entry == Recorded.Native[i].Entry && sha == Recorded.Native[i].Sha, $"native shape {i}: {entry} {sha}, the source's {Recorded.Native[i].Entry} {Recorded.Native[i].Sha}");
        }

        string pack = Data.ShaText(NativePackX(NativeKind.BF16));
        t.Check(pack == Recorded.PackX, $"the x pack's text {pack}, the source's {Recorded.PackX}");
        string widen = Data.ShaText(SuperKernels.WidenModule(SuperBlocks.Q4K));
        t.Check(widen == Recorded.Widen, $"q4_k's widening module {widen}, the source's {Recorded.Widen}");
        string sum = Data.ShaText(TypedSumModule());
        t.Check(sum == Recorded.Sum, $"the splits' sum module {sum}, the source's {Recorded.Sum}");
    }
}
