// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;

namespace Idrak.PluginTests;

/// <summary>
/// The kernels a plug-in writes for a GPU, read as the library reads them: a CUDA kernel's parameters from its PTX
/// (<see cref="CudaKernel"/>) and a HIP kernel's arguments from its GCN text's .arg lines (<see cref="GcnKernel"/>). The
/// test runner in tests/Idrak.Tests runs <see cref="All"/> on every device; a failed check throws.
/// </summary>
public static class KernelPluginTests
{
    /// <summary>The tests, by name.</summary>
    public static IReadOnlyList<(string Name, Action<Device> Run)> All { get; } =
    [
        ("outside plug-in kernels: CUDA kernels read their parameters from the PTX, HIP's GCN kernels theirs from the .arg lines (refusing text whose declarations do not read, with the line and column)", GpuKernelShapes),
    ];

    private static void GpuKernelShapes(Device device)
    {
        Check(new CudaKernel(".visible .entry k(.param .u64 a, .param .b32 b, .param .f32 c, .param .s32 d, .param .align 8 .b8 e[16]) { ret; }", "k").Parameters == "xwfi?",
            "PTX parameter types");

        // GCN: the arguments from the .arg lines ('p' a buffer, 'w' a 4-byte value, 'l' an 8-byte one).
        Check(new GcnKernel(".kernel k\n.arg a, 8, global\n.arg b, 4, value\n.arg c, 8, value\n.arg d, 2, value\n  s_endpgm\n", "k").Parameters == "pwl?", "GCN argument kinds");
        string? refused = Message(() => new GcnKernel(".kernel k\n.arg a, 8, global\n.arg b, 4, bytes\n  s_endpgm\n", "k"));
        Check(refused is not null && refused.Contains("line 3, column 12", StringComparison.Ordinal), $"GCN declarations refused with the line and column: {refused}");
    }

    // The message of the ArgumentException the action throws, or null when it throws none.
    private static string? Message(Action action)
    {
        try
        {
            action();
            return null;
        }
        catch (ArgumentException e)
        {
            return e.Message;
        }
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
