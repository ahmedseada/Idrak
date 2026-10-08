// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Operations;

namespace Idrak.Gpu.Cuda;

/// <summary>
/// A compute kernel for a CUDA device (<see cref="Launch(Backend, uint, uint, uint, uint, uint, uint, ReadOnlySpan{KernelArgument}, uint)"/>):
/// PTX text and the name of one of its <c>.entry</c> functions. Each device JIT-compiles a PTX text once, on the first
/// launch there of a kernel from it, and keeps the module (kernels built from the same text share it); the function is
/// looked up once per device too. The parameters are read from the entry's declaration, and every launch is checked
/// against them: a <c>.u64</c>/<c>.s64</c>/<c>.b64</c> parameter takes a storage or a 64-bit integer (a storage only when
/// declared <c>.ptr</c>), <c>.u32</c>/<c>.s32</c> an integer, <c>.f32</c> a float, <c>.b32</c> either. A launch goes on
/// the device's stream, in order with its other work, and into a CUDA graph while one records. The library's own kernels
/// are generated in C#; a plug-in ships the PTX of its own (from <c>nvcc -ptx</c>, say: target the oldest
/// architecture it supports, the driver compiles it for newer ones) and launches it from a kernel it registers for the
/// "cuda" kind (<see cref="Kernels.Register(Operation, string, Delegate, Func{Backend, bool})"/>). Driver errors (a PTX
/// the JIT compiler refuses, a failed launch) raise <see cref="CudaException"/>.
/// </summary>
/// <example>
/// <code>
/// static readonly CudaKernel Scale = new(ScalePtx, "my_scale_rows");
///
/// Kernels.Register(MyOps.ScaleRows, "cuda", (b, rows, scales, output, n, width) =&gt;
///     Scale.Launch(b, (uint)Math.Clamp((n + 255) / 256, 1, 65535), 1, 1, 256, 1, 1, [rows, scales, output, n, width]));
/// </code>
/// </example>
public sealed class CudaKernel
{
    /// <summary>A kernel from PTX text and the name of its <c>.entry</c>; <paramref name="name"/> is for messages and the profiler (the entry's when null).</summary>
    /// <exception cref="ArgumentException">The text declares no <c>.entry</c> named <paramref name="entry"/>.</exception>
    public CudaKernel(string ptx, string entry, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(ptx);
        ArgumentException.ThrowIfNullOrWhiteSpace(entry);
        Name = name ?? entry;
        Parameters = ReadParameters(ptx, entry)
                     ?? throw new ArgumentException($"CUDA kernel '{Name}': the PTX declares no .entry named '{entry}'.", nameof(entry));
        Ptx = ptx;
        Entry = entry;
    }

    /// <summary>The PTX text.</summary>
    public string Ptx { get; }

    /// <summary>The <c>.entry</c> function launched.</summary>
    public string Entry { get; }

    /// <summary>A name for messages and the profiler.</summary>
    public string Name { get; }

    /// <summary>
    /// Its parameters as read from the PTX, one letter each: 'p' a storage (<c>.u64 .ptr</c>), 'x' a storage or 64-bit
    /// integer (<c>.u64</c>, <c>.s64</c>, <c>.b64</c>), 'i' a 32-bit integer, 'f' a float, 'w' either (<c>.b32</c>), '?'
    /// a type a launch cannot pass (an array or another width).
    /// </summary>
    public string Parameters { get; }

    /// <summary>
    /// Queues the kernel on the CUDA device <paramref name="backend"/> over gridX × gridY × gridZ blocks of
    /// blockX × blockY × blockZ threads, with <paramref name="arguments"/> in the order the entry declares its parameters
    /// (storages of that device, integers, floats) and <paramref name="sharedBytes"/> of dynamic shared memory. It runs
    /// in order with the device's other work; reading a storage back waits for it. A grid with a zero dimension launches
    /// nothing. The first launch on a device loads the PTX there (launch it once before recording a graph when the
    /// module is large: the JIT compilation is not part of the graph either way).
    /// </summary>
    /// <exception cref="ArgumentException">
    /// <paramref name="backend"/> is not a CUDA device, a storage is not on it, or the arguments do not match the parameters.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">The grid, the block or the shared memory exceed the device's limits.</exception>
    /// <exception cref="CudaException">The driver refused the PTX or the launch.</exception>
    public void Launch(Backend backend, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, ReadOnlySpan<KernelArgument> arguments, uint sharedBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (backend is not CudaBackend cuda)
        {
            throw new ArgumentException($"CUDA kernel '{Name}' runs on a CUDA device, not on {backend.Name} ({backend.Kind}).", nameof(backend));
        }

        KernelArgument.CheckAgainst(Name, Parameters, arguments);
        cuda.Launch(this, gridX, gridY, gridZ, blockX, blockY, blockZ, arguments, sharedBytes);
    }

    /// <summary>The function loaded for it on the device that launched it last (looked up without a dictionary; one
    /// reference, so a thread never reads one device's owner with another's function).</summary>
    internal LoadedFunction? LastFunction;

    /// <summary>A function of this kernel loaded on a device.</summary>
    internal sealed record LoadedFunction(CudaBackend Owner, IntPtr Function);

    // The letters of the parameters of .entry `entry`, or null when the text declares none of that name.
    internal static string? ReadParameters(string ptx, string entry)
    {
        var match = Regex.Match(ptx, $@"\.entry\s+{Regex.Escape(entry)}\s*\(([^)]*)\)");
        if (!match.Success)
        {
            return null;
        }

        var letters = new StringBuilder();
        foreach (string declaration in match.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string[] tokens = declaration.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0 || tokens[0] != ".param")
            {
                continue;
            }

            bool array = declaration.Contains('[', StringComparison.Ordinal);
            bool pointer = tokens.Any(t => t.StartsWith(".ptr", StringComparison.Ordinal));
            string? type = tokens.Skip(1).FirstOrDefault(t => t is ".u64" or ".s64" or ".b64" or ".u32" or ".s32" or ".b32" or ".f32"
                or ".u8" or ".s8" or ".b8" or ".u16" or ".s16" or ".b16" or ".f16" or ".f64");
            letters.Append(array ? '?' : type switch
            {
                ".u64" or ".s64" or ".b64" => pointer ? 'p' : 'x',
                ".u32" or ".s32" => 'i',
                ".f32" => 'f',
                ".b32" => 'w',
                _ => '?',
            });
        }

        return letters.ToString();
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}
