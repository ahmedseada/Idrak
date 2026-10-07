// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Operations;

namespace Idrak.Gpu.Hip;

/// <summary>
/// A compute kernel for a HIP device (<see cref="Launch(Backend, uint, uint, uint, uint, uint, uint, ReadOnlySpan{KernelArgument}, uint)"/>):
/// HIP C++ source and the name of one of its <c>extern "C" __global__</c> functions. Each device compiles a source with
/// hipRTC once, for its own architecture, on the first launch there of a kernel from it, and keeps the module (kernels
/// built from the same source and options share it); the code object is also kept in the HIP kernel cache on disk
/// (IDRAK_HIP_KERNEL_CACHE), so a later run loads it without compiling. The parameters are read from the function's
/// declaration (a pointer: a storage; <c>int</c>, <c>unsigned</c>: a 32-bit integer; <c>float</c>; <c>long long</c>,
/// <c>size_t</c>, <c>int64_t</c>: a 64-bit integer) and every launch is checked against them. A launch goes on the
/// device's stream, in order with its other work. The library's own HIP kernels are HIP C compiled the same way; a
/// plug-in ships the source of its own and launches it from a kernel it registers for the "hip" kind
/// (<see cref="Kernels.Register(Operation, string, Delegate, Func{Backend, bool})"/>). A compiler error, no hipRTC on the
/// machine or a failed launch raise <see cref="HipException"/>.
/// </summary>
/// <example>
/// <code>
/// const string Source = """
///     extern "C" __global__ void my_scale_rows(const float* rows, const float* scales, float* output, int n, int width)
///     {
///         for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i &lt; n; i += (long long)blockDim.x * gridDim.x)
///             output[i] = rows[i] * scales[i / width];
///     }
///     """;
/// static readonly HipKernel Scale = new(Source, "my_scale_rows");
///
/// Kernels.Register(MyOps.ScaleRows, "hip", (b, rows, scales, output, n, width) =&gt;
///     Scale.Launch(b, (uint)Math.Clamp((n + 255) / 256, 1, 65535), 1, 1, 256, 1, 1, [rows, scales, output, n, width]));
/// </code>
/// </example>
public sealed class HipKernel
{
    /// <summary>
    /// A kernel from HIP C++ source and the name of its <c>extern "C" __global__</c> function; <paramref name="options"/>
    /// are passed to hipRTC after the library's own (<c>-O3</c> and the device's architecture), <paramref name="name"/>
    /// is for messages (the function's when null).
    /// </summary>
    /// <exception cref="ArgumentException">The source declares no <c>__global__</c> function named <paramref name="entry"/>.</exception>
    public HipKernel(string source, string entry, string? name = null, IReadOnlyList<string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(entry);
        Name = name ?? entry;
        Parameters = ReadParameters(source, entry)
                     ?? throw new ArgumentException($"HIP kernel '{Name}': the source declares no __global__ function named '{entry}'.", nameof(entry));
        Source = source;
        Entry = entry;
        Options = options is null ? [] : [.. options];
    }

    /// <summary>The HIP C++ source.</summary>
    public string Source { get; }

    /// <summary>The <c>extern "C" __global__</c> function launched.</summary>
    public string Entry { get; }

    /// <summary>A name for messages.</summary>
    public string Name { get; }

    /// <summary>hipRTC options added to the library's own.</summary>
    public IReadOnlyList<string> Options { get; }

    /// <summary>
    /// Its parameters as read from the source, one letter each: 'p' a storage (a pointer), 'i' a 32-bit integer, 'f' a
    /// float, 'l' a 64-bit integer, '?' a type a launch cannot pass (double, a structure, a plain <c>long</c>, whose
    /// width differs between Linux and Windows).
    /// </summary>
    public string Parameters { get; }

    /// <summary>
    /// Queues the kernel on the HIP device <paramref name="backend"/> over gridX × gridY × gridZ blocks of
    /// blockX × blockY × blockZ threads, with <paramref name="arguments"/> in the order the function declares its
    /// parameters (storages of that device, integers, floats) and <paramref name="sharedBytes"/> of dynamic shared
    /// memory. It runs in order with the device's other work; reading a storage back waits for it. A grid with a zero
    /// dimension launches nothing. The first launch on a device compiles the source there (or reads it from the cache).
    /// </summary>
    /// <exception cref="ArgumentException">
    /// <paramref name="backend"/> is not a HIP device, a storage is not on it, or the arguments do not match the parameters.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">The grid, the block or the shared memory exceed the device's limits.</exception>
    /// <exception cref="HipException">hipRTC is missing or refused the source, or the runtime refused the launch.</exception>
    public void Launch(Backend backend, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, ReadOnlySpan<KernelArgument> arguments, uint sharedBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (backend is not HipBackend hip)
        {
            throw new ArgumentException($"HIP kernel '{Name}' runs on a HIP device, not on {backend.Name} ({backend.Kind}).", nameof(backend));
        }

        KernelArgument.CheckAgainst(Name, Parameters, arguments);
        hip.Launch(this, gridX, gridY, gridZ, blockX, blockY, blockZ, arguments, sharedBytes);
    }

    /// <summary>
    /// Why plug-in kernels cannot be compiled for <paramref name="backend"/> (null when they can): it is not a HIP
    /// device, or hipRTC is not on the machine (the HIP runtime alone runs code objects but does not compile them). A
    /// plug-in registers its kernel with the requirement <c>b =&gt; HipKernel.UnavailableReason(b) is null</c>, so such a
    /// machine runs the operation's default kernel instead of failing.
    /// </summary>
    public static string? UnavailableReason(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return backend is not HipBackend ? $"{backend.Name} ({backend.Kind}) is not a HIP device"
            : HipRtc.UnavailableReason is { Length: > 0 } missing ? missing
            : null;
    }

    /// <summary>The function loaded for it on the device that launched it last (looked up without a dictionary).</summary>
    internal (HipBackend Owner, IntPtr Function)? LastFunction;

    // The letters of the parameters of __global__ function `entry`, or null when the source declares none of that name.
    internal static string? ReadParameters(string source, string entry)
    {
        var match = Regex.Match(source, $@"__global__(?:[^;{{(]|\([^)]*\))*?\b{Regex.Escape(entry)}\s*\(([^)]*)\)");
        if (!match.Success)
        {
            return null;
        }

        string list = match.Groups[1].Value.Trim();
        if (list.Length == 0 || list == "void")
        {
            return "";
        }

        var letters = new StringBuilder();
        foreach (string declaration in list.Split(','))
        {
            string d = Regex.Replace(declaration, @"\s+", " ").Trim();
            letters.Append(
                d.Contains('*', StringComparison.Ordinal) ? 'p'
                : Regex.IsMatch(d, @"\b(double|struct)\b") || d.Contains('[', StringComparison.Ordinal) ? '?'
                : Regex.IsMatch(d, @"\bfloat\b") ? 'f'
                : Regex.IsMatch(d, @"\blong long\b|\b(u?int64_t|size_t|ptrdiff_t)\b") ? 'l'
                : Regex.IsMatch(d, @"\blong\b") ? '?'
                : Regex.IsMatch(d, @"\b(int|unsigned|u?int32_t)\b") ? 'i'
                : '?');
        }

        return letters.ToString();
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}
