// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Gpu.Hip;

/// <summary>Thrown when a HIP runtime call or kernel fails (a <see cref="DeviceException"/>: reported to telemetry).</summary>
public sealed class HipException(string message) : DeviceException("hip", message);

/// <summary>
/// Bindings to the HIP runtime (libamdhip64.so on Linux, ROCm's runtime library; amdhip64_N.dll on Windows, installed by
/// the graphics driver), loaded at run time and called through function pointers resolved by name. Nothing is bound at
/// build time, so a machine without the runtime loads the library as before: the probe finds no library, reports why,
/// and there are no HIP devices. Entry points a runtime release lacks are null (optional ones) or make the probe fail
/// with their name (required ones), never a crash at the first call.
/// </summary>
internal static unsafe class HipRuntime
{
    private static IntPtr s_handle;

    /// <summary>The file the runtime was opened from (a path, or a bare name the system's search found); empty until loaded.</summary>
    public static string LoadedFrom { get; private set; } = "";

    /// <summary>The runtime library's file names tried in order on <paramref name="os"/> ("windows" or "linux"; none elsewhere).</summary>
    internal static string[] CandidateNames(string os) => os switch
    {
        // The runtime's major version is part of the file name (amdhip64_6.dll); older drivers install amdhip64.dll.
        "windows" => ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"],
        "linux" => ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5"],
        _ => [],
    };

    /// <summary>
    /// Folders the runtime is looked for in when the default search does not find it: on Linux, the ROCm root named by
    /// ROCM_PATH, then ROCm's default prefix. On Windows none: the system's search finds the runtime the graphics driver
    /// installed, and no SDK's folder is looked in (the library depends on no part of the HIP SDK).
    /// </summary>
    internal static IEnumerable<string> SearchFolders(string os, Func<string, string?> environment)
    {
        if (os != "linux")
        {
            yield break;
        }

        if (environment("ROCM_PATH") is { Length: > 0 } root)
        {
            yield return Path.Combine(root, "lib");
        }

        yield return "/opt/rocm/lib";
    }

    internal static string CurrentOs => OperatingSystem.IsWindows() ? "windows" : OperatingSystem.IsLinux() ? "linux" : "";

    /// <summary>
    /// Opens the first library of <paramref name="names"/> found by the system's default search, else in
    /// <paramref name="folders"/> (ROCm's on Linux); false (with the names tried) when none is there; <paramref name="opened"/>
    /// is the name or path opened. The default search comes first: on Windows it finds the runtime the graphics driver
    /// installed, the one matched to that driver (a newer runtime beside an older driver can crash the process inside
    /// hipInit, which no handler catches). IDRAK_HIP_RUNTIME names another runtime to try first.
    /// Touches no shared state, so tests can ask for libraries that do not exist.
    /// </summary>
    internal static bool TryOpen(IReadOnlyList<string> names, IEnumerable<string> folders, out IntPtr handle, out string reason, out string opened)
    {
        foreach (string name in names)
        {
            if (NativeLibrary.TryLoad(name, out handle))
            {
                (reason, opened) = ("", name);
                return true;
            }
        }

        foreach (string folder in folders)
        {
            foreach (string name in names)
            {
                string path = Path.Combine(folder, name);
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out handle))
                {
                    (reason, opened) = ("", path);
                    return true;
                }
            }
        }

        handle = IntPtr.Zero;
        opened = "";
        reason = names.Count == 0 ? "HIP is not used on this operating system"
            : $"the HIP runtime ({string.Join(" / ", names)}) was not found; install the GPU's graphics driver (Windows) or ROCm's runtime (Linux)";
        return false;
    }

    /// <summary>Loads the runtime and resolves its entry points; false (with a reason) when it is missing or too old.</summary>
    public static bool TryLoad(out string reason)
    {
        if (s_handle != IntPtr.Zero)
        {
            reason = "";
            return true;
        }

        string os = CurrentOs;
        string[] names = Environment.GetEnvironmentVariable("IDRAK_HIP_RUNTIME") is { Length: > 0 } chosen
            ? [chosen.Trim(), .. CandidateNames(os)] : CandidateNames(os);                  // a runtime (name or path) to try first
        if (!TryOpen(names, SearchFolders(os, Environment.GetEnvironmentVariable), out var handle, out reason, out string opened))
        {
            return false;
        }

        var missing = new List<string>();
        IntPtr Required(string name)
        {
            if (NativeLibrary.TryGetExport(handle, name, out var address))
            {
                return address;
            }

            missing.Add(name);
            return IntPtr.Zero;
        }

        IntPtr Optional(string name) => NativeLibrary.TryGetExport(handle, name, out var address) ? address : IntPtr.Zero;

        hipInit = (delegate* unmanaged<uint, int>)Required(nameof(hipInit));
        hipDriverGetVersion = (delegate* unmanaged<int*, int>)Required(nameof(hipDriverGetVersion));
        hipRuntimeGetVersion = (delegate* unmanaged<int*, int>)Required(nameof(hipRuntimeGetVersion));
        hipGetDeviceCount = (delegate* unmanaged<int*, int>)Required(nameof(hipGetDeviceCount));
        hipDeviceGet = (delegate* unmanaged<int*, int, int>)Required(nameof(hipDeviceGet));
        hipDeviceGetName = (delegate* unmanaged<byte*, int, int, int>)Required(nameof(hipDeviceGetName));
        hipDeviceGetAttribute = (delegate* unmanaged<int*, int, int, int>)Required(nameof(hipDeviceGetAttribute));
        hipDeviceTotalMem = (delegate* unmanaged<nuint*, int, int>)Required(nameof(hipDeviceTotalMem));
        hipSetDevice = (delegate* unmanaged<int, int>)Required(nameof(hipSetDevice));
        hipMalloc = (delegate* unmanaged<ulong*, nuint, int>)Required(nameof(hipMalloc));
        hipFree = (delegate* unmanaged<ulong, int>)Required(nameof(hipFree));
        hipMemGetInfo = (delegate* unmanaged<nuint*, nuint*, int>)Required(nameof(hipMemGetInfo));
        hipMemcpyAsync = (delegate* unmanaged<void*, void*, nuint, int, IntPtr, int>)Required(nameof(hipMemcpyAsync));
        hipMemcpy2DAsync = (delegate* unmanaged<void*, nuint, void*, nuint, nuint, nuint, int, IntPtr, int>)Required(nameof(hipMemcpy2DAsync));
        hipMemsetD32Async = (delegate* unmanaged<ulong, int, nuint, IntPtr, int>)Required(nameof(hipMemsetD32Async));
        hipStreamCreate = (delegate* unmanaged<IntPtr*, int>)Required(nameof(hipStreamCreate));
        hipStreamSynchronize = (delegate* unmanaged<IntPtr, int>)Required(nameof(hipStreamSynchronize));
        hipModuleLoadData = (delegate* unmanaged<IntPtr*, void*, int>)Required(nameof(hipModuleLoadData));
        hipModuleGetFunction = (delegate* unmanaged<IntPtr*, IntPtr, byte*, int>)Required(nameof(hipModuleGetFunction));
        hipModuleLaunchKernel = (delegate* unmanaged<IntPtr, uint, uint, uint, uint, uint, uint, uint, IntPtr, void**, void**, int>)
            Required(nameof(hipModuleLaunchKernel));

        // Optional: names and properties (the full device-properties call was renamed when its struct grew in 6.0),
        // events (for timing, unused until kernels are measured), and error names.
        hipGetDevicePropertiesR0600 = (delegate* unmanaged<byte*, int, int>)Optional(nameof(hipGetDevicePropertiesR0600));
        hipGetDeviceProperties = (delegate* unmanaged<byte*, int, int>)Optional(nameof(hipGetDeviceProperties));
        hipDeviceGetUuid = (delegate* unmanaged<byte*, int, int>)Optional(nameof(hipDeviceGetUuid));
        hipEventCreate = (delegate* unmanaged<IntPtr*, int>)Optional(nameof(hipEventCreate));
        hipEventRecord = (delegate* unmanaged<IntPtr, IntPtr, int>)Optional(nameof(hipEventRecord));
        hipEventSynchronize = (delegate* unmanaged<IntPtr, int>)Optional(nameof(hipEventSynchronize));
        hipEventElapsedTime = (delegate* unmanaged<float*, IntPtr, IntPtr, int>)Optional(nameof(hipEventElapsedTime));
        hipGetErrorName = (delegate* unmanaged<int, byte*>)Optional(nameof(hipGetErrorName));
        hipStreamQuery = (delegate* unmanaged<IntPtr, int>)Optional(nameof(hipStreamQuery));   // polled waits (HipBackend.WaitStream)
        hipModuleUnload = (delegate* unmanaged<IntPtr, int>)Optional(nameof(hipModuleUnload));   // GCN modules given back (HipBackend.GcnFunction)

        if (missing.Count > 0)
        {
            NativeLibrary.Free(handle);
            reason = $"the HIP runtime found lacks {string.Join(", ", missing)} (a newer graphics driver or ROCm runtime is needed)";
            return false;
        }

        s_handle = handle;                                                      // kept for the process
        LoadedFrom = opened;
        reason = "";
        return true;
    }

    public static void Check(int result, string call)
    {
        if (result != Success)
        {
            throw new HipException($"{call} failed: {Describe(result)}");
        }
    }

    // The runtime's name for an error code, with the code.
    public static string Describe(int result)
    {
        string name = "hipError";
        if (hipGetErrorName != null && hipGetErrorName(result) is var text && text != null)
        {
            name = Marshal.PtrToStringAnsi((IntPtr)text) ?? name;
        }

        return $"{name} ({result})";
    }

    // Error codes (hipError_t).
    public const int Success = 0, ErrorOutOfMemory = 2, ErrorNoDevice = 100, ErrorNotReady = 600;

    // hipModuleLaunchKernel's `extra` markers (HIP_LAUNCH_PARAM_*): the kernel arguments' buffer, its size, the end.
    public const nint LaunchParamBufferPointer = 1, LaunchParamBufferSize = 2, LaunchParamEnd = 3;

    // hipMemcpyKind.
    public const int MemcpyHostToDevice = 1, MemcpyDeviceToHost = 2, MemcpyDeviceToDevice = 3;

    // hipDeviceAttribute_t (the CUDA-compatible range, numbered as since ROCm 5.0). Each value read is checked for
    // plausibility (HipDeviceLimits.Problem): a runtime numbering them differently disables the kernels instead of
    // launching them with wrong sizes.
    public const int AttributeIntegrated = 16, AttributeL2CacheSize = 19, AttributeComputeCapabilityMajor = 23,
        AttributeMaxBlockDimX = 26, AttributeMaxGridDimX = 29, AttributeMaxGridDimY = 30, AttributeMaxGridDimZ = 31, AttributeMaxThreadsPerBlock = 56,
        AttributeMemoryBusWidth = 59, AttributeComputeCapabilityMinor = 61, AttributeMultiprocessorCount = 63,
        AttributePciBusId = 67, AttributePciDeviceId = 68, AttributePciDomainId = 69, AttributeMaxSharedMemoryPerBlock = 74,
        AttributeWarpSize = 87;

    public static delegate* unmanaged<uint, int> hipInit;
    public static delegate* unmanaged<int*, int> hipDriverGetVersion;
    public static delegate* unmanaged<int*, int> hipRuntimeGetVersion;
    public static delegate* unmanaged<int*, int> hipGetDeviceCount;
    public static delegate* unmanaged<int*, int, int> hipDeviceGet;
    public static delegate* unmanaged<byte*, int, int, int> hipDeviceGetName;
    public static delegate* unmanaged<int*, int, int, int> hipDeviceGetAttribute;
    public static delegate* unmanaged<nuint*, int, int> hipDeviceTotalMem;
    public static delegate* unmanaged<int, int> hipSetDevice;
    public static delegate* unmanaged<ulong*, nuint, int> hipMalloc;
    public static delegate* unmanaged<ulong, int> hipFree;
    public static delegate* unmanaged<nuint*, nuint*, int> hipMemGetInfo;
    public static delegate* unmanaged<void*, void*, nuint, int, IntPtr, int> hipMemcpyAsync;
    public static delegate* unmanaged<void*, nuint, void*, nuint, nuint, nuint, int, IntPtr, int> hipMemcpy2DAsync;
    public static delegate* unmanaged<ulong, int, nuint, IntPtr, int> hipMemsetD32Async;
    public static delegate* unmanaged<IntPtr*, int> hipStreamCreate;
    public static delegate* unmanaged<IntPtr, int> hipStreamSynchronize;
    public static delegate* unmanaged<IntPtr, int> hipStreamQuery;
    public static delegate* unmanaged<IntPtr*, void*, int> hipModuleLoadData;
    public static delegate* unmanaged<IntPtr*, IntPtr, byte*, int> hipModuleGetFunction;
    public static delegate* unmanaged<IntPtr, uint, uint, uint, uint, uint, uint, uint, IntPtr, void**, void**, int> hipModuleLaunchKernel;
    public static delegate* unmanaged<IntPtr, int> hipModuleUnload;
    public static delegate* unmanaged<byte*, int, int> hipGetDevicePropertiesR0600;
    public static delegate* unmanaged<byte*, int, int> hipGetDeviceProperties;
    public static delegate* unmanaged<byte*, int, int> hipDeviceGetUuid;
    public static delegate* unmanaged<IntPtr*, int> hipEventCreate;
    public static delegate* unmanaged<IntPtr, IntPtr, int> hipEventRecord;
    public static delegate* unmanaged<IntPtr, int> hipEventSynchronize;
    public static delegate* unmanaged<float*, IntPtr, IntPtr, int> hipEventElapsedTime;
    public static delegate* unmanaged<int, byte*> hipGetErrorName;
}
