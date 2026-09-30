using System.Reflection;
using System.Runtime.InteropServices;

namespace Idrak.Backends.Cuda;

/// <summary>Thrown when a CUDA driver call fails.</summary>
public sealed class CudaException(string message) : Exception(message);

/// <summary>
/// Bindings to the CUDA driver API. The driver ships with the NVIDIA display driver
/// (nvcuda.dll on Windows, libcuda.so.1 on Linux), so nothing beyond a GPU driver is required:
/// no CUDA toolkit, cuBLAS, cuDNN or TensorFlow binaries.
/// </summary>
internal static unsafe partial class CudaDriver
{
    private const string Library = "cuda";

    static CudaDriver()
    {
        NativeLibrary.SetDllImportResolver(typeof(CudaDriver).Assembly, Resolve);
    }

    /// <summary>Loads the driver library; returns false (with a reason) when no NVIDIA driver is installed.</summary>
    public static bool TryLoad(out string reason)
    {
        foreach (var name in CandidateNames())
        {
            if (NativeLibrary.TryLoad(name, out var handle))
            {
                NativeLibrary.Free(handle);
                reason = "";
                return true;
            }
        }

        reason = $"the NVIDIA driver library ({string.Join(" / ", CandidateNames())}) was not found";
        return false;
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Library)
        {
            return IntPtr.Zero;
        }

        foreach (var candidate in CandidateNames())
        {
            if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out var handle))
            {
                return handle;
            }
        }

        return IntPtr.Zero;
    }

    private static string[] CandidateNames() =>
        OperatingSystem.IsWindows() ? ["nvcuda.dll"] :
        OperatingSystem.IsLinux() ? ["libcuda.so.1", "libcuda.so"] :
        [];

    public static void Check(int result, string call)
    {
        if (result != 0)
        {
            throw new CudaException($"{call} failed: {Describe(result)}");
        }
    }

    // The driver's name for an error code, with the code.
    public static string Describe(int result)
    {
        string name = "CUDA_ERROR";
        if (cuGetErrorName(result, out var text) == 0 && text != null)
        {
            name = Marshal.PtrToStringAnsi((IntPtr)text) ?? name;
        }

        return $"{name} ({result})";
    }

    public const int ErrorOutOfMemory = 2;

    public const int JitErrorLogBuffer = 5;
    public const int JitErrorLogBufferSizeBytes = 6;

    public const int AttributeMultiprocessorCount = 16;
    public const int AttributeComputeCapabilityMajor = 75;
    public const int AttributeComputeCapabilityMinor = 76;

    [LibraryImport(Library)]
    public static partial int cuInit(uint flags);

    [LibraryImport(Library)]
    public static partial int cuDriverGetVersion(out int version);

    [LibraryImport(Library)]
    public static partial int cuDeviceGetCount(out int count);

    [LibraryImport(Library)]
    public static partial int cuDeviceGet(out int device, int ordinal);

    [LibraryImport(Library)]
    public static partial int cuDeviceGetName(byte* name, int length, int device);

    [LibraryImport(Library)]
    public static partial int cuDeviceGetAttribute(out int value, int attribute, int device);

    [LibraryImport(Library, EntryPoint = "cuDeviceTotalMem_v2")]
    public static partial int cuDeviceTotalMem(out nuint bytes, int device);

    [LibraryImport(Library, EntryPoint = "cuDevicePrimaryCtxRetain")]
    public static partial int cuDevicePrimaryCtxRetain(out IntPtr context, int device);

    [LibraryImport(Library)]
    public static partial int cuCtxSetCurrent(IntPtr context);

    [LibraryImport(Library)]
    public static partial int cuCtxSynchronize();

    [LibraryImport(Library)]
    public static partial int cuModuleLoadDataEx(out IntPtr module, byte* image, uint numOptions, int* options, void** optionValues);

    [LibraryImport(Library)]
    public static partial int cuModuleGetFunction(out IntPtr function, IntPtr module, byte* name);

    public const int FunctionAttributeMaxDynamicSharedSizeBytes = 8;

    [LibraryImport(Library)]
    public static partial int cuFuncSetAttribute(IntPtr function, int attribute, int value);

    [LibraryImport(Library, EntryPoint = "cuMemAlloc_v2")]
    public static partial int cuMemAlloc(out ulong pointer, nuint bytes);

    [LibraryImport(Library, EntryPoint = "cuMemFree_v2")]
    public static partial int cuMemFree(ulong pointer);

    [LibraryImport(Library, EntryPoint = "cuMemGetInfo_v2")]
    public static partial int cuMemGetInfo(out nuint free, out nuint total);

    [LibraryImport(Library)]
    public static partial int cuMemHostAlloc(out IntPtr pointer, nuint bytes, uint flags);

    [LibraryImport(Library, EntryPoint = "cuMemHostGetDevicePointer_v2")]
    public static partial int cuMemHostGetDevicePointer(out ulong device, IntPtr host, uint flags);

    [LibraryImport(Library)]
    public static partial int cuMemFreeHost(IntPtr pointer);

    public const uint HostAllocPortable = 1, HostAllocDeviceMap = 2;

    [LibraryImport(Library, EntryPoint = "cuMemcpyHtoD_v2")]
    public static partial int cuMemcpyHtoD(ulong destination, void* source, nuint bytes);

    [LibraryImport(Library, EntryPoint = "cuMemcpyDtoH_v2")]
    public static partial int cuMemcpyDtoH(void* destination, ulong source, nuint bytes);

    [LibraryImport(Library, EntryPoint = "cuMemcpyDtoD_v2")]
    public static partial int cuMemcpyDtoD(ulong destination, ulong source, nuint bytes);

    [LibraryImport(Library, EntryPoint = "cuMemsetD32_v2")]
    public static partial int cuMemsetD32(ulong destination, uint value, nuint count);

    [LibraryImport(Library)]
    public static partial int cuLaunchKernel(
        IntPtr function,
        uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ,
        uint sharedMemBytes, IntPtr stream, void** kernelParams, void** extra);

    [LibraryImport(Library)]
    public static partial int cuGetErrorName(int error, out byte* text);

    // Streams, asynchronous copies and CUDA Graphs (stream capture).

    public const int StreamCaptureModeRelaxed = 2;

    [LibraryImport(Library)]
    public static partial int cuStreamCreate(out IntPtr stream, uint flags);

    [LibraryImport(Library)]
    public static partial int cuStreamSynchronize(IntPtr stream);

    [LibraryImport(Library)]
    public static partial int cuEventCreate(out IntPtr e, uint flags);

    [LibraryImport(Library)]
    public static partial int cuEventRecord(IntPtr e, IntPtr stream);

    [LibraryImport(Library)]
    public static partial int cuEventSynchronize(IntPtr e);

    [LibraryImport(Library)]
    public static partial int cuEventElapsedTime(out float milliseconds, IntPtr start, IntPtr end);

    [LibraryImport(Library, EntryPoint = "cuEventDestroy_v2")]
    public static partial int cuEventDestroy(IntPtr e);

    [LibraryImport(Library)]
    public static partial int cuMemsetD32Async(ulong destination, uint value, nuint count, IntPtr stream);

    [LibraryImport(Library, EntryPoint = "cuMemcpyDtoDAsync_v2")]
    public static partial int cuMemcpyDtoDAsync(ulong destination, ulong source, nuint bytes, IntPtr stream);

    [LibraryImport(Library, EntryPoint = "cuStreamBeginCapture_v2")]
    public static partial int cuStreamBeginCapture(IntPtr stream, int mode);

    [LibraryImport(Library)]
    public static partial int cuStreamEndCapture(IntPtr stream, out IntPtr graph);

    [LibraryImport(Library)]
    public static partial int cuGraphInstantiateWithFlags(out IntPtr executable, IntPtr graph, ulong flags);

    [LibraryImport(Library)]
    public static partial int cuGraphLaunch(IntPtr executable, IntPtr stream);

    [LibraryImport(Library)]
    public static partial int cuGraphExecDestroy(IntPtr executable);

    [LibraryImport(Library)]
    public static partial int cuGraphDestroy(IntPtr graph);
}
