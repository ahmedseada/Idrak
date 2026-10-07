// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Gpu.Hip;

/// <summary>
/// Bindings to hipRTC, HIP's run-time compiler (libhiprtc.so with ROCm; hiprtcMMmm.dll with the Windows HIP SDK), which
/// turns HIP C++ source into a code object for one GPU architecture. Loaded like the runtime (<see cref="HipRuntime"/>):
/// at run time, by name, optional. Without it the HIP backend still runs, with every operation on the host fallback
/// except memory, copies and fills; the graphics driver alone (Windows) installs the runtime but not the compiler.
/// </summary>
internal static unsafe class HipRtc
{
    private static readonly Lazy<string> Loaded = new(Load);

    /// <summary>Why the compiler is unavailable ("" when it loaded).</summary>
    public static string UnavailableReason => Loaded.Value;

    /// <summary>The compiler's version ("major.minor"), or "" when it did not load.</summary>
    public static string Version { get; private set; } = "";

    /// <summary>The compiler's file names tried in order on <paramref name="os"/> ("windows" or "linux"; none elsewhere).</summary>
    internal static string[] CandidateNames(string os)
    {
        if (os == "linux")
        {
            return ["libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so.5"];
        }

        if (os != "windows")
        {
            return [];
        }

        // The Windows SDK names the file after its release (hiprtc0602.dll for 6.2): newest first.
        var names = new List<string>();
        for (int major = 7; major >= 5; major--)
        {
            for (int minor = 9; minor >= 0; minor--)
            {
                names.Add($"hiprtc{major:00}{minor:00}.dll");
            }
        }

        names.Add("hiprtc.dll");
        return [.. names];
    }

    /// <summary>The options of a compilation for the architecture <paramref name="architecture"/> (null: the current device's).</summary>
    internal static string[] Options(string? architecture, int blockSize)
    {
        var options = new List<string> { "-O3", $"-DIDRAK_BLOCK={blockSize}" };
        if (!string.IsNullOrEmpty(architecture))
        {
            options.Add($"--gpu-architecture={architecture}");               // the device's own target, features included (hipRTC's documented form)
        }

        return [.. options];
    }

    private static string Load()
    {
        string os = HipRuntime.CurrentOs;
        var folders = HipRuntime.SearchFolders(os, Environment.GetEnvironmentVariable).ToList();
        string[] names = CandidateNames(os);
        if (!HipRuntime.TryOpen(names, folders, out var handle, out _))
        {
            // A Windows SDK release newer than the names above: any hiprtcMMmm.dll in its bin folder.
            string? found = os != "windows" ? null : folders.Where(Directory.Exists)
                .SelectMany(f => Directory.EnumerateFiles(f, "hiprtc0*.dll"))
                .Where(f => !Path.GetFileName(f).Contains("builtins", StringComparison.OrdinalIgnoreCase))
                .OrderByDescending(f => f, StringComparer.OrdinalIgnoreCase).FirstOrDefault();
            if (found is null || !NativeLibrary.TryLoad(found, out handle))
            {
                return $"the HIP run-time compiler ({names.FirstOrDefault() ?? "hiprtc"}) was not found; install ROCm (Linux) or the HIP SDK (Windows)";
            }
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

        hiprtcVersion = (delegate* unmanaged<int*, int*, int>)Required(nameof(hiprtcVersion));
        hiprtcCreateProgram = (delegate* unmanaged<IntPtr*, byte*, byte*, int, byte**, byte**, int>)Required(nameof(hiprtcCreateProgram));
        hiprtcCompileProgram = (delegate* unmanaged<IntPtr, int, byte**, int>)Required(nameof(hiprtcCompileProgram));
        hiprtcGetProgramLogSize = (delegate* unmanaged<IntPtr, nuint*, int>)Required(nameof(hiprtcGetProgramLogSize));
        hiprtcGetProgramLog = (delegate* unmanaged<IntPtr, byte*, int>)Required(nameof(hiprtcGetProgramLog));
        hiprtcGetCodeSize = (delegate* unmanaged<IntPtr, nuint*, int>)Required(nameof(hiprtcGetCodeSize));
        hiprtcGetCode = (delegate* unmanaged<IntPtr, byte*, int>)Required(nameof(hiprtcGetCode));
        hiprtcDestroyProgram = (delegate* unmanaged<IntPtr*, int>)Required(nameof(hiprtcDestroyProgram));
        hiprtcGetErrorString = NativeLibrary.TryGetExport(handle, nameof(hiprtcGetErrorString), out var errors)
            ? (delegate* unmanaged<int, byte*>)errors : null;
        if (missing.Count > 0)
        {
            return $"the HIP run-time compiler found lacks {string.Join(", ", missing)}";
        }

        int major = 0, minor = 0;
        if (hiprtcVersion(&major, &minor) == 0)
        {
            Version = $"{major}.{minor}";
        }

        return "";
    }

    /// <summary>
    /// Compiles <paramref name="source"/> with <paramref name="options"/> into a code object; throws a
    /// <see cref="HipException"/> with the compiler's log when it fails.
    /// </summary>
    public static byte[] Compile(string source, string name, IReadOnlyList<string> options)
    {
        if (UnavailableReason.Length > 0)
        {
            throw new HipException(UnavailableReason);
        }

        byte[] sourceBytes = Encoding.UTF8.GetBytes(source + "\0");
        byte[] nameBytes = Encoding.UTF8.GetBytes(name + "\0");
        var optionBytes = options.Select(o => Encoding.UTF8.GetBytes(o + "\0")).ToArray();
        var pins = optionBytes.Select(o => GCHandle.Alloc(o, GCHandleType.Pinned)).ToArray();
        try
        {
            byte** optionPointers = stackalloc byte*[Math.Max(1, optionBytes.Length)];
            for (int i = 0; i < optionBytes.Length; i++)
            {
                optionPointers[i] = (byte*)pins[i].AddrOfPinnedObject();
            }

            IntPtr program;
            fixed (byte* s = sourceBytes)
            fixed (byte* n = nameBytes)
            {
                Check(hiprtcCreateProgram(&program, s, n, 0, null, null), nameof(hiprtcCreateProgram));
            }

            try
            {
                int result = hiprtcCompileProgram(program, optionBytes.Length, optionPointers);
                if (result != 0)
                {
                    throw new HipException($"hipRTC could not compile the Idrak kernels ({Describe(result)}): {Log(program)}");
                }

                nuint size;
                Check(hiprtcGetCodeSize(program, &size), nameof(hiprtcGetCodeSize));
                byte[] code = new byte[(int)size];
                fixed (byte* c = code)
                {
                    Check(hiprtcGetCode(program, c), nameof(hiprtcGetCode));
                }

                return code;
            }
            finally
            {
                hiprtcDestroyProgram(&program);
            }
        }
        finally
        {
            foreach (var pin in pins)
            {
                pin.Free();
            }
        }
    }

    // The compiler's messages for a program.
    private static string Log(IntPtr program)
    {
        nuint size;
        if (hiprtcGetProgramLogSize(program, &size) != 0 || size <= 1)
        {
            return "(no log)";
        }

        byte[] log = new byte[(int)size];
        fixed (byte* l = log)
        {
            return hiprtcGetProgramLog(program, l) == 0 ? Encoding.UTF8.GetString(log, 0, (int)size - 1).Trim() : "(no log)";
        }
    }

    private static void Check(int result, string call)
    {
        if (result != 0)
        {
            throw new HipException($"{call} failed: {Describe(result)}");
        }
    }

    private static string Describe(int result) =>
        (hiprtcGetErrorString != null && hiprtcGetErrorString(result) is var text && text != null
            ? Marshal.PtrToStringAnsi((IntPtr)text) ?? "HIPRTC_ERROR" : "HIPRTC_ERROR") + $" ({result})";

    private static delegate* unmanaged<int*, int*, int> hiprtcVersion;
    private static delegate* unmanaged<IntPtr*, byte*, byte*, int, byte**, byte**, int> hiprtcCreateProgram;
    private static delegate* unmanaged<IntPtr, int, byte**, int> hiprtcCompileProgram;
    private static delegate* unmanaged<IntPtr, nuint*, int> hiprtcGetProgramLogSize;
    private static delegate* unmanaged<IntPtr, byte*, int> hiprtcGetProgramLog;
    private static delegate* unmanaged<IntPtr, nuint*, int> hiprtcGetCodeSize;
    private static delegate* unmanaged<IntPtr, byte*, int> hiprtcGetCode;
    private static delegate* unmanaged<IntPtr*, int> hiprtcDestroyProgram;
    private static delegate* unmanaged<int, byte*> hiprtcGetErrorString;
}
