// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Backends.Vulkan;

// The generated SPIR-V kernels: each builds, keeps the Vulkan contract (bindings, push constants ≤ 128 bytes), and passes
// the Khronos validator (spirv-val --target-env vulkan1.1) when it is installed. No Vulkan device is needed.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] SpirvGroup =
    [
        ("spirv: every Vulkan kernel builds at every workgroup width, with and without subgroup reductions, keeps the contract and passes spirv-val (when installed)", SpirvKernelsValidate),
    ];

    private static void SpirvKernelsValidate(Device device)
    {
        // Every width a device may be given: powers of two from MinWidth to MaxWidth.
        var widths = Enumerable.Range(0, 16).Select(i => 1 << i).Where(w => w >= VulkanKernels.MinWidth && w <= VulkanKernels.MaxWidth).ToArray();
        // … each with its reductions through workgroup memory and through subgroup arithmetic.
        var kernels = VulkanKernels.Names.SelectMany(n => widths.SelectMany(w => new[] { false, true }
            .Select(sub => (Width: w, Kernel: VulkanKernels.Get(n, w, sub), Subgroups: sub)))).ToList();
        Check(kernels.Count > 40 * widths.Length, $"only {kernels.Count} kernels");
        foreach (var (width, kernel, _) in kernels)
        {
            Check(kernel.SharedBytes <= VulkanKernels.SharedBytesBound(width),
                $"{kernel.Name} at width {width}: {kernel.SharedBytes} bytes of workgroup memory, over the bound {VulkanKernels.SharedBytesBound(width)}");
            Check(kernel.Words.Length > 5 && kernel.Words[0] == 0x07230203 && kernel.Words[1] == SpirvModule.Version, $"{kernel.Name}: bad header");
            Check(kernel.PushBytes is >= 0 and <= 128 && kernel.PushBytes % 4 == 0, $"{kernel.Name}: {kernel.PushBytes} push-constant bytes");
            Check(kernel.Bindings == kernel.BindingNames.Length && kernel.Bindings > 0, $"{kernel.Name}: {kernel.Bindings} bindings");
            Check(kernel.LocalSize > 0 && kernel.LocalSize <= width, $"{kernel.Name}: local size {kernel.LocalSize} at width {width}");
        }

        // Same words on a second build (deterministic generation).
        Check(VulkanKernels.Unary(Idrak.Backends.UnaryOp.Gelu).Words.SequenceEqual(VulkanKernels.Get("unary_gelu").Words), "lookup by op");

        if (FindOnPath("spirv-val") is not { } validator)
        {
            Console.WriteLine("    (spirv-val not found on PATH: validation skipped)");
            return;
        }

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-spirv-{Environment.ProcessId}");
        Directory.CreateDirectory(folder);
        try
        {
            var failures = new List<string>();
            Parallel.ForEach(kernels, entry =>
            {
                var kernel = entry.Kernel;
                string file = Path.Combine(folder, $"{kernel.Name}_w{entry.Width}{(entry.Subgroups ? "_subgroups" : "")}.spv");
                WriteSpirv(file, kernel.Words);
                var start = new ProcessStartInfo(validator, ["--target-env", "vulkan1.1", file]) { RedirectStandardError = true, RedirectStandardOutput = true };
                using var process = Process.Start(start)!;
                string output = process.StandardError.ReadToEnd() + process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                if (process.ExitCode != 0)
                {
                    lock (failures)
                    {
                        failures.Add($"{kernel.Name} (width {entry.Width}{(entry.Subgroups ? ", subgroups" : "")}): {output.Trim()}");
                    }
                }
            });
            Check(failures.Count == 0, $"spirv-val rejected {failures.Count} kernel(s):\n{string.Join("\n", failures)}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    /// <summary>Writes every Vulkan kernel to folder/name.spv, and their contracts (bindings, push constants) to kernels.txt.</summary>
    public static int DumpSpirv(string folder)
    {
        Directory.CreateDirectory(folder);
        var lines = new List<string>();
        foreach (string name in VulkanKernels.Names.Order(StringComparer.Ordinal))
        {
            var kernel = VulkanKernels.Get(name);
            WriteSpirv(Path.Combine(folder, name + ".spv"), kernel.Words);
            lines.Add(kernel.ToString());
        }

        File.WriteAllLines(Path.Combine(folder, "kernels.txt"), lines);
        Console.WriteLine($"Wrote {lines.Count} SPIR-V kernels to {folder}");
        return 0;
    }

    private static void WriteSpirv(string path, uint[] words)
    {
        var bytes = new byte[words.Length * 4];
        Buffer.BlockCopy(words, 0, bytes, 0, bytes.Length);              // little-endian words, as spirv-val reads them
        File.WriteAllBytes(path, bytes);
    }

    private static string? FindOnPath(string tool) =>
        (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
        .SelectMany(d => new[] { Path.Combine(d, tool), Path.Combine(d, tool + ".exe") })
        .FirstOrDefault(File.Exists);
}
