// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Backends.Vulkan;

// The generated SPIR-V kernels: each builds, keeps the Vulkan contract (bindings, push constants ≤ 128 bytes), and passes
// the Khronos validator (spirv-val --target-env vulkan1.1) when it is installed. No Vulkan device is needed. The
// cooperative-matrix products (built only for devices that report cooperative matrices) are validated the same way, in
// every shape the kernels take, at several widths, and in their emulated form.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] SpirvGroup =
    [
        ("spirv: every Vulkan kernel builds at every workgroup width, with and without subgroup reductions, keeps the contract and passes spirv-val (when installed)", SpirvKernelsValidate),
        ("spirv: the cooperative-matrix products (float32, int8, int4, bfloat16; every matrix shape they take; several widths; emulated too) keep the contract and pass spirv-val (when installed)", SpirvCoopKernelsValidate),
    ];

    private static void SpirvCoopKernelsValidate(Device device)
    {
        _ = device;
        int[] sizes = [8, 16, 32];
        var shapes = sizes.SelectMany(m => sizes.SelectMany(n => sizes.Select(k => new VulkanKernels.CoopShape(m, n, k)))).ToArray();
        VulkanKernels.PackedFormat?[] formats = [null, VulkanKernels.PackedFormat.Int8, VulkanKernels.PackedFormat.Int4, VulkanKernels.PackedFormat.BFloat16];
        var specs = new List<VulkanKernels.CoopSpec>();
        foreach (var shape in shapes)
        {
            foreach (var format in formats)
            {
                foreach (int depth in VulkanKernels.CoopDepths.Where(d => d >= shape.K))
                {
                    specs.AddRange(new[] { 16, 32, 128, 256, 1024 }.Select(w => new VulkanKernels.CoopSpec(format, shape, w, 0, depth)));
                    foreach (int subgroup in new[] { 8, 16, 32 })
                    {
                        specs.AddRange(new[] { 32, 64 }.Where(w => w % subgroup == 0 && shape.M * shape.N % subgroup == 0)
                            .Select(w => new VulkanKernels.CoopSpec(format, shape, w, subgroup, depth)));
                    }
                }
            }
        }

        Check(!VulkanKernels.CoopShapeUsable(new VulkanKernels.CoopShape(16, 16, 4)) && !VulkanKernels.CoopShapeUsable(new VulkanKernels.CoopShape(64, 16, 16)),
            "shapes the kernels do not tile are refused");
        var kernels = specs.Select(s => (Spec: s, Kernel: VulkanKernels.Coop(s))).ToList();
        foreach (var (spec, kernel) in kernels)
        {
            Check(kernel.Name == spec.Name && kernel.LocalSize == spec.Width, $"{kernel.Name}: local size {kernel.LocalSize}");
            // At most 36 KiB staging 32 rows of k, 27 KiB staging 16 (the backend takes the deepest the device has room for).
            Check(kernel.SharedBytes <= (spec.Depth == 32 ? 36 : 27) << 10, $"{kernel.Name}: {kernel.SharedBytes} bytes of workgroup memory");
            Check(kernel.PushBytes is > 0 and <= 128 && kernel.Bindings == (spec.Format is null or VulkanKernels.PackedFormat.BFloat16 ? 3 : 4),
                $"{kernel.Name}: {kernel.Bindings} bindings, {kernel.PushBytes} push-constant bytes");
            Check(kernel.Writes == 1UL << (kernel.Bindings - 1), $"{kernel.Name}: writes {kernel.Writes:X}");
        }

        ValidateWithSpirvVal(kernels.Select(e => (e.Kernel, $"{e.Kernel.Name}.spv")).ToList());
    }

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

        ValidateWithSpirvVal(kernels.Select(e => (e.Kernel, $"{e.Kernel.Name}_w{e.Width}{(e.Subgroups ? "_subgroups" : "")}.spv")).ToList());
    }

    // Runs spirv-val --target-env vulkan1.1 on every kernel (skipped when it is not installed).
    private static void ValidateWithSpirvVal(List<(SpirvKernel Kernel, string File)> kernels)
    {
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
                string file = Path.Combine(folder, entry.File);
                WriteSpirv(file, entry.Kernel.Words);
                var start = new ProcessStartInfo(validator, ["--target-env", "vulkan1.1", file]) { RedirectStandardError = true, RedirectStandardOutput = true };
                using var process = Process.Start(start)!;
                string output = process.StandardError.ReadToEnd() + process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                if (process.ExitCode != 0)
                {
                    lock (failures)
                    {
                        failures.Add($"{entry.File}: {output.Trim()}");
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

    /// <summary>Writes every Vulkan kernel (and the cooperative-matrix products in one shape) to folder/name.spv, and their contracts (bindings, push constants) to kernels.txt.</summary>
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

        // The cooperative-matrix products in the common 16 × 16 × 16 shape (devices build them in the shape they report).
        foreach (var format in new VulkanKernels.PackedFormat?[] { null, VulkanKernels.PackedFormat.Int8, VulkanKernels.PackedFormat.Int4, VulkanKernels.PackedFormat.BFloat16 })
        {
            var kernel = VulkanKernels.Coop(new VulkanKernels.CoopSpec(format, new VulkanKernels.CoopShape(16, 16, 16), 128));
            WriteSpirv(Path.Combine(folder, kernel.Name + ".spv"), kernel.Words);
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
