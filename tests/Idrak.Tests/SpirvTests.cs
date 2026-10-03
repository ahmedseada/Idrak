// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Backends.Vulkan;

// The generated SPIR-V kernels: each builds, keeps the Vulkan contract (bindings, push constants ≤ 128 bytes), and passes
// the Khronos validator (spirv-val --target-env vulkan1.1) when it is installed. No Vulkan device is needed. The
// cooperative-matrix products (built only for devices that report cooperative matrices) are validated the same way, in
// every shape the kernels take, at several widths, and in their emulated form, float32-accurate and reduced precision
// (16-bit float and bfloat16 operands). The bfloat16 kernels need a validator that knows SPV_KHR_bfloat16 (SPIRV-Tools
// 2025.2 and later): an older one skips them, saying so. IDRAK_SPIRV_VAL names a validator other than spirv-val on PATH.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] SpirvGroup =
    [
        ("spirv: every Vulkan kernel builds at every workgroup width, with and without subgroup reductions, keeps the contract and passes spirv-val (when installed)", SpirvKernelsValidate),
        ("spirv: the cooperative-matrix products (float32, int8, int4, bfloat16; every matrix shape they take; several widths; emulated too) keep the contract and pass spirv-val (when installed)", SpirvCoopKernelsValidate),
        ("spirv: the reduced-precision cooperative-matrix products (16-bit float and bfloat16 operands; every format and matrix shape; several widths; emulated too) keep the contract, do one product per step and pass spirv-val (when installed; bfloat16 when it knows SPV_KHR_bfloat16)", SpirvMixedCoopKernelsValidate),
    ];

    private static void SpirvMixedCoopKernelsValidate(Device device)
    {
        _ = device;
        int[] sizes = [8, 16, 32];
        var shapes = sizes.SelectMany(m => sizes.SelectMany(n => sizes.Select(k => new VulkanKernels.CoopShape(m, n, k)))).ToArray();
        VulkanKernels.PackedFormat?[] formats = [null, VulkanKernels.PackedFormat.Int8, VulkanKernels.PackedFormat.Int4, VulkanKernels.PackedFormat.BFloat16];
        var specs = new List<VulkanKernels.CoopSpec>();
        foreach (var precision in new[] { VulkanKernels.CoopPrecision.Float16, VulkanKernels.CoopPrecision.BFloat16 })
        {
            foreach (var shape in shapes)
            {
                foreach (var format in formats)
                {
                    foreach (int depth in VulkanKernels.CoopDepths.Where(d => d >= shape.K))
                    {
                        specs.AddRange(new[] { 16, 128, 1024 }.Select(w => new VulkanKernels.CoopSpec(format, shape, w, 0, depth, precision)));
                        foreach (int subgroup in new[] { 8, 32 })
                        {
                            specs.AddRange(new[] { 32, 64 }.Where(w => w % subgroup == 0 && shape.M * shape.N % subgroup == 0)
                                .Select(w => new VulkanKernels.CoopSpec(format, shape, w, subgroup, depth, precision)));
                        }
                    }
                }
            }
        }

        var kernels = specs.Select(s => (Spec: s, Kernel: VulkanKernels.Coop(s))).ToList();
        foreach (var (spec, kernel) in kernels)
        {
            var split = VulkanKernels.Coop(spec with { Precision = VulkanKernels.CoopPrecision.Split });
            Check(kernel.Name == spec.Name && kernel.Name.EndsWith(VulkanKernels.RoundSuffix(spec.Precision) + (spec.Emulated > 0 ? $"_emulated{spec.Emulated}" : ""), StringComparison.Ordinal)
                && kernel.LocalSize == spec.Width, $"{kernel.Name}: local size {kernel.LocalSize}");
            // Within the float32-accurate kernels' bounds (the backend picks the depth by them); on a device, no more than those kernels.
            Check(kernel.SharedBytes <= (spec.Depth == 32 ? 36 : 27) << 10 && (spec.Emulated > 0 || kernel.SharedBytes <= split.SharedBytes),
                $"{kernel.Name}: {kernel.SharedBytes} bytes of workgroup memory (float32-accurate: {split.SharedBytes})");
            Check(kernel.PushBytes == split.PushBytes && kernel.Bindings == split.Bindings && kernel.Writes == split.Writes,
                $"{kernel.Name}: {kernel.Bindings} bindings, {kernel.PushBytes} push-constant bytes (the float32-accurate kernel's contract)");
            if (spec.Emulated == 0)
            {
                // One matrix product per tile and matrix depth (the float32-accurate kernel: two or three).
                int tiles = 32 / spec.Shape.M * (32 / spec.Shape.N) * (spec.Depth / spec.Shape.K);
                int products = CountInstructions(kernel.Words, 4459), splitProducts = CountInstructions(split.Words, 4459);   // OpCooperativeMatrixMulAddKHR
                Check(products == tiles && splitProducts >= 2 * tiles, $"{kernel.Name}: {products} matrix products ({splitProducts} float32-accurate), expected {tiles}");
                // bfloat16 (OpCapability BFloat16TypeKHR) exactly where the operands are bfloat16.
                Check((CountInstructions(kernel.Words, 17, 5116) == 1) == (spec.Precision == VulkanKernels.CoopPrecision.BFloat16),
                    $"{kernel.Name}: bfloat16 declared {CountInstructions(kernel.Words, 17, 5116)} times");
            }
        }

        string? validator = SpirvValidator();
        bool knowsBFloat16 = validator is not null && ValidatorKnowsBFloat16(validator);
        var checkedKernels = kernels.Where(e => knowsBFloat16 || e.Spec.Precision != VulkanKernels.CoopPrecision.BFloat16 || e.Spec.Emulated > 0).ToList();
        if (validator is not null && !knowsBFloat16)
        {
            Console.WriteLine($"    ({validator} does not know SPV_KHR_bfloat16: {kernels.Count - checkedKernels.Count} bfloat16 kernels not validated; IDRAK_SPIRV_VAL names a newer one)");
        }

        ValidateWithSpirvVal(checkedKernels.Select(e => (e.Kernel, $"{e.Kernel.Name}.spv")).ToList());
    }

    // Instructions of `opcode` in a module (with `operand` as their first operand when given).
    private static int CountInstructions(uint[] words, uint opcode, uint? operand = null)
    {
        int count = 0;
        for (int i = 5; i < words.Length;)
        {
            uint length = words[i] >> 16;
            if ((words[i] & 0xFFFF) == opcode && (operand is null || length > 1 && words[i + 1] == operand))
            {
                count++;
            }

            i += (int)Math.Max(1, length);
        }

        return count;
    }

    // Whether the validator accepts a minimal module with bfloat16 cooperative matrices (SPV_KHR_bfloat16).
    private static bool ValidatorKnowsBFloat16(string validator)
    {
        var k = new KernelBuilder("bfloat16_probe", 32);
        var staged = k.SharedOperand("staged", 256, OperandType.BFloat16);
        var target = k.Shared("target", 256);
        uint a = k.MatrixType(16, 16, MatrixUse.A, OperandType.BFloat16);
        uint b = k.MatrixType(16, 16, MatrixUse.B, OperandType.BFloat16);
        uint c = k.MatrixType(16, 16, MatrixUse.Accumulator, OperandType.BFloat16);
        uint sum = k.MatrixLocal(c);
        k.MatrixZero(sum, c);
        k.MatrixMulAdd(sum, c, k.MatrixLoad(a, staged, k.Int(0), 16), k.MatrixLoad(b, staged, k.Int(0), 16));
        k.MatrixStore(sum, c, target, k.Int(0), 16);
        k.Buffer("unused");
        var built = k.Build();
        string file = Path.Combine(Path.GetTempPath(), $"idrak-bfloat16-probe-{Environment.ProcessId}.spv");
        try
        {
            var (passed, output) = Validate(file, built.Words);
            if (!passed && output.Contains("VulkanMemoryModel capability must also be declared", StringComparison.Ordinal))
            {
                (passed, _) = Validate(file, WithVulkanMemoryModel(built.Words));
            }

            return passed;
        }
        finally
        {
            File.Delete(file);
        }
    }

    // The validator: IDRAK_SPIRV_VAL, else spirv-val on PATH, else null.
    private static string? SpirvValidator() =>
        Environment.GetEnvironmentVariable("IDRAK_SPIRV_VAL") is { Length: > 0 } named && File.Exists(named) ? named : FindOnPath("spirv-val");

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
        if (SpirvValidator() is not { } validator)
        {
            Console.WriteLine("    (spirv-val not found on PATH: validation skipped)");
            return;
        }

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-spirv-{Environment.ProcessId}");
        Directory.CreateDirectory(folder);
        try
        {
            var failures = new List<string>();
            int memoryModel = 0;
            Parallel.ForEach(kernels, entry =>
            {
                string file = Path.Combine(folder, entry.File);
                var (passed, output) = Validate(file, entry.Kernel.Words);
                if (!passed && output.Contains("VulkanMemoryModel capability must also be declared", StringComparison.Ordinal))
                {
                    // Newer validators require the Vulkan memory model with cooperative matrices (and stop there); the
                    // kernels declare the GLSL450 model, which drivers accept. The rest of the module is validated with
                    // the Vulkan memory model declared instead.
                    (passed, output) = Validate(file, WithVulkanMemoryModel(entry.Kernel.Words));
                    Interlocked.Increment(ref memoryModel);
                }

                if (!passed)
                {
                    lock (failures)
                    {
                        failures.Add($"{entry.File}: {output.Trim()}");
                    }
                }
            });
            if (memoryModel > 0)
            {
                Console.WriteLine($"    ({validator} requires the Vulkan memory model with cooperative matrices: {memoryModel} kernels validated with it declared)");
            }

            Check(failures.Count == 0, $"spirv-val rejected {failures.Count} kernel(s):\n{string.Join("\n", failures)}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // Runs the validator on one module: whether it passed, and what it said.
    private static (bool Passed, string Output) Validate(string file, uint[] words)
    {
        WriteSpirv(file, words);
        var start = new ProcessStartInfo(SpirvValidator()!, ["--target-env", "vulkan1.1", file]) { RedirectStandardError = true, RedirectStandardOutput = true };
        using var process = Process.Start(start)!;
        string output = process.StandardError.ReadToEnd() + process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode == 0, output);
    }

    // The module with the Vulkan memory model declared (capability VulkanMemoryModel, SPV_KHR_vulkan_memory_model,
    // OpMemoryModel Logical Vulkan) in place of GLSL450.
    private static uint[] WithVulkanMemoryModel(uint[] words)
    {
        var result = new List<uint>(words[..5]) { 2u << 16 | 17, 5345 };       // OpCapability VulkanMemoryModel
        bool extended = false;
        for (int i = 5; i < words.Length;)
        {
            int length = (int)Math.Max(1, words[i] >> 16);
            uint opcode = words[i] & 0xFFFF;
            if (!extended && opcode is 10 or 11 or 14)                          // the first OpExtension, OpExtInstImport or OpMemoryModel
            {
                var name = System.Text.Encoding.ASCII.GetBytes("SPV_KHR_vulkan_memory_model\0\0\0\0\0");
                int count = (27 + 4) / 4;                                       // the name and its terminating zero, in words
                result.Add((uint)(count + 1) << 16 | 10);
                for (int w = 0; w < count; w++)
                {
                    result.Add(BitConverter.ToUInt32(name, 4 * w));
                }

                extended = true;
            }

            var instruction = words[i..(i + length)];
            if (opcode == 14)
            {
                instruction[2] = 3;                                             // Vulkan
            }

            result.AddRange(instruction);
            i += length;
        }

        return [.. result];
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
