// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Gpu.Vulkan;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak kernels dump [ptx|spirv|gcn]</c>: writes the kernels the GPU backends generate, for debugging: the CUDA
/// PTX (what the test runner's <c>--dump-ptx</c> writes), the Vulkan SPIR-V modules with their contracts (its
/// <c>--dump-spirv</c>) and HIP's GCN kernels (plan 35), each as written (virtual registers) and as Idrak's assembler
/// prints it for a target (physical registers, waits and wait states: text LLVM's llvm-mc reads). No GPU is needed: the
/// kernels are generated, not read from a device.
/// </summary>
internal sealed class KernelsDumpCommand : Command
{
    /// <summary>The kinds of kernels, in the order they are written.</summary>
    public static readonly string[] Kinds = ["ptx", "spirv", "gcn"];

    public override string Name => "kernels dump";

    public override string Summary => "Writes the generated GPU kernels (PTX, SPIR-V, GCN) for debugging";

    public override string Usage =>
        "[ptx|spirv|gcn] [-o DIR] [--target TARGET]\n\n" +
        "Writes the generated kernels to DIR (default ./kernels); without a kind, all three:\n" +
        "  ptx     the CUDA kernels: idrak.ptx and idrak.tensorcore.ptx (the default kernel shapes)\n" +
        "  spirv   the Vulkan kernels: spirv/NAME.spv and spirv/kernels.txt (bindings, push constants); the\n" +
        "          cooperative-matrix products in the 16 x 16 x 16 shape (a device builds them in the shape it reports)\n" +
        "  gcn     HIP's kernels: gcn/virtual/NAME.s as written (virtual registers) and gcn/TARGET/NAME.s as Idrak's\n" +
        "          assembler prints them for TARGET (physical registers, waits and wait states; llvm-mc reads it)\n\n" +
        "Options:\n" +
        "  -o, --out DIR       the folder to write to (default ./kernels)\n" +
        "  --target TARGET     gcn: the GCN target to print for, as a device reports it (default: hip:0's, else the\n" +
        "                      first processor a registered family handles)\n\n" +
        "Examples:\n" +
        "  idrak kernels dump\n" +
        "  idrak kernels dump spirv -o /tmp/spv && spirv-val /tmp/spv/spirv/add.spv\n" +
        "  idrak kernels dump gcn -o test-output/kernels";

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--target"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 1)
        {
            throw new UsageException($"Unexpected argument '{context.Positional[1]}'; give one kind (ptx, spirv or gcn) or none for all.");
        }

        string[] kinds = context.Positional.Count == 0 ? Kinds : [context.Positional[0].ToLowerInvariant()];
        if (!Kinds.Contains(kinds[0]))
        {
            throw new UsageException($"Unknown kind '{context.Positional[0]}'; choose ptx, spirv or gcn.");
        }

        GcnIsaVersion? target = null;
        if (context.Option("--target") is { } named)
        {
            target = GcnIsaVersion.TryParse(named, out var parsed) && GcnTargets.TryFor(parsed, out _) ? parsed
                : throw new UsageException($"--target {named}: not a GCN target a registered family handles (families: {string.Join(", ", GcnTargets.Names)}).");
        }

        string folder = Path.GetFullPath(context.Option("--out") ?? "kernels");
        Directory.CreateDirectory(folder);
        var files = new List<(string Kind, string Path, long Bytes)>();
        foreach (string kind in kinds)
        {
            int before = files.Count;
            switch (kind)
            {
                case "ptx":
                    files.Add((kind, WriteText(Path.Combine(folder, "idrak.ptx"), PtxKernels.Source), 0));
                    files.Add((kind, WriteText(Path.Combine(folder, "idrak.tensorcore.ptx"), PtxKernels.TensorCoreSource), 0));
                    break;
                case "gcn":
                    files.AddRange(DumpGcn(Path.Combine(folder, "gcn"), target ?? GcnKernels.DefaultTarget()).Select(p => (kind, p, 0L)));
                    break;
                default:
                    files.AddRange(DumpSpirv(Path.Combine(folder, "spirv")).Select(p => (kind, p, 0L)));
                    break;
            }

            for (int i = before; i < files.Count; i++)
            {
                files[i] = files[i] with { Bytes = new FileInfo(files[i].Path).Length };
            }

            var mine = files.Skip(before).ToList();
            string where = kind == "gcn" ? Path.Combine(folder, "gcn") : Path.GetDirectoryName(mine[0].Path)!;
            context.Write(string.Create(CultureInfo.InvariantCulture, $"{kind,-6} {mine.Count,4} files  {mine.Sum(f => f.Bytes),12:N0} bytes  {where}"));
        }

        foreach (var file in files)
        {
            context.Detail(string.Create(CultureInfo.InvariantCulture, $"  {Path.GetRelativePath(folder, file.Path)}  {file.Bytes:N0} bytes"));
        }

        context.WriteJson(new JsonObject
        {
            ["folder"] = folder,
            ["files"] = new JsonArray([.. files.Select(f => (JsonNode)new JsonObject { ["kind"] = f.Kind, ["path"] = f.Path, ["bytes"] = f.Bytes })]),
        });
        return ExitCodes.Ok;
    }

    // HIP's kernels: each as written (virtual registers, the same for every target) and as printed for the target.
    private static IEnumerable<string> DumpGcn(string folder, GcnIsaVersion target)
    {
        string written = Path.Combine(folder, "virtual"), printed = Path.Combine(folder, target.Processor);
        Directory.CreateDirectory(written);
        Directory.CreateDirectory(printed);
        var paths = new List<string>();
        foreach (var (name, text, physical) in GcnKernels.Dump(target))
        {
            paths.Add(WriteText(Path.Combine(written, name + ".s"), text));
            paths.Add(WriteText(Path.Combine(printed, name + ".s"), $"; {name} for {target}, printed by Idrak's GCN assembler\n{physical}"));
        }

        return paths;
    }

    // The test runner's --dump-spirv: every kernel at its default width, and the cooperative-matrix products in one shape.
    private static IEnumerable<string> DumpSpirv(string folder)
    {
        Directory.CreateDirectory(folder);
        var lines = new List<string>();
        var paths = new List<string>();
        foreach (string name in VulkanKernels.Names.Order(StringComparer.Ordinal))
        {
            var kernel = VulkanKernels.Get(name);
            paths.Add(WriteWords(Path.Combine(folder, name + ".spv"), kernel.Words));
            lines.Add(kernel.ToString());
        }

        foreach (var format in new VulkanKernels.PackedFormat?[] { null, VulkanKernels.PackedFormat.Int8, VulkanKernels.PackedFormat.Int4, VulkanKernels.PackedFormat.BFloat16 })
        {
            var kernel = VulkanKernels.Coop(new VulkanKernels.CoopSpec(format, new VulkanKernels.CoopShape(16, 16, 16), 128));
            paths.Add(WriteWords(Path.Combine(folder, kernel.Name + ".spv"), kernel.Words));
            lines.Add(kernel.ToString());
        }

        string contracts = Path.Combine(folder, "kernels.txt");
        File.WriteAllLines(contracts, lines);
        paths.Add(contracts);
        return paths;
    }

    private static string WriteWords(string path, uint[] words)
    {
        var bytes = new byte[words.Length * 4];
        Buffer.BlockCopy(words, 0, bytes, 0, bytes.Length);                  // little-endian words, as spirv-val reads them
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private static string WriteText(string path, string text)
    {
        File.WriteAllText(path, text);
        return path;
    }
}
