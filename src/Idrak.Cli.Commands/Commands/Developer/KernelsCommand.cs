// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak kernels [-d DEVICE]</c>: which kernel each operation of the device contract runs on a device (plan 9, "the
/// chain made visible"): a kernel registered for the device's kind, the device's own, the composed default, the host
/// fallback, or none. Read from the dispatcher (<see cref="Kernels.Chain"/>), so plug-ins loaded with <c>--plugin</c>
/// show their registered kernels and the operations they declare (<see cref="PluginOperations"/>, "plug-in" rows). A
/// device whose kernels load on first need or can be turned off (<see cref="IKernelAvailability"/>: HIP's GCN set,
/// Vulkan's) says whether they are loaded, or why not: the operations marked device then take the host fallback.
/// </summary>
internal sealed class KernelsCommand : Command
{
    public override string Name => "kernels";

    public override string Summary => "Which kernel each operation runs on a device: registered, its own, composed, host fallback or none";

    public override string Usage =>
        "[-d DEVICE] [--source KIND]\n\n" +
        "Lists every operation of the device contract with the kernel it runs on the device (the default device without -d):\n" +
        "  registered  a kernel registered for the device's kind (Kernels.Register, from a plug-in, say)\n" +
        "  device      the device's own kernel\n" +
        "  composed    other operations on the same device\n" +
        "  host        the host fallback: the operands are copied to system memory and the CPU runs it\n" +
        "  none        no kernel: the caller takes another path (unfused products, say)\n" +
        "The third column says what runs where a device has no kernel of its own; the last, whether the library or a plug-in\n" +
        "declared the operation (PluginOperations.Register: a packed format, a cache layout, a graph operation, ...).\n" +
        "A device whose kernels load on first need (HIP's GCN kernels, Vulkan's) says whether they loaded, or why not:\n" +
        "then its device rows take the host fallback (IDRAK_HIP_KERNELS=0, IDRAK_VULKAN_KERNELS=0 turn them off).\n\n" +
        "Options:\n" +
        "  --source KIND       only the operations whose kernel is KIND (registered, device, composed, host or none)\n\n" +
        "Examples:\n" +
        "  idrak kernels\n" +
        "  idrak kernels -d vulkan:0 --source host\n" +
        "  idrak kernels -d cuda:0 -j\n" +
        "  idrak kernels -d hip:0 --source device";

    public override IReadOnlyCollection<string> ValueOptions => ["--source"];

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"Unexpected argument '{context.Positional[0]}'; choose the device with -d.");
        }

        KernelSource? only = null;
        if (context.Option("--source") is { } source)
        {
            only = Enum.TryParse<KernelSource>(source, ignoreCase: true, out var parsed) && Enum.IsDefined(parsed) && !source.All(char.IsDigit) ? parsed
                : throw new UsageException($"Unknown kernel source '{source}'; choose registered, device, composed, host or none.");
        }

        var device = context.Device;
        var backend = device.Backend;
        var chain = Kernels.Chain(backend);
        var shown = chain.Where(c => only is null || c.Source == only).ToList();
        context.Write(Messages.T("Kernels on {0} ({1}): {2}", device, backend.Name,
            string.Join(", ", Enum.GetValues<KernelSource>().Select(s => $"{chain.Count(c => c.Source == s)} {Shown((s))}"))));
        // The device's kernels as a whole (a device whose kernels load on first need, or may be turned off: HIP, Vulkan).
        var availability = backend as IKernelAvailability;
        string? unavailable = availability?.KernelsUnavailable();
        if (availability is not null)
        {
            context.Write(unavailable is null ? Messages.T("Its kernels are loaded: the operations marked device run on it.")
                : Messages.T("Its kernels are unavailable ({0}): the operations marked device take the host fallback.", unavailable));
        }

        context.Table([Messages.T("Operation"), Messages.T("Kernel"), Messages.T("Without its own"), Messages.T("Declared by")],
            shown.Select(c => (IReadOnlyList<string>)[c.Operation.Name, Shown((c.Source)), Shown((c.Operation.Fallback)),
                c.Operation.IsPlugin ? Messages.T("plug-in") : Messages.T("library")]));
        context.WriteJson(new JsonObject
        {
            ["device"] = device.ToString(),
            ["kind"] = backend.Kind,
            ["name"] = backend.Name,
            ["kernels"] = availability is null ? null : unavailable is null ? "loaded" : "unavailable",
            ["kernels_unavailable"] = unavailable,
            ["counts"] = new JsonObject(Enum.GetValues<KernelSource>().Select(s => KeyValuePair.Create(Word(s), (JsonNode?)chain.Count(c => c.Source == s)))),
            ["operations"] = new JsonArray([.. shown.Select(c => (JsonNode)new JsonObject
            {
                ["name"] = c.Operation.Name,
                ["kernel"] = Word(c.Source),
                ["fallback"] = Word(c.Operation.Fallback),
                ["plugin"] = c.Operation.IsPlugin,
            })]),
        });
        return ExitCodes.Ok;
    }

    // The source's word in the listing, in the tool's language.
    private static string Shown(KernelSource source) => source switch
    {
        KernelSource.Registered => Messages.T("registered"),
        KernelSource.Device => Messages.T("device"),
        KernelSource.Composed => Messages.T("composed"),
        KernelSource.Host => Messages.T("host"),
        _ => Messages.T("none"),
    };

    // The source's word in the JSON (never translated).
    private static string Word(KernelSource source) => source switch
    {
        KernelSource.Registered => "registered",
        KernelSource.Device => "device",
        KernelSource.Composed => "composed",
        KernelSource.Host => "host",
        _ => "none",
    };
}
