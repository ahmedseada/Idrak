// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak devices</c> (<c>dev</c>): every device with its backend, name, memory, compute units, subgroup size, matrix
/// units, kernel width and whether a plain run uses it (the test runner's <c>--list-devices</c>), and each backend that
/// found none with the reason.
/// </summary>
internal sealed class DevicesCommand : Command
{
    public override string Name => "devices";

    public override IReadOnlyCollection<string> Aliases => ["dev"];

    public override string Summary => "Every device: backend, name, memory, compute units, subgroup size, matrix units, kernel width";

    public override string Usage => """

        Lists the CPU and every device of every backend, including devices reached by name only (such as a software
        Vulkan driver). With --device, only that device.

        Columns: Memory (the device's memory; the CPU: what the process may use), CUs (compute units; the CPU: logical
        processors), Lanes (subgroup, warp or wavefront size; the CPU: float lanes per vector), Width (threads per
        workgroup of the kernels, chosen from the device's limits), Matrix (matrix units), In a plain run (yes: listed
        by Device.Available and tested by a plain test run; by name: only when named).

        Examples:
          idrak devices
          idrak dev -j
          idrak devices -d vulkan:0
        """;

    public override int Run(CommandContext context)
    {
        var devices = Machine.Devices(context);
        var missing = context.Option("--device") is null ? DeviceListing.Backends.Where(b => b.Count == 0).ToList() : [];
        string yes = Messages.T("yes"), no = Messages.T("no");
        var rows = devices.Select(d => (IReadOnlyList<string>)
        [
            d.Device, d.Backend, d.Error is null ? d.Name : Messages.T("cannot start: {0}", d.Error), Machine.Bytes(d.MemoryBytes), Machine.Number(d.ComputeUnits),
            Machine.Number(d.SubgroupSize), Machine.Number(d.KernelWidth), d.Error is null ? d.MatrixUnits ? yes : no : "-",
            d.Listed ? yes : Messages.T("by name"), d.IsDefault ? Messages.T("default device") : d.Note ?? "",
        ]).Concat(missing.Select(b => (IReadOnlyList<string>)[$"{b.Kind}:-", b.Display, Messages.T("none found: {0}", b.UnavailableReason), "-", "-", "-", "-", "-", "-", ""]));
        context.Table([Messages.T("Device"), Messages.T("Backend"), Messages.T("Name"), Messages.T("Memory"), Messages.T("CUs"), Messages.T("Lanes"),
            Messages.T("Width"), Messages.T("Matrix"), Messages.T("In a plain run"), Messages.T("Note")], rows);
        context.WriteJson(new JsonObject
        {
            ["default"] = Device.Default.ToString(),
            ["devices"] = new JsonArray([.. devices.Select(d => (JsonNode)Machine.Json(d))]),
            ["backends"] = new JsonArray([.. DeviceListing.Backends.Select(b => (JsonNode)Machine.Json(b))]),
        });
        // The listing itself succeeds whatever a device reports; a device asked for by name that cannot start fails.
        return context.Option("--device") is not null && devices[0].Error is not null ? ExitCodes.Failed : ExitCodes.Ok;
    }
}
