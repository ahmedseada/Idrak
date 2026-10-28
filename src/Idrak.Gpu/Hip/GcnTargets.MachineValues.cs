// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Abstractions;

// Format data: what LLVM writes into a code object's header for each processor. Nothing branches on a name here; a
// family looks its processor up by the version the device reports.
public static partial class GcnTargets
{
    /// <summary>The target triple of HSA code objects (architecture, vendor and OS fields), before the target id in the metadata.</summary>
    internal const string Triple = "amdgcn-amd-amdhsa";

    /// <summary>The owner of a code object's metadata note (NT_AMDGPU_METADATA): its name and terminator, padded to 4 bytes.</summary>
    internal static ReadOnlySpan<byte> NoteOwner => "AMDGPU\0\0"u8;

    /// <summary>
    /// Every processor LLVM 18 assembles for, with its EF_AMDGPU_MACH_* value (the low byte of e_flags) and whether it has
    /// an xnack and an sramecc setting. Read on 2026-10-28 from LLVM 18.1.3 itself: for each processor
    /// <c>llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=…</c> assembled <c>.amdgcn_target "amdgcn-amd-amdhsa--…"</c>, and
    /// <c>llvm-readelf -h</c> gave e_flags (the low byte the machine, 0x100 an xnack setting, 0x400 an sramecc one, both
    /// "any" when the target names neither).
    /// </summary>
    private static readonly (string Processor, byte Machine, bool Xnack, bool Sramecc)[] Machines =
    [
        ("gfx600", 0x20, false, false),
        ("gfx601", 0x21, false, false),
        ("gfx602", 0x3A, false, false),
        ("gfx700", 0x22, false, false),
        ("gfx701", 0x23, false, false),
        ("gfx702", 0x24, false, false),
        ("gfx703", 0x25, false, false),
        ("gfx704", 0x26, false, false),
        ("gfx705", 0x3B, false, false),
        ("gfx801", 0x28, true, false),
        ("gfx802", 0x29, false, false),
        ("gfx803", 0x2A, false, false),
        ("gfx805", 0x3C, false, false),
        ("gfx810", 0x2B, true, false),
        ("gfx900", 0x2C, true, false),
        ("gfx902", 0x2D, true, false),
        ("gfx904", 0x2E, true, false),
        ("gfx906", 0x2F, true, true),
        ("gfx908", 0x30, true, true),
        ("gfx909", 0x31, true, false),
        ("gfx90a", 0x3F, true, true),
        ("gfx90c", 0x32, true, false),
        ("gfx940", 0x40, true, true),
        ("gfx941", 0x4B, true, true),
        ("gfx942", 0x4C, true, true),
        ("gfx1010", 0x33, true, false),
        ("gfx1011", 0x34, true, false),
        ("gfx1012", 0x35, true, false),
        ("gfx1013", 0x42, true, false),
        ("gfx1030", 0x36, false, false),
        ("gfx1031", 0x37, false, false),
        ("gfx1032", 0x38, false, false),
        ("gfx1033", 0x39, false, false),
        ("gfx1034", 0x3E, false, false),
        ("gfx1035", 0x3D, false, false),
        ("gfx1036", 0x45, false, false),
        ("gfx1100", 0x41, false, false),
        ("gfx1101", 0x46, false, false),
        ("gfx1102", 0x47, false, false),
        ("gfx1103", 0x44, false, false),
        ("gfx1150", 0x43, false, false),
        ("gfx1151", 0x4A, false, false),
        ("gfx1200", 0x48, false, false),
        ("gfx1201", 0x4E, false, false),
    ];

    // The processors' versions, read once from their names, in the order of Machines (in this file: static fields of
    // one file are set in order, those of a partial class's other files in none).
    private static readonly GcnIsaVersion[] MachineVersions = [.. Machines.Select(m => GcnIsaVersion.Parse(m.Processor))];
}
