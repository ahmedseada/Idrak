// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

/// <summary>
/// A compute kernel for <see cref="VulkanBackend.Dispatch"/>: SPIR-V 1.3 words (Vulkan 1.1), entry point "main", its local
/// size in the module. Descriptor set 0 binding i is the i-th storage of a dispatch (a storage buffer of 32-bit words);
/// its scalars are one push-constant block of <see cref="PushConstantBytes"/> bytes. Each backend builds the pipeline
/// once, on the kernel's first dispatch there, and keeps it. <see cref="Writes"/> tells the runtime which bindings the
/// kernel stores into, so it orders a dispatch after earlier ones only where they touch the same storages.
/// </summary>
internal sealed class VulkanKernel
{
    /// <summary>The largest push-constant block a kernel may declare (the least every Vulkan device supports).</summary>
    public const int MaxPushConstantBytes = 128;

    /// <summary>The most storages a kernel binds (one bit each in <see cref="Writes"/>).</summary>
    public const int MaxBindings = 64;

    private const uint SpirvMagic = 0x07230203;

    /// <summary>A kernel from its SPIR-V words; <paramref name="writes"/> has bit i set when it writes binding i (all of them when null).</summary>
    public VulkanKernel(uint[] spirv, int bindings, int pushConstantBytes, string name = "kernel", ulong? writes = null)
    {
        ArgumentNullException.ThrowIfNull(spirv);
        ArgumentNullException.ThrowIfNull(name);
        if (spirv.Length < 5 || spirv[0] != SpirvMagic)
        {
            throw new ArgumentException($"Vulkan kernel '{name}': not a SPIR-V module (no magic number).", nameof(spirv));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(bindings);
        if (pushConstantBytes < 0 || pushConstantBytes > MaxPushConstantBytes || pushConstantBytes % 4 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pushConstantBytes), pushConstantBytes,
                $"Vulkan kernel '{name}': push constants are 4-byte members, at most {MaxPushConstantBytes} bytes.");
        }

        if (bindings > MaxBindings)
        {
            throw new ArgumentOutOfRangeException(nameof(bindings), bindings, $"Vulkan kernel '{name}': at most {MaxBindings} storages.");
        }

        Spirv = spirv;
        Bindings = bindings;
        Writes = writes ?? (bindings == MaxBindings ? ulong.MaxValue : (1UL << bindings) - 1);
        PushConstantBytes = pushConstantBytes;
        Name = name;
    }

    /// <summary>The module's words.</summary>
    public uint[] Spirv { get; }

    /// <summary>The storage buffers it reads and writes (bindings 0 to Bindings - 1 of set 0).</summary>
    public int Bindings { get; }

    /// <summary>Bit i set when the kernel writes binding i; the others it only reads.</summary>
    public ulong Writes { get; }

    /// <summary>The size of its push-constant block (0: none).</summary>
    public int PushConstantBytes { get; }

    /// <summary>A name for messages.</summary>
    public string Name { get; }

    /// <summary>The pipeline built for it on the backend that dispatched it last (looked up without a dictionary).</summary>
    internal object? LastPipeline;

    /// <summary>The sized variant chosen for it by the backend that chose last (VulkanBackend.SubgroupSize.cs).</summary>
    internal object? LastSized;

    /// <summary>The subgroup size its pipeline requires (VK_EXT_subgroup_size_control), or 0 for the device's default.</summary>
    internal int RequiredSubgroupSize { get; init; }

    /// <summary>Whether the module uses subgroup operations (declares the GroupNonUniform capability).</summary>
    internal bool UsesSubgroups => (_usesSubgroups ??= DeclaresCapability(Spirv, 61)) == true;

    private bool? _usesSubgroups;

    /// <summary>The module's workgroup width (its LocalSize execution mode's x; 0 when it declares none).</summary>
    internal int LocalSizeX => _localSizeX ??= ReadLocalSizeX(Spirv);

    private int? _localSizeX;

    // Whether the module declares `capability` (OpCapability, which comes before every other instruction but the header).
    private static bool DeclaresCapability(uint[] words, uint capability)
    {
        for (int i = 5; i < words.Length;)
        {
            uint op = words[i] & 0xFFFF, count = words[i] >> 16;
            if (op != 17 || count == 0)
            {
                return false;                                                  // past the capabilities
            }

            if (count >= 2 && i + 1 < words.Length && words[i + 1] == capability)
            {
                return true;
            }

            i += (int)count;
        }

        return false;
    }

    // The x of the first OpExecutionMode LocalSize (opcode 16, mode 17), or 0.
    private static int ReadLocalSizeX(uint[] words)
    {
        for (int i = 5; i < words.Length;)
        {
            uint op = words[i] & 0xFFFF, count = words[i] >> 16;
            if (count == 0)
            {
                return 0;
            }

            if (op == 16 && count >= 4 && i + 3 < words.Length && words[i + 2] == 17)
            {
                return (int)words[i + 3];
            }

            i += (int)count;
        }

        return 0;
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}
