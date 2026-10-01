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

    /// <inheritdoc />
    public override string ToString() => Name;
}
