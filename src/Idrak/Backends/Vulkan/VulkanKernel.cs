// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

/// <summary>
/// A compute kernel for <see cref="VulkanBackend.Dispatch"/>: SPIR-V 1.3 words (Vulkan 1.1), entry point "main", its local
/// size in the module. Descriptor set 0 binding i is the i-th storage of a dispatch (a storage buffer of 32-bit words);
/// its scalars are one push-constant block of <see cref="PushConstantBytes"/> bytes. Each backend builds the pipeline
/// once, on the kernel's first dispatch there, and keeps it.
/// </summary>
internal sealed class VulkanKernel
{
    /// <summary>The largest push-constant block a kernel may declare (the least every Vulkan device supports).</summary>
    public const int MaxPushConstantBytes = 128;

    private const uint SpirvMagic = 0x07230203;

    public VulkanKernel(uint[] spirv, int bindings, int pushConstantBytes, string name = "kernel")
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

        Spirv = spirv;
        Bindings = bindings;
        PushConstantBytes = pushConstantBytes;
        Name = name;
    }

    /// <summary>The module's words.</summary>
    public uint[] Spirv { get; }

    /// <summary>The storage buffers it reads and writes (bindings 0 to Bindings - 1 of set 0).</summary>
    public int Bindings { get; }

    /// <summary>The size of its push-constant block (0: none).</summary>
    public int PushConstantBytes { get; }

    /// <summary>A name for messages.</summary>
    public string Name { get; }

    /// <inheritdoc />
    public override string ToString() => Name;
}
