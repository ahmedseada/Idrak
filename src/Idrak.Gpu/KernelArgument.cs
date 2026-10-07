// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu;

/// <summary>
/// One argument of a plug-in's CUDA or HIP kernel launch (<see cref="Cuda.CudaKernel.Launch(Backend, uint, uint, uint, uint, uint, uint, ReadOnlySpan{KernelArgument}, uint)"/>,
/// <see cref="Hip.HipKernel.Launch(Backend, uint, uint, uint, uint, uint, uint, ReadOnlySpan{KernelArgument}, uint)"/>):
/// the device address of a storage of the launching device (a <c>float*</c> parameter, 64 bits), a 32-bit integer, a
/// float or a 64-bit integer. Storages, <c>int</c>, <c>uint</c>, <c>float</c> and <c>long</c> convert to it implicitly,
/// so a launch reads <c>kernel.Launch(backend, blocks, 1, 1, 256, 1, 1, [rows, scales, output, n, width])</c>. Each
/// argument is checked against the kernel's declared parameter before the launch.
/// </summary>
public readonly struct KernelArgument
{
    private readonly ulong _bits;

    private KernelArgument(KernelArgumentKind kind, ulong bits, Storage? storage = null, long offset = 0)
    {
        Kind = kind;
        _bits = bits;
        Storage = storage;
        Offset = offset;
    }

    /// <summary>What the argument is.</summary>
    internal KernelArgumentKind Kind { get; }

    /// <summary>The storage whose address is passed (pointer arguments only).</summary>
    internal Storage? Storage { get; }

    /// <summary>Floats past the start of <see cref="Storage"/> the address points at.</summary>
    internal long Offset { get; }

    /// <summary>The value's bits, widened to 64 (scalars only; a pointer's address is the device's to resolve).</summary>
    internal ulong Bits => _bits;

    /// <summary>The address of <paramref name="storage"/>'s float <paramref name="offset"/> (0: its start), on the launching device.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="offset"/> is negative or past the storage's end.</exception>
    public static KernelArgument Pointer(Storage storage, long offset = 0)
    {
        ArgumentNullException.ThrowIfNull(storage);
        if (offset < 0 || offset > storage.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), offset, $"Past the end of a storage of {storage.Length} floats.");
        }

        return new KernelArgument(KernelArgumentKind.Pointer, 0, storage, offset);
    }

    /// <summary>A 32-bit integer (an <c>int</c> or <c>unsigned int</c> parameter).</summary>
    public static KernelArgument Int32(int value) => new(KernelArgumentKind.Int32, (uint)value);

    /// <summary>A float (a <c>float</c> parameter).</summary>
    public static KernelArgument Float32(float value) => new(KernelArgumentKind.Float32, BitConverter.SingleToUInt32Bits(value));

    /// <summary>A 64-bit integer (a <c>long long</c> or <c>size_t</c> parameter).</summary>
    public static KernelArgument Int64(long value) => new(KernelArgumentKind.Int64, (ulong)value);

    /// <summary>The address of the storage's first float.</summary>
    public static implicit operator KernelArgument(Storage storage) => Pointer(storage);

    /// <summary>A 32-bit integer.</summary>
    public static implicit operator KernelArgument(int value) => Int32(value);

    /// <summary>A 32-bit unsigned integer (passed as its bits).</summary>
    public static implicit operator KernelArgument(uint value) => Int32((int)value);

    /// <summary>A float.</summary>
    public static implicit operator KernelArgument(float value) => Float32(value);

    /// <summary>A 64-bit integer.</summary>
    public static implicit operator KernelArgument(long value) => Int64(value);

    /// <inheritdoc />
    public override string ToString() => Kind switch
    {
        KernelArgumentKind.Pointer => Offset == 0 ? $"storage of {Storage!.Length} floats" : $"storage of {Storage!.Length} floats at {Offset}",
        KernelArgumentKind.Int32 => ((int)(uint)_bits).ToString(CultureInfo.InvariantCulture),
        KernelArgumentKind.Float32 => BitConverter.UInt32BitsToSingle((uint)_bits).ToString(CultureInfo.InvariantCulture) + "f",
        KernelArgumentKind.Int64 => ((long)_bits).ToString(CultureInfo.InvariantCulture) + "L",
        _ => "(none)",
    };

    /// <summary>
    /// Checks <paramref name="arguments"/> against a kernel's parameters, one letter each ('p' a device pointer, 'i' a
    /// 32-bit integer, 'f' a float, 'l' a 64-bit integer; 'w' a 32-bit word that takes an integer or a float, 'x' a
    /// 64-bit word that takes a pointer or a 64-bit integer).
    /// </summary>
    internal static void CheckAgainst(string kernel, string parameters, ReadOnlySpan<KernelArgument> arguments)
    {
        if (arguments.Length != parameters.Length)
        {
            throw new ArgumentException($"Kernel '{kernel}' declares {parameters.Length} parameters ({parameters}); {arguments.Length} arguments given.", nameof(arguments));
        }

        for (int i = 0; i < arguments.Length; i++)
        {
            var kind = arguments[i].Kind;
            bool fits = parameters[i] switch
            {
                'p' => kind == KernelArgumentKind.Pointer,
                'i' => kind == KernelArgumentKind.Int32,
                'f' => kind == KernelArgumentKind.Float32,
                'l' => kind == KernelArgumentKind.Int64,
                'w' => kind is KernelArgumentKind.Int32 or KernelArgumentKind.Float32,
                'x' => kind is KernelArgumentKind.Pointer or KernelArgumentKind.Int64,
                _ => false,
            };
            if (!fits)
            {
                throw new ArgumentException(
                    $"Kernel '{kernel}': argument {i} is {Describe(kind)} ({arguments[i]}), but parameter {i} takes {Describe(parameters[i])}.", nameof(arguments));
            }
        }
    }

    private static string Describe(KernelArgumentKind kind) => kind switch
    {
        KernelArgumentKind.Pointer => "a storage",
        KernelArgumentKind.Int32 => "a 32-bit integer",
        KernelArgumentKind.Float32 => "a float",
        KernelArgumentKind.Int64 => "a 64-bit integer",
        _ => "missing",
    };

    private static string Describe(char parameter) => parameter switch
    {
        'p' => "a storage",
        'i' => "a 32-bit integer",
        'f' => "a float",
        'l' => "a 64-bit integer",
        'w' => "a 32-bit integer or a float",
        'x' => "a storage or a 64-bit integer",
        _ => $"an unsupported type ('{parameter}')",
    };
}

/// <summary>What a <see cref="KernelArgument"/> holds (default: nothing, refused at launch).</summary>
internal enum KernelArgumentKind : byte
{
    None,
    Pointer,
    Int32,
    Float32,
    Int64,
}
