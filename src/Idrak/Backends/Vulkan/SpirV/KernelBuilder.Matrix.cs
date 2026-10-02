// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// What the cooperative-matrix products need beyond 32-bit values: workgroup arrays of 16-bit floats (capability Float16,
// for devices that report shaderFloat16) and cooperative matrices (SPV_KHR_cooperative_matrix, capability
// CooperativeMatrixKHR, subgroup scope) loaded from and stored to workgroup memory, row-major. Values in registers stay
// 32-bit: a 16-bit float exists only in workgroup memory and inside a matrix.
internal sealed partial class KernelBuilder
{
    /// <summary>A workgroup-shared array of 16-bit floats: stores round a float to the nearest 16-bit float, loads widen it.</summary>
    public HalfArray SharedHalf(string name, int length)
    {
        _m.Capability(9);                                                // Float16
        uint type = _m.TypeArray(_m.TypeHalf(), (uint)length);
        uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Workgroup, type), StorageClass.Workgroup);
        _m.Name(variable, name);
        _sharedBytes += 2 * length;
        return new HalfArray(this, variable, length);
    }

    /// <summary>Whether a float is infinite.</summary>
    public Val IsInf(Val x) => Op(SpirvOp.IsInf, ScalarKind.Bool, x);

    /// <summary>
    /// A cooperative matrix type of <paramref name="rows"/> × <paramref name="columns"/>: 16-bit floats for the operands
    /// (<paramref name="use"/> A or B), 32-bit floats for the accumulator. Declares the extension and its capability.
    /// </summary>
    public uint MatrixType(int rows, int columns, MatrixUse use)
    {
        _m.Capability(9);                                                // Float16
        _m.Capability(6022);                                             // CooperativeMatrixKHR
        _m.Extension("SPV_KHR_cooperative_matrix");
        uint component = use == MatrixUse.Accumulator ? TypeOf(ScalarKind.Float) : _m.TypeHalf();
        return _m.TypeCooperativeMatrix(component, (uint)rows, (uint)columns, (uint)use);
    }

    /// <summary>A function-local cooperative matrix of <paramref name="type"/> (an accumulator kept across loop iterations).</summary>
    public uint MatrixLocal(uint type) => _m.LocalVariable(_m.TypePointer(StorageClass.Function, type));

    /// <summary>Stores a matrix of zeros into the local <paramref name="local"/> of <paramref name="type"/>.</summary>
    public void MatrixZero(uint local, uint type) => _m.Code(SpirvOp.Store, local, _m.ConstantComposite(type, _m.ConstantFloat(0f)));

    /// <summary>
    /// Loads a matrix of <paramref name="type"/> from <paramref name="source"/>, row-major: row r starts at element
    /// <paramref name="offset"/> + r · <paramref name="stride"/>. Every invocation of the subgroup must take part.
    /// </summary>
    public uint MatrixLoad(uint type, HalfArray source, Val offset, int stride) =>
        _m.Value(SpirvOp.CooperativeMatrixLoadKHR, type, source.Pointer(offset), _m.ConstantUInt(0), _m.ConstantUInt((uint)stride));   // layout 0: RowMajorKHR

    /// <summary>local = a · b + local, the accumulator's sums in 32-bit floats. Every invocation of the subgroup must take part.</summary>
    public void MatrixMulAdd(uint local, uint accumulatorType, uint a, uint b)
    {
        uint c = _m.Value(SpirvOp.Load, accumulatorType, local);
        _m.Code(SpirvOp.Store, local, _m.Value(SpirvOp.CooperativeMatrixMulAddKHR, accumulatorType, a, b, c));
    }

    /// <summary>Stores the accumulator <paramref name="local"/> into <paramref name="target"/>, row-major (as <see cref="MatrixLoad"/>).</summary>
    public void MatrixStore(uint local, uint accumulatorType, SharedArray target, Val offset, int stride)
    {
        uint value = _m.Value(SpirvOp.Load, accumulatorType, local);
        _m.Code(SpirvOp.CooperativeMatrixStoreKHR, target.Pointer(offset), value, _m.ConstantUInt(0), _m.ConstantUInt((uint)stride));
    }

    internal uint HalfElement(uint variable, Val index) =>
        _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.Workgroup, _m.TypeHalf()), variable, index.Id);
}

/// <summary>What a cooperative matrix is for (the Use operand of OpTypeCooperativeMatrixKHR).</summary>
internal enum MatrixUse : uint
{
    A = 0,
    B = 1,
    Accumulator = 2,
}

/// <summary>A workgroup-shared array of 16-bit floats, read and written as floats.</summary>
internal sealed class HalfArray(KernelBuilder builder, uint variable, int length)
{
    /// <summary>Elements.</summary>
    public int Length => length;

    /// <summary>Element <paramref name="index"/>, widened to a float; a stored float is rounded to 16 bits.</summary>
    public Val this[Val index]
    {
        get
        {
            var m = builder.Module;
            uint half = m.Value(SpirvOp.Load, m.TypeHalf(), builder.HalfElement(variable, index));
            return new Val(builder, m.Value(SpirvOp.FConvert, builder.TypeOf(ScalarKind.Float), half), ScalarKind.Float);
        }
        set
        {
            KernelBuilder.Expect(value, ScalarKind.Float);
            var m = builder.Module;
            m.Code(SpirvOp.Store, builder.HalfElement(variable, index), m.Value(SpirvOp.FConvert, m.TypeHalf(), value.Id));
        }
    }

    /// <summary>A pointer to element <paramref name="index"/> (where a cooperative matrix is loaded from).</summary>
    internal uint Pointer(Val index) => builder.HalfElement(variable, index);
}
