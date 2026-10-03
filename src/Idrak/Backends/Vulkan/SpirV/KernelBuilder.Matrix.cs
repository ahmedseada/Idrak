// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// What the cooperative-matrix products need beyond 32-bit values: workgroup arrays of 16-bit floats (capability Float16,
// for devices that report shaderFloat16) or of bfloat16 values (SPV_KHR_bfloat16, capability BFloat16TypeKHR, for devices
// that report shaderBFloat16Type), and cooperative matrices of either (SPV_KHR_cooperative_matrix, capability
// CooperativeMatrixKHR, subgroup scope; bfloat16 ones with BFloat16CooperativeMatrixKHR) loaded from and stored to
// workgroup memory, row-major. Values in registers stay 32-bit: a 16-bit value exists only in workgroup memory and
// inside a matrix.
internal sealed partial class KernelBuilder
{
    /// <summary>A workgroup-shared array of 16-bit floats: stores round a float to the nearest 16-bit float, loads widen it.</summary>
    public OperandArray SharedHalf(string name, int length) => SharedOperand(name, length, OperandType.Float16);

    /// <summary>
    /// A workgroup-shared array of matrix operands of <paramref name="type"/>: stores convert a float to it (the caller
    /// rounds first where the result must not depend on the device's rounding), loads widen it to a float.
    /// </summary>
    public OperandArray SharedOperand(string name, int length, OperandType type)
    {
        uint element = OperandElement(type);
        uint array = _m.TypeArray(element, (uint)length);
        uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Workgroup, array), StorageClass.Workgroup);
        _m.Name(variable, name);
        _sharedBytes += (type == OperandType.Float32 ? 4 : 2) * length;
        return new OperandArray(this, variable, length, type);
    }

    /// <summary>Whether a float is infinite.</summary>
    public Val IsInf(Val x) => Op(SpirvOp.IsInf, ScalarKind.Bool, x);

    /// <summary>
    /// A cooperative matrix type of <paramref name="rows"/> × <paramref name="columns"/>: <paramref name="operand"/>
    /// components for the operands (<paramref name="use"/> A or B: 16-bit floats or bfloat16), 32-bit floats for the
    /// accumulator. Declares the extensions and capabilities.
    /// </summary>
    public uint MatrixType(int rows, int columns, MatrixUse use, OperandType operand = OperandType.Float16)
    {
        if (operand == OperandType.Float32)
        {
            throw new ArgumentOutOfRangeException(nameof(operand), operand, "Matrix operands are 16-bit floats or bfloat16.");
        }

        _m.Capability(6022);                                             // CooperativeMatrixKHR
        _m.Extension("SPV_KHR_cooperative_matrix");
        if (operand == OperandType.BFloat16)
        {
            _m.Capability(5118);                                         // BFloat16CooperativeMatrixKHR
        }

        uint component = use == MatrixUse.Accumulator ? TypeOf(ScalarKind.Float) : OperandElement(operand);
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
    public uint MatrixLoad(uint type, OperandArray source, Val offset, int stride) =>
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

    // The scalar type of an operand array's elements, with its capability and extension declared.
    internal uint OperandElement(OperandType type)
    {
        switch (type)
        {
            case OperandType.Float16:
                _m.Capability(9);                                        // Float16
                return _m.TypeHalf();
            case OperandType.BFloat16:
                _m.Capability(5116);                                     // BFloat16TypeKHR
                _m.Extension("SPV_KHR_bfloat16");
                return _m.TypeBFloat16();
            default:
                return TypeOf(ScalarKind.Float);
        }
    }

    internal uint OperandPointer(uint variable, OperandType type, Val index) =>
        _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.Workgroup, OperandElement(type)), variable, index.Id);
}

/// <summary>What a cooperative matrix is for (the Use operand of OpTypeCooperativeMatrixKHR).</summary>
internal enum MatrixUse : uint
{
    A = 0,
    B = 1,
    Accumulator = 2,
}

/// <summary>The element type of a workgroup array of matrix operands.</summary>
internal enum OperandType
{
    /// <summary>16-bit IEEE floats (capability Float16).</summary>
    Float16,

    /// <summary>bfloat16 (SPV_KHR_bfloat16): float32's sign, exponent and first 7 mantissa bits.</summary>
    BFloat16,

    /// <summary>32-bit floats: only for emulated matrices (tests), holding values already rounded to bfloat16.</summary>
    Float32,
}

/// <summary>A workgroup-shared array of matrix operands (16-bit floats, bfloat16 or, emulated, floats), read and written as floats.</summary>
internal sealed class OperandArray(KernelBuilder builder, uint variable, int length, OperandType type)
{
    /// <summary>Elements.</summary>
    public int Length => length;

    /// <summary>The element type.</summary>
    public OperandType Type => type;

    /// <summary>Element <paramref name="index"/>, widened to a float; a stored float is converted to the element type.</summary>
    public Val this[Val index]
    {
        get
        {
            var m = builder.Module;
            uint floatType = builder.TypeOf(ScalarKind.Float);
            if (type == OperandType.Float32)
            {
                return new Val(builder, m.Value(SpirvOp.Load, floatType, Pointer(index)), ScalarKind.Float);
            }

            uint narrow = m.Value(SpirvOp.Load, builder.OperandElement(type), Pointer(index));
            return new Val(builder, m.Value(SpirvOp.FConvert, floatType, narrow), ScalarKind.Float);
        }
        set
        {
            KernelBuilder.Expect(value, ScalarKind.Float);
            var m = builder.Module;
            m.Code(SpirvOp.Store, Pointer(index), type == OperandType.Float32 ? value.Id : m.Value(SpirvOp.FConvert, builder.OperandElement(type), value.Id));
        }
    }

    /// <summary>A pointer to element <paramref name="index"/> (where a cooperative matrix is loaded from).</summary>
    internal uint Pointer(Val index) => builder.OperandPointer(variable, type, index);
}
