// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Backends.Vulkan;

/// <summary>
/// Writes one SPIR-V 1.3 compute module (Vulkan 1.1): capability Shader, the Logical / GLSL450 memory model, the
/// GLSL.std.450 instructions, one entry point "main" with its local size, and a single function. Types and constants
/// are created once and shared; every other instruction goes into its section in the order the specification requires.
/// A module may declare extensions (SPV_KHR_cooperative_matrix) for the devices that report them.
/// </summary>
internal sealed class SpirvModule
{
    /// <summary>SPIR-V 1.3, the version Vulkan 1.1 consumes without extensions (StorageBuffer is core).</summary>
    public const uint Version = 0x00010300;

    private const uint Magic = 0x07230203;

    private readonly List<uint> _annotations = [];
    private readonly List<uint> _debug = [];
    private readonly List<uint> _globals = [];                           // types, constants and global variables
    private readonly List<uint> _variables = [];                         // the function's variables (first block)
    private readonly List<uint> _prologue = [];                          // loads at the start of the first block
    private readonly List<uint> _body = [];                              // the function's code after its first label
    private readonly Dictionary<string, uint> _types = [];
    private readonly Dictionary<(uint Type, uint Bits), uint> _constants = [];
    private readonly List<uint> _interface = [];
    private readonly SortedSet<uint> _capabilities = [];
    private readonly SortedSet<string> _extensions = new(StringComparer.Ordinal);
    private uint _next = 1;
    private bool _terminated;

    /// <summary>Creates the module and its fixed parts: the extended instruction set and the function's first block.</summary>
    public SpirvModule()
    {
        GlslSet = Id();
        Function = Id();
        EntryLabel = Id();
    }

    /// <summary>The id of the GLSL.std.450 import.</summary>
    public uint GlslSet { get; }

    /// <summary>The id of the entry-point function.</summary>
    public uint Function { get; }

    /// <summary>The label of the function's first block, where its variables are declared.</summary>
    public uint EntryLabel { get; }

    /// <summary>Whether the current block already ended with a branch or return.</summary>
    public bool Terminated => _terminated;

    /// <summary>A fresh result id.</summary>
    public uint Id() => _next++;

    /// <summary>Declares a capability beyond Shader (61 GroupNonUniform, 63 GroupNonUniformArithmetic, …), once.</summary>
    public void Capability(uint capability) => _capabilities.Add(capability);

    /// <summary>Declares a SPIR-V extension ("SPV_KHR_cooperative_matrix"), once.</summary>
    public void Extension(string name) => _extensions.Add(name);

    // ------------------------------------------------------------------ types

    public uint TypeVoid() => Type("void", () => Emit(_globals, SpirvOp.TypeVoid, Id()));

    public uint TypeBool() => Type("bool", () => Emit(_globals, SpirvOp.TypeBool, Id()));

    /// <summary>A 32-bit integer type, signed or not.</summary>
    public uint TypeInt(bool signed) => Type(signed ? "int" : "uint", () => Emit(_globals, SpirvOp.TypeInt, Id(), 32, signed ? 1u : 0u));

    public uint TypeFloat() => Type("float", () => Emit(_globals, SpirvOp.TypeFloat, Id(), 32));

    /// <summary>The 16-bit float type (capability Float16, declared by the caller).</summary>
    public uint TypeHalf() => Type("half", () => Emit(_globals, SpirvOp.TypeFloat, Id(), 16));

    /// <summary>
    /// The bfloat16 type (SPV_KHR_bfloat16: a 16-bit float with encoding BFloat16KHR, 0; capability BFloat16TypeKHR and the
    /// extension declared by the caller).
    /// </summary>
    public uint TypeBFloat16() => Type("bfloat16", () => Emit(_globals, SpirvOp.TypeFloat, Id(), 16, 0));

    /// <summary>
    /// A cooperative matrix type (SPV_KHR_cooperative_matrix) of <paramref name="component"/>, subgroup scope,
    /// <paramref name="rows"/> × <paramref name="columns"/>, for use <paramref name="use"/> (0 the left operand A, 1 the
    /// right operand B, 2 the accumulator).
    /// </summary>
    public uint TypeCooperativeMatrix(uint component, uint rows, uint columns, uint use)
    {
        uint scope = ConstantUInt(3), r = ConstantUInt(rows), c = ConstantUInt(columns), u = ConstantUInt(use);   // scope 3: Subgroup
        return Type($"coop:{component}:{rows}:{columns}:{use}", () => Emit(_globals, SpirvOp.TypeCooperativeMatrixKHR, Id(), component, scope, r, c, u));
    }

    public uint TypeVector(uint component, uint count) =>
        Type($"vec{count}:{component}", () => Emit(_globals, SpirvOp.TypeVector, Id(), component, count));

    /// <summary>A fixed-length array (workgroup memory: no explicit layout).</summary>
    public uint TypeArray(uint element, uint length)
    {
        uint lengthId = Constant(TypeInt(false), length);
        return Type($"array:{element}:{length}", () => Emit(_globals, SpirvOp.TypeArray, Id(), element, lengthId));
    }

    /// <summary>A runtime array with a 4-byte stride (the storage buffers' 32-bit words).</summary>
    public uint TypeRuntimeArray(uint element) => Type($"rtarray:{element}", () =>
    {
        uint id = Emit(_globals, SpirvOp.TypeRuntimeArray, Id(), element);
        Decorate(id, Decoration.ArrayStride, 4);
        return id;
    });

    /// <summary>A new struct type (never shared: structs carry their own layout decorations).</summary>
    public uint TypeStruct(params uint[] members) => TypeStructAs(Id(), members);

    /// <summary>A new struct type with an id reserved earlier (code may refer to it before it is defined).</summary>
    public uint TypeStructAs(uint id, uint[] members) => Emit(_globals, SpirvOp.TypeStruct, [id, .. members]);

    public uint TypePointer(StorageClass storage, uint type) =>
        Type($"ptr:{(uint)storage}:{type}", () => Emit(_globals, SpirvOp.TypePointer, Id(), (uint)storage, type));

    public uint TypeFunction(uint result) => Type($"fn:{result}", () => Emit(_globals, SpirvOp.TypeFunction, Id(), result));

    private uint Type(string key, Func<uint> create)
    {
        if (!_types.TryGetValue(key, out uint id))
        {
            id = create();
            _types[key] = id;
        }

        return id;
    }

    // ------------------------------------------------------------------ constants

    /// <summary>A 32-bit constant of <paramref name="type"/> with these bits.</summary>
    public uint Constant(uint type, uint bits)
    {
        if (!_constants.TryGetValue((type, bits), out uint id))
        {
            id = Emit(_globals, SpirvOp.Constant, type, Id(), bits);
            _constants[(type, bits)] = id;
        }

        return id;
    }

    public uint ConstantInt(int value) => Constant(TypeInt(true), unchecked((uint)value));

    public uint ConstantUInt(uint value) => Constant(TypeInt(false), value);

    public uint ConstantFloat(float value) => Constant(TypeFloat(), BitConverter.SingleToUInt32Bits(value));

    /// <summary>A composite constant with every component <paramref name="constituent"/> (a cooperative matrix takes one).</summary>
    public uint ConstantComposite(uint type, uint constituent) =>
        Type($"composite:{type}:{constituent}", () => Emit(_globals, SpirvOp.ConstantComposite, type, Id(), constituent));

    public uint ConstantBool(bool value) => Type(value ? "true" : "false",
        () => Emit(_globals, value ? SpirvOp.ConstantTrue : SpirvOp.ConstantFalse, TypeBool(), Id()));

    // ------------------------------------------------------------------ globals, decorations, names

    /// <summary>A module-scope variable; Input variables are added to the entry point's interface.</summary>
    public uint GlobalVariable(uint pointerType, StorageClass storage) => GlobalVariableAs(Id(), pointerType, storage);

    /// <summary>A module-scope variable with an id reserved earlier.</summary>
    public uint GlobalVariableAs(uint id, uint pointerType, StorageClass storage)
    {
        Emit(_globals, SpirvOp.Variable, pointerType, id, (uint)storage);
        if (storage == StorageClass.Input)
        {
            _interface.Add(id);
        }

        return id;
    }

    public void Decorate(uint target, Decoration decoration, params uint[] operands) =>
        Emit(_annotations, SpirvOp.Decorate, [target, (uint)decoration, .. operands]);

    public void MemberDecorate(uint structType, uint member, Decoration decoration, params uint[] operands) =>
        Emit(_annotations, SpirvOp.MemberDecorate, [structType, member, (uint)decoration, .. operands]);

    /// <summary>A debug name (shown by spirv-dis and driver tools).</summary>
    public void Name(uint target, string name) => Emit(_debug, SpirvOp.Name, [target, .. Text(name)]);

    public void MemberName(uint structType, uint member, string name) => Emit(_debug, SpirvOp.MemberName, [structType, member, .. Text(name)]);

    // ------------------------------------------------------------------ function code

    /// <summary>A Function-storage variable, declared in the first block as SPIR-V requires.</summary>
    public uint LocalVariable(uint pointerType) => Emit(_variables, SpirvOp.Variable, pointerType, Id(), (uint)StorageClass.Function);

    /// <summary>
    /// An instruction with a result placed at the start of the first block (after the variables), so its value dominates
    /// all the code: loads of built-in variables and push constants, which may first be asked for inside a branch.
    /// </summary>
    public uint PrologueValue(SpirvOp op, uint type, params uint[] operands)
    {
        uint id = Id();
        Emit(_prologue, op, [type, id, .. operands]);
        return id;
    }

    /// <summary>Appends an instruction to the current block.</summary>
    public void Code(SpirvOp op, params uint[] operands)
    {
        if (_terminated)
        {
            throw new InvalidOperationException($"SPIR-V: {op} after the block ended.");
        }

        Emit(_body, op, operands);
    }

    /// <summary>Appends an instruction with a result (type, new id, operands) and returns the id.</summary>
    public uint Value(SpirvOp op, uint type, params uint[] operands)
    {
        uint id = Id();
        Code(op, [type, id, .. operands]);
        return id;
    }

    /// <summary>A GLSL.std.450 instruction on its operands.</summary>
    public uint Ext(Glsl instruction, uint type, params uint[] operands) => Value(SpirvOp.ExtInst, type, [GlslSet, (uint)instruction, .. operands]);

    /// <summary>Ends the current block with a branch, a conditional branch or a return.</summary>
    public void Terminate(SpirvOp op, params uint[] operands)
    {
        Code(op, operands);
        _terminated = true;
    }

    /// <summary>Starts a new block (the previous one must have ended).</summary>
    public void Label(uint label)
    {
        if (!_terminated)
        {
            throw new InvalidOperationException("SPIR-V: a block starts before the previous one ended.");
        }

        _terminated = false;
        Emit(_body, SpirvOp.Label, label);
    }

    // ------------------------------------------------------------------ the module

    /// <summary>The finished module's words: header, then each section in the order of the specification's logical layout.</summary>
    public uint[] Finish(uint localX, uint localY, uint localZ)
    {
        if (!_terminated)
        {
            Terminate(SpirvOp.Return);
        }

        uint voidType = TypeVoid();
        uint functionType = TypeFunction(voidType);
        var words = new List<uint>(_globals.Count + _body.Count + 128) { Magic, Version, 0, 0, 0 };
        Emit(words, SpirvOp.Capability, 1);                              // Shader
        foreach (uint capability in _capabilities)
        {
            Emit(words, SpirvOp.Capability, capability);
        }

        foreach (string extension in _extensions)
        {
            Emit(words, SpirvOp.Extension, Text(extension));
        }

        Emit(words, SpirvOp.ExtInstImport, [GlslSet, .. Text("GLSL.std.450")]);
        Emit(words, SpirvOp.MemoryModel, 0, 1);                          // Logical, GLSL450
        Emit(words, SpirvOp.EntryPoint, [5, Function, .. Text("main"), .. _interface]);   // GLCompute
        Emit(words, SpirvOp.ExecutionMode, Function, 17, localX, localY, localZ);         // LocalSize
        words.AddRange(_debug);
        words.AddRange(_annotations);
        words.AddRange(_globals);
        Emit(words, SpirvOp.Function, voidType, Function, 0, functionType);
        Emit(words, SpirvOp.Label, EntryLabel);
        words.AddRange(_variables);
        words.AddRange(_prologue);
        words.AddRange(_body);
        Emit(words, SpirvOp.FunctionEnd);
        words[3] = _next;                                                // the id bound
        return [.. words];
    }

    private static uint Emit(List<uint> section, SpirvOp op, params uint[] operands)
    {
        section.Add((uint)(operands.Length + 1) << 16 | (uint)op);
        section.AddRange(operands);
        return operands.Length switch
        {
            0 => 0,
            _ when op is SpirvOp.Constant or SpirvOp.ConstantComposite or SpirvOp.ConstantTrue or SpirvOp.ConstantFalse or SpirvOp.Variable => operands[1],
            _ => operands[0],
        };
    }

    // A literal string: UTF-8, nul-terminated, padded with zeros to whole words.
    private static uint[] Text(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        var words = new uint[bytes.Length / 4 + 1];
        for (int i = 0; i < bytes.Length; i++)
        {
            words[i / 4] |= (uint)bytes[i] << (8 * (i % 4));
        }

        return words;
    }
}
