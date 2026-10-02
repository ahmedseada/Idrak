// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

/// <summary>
/// A generated compute kernel as the Vulkan backend loads it: SPIR-V words with entry point "main", storage buffers at
/// descriptor set 0, bindings 0 … <see cref="Bindings"/> - 1 (in the order the kernel declared them), and
/// <see cref="PushBytes"/> bytes of push constants (4-byte members in declaration order).
/// </summary>
/// <param name="Name">The kernel's name (the generator's catalog key).</param>
/// <param name="Words">The SPIR-V module.</param>
/// <param name="Bindings">Storage buffers the dispatch passes, in binding order.</param>
/// <param name="PushBytes">Size of the push-constant block (0 when there is none, at most 128).</param>
/// <param name="LocalSizeX">Workgroup width.</param>
/// <param name="LocalSizeY">Workgroup height.</param>
/// <param name="LocalSizeZ">Workgroup depth.</param>
/// <param name="BindingNames">Each binding's name, for messages and documentation.</param>
/// <param name="PushNames">Each push constant's name with its type ("n:int", "alpha:float").</param>
/// <param name="Writes">Bit i set when the kernel writes binding i (the others are only read, and decorated NonWritable).</param>
/// <param name="SharedBytes">Workgroup memory the kernel declares (bytes), checked against the device's limit.</param>
internal sealed record SpirvKernel(string Name, uint[] Words, int Bindings, int PushBytes, int LocalSizeX, int LocalSizeY, int LocalSizeZ,
    string[] BindingNames, string[] PushNames, ulong Writes, int SharedBytes = 0)
{
    /// <summary>Invocations per workgroup.</summary>
    public int LocalSize => LocalSizeX * LocalSizeY * LocalSizeZ;

    /// <summary>The contract in one line: "name(x, y | n:int, alpha:float) local 256".</summary>
    public override string ToString() =>
        $"{Name}({string.Join(", ", BindingNames)} | {string.Join(", ", PushNames)}) local {LocalSizeX}x{LocalSizeY}x{LocalSizeZ}";
}

/// <summary>The scalar types kernels compute with.</summary>
internal enum ScalarKind
{
    Bool,
    Int,
    UInt,
    Float,
}

/// <summary>
/// Builds a compute kernel in C# that reads like the code it generates:
/// <code>
/// var k = new KernelBuilder("axpy", 256);
/// var x = k.Buffer("x"); var y = k.Buffer("y");
/// var n = k.PushInt("n"); var alpha = k.PushFloat("alpha");
/// k.For(k.GlobalX, n, i => y[i] = k.Fma(x[i], alpha, y[i]), step: k.GridStrideX);
/// SpirvKernel kernel = k.Build();
/// </code>
/// Values (<see cref="Val"/>) are SSA results valid in the block that made them and the blocks it dominates; a value
/// needed after an If or a loop goes through a <see cref="Var"/>. Control flow is structured (each If and loop has
/// its merge block), as Vulkan requires.
/// </summary>
internal sealed class KernelBuilder
{
    private readonly SpirvModule _m = new();
    private readonly string _name;
    private readonly int _localX, _localY, _localZ;
    private readonly List<string> _bindingNames = [];
    private readonly List<uint> _bindingVariables = [];
    private ulong _writes;
    private readonly List<(string Name, ScalarKind Kind)> _push = [];
    private readonly uint _pushStruct, _pushVariable;
    private readonly uint _bufferPointer, _bufferElement;
    private readonly Dictionary<BuiltIn, uint> _builtIns = [];
    private readonly Dictionary<(BuiltIn, int), Val> _builtInValues = [];
    private int _sharedBytes;
    private readonly bool _subgroups = t_subgroups;

    /// <summary>
    /// Whether kernels built on this thread from now on reduce through subgroup arithmetic (GroupNonUniformArithmetic,
    /// for devices that report it for compute shaders) instead of workgroup memory alone. Set around a build by the
    /// kernel catalog.
    /// </summary>
    [ThreadStatic]
    internal static bool t_subgroups;

    /// <summary>Starts a kernel with this workgroup size.</summary>
    public KernelBuilder(string name, int localSizeX, int localSizeY = 1, int localSizeZ = 1)
    {
        _name = name;
        (_localX, _localY, _localZ) = (localSizeX, localSizeY, localSizeZ);
        _m.Name(_m.Function, name);
        _pushStruct = _m.Id();
        _pushVariable = _m.Id();

        // Every storage buffer is struct { float data[]; } (Block, member offset 0, stride 4).
        uint array = _m.TypeRuntimeArray(TypeOf(ScalarKind.Float));
        uint block = _m.TypeStruct(array);
        _m.Decorate(block, Decoration.Block);
        _m.MemberDecorate(block, 0, Decoration.Offset, 0);
        _m.Name(block, "Words");
        _m.MemberName(block, 0, "data");
        _bufferPointer = _m.TypePointer(StorageClass.StorageBuffer, block);
        _bufferElement = _m.TypePointer(StorageClass.StorageBuffer, TypeOf(ScalarKind.Float));
    }

    internal SpirvModule Module => _m;

    /// <summary>Workgroup width.</summary>
    public int LocalSizeX => _localX;

    /// <summary>Whether this kernel's reductions use subgroup arithmetic (see <see cref="t_subgroups"/>).</summary>
    public bool Subgroups => _subgroups;

    // ------------------------------------------------------------------ declarations

    /// <summary>The next storage buffer (binding = how many were declared before it).</summary>
    /// <param name="name">The binding's name.</param>
    /// <param name="reread">Every load reads memory again (Volatile, Coherent): for a binding one invocation stores into
    /// and loads back from at addresses only known at run time, so no compiler keeps an earlier value of it.</param>
    public Buf Buffer(string name, bool reread = false)
    {
        uint variable = _m.GlobalVariable(_bufferPointer, StorageClass.StorageBuffer);
        _m.Decorate(variable, Decoration.DescriptorSet, 0);
        _m.Decorate(variable, Decoration.Binding, (uint)_bindingNames.Count);
        _m.Decorate(variable, Decoration.Aliased);                       // a dispatch may pass one storage twice (in place)
        if (reread)
        {
            _m.Decorate(variable, Decoration.Volatile);
            _m.Decorate(variable, Decoration.Coherent);
        }

        _m.Name(variable, name);
        _bindingNames.Add(name);
        _bindingVariables.Add(variable);
        return new Buf(this, variable, _bindingNames.Count - 1);
    }

    // Records that the kernel stores into binding `binding` (the runtime orders dispatches by what they write).
    internal void Written(int binding) => _writes |= 1UL << binding;

    /// <summary>The next push constant, a signed 32-bit integer.</summary>
    public Val PushInt(string name) => Push(name, ScalarKind.Int);

    /// <summary>The next push constant, an unsigned 32-bit integer.</summary>
    public Val PushUInt(string name) => Push(name, ScalarKind.UInt);

    /// <summary>The next push constant, a 32-bit float.</summary>
    public Val PushFloat(string name) => Push(name, ScalarKind.Float);

    private Val Push(string name, ScalarKind kind)
    {
        if (_push.Count == 32)
        {
            throw new InvalidOperationException($"{_name}: more than 128 bytes of push constants.");
        }

        uint member = _m.ConstantInt(_push.Count);
        _push.Add((name, kind));
        uint pointer = _m.PrologueValue(SpirvOp.AccessChain, _m.TypePointer(StorageClass.PushConstant, TypeOf(kind)), _pushVariable, member);
        return new Val(this, _m.PrologueValue(SpirvOp.Load, TypeOf(kind), pointer), kind);
    }

    /// <summary>A workgroup-shared float array.</summary>
    public SharedArray Shared(string name, int length)
    {
        uint type = _m.TypeArray(TypeOf(ScalarKind.Float), (uint)length);
        uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Workgroup, type), StorageClass.Workgroup);
        _m.Name(variable, name);
        _sharedBytes += 4 * length;
        return new SharedArray(this, variable, length);
    }

    /// <summary>A function-local variable (for values that live across Ifs and loop iterations).</summary>
    public Var Local(ScalarKind kind) => new(this, _m.LocalVariable(_m.TypePointer(StorageClass.Function, TypeOf(kind))), kind);

    /// <summary>A local variable starting at <paramref name="initial"/>.</summary>
    public Var Local(Val initial)
    {
        var v = Local(initial.Kind);
        v.V = initial;
        return v;
    }

    /// <summary>A float local variable starting at <paramref name="initial"/>.</summary>
    public Var Local(float initial) => Local(Float(initial));

    /// <summary>An int local variable starting at <paramref name="initial"/>.</summary>
    public Var Local(int initial) => Local(Int(initial));

    // ------------------------------------------------------------------ built-in indices (signed ints)

    /// <summary>gl_GlobalInvocationID.x.</summary>
    public Val GlobalX => BuiltInValue(BuiltIn.GlobalInvocationId, 0);

    /// <summary>gl_GlobalInvocationID.y.</summary>
    public Val GlobalY => BuiltInValue(BuiltIn.GlobalInvocationId, 1);

    /// <summary>gl_LocalInvocationID.x.</summary>
    public Val LocalX => BuiltInValue(BuiltIn.LocalInvocationId, 0);

    /// <summary>gl_LocalInvocationID.y.</summary>
    public Val LocalY => BuiltInValue(BuiltIn.LocalInvocationId, 1);

    /// <summary>gl_WorkGroupID.x.</summary>
    public Val GroupX => BuiltInValue(BuiltIn.WorkgroupId, 0);

    /// <summary>gl_WorkGroupID.y.</summary>
    public Val GroupY => BuiltInValue(BuiltIn.WorkgroupId, 1);

    /// <summary>gl_WorkGroupID.z.</summary>
    public Val GroupZ => BuiltInValue(BuiltIn.WorkgroupId, 2);

    /// <summary>gl_NumWorkGroups.x.</summary>
    public Val GroupsX => BuiltInValue(BuiltIn.NumWorkgroups, 0);

    /// <summary>gl_NumWorkGroups.y.</summary>
    public Val GroupsY => BuiltInValue(BuiltIn.NumWorkgroups, 1);

    /// <summary>gl_NumWorkGroups.z.</summary>
    public Val GroupsZ => BuiltInValue(BuiltIn.NumWorkgroups, 2);

    /// <summary>Invocations in the whole dispatch along x (the step of a grid-stride loop).</summary>
    public Val GridStrideX => GroupsX * _localX;

    /// <summary>gl_SubgroupSize (subgroup kernels only).</summary>
    public Val SubgroupSize => ScalarBuiltIn(BuiltIn.SubgroupSize);

    /// <summary>gl_NumSubgroups (subgroup kernels only).</summary>
    public Val NumSubgroups => ScalarBuiltIn(BuiltIn.NumSubgroups);

    /// <summary>gl_SubgroupID (subgroup kernels only).</summary>
    public Val SubgroupId => ScalarBuiltIn(BuiltIn.SubgroupId);

    /// <summary>gl_SubgroupInvocationID (subgroup kernels only).</summary>
    public Val SubgroupLocalId => ScalarBuiltIn(BuiltIn.SubgroupLocalInvocationId);

    // A scalar uint built-in (the subgroup ones), loaded once at the start, as a signed int.
    private Val ScalarBuiltIn(BuiltIn builtIn)
    {
        if (_builtInValues.TryGetValue((builtIn, 0), out var cached))
        {
            return cached;
        }

        _m.Capability(61);                                               // GroupNonUniform
        uint uintType = TypeOf(ScalarKind.UInt);
        uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Input, uintType), StorageClass.Input);
        _m.Decorate(variable, Decoration.BuiltIn, (uint)builtIn);
        uint loaded = _m.PrologueValue(SpirvOp.Load, uintType, variable);
        var value = new Val(this, _m.PrologueValue(SpirvOp.Bitcast, TypeOf(ScalarKind.Int), loaded), ScalarKind.Int);
        _builtInValues[(builtIn, 0)] = value;
        return value;
    }

    /// <summary>The sum (or maximum) of a float over the invocations of this subgroup, returned to all of them (subgroup
    /// kernels only; every invocation of the subgroup must take part).</summary>
    public Val SubgroupReduce(Val value, bool max)
    {
        Expect(value, ScalarKind.Float);
        _m.Capability(61);                                               // GroupNonUniform
        _m.Capability(63);                                               // GroupNonUniformArithmetic
        uint scope = _m.ConstantUInt(3);                                 // Subgroup
        return new Val(this, _m.Value(max ? SpirvOp.GroupNonUniformFMax : SpirvOp.GroupNonUniformFAdd, TypeOf(ScalarKind.Float), scope, 0, value.Id), ScalarKind.Float);
    }

    private Val BuiltInValue(BuiltIn builtIn, int component)
    {
        if (_builtInValues.TryGetValue((builtIn, component), out var cached))
        {
            return cached;
        }

        uint uint3 = _m.TypeVector(TypeOf(ScalarKind.UInt), 3);
        if (!_builtIns.TryGetValue(builtIn, out uint loaded))
        {
            uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Input, uint3), StorageClass.Input);
            _m.Decorate(variable, Decoration.BuiltIn, (uint)builtIn);
            loaded = _m.PrologueValue(SpirvOp.Load, uint3, variable);
            _builtIns[builtIn] = loaded;
        }

        uint part = _m.PrologueValue(SpirvOp.CompositeExtract, TypeOf(ScalarKind.UInt), loaded, (uint)component);
        var value = new Val(this, _m.PrologueValue(SpirvOp.Bitcast, TypeOf(ScalarKind.Int), part), ScalarKind.Int);
        _builtInValues[(builtIn, component)] = value;
        return value;
    }

    // ------------------------------------------------------------------ constants

    /// <summary>A signed integer constant.</summary>
    public Val Int(int value) => new(this, _m.ConstantInt(value), ScalarKind.Int);

    /// <summary>An unsigned integer constant.</summary>
    public Val UInt(uint value) => new(this, _m.ConstantUInt(value), ScalarKind.UInt);

    /// <summary>A float constant.</summary>
    public Val Float(float value) => new(this, _m.ConstantFloat(value), ScalarKind.Float);

    /// <summary>A boolean constant.</summary>
    public Val Bool(bool value) => new(this, _m.ConstantBool(value), ScalarKind.Bool);

    // ------------------------------------------------------------------ math

    /// <summary>e^x.</summary>
    public Val Exp(Val x) => Ext(Glsl.Exp, x);

    /// <summary>Natural logarithm.</summary>
    public Val Log(Val x) => Ext(Glsl.Log, x);

    /// <summary>Base-2 logarithm.</summary>
    public Val Log2(Val x) => Ext(Glsl.Log2, x);

    /// <summary>tanh as the driver computes it (GLSL.std.450; may lose accuracy or overflow for large |x| on some drivers).</summary>
    public Val Tanh(Val x) => Ext(Glsl.Tanh, x);

    /// <summary>Square root.</summary>
    public Val Sqrt(Val x) => Ext(Glsl.Sqrt, x);

    /// <summary>1 / sqrt(x).</summary>
    public Val InverseSqrt(Val x) => Ext(Glsl.InverseSqrt, x);

    /// <summary>x^y.</summary>
    public Val Pow(Val x, Val y) => Ext(Glsl.Pow, x, y);

    /// <summary>Largest integer value not above x.</summary>
    public Val Floor(Val x) => Ext(Glsl.Floor, x);

    /// <summary>x rounded to the nearest integer value, ties to even.</summary>
    public Val RoundEven(Val x) => Ext(Glsl.RoundEven, x);

    /// <summary>a · b + c (fused where the device fuses).</summary>
    public Val Fma(Val a, Val b, Val c) => Ext(Glsl.Fma, a, b, c);

    /// <summary>|x| (float or signed int).</summary>
    public Val Abs(Val x) => Ext(x.Kind == ScalarKind.Float ? Glsl.FAbs : Glsl.SAbs, x);

    /// <summary>The larger of two values of the same type.</summary>
    public Val Max(Val a, Val b) => Ext(a.Kind switch { ScalarKind.Float => Glsl.FMax, ScalarKind.UInt => Glsl.UMax, _ => Glsl.SMax }, a, b);

    /// <summary>The smaller of two values of the same type.</summary>
    public Val Min(Val a, Val b) => Ext(a.Kind switch { ScalarKind.Float => Glsl.FMin, ScalarKind.UInt => Glsl.UMin, _ => Glsl.SMin }, a, b);

    /// <summary>x limited to [lo, hi].</summary>
    public Val Clamp(Val x, Val lo, Val hi) => Ext(x.Kind == ScalarKind.Float ? Glsl.FClamp : Glsl.SClamp, x, lo, hi);

    /// <summary>condition ? a : b (both evaluated).</summary>
    public Val Select(Val condition, Val a, Val b)
    {
        Same(a, b);
        return new(this, _m.Value(SpirvOp.Select, TypeOf(a.Kind), condition.Id, a.Id, b.Id), a.Kind);
    }

    private Val Ext(Glsl op, params Val[] args)
    {
        foreach (var a in args)
        {
            Same(args[0], a);
        }

        return new(this, _m.Ext(op, TypeOf(args[0].Kind), [.. args.Select(a => a.Id)]), args[0].Kind);
    }

    // ------------------------------------------------------------------ control flow

    /// <summary>if (condition) then(); else otherwise();</summary>
    public void If(Val condition, Action then, Action? otherwise = null)
    {
        Expect(condition, ScalarKind.Bool);
        uint thenLabel = _m.Id(), merge = _m.Id(), elseLabel = otherwise is null ? merge : _m.Id();
        _m.Code(SpirvOp.SelectionMerge, merge, 0);
        _m.Terminate(SpirvOp.BranchConditional, condition.Id, thenLabel, elseLabel);
        _m.Label(thenLabel);
        then();
        _m.Terminate(SpirvOp.Branch, merge);
        if (otherwise is not null)
        {
            _m.Label(elseLabel);
            otherwise();
            _m.Terminate(SpirvOp.Branch, merge);
        }

        _m.Label(merge);
    }

    /// <summary>for (int i = start; i &lt; end; i += step) body(i); with step 1 when not given.</summary>
    public void For(Val start, Val end, Action<Val> body, Val? step = null)
    {
        Expect(start, ScalarKind.Int);
        var i = Local(start);
        Val by = step ?? Int(1);
        Loop(() => i.V < end, () => body(i.V), () => i.V = i.V + by);
    }

    /// <summary>for (int i = start; i &lt; end; i += step) body(i); with a constant step.</summary>
    public void For(Val start, Val end, int step, Action<Val> body) => For(start, end, body, Int(step));

    /// <summary>while (condition()) body();</summary>
    public void While(Func<Val> condition, Action body) => Loop(condition, body, static () => { });

    // header: merge declaration → condition block: exit or body → body → continue block: next → header.
    private void Loop(Func<Val> condition, Action body, Action next)
    {
        uint header = _m.Id(), test = _m.Id(), bodyLabel = _m.Id(), continueLabel = _m.Id(), merge = _m.Id();
        _m.Terminate(SpirvOp.Branch, header);
        _m.Label(header);
        _m.Code(SpirvOp.LoopMerge, merge, continueLabel, 0);
        _m.Terminate(SpirvOp.Branch, test);
        _m.Label(test);
        var c = condition();
        Expect(c, ScalarKind.Bool);
        _m.Terminate(SpirvOp.BranchConditional, c.Id, bodyLabel, merge);
        _m.Label(bodyLabel);
        body();
        _m.Terminate(SpirvOp.Branch, continueLabel);
        _m.Label(continueLabel);
        next();
        _m.Terminate(SpirvOp.Branch, header);
        _m.Label(merge);
    }

    /// <summary>
    /// barrier(): every invocation of the workgroup waits here, and their workgroup-memory writes before it are visible
    /// after it. Only in control flow that is uniform across the workgroup.
    /// </summary>
    public void Barrier() =>
        _m.Code(SpirvOp.ControlBarrier, _m.ConstantUInt(2), _m.ConstantUInt(2), _m.ConstantUInt(0x108));   // Workgroup, AcquireRelease | WorkgroupMemory

    /// <summary>
    /// <see cref="Barrier"/> that also orders storage-buffer memory: the workgroup's buffer writes before it are visible to
    /// its invocations after it (a buffer written by some invocations, then read or written by others). Uniform control flow only.
    /// </summary>
    public void BufferBarrier() =>
        _m.Code(SpirvOp.ControlBarrier, _m.ConstantUInt(2), _m.ConstantUInt(2), _m.ConstantUInt(0x148));   // + UniformMemory

    /// <summary>
    /// The sum of <paramref name="value"/> over the workgroup's x invocations (a power of two), returned to all of them
    /// through <paramref name="scratch"/> (at least the workgroup width long). Uniform control flow only.
    /// </summary>
    public Val ReduceSum(SharedArray scratch, Val value) => Reduce(scratch, value, (a, b) => a + b, max: false);

    /// <summary>The largest <paramref name="value"/> over the workgroup's x invocations (see <see cref="ReduceSum"/>).</summary>
    public Val ReduceMax(SharedArray scratch, Val value) => Reduce(scratch, value, Max, max: true);

    // Without subgroups: a tree through workgroup memory (log2 width rounds, a barrier each). With them: each subgroup
    // reduces its values, its first invocation stores the result in scratch[subgroup], and after a barrier every
    // subgroup reduces those (a subgroup's width of them at a time, in order: the same result in every invocation),
    // whatever the subgroup size; a barrier then frees the scratch. Two barriers instead of log2 width + 2.
    private Val Reduce(SharedArray scratch, Val value, Func<Val, Val, Val> combine, bool max)
    {
        if (_localY * _localZ != 1 || (_localX & (_localX - 1)) != 0 || scratch.Length < _localX)
        {
            throw new InvalidOperationException($"{_name}: reductions need a 1-D power-of-two workgroup and scratch as wide.");
        }

        if (_subgroups)
        {
            var mine = SubgroupReduce(value, max);
            If(SubgroupLocalId.Eq(0), () => scratch[SubgroupId] = mine);
            Barrier();
            var identity = Float(max ? float.NegativeInfinity : 0f);
            var acc = Local(identity);
            For(Int(0), NumSubgroups, start =>
            {
                var at = start + SubgroupLocalId;
                var part = Select(at < NumSubgroups, scratch[Min(at, Int(_localX - 1))], identity);
                acc.V = combine(acc.V, SubgroupReduce(part, max));
            }, SubgroupSize);
            var result = acc.V;
            Barrier();                                                   // everyone has read the scratch before it is reused
            return result;
        }

        var lane = LocalX;
        scratch[lane] = value;
        Barrier();
        for (int stride = _localX / 2; stride > 0; stride /= 2)
        {
            int s = stride;
            If(lane < s, () => scratch[lane] = combine(scratch[lane], scratch[lane + s]));
            Barrier();
        }

        var total = scratch[Int(0)];
        Barrier();                                                       // everyone has read it before the scratch is reused
        return total;
    }

    // ------------------------------------------------------------------ the module

    /// <summary>The finished kernel.</summary>
    public SpirvKernel Build()
    {
        if (_push.Count > 0)
        {
            uint[] members = [.. _push.Select(p => TypeOf(p.Kind))];
            _m.TypeStructAs(_pushStruct, members);
            _m.Decorate(_pushStruct, Decoration.Block);
            for (int i = 0; i < _push.Count; i++)
            {
                _m.MemberDecorate(_pushStruct, (uint)i, Decoration.Offset, (uint)(4 * i));
                _m.MemberName(_pushStruct, (uint)i, _push[i].Name);
            }

            _m.Name(_pushStruct, "Push");
            _m.GlobalVariableAs(_pushVariable, _m.TypePointer(StorageClass.PushConstant, _pushStruct), StorageClass.PushConstant);
        }

        // Bindings the kernel only reads are NonWritable (drivers may read them through faster caches).
        for (int i = 0; i < _bindingVariables.Count; i++)
        {
            if ((_writes & (1UL << i)) == 0)
            {
                _m.Decorate(_bindingVariables[i], Decoration.NonWritable);
            }
        }

        var words = _m.Finish((uint)_localX, (uint)_localY, (uint)_localZ);
        return new SpirvKernel(_name, words, _bindingNames.Count, 4 * _push.Count, _localX, _localY, _localZ,
            [.. _bindingNames], [.. _push.Select(p => $"{p.Name}:{p.Kind.ToString().ToLowerInvariant()}")], _writes, _sharedBytes);
    }

    // ------------------------------------------------------------------ helpers for Val

    internal uint TypeOf(ScalarKind kind) => kind switch
    {
        ScalarKind.Bool => _m.TypeBool(),
        ScalarKind.Int => _m.TypeInt(true),
        ScalarKind.UInt => _m.TypeInt(false),
        _ => _m.TypeFloat(),
    };

    internal Val Op(SpirvOp op, ScalarKind result, params Val[] operands) =>
        new(this, _m.Value(op, TypeOf(result), [.. operands.Select(o => o.Id)]), result);

    internal static void Same(Val a, Val b)
    {
        if (a.Kind != b.Kind)
        {
            throw new InvalidOperationException($"SPIR-V kernel: {a.Kind} combined with {b.Kind} (convert one first).");
        }
    }

    internal static void Expect(Val v, ScalarKind kind)
    {
        if (v.Kind != kind)
        {
            throw new InvalidOperationException($"SPIR-V kernel: expected {kind}, got {v.Kind}.");
        }
    }

    internal uint BufferElement(uint variable, Val index)
    {
        if (index.Kind is not (ScalarKind.Int or ScalarKind.UInt))
        {
            throw new InvalidOperationException("SPIR-V kernel: buffer index must be an integer.");
        }

        return _m.Value(SpirvOp.AccessChain, _bufferElement, variable, _m.ConstantInt(0), index.Id);
    }

    internal uint SharedElement(uint variable, Val index) =>
        _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.Workgroup, TypeOf(ScalarKind.Float)), variable, index.Id);
}

/// <summary>A 32-bit value computed by a kernel (an SSA id with its type), with C# operators that emit SPIR-V.</summary>
internal readonly struct Val
{
    internal Val(KernelBuilder builder, uint id, ScalarKind kind)
    {
        Builder = builder;
        Id = id;
        Kind = kind;
    }

    /// <summary>The kernel this value belongs to.</summary>
    public KernelBuilder Builder { get; }

    /// <summary>The SPIR-V result id.</summary>
    public uint Id { get; }

    /// <summary>The value's type.</summary>
    public ScalarKind Kind { get; }

    // A literal of this value's type (an int literal next to a float value is a float).
    private Val Literal(int v) => Kind switch
    {
        ScalarKind.Float => Builder.Float(v),
        ScalarKind.UInt => Builder.UInt((uint)v),
        _ => Builder.Int(v),
    };

    private Val Literal(float v)
    {
        KernelBuilder.Expect(this, ScalarKind.Float);
        return Builder.Float(v);
    }

    private static Val Arith(Val a, Val b, SpirvOp signed, SpirvOp unsigned, SpirvOp floating)
    {
        KernelBuilder.Same(a, b);
        var op = a.Kind switch
        {
            ScalarKind.Float => floating,
            ScalarKind.UInt => unsigned,
            ScalarKind.Int => signed,
            _ => throw new InvalidOperationException("SPIR-V kernel: arithmetic on a boolean."),
        };
        return a.Builder.Op(op, a.Kind, a, b);
    }

    private static Val Compare(Val a, Val b, SpirvOp signed, SpirvOp unsigned, SpirvOp floating)
    {
        KernelBuilder.Same(a, b);
        var op = a.Kind switch
        {
            ScalarKind.Float => floating,
            ScalarKind.UInt => unsigned,
            ScalarKind.Int => signed,
            _ => throw new InvalidOperationException("SPIR-V kernel: ordering booleans."),
        };
        return a.Builder.Op(op, ScalarKind.Bool, a, b);
    }

    public static Val operator +(Val a, Val b) => Arith(a, b, SpirvOp.IAdd, SpirvOp.IAdd, SpirvOp.FAdd);

    public static Val operator -(Val a, Val b) => Arith(a, b, SpirvOp.ISub, SpirvOp.ISub, SpirvOp.FSub);

    public static Val operator *(Val a, Val b) => Arith(a, b, SpirvOp.IMul, SpirvOp.IMul, SpirvOp.FMul);

    public static Val operator /(Val a, Val b) => Arith(a, b, SpirvOp.SDiv, SpirvOp.UDiv, SpirvOp.FDiv);

    /// <summary>Remainder with the sign of the dividend (as in C#).</summary>
    public static Val operator %(Val a, Val b) => Arith(a, b, SpirvOp.SRem, SpirvOp.UMod, SpirvOp.FRem);

    public static Val operator +(Val a, int b) => a + a.Literal(b);

    public static Val operator +(int a, Val b) => b.Literal(a) + b;

    public static Val operator -(Val a, int b) => a - a.Literal(b);

    public static Val operator -(int a, Val b) => b.Literal(a) - b;

    public static Val operator *(Val a, int b) => a * a.Literal(b);

    public static Val operator *(int a, Val b) => b.Literal(a) * b;

    public static Val operator /(Val a, int b) => a / a.Literal(b);

    public static Val operator /(int a, Val b) => b.Literal(a) / b;

    public static Val operator %(Val a, int b) => a % a.Literal(b);

    public static Val operator +(Val a, float b) => a + a.Literal(b);

    public static Val operator +(float a, Val b) => b.Literal(a) + b;

    public static Val operator -(Val a, float b) => a - a.Literal(b);

    public static Val operator -(float a, Val b) => b.Literal(a) - b;

    public static Val operator *(Val a, float b) => a * a.Literal(b);

    public static Val operator *(float a, Val b) => b.Literal(a) * b;

    public static Val operator /(Val a, float b) => a / a.Literal(b);

    public static Val operator /(float a, Val b) => b.Literal(a) / b;

    public static Val operator -(Val a) => a.Kind == ScalarKind.Float
        ? a.Builder.Op(SpirvOp.FNegate, a.Kind, a)
        : a.Builder.Op(SpirvOp.SNegate, a.Kind, a);

    public static Val operator <(Val a, Val b) => Compare(a, b, SpirvOp.SLessThan, SpirvOp.ULessThan, SpirvOp.FOrdLessThan);

    public static Val operator >(Val a, Val b) => Compare(a, b, SpirvOp.SGreaterThan, SpirvOp.UGreaterThan, SpirvOp.FOrdGreaterThan);

    public static Val operator <=(Val a, Val b) => Compare(a, b, SpirvOp.SLessThanEqual, SpirvOp.ULessThanEqual, SpirvOp.FOrdLessThanEqual);

    public static Val operator >=(Val a, Val b) => Compare(a, b, SpirvOp.SGreaterThanEqual, SpirvOp.UGreaterThanEqual, SpirvOp.FOrdGreaterThanEqual);

    public static Val operator <(Val a, int b) => a < a.Literal(b);

    public static Val operator >(Val a, int b) => a > a.Literal(b);

    public static Val operator <=(Val a, int b) => a <= a.Literal(b);

    public static Val operator >=(Val a, int b) => a >= a.Literal(b);

    public static Val operator <(Val a, float b) => a < a.Literal(b);

    public static Val operator >(Val a, float b) => a > a.Literal(b);

    public static Val operator <=(Val a, float b) => a <= a.Literal(b);

    public static Val operator >=(Val a, float b) => a >= a.Literal(b);

    /// <summary>Bitwise and of integers, logical and of booleans.</summary>
    public static Val operator &(Val a, Val b) => a.Kind == ScalarKind.Bool
        ? a.Builder.Op(SpirvOp.LogicalAnd, ScalarKind.Bool, a, b)
        : Arith(a, b, SpirvOp.BitwiseAnd, SpirvOp.BitwiseAnd, SpirvOp.BitwiseAnd);

    /// <summary>Bitwise or of integers, logical or of booleans.</summary>
    public static Val operator |(Val a, Val b) => a.Kind == ScalarKind.Bool
        ? a.Builder.Op(SpirvOp.LogicalOr, ScalarKind.Bool, a, b)
        : Arith(a, b, SpirvOp.BitwiseOr, SpirvOp.BitwiseOr, SpirvOp.BitwiseOr);

    public static Val operator ^(Val a, Val b) => Arith(a, b, SpirvOp.BitwiseXor, SpirvOp.BitwiseXor, SpirvOp.BitwiseXor);

    public static Val operator &(Val a, int b) => a & a.Literal(b);

    public static Val operator |(Val a, int b) => a | a.Literal(b);

    public static Val operator ^(Val a, int b) => a ^ a.Literal(b);

    /// <summary>Logical not of a boolean, bitwise not of an integer.</summary>
    public static Val operator !(Val a) => a.Builder.Op(a.Kind == ScalarKind.Bool ? SpirvOp.LogicalNot : SpirvOp.Not, a.Kind, a);

    public static Val operator <<(Val a, int bits) => a.Builder.Op(SpirvOp.ShiftLeftLogical, a.Kind, a, a.Builder.Int(bits));

    /// <summary>Arithmetic shift for signed values, logical for unsigned.</summary>
    public static Val operator >>(Val a, int bits) =>
        a.Builder.Op(a.Kind == ScalarKind.UInt ? SpirvOp.ShiftRightLogical : SpirvOp.ShiftRightArithmetic, a.Kind, a, a.Builder.Int(bits));

    /// <summary>Shift left by a computed amount.</summary>
    public Val ShiftLeft(Val bits) => Builder.Op(SpirvOp.ShiftLeftLogical, Kind, this, bits);

    /// <summary>Shift right (arithmetic for signed values) by a computed amount.</summary>
    public Val ShiftRight(Val bits) => Builder.Op(Kind == ScalarKind.UInt ? SpirvOp.ShiftRightLogical : SpirvOp.ShiftRightArithmetic, Kind, this, bits);

    /// <summary>this == other.</summary>
    public Val Eq(Val other)
    {
        KernelBuilder.Same(this, other);
        var op = Kind switch { ScalarKind.Float => SpirvOp.FOrdEqual, ScalarKind.Bool => SpirvOp.LogicalEqual, _ => SpirvOp.IEqual };
        return Builder.Op(op, ScalarKind.Bool, this, other);
    }

    /// <summary>this == literal.</summary>
    public Val Eq(int other) => Eq(Literal(other));

    /// <summary>this != other.</summary>
    public Val Ne(Val other)
    {
        KernelBuilder.Same(this, other);
        var op = Kind switch { ScalarKind.Float => SpirvOp.FOrdNotEqual, ScalarKind.Bool => SpirvOp.LogicalNotEqual, _ => SpirvOp.INotEqual };
        return Builder.Op(op, ScalarKind.Bool, this, other);
    }

    /// <summary>this != literal.</summary>
    public Val Ne(int other) => Ne(Literal(other));

    /// <summary>The value converted to float (from a signed or unsigned integer).</summary>
    public Val ToFloat() => Kind switch
    {
        ScalarKind.Float => this,
        ScalarKind.UInt => Builder.Op(SpirvOp.ConvertUToF, ScalarKind.Float, this),
        _ => Builder.Op(SpirvOp.ConvertSToF, ScalarKind.Float, this),
    };

    /// <summary>The value converted to a signed integer (floats truncate toward zero; unsigned values keep their bits).</summary>
    public Val ToInt() => Kind switch
    {
        ScalarKind.Int => this,
        ScalarKind.Float => Builder.Op(SpirvOp.ConvertFToS, ScalarKind.Int, this),
        _ => Builder.Op(SpirvOp.Bitcast, ScalarKind.Int, this),
    };

    /// <summary>The value converted to an unsigned integer (floats truncate; signed values keep their bits).</summary>
    public Val ToUInt() => Kind switch
    {
        ScalarKind.UInt => this,
        ScalarKind.Float => Builder.Op(SpirvOp.ConvertFToU, ScalarKind.UInt, this),
        _ => Builder.Op(SpirvOp.Bitcast, ScalarKind.UInt, this),
    };

    /// <summary>The same 32 bits read as a float.</summary>
    public Val AsFloat() => Kind == ScalarKind.Float ? this : Builder.Op(SpirvOp.Bitcast, ScalarKind.Float, this);

    /// <summary>The same 32 bits read as a signed integer.</summary>
    public Val AsInt() => Kind == ScalarKind.Int ? this : Builder.Op(SpirvOp.Bitcast, ScalarKind.Int, this);

    /// <summary>The same 32 bits read as an unsigned integer.</summary>
    public Val AsUInt() => Kind == ScalarKind.UInt ? this : Builder.Op(SpirvOp.Bitcast, ScalarKind.UInt, this);

    /// <summary>Whether a float is NaN.</summary>
    public Val IsNan() => Builder.Op(SpirvOp.IsNan, ScalarKind.Bool, this);
}

/// <summary>A function-local variable: <c>v.V</c> loads it, <c>v.V = x</c> stores.</summary>
internal sealed class Var(KernelBuilder builder, uint pointer, ScalarKind kind)
{
    /// <summary>The variable's type.</summary>
    public ScalarKind Kind => kind;

    /// <summary>The current value (a load), or a new one (a store).</summary>
    public Val V
    {
        get => new(builder, builder.Module.Value(SpirvOp.Load, builder.TypeOf(kind), pointer), kind);
        set
        {
            KernelBuilder.Expect(value, kind);
            builder.Module.Code(SpirvOp.Store, pointer, value.Id);
        }
    }
}

/// <summary>A storage buffer of 32-bit words: <c>b[i]</c> reads or writes word i as a float; <see cref="Int"/> and
/// <see cref="UInt"/> read its bits as integers.</summary>
internal sealed class Buf(KernelBuilder builder, uint variable, int binding)
{
    /// <summary>Word <paramref name="index"/> as a float.</summary>
    public Val this[Val index]
    {
        get => new(builder, builder.Module.Value(SpirvOp.Load, builder.TypeOf(ScalarKind.Float), builder.BufferElement(variable, index)), ScalarKind.Float);
        set
        {
            builder.Written(binding);
            builder.Module.Code(SpirvOp.Store, builder.BufferElement(variable, index), value.AsFloat().Id);
        }
    }

    /// <summary>Word <paramref name="index"/>'s bits as a signed integer.</summary>
    public Val Int(Val index) => this[index].AsInt();

    /// <summary>Word <paramref name="index"/>'s bits as an unsigned integer.</summary>
    public Val UInt(Val index) => this[index].AsUInt();
}

/// <summary>A workgroup-shared float array.</summary>
internal sealed class SharedArray(KernelBuilder builder, uint variable, int length)
{
    /// <summary>Elements.</summary>
    public int Length => length;

    /// <summary>Element <paramref name="index"/>.</summary>
    public Val this[Val index]
    {
        get => new(builder, builder.Module.Value(SpirvOp.Load, builder.TypeOf(ScalarKind.Float), builder.SharedElement(variable, index)), ScalarKind.Float);
        set
        {
            KernelBuilder.Expect(value, ScalarKind.Float);
            builder.Module.Code(SpirvOp.Store, builder.SharedElement(variable, index), value.Id);
        }
    }
}
