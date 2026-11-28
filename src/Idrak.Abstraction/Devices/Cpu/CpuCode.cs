// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection;
using System.Reflection.Emit;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>
/// The code a measured choice chooses between, as a fingerprint of 16 hex digits: the IL every one of its candidates can
/// run, so that a kept choice (CpuChoices) is read only by a build whose candidates are the code they were when it was
/// measured, and a build that changed them measures again, as a GPU backend's kept choices name a hash of its generated
/// kernels. From the root types (a choice's candidates' runs, closed over the kernels they run): every method of each,
/// then every method a walked method calls or takes the address of, and every method of each type it names as a generic
/// argument, a local or a type operand (how a kernel reaches its instruction set and its plan's steps: type parameters),
/// within the roots' namespaces; a static field read brings its type's initializer. Each method is hashed as its
/// attributes, locals and instructions, every token written as the name of what it refers to and the compiler's numbering
/// of lambdas and closures left out, so a change elsewhere in the assembly (which moves tokens and that numbering) leaves
/// the fingerprint as it was. Made once a process for each set of roots, and kept in the choices file for the modules'
/// builds, so a process of the same build walks nothing; where the IL cannot be read (code compiled ahead of time), the
/// fingerprint is the modules' builds alone.
/// </summary>
internal static partial class CpuCode
{
    // A walk's fingerprint for each set of roots, made in this process.
    private static readonly ConcurrentDictionary<string, string> s_made = new(StringComparer.Ordinal);

    /// <summary>The fingerprint of the code reachable from <paramref name="roots"/> (16 hex digits).</summary>
    public static string Of(params Type[] roots)
    {
        var names = new StringBuilder();
        foreach (var root in roots)
        {
            names.Append(root).Append(';');                    // a closed generic with its arguments
        }

        string name = names.ToString();
        return s_made.TryGetValue(name, out var made) ? made : s_made.GetOrAdd(name, Make(name, roots));
    }

    /// <summary>The methods the walk from <paramref name="roots"/> hashes, by name (tests).</summary>
    [UnconditionalSuppressMessage("Trimming", "IL2026", Justification = "A fingerprint of the code that runs: a trimmed assembly's IL is the code it runs.")]
    internal static IReadOnlyList<string> Walked(params Type[] roots) => [.. new Walker(roots).Run(out _).Select(t => t.Key)];

    // The fingerprint for these roots: kept for the modules' builds, else walked (and kept), else the builds alone.
    [UnconditionalSuppressMessage("Trimming", "IL2026", Justification = "A fingerprint of the code that runs: a trimmed assembly's IL is the code it runs, and where none can be read the modules' builds serve.")]
    private static string Make(string name, Type[] roots)
    {
        string builds = Builds(roots), group = "code " + Hex(name), build = "b" + Hex(builds);
        if (CpuChoices.TryGet(group, build, out var kept))
        {
            return kept;
        }

        string fingerprint;
        try
        {
            fingerprint = Walk(roots) ?? Hex(builds);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or ArgumentException or MemberAccessException or TypeLoadException
            or BadImageFormatException or IOException)
        {
            fingerprint = Hex(builds);
        }

        CpuChoices.Keep(group, build, fingerprint);
        return fingerprint;
    }

    // The builds (module version ids) of the roots' modules and their generic arguments', in order.
    private static string Builds(Type[] roots)
    {
        var modules = new SortedSet<string>(StringComparer.Ordinal);
        void Add(Type t)
        {
            if (t.HasElementType)
            {
                Add(t.GetElementType()!);
                return;
            }

            string build;
            try
            {
                build = t.Module.ModuleVersionId.ToString("N");
            }
            catch (Exception e) when (e is NotSupportedException or InvalidOperationException)
            {
                build = t.Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? t.Assembly.FullName ?? "";
            }

            modules.Add(t.Assembly.GetName().Name + " " + build);
            if (t.IsGenericType && !t.IsGenericTypeDefinition)
            {
                foreach (var argument in t.GetGenericArguments())
                {
                    Add(argument);
                }
            }
        }

        foreach (var root in roots)
        {
            Add(root);
        }

        return string.Join(";", modules);
    }

    private static string Hex(string text) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)))[..16];

    // The walk's fingerprint, or null where a method that has IL gave none (code compiled ahead of time).
    [RequiresUnreferencedCode("Reads the walked methods' IL and resolves their tokens.")]
    private static string? Walk(Type[] roots)
    {
        var texts = new Walker(roots).Run(out bool complete);
        if (!complete)
        {
            return null;
        }

        var all = new StringBuilder();
        foreach (var (key, text) in texts)
        {
            all.Append(key).Append('\n').Append(text).Append('\n');
        }

        return Hex(all.ToString());
    }

    // The compiler's numbering in the names it makes (a lambda's, a closure's, a local function's, a cached delegate's, an
    // iterator's): the method's ordinal in its type, which a method added anywhere before it moves.
    [GeneratedRegex(@"(__DisplayClass|b__|<>9__|\||d__)\d+_?")]
    private static partial Regex Numbering();

    private static string Named(string text) => Numbering().Replace(text, "$1");

    // The IL's opcodes by their value (one byte, or 0xFE and a second).
    private static readonly OpCode[] s_one = new OpCode[256], s_two = new OpCode[256];

    static CpuCode()
    {
        foreach (var field in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (field.GetValue(null) is OpCode op)
            {
                ushort value = (ushort)op.Value;
                (op.Size == 1 ? s_one : s_two)[value & 0xFF] = op;
            }
        }
    }

    // One walk: the methods reached and each one's text.
    [RequiresUnreferencedCode("Reads the walked methods' IL and resolves their tokens.")]
    private sealed class Walker
    {
        private const BindingFlags Declared = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        private readonly HashSet<(Assembly, string?)> _scope = [];
        private readonly HashSet<MethodBase> _seen = [];
        private readonly HashSet<Type> _types = [];
        private readonly Queue<MethodBase> _queue = new();
        private readonly List<(string Key, string Text)> _texts = [];

        public Walker(Type[] roots)
        {
            foreach (var root in roots)
            {
                _scope.Add((root.Assembly, root.Namespace));
            }

            foreach (var root in roots)
            {
                AddType(root, root: true);
            }
        }

        // Every reached method's text by its name (and each walked type's fields), in order of the names and then the texts
        // (two lambdas of overloads share a name once their numbering is left out); `complete` false where a method with IL
        // gave none.
        public List<(string Key, string Text)> Run(out bool complete)
        {
            complete = true;
            while (_queue.TryDequeue(out var method))
            {
                var text = new StringBuilder();
                text.Append(CultureInfo.InvariantCulture, $"{method.Attributes} {method.MethodImplementationFlags};");
                var body = method.GetMethodBody();
                if (body?.GetILAsByteArray() is { } il)
                {
                    Body(method, body, il, text);
                }
                else if (!method.IsAbstract && (method.MethodImplementationFlags & (MethodImplAttributes.InternalCall | MethodImplAttributes.Runtime)) == 0
                    && (method.Attributes & MethodAttributes.PinvokeImpl) == 0)
                {
                    complete = false;
                }

                _texts.Add((Named($"{method.DeclaringType}::{method}"), Named(text.ToString())));
            }

            _texts.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key) is var order and not 0 ? order : string.CompareOrdinal(a.Text, b.Text));
            return _texts;
        }

        private bool InScope(Type t) => _scope.Contains((t.Assembly, t.Namespace));

        private void Enqueue(MethodBase method)
        {
            if (_seen.Add(method))
            {
                _queue.Enqueue(method);
            }
        }

        // A called method's own definition (the method of the generic type or method it instantiates), walked where it is
        // the roots' code.
        private void AddMethod(MethodBase method)
        {
            if (method.DeclaringType is not { HasElementType: false } owner || !InScope(owner) || method.MetadataToken >> 24 != 0x06)
            {
                return;
            }

            if (method.Module.ResolveMethod(method.MetadataToken) is { } definition)
            {
                Enqueue(definition);
            }
        }

        // A type named: its generic arguments, and (the roots' code) every method of its definition, its fields and layout.
        private void AddType(Type t, bool root = false)
        {
            if (t.IsGenericParameter)
            {
                return;
            }

            if (t.HasElementType)
            {
                AddType(t.GetElementType()!);
                return;
            }

            if (t.IsFunctionPointer)
            {
                AddType(t.GetFunctionPointerReturnType());
                foreach (var parameter in t.GetFunctionPointerParameterTypes())
                {
                    AddType(parameter);
                }

                return;
            }

            if (t.IsGenericType && !t.IsGenericTypeDefinition)
            {
                foreach (var argument in t.GetGenericArguments())
                {
                    AddType(argument);
                }

                t = t.GetGenericTypeDefinition();
            }

            if (!(root || InScope(t)) || !_types.Add(t))
            {
                return;
            }

            // Its fields and attributes (a layout or an inline array's length changes its code without its IL).
            var layout = new StringBuilder();
            foreach (var field in t.GetFields(Declared))
            {
                layout.Append(CultureInfo.InvariantCulture, $"{field.Attributes} {field.FieldType} {field.Name};");
            }

            foreach (var attribute in t.GetCustomAttributesData())
            {
                layout.Append(attribute).Append(';');
            }

            _texts.Add((Named($"{t} (type)"), Named(layout.ToString())));
            foreach (var method in t.GetMethods(Declared))
            {
                Enqueue(method);
            }

            foreach (var constructor in t.GetConstructors(Declared))
            {
                Enqueue(constructor);
            }

            if (t.TypeInitializer is { } initializer)
            {
                Enqueue(initializer);
            }
        }

        // A method's locals, handlers and instructions, each token as the name of what it refers to.
        private void Body(MethodBase method, MethodBody body, byte[] il, StringBuilder text)
        {
            text.Append(body.InitLocals ? "init;" : "skip;");
            foreach (var local in body.LocalVariables)
            {
                text.Append(local.LocalType).Append(local.IsPinned ? " pinned;" : ";");
                AddType(local.LocalType);
            }

            foreach (var clause in body.ExceptionHandlingClauses)
            {
                text.Append(CultureInfo.InvariantCulture, $"{clause.Flags} {clause.TryOffset} {clause.TryLength} {clause.HandlerOffset} {clause.HandlerLength} ");
                if (clause.Flags == ExceptionHandlingClauseOptions.Clause)
                {
                    text.Append(clause.CatchType);
                }

                text.Append(';');
            }

            Type[]? typeArguments = method.DeclaringType is { IsGenericType: true } owner ? owner.GetGenericArguments() : null;
            Type[]? methodArguments = method is MethodInfo { IsGenericMethod: true } generic ? generic.GetGenericArguments() : null;
            var module = method.Module;
            int i = 0;
            while (i < il.Length)
            {
                var op = il[i] == 0xFE && i + 1 < il.Length ? s_two[il[++i]] : s_one[il[i]];
                i++;
                text.Append(op.Value.ToString(CultureInfo.InvariantCulture)).Append(' ');
                switch (op.OperandType)
                {
                    case OperandType.InlineNone:
                        break;
                    case OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar:
                        text.Append(il[i].ToString(CultureInfo.InvariantCulture)).Append(' ');
                        i += 1;
                        break;
                    case OperandType.InlineVar:
                        text.Append(BitConverter.ToUInt16(il, i).ToString(CultureInfo.InvariantCulture)).Append(' ');
                        i += 2;
                        break;
                    case OperandType.InlineBrTarget or OperandType.InlineI or OperandType.ShortInlineR:
                        text.Append(BitConverter.ToInt32(il, i).ToString(CultureInfo.InvariantCulture)).Append(' ');
                        i += 4;
                        break;
                    case OperandType.InlineI8 or OperandType.InlineR:
                        text.Append(BitConverter.ToInt64(il, i).ToString(CultureInfo.InvariantCulture)).Append(' ');
                        i += 8;
                        break;
                    case OperandType.InlineSwitch:
                        int targets = BitConverter.ToInt32(il, i);
                        i += 4;
                        for (int s = 0; s < targets; s++, i += 4)
                        {
                            text.Append(BitConverter.ToInt32(il, i).ToString(CultureInfo.InvariantCulture)).Append(',');
                        }

                        text.Append(' ');
                        break;
                    default:
                        Token(module, op, BitConverter.ToInt32(il, i), typeArguments, methodArguments, text);
                        i += 4;
                        break;
                }
            }
        }

        // A token as the name of what it refers to, walking what it reaches.
        private void Token(System.Reflection.Module module, OpCode op, int token, Type[]? typeArguments, Type[]? methodArguments, StringBuilder text)
        {
            try
            {
                switch (op.OperandType)
                {
                    case OperandType.InlineString:
                        text.Append('"').Append(module.ResolveString(token)).Append("\" ");
                        return;
                    case OperandType.InlineSig:
                        text.Append(Convert.ToHexString(module.ResolveSignature(token))).Append(' ');
                        return;
                }

                var member = module.ResolveMember(token, typeArguments, methodArguments);
                text.Append(member?.DeclaringType).Append("::").Append(member).Append(' ');
                switch (member)
                {
                    case MethodBase called:
                        AddMethod(called);
                        if (called.DeclaringType is { IsGenericType: true } calledType)
                        {
                            foreach (var argument in calledType.GetGenericArguments())
                            {
                                AddType(argument);
                            }
                        }

                        if (called is MethodInfo { IsGenericMethod: true } generic)
                        {
                            foreach (var argument in generic.GetGenericArguments())
                            {
                                AddType(argument);
                            }
                        }

                        break;
                    case FieldInfo { IsStatic: true, DeclaringType: { } fieldType }:
                        var definition = fieldType.IsGenericType ? fieldType.GetGenericTypeDefinition() : fieldType;
                        if (InScope(definition) && definition.TypeInitializer is { } initializer)
                        {
                            Enqueue(initializer);
                        }

                        break;
                    case Type type:
                        AddType(type);
                        break;
                }
            }
            catch (Exception e) when (e is ArgumentException or MissingMemberException or TypeLoadException or BadImageFormatException)
            {
                text.Append("? ");
            }
        }
    }
}
