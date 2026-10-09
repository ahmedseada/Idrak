// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;

// Plan 10, phase 7 (W4.4): the public surface of every library package, written to api/<package>.txt from the built
// assemblies' metadata (types, members, signatures with nullability, the attributes that change what callers may do), and
// compared on every run. An intended change rewrites the files (IDRAK_UPDATE_API=1) in the same commit, with a changelog
// line; an accidental one fails here.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ApiSurfaceGroup =
    [
        ("public API: api/*.txt matches the public surface of every library package (IDRAK_UPDATE_API=1 rewrites them)", ApiSurfaceCurrent),
        ("IdrakFromSource: build/IdrakFromSource.targets swaps exactly the library's packages (every packable project under src/ but the tool)", IdrakFromSourceList),
    ];

    private static void IdrakFromSourceList(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(IdrakFromSourceList)))
        {
            return;
        }

        string root = RepositoryRoot();
        string targets = File.ReadAllText(Path.Combine(root, "build", "IdrakFromSource.targets"));
        var match = System.Text.RegularExpressions.Regex.Match(targets, "<_IdrakSourcePackage Include=\"([^\"]+)\"");
        Check(match.Success, "no _IdrakSourcePackage list in build/IdrakFromSource.targets");
        var listed = match.Groups[1].Value.Split(';').Order(StringComparer.Ordinal).ToList();
        var packages = Directory.GetDirectories(Path.Combine(root, "src")).Select(Path.GetFileName)
            .Where(name => File.Exists(Path.Combine(root, "src", name!, name + ".csproj")) && name != "Idrak.Cli"
                           && !File.ReadAllText(Path.Combine(root, "src", name!, name + ".csproj")).Contains("<IsPackable>false</IsPackable>", StringComparison.Ordinal))
            .Order(StringComparer.Ordinal).ToList();
        Check(listed.SequenceEqual(packages), $"listed: {string.Join(", ", listed)}; src/: {string.Join(", ", packages)}");
        Check(packages.ToHashSet().SetEquals(ApiAssemblyNames), $"the API guard covers {string.Join(", ", ApiAssemblyNames)}");
    }

    // The packages whose surface is guarded: the library's, the testing kit included (apps build on it), the CLI not (an app).
    private static readonly string[] ApiAssemblyNames =
        ["Idrak.Abstraction", "Idrak.Abstraction.Testing", "Idrak.Gpu", "Idrak", "Idrak.Data", "Idrak.Nlp", "Idrak.Vision", "Idrak.Mcp", "Idrak.AspNetCore", "Idrak.Onnx.Runtime"];

    private static bool UpdateApi => Environment.GetEnvironmentVariable("IDRAK_UPDATE_API") == "1";

    private static void ApiSurfaceCurrent(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ApiSurfaceCurrent)))
        {
            return;
        }

        string folder = Path.Combine(RepositoryRoot(), "api");
        var problems = new List<string>();
        foreach (string name in ApiAssemblyNames)
        {
            string path = Path.Combine(folder, name + ".txt");
            string actual = ApiSurface.Describe(Assembly.Load(name));
            if (UpdateApi)
            {
                Directory.CreateDirectory(folder);
                File.WriteAllText(path, actual);
                continue;
            }

            string expected = File.Exists(path) ? File.ReadAllText(path).ReplaceLineEndings("\n") : "";
            if (expected != actual)
            {
                problems.Add($"{name}: {ApiSurface.FirstDifference(expected, actual)}");
            }
        }

        var stray = Directory.Exists(folder) ? Directory.GetFiles(folder, "*.txt").Select(Path.GetFileNameWithoutExtension).Except(ApiAssemblyNames).ToList() : [];
        if (stray.Count > 0 && !UpdateApi)
        {
            problems.Add($"api/ has files for no guarded package: {string.Join(", ", stray)}");
        }

        Check(problems.Count == 0, "the public API changed:\n  " + string.Join("\n  ", problems)
            + "\nIf the change is intended: IDRAK_UPDATE_API=1 IDRAK_DEVICES=cpu IDRAK_FILTER=\"public API\" dotnet run -c Release --project tests/Idrak.Tests,"
            + " then commit api/ with a CHANGELOG line.");
    }

    // The public surface of an assembly as sorted text: one block per type (its declaration, then its members).
    internal static class ApiSurface
    {
        private const BindingFlags Declared = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        // Attributes that change what callers may do or see, kept in the declarations.
        private static readonly HashSet<string> KeptAttributes =
        [
            "System.ObsoleteAttribute", "System.FlagsAttribute", "System.Diagnostics.CodeAnalysis.ExperimentalAttribute",
            "System.Diagnostics.CodeAnalysis.RequiresUnreferencedCodeAttribute", "System.Diagnostics.CodeAnalysis.RequiresDynamicCodeAttribute",
            "System.Diagnostics.CodeAnalysis.SetsRequiredMembersAttribute",
        ];

        public static string Describe(Assembly assembly)
        {
            var text = new StringBuilder();
            text.Append("// The public API of ").Append(assembly.GetName().Name).Append(" (generated by tests/Idrak.Tests/ApiSurfaceTests.cs; IDRAK_UPDATE_API=1 rewrites it).\n");
            foreach (var type in assembly.GetExportedTypes().Where(Visible).OrderBy(t => t.FullName, StringComparer.Ordinal))
            {
                text.Append('\n').Append(Declaration(type)).Append('\n');
                foreach (string member in Members(type).Order(StringComparer.Ordinal))
                {
                    text.Append("  ").Append(member).Append('\n');
                }
            }

            return text.ToString();
        }

        public static string FirstDifference(string expected, string actual)
        {
            string[] e = expected.Split('\n'), a = actual.Split('\n');
            var removed = e.Except(a).Where(l => l.Length > 0).Take(3).ToList();
            var added = a.Except(e).Where(l => l.Length > 0).Take(3).ToList();
            return expected.Length == 0 ? "no api file yet"
                : $"{(removed.Count > 0 ? "removed: " + string.Join(" | ", removed.Select(l => l.Trim())) : "")}"
                  + $"{(removed.Count > 0 && added.Count > 0 ? "; " : "")}{(added.Count > 0 ? "added: " + string.Join(" | ", added.Select(l => l.Trim())) : "")}"
                  + (removed.Count == 0 && added.Count == 0 ? "the order or the blank lines changed" : "");
        }

        // A type outsiders can name: public, nested only in such types, not made by the compiler.
        private static bool Visible(Type type) =>
            !type.Name.Contains('<', StringComparison.Ordinal) && !type.IsDefined(typeof(CompilerGeneratedAttribute), false)
            && (type.IsPublic || type.IsNestedPublic || (type.IsNestedFamily || type.IsNestedFamORAssem) && type.DeclaringType is { IsSealed: false })
            && (type.DeclaringType is null || Visible(type.DeclaringType));

        private static string Declaration(Type type)
        {
            var parts = new List<string>(Attributes(type.CustomAttributes));
            parts.Add(type.IsNested && !type.IsNestedPublic ? "protected" : "public");
            if (type.IsSubclassOf(typeof(MulticastDelegate)))
            {
                var invoke = type.GetMethod("Invoke")!;
                parts.Add($"delegate {TypeName(invoke.ReturnType, Nullability(invoke.ReturnParameter))} {Name(type)}{Constraints(type)}({Parameters(invoke)})");
                return string.Join(" ", parts);
            }

            if (type.IsEnum)
            {
                parts.Add($"enum {Name(type)} : {TypeName(Enum.GetUnderlyingType(type), null)}");
                return string.Join(" ", parts);
            }

            bool record = type.GetMethod("<Clone>$", Declared) is not null || type.IsValueType && type.GetMethod("PrintMembers", Declared) is not null;
            if (type.IsInterface)
            {
                parts.Add("interface");
            }
            else if (type.IsValueType)
            {
                if (type.IsDefined(typeof(IsReadOnlyAttribute), false))
                {
                    parts.Add("readonly");
                }

                if (type.IsByRefLike)
                {
                    parts.Add("ref");
                }

                parts.Add(record ? "record struct" : "struct");
            }
            else
            {
                parts.Add(type is { IsAbstract: true, IsSealed: true } ? "static" : type.IsAbstract ? "abstract" : type.IsSealed ? "sealed" : "");
                parts.Add(record ? "record" : "class");
            }

            var bases = new List<string>();
            if (type is { IsClass: true, BaseType: { } baseType } && baseType != typeof(object))
            {
                bases.Add(TypeName(baseType, null));
            }

            var inherited = type.BaseType?.GetInterfaces() ?? [];
            bases.AddRange(type.GetInterfaces().Where(i => !inherited.Contains(i) && (i.IsPublic || i.IsNestedPublic)).Select(i => TypeName(i, null)).Order(StringComparer.Ordinal));
            parts.Add(Name(type) + (bases.Count > 0 ? " : " + string.Join(", ", bases) : "") + Constraints(type));
            return string.Join(" ", parts.Where(p => p.Length > 0));
        }

        private static IEnumerable<string> Members(Type type)
        {
            bool outsidersDerive = !type.IsSealed && !type.IsValueType && !type.IsInterface && type.GetConstructors(Declared).Any(c => c.IsPublic || c.IsFamily || c.IsFamilyOrAssembly);
            bool Shown(MethodBase? m) => m is not null && (m.IsPublic || outsidersDerive && (m.IsFamily || m.IsFamilyOrAssembly));
            string Access(MethodBase m) => m.IsPublic ? "public" : "protected";
            bool Generated(MemberInfo m) => m.Name.Contains('<', StringComparison.Ordinal) || m.IsDefined(typeof(CompilerGeneratedAttribute), false);

            if (type.IsEnum)
            {
                foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.Static))
                {
                    yield return $"{field.Name} = {Convert.ToString(field.GetRawConstantValue(), CultureInfo.InvariantCulture)}";
                }

                yield break;
            }

            if (type.IsSubclassOf(typeof(MulticastDelegate)))
            {
                yield break;
            }

            foreach (var field in type.GetFields(Declared))
            {
                if (Generated(field) || !(field.IsPublic || outsidersDerive && (field.IsFamily || field.IsFamilyOrAssembly)))
                {
                    continue;
                }

                string modifiers = (field.IsPublic ? "public" : "protected") + (field.IsLiteral ? " const" : field.IsStatic ? " static" : "") + (field.IsInitOnly ? " readonly" : "");
                string value = field.IsLiteral ? " = " + Literal(field.GetRawConstantValue()) : "";
                yield return $"{Attributes(field.CustomAttributes, " ")}{modifiers} {TypeName(field.FieldType, Nullability(field))} {field.Name}{value}";
            }

            foreach (var constructor in type.GetConstructors(Declared))
            {
                if (Shown(constructor) && !constructor.IsStatic)
                {
                    yield return $"{Attributes(constructor.CustomAttributes, " ")}{Access(constructor)} {Name(type, generic: false)}({Parameters(constructor)})";
                }
            }

            foreach (var property in type.GetProperties(Declared))
            {
                var get = property.GetMethod;
                var set = property.SetMethod;
                var any = Shown(get) ? get! : Shown(set) ? set! : null;
                if (any is null || Generated(property) || property.Name == "EqualityContract")    // a record's own, made by the compiler
                {
                    continue;
                }

                bool init = set?.ReturnParameter.GetRequiredCustomModifiers().Any(m => m.FullName == "System.Runtime.CompilerServices.IsExternalInit") == true;
                string accessors = string.Join(" ", new[]
                {
                    Shown(get) ? (get!.IsPublic || any.IsFamily ? "" : "protected ") + "get;" : null,
                    Shown(set) ? (set!.IsPublic || any.IsFamily ? "" : "protected ") + (init ? "init;" : "set;") : null,
                }.Where(a => a is not null));
                var indexes = property.GetIndexParameters();
                string name = indexes.Length == 0 ? property.Name : $"this[{string.Join(", ", indexes.Select(Parameter))}]";
                string required = property.IsDefined(typeof(RequiredMemberAttribute), false) ? "required " : "";
                yield return $"{Attributes(property.CustomAttributes, " ")}{Access(any)}{Modifiers(any, type)} {required}{TypeName(property.PropertyType, Nullability(property))} {name} {{ {accessors} }}";
            }

            foreach (var e in type.GetEvents(Declared))
            {
                var add = e.AddMethod;
                if (Shown(add) && !Generated(e))
                {
                    yield return $"{Access(add!)}{Modifiers(add!, type)} event {TypeName(e.EventHandlerType!, Nullability(e))} {e.Name}";
                }
            }

            foreach (var method in type.GetMethods(Declared))
            {
                if (!Shown(method) || Generated(method) || method.IsSpecialName && !method.Name.StartsWith("op_", StringComparison.Ordinal)
                    || method.Name is "PrintMembers" or "<Clone>$" || method.Name == "Deconstruct" && IsRecordMember(type))
                {
                    continue;
                }

                string generic = method.IsGenericMethodDefinition ? $"<{string.Join(", ", method.GetGenericArguments().Select(a => a.Name))}>" : "";
                string constraints = method.IsGenericMethodDefinition ? string.Concat(method.GetGenericArguments().Select(Constraint)) : "";
                yield return $"{Attributes(method.CustomAttributes, " ")}{Access(method)}{Modifiers(method, type)} {TypeName(method.ReturnType, Nullability(method.ReturnParameter))} {method.Name}{generic}({Parameters(method)}){constraints}";
            }

            foreach (var nested in type.GetNestedTypes(Declared).Where(Visible))
            {
                yield return $"nested {Name(nested)} (below)";
            }
        }

        // The members a record writes itself (Deconstruct of a positional record is part of its shape, kept by the declaration).
        private static bool IsRecordMember(Type type) => type.GetMethod("<Clone>$", Declared) is not null;

        private static string Modifiers(MethodInfo method, Type owner)
        {
            if (owner.IsInterface)
            {
                return method.IsStatic ? (method.IsAbstract ? " static abstract" : " static") : method.IsAbstract ? "" : " (default)";
            }

            if (method.IsStatic)
            {
                return " static";
            }

            if (method.IsAbstract)
            {
                return " abstract";
            }

            if (method.IsVirtual && !method.IsFinal)
            {
                return method.GetBaseDefinition().DeclaringType == method.DeclaringType ? " virtual" : " override";
            }

            if (method.IsVirtual && method.GetBaseDefinition().DeclaringType != method.DeclaringType)
            {
                return " sealed override";
            }

            return "";
        }

        private static string Parameters(MethodBase method) => string.Join(", ", method.GetParameters().Select((p, i) =>
            (i == 0 && method.IsDefined(typeof(ExtensionAttribute), false) ? "this " : "") + Parameter(p)));

        private static string Parameter(ParameterInfo p)
        {
            var type = p.ParameterType;
            string prefix = "";
            if (type.IsByRef)
            {
                type = type.GetElementType()!;
                prefix = p.IsOut ? "out " : p.IsIn || p.IsDefined(typeof(IsReadOnlyAttribute), false) ? "in " : "ref ";
            }

            if (p.IsDefined(typeof(ParamArrayAttribute), false) || p.IsDefined(typeof(ParamCollectionAttribute), false))
            {
                prefix = "params " + prefix;
            }

            string scoped = p.IsDefined(typeof(ScopedRefAttribute), false) ? "scoped " : "";
            string value = p.HasDefaultValue ? " = " + Literal(p.RawDefaultValue, type) : p.IsOptional ? " = default" : "";
            return $"{scoped}{prefix}{TypeName(type, Nullability(p))} {p.Name}{value}";
        }

        private static string Literal(object? value, Type? type = null) => value switch
        {
            null or DBNull or Missing => type is { IsValueType: true } && Nullable.GetUnderlyingType(type) is null ? "default" : "null",
            // A multi-line raw string constant holds the checkout's line endings (CRLF on Windows with autocrlf): written as "\n" whatever they are.
            string s => "\"" + s.ReplaceLineEndings("\n").Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal) + "\"",
            char c => $"'{c}'",
            bool b => b ? "true" : "false",
            float f => f.ToString("R", CultureInfo.InvariantCulture) + "f",
            double d => d.ToString("R", CultureInfo.InvariantCulture),
            _ when type is { IsEnum: true } || Nullable.GetUnderlyingType(type ?? typeof(object)) is { IsEnum: true } =>
                $"({TypeName(Nullable.GetUnderlyingType(type!) ?? type!, null)}){Convert.ToString(value, CultureInfo.InvariantCulture)}",
            _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? "",
        };

        private static IEnumerable<string> Attributes(IEnumerable<CustomAttributeData> attributes) =>
            attributes.Where(a => KeptAttributes.Contains(a.AttributeType.FullName ?? "")).Select(a =>
                $"[{a.AttributeType.Name[..^"Attribute".Length]}{(a.ConstructorArguments.Count > 0 ? "(" + string.Join(", ", a.ConstructorArguments.Select(c => Literal(c.Value))) + ")" : "")}]")
            .Order(StringComparer.Ordinal);

        private static string Attributes(IEnumerable<CustomAttributeData> attributes, string separator) =>
            string.Concat(Attributes(attributes).Select(a => a + separator));

        // A type's name with its generic parameters, outer types first (Outer.Inner<T>).
        private static string Name(Type type, bool generic = true)
        {
            string name = type.Name;
            int tick = name.IndexOf('`', StringComparison.Ordinal);
            name = tick >= 0 ? name[..tick] : name;
            if (!generic)
            {
                return name;
            }

            var own = type.GetGenericArguments().Skip(type.DeclaringType?.GetGenericArguments().Length ?? 0).ToArray();
            string arguments = own.Length > 0 ? $"<{string.Join(", ", own.Select(a => (a.GenericParameterAttributes.HasFlag(GenericParameterAttributes.Covariant) ? "out " : a.GenericParameterAttributes.HasFlag(GenericParameterAttributes.Contravariant) ? "in " : "") + a.Name))}>" : "";
            return (type.DeclaringType is { } outer ? Name(outer) + "." : type.Namespace + ".") + name + arguments;
        }

        private static string Constraints(Type type) =>
            type.IsGenericTypeDefinition ? string.Concat(type.GetGenericArguments().Skip(type.DeclaringType?.GetGenericArguments().Length ?? 0).Select(Constraint)) : "";

        private static string Constraint(Type parameter)
        {
            var parts = new List<string>();
            var a = parameter.GenericParameterAttributes;
            if (a.HasFlag(GenericParameterAttributes.ReferenceTypeConstraint))
            {
                parts.Add("class");
            }

            if (a.HasFlag(GenericParameterAttributes.NotNullableValueTypeConstraint))
            {
                parts.Add(parameter.GetGenericParameterConstraints().Any(c => c == typeof(ValueType)) ? "struct" : "unmanaged");
            }
            else if (!a.HasFlag(GenericParameterAttributes.ReferenceTypeConstraint) && NullableFlag(parameter) == 1)
            {
                parts.Add("notnull");
            }

            parts.AddRange(parameter.GetGenericParameterConstraints().Where(c => c != typeof(ValueType)).Select(c => TypeName(c, null)).Order(StringComparer.Ordinal));
            if (a.HasFlag(GenericParameterAttributes.DefaultConstructorConstraint) && !a.HasFlag(GenericParameterAttributes.NotNullableValueTypeConstraint))
            {
                parts.Add("new()");
            }

            if (a.HasFlag((GenericParameterAttributes)0x20))                       // AllowByRefLike
            {
                parts.Add("allows ref struct");
            }

            return parts.Count > 0 ? $" where {parameter.Name} : {string.Join(", ", parts)}" : "";
        }

        // A type parameter's nullable flag: its own NullableAttribute, else the nearest NullableContextAttribute (method,
        // then the types it is nested in); 1 is not-null (the `notnull` constraint), 2 maybe-null, 0 oblivious.
        private static byte NullableFlag(Type parameter)
        {
            static byte? Flag(IEnumerable<CustomAttributeData> attributes, string name) =>
                attributes.FirstOrDefault(c => c.AttributeType.Name == name) is { } c && c.ConstructorArguments.Count == 1
                    ? c.ConstructorArguments[0].Value switch { byte b => b, System.Collections.ObjectModel.ReadOnlyCollection<CustomAttributeTypedArgument> list => (byte)list[0].Value!, _ => null }
                    : null;

            if (Flag(parameter.CustomAttributes, "NullableAttribute") is { } own)
            {
                return own;
            }

            if (parameter.DeclaringMethod is { } method && Flag(method.CustomAttributes, "NullableContextAttribute") is { } inMethod)
            {
                return inMethod;
            }

            for (var type = parameter.DeclaringType; type is not null; type = type.DeclaringType)
            {
                if (Flag(type.CustomAttributes, "NullableContextAttribute") is { } inType)
                {
                    return inType;
                }
            }

            return 0;
        }

        private static NullabilityInfo? Nullability(ParameterInfo p) => Safe(() => new NullabilityInfoContext().Create(p));

        private static NullabilityInfo? Nullability(PropertyInfo p) => Safe(() => new NullabilityInfoContext().Create(p));

        private static NullabilityInfo? Nullability(FieldInfo f) => Safe(() => new NullabilityInfoContext().Create(f));

        private static NullabilityInfo? Nullability(EventInfo e) => Safe(() => new NullabilityInfoContext().Create(e));

        private static NullabilityInfo? Safe(Func<NullabilityInfo> create)
        {
            try
            {
                return create();
            }
            catch (Exception e) when (e is ArgumentException or NotSupportedException or InvalidOperationException)
            {
                return null;
            }
        }

        // A type as C# writes it, with ? where the metadata says the reference may be null.
        private static string TypeName(Type type, NullabilityInfo? nullability)
        {
            if (type.IsByRef)
            {
                return "ref " + TypeName(type.GetElementType()!, nullability?.ElementType ?? nullability);
            }

            string mark = nullability is { ReadState: NullabilityState.Nullable } && !type.IsValueType ? "?" : "";
            if (type.IsArray)
            {
                string rank = type.GetArrayRank() == 1 ? "[]" : $"[{new string(',', type.GetArrayRank() - 1)}]";
                return TypeName(type.GetElementType()!, nullability?.ElementType) + rank + mark;
            }

            if (type.IsPointer)
            {
                return TypeName(type.GetElementType()!, null) + "*";
            }

            if (type.IsGenericParameter)
            {
                return type.Name;                    // T? on an unconstrained T reads the same as T in the metadata: not shown
            }

            if (Nullable.GetUnderlyingType(type) is { } underlying)
            {
                return TypeName(underlying, nullability?.GenericTypeArguments.FirstOrDefault()) + "?";
            }

            string? keyword = Type.GetTypeCode(type) switch
            {
                TypeCode.Boolean => "bool", TypeCode.Byte => "byte", TypeCode.SByte => "sbyte", TypeCode.Char => "char",
                TypeCode.Int16 => "short", TypeCode.UInt16 => "ushort", TypeCode.Int32 => "int", TypeCode.UInt32 => "uint",
                TypeCode.Int64 => "long", TypeCode.UInt64 => "ulong", TypeCode.Single => "float", TypeCode.Double => "double",
                TypeCode.Decimal => "decimal", TypeCode.String => "string", _ when type == typeof(object) => "object", _ when type == typeof(void) => "void",
                _ when type == typeof(nint) => "nint", _ when type == typeof(nuint) => "nuint", _ => null,
            };
            if (keyword is not null && !type.IsEnum)
            {
                return keyword + mark;
            }

            if (type.IsGenericType && type.FullName?.StartsWith("System.ValueTuple`", StringComparison.Ordinal) == true
                || type.IsGenericType && type.GetGenericTypeDefinition().FullName?.StartsWith("System.ValueTuple`", StringComparison.Ordinal) == true)
            {
                var items = type.GetGenericArguments();
                return "(" + string.Join(", ", items.Select((t, i) => TypeName(t, nullability?.GenericTypeArguments.ElementAtOrDefault(i)))) + ")" + mark;
            }

            string name = type.Name;
            int tick = name.IndexOf('`', StringComparison.Ordinal);
            name = tick >= 0 ? name[..tick] : name;
            var arguments = type.GetGenericArguments();
            int outer = type.DeclaringType?.GetGenericArguments().Length ?? 0;
            string own = arguments.Length > outer
                ? "<" + string.Join(", ", arguments.Skip(outer).Select((t, i) => TypeName(t, nullability?.GenericTypeArguments.ElementAtOrDefault(i + outer)))) + ">"
                : "";
            string prefix = type.DeclaringType is { } declaring && !type.IsGenericParameter ? TypeName(declaring, null) + "." : (type.Namespace is { } ns ? ns + "." : "");
            if (type.DeclaringType is not null && prefix.Contains('<', StringComparison.Ordinal) && arguments.Length > 0)
            {
                prefix = Name(type.DeclaringType, generic: false) + ".";
            }

            return prefix + name + own + mark;
        }
    }
}
