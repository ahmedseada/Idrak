// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;
using System.Text;
using Idrak;

// Plan 10, phase 0: the inventory of every abstraction in the library packages (interfaces, abstract classes,
// registries, the device contract and the internals other assemblies reach), generated from the built assemblies so it
// cannot drift; and the rule that every abstraction is declared under Idrak.Abstraction.*, with today's violations as
// an allow list that may only shrink.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] InventoryGroup =
    [
        ("abstraction inventory: plans/10-abstraction-inventory.md matches the library (IDRAK_UPDATE_INVENTORY=1 rewrites it)", InventoryCurrent),
        ("abstraction inventory: every interface, abstract class and registry outside Idrak.Abstraction.* is on the allow list, and the list names nothing that moved or is gone", AbstractionNamespaces),
    ];

    // The library packages (the CLI is an application: its own helpers are outside the rule).
    private static readonly string[] LibraryAssemblyNames =
        ["Idrak.Abstraction", "Idrak", "Idrak.LanguageModels", "Idrak.Datasets", "Idrak.Onnx", "Idrak.Onnx.Runtime", "Idrak.AspNetCore", "Idrak.Mcp"];

    // Each assembly whose internals others see, with those others (the tests aside): what phases 2 and 4 must turn into
    // public contract. Idrak.Abstraction's list must be empty after phase 4.
    private static readonly (string Target, string[] Users)[] FriendAssemblies =
    [
        ("Idrak.Abstraction", ["Idrak", "Idrak.LanguageModels", "Idrak.Onnx", "Idrak.Cli"]),
        ("Idrak", ["Idrak.Onnx", "Idrak.LanguageModels", "Idrak.Cli"]),
    ];

    private const string InventoryPath = "plans/10-abstraction-inventory.md", AllowListPath = "tests/Idrak.Tests/data/abstraction-allow-list.txt";

    private const BindingFlags AllDeclared = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

    private static bool UpdateInventory => Environment.GetEnvironmentVariable("IDRAK_UPDATE_INVENTORY") == "1";

    // These checks do not depend on the device: each runs once, on the first device of the run. The group runs first so
    // the registries hold their built-in names only (later tests register plug-ins of their own).
    private static readonly HashSet<string> InventoryDone = [];

    private static bool FirstRun(string check)
    {
        lock (InventoryDone)
        {
            return InventoryDone.Add(check);
        }
    }

    private static void InventoryCurrent(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(InventoryCurrent)))
        {
            return;
        }

        string path = Path.Combine(RepositoryRoot(), InventoryPath);
        string expected = Inventory();
        string actual = File.Exists(path) ? File.ReadAllText(path).ReplaceLineEndings("\n") : "";
        if (UpdateInventory)
        {
            File.WriteAllText(path, expected);
            return;
        }

        if (actual != expected)
        {
            string[] a = actual.Split('\n'), e = expected.Split('\n');
            int line = Enumerable.Range(0, Math.Max(a.Length, e.Length)).First(i => i >= a.Length || i >= e.Length || a[i] != e[i]);
            Check(false, $"{InventoryPath} is out of date at line {line + 1}: has \"{(line < a.Length ? a[line] : "")}\", the library gives "
                         + $"\"{(line < e.Length ? e[line] : "")}\" (IDRAK_UPDATE_INVENTORY=1 rewrites it)");
        }
    }

    private static void AbstractionNamespaces(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(AbstractionNamespaces)))
        {
            return;
        }

        var outside = Abstractions().Where(a => !InAbstractionNamespace(a.Type)).Select(a => Display(a.Type)).ToHashSet(StringComparer.Ordinal);
        string path = Path.Combine(RepositoryRoot(), AllowListPath);
        bool exists = File.Exists(path);
        var allowed = exists
            ? File.ReadLines(path).Select(l => l.Trim()).Where(l => l.Length > 0 && !l.StartsWith('#')).ToHashSet(StringComparer.Ordinal)
            : [];
        var added = outside.Except(allowed).Order(StringComparer.Ordinal).ToList();
        var gone = allowed.Except(outside).Order(StringComparer.Ordinal).ToList();

        // The list only shrinks: an update removes what moved or is gone, and writes the list only the first time.
        if (UpdateInventory && (!exists || gone.Count > 0))
        {
            var kept = exists ? allowed.Except(gone) : outside;
            File.WriteAllText(path, "# Plan 10: abstractions still declared outside Idrak.Abstraction.*. This list may only shrink: declare new\n"
                                    + "# interfaces, abstract classes and registries under Idrak.Abstraction.* (tests/Idrak.Tests/InventoryTests.cs).\n"
                                    + string.Concat(kept.Order(StringComparer.Ordinal).Select(l => l + "\n")));
            gone = [];
            if (!exists)
            {
                added = [];
            }
        }

        Check(added.Count == 0, $"declared outside Idrak.Abstraction.* (plan 10): {string.Join(", ", added)}");
        Check(gone.Count == 0, $"{AllowListPath} names abstractions that moved or are gone; remove them: {string.Join(", ", gone)}");
    }

    private static bool InAbstractionNamespace(Type type) =>
        type.Namespace is { } ns && (ns == "Idrak.Abstraction" || ns.StartsWith("Idrak.Abstraction.", StringComparison.Ordinal));

    private sealed record Abstraction(Type Type, string Kind);

    private static List<Assembly> LibraryAssemblies() => [.. LibraryAssemblyNames.Select(n => Assembly.Load(n))];

    // Every interface, abstract class and registry (a type declaring a static Register) in the library packages, public
    // or internal; compiler-generated types and those private to a class (implementation details, not contracts) aside.
    private static List<Abstraction> Abstractions()
    {
        var found = new List<Abstraction>();
        foreach (var type in LibraryAssemblies().SelectMany(a => a.GetTypes()))
        {
            if (Generated(type) || Visibility(type) == "private")
            {
                continue;
            }

            bool registry = type.GetMethods(AllDeclared).Any(m => m.IsStatic && m.Name == "Register");
            string? kind = type.IsInterface ? "interface"
                : type.IsClass && type.IsAbstract && !type.IsSealed ? (registry ? "abstract class, registry" : "abstract class")
                : registry ? "registry"
                : null;
            if (kind is not null)
            {
                found.Add(new(type, kind));
            }
        }

        return [.. found.OrderBy(a => Display(a.Type), StringComparer.Ordinal)];
    }

    private static bool Generated(Type type)
    {
        for (var t = type; t is not null; t = t.DeclaringType)
        {
            if (t.Name.Contains('<') || t.IsDefined(typeof(CompilerGeneratedAttribute), inherit: false))
            {
                return true;
            }
        }

        return false;
    }

    // The type's name as C# writes it: nested types with a dot, generic parameters by name.
    private static string Display(Type type)
    {
        string name = type.Name;
        int tick = name.IndexOf('`');
        if (tick >= 0)
        {
            name = name[..tick];
        }

        var own = type.GetGenericArguments().Skip(type.DeclaringType?.GetGenericArguments().Length ?? 0).ToArray();
        if (own.Length > 0)
        {
            name += "<" + string.Join(", ", own.Select(a => a.Name)) + ">";
        }

        return type.DeclaringType is { } outer ? Display(outer) + "." + name : (type.Namespace is { } ns ? ns + "." : "") + name;
    }

    private static string ShortName(Type type) => Display(type)[((type.Namespace?.Length + 1) ?? 0)..];

    private static string Visibility(Type type)
    {
        for (var t = type; t is not null; t = t.DeclaringType)
        {
            if (!(t.IsPublic || t.IsNestedPublic || t.IsNestedFamily || t.IsNestedFamORAssem))
            {
                return t.IsNestedPrivate ? "private" : "internal";
            }
        }

        return "public";
    }

    // The core types a contract's members mention: why it cannot move without them.
    private static readonly string[] CoreTypes = ["Tensor", "Module", "Device", "Backend", "Storage"];

    private static string Mentions(Type type)
    {
        var seen = new SortedSet<string>(StringComparer.Ordinal);
        void Add(Type? t)
        {
            if (t is null)
            {
                return;
            }

            if (t.HasElementType)
            {
                Add(t.GetElementType());
                return;
            }

            if (t.IsGenericType)
            {
                foreach (var a in t.GetGenericArguments())
                {
                    Add(a);
                }
            }

            if (t.Assembly.GetName().Name is "Idrak" or "Idrak.Abstraction" && CoreTypes.Contains(t.Name) && t != type)
            {
                seen.Add(t.Name);
            }
        }

        foreach (var member in type.GetMembers(AllDeclared))
        {
            switch (member)
            {
                case MethodBase m when !m.IsPrivate:
                    Add((m as MethodInfo)?.ReturnType);
                    foreach (var p in m.GetParameters())
                    {
                        Add(p.ParameterType);
                    }

                    break;
                case PropertyInfo p:
                    Add(p.PropertyType);
                    break;
                case FieldInfo f when !f.IsPrivate:
                    Add(f.FieldType);
                    break;
            }
        }

        Add(type.BaseType);
        return seen.Count == 0 ? "—" : string.Join(", ", seen);
    }

    // The concrete types in the library packages that implement an interface or derive from an abstract class.
    private static string Implementations(Type contract, List<Type> concrete)
    {
        static bool Derives(Type t, Type contract)
        {
            for (var b = t.BaseType; b is not null; b = b.BaseType)
            {
                if (b == contract || b.IsGenericType && b.GetGenericTypeDefinition() == contract)
                {
                    return true;
                }
            }

            return false;
        }

        var names = concrete
            .Where(t => contract.IsInterface
                ? t.GetInterfaces().Any(i => i == contract || i.IsGenericType && i.GetGenericTypeDefinition() == contract)
                : Derives(t, contract))
            .Select(ShortName).Order(StringComparer.Ordinal).ToList();
        return names.Count switch
        {
            0 => "—",
            <= 6 => string.Join(", ", names),
            _ => string.Join(", ", names.Take(6)) + $", +{names.Count - 6}",
        };
    }

    // A registry's built-in names, read from its static Names property when it has one.
    private static string RegisteredNames(Type type)
    {
        var names = type.GetProperty("Names", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
        if (names is null || !typeof(IEnumerable<string>).IsAssignableFrom(names.PropertyType) || type.ContainsGenericParameters)
        {
            return "—";
        }

        var list = ((IEnumerable<string>)names.GetValue(null)!).Order(StringComparer.OrdinalIgnoreCase).ToList();
        return list.Count == 0 ? "—" : list.Count <= 8 ? string.Join(", ", list) : string.Join(", ", list.Take(8)) + $", +{list.Count - 8}";
    }

    // The area under Idrak.Abstraction a contract is proposed to move to (plan 10, "Namespaces"); phases 1 to 3 settle it.
    private static string TargetNamespace(Type type)
    {
        string name = ShortName(type), ns = type.Namespace ?? "";
        string area = name switch
        {
            "Tensor" or "TensorScope" => "",
            "CpuBackend.IRangeKernel" => "Operations",
            "GraphOps" or "Autograd" or "DifferentiableFunction" => "Autograd",
            "IScaler" or "DistillationTeacher" or "TeacherDistributions" => "Training",
            "IWeightSource" or "WeightCodec" or "CheckpointFormats" or "ICheckpointFormat" or "ModelSources" or "IModelSource" or "ITensorStore"
                or "GgufTypes" or "GgufArchitectures" => "Formats",
            "PackedWeight" or "KeyValueLayout" or "KeyValueLayouts" or "RopeScalings" or "ITokenSampler" => "Generation",
            _ => ns switch
            {
                "Idrak.Backends" or "Idrak.Backends.Cuda" or "Idrak.Backends.Vulkan" or "Idrak.Backends.Hip" => "Devices",
                "Idrak.Layers" => "Modules",
                "Idrak.Optimizers" or "Idrak.Training" => "Training",
                "Idrak.Data" or "Idrak.Datasets" => "Data",
                "Idrak.Vision" => "Vision",
                "Idrak.Retrieval" => "Retrieval",
                "Idrak.Generation" or "Idrak.LanguageModels" => "Generation",
                "Idrak.Onnx" or "Idrak.Onnx.Runtime" => "Formats",
                "Idrak.Inference" or "Idrak.AspNetCore" or "Idrak.Mcp" => "Serving",
                "Idrak.Diagnostics" => "Diagnostics",
                _ when ns.StartsWith("Idrak.Datasets.", StringComparison.Ordinal) => "Data",
                _ when ns.StartsWith("Idrak.LanguageModels.", StringComparison.Ordinal) => "Generation",
                _ when ns.StartsWith("Idrak.Abstraction", StringComparison.Ordinal) => ns["Idrak.Abstraction".Length..].TrimStart('.'),
                _ => "",
            },
        };

        return area.Length == 0 ? "Idrak.Abstraction" : "Idrak.Abstraction." + area;
    }

    private static string Inventory()
    {
        var assemblies = LibraryAssemblies();
        var concrete = assemblies.SelectMany(a => a.GetTypes()).Where(t => t is { IsClass: true, IsAbstract: false } or { IsValueType: true } && !Generated(t)).ToList();
        var abstractions = Abstractions();
        var text = new StringBuilder();
        void Line(string line = "") => text.Append(line).Append('\n');
        static string Cell(string s) => s.Replace("|", "\\|");

        Line("# Plan 10, phase 0: inventory of the library's abstractions");
        Line();
        Line("Generated by `tests/Idrak.Tests/InventoryTests.cs` from the built assemblies; do not edit by hand. The test fails when");
        Line("this file and the library disagree; `IDRAK_UPDATE_INVENTORY=1 IDRAK_DEVICES=cpu IDRAK_FILTER=inventory dotnet run");
        Line("--project tests/Idrak.Tests` rewrites it. See [10-abstraction.md](10-abstraction.md).");
        Line();
        Line($"Scanned: {string.Join(", ", LibraryAssemblyNames.Select(n => $"`{n}`"))}. The CLI (`Idrak.Cli`) is an application and is");
        Line("scanned only as a user of `Idrak`'s internals.");
        Line();

        Line("## Summary");
        Line();
        Line("| Assembly | Interfaces | Abstract classes | Registries | Total | Outside `Idrak.Abstraction.*` |");
        Line("|---|---|---|---|---|---|");
        foreach (var assembly in assemblies)
        {
            var own = abstractions.Where(a => a.Type.Assembly == assembly).ToList();
            Line($"| `{assembly.GetName().Name}` | {own.Count(a => a.Kind == "interface")} | {own.Count(a => a.Kind.StartsWith("abstract", StringComparison.Ordinal))} "
                 + $"| {own.Count(a => a.Kind.Contains("registry", StringComparison.Ordinal))} | {own.Count} | {own.Count(a => !InAbstractionNamespace(a.Type))} |");
        }

        Line();
        Line("An abstract class that also declares `Register` counts once under abstract classes and once under registries.");
        Line();

        Line("## Abstractions");
        Line();
        Line("*Mentions*: the core types its members name (why it cannot move without them). *Implementations*: the concrete");
        Line("types in the library packages (the default is among them). *Registered*: a registry's built-in names. *Target*: the");
        Line("proposed namespace (phases 1 to 3 settle it).");
        foreach (var group in abstractions.GroupBy(a => a.Type.Assembly.GetName().Name!).OrderBy(g => Array.IndexOf(LibraryAssemblyNames, g.Key)))
        {
            Line();
            Line($"### {group.Key}");
            Line();
            Line("| Type | Kind | Visibility | Mentions | Implementations | Registered | Target |");
            Line("|---|---|---|---|---|---|---|");
            foreach (var a in group)
            {
                string registered = a.Kind.Contains("registry", StringComparison.Ordinal) ? RegisteredNames(a.Type) : "";
                string implementations = a.Kind == "registry" ? "" : Implementations(a.Type, concrete);
                Line($"| `{Cell(Display(a.Type))}` | {a.Kind} | {Visibility(a.Type)} | {Mentions(a.Type)} | {Cell(implementations)} | {Cell(registered)} | `{TargetNamespace(a.Type)}` |");
            }
        }

        Line();
        Line("## The device contract (`Backend`)");
        Line();
        BackendOperations(Line);

        Line();
        foreach (var (target, users) in FriendAssemblies)
        {
            Line();
            Line($"## Internals of `{target}` other assemblies use");
            Line();
            Line($"What each assembly `{target}` names in `InternalsVisibleTo` (the tests aside) references among its non-public types");
            Line("and members, read from that assembly's metadata. Phases 2 and 4 make each one public contract, or justify it line by line.");
            foreach (string user in users)
            {
                Line();
                Line($"### {user}");
                Line();
                var used = InternalsUsed(Assembly.Load(user), Assembly.Load(target));
                if (used.Count == 0)
                {
                    Line("None.");
                    continue;
                }

                Line($"{used.Values.Sum(v => v.Count)} members on {used.Count} types.");
                Line();
                Line("| Type | Internal members used |");
                Line("|---|---|");
                foreach (var (type, members) in used)
                {
                    Line($"| `{Cell(type)}` | {Cell(string.Join(", ", members))} |");
                }
            }
        }

        return text.ToString();
    }

    private static void BackendOperations(Action<string> line)
    {
        var backend = typeof(Device).Assembly.GetType("Idrak.Abstraction.Devices.Backend", throwOnError: true)!;
        var devices = new[] { typeof(Device).Assembly, typeof(Idrak.Layers.Sequential).Assembly }.SelectMany(a => a.GetTypes()).Where(t => !t.IsAbstract && backend.IsAssignableFrom(t) && !Generated(t)).OrderBy(t => t.Name, StringComparer.Ordinal).ToList();
        var operations = backend.GetMethods(AllDeclared).Where(m => (m.IsAbstract || m.IsVirtual && !m.IsFinal) && m.GetBaseDefinition() == m && !m.IsSpecialName)
            .OrderBy(m => m.Name, StringComparer.Ordinal).ThenBy(m => m.GetParameters().Length).ToList();
        var properties = backend.GetProperties(AllDeclared).Where(p => p.GetMethod is { } g && (g.IsAbstract || g.IsVirtual)).OrderBy(p => p.Name, StringComparer.Ordinal).ToList();

        bool Overrides(Type device, MethodInfo operation)
        {
            for (var t = device; t is not null && t != backend; t = t.BaseType)
            {
                if (t.GetMethods(AllDeclared).Any(m => m.GetBaseDefinition() == operation))
                {
                    return true;
                }
            }

            return false;
        }

        line($"`Idrak.Abstraction.Devices.Backend` ({Visibility(backend)}): {operations.Count(m => m.IsAbstract)} abstract and {operations.Count(m => !m.IsAbstract)} virtual "
             + $"methods, {properties.Count} abstract or virtual properties. Devices: {string.Join(", ", devices.Select(d => $"`{d.Name}`"))}. Each");
        line($"operation ({Idrak.Abstraction.Operations.OperationIndex.Count}, `Ops`) is a `NameKernel` method, the device's own kernel; `Backend.Name(...)` runs the kernel registered");
        line("for the device in `Kernels` instead, where there is one (plan 9). The rest is device plumbing: memory, copies, graphs, profiling.");
        line("");
        line("| Device | Methods overridden |");
        line("|---|---|");
        foreach (var device in devices)
        {
            line($"| `{device.Name}` | {operations.Count(o => Overrides(device, o))} of {operations.Count} |");
        }

        line("");
        line("| Operation | Kind | " + string.Join(" | ", devices.Select(d => d.Name.Replace("Backend", ""))) + " |");
        line("|---|---|" + string.Concat(devices.Select(_ => "---|")));
        foreach (var operation in operations)
        {
            string signature = $"{operation.Name}({operation.GetParameters().Length})";
            line($"| `{signature}` | {(operation.IsAbstract ? "abstract" : "virtual")} | " + string.Join(" | ", devices.Select(d => Overrides(d, operation) ? "✓" : "")) + " |");
        }
    }

    // The non-public types and members of `target` that `user` references, by type, read from user's metadata tables.
    private static SortedDictionary<string, List<string>> InternalsUsed(Assembly user, Assembly target)
    {
        var used = new SortedDictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        string targetName = target.GetName().Name!;
        using var stream = File.OpenRead(user.Location);
        using var pe = new PEReader(stream);
        var reader = pe.GetMetadataReader();
        var provider = new NameProvider(reader);

        Type? Resolve(TypeReferenceHandle handle)
        {
            var reference = reader.GetTypeReference(handle);
            string name = reader.GetString(reference.Name);
            switch (reference.ResolutionScope.Kind)
            {
                case HandleKind.TypeReference:
                    return Resolve((TypeReferenceHandle)reference.ResolutionScope)?.GetNestedType(name, BindingFlags.Public | BindingFlags.NonPublic);
                case HandleKind.AssemblyReference:
                    var assembly = reader.GetAssemblyReference((AssemblyReferenceHandle)reference.ResolutionScope);
                    if (reader.GetString(assembly.Name) != targetName)
                    {
                        return null;
                    }

                    string ns = reader.GetString(reference.Namespace);
                    return target.GetType(ns.Length == 0 ? name : ns + "." + name);
                default:
                    return null;
            }
        }

        void Add(Type type, string member) => (used.TryGetValue(Display(type), out var set) ? set : used[Display(type)] = new(StringComparer.Ordinal)).Add(member);

        foreach (var handle in reader.TypeReferences)
        {
            if (Resolve(handle) is { } type && Visibility(type) != "public")
            {
                Add(type, "(the type)");
            }
        }

        foreach (var handle in reader.MemberReferences)
        {
            var reference = reader.GetMemberReference(handle);
            Type? parent = reference.Parent.Kind switch
            {
                HandleKind.TypeReference => Resolve((TypeReferenceHandle)reference.Parent),
                HandleKind.TypeSpecification => reader.GetTypeSpecification((TypeSpecificationHandle)reference.Parent).DecodeSignature(provider, null) is { } spec
                                                && provider.Generic.TryGetValue(spec, out var definition) && definition.Kind == HandleKind.TypeReference
                    ? Resolve((TypeReferenceHandle)definition)
                    : null,
                _ => null,
            };
            if (parent is null)
            {
                continue;
            }

            string name = reader.GetString(reference.Name);
            if (reference.GetKind() == MemberReferenceKind.Field)
            {
                if (parent.GetField(name, AllDeclared) is { } field && !(field.IsPublic || field.IsFamily || field.IsFamilyOrAssembly))
                {
                    Add(parent, name);
                }

                continue;
            }

            int count = reference.DecodeMethodSignature(provider, null).ParameterTypes.Length;
            var candidates = parent.GetMembers(AllDeclared).OfType<MethodBase>().Where(m => m.Name == name && m.GetParameters().Length == count).ToList();
            if (candidates.Count > 0 && candidates.All(m => !(m.IsPublic || m.IsFamily || m.IsFamilyOrAssembly)))
            {
                string shown = name switch
                {
                    ".ctor" => "constructor",
                    _ when name.StartsWith("get_", StringComparison.Ordinal) || name.StartsWith("set_", StringComparison.Ordinal) => name[4..],
                    _ when name.StartsWith("add_", StringComparison.Ordinal) => name[4..],
                    _ => name + "()",
                };
                Add(parent, shown);
            }
        }

        return new(used.ToDictionary(p => p.Key, p => p.Value.ToList()), StringComparer.Ordinal);
    }

    // Decodes signatures only far enough to count parameters and find the definition of a generic instance.
    private sealed class NameProvider(MetadataReader reader) : ISignatureTypeProvider<string, object?>
    {
        public Dictionary<string, EntityHandle> Generic { get; } = new(StringComparer.Ordinal);

        public string GetTypeFromReference(MetadataReader r, TypeReferenceHandle handle, byte rawTypeKind) =>
            "ref:" + MetadataTokens.GetToken(handle);

        public string GetTypeFromDefinition(MetadataReader r, TypeDefinitionHandle handle, byte rawTypeKind) =>
            "def:" + MetadataTokens.GetToken(handle);

        public string GetGenericInstantiation(string genericType, ImmutableArray<string> typeArguments)
        {
            string key = genericType + "<" + string.Join(",", typeArguments) + ">";
            if (genericType.StartsWith("ref:", StringComparison.Ordinal))
            {
                Generic[key] = MetadataTokens.EntityHandle(int.Parse(genericType[4..], System.Globalization.CultureInfo.InvariantCulture));
            }

            return key;
        }

        public string GetTypeFromSpecification(MetadataReader r, object? context, TypeSpecificationHandle handle, byte rawTypeKind) =>
            reader.GetTypeSpecification(handle).DecodeSignature(this, context);

        public string GetPrimitiveType(PrimitiveTypeCode typeCode) => typeCode.ToString();

        public string GetArrayType(string elementType, ArrayShape shape) => elementType + "[,]";

        public string GetSZArrayType(string elementType) => elementType + "[]";

        public string GetByReferenceType(string elementType) => elementType + "&";

        public string GetPointerType(string elementType) => elementType + "*";

        public string GetPinnedType(string elementType) => elementType;

        public string GetGenericMethodParameter(object? context, int index) => "!!" + index;

        public string GetGenericTypeParameter(object? context, int index) => "!" + index;

        public string GetFunctionPointerType(MethodSignature<string> signature) => "fnptr";

        public string GetModifiedType(string modifier, string unmodifiedType, bool isRequired) => unmodifiedType;
    }
}
