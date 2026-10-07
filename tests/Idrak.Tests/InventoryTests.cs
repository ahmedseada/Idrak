// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;
using System.Text;
using Idrak;

// Plan 10, phase 0: the inventory of every abstraction in the library packages (interfaces, abstract classes,
// registries, the device contract and the internals other assemblies reach), generated from the built assemblies so it
// cannot drift; the rule that every abstraction is declared under Idrak.Abstraction.* or a package's own *.Abstractions
// namespace (its allow list is empty and may only shrink); and, from phase 8c, decision 10: a contract lives in
// Idrak.Abstraction when Abstraction itself or several library packages use it, else in the one package that does.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] InventoryGroup =
    [
        ("abstraction inventory: plans/10-abstraction-inventory.md matches the library (IDRAK_UPDATE_INVENTORY=1 rewrites it)", InventoryCurrent),
        ("abstraction inventory: every interface, abstract class and registry outside Idrak.Abstraction.* and the packages' own *.Abstractions is on the allow list, and the list names nothing that moved or is gone", AbstractionNamespaces),
        ("abstraction inventory: every contract lives with its users (decision 10): in Idrak.Abstraction when Abstraction or several library packages use it, else in the one package that does", ContractsWithTheirUsers),
        ("abstraction inventory: Idrak.Abstraction grants its internals to the tests only; every internal of Idrak another library assembly uses is justified in the list, and the list names nothing no longer used", InternalsJustified),
    ];

    // The library packages (the CLI is an application: its own helpers are outside the rule).
    private static readonly string[] LibraryAssemblyNames =
        ["Idrak.Abstraction", "Idrak", "Idrak.Nlp", "Idrak.Data", "Idrak.Vision", "Idrak.Onnx.Runtime", "Idrak.AspNetCore", "Idrak.Mcp"];

    // Each assembly whose internals others see, with those others (the tests aside): what must turn into public contract
    // or be justified. Idrak.Abstraction is not one since phase 4: it grants no library assembly its internals.
    private static readonly (string Target, string[] Users)[] FriendAssemblies =
    [
        ("Idrak", ["Idrak.Nlp", "Idrak.Cli"]),
    ];

    private const string InventoryPath = "plans/10-abstraction-inventory.md", AllowListPath = "tests/Idrak.Tests/data/abstraction-allow-list.txt",
        JustifiedPath = "tests/Idrak.Tests/data/internals-justified.txt";

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

        var outside = Abstractions().Where(a => !DeclaredWhereAllowed(a.Type)).Select(a => Display(a.Type)).ToHashSet(StringComparer.Ordinal);
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
            File.WriteAllText(path, "# Plan 10: abstractions declared outside Idrak.Abstraction.* and the packages' own *.Abstractions namespaces. This\n"
                                    + "# list may only shrink: declare new interfaces, abstract classes and registries there (tests/Idrak.Tests/InventoryTests.cs).\n"
                                    + string.Concat(kept.Order(StringComparer.Ordinal).Select(l => l + "\n")));
            gone = [];
            if (!exists)
            {
                added = [];
            }
        }

        Check(added.Count == 0, $"declared outside Idrak.Abstraction.* (in Idrak.Abstraction) or the package's own *.Abstractions namespace (plan 10): {string.Join(", ", added)}");
        Check(gone.Count == 0, $"{AllowListPath} names abstractions that moved or are gone; remove them: {string.Join(", ", gone)}");
    }

    // Plan 10, phase 2's bar: every internal another library assembly uses is listed with why (Idrak's use of
    // Idrak.Abstraction is phase 4's); the list only shrinks.
    private static void InternalsJustified(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(InternalsJustified)))
        {
            return;
        }

        // Plan 10, phase 4: the GPU devices in Idrak, and every other library assembly, see Idrak.Abstraction's public surface only.
        var friends = typeof(Device).Assembly.GetCustomAttributes<InternalsVisibleToAttribute>().Select(a => a.AssemblyName).Order(StringComparer.Ordinal).ToList();
        Check(friends.SequenceEqual(["Idrak.Tests"]), $"Idrak.Abstraction grants its internals to {string.Join(", ", friends)}; only Idrak.Tests may see them");

        var used = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (target, users) in FriendAssemblies)
        {
            foreach (string user in users)
            {
                foreach (var (type, members) in InternalsUsed(Assembly.Load(user), Assembly.Load(target)))
                {
                    used.UnionWith(members.Select(m => $"{user} | {type} | {m}"));
                }
            }
        }

        var listed = File.ReadLines(Path.Combine(RepositoryRoot(), JustifiedPath))
            .Where(l => l.Length > 0 && !l.StartsWith('#'))
            .Select(l => l.Split(" | ", 4))
            .ToList();
        Check(listed.All(f => f.Length == 4 && f[3].Trim().Length > 0), $"{JustifiedPath}: every line is user | type | member | why");
        var justified = listed.Select(f => string.Join(" | ", f.Take(3))).ToHashSet(StringComparer.Ordinal);
        var missing = used.Except(justified).Order(StringComparer.Ordinal).ToList();
        var stale = justified.Except(used).Order(StringComparer.Ordinal).ToList();
        Check(missing.Count == 0, $"internals used without a reason in {JustifiedPath} (make them public contract, or add a line): {string.Join("; ", missing)}");
        Check(stale.Count == 0, $"{JustifiedPath} lists internals no longer used; remove: {string.Join("; ", stale)}");
    }

    // Decision 10: a contract in Idrak.Abstraction that one library package alone uses moves to that package; one in a
    // package that several use (or another package alone) moves to Abstraction (or to that package). Users are measured
    // from the assemblies' metadata, see ContractUsers.
    private static void ContractsWithTheirUsers(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ContractsWithTheirUsers)))
        {
            return;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_USAGE_DUMP") is { } dump)
        {
            var u = ContractUsers();
            File.WriteAllLines(dump, u.Users.Where(p => p.Key.Assembly == typeof(Device).Assembly || p.Key.Assembly.GetName().Name != "x").OrderBy(p => p.Key.Assembly.GetName().Name).ThenBy(p => p.Key.FullName)
                .Select(p => $"{p.Key.Assembly.GetName().Name} {p.Key.FullName} :: {string.Join(",", p.Value.Order())} <- {string.Join(",", u.Referrers[p.Key].Select(r => r.Name).Order().Take(12))}"));
        }

        var misplaced = Abstractions().Select(a => Misplaced(a.Type)).OfType<string>().ToList();
        Check(misplaced.Count == 0, $"contracts that do not live with their users (decision 10): {string.Join("; ", misplaced)}");
    }

    private const string AbstractionAssembly = "Idrak.Abstraction";

    // Where a contract may be declared: under Idrak.Abstraction.* in the Abstraction assembly, and under the package's own
    // *.Abstractions namespace in a package (never the bare Idrak.Abstractions, too close to Idrak.Abstraction).
    private static bool DeclaredWhereAllowed(Type type)
    {
        string ns = type.Namespace ?? "", assembly = type.Assembly.GetName().Name!;
        if (assembly == AbstractionAssembly)
        {
            return InAbstractionNamespace(type);
        }

        // The .Abstractions sub-namespace of a namespace this assembly's implementations use (Idrak.Vision.Abstractions
        // beside Idrak.Vision, Idrak.Generation.Abstractions beside Nlp's Idrak.Generation).
        return ns.EndsWith(".Abstractions", StringComparison.Ordinal) && ns != "Idrak.Abstractions" && !InAbstractionNamespace(type)
               && type.Assembly.GetTypes().Any(t => t.Namespace == ns[..^".Abstractions".Length]);
    }

    // The assembly a contract belongs in by decision 10: Abstraction when Abstraction or more than one library package
    // uses it; the one package that uses it otherwise; where it is when no library assembly does.
    private static string BelongsIn(Type type)
    {
        var users = ContractUsers().Users[Outermost(type)];
        return users.Count switch
        {
            0 => type.Assembly.GetName().Name!,
            1 => users.Single(),
            _ => AbstractionAssembly,
        };
    }

    // Why a contract is in the wrong assembly, with where it goes; null when it is where decision 10 puts it.
    private static string? Misplaced(Type type)
    {
        string assembly = type.Assembly.GetName().Name!, target = BelongsIn(type);
        if (target == assembly)
        {
            return null;
        }

        string users = string.Join(", ", ContractUsers().Users[Outermost(type)].Order(StringComparer.Ordinal));
        return target == AbstractionAssembly
            ? $"{Display(type)} (in {assembly}) is used by {users}: move it to {AbstractionArea(type)}"
            : $"{Display(type)} (in {assembly}) is used by {target} alone: move it to {target}, namespace {PackageNamespace(type, target)}";
    }

    // The namespace a contract takes in the one package that uses it: the .Abstractions sub-namespace of the namespace most
    // of that package's types naming it use (through the Abstraction types that move with it when none names it directly).
    private static string PackageNamespace(Type type, string package)
    {
        var referrers = ContractUsers().Referrers;
        var seen = new HashSet<Type> { Outermost(type) };
        var level = new List<Type> { Outermost(type) };
        while (level.Count > 0)
        {
            var next = level.SelectMany(t => referrers.TryGetValue(t, out var r) ? r : []).Where(seen.Add).ToList();
            var namespaces = next.Where(t => t.Assembly.GetName().Name == package)
                .Select(t => t.Namespace is { } ns && ns.EndsWith(".Abstractions", StringComparison.Ordinal) ? ns[..^".Abstractions".Length] : t.Namespace ?? "")
                .Where(ns => ns.Length > 0 && ns != "Idrak")
                .GroupBy(ns => ns).OrderByDescending(g => g.Count()).ThenBy(g => g.Key, StringComparer.Ordinal).ToList();
            if (namespaces.Count > 0)
            {
                return namespaces[0].Key + ".Abstractions";
            }

            level = [.. next.Where(t => t.Assembly.GetName().Name == AbstractionAssembly)];
        }

        return package == "Idrak" ? "Idrak.<area>.Abstractions" : package + ".Abstractions";
    }

    // The area under Idrak.Abstraction a contract moving into Abstraction takes, by the namespace of its implementations.
    private static string AbstractionArea(Type type)
    {
        string ns = type.Namespace ?? "";
        if (ns.EndsWith(".Abstractions", StringComparison.Ordinal))
        {
            ns = ns[..^".Abstractions".Length];
        }

        string area = ns switch
        {
            "Idrak.Layers" => "Modules",
            "Idrak.Optimizers" or "Idrak.Training" => "Training",
            "Idrak.Data" or "Idrak.Data.Parquet" => "Data",
            "Idrak.Vision" => "Vision",
            "Idrak.Retrieval" => "Retrieval",
            "Idrak.Generation" or "Idrak.Nlp" => "Generation",
            "Idrak.Models" or "Idrak.Onnx" or "Idrak.Onnx.Runtime" => "Formats",
            "Idrak.Inference" or "Idrak.AspNetCore" or "Idrak.Mcp" => "Serving",
            "Idrak.Diagnostics" => "Diagnostics",
            _ when ns.StartsWith("Idrak.Backends", StringComparison.Ordinal) || ns.StartsWith("Idrak.Gpu", StringComparison.Ordinal) => "Devices",
            _ => "",
        };
        return area.Length == 0 ? AbstractionAssembly : AbstractionAssembly + "." + area;
    }

    // Which library assemblies use each type (by its outermost type: nested types belong to theirs), and which types refer
    // to it. Another assembly uses a type when its metadata references it (the TypeReferences table, and the signatures and
    // method bodies of its types). The defining assembly uses it when one of its other types refers to it; in
    // Idrak.Abstraction that need is counted through the referring type: when that type has one user, its user inherits the
    // reference (the two move together), and when it has several (or none but Abstraction), Abstraction does.
    private sealed record Usage(Dictionary<Type, HashSet<string>> Users, Dictionary<Type, HashSet<Type>> Referrers);

    private static Usage? _contractUsers;

    private static Usage ContractUsers()
    {
        if (_contractUsers is { } done)
        {
            return done;
        }

        var assemblies = LibraryAssemblies();
        var library = assemblies.ToHashSet();
        var users = new Dictionary<Type, HashSet<string>>();
        var referrers = new Dictionary<Type, HashSet<Type>>();
        foreach (var unit in assemblies.SelectMany(a => a.GetTypes()).Select(Outermost).Where(t => !t.Name.StartsWith('<')))
        {
            users.TryAdd(unit, []);
            referrers.TryAdd(unit, []);
        }

        foreach (var assembly in assemblies)
        {
            string name = assembly.GetName().Name!;
            var (byType, referenced) = References(assembly, library);
            foreach (var target in referenced.Where(t => t.Assembly != assembly && users.ContainsKey(t)))
            {
                users[target].Add(name);
            }

            foreach (var (from, targets) in byType)
            {
                foreach (var target in targets.Where(t => t != from && users.ContainsKey(t)))
                {
                    referrers[target].Add(from);
                    if (target.Assembly != assembly)
                    {
                        users[target].Add(name);
                    }
                }
            }
        }

        // A package uses its own type when another of its types refers to it.
        foreach (var (target, from) in referrers.Where(p => p.Key.Assembly.GetName().Name != AbstractionAssembly))
        {
            if (from.Any(r => r.Assembly == target.Assembly))
            {
                users[target].Add(target.Assembly.GetName().Name!);
            }
        }

        // Abstraction's own references, to a fixed point: a referring type passes on its users, and Abstraction itself when
        // it has several (it stays in Abstraction, and so must what it names).
        var own = referrers.Where(p => p.Key.Assembly.GetName().Name == AbstractionAssembly)
            .Select(p => (Target: p.Key, From: p.Value.Where(r => r.Assembly == p.Key.Assembly).ToList())).ToList();
        for (bool changed = true; changed;)
        {
            changed = false;
            foreach (var (target, from) in own)
            {
                foreach (var r in from)
                {
                    int before = users[target].Count;
                    users[target].UnionWith(users[r]);
                    if (users[r].Count > 1)
                    {
                        users[target].Add(AbstractionAssembly);
                    }

                    changed |= users[target].Count != before;
                }
            }
        }

        return _contractUsers = new(users, referrers);
    }

    private static Type Outermost(Type type)
    {
        while (type.DeclaringType is { } outer)
        {
            type = outer;
        }

        return type;
    }

    // The operand size of every IL instruction, by its opcode value.
    private static readonly Dictionary<short, OperandType> OperandTypes = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(f => (OpCode)f.GetValue(null)!).ToDictionary(o => o.Value, o => o.OperandType);

    // What each type of `assembly` (by outermost type) refers to among the library's types, read from its metadata: base
    // type, interfaces, generic constraints, the signatures of its fields, methods, properties and events, and the types,
    // methods and fields its method bodies name. Also every library type in its TypeReferences table.
    private static (Dictionary<Type, HashSet<Type>> ByType, HashSet<Type> Referenced) References(Assembly assembly, HashSet<Assembly> library)
    {
        using var stream = File.OpenRead(assembly.Location);
        using var pe = new PEReader(stream);
        var reader = pe.GetMetadataReader();
        var module = assembly.ManifestModule;
        var collector = new TypeCollector(reader);

        Type? Resolve(EntityHandle handle)
        {
            try
            {
                var type = module.ResolveType(MetadataTokens.GetToken(handle));
                return library.Contains(type.Assembly) ? Outermost(type) : null;
            }
            catch (Exception e) when (e is ArgumentException or TypeLoadException or FileNotFoundException or BadImageFormatException)
            {
                return null;
            }
        }

        var referenced = reader.TypeReferences.Select(h => Resolve(h)).OfType<Type>().ToHashSet();
        var byType = new Dictionary<Type, HashSet<Type>>();
        foreach (var handle in reader.TypeDefinitions)
        {
            if (Resolve(handle) is not { } unit || unit.Name.StartsWith('<'))
            {
                continue;
            }

            collector.Seen.Clear();
            var definition = reader.GetTypeDefinition(handle);
            collector.Add(definition.BaseType);
            foreach (var implementation in definition.GetInterfaceImplementations())
            {
                collector.Add(reader.GetInterfaceImplementation(implementation).Interface);
            }

            collector.AddConstraints(definition.GetGenericParameters());
            foreach (var field in definition.GetFields())
            {
                reader.GetFieldDefinition(field).DecodeSignature(collector, null);
            }

            foreach (var property in definition.GetProperties())
            {
                reader.GetPropertyDefinition(property).DecodeSignature(collector, null);
            }

            foreach (var e in definition.GetEvents())
            {
                collector.Add(reader.GetEventDefinition(e).Type);
            }

            foreach (var methodHandle in definition.GetMethods())
            {
                var method = reader.GetMethodDefinition(methodHandle);
                method.DecodeSignature(collector, null);
                collector.AddConstraints(method.GetGenericParameters());
                if (method.RelativeVirtualAddress != 0)
                {
                    var body = pe.GetMethodBody(method.RelativeVirtualAddress);
                    if (!body.LocalSignature.IsNil)
                    {
                        reader.GetStandaloneSignature(body.LocalSignature).DecodeLocalSignature(collector, null);
                    }

                    collector.AddBody(body.GetILBytes()!);
                }
            }

            var targets = byType.TryGetValue(unit, out var set) ? set : byType[unit] = [];
            targets.UnionWith(collector.Seen.Select(Resolve).OfType<Type>());
        }

        return (byType, referenced);
    }

    // Records the type definitions and references a signature or a method body names (type arguments included).
    private sealed class TypeCollector(MetadataReader reader) : ISignatureTypeProvider<int, object?>
    {
        public HashSet<EntityHandle> Seen { get; } = [];

        public void Add(EntityHandle handle)
        {
            if (handle.IsNil)
            {
                return;
            }

            switch (handle.Kind)
            {
                case HandleKind.TypeDefinition or HandleKind.TypeReference:
                    Seen.Add(handle);
                    break;
                case HandleKind.TypeSpecification:
                    reader.GetTypeSpecification((TypeSpecificationHandle)handle).DecodeSignature(this, null);
                    break;
                case HandleKind.MethodDefinition:
                    Seen.Add(reader.GetMethodDefinition((MethodDefinitionHandle)handle).GetDeclaringType());
                    break;
                case HandleKind.FieldDefinition:
                    Seen.Add(reader.GetFieldDefinition((FieldDefinitionHandle)handle).GetDeclaringType());
                    break;
                case HandleKind.MemberReference:
                    var member = reader.GetMemberReference((MemberReferenceHandle)handle);
                    Add(member.Parent);
                    if (member.GetKind() == MemberReferenceKind.Method)
                    {
                        member.DecodeMethodSignature(this, null);
                    }
                    else
                    {
                        member.DecodeFieldSignature(this, null);
                    }

                    break;
                case HandleKind.MethodSpecification:
                    var specification = reader.GetMethodSpecification((MethodSpecificationHandle)handle);
                    Add(specification.Method);
                    specification.DecodeSignature(this, null);
                    break;
            }
        }

        public void AddConstraints(GenericParameterHandleCollection parameters)
        {
            foreach (var parameter in parameters)
            {
                foreach (var constraint in reader.GetGenericParameter(parameter).GetConstraints())
                {
                    Add(reader.GetGenericParameterConstraint(constraint).Type);
                }
            }
        }

        // Walks the instructions and adds what each token operand names.
        public void AddBody(byte[] il)
        {
            for (int i = 0; i < il.Length;)
            {
                short code = il[i++];
                if (code == 0xFE)
                {
                    code = unchecked((short)(0xFE00 | il[i++]));
                }

                var operand = OperandTypes[code];
                switch (operand)
                {
                    case OperandType.InlineField or OperandType.InlineMethod or OperandType.InlineType or OperandType.InlineTok:
                        Add(MetadataTokens.EntityHandle(BitConverter.ToInt32(il, i)));
                        i += 4;
                        break;
                    case OperandType.InlineSwitch:
                        i += 4 + 4 * BitConverter.ToInt32(il, i);
                        break;
                    case OperandType.InlineNone:
                        break;
                    case OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar:
                        i += 1;
                        break;
                    case OperandType.InlineVar:
                        i += 2;
                        break;
                    case OperandType.InlineI8 or OperandType.InlineR:
                        i += 8;
                        break;
                    default:   // InlineBrTarget, InlineI, InlineSig, InlineString, ShortInlineR
                        i += 4;
                        break;
                }
            }
        }

        public int GetTypeFromDefinition(MetadataReader r, TypeDefinitionHandle handle, byte rawTypeKind)
        {
            Seen.Add(handle);
            return 0;
        }

        public int GetTypeFromReference(MetadataReader r, TypeReferenceHandle handle, byte rawTypeKind)
        {
            Seen.Add(handle);
            return 0;
        }

        public int GetTypeFromSpecification(MetadataReader r, object? context, TypeSpecificationHandle handle, byte rawTypeKind) =>
            reader.GetTypeSpecification(handle).DecodeSignature(this, context);

        public int GetGenericInstantiation(int genericType, ImmutableArray<int> typeArguments) => 0;

        public int GetPrimitiveType(PrimitiveTypeCode typeCode) => 0;

        public int GetArrayType(int elementType, ArrayShape shape) => 0;

        public int GetSZArrayType(int elementType) => 0;

        public int GetByReferenceType(int elementType) => 0;

        public int GetPointerType(int elementType) => 0;

        public int GetPinnedType(int elementType) => 0;

        public int GetGenericMethodParameter(object? context, int index) => 0;

        public int GetGenericTypeParameter(object? context, int index) => 0;

        public int GetFunctionPointerType(MethodSignature<int> signature) => 0;

        public int GetModifiedType(int modifier, int unmodifiedType, bool isRequired) => 0;
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
        Line("| Assembly | Interfaces | Abstract classes | Registries | Total | Misplaced (decision 10) |");
        Line("|---|---|---|---|---|---|");
        foreach (var assembly in assemblies)
        {
            var own = abstractions.Where(a => a.Type.Assembly == assembly).ToList();
            Line($"| `{assembly.GetName().Name}` | {own.Count(a => a.Kind == "interface")} | {own.Count(a => a.Kind.StartsWith("abstract", StringComparison.Ordinal))} "
                 + $"| {own.Count(a => a.Kind.Contains("registry", StringComparison.Ordinal))} | {own.Count} | {own.Count(a => Misplaced(a.Type) is not null)} |");
        }

        Line();
        Line("An abstract class that also declares `Register` counts once under abstract classes and once under registries.");
        Line();

        Line("## Abstractions");
        Line();
        Line("*Mentions*: the core types its members name (why it cannot move without them). *Implementations*: the concrete");
        Line("types in the library packages (the default is among them). *Registered*: a registry's built-in names. *Users*: the");
        Line("library assemblies that use it (without the `Idrak.` prefix): another assembly uses it when its metadata names it; the");
        Line("defining one when another of its types does. In Abstraction that need counts through the referring type, which");
        Line("passes on its own users (they move together), or Abstraction when it has several. *Belongs in*: where decision 10");
        Line("puts it (Abstraction when Abstraction or several packages use it, else the one package that does).");
        foreach (var group in abstractions.GroupBy(a => a.Type.Assembly.GetName().Name!).OrderBy(g => Array.IndexOf(LibraryAssemblyNames, g.Key)))
        {
            Line();
            Line($"### {group.Key}");
            Line();
            Line("| Type | Kind | Visibility | Mentions | Implementations | Registered | Users | Belongs in |");
            Line("|---|---|---|---|---|---|---|---|");
            foreach (var a in group)
            {
                string registered = a.Kind.Contains("registry", StringComparison.Ordinal) ? RegisteredNames(a.Type) : "";
                string implementations = a.Kind == "registry" ? "" : Implementations(a.Type, concrete);
                Line($"| `{Cell(Display(a.Type))}` | {a.Kind} | {Visibility(a.Type)} | {Mentions(a.Type)} | {Cell(implementations)} | {Cell(registered)} | {UsersCell(a.Type)} | {BelongsInCell(a.Type)} |");
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

    private static string ShortAssembly(string name) => name == "Idrak" ? name : name["Idrak.".Length..];

    private static string UsersCell(Type type)
    {
        var users = ContractUsers().Users[Outermost(type)];
        return users.Count == 0 ? "—" : string.Join(", ", users.Select(ShortAssembly).Order(StringComparer.Ordinal));
    }

    private static string BelongsInCell(Type type)
    {
        string target = BelongsIn(type);
        return target == type.Assembly.GetName().Name ? ShortAssembly(target) : $"**{ShortAssembly(target)}** (move)";
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
