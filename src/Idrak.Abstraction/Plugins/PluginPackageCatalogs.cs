// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Plugins;

/// <summary>
/// Declares names a plug-in package registers into one of the library's registries, so a refusal can name the package
/// that holds a name nobody registered without loading or registering it:
/// <c>[assembly: IdrakPluginNames(nameof(PretrainedArchitectures), QwenPlugin.Qwen2ForCausalLM, ...)]</c>, one per
/// registry. Read by <see cref="PluginPackageCatalogs"/>' library catalog from the packages loaded and from the program's
/// files (their metadata only). The library declares no family: the packages do.
/// </summary>
/// <param name="registry">The registry the names go into, as its refusals name it (<c>nameof(PretrainedArchitectures)</c>).</param>
/// <param name="names">The names the package registers there.</param>
[AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
public sealed class IdrakPluginNamesAttribute(string registry, params string[] names) : Attribute
{
    /// <summary>The registry the names go into.</summary>
    public string Registry { get; } = registry;

    /// <summary>The names the package registers there.</summary>
    public IReadOnlyList<string> Names { get; } = names;
}

/// <summary>
/// Knows which plug-in package holds a name of a registry (an architecture, a family): what a registry's refusal asks to
/// name the package to reference instead of guessing one. Registered in <see cref="PluginPackageCatalogs"/>.
/// </summary>
public interface IPluginPackageCatalog
{
    /// <summary>The catalog's name in <see cref="PluginPackageCatalogs"/>.</summary>
    string Name { get; }

    /// <summary>The name of the package that registers <paramref name="name"/> into <paramref name="registry"/>; null when the catalog knows none.</summary>
    string? PackageOf(string registry, string name);
}

/// <summary>
/// The catalogs of plug-in packages (<see cref="IPluginPackageCatalog"/>), asked in registration order by
/// <see cref="PackageOf"/>. The library's own, <see cref="Present"/>, reads the <see cref="IdrakPluginNamesAttribute"/>
/// declarations of the packages the program has (loaded, or among its files); an app adds one that knows packages it
/// does not ship (a package index, say) with <see cref="Register"/>.
/// </summary>
public static class PluginPackageCatalogs
{
    /// <summary>The library's catalog: the declarations of the plug-in packages the program has, loaded or as files.</summary>
    public const string Present = "present";

    private static readonly SlotTable<string, IPluginPackageCatalog> Table = new(nameof(PluginPackageCatalogs), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a catalog only names a package in a refusal's message; nothing it answers is run or loaded");

    static PluginPackageCatalogs() => Table.RegisterDefault(Present, new PresentPluginPackages());

    /// <summary>Registers a catalog under its <see cref="IPluginPackageCatalog.Name"/>; one of a taken name takes its place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPluginPackageCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentException.ThrowIfNullOrWhiteSpace(catalog.Name);
        Table.Register(catalog.Name, catalog, Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the catalog <paramref name="name"/> an app registered (the library's comes back); false when none did.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered catalogs' names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The catalog registered as <paramref name="name"/>, or null.</summary>
    public static IPluginPackageCatalog? Find(string name) => Table.Find(name);

    /// <summary>The catalog registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is.</exception>
    public static IPluginPackageCatalog Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(PluginPackageCatalogs), name, Table.Keys,
            $"No plug-in package catalog '{name}' is registered (registered: {string.Join(", ", Table.Keys)}); add it with PluginPackageCatalogs.Register.");

    /// <summary>The library's catalog <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IPluginPackageCatalog? Default(string name) => Table.Default(name);

    /// <summary>Who registered the catalog <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// The package that registers <paramref name="name"/> into <paramref name="registry"/>, from the first catalog that
    /// knows one; null when none does (a refusal then names no package rather than guess one).
    /// </summary>
    public static string? PackageOf(string registry, string name)
    {
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentNullException.ThrowIfNull(name);
        foreach (var catalog in Table.Values)
        {
            if (catalog.PackageOf(registry, name) is { Length: > 0 } package)
            {
                return package;
            }
        }

        return null;
    }
}

/// <summary>
/// The library's catalog (<see cref="PluginPackageCatalogs.Present"/>): the <see cref="IdrakPluginNamesAttribute"/>
/// declarations of the assemblies loaded now, then of the program's files that reference Idrak.Abstraction (read from
/// their metadata, without loading them, once a process: rule 75). Asked only when a registry refuses a name.
/// </summary>
internal sealed class PresentPluginPackages : IPluginPackageCatalog
{
    private const string AttributeNamespace = "Idrak.Abstraction.Plugins", AttributeName = nameof(IdrakPluginNamesAttribute), Abstraction = "Idrak.Abstraction";

    private readonly Lazy<Declaration[]> _files = new(ReadFiles);

    /// <summary>One declaration: the package (its assembly's name), the registry and its names.</summary>
    internal sealed record Declaration(string Package, string Registry, string[] Names);

    public string Name => PluginPackageCatalogs.Present;

    public string? PackageOf(string registry, string name)
    {
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic)
            {
                continue;
            }

            foreach (var declared in assembly.GetCustomAttributes<IdrakPluginNamesAttribute>())
            {
                if (declared.Registry == registry && declared.Names.Contains(name))
                {
                    return assembly.GetName().Name;
                }
            }
        }

        foreach (var declared in _files.Value)
        {
            if (declared.Registry == registry && Array.IndexOf(declared.Names, name) >= 0)
            {
                return declared.Package;
            }
        }

        return null;
    }

    private static Declaration[] ReadFiles()
    {
        if (LibraryFiles.Present is not { } files)
        {
            return [];
        }

        var declarations = new List<Declaration>();
        foreach (string path in files.Program)
        {
            try
            {
                Read(path, declarations);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or BadImageFormatException or InvalidOperationException)
            {
                // A file that is not an assembly, or cannot be read, declares nothing.
            }
        }

        return [.. declarations];
    }

    /// <summary>Adds the declarations of the assembly at <paramref name="path"/> (none when it does not reference Idrak.Abstraction).</summary>
    internal static void Read(string path, List<Declaration> declarations)
    {
        using var stream = File.OpenRead(path);
        using var pe = new PEReader(stream);
        if (!pe.HasMetadata)
        {
            return;
        }

        var reader = pe.GetMetadataReader();
        if (!reader.IsAssembly || !References(reader, Abstraction))
        {
            return;
        }

        var definition = reader.GetAssemblyDefinition();
        string? package = null;
        foreach (var handle in definition.GetCustomAttributes())
        {
            var attribute = reader.GetCustomAttribute(handle);
            if (attribute.Constructor.Kind != HandleKind.MemberReference
                || reader.GetMemberReference((MemberReferenceHandle)attribute.Constructor).Parent is not { Kind: HandleKind.TypeReference } parent)
            {
                continue;
            }

            var type = reader.GetTypeReference((TypeReferenceHandle)parent);
            if (!reader.StringComparer.Equals(type.Name, AttributeName) || !reader.StringComparer.Equals(type.Namespace, AttributeNamespace))
            {
                continue;
            }

            // The blob (ECMA-335 II.23.3): prolog 0x0001, the registry (a serialized string), the names (a count, then each).
            var blob = reader.GetBlobReader(attribute.Value);
            if (blob.ReadUInt16() != 1 || blob.ReadSerializedString() is not { } registry)
            {
                continue;
            }

            int count = blob.ReadInt32();
            var names = new string[Math.Max(count, 0)];
            for (int i = 0; i < names.Length; i++)
            {
                names[i] = blob.ReadSerializedString() ?? "";
            }

            package ??= reader.GetString(definition.Name);
            declarations.Add(new Declaration(package, registry, names));
        }
    }

    private static bool References(MetadataReader reader, string name)
    {
        foreach (var handle in reader.AssemblyReferences)
        {
            if (reader.StringComparer.Equals(reader.GetAssemblyReference(handle).Name, name))
            {
                return true;
            }
        }

        return false;
    }
}
