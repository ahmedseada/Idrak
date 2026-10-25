// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Formats;

/// <summary>
/// A runtime that runs an ONNX model as a <see cref="Module"/>, outside Idrak's own import of it (Idrak.Onnx.Runtime
/// registers "onnxruntime": ONNX Runtime). A host finds one through <see cref="OnnxRuntimeLoaders"/>, by name, without a
/// reference to the runtime's assembly: the idrak tool's <c>onnx check</c> compares Idrak's import with the first one
/// registered.
/// </summary>
public interface IOnnxRuntimeLoader : IKind
{
    /// <summary>The model whose .onnx file's bytes are <paramref name="model"/>, run by this runtime (inference only).</summary>
    /// <exception cref="NotSupportedException">The model has inputs or outputs the runtime does not take.</exception>
    Module Load(byte[] model);
}

/// <summary>
/// The ONNX runtimes (<see cref="IOnnxRuntimeLoader"/>) by name, in the order they were registered. The library holds
/// none itself: Idrak.Onnx.Runtime registers ONNX Runtime ("onnxruntime") when the application ships it, or when a host
/// loads it later (the idrak tool's <c>-P</c>, <see cref="Plugins.IdrakPlugins.Load"/>).
/// </summary>
public static class OnnxRuntimeLoaders
{
    private static readonly KindTable<IOnnxRuntimeLoader> Table = new(nameof(OnnxRuntimeLoaders), "ONNX runtime",
        "add it with OnnxRuntimeLoaders.Register, or reference Idrak.Onnx.Runtime",
        unguarded: "a runtime is what a model's outputs are compared with: another one answering in its place would compare against something else");

    static OnnxRuntimeLoaders() => LibraryDefaults.Ensure(typeof(OnnxRuntimeLoaders));

    /// <summary>Registers <paramref name="loader"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IOnnxRuntimeLoader loader) => Table.Register(loader, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's runtime <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in order.</summary>
    public static IReadOnlyList<string> Names => Table.Names;

    /// <summary>The registered runtimes, in the order of <see cref="Names"/>.</summary>
    public static IReadOnlyList<IOnnxRuntimeLoader> Registered => Table.Kinds;

    /// <summary>The runtime registered as <paramref name="name"/>, or null.</summary>
    public static IOnnxRuntimeLoader? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The runtime registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones.</exception>
    public static IOnnxRuntimeLoader Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>Who registered the runtime <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered runtime as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();
}
