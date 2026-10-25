// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Layers.Abstractions;

/// <summary>
/// A layer type a <c>GraphModule</c> can hold, registered by its <see cref="IKind.Name"/> (the "type" its JSON writes) in
/// <see cref="LayerTypes"/>: the modules it describes (<see cref="ModuleType"/>, and types derived from it without a type
/// of their own), how it describes one's settings and how it creates one again from them (its weights are loaded
/// separately). Implement <see cref="ILayerType{T}"/>, typed by the module.
/// </summary>
public interface ILayerType : IKind
{
    /// <summary>The module type it describes.</summary>
    Type ModuleType { get; }

    /// <summary>The layer's settings as JSON (the "type" key is added by <see cref="LayerTypes.Describe"/>).</summary>
    JsonObject Describe(Module module);

    /// <summary>A new layer on <paramref name="device"/> from the settings <see cref="Describe"/> wrote.</summary>
    Module Create(JsonObject description, Device device);
}

/// <summary>A layer type (<see cref="ILayerType"/>) of modules of type <typeparamref name="T"/>.</summary>
/// <typeparam name="T">The module type.</typeparam>
public interface ILayerType<T> : ILayerType where T : Module
{
    /// <summary>The layer's settings as JSON (the "type" key is added by <see cref="LayerTypes.Describe"/>).</summary>
    JsonObject Describe(T module);

    /// <summary>A new layer on <paramref name="device"/> from the settings <see cref="Describe(T)"/> wrote.</summary>
    new T Create(JsonObject description, Device device);

    Type ILayerType.ModuleType => typeof(T);

    JsonObject ILayerType.Describe(Module module) => Describe((T)module);

    Module ILayerType.Create(JsonObject description, Device device) => Create(description, device);
}

/// <summary>
/// The layer types a <c>GraphModule</c> can hold, by the "type" name its JSON writes (<see cref="ILayerType"/>). The
/// standard layers are registered here ("linear", "conv2d", "batchnorm", ..., and "sequential" for blocks of them); add
/// your own with <see cref="Register(ILayerType)"/>, so a graph holding them survives <c>GraphModule.ToJson</c>,
/// <c>GraphModule.FromJson</c> and model packages.
/// </summary>
public static class LayerTypes
{
    // Describing is one call; creating makes the layer once (its weights too), so under Shadow only describing is compared.
    private static readonly KindTable<ILayerType> Registry = new(nameof(LayerTypes), "Layer type", "add it with LayerTypes.Register",
        (slot, app, library) => new GuardedType(slot, app, library), StringComparer.Ordinal);

    // The types by module type, from the registry's current entries (a type registered under a later name wins).
    private static (IReadOnlyList<ILayerType> Source, Dictionary<Type, ILayerType> ByType) _byType = ([], []);

    // Idrak's layers (linear, conv2d, attention, ..., sequential) are registered before the first use.
    static LayerTypes() => Overrides.AsLibraryDefaults(LibraryLayerTypes.RegisterAll);   // the built-in layer types, on first use

    /// <summary>
    /// Registers <paramref name="type"/> under its <see cref="IKind.Name"/>. Under a built-in name it overrides the
    /// library's, which stays behind it (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ILayerType type) => Registry.Register(type, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>
    /// Registers the layer type <paramref name="type"/> for modules of type <typeparamref name="T"/> from two functions
    /// (made into an <see cref="ILayerType{T}"/>): <paramref name="describe"/> returns the layer's settings as JSON (the
    /// "type" key is added), <paramref name="create"/> makes a new layer on the given device from that JSON. Under a
    /// built-in name it overrides the library's, which stays behind it (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register<T>(string type, Func<T, JsonObject> describe, Func<JsonObject, Device, T> create) where T : Module
    {
        ArgumentException.ThrowIfNullOrEmpty(type);
        ArgumentNullException.ThrowIfNull(describe);
        ArgumentNullException.ThrowIfNull(create);
        Registry.Slots.Register(type, new FunctionType<T>(type, describe, create), System.Reflection.Assembly.GetCallingAssembly(), create);
    }

    /// <summary>Removes the app's layer type <paramref name="type"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string type) => Registry.Unregister(type);

    /// <summary>The registered layer type names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>Who registered the layer type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string type) => Registry.Origin(type);

    /// <summary>Whether the library has a layer type <paramref name="type"/> of its own, whatever an app registered over it.</summary>
    public static bool HasDefault(string type) => Registry.Slots.HasDefault(type);

    /// <summary>
    /// What happens when the app's layer type <paramref name="type"/> fails (<see cref="SlotPolicy.Throw"/> unless set:
    /// the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's): describing falls back per call, creating when the layer is made.
    /// </summary>
    public static void SetPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(type, policy, shadowRate);

    /// <summary>Whether <paramref name="module"/>'s type (or a base type) is registered.</summary>
    public static bool CanDescribe(Module module) => Find(module.GetType()) is not null;

    /// <summary>The layer's settings as JSON, with its "type" name first.</summary>
    public static JsonObject Describe(Module module)
    {
        ArgumentNullException.ThrowIfNull(module);
        var type = Find(module.GetType()) ?? throw new NotSupportedException(
            $"{module.GetType().Name} cannot be described (layer types: {string.Join(", ", Names)}); add it with LayerTypes.Register.");
        var description = new JsonObject { ["type"] = type.Name };
        foreach (var (key, value) in type.Describe(module))
        {
            if (key != "type")
            {
                description[key] = value?.DeepClone();
            }
        }

        return description;
    }

    /// <summary>A new layer from a description written by <see cref="Describe"/>, on <paramref name="device"/> (the default device when null).</summary>
    public static Module Create(JsonObject description, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(description);
        string type = (string?)description["type"] ?? throw new InvalidDataException("The layer description has no \"type\".");
        return Registry.Find(type) is { } entry
            ? entry.Create(description, device ?? Device.Default)
            : throw new InvalidDataException($"Unknown layer type '{type}' (registered: {string.Join(", ", Names)}); add it with LayerTypes.Register.");
    }

    // The layer type for the module type or its nearest registered base type.
    private static ILayerType? Find(Type? type)
    {
        var entries = Registry.Kinds;
        var cache = _byType;
        if (!ReferenceEquals(cache.Source, entries))
        {
            var byType = new Dictionary<Type, ILayerType>();
            foreach (var entry in entries)
            {
                byType[entry.ModuleType] = entry;
            }

            _byType = cache = (entries, byType);
        }

        for (; type is not null && type != typeof(object); type = type.BaseType)
        {
            if (cache.ByType.TryGetValue(type, out var entry))
            {
                return entry;
            }
        }

        return null;
    }

    // A layer type from its two functions.
    private sealed class FunctionType<T>(string name, Func<T, JsonObject> describe, Func<JsonObject, Device, T> create) : ILayerType<T> where T : Module
    {
        public string Name => name;

        public string Summary => typeof(T).Name;

        public JsonObject Describe(T module) => describe(module);

        public T Create(JsonObject description, Device device) => create(description, device);
    }

    private sealed class GuardedType(Slot slot, ILayerType app, ILayerType library) : ILayerType
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public Type ModuleType => app.ModuleType;

        public JsonObject Describe(Module module) =>
            slot.Call(() => app.Describe(module), () => library.Describe(module), (a, b) => Comparisons.Exact(a.ToJsonString(), b.ToJsonString()));

        public Module Create(JsonObject description, Device device) =>
            slot.Call(() => app.Create(description, device), () => library.Create(description, device), effects: true);
    }

    private static int I(JsonObject d, string key) => (int)d[key]!;

    private static float F(JsonObject d, string key) => (float)d[key]!;

    private static bool B(JsonObject d, string key) => (bool)d[key]!;
}
