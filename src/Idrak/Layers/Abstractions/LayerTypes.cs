// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Layers.Abstractions;

/// <summary>
/// The layer types a <c>GraphModule</c> can hold, by the "type" name its JSON writes: how to describe a layer's
/// settings and how to create the layer again from them (its weights are loaded separately). The standard layers are
/// registered here ("linear", "conv2d", "batchnorm", ..., and "sequential" for blocks of them); add your own with
/// <see cref="Register{T}"/>, so a graph holding them survives <c>GraphModule.ToJson</c>,
/// <c>GraphModule.FromJson</c> and model packages.
/// </summary>
public static class LayerTypes
{
    private sealed record Entry(string Name, Type Type, Func<Module, JsonObject> Describe, Func<JsonObject, Device, Module> Create);

    // Describing is one call; creating makes the layer once (its weights too), so under Shadow only describing is compared.
    private static readonly SlotTable<string, Entry> Registry = new(nameof(LayerTypes), (slot, app, library) => app with
    {
        Describe = module => slot.Call(() => app.Describe(module), () => library.Describe(module), (a, b) => Comparisons.Exact(a.ToJsonString(), b.ToJsonString())),
        Create = (description, device) => slot.Call(() => app.Create(description, device), () => library.Create(description, device), effects: true),
    }, StringComparer.Ordinal);

    // The entries by module type, from the registry's current entries (a type registered under a later name wins).
    private static (IReadOnlyList<Entry> Source, Dictionary<Type, Entry> ByType) _byType = ([], []);

    // Idrak's layers (linear, conv2d, attention, ..., sequential) are registered before the first use.
    static LayerTypes() => Overrides.AsLibraryDefaults(LibraryLayerTypes.RegisterAll);   // the built-in layer types, on first use

    /// <summary>
    /// Registers the layer type <paramref name="type"/> for modules of type <typeparamref name="T"/>:
    /// <paramref name="describe"/> returns the layer's settings as JSON (the "type" key is added), <paramref name="create"/>
    /// makes a new layer on the given device from that JSON. Under a built-in name it overrides the library's, which stays
    /// behind it as its fallback (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    public static void Register<T>(string type, Func<T, JsonObject> describe, Func<JsonObject, Device, T> create) where T : Module
    {
        ArgumentException.ThrowIfNullOrEmpty(type);
        ArgumentNullException.ThrowIfNull(describe);
        ArgumentNullException.ThrowIfNull(create);
        Registry.Register(type, new Entry(type, typeof(T), m => describe((T)m), (d, device) => create(d, device)), create);
    }

    /// <summary>Removes the app's layer type <paramref name="type"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string type) => Registry.Unregister(type);

    /// <summary>The registered layer type names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>Who registered the layer type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string type) => Registry.Origin(type);

    /// <summary>Whether the library has a layer type <paramref name="type"/> of its own, whatever an app registered over it.</summary>
    public static bool HasDefault(string type) => Registry.HasDefault(type);

    /// <summary>
    /// What happens when the app's layer type <paramref name="type"/> fails (<see cref="SlotPolicy.FallBack"/> to the
    /// library's unless set): describing falls back per call, creating when the layer is made.
    /// </summary>
    public static void SetPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(type, policy, shadowRate);

    /// <summary>Whether <paramref name="module"/>'s type (or a base type) is registered.</summary>
    public static bool CanDescribe(Module module) => Find(module.GetType()) is not null;

    /// <summary>The layer's settings as JSON, with its "type" name first.</summary>
    public static JsonObject Describe(Module module)
    {
        ArgumentNullException.ThrowIfNull(module);
        var entry = Find(module.GetType()) ?? throw new NotSupportedException(
            $"{module.GetType().Name} cannot be described (layer types: {string.Join(", ", Names)}); add it with LayerTypes.Register.");
        var description = new JsonObject { ["type"] = entry.Name };
        foreach (var (key, value) in entry.Describe(module))
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

    // The entry for the type or its nearest registered base type.
    private static Entry? Find(Type? type)
    {
        var entries = Registry.Values;
        var cache = _byType;
        if (!ReferenceEquals(cache.Source, entries))
        {
            var byType = new Dictionary<Type, Entry>();
            foreach (var entry in entries)
            {
                byType[entry.Type] = entry;
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

    private static int I(JsonObject d, string key) => (int)d[key]!;

    private static float F(JsonObject d, string key) => (float)d[key]!;

    private static bool B(JsonObject d, string key) => (bool)d[key]!;
}
