// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Layers;

/// <summary>
/// The layer types a <see cref="GraphModule"/> can hold, by the "type" name its JSON writes: how to describe a layer's
/// settings and how to create the layer again from them (its weights are loaded separately). The standard layers are
/// registered here ("linear", "conv2d", "batchnorm", ..., and "sequential" for blocks of them); add your own with
/// <see cref="Register{T}"/>, so a graph holding them survives <see cref="GraphModule.ToJson"/>,
/// <see cref="GraphModule.FromJson"/> and model packages.
/// </summary>
public static class LayerTypes
{
    private sealed record Entry(string Name, Type Type, Func<Module, JsonObject> Describe, Func<JsonObject, Device, Module> Create);

    private static readonly Dictionary<string, Entry> Registry = new(StringComparer.Ordinal);
    private static readonly Dictionary<Type, Entry> ByType = [];

    static LayerTypes()
    {
        Register<Linear>("linear", l => new() { ["in"] = l.InFeatures, ["out"] = l.OutFeatures, ["bias"] = l.Bias is not null },
            (d, device) => new Linear(I(d, "in"), I(d, "out"), B(d, "bias"), device));
        Register<Conv2d>("conv2d", c => new()
        {
            ["in"] = c.InChannels, ["out"] = c.OutChannels, ["kernel"] = c.KernelSize, ["stride"] = c.Stride, ["padding"] = c.Padding, ["bias"] = c.Bias is not null,
        }, (d, device) => new Conv2d(I(d, "in"), I(d, "out"), I(d, "kernel"), I(d, "stride"), I(d, "padding"), B(d, "bias"), device));
        Register<BatchNorm>("batchnorm", b => new() { ["channels"] = b.Channels, ["momentum"] = b.Momentum, ["epsilon"] = b.Epsilon },
            (d, device) => new BatchNorm(I(d, "channels"), F(d, "momentum"), F(d, "epsilon"), device));
        Register<LayerNorm>("layernorm", n => new() { ["features"] = n.Features, ["epsilon"] = n.Epsilon },
            (d, device) => new LayerNorm(I(d, "features"), F(d, "epsilon"), device));
        Register<Embedding>("embedding", e => new() { ["vocabulary"] = e.Vocabulary, ["dim"] = e.Dim },
            (d, device) => new Embedding(I(d, "vocabulary"), I(d, "dim"), device));
        Register<PositionalEncoding>("positional", p => new() { ["maxLength"] = p.MaxLength, ["dim"] = p.Dim },
            (d, device) => new PositionalEncoding(I(d, "maxLength"), I(d, "dim"), device));
        Register<MultiHeadAttention>("attention", a => new() { ["dim"] = a.Dim, ["heads"] = a.Heads, ["causal"] = a.Causal },
            (d, device) => new MultiHeadAttention(I(d, "dim"), I(d, "heads"), B(d, "causal"), 0f, device));
        Register<TransformerEncoderLayer>("transformer", t => t.Children().ToList() is [_, MultiHeadAttention a, _, Linear ff, ..]
            ? new() { ["dim"] = t.Dim, ["heads"] = a.Heads, ["ffDim"] = ff.OutFeatures, ["causal"] = a.Causal }
            : throw new NotSupportedException("a transformer layer without its standard sublayers cannot be described"),
            (d, device) => new TransformerEncoderLayer(I(d, "dim"), I(d, "heads"), I(d, "ffDim"), 0f, B(d, "causal"), device));
        Register<LSTM>("lstm", r => new() { ["in"] = r.InputSize, ["hidden"] = r.HiddenSize, ["sequences"] = r.ReturnSequences },
            (d, device) => new LSTM(I(d, "in"), I(d, "hidden"), B(d, "sequences"), device));
        Register<GRU>("gru", r => new() { ["in"] = r.InputSize, ["hidden"] = r.HiddenSize, ["sequences"] = r.ReturnSequences },
            (d, device) => new GRU(I(d, "in"), I(d, "hidden"), B(d, "sequences"), device));
        Register<MaxPool2d>("maxpool2d", m => new() { ["kernel"] = m.KernelSize, ["stride"] = m.Stride, ["padding"] = m.Padding },
            (d, _) => new MaxPool2d(I(d, "kernel"), I(d, "stride"), I(d, "padding")));
        Register<GlobalAveragePool2d>("globalavgpool2d", _ => [], (_, _) => new GlobalAveragePool2d());
        Register<Flatten>("flatten", _ => [], (_, _) => new Flatten());
        Register<ReLU>("relu", _ => [], (_, _) => new ReLU());
        Register<Tanh>("tanh", _ => [], (_, _) => new Tanh());
        Register<Sigmoid>("sigmoid", _ => [], (_, _) => new Sigmoid());
        Register<GELU>("gelu", _ => [], (_, _) => new GELU());
        Register<Softmax>("softmax", _ => [], (_, _) => new Softmax());
        Register<Dropout>("dropout", d => new() { ["p"] = d.Probability }, (d, _) => new Dropout(F(d, "p")));

        // A block: the network builder's description when it built the block (so registered builder steps, lambdas
        // included, survive), otherwise each layer's own description.
        Register<Sequential>("sequential", s => Network.ArchitectureOf(s) is { } network
            ? new() { ["network"] = network.DeepClone() }
            : new() { ["layers"] = new JsonArray([.. s.Select(m => (JsonNode)Describe(m))]) },
            (d, device) => d["network"] is JsonObject network
                ? Network.FromJson(network).OnDevice(device).Build()
                : new Sequential([.. d["layers"]!.AsArray().Select(l => Create(l!.AsObject(), device))]));
    }

    /// <summary>
    /// Registers (or replaces) the layer type <paramref name="type"/> for modules of type <typeparamref name="T"/>:
    /// <paramref name="describe"/> returns the layer's settings as JSON (the "type" key is added), <paramref name="create"/>
    /// makes a new layer on the given device from that JSON.
    /// </summary>
    public static void Register<T>(string type, Func<T, JsonObject> describe, Func<JsonObject, Device, T> create) where T : Module
    {
        ArgumentException.ThrowIfNullOrEmpty(type);
        ArgumentNullException.ThrowIfNull(describe);
        ArgumentNullException.ThrowIfNull(create);
        var entry = new Entry(type, typeof(T), m => describe((T)m), (d, device) => create(d, device));
        lock (Registry)
        {
            if (Registry.Remove(type, out var old))
            {
                ByType.Remove(old.Type);
            }

            if (ByType.Remove(typeof(T), out var other))
            {
                Registry.Remove(other.Name);
            }

            Registry[type] = entry;
            ByType[typeof(T)] = entry;
        }
    }

    /// <summary>Removes the layer type <paramref name="type"/>; returns whether it was registered.</summary>
    public static bool Unregister(string type)
    {
        lock (Registry)
        {
            if (!Registry.Remove(type, out var entry))
            {
                return false;
            }

            ByType.Remove(entry.Type);
            return true;
        }
    }

    /// <summary>The registered layer type names.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

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
        Entry? entry;
        lock (Registry)
        {
            Registry.TryGetValue(type, out entry);
        }

        return entry is null
            ? throw new InvalidDataException($"Unknown layer type '{type}' (registered: {string.Join(", ", Names)}); add it with LayerTypes.Register.")
            : entry.Create(description, device ?? Device.Default);
    }

    // The entry for the type or its nearest registered base type.
    private static Entry? Find(Type? type)
    {
        lock (Registry)
        {
            for (; type is not null && type != typeof(object); type = type.BaseType)
            {
                if (ByType.TryGetValue(type, out var entry))
                {
                    return entry;
                }
            }

            return null;
        }
    }

    private static int I(JsonObject d, string key) => (int)d[key]!;

    private static float F(JsonObject d, string key) => (float)d[key]!;

    private static bool B(JsonObject d, string key) => (bool)d[key]!;
}
