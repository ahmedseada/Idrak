// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Layers;

/// <summary>
/// Adds the layers of one network step to <paramref name="builder"/>, reading the step's arguments from
/// <paramref name="arguments"/>, and returns the builder. Register it with <see cref="NetworkOps.Register"/>.
/// </summary>
public delegate NetworkBuilder NetworkOp(NetworkBuilder builder, NetworkOpArguments arguments);

/// <summary>
/// The arguments of one network step: the JSON object <see cref="NetworkBuilder.ToJson"/> writes for it
/// (<c>{"op": "linear", "out": 10, "bias": true}</c>), read with the typed helpers.
/// </summary>
public sealed class NetworkOpArguments
{
    internal NetworkOpArguments(JsonObject step) => Json = step;

    /// <summary>The step's name ("op").</summary>
    public string Op => (string)Json["op"]!;

    /// <summary>The whole step, for arguments the helpers do not cover.</summary>
    public JsonObject Json { get; }

    /// <summary>Whether the step has the argument <paramref name="key"/>.</summary>
    public bool Has(string key) => Json[key] is not null;

    /// <summary>The integer argument <paramref name="key"/>; it must be present.</summary>
    public int Int(string key) => (int)Required(key);

    /// <summary>The integer argument <paramref name="key"/>, or null when the step leaves it out.</summary>
    public int? OptionalInt(string key) => (int?)Json[key];

    /// <summary>The number argument <paramref name="key"/>; it must be present.</summary>
    public float Float(string key) => (float)Required(key);

    /// <summary>The true/false argument <paramref name="key"/>; it must be present.</summary>
    public bool Bool(string key) => (bool)Required(key);

    /// <summary>The number array argument <paramref name="key"/> (for example per-channel means); it must be present.</summary>
    public float[] Floats(string key) => [.. Required(key).AsArray().Select(v => (float)v!)];

    /// <summary>The integer array argument <paramref name="key"/> (for example a shape); it must be present.</summary>
    public int[] Ints(string key) => [.. Required(key).AsArray().Select(v => (int)v!)];

    private JsonNode Required(string key) => Json[key] ?? throw new InvalidDataException($"The network step '{Op}' needs the argument '{key}'.");
}

/// <summary>
/// The network steps <see cref="Network.FromJson"/> and <see cref="NetworkBuilder.Op"/> know, by the "op" name
/// <see cref="NetworkBuilder.ToJson"/> writes. Every builder layer is registered ("linear", "relu", "conv2d",
/// "transformer", ...); add your own with <see cref="Register"/>, so a builder using them can still be written to JSON,
/// saved in a model package and read back.
/// </summary>
public static class NetworkOps
{
    private static readonly Dictionary<string, NetworkOp> Registry = new(StringComparer.Ordinal)
    {
        ["linear"] = (b, a) => b.Linear(a.Int("out"), a.Bool("bias")),
        ["relu"] = (b, _) => b.ReLU(),
        ["tanh"] = (b, _) => b.Tanh(),
        ["sigmoid"] = (b, _) => b.Sigmoid(),
        ["gelu"] = (b, _) => b.GELU(),
        ["softmax"] = (b, _) => b.Softmax(),
        ["dropout"] = (b, a) => b.Dropout(a.Float("p")),
        ["batchnorm"] = (b, a) => b.BatchNorm(a.Float("momentum"), a.Float("epsilon")),
        ["layernorm"] = (b, a) => b.LayerNorm(a.Float("epsilon")),
        ["normalize"] = (b, a) => b.Normalize(a.Floats("mean"), a.Floats("std")),
        ["conv2d"] = (b, a) => b.Conv2d(a.Int("out"), a.Int("kernel"), a.Int("stride"), a.Int("padding"), a.Bool("bias")),
        ["maxpool2d"] = (b, a) => b.MaxPool2d(a.Int("kernel"), a.OptionalInt("stride"), a.Int("padding")),
        ["globalavgpool2d"] = (b, _) => b.GlobalAveragePool2d(),
        ["flatten"] = (b, _) => b.Flatten(),
        ["embedding"] = (b, a) => b.Embedding(a.Int("vocabulary"), a.Int("dim")),
        ["positional"] = (b, a) => b.PositionalEncoding(a.OptionalInt("maxLength")),
        ["transformer"] = (b, a) => b.TransformerEncoderLayer(a.Int("heads"), a.OptionalInt("ffDim"), a.Float("dropout"), a.Bool("causal")),
        ["attention"] = (b, a) => b.MultiHeadAttention(a.Int("heads"), a.Bool("causal"), a.Float("dropout")),
        ["lstm"] = (b, a) => b.LSTM(a.Int("hidden"), a.Bool("returnSequences")),
        ["gru"] = (b, a) => b.GRU(a.Int("hidden"), a.Bool("returnSequences")),
        ["meanOverTime"] = (b, _) => b.MeanOverTime(),
        ["lastStep"] = (b, _) => b.LastStep(),
        ["firstStep"] = (b, _) => b.FirstStep(),
        ["reshape"] = (b, a) => b.Reshape(a.Ints("shape")),
    };

    /// <summary>Registers (or replaces) the network step <paramref name="name"/>.</summary>
    public static void Register(string name, NetworkOp op)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(op);
        lock (Registry)
        {
            Registry[name] = op;
        }
    }

    /// <summary>Removes the network step <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.Remove(name);
        }
    }

    /// <summary>The registered step names.</summary>
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

    /// <summary>The step registered as <paramref name="name"/>.</summary>
    public static NetworkOp Get(string name) =>
        TryGet(name) ?? throw new NotSupportedException($"No network step '{name}' is registered ({string.Join(", ", Names)}); add it with NetworkOps.Register.");

    internal static NetworkOp? TryGet(string name)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(name, out var op) ? op : null;
        }
    }
}
