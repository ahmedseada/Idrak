// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Modules;

/// <summary>
/// Adds the layers of one network step to <paramref name="builder"/>, reading the step's arguments from
/// <paramref name="arguments"/>, and returns the builder. Register it with <see cref="NetworkOps.Register"/>.
/// </summary>
public delegate INetworkBuilder NetworkOp(INetworkBuilder builder, NetworkOpArguments arguments);

/// <summary>
/// The arguments of one network step: the JSON object the network builder writes for it (<c>NetworkBuilder.ToJson</c>:
/// <c>{"op": "linear", "out": 10, "bias": true}</c>), read with the typed helpers.
/// </summary>
public sealed class NetworkOpArguments
{
    /// <summary>The arguments in <paramref name="step"/>, the step's JSON object with its "op".</summary>
    public NetworkOpArguments(JsonObject step)
    {
        ArgumentNullException.ThrowIfNull(step);
        Json = step;
    }

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
/// The network steps <c>Network.FromJson</c> and <see cref="INetworkBuilder.Op"/> know, by the "op" name the network
/// builder writes (<c>NetworkBuilder.ToJson</c>). Idrak registers one step per builder layer ("linear", "relu",
/// "conv2d", "transformer", ...); add your own with <see cref="Register"/>, so a builder using them can still be written
/// to JSON, saved in a model package and read back. A step sees the builder through <see cref="INetworkBuilder"/>.
/// </summary>
public static class NetworkOps
{
    private static readonly Dictionary<string, NetworkOp> Registry = new(StringComparer.Ordinal);

    // Idrak's steps (one per builder layer: linear, relu, conv2d, transformer, ...) are registered before the first use.
    static NetworkOps() => LibraryDefaults.Ensure(typeof(NetworkOps));

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
    public static NetworkOp Get(string name)
    {
        lock (Registry)
        {
            if (Registry.TryGetValue(name, out var op))
            {
                return op;
            }
        }

        throw new NotSupportedException($"No network step '{name}' is registered ({string.Join(", ", Names)}); add it with NetworkOps.Register.");
    }

    /// <summary>Whether a step is registered as <paramref name="name"/>.</summary>
    public static bool Contains(string name)
    {
        lock (Registry)
        {
            return Registry.ContainsKey(name);
        }
    }
}
