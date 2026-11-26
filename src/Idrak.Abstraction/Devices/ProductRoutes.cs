// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// The route products of described weights take: each device's described lane (one lane that reads any type from its
/// description, <see cref="TypeDescription"/>), its variant at each shape measured once on the device. Each device records
/// what it chose for each shape (<see cref="Choices"/>), so a benchmark can say which variant ran.
/// </summary>
public static class ProductRoutes
{
    private static readonly ConcurrentDictionary<string, string> s_choices = new(StringComparer.Ordinal);

    /// <summary>What each device chose: "device: product" to the variant it measured fastest.</summary>
    public static IReadOnlyDictionary<string, string> Choices => new Dictionary<string, string>(s_choices, StringComparer.Ordinal);

    /// <summary>Records the variant <paramref name="device"/> chose for <paramref name="product"/> (a backend, once per shape).</summary>
    public static void Chose(string device, string product, string lane)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(product);
        ArgumentNullException.ThrowIfNull(lane);
        s_choices[$"{device}: {product}"] = lane;
    }

    /// <summary>Forgets the recorded choices (a benchmark between its rows, tests).</summary>
    public static void ForgetChoices() => s_choices.Clear();
}
