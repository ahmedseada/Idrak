// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// How a product of weights stored in a type other than float32 picks its kernel: <see cref="Name"/>, today's routes, one kernel per format by the format's name (bfloat16, int8 and the
/// stored formats each their own operations and kernels); <see cref="Generic"/>, one product per device that widens any type
/// from its description (<see cref="ElementType.Float"/>, the integer bits, <see cref="ElementType.Scaled"/>), with no
/// kernel of a type's own; <see cref="Measured"/>, the device's capability, then the description, then a measurement at the
/// shape against the generic product, kept per shape: a named kernel is taken only where it measured faster.
/// </summary>
public enum ProductRoute
{
    /// <summary>Today's kernels, chosen by the format's name.</summary>
    Name,

    /// <summary>One product per device, widening any type from its description; no kernel of a type's own.</summary>
    Generic,

    /// <summary>The faster of the generic product and the format's own kernel, measured once per shape.</summary>
    Measured,
}

/// <summary>
/// The lanes a product of weights read where they are stored may take under <see cref="ProductRoute.Measured"/>
/// (<see cref="ProductRoutes.Lanes"/>): the described lane (one lane that reads any type from its description,
/// <see cref="ElementType.Description"/>), the kernels written for a type, or both, the faster at each shape measured once.
/// </summary>
public enum RouteLanes
{
    /// <summary>The described lane and the written kernels as candidates, the faster at each shape measured once.</summary>
    All,

    /// <summary>The described lane alone (a type it does not read takes the written kernels).</summary>
    Described,

    /// <summary>The written kernels alone, the described lane not a candidate.</summary>
    Written,
}

/// <summary>
/// The route products of typed weights take (<see cref="ProductRoute"/>): <see cref="EnvironmentVariable"/> chooses it for
/// the process (read once, before the first product), <see cref="Current"/> sets it in code. Under
/// <see cref="ProductRoute.Measured"/> each device records what it chose for each shape (<see cref="Choices"/>), so a
/// benchmark can say which kernel ran.
/// </summary>
public static class ProductRoutes
{
    /// <summary>The variable that chooses the route: "name" (the default), "generic" or "measured".</summary>
    public const string EnvironmentVariable = "IDRAK_ROUTES";

    private static int s_route = -1;
    private static readonly ConcurrentDictionary<string, string> s_choices = new(StringComparer.Ordinal);

    /// <summary>The route products take now: <see cref="EnvironmentVariable"/> until set here.</summary>
    /// <exception cref="InvalidOperationException">The variable holds none of the routes' names.</exception>
    public static ProductRoute Current
    {
        get
        {
            int route = Volatile.Read(ref s_route);
            if (route < 0)
            {
                route = (int)Parse(Environment.GetEnvironmentVariable(EnvironmentVariable));
                Volatile.Write(ref s_route, route);
            }

            return (ProductRoute)route;
        }
        set => Volatile.Write(ref s_route, (int)value);
    }

    /// <summary>The route <paramref name="text"/> names: "name" (or nothing), "generic" or "measured", ignoring case.</summary>
    /// <exception cref="InvalidOperationException">It names none of them.</exception>
    public static ProductRoute Parse(string? text) => text?.Trim().ToLowerInvariant() switch
    {
        null or "" or "name" => ProductRoute.Name,
        "generic" => ProductRoute.Generic,
        "measured" => ProductRoute.Measured,
        _ => throw new InvalidOperationException($"{EnvironmentVariable} is '{text}': name, generic or measured."),
    };

    /// <summary>The variable that chooses the lanes under the measured route: "all" (the default), "described" or "written".</summary>
    public const string LanesVariable = "IDRAK_ROUTE_LANES";

    private static int s_lanes = -1;

    /// <summary>The lanes products may take under <see cref="ProductRoute.Measured"/>: <see cref="LanesVariable"/> until set here.</summary>
    /// <exception cref="InvalidOperationException">The variable holds none of the choices' names.</exception>
    public static RouteLanes Lanes
    {
        get
        {
            int lanes = Volatile.Read(ref s_lanes);
            if (lanes < 0)
            {
                lanes = (int)ParseLanes(Environment.GetEnvironmentVariable(LanesVariable));
                Volatile.Write(ref s_lanes, lanes);
            }

            return (RouteLanes)lanes;
        }
        set => Volatile.Write(ref s_lanes, (int)value);
    }

    /// <summary>The lanes <paramref name="text"/> names: "all" (or nothing), "described" or "written", ignoring case.</summary>
    /// <exception cref="InvalidOperationException">It names none of them.</exception>
    public static RouteLanes ParseLanes(string? text) => text?.Trim().ToLowerInvariant() switch
    {
        null or "" or "all" => RouteLanes.All,
        "described" => RouteLanes.Described,
        "written" => RouteLanes.Written,
        _ => throw new InvalidOperationException($"{LanesVariable} is '{text}': all, described or written."),
    };

    /// <summary>What each device chose under <see cref="ProductRoute.Measured"/>: "device: product" to the kernel's lane.</summary>
    public static IReadOnlyDictionary<string, string> Choices => new Dictionary<string, string>(s_choices, StringComparer.Ordinal);

    /// <summary>Records the lane <paramref name="device"/> chose for <paramref name="product"/> (a backend, once per shape).</summary>
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
