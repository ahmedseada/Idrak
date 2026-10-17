// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>Draws beyond <see cref="Random"/>'s own, from a seeded generator so a seed gives the same values everywhere.</summary>
public static class RandomDraws
{
    /// <summary>A standard normal draw (Box-Muller: two uniform draws a value).</summary>
    public static double NextGaussian(this Random random)
    {
        ArgumentNullException.ThrowIfNull(random);
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }
}
