// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// One way of measuring a choice between two paths that give the same result: each path warmed
/// once; a round times a path over enough calls to fill about 2 ms (the count from the warm call); the paths timed in
/// alternating rounds, each one's minimum per call kept; at least three rounds, and on until neither minimum has improved
/// by more than 2 % and the runtime has compiled nothing for three rounds in a row, at most 24 rounds.
/// </summary>
internal static class Faster
{
    private const double RoundMs = 2.0, Gain = 0.98;
    private const int MinRounds = 3, Settled = 3, MaxRounds = 24, MaxCalls = 64;

    /// <summary>The settled time per call of each path, in ms; <paramref name="sync"/> (a device's) runs after a path's calls, timed with them.</summary>
    public static (double First, double Second) Time(Action first, Action second, Action? sync = null)
    {
        double Once(Action run, int calls)
        {
            sync?.Invoke();
            long t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < calls; i++)
            {
                run();
            }

            sync?.Invoke();
            return Stopwatch.GetElapsedTime(t0).TotalMilliseconds / calls;
        }

        int Calls(double ms) => (int)Math.Clamp(Math.Ceiling(RoundMs / Math.Max(ms, 1e-3)), 1, MaxCalls);

        int n1 = Calls(Once(first, 1)), n2 = Calls(Once(second, 1));
        double t1 = double.MaxValue, t2 = double.MaxValue;
        int still = 0;
        for (int round = 0; round < MaxRounds; round++)
        {
            long compiled = System.Runtime.JitInfo.GetCompiledMethodCount();
            double a = Once(first, n1), b = Once(second, n2);
            bool gained = a < t1 * Gain || b < t2 * Gain || System.Runtime.JitInfo.GetCompiledMethodCount() != compiled;
            t1 = Math.Min(t1, a);
            t2 = Math.Min(t2, b);
            still = gained ? 0 : still + 1;
            if (round + 1 >= MinRounds && still >= Settled)
            {
                break;
            }
        }

        return (t1, t2);
    }
}
