// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction.Training;

/// <summary>A training step that ran out of memory, as an <see cref="IOutOfMemoryPolicy"/> sees it.</summary>
/// <param name="Error">The allocation failure (<see cref="OutOfMemoryPolicies.IsAllocationFailure"/>).</param>
/// <param name="Samples">The step's samples.</param>
/// <param name="Parts">The parts the step was cut into when it failed (1: whole).</param>
/// <param name="Attempt">How many times this step has failed so far (1 on the first failure).</param>
/// <param name="Where">The trainer that failed ("Trainer", or a trainer's own name).</param>
public sealed record OutOfMemoryFailure(Exception Error, int Samples, int Parts, int Attempt, string Where);

/// <summary>
/// What a training step does when it runs out of memory (<see cref="OutOfMemoryPolicies"/>): into how many parts the
/// step's samples are cut for the next attempt, their gradients added up so the step's update is the whole step's; or none,
/// and the failure reaches the caller. A trainer asks it after each failure of a step, with the parts it failed in.
/// </summary>
public interface IOutOfMemoryPolicy
{
    /// <summary>The parts the next attempt cuts the step into (more than <see cref="OutOfMemoryFailure.Parts"/>), or 0 to give up: the failure is thrown.</summary>
    int NextParts(OutOfMemoryFailure failure);
}

/// <summary>
/// The out-of-memory policies of training steps, by name (ignoring case). The library registers <see cref="Split"/>
/// (the
/// step cut in twice as many parts after each failure, down to one sample a part: the default) and <see cref="None"/>
/// (no
/// recovery). A trainer takes its policy from its options, else <c>EnvironmentVariable</c> (IDRAK_OOM_POLICY), else
/// <see cref="Split"/>; a name nobody registered is refused with "not registered". An allocation failure is told apart
/// from
/// any other error by its type alone (<see cref="IsAllocationFailure"/>, rule 90), and every recovery is reported to
/// the
/// telemetry (<see cref="Report"/>, <see cref="OutOfMemoryRecovered"/>) by every trainer that recovers, whatever its
/// own way.
/// </summary>
public static class OutOfMemoryPolicies
{
    /// <summary>The step cut in twice as many parts after each failure, down to one sample a part (the library's default).</summary>
    public const string Split = "split";

    /// <summary>No recovery: the failure reaches the caller.</summary>
    public const string None = "none";

    /// <summary>
    /// Whether <paramref name="error"/> is an allocation failure, by its type: the library's <see cref="ResourceLimitExceededException"/>
    /// (a device's memory, or a limit set on <see cref="ComputeResources"/>) or the runtime's <see cref="OutOfMemoryException"/>
    /// (host memory); an <see cref="AggregateException"/> (work spread over threads) when every error inside it is one.
    /// </summary>
    public static bool IsAllocationFailure(Exception? error) => error switch
    {
        ResourceLimitExceededException or OutOfMemoryException => true,
        AggregateException aggregate => aggregate.InnerExceptions.Count > 0 && aggregate.InnerExceptions.All(IsAllocationFailure),
        _ => false,
    };

    /// <summary>Reports a recovery to the telemetry (<see cref="TelemetryLevel.Devices"/>, which the built-in loggers always show).</summary>
    public static void Report(in OutOfMemoryRecovered recovered) => Telemetry.OutOfMemoryRecovered(in recovered);
}

internal sealed class SplitPolicy : IOutOfMemoryPolicy
{
    public string Name => OutOfMemoryPolicies.Split;

    public int NextParts(OutOfMemoryFailure failure) => failure.Parts >= failure.Samples ? 0 : Math.Min(failure.Samples, Math.Max(2, failure.Parts * 2));
}

internal sealed class NonePolicy : IOutOfMemoryPolicy
{
    public string Name => OutOfMemoryPolicies.None;

    public int NextParts(OutOfMemoryFailure failure) => 0;
}
