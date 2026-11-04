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
public interface IOutOfMemoryPolicy : IKind
{
    /// <summary>The parts the next attempt cuts the step into (more than <see cref="OutOfMemoryFailure.Parts"/>), or 0 to give up: the failure is thrown.</summary>
    int NextParts(OutOfMemoryFailure failure);
}

/// <summary>
/// The out-of-memory policies of training steps, by name (ignoring case). The library registers <see cref="Split"/> (the
/// step cut in twice as many parts after each failure, down to one sample a part: the default) and <see cref="None"/> (no
/// recovery). A trainer takes its policy from its options, else <see cref="EnvironmentVariable"/> (IDRAK_OOM_POLICY), else
/// <see cref="Split"/>; a name nobody registered is refused with "not registered". An allocation failure is told apart from
/// any other error by its type alone (<see cref="IsAllocationFailure"/>, rule 90), and every recovery is reported to the
/// telemetry (<see cref="Report"/>, <see cref="OutOfMemoryRecovered"/>) by every trainer that recovers, whatever its own way.
/// </summary>
public static class OutOfMemoryPolicies
{
    /// <summary>The step cut in twice as many parts after each failure, down to one sample a part (the library's default).</summary>
    public const string Split = "split";

    /// <summary>No recovery: the failure reaches the caller.</summary>
    public const string None = "none";

    /// <summary>The variable that names the policy when an option does not: "IDRAK_OOM_POLICY".</summary>
    public const string EnvironmentVariable = "IDRAK_OOM_POLICY";

    private static readonly SlotTable<string, IOutOfMemoryPolicy> Table = new(nameof(OutOfMemoryPolicies), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a policy decides how a failed step runs again: falling back to another midway would cut the step another way than asked");

    static OutOfMemoryPolicies() => Overrides.AsLibraryDefaults(() =>
    {
        Register(new SplitPolicy());
        Register(new NonePolicy());
    });

    /// <summary>The policy named by <see cref="EnvironmentVariable"/>, or <see cref="Split"/> when it is not set.</summary>
    public static string DefaultName => System.Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim() : Split;

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

    /// <summary>Registers <paramref name="policy"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IOutOfMemoryPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentException.ThrowIfNullOrWhiteSpace(policy.Name);
        Table.Register(policy.Name, policy, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's policy <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The policy registered as <paramref name="name"/>, or null.</summary>
    public static IOutOfMemoryPolicy? Find(string name) => Table.Find(name);

    /// <summary>The policy registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static IOutOfMemoryPolicy Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(OutOfMemoryPolicies), name, Table.Keys,
            $"Out-of-memory policy '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with OutOfMemoryPolicies.Register.");

    /// <summary>The library's policy <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IOutOfMemoryPolicy? Default(string name) => Table.Default(name);

    /// <summary>Who registered the policy <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered policy as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(p => $"{p.Name}: {p.Summary}"));

    private sealed class SplitPolicy : IOutOfMemoryPolicy
    {
        public string Name => Split;

        public string Summary => "the step cut in twice as many parts after each failure (their gradients added up), down to one sample a part";

        public int NextParts(OutOfMemoryFailure failure) => failure.Parts >= failure.Samples ? 0 : Math.Min(failure.Samples, Math.Max(2, failure.Parts * 2));
    }

    private sealed class NonePolicy : IOutOfMemoryPolicy
    {
        public string Name => None;

        public string Summary => "no recovery: the failure reaches the caller";

        public int NextParts(OutOfMemoryFailure failure) => 0;
    }
}
