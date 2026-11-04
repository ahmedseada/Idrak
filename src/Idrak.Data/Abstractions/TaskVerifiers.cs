// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Data.Abstractions;

/// <summary>
/// What a <see cref="ITaskVerifier"/> found about a task folder: whether its check fails on the workspace (else the task
/// tests nothing) and passes with its solution (else it cannot be solved), with the commands' output.
/// </summary>
/// <param name="StartFails">The check fails on the starting workspace.</param>
/// <param name="SolutionPasses">The check passes with the solution laid over the workspace.</param>
/// <param name="Output">Both runs' setup and check output.</param>
public sealed record TaskProof(bool StartFails, bool SolutionPasses, string Output)
{
    /// <summary>Whether the task is proven: its check fails on the workspace and passes with the solution.</summary>
    public bool Proven => StartFails && SolutionPasses;

    /// <summary>Why a task that is not proven is not, in words ("the check passes on the workspace", …); null when proven.</summary>
    public string? Reason => Proven ? null
        : !StartFails && !SolutionPasses ? "the check passes on the workspace and fails with the solution"
        : !StartFails ? "the check passes on the workspace (the task tests nothing)"
        : "the check fails with the solution (the task cannot be solved)";
}

/// <summary>
/// Proves a task folder (<see cref="TaskVerifiers"/>): the layout a code extractor writes (<c>task.json</c> with the prompt,
/// the base folder, the setup and check commands and their variables; <c>workspace/</c>, <c>verify/</c>, <c>solution/</c>)
/// is run without a model, once as it starts and once with its solution. Idrak.Data runs no commands itself: a package
/// that does (Idrak.Reinforcement.Plugins' coding tasks) registers one.
/// </summary>
public interface ITaskVerifier : IKind
{
    /// <summary>Runs the check of the task in <paramref name="taskFolder"/> on its workspace and with its solution, using <paramref name="workFolder"/> (created) for both runs.</summary>
    /// <param name="taskFolder">The task's folder (its <c>task.json</c>).</param>
    /// <param name="workFolder">An empty or missing folder the runs may fill; the caller removes it.</param>
    /// <param name="cancellationToken">Stops the runs.</param>
    Task<TaskProof> ProveAsync(string taskFolder, string workFolder, CancellationToken cancellationToken = default);
}

/// <summary>
/// The task verifiers by name (ignoring case), for <c>idrak data code --verifier</c> and <c>CodeExtractorOptions.Verifier</c>.
/// The library registers none (it runs no commands); Idrak.Reinforcement.Plugins registers its coding tasks' ("coding"). With
/// none registered, or "none" asked, a code extractor writes its drafts as unproven, never as tasks. A name nobody registered
/// is refused with "not registered".
/// </summary>
public static class TaskVerifiers
{
    /// <summary>The name that asks for no verifier: drafts are written as unproven.</summary>
    public const string None = "none";

    private static readonly SlotTable<string, ITaskVerifier> Table = new(nameof(TaskVerifiers), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a proof runs commands with effects in a work folder: a failed proof is a rejected task, not retried on another verifier");

    /// <summary>Registers <paramref name="verifier"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITaskVerifier verifier)
    {
        ArgumentNullException.ThrowIfNull(verifier);
        ArgumentException.ThrowIfNullOrWhiteSpace(verifier.Name);
        if (verifier.Name.Equals(None, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"'{None}' asks for no verifier; a verifier is registered under another name.", nameof(verifier));
        }

        Table.Register(verifier.Name, verifier, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's verifier <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The verifier registered as <paramref name="name"/>, or null.</summary>
    public static ITaskVerifier? Find(string name) => Table.Find(name);

    /// <summary>The verifier registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static ITaskVerifier Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(TaskVerifiers), name, Table.Keys,
            $"Task verifier '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); add it with TaskVerifiers.Register, or ask for '{None}'.");

    /// <summary>
    /// The verifier a run uses for <paramref name="name"/>: none for "none", the registered one of that name, or (for null)
    /// the first registered, or none when nobody registered one.
    /// </summary>
    /// <exception cref="NotSupportedException">A name nobody registered.</exception>
    public static ITaskVerifier? Choose(string? name) => name is null ? Table.Values.FirstOrDefault()
        : name.Equals(None, StringComparison.OrdinalIgnoreCase) ? null
        : Get(name);

    /// <summary>The library's verifier <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITaskVerifier? Default(string name) => Table.Default(name);

    /// <summary>Who registered the verifier <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered verifier as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(v => $"{v.Name}: {v.Summary}"));
}
