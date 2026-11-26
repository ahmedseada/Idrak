// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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
/// Proves a task folder: the layout a code extractor writes (<c>task.json</c> with the prompt,
/// the base folder, the setup and check commands and their variables; <c>workspace/</c>, <c>verify/</c>, <c>solution/</c>)
/// is run without a model, once as it starts and once with its solution. Idrak.Data runs no commands itself: a package
/// that does (Idrak.Reinforcement.Plugins' coding tasks) brings one.
/// </summary>
public interface ITaskVerifier
{
    /// <summary>The verifier's name ("coding").</summary>
    string Name { get; }

    /// <summary>Runs the check of the task in <paramref name="taskFolder"/> on its workspace and with its solution, using <paramref name="workFolder"/> (created) for both runs.</summary>
    /// <param name="taskFolder">The task's folder (its <c>task.json</c>).</param>
    /// <param name="workFolder">An empty or missing folder the runs may fill; the caller removes it.</param>
    /// <param name="cancellationToken">Stops the runs.</param>
    Task<TaskProof> ProveAsync(string taskFolder, string workFolder, CancellationToken cancellationToken = default);
}

/// <summary>The word for no verifier (<c>idrak data code --verifier none</c>, <c>CodeExtractorOptions.Verifier</c>).</summary>
public static class TaskVerifiers
{
    /// <summary>The name that asks for no verifier: drafts are written as unproven.</summary>
    public const string None = "none";
}
