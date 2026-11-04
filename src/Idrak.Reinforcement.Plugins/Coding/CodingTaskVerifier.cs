// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Abstractions;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// The coding tasks' proof as a task verifier (<see cref="TaskVerifiers"/>, "coding"): a task folder a code extractor wrote
/// is loaded as a <see cref="CodingTask"/> and checked by <see cref="CodingVerifier.CheckTaskAsync"/>, in the sandbox and
/// under the limits of its options, its setup and verification commands with the task's own variables. Registered by
/// <see cref="CodingPlugin"/>; <c>idrak data code</c> uses it unless asked for another or none.
/// </summary>
/// <param name="options">The commands' tools and time limit; null for the defaults.</param>
public sealed class CodingTaskVerifier(CodingVerifierOptions? options = null) : ITaskVerifier
{
    /// <summary>Its name in <see cref="TaskVerifiers"/>: "coding".</summary>
    public const string VerifierName = "coding";

    private readonly CodingVerifier _verifier = new(options);

    /// <inheritdoc/>
    public string Name => VerifierName;

    /// <inheritdoc/>
    public string Summary => "the coding tasks' check (agent-check): the task's commands fail on its workspace and pass with its solution";

    /// <inheritdoc/>
    public async Task<TaskProof> ProveAsync(string taskFolder, string workFolder, CancellationToken cancellationToken = default)
    {
        var task = CodingTask.Load(taskFolder);
        var check = await _verifier.CheckTaskAsync(task, workFolder, cancellationToken).ConfigureAwait(false);
        return new TaskProof(check.StartFails, check.SolutionPasses, check.Output);
    }
}
