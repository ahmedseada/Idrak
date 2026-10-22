// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Text.Generation;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>How a coding task's run ended (<see cref="CodingRun.Outcome"/>).</summary>
public enum CodingOutcome
{
    /// <summary>The agent finished and every verification command succeeded (or the task has none).</summary>
    Passed,

    /// <summary>A verification command failed.</summary>
    Failed,

    /// <summary>The agent used all its rounds or time; verification still ran and failed.</summary>
    OutOfBudget,

    /// <summary>A setup command failed: the task, not the agent, is broken.</summary>
    SetupFailed,

    /// <summary>The model failed (for example the server was unreachable).</summary>
    Error,
}

/// <summary>How <see cref="CodingVerifier"/> runs a task's setup and verification commands.</summary>
public sealed record CodingVerifierOptions
{
    /// <summary>What the commands may do (the programs allowed, the sandbox, its limits); their time limit is <see cref="Timeout"/>.</summary>
    public CodingToolOptions Tools { get; init; } = new();

    /// <summary>Time limit of each setup and verification command (10 minutes by default).</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromMinutes(10);
}

/// <summary>What a task's setup or verification commands did (<see cref="CodingVerifier.SetupAsync"/>, <see cref="CodingVerifier.VerifyAsync"/>).</summary>
/// <param name="Passed">Every command exited with 0 (true for none).</param>
/// <param name="Commands">The task's commands.</param>
/// <param name="PassedCommands">The commands that passed before the first that failed (the commands run in order, and stop there).</param>
/// <param name="Output">Each command run, its exit and its output ("$ command", "exit code 0", ...).</param>
public sealed record CodingVerification(bool Passed, int Commands, int PassedCommands, string Output)
{
    /// <summary>The share of the commands that passed (for none: 1 when passed).</summary>
    public float Fraction => Commands == 0 ? (Passed ? 1f : 0f) : (float)PassedCommands / Commands;
}

/// <summary>What <see cref="CodingVerifier.CheckTaskAsync"/> found.</summary>
/// <param name="StartFails">Verification fails on the starting files (else the task tests nothing).</param>
/// <param name="SolutionPasses">Verification passes with the task's solution (else it cannot be solved).</param>
/// <param name="Output">Both runs' setup and verification output.</param>
public sealed record CodingTaskCheck(bool StartFails, bool SolutionPasses, string Output);

/// <summary>One run of the agent on a coding task: the agent's conversation, how it ended, and the verification's output.</summary>
/// <param name="Task">The task.</param>
/// <param name="Agent">The agent's loop (<see cref="CodingAgent.RunAsync"/>): messages, tools, figures.</param>
/// <param name="Outcome">How it ended.</param>
/// <param name="VerifyOutput">The verification commands' output (or the setup's, or the error).</param>
public sealed record CodingRun(CodingTask Task, CodingAgentRun Agent, CodingOutcome Outcome, string VerifyOutput)
{
    /// <summary>The run as one JSON line: the conversation in the chat format with the task, the outcome and the figures as extra fields.</summary>
    public JsonObject ToJson()
    {
        var json = ChatJson.Transcript(Agent.Messages, Agent.Tools, Agent.Think);
        json["task"] = Task.Id;
        if (Task.Language is { } language)
        {
            json["language"] = language;
        }

        json["outcome"] = Outcome.ToString();
        Agent.AddFigures(json);
        return json;
    }
}

/// <summary>
/// Runs coding tasks: their setup and verification commands (through <see cref="CodingTools"/>, so in the sandbox and
/// under the limits the options name), an agent on a task in a fresh copy of its project (<see cref="RunAsync"/>: an
/// evaluation of the model on the task), and a task's own check without a model (<see cref="CheckTaskAsync"/>).
/// </summary>
/// <param name="options">The commands' tools and time limit; null for the defaults.</param>
public sealed class CodingVerifier(CodingVerifierOptions? options = null)
{
    /// <summary>The commands' tools and time limit.</summary>
    public CodingVerifierOptions Options { get; } = options ?? new CodingVerifierOptions();

    /// <summary>Called with progress lines (setup, verification).</summary>
    public Action<string>? OnStatus { get; init; }

    /// <summary>The tools setup and verification commands run with in <paramref name="workspace"/> (the options' tools with the verification time limit).</summary>
    public CodingTools Checks(string workspace) => new(workspace, Options.Tools with { CommandTimeout = Options.Timeout });

    /// <summary>Runs the task's setup commands in order with <paramref name="checks"/>, stopping at the first that fails.</summary>
    public async Task<CodingVerification> SetupAsync(CodingTask task, CodingTools checks, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(checks);
        var setup = task.Setup ?? [];
        for (int i = 0; i < setup.Count; i++)
        {
            OnStatus?.Invoke($"setup: {setup[i]}");
            var result = await checks.ExecuteAsync(setup[i], ".", null, cancellationToken).ConfigureAwait(false);
            if (!result.Succeeded)
            {
                return new CodingVerification(false, setup.Count, i, $"$ {setup[i]}\n{result.Output}");
            }
        }

        return new CodingVerification(true, setup.Count, setup.Count, "");
    }

    /// <summary>
    /// Copies the task's hidden files into the workspace of <paramref name="checks"/> (over any copy the agent made), then
    /// runs its verification commands in order, stopping at the first that fails.
    /// </summary>
    public async Task<CodingVerification> VerifyAsync(CodingTask task, CodingTools checks, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(checks);
        if (task.VerifyFiles is { } hidden)
        {
            CodingWorkspace.Overlay(hidden, checks.Workspace);
        }

        var output = new StringBuilder();
        var verify = task.Verify ?? [];
        for (int i = 0; i < verify.Count; i++)
        {
            OnStatus?.Invoke($"verify: {verify[i]}");
            var result = await checks.ExecuteAsync(verify[i], ".", null, cancellationToken).ConfigureAwait(false);
            output.Append($"$ {verify[i]}\n{(result.TimedOut ? "timed out" : $"exit code {result.ExitCode}")}\n{result.Output}\n");
            if (!result.Succeeded)
            {
                return new CodingVerification(false, verify.Count, i, output.ToString());
            }
        }

        return new CodingVerification(true, verify.Count, verify.Count, output.ToString());
    }

    /// <summary>
    /// Runs <paramref name="task"/> in a fresh copy of its project under <paramref name="workFolder"/> (created; left in
    /// place for inspection): setup, the agent on the task's prompt (with its own tool options), the hidden files,
    /// verification.
    /// </summary>
    public async Task<CodingRun> RunAsync(CodingAgent agent, CodingTask task, string workFolder, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(agent);
        ArgumentNullException.ThrowIfNull(task);
        CodingWorkspace.Prepare(task, workFolder);
        var tools = new CodingTools(workFolder, agent.Options.Tools);
        var checks = Checks(workFolder);
        var setup = await SetupAsync(task, checks, cancellationToken).ConfigureAwait(false);
        if (!setup.Passed)
        {
            var none = new CodingAgentRun([], [.. tools.All.Select(t => t.Definition)], agent.Options.Think, CodingAgentEnd.Error, "not run: the setup failed", 0, 0, 0, 0,
                TimeSpan.Zero, TimeSpan.Zero, tools.Workspace);
            return new CodingRun(task, none, CodingOutcome.SetupFailed, setup.Output);
        }

        CodingWorkspace.CommitStart(workFolder);
        var run = await agent.RunAsync(task.Prompt, tools, cancellationToken).ConfigureAwait(false);
        if (run.End == CodingAgentEnd.Error)
        {
            return new CodingRun(task, run, CodingOutcome.Error, run.Note);
        }

        var verification = await VerifyAsync(task, checks, cancellationToken).ConfigureAwait(false);
        var outcome = verification.Passed ? CodingOutcome.Passed : run.End == CodingAgentEnd.OutOfBudget ? CodingOutcome.OutOfBudget : CodingOutcome.Failed;
        return new CodingRun(task, run, outcome, verification.Output);
    }

    /// <summary>
    /// Checks a task without a model: its verification must fail on the starting files (else the task tests nothing) and
    /// pass with its <see cref="CodingTask.Solution"/> (else it cannot be solved). Uses two folders under <paramref name="workFolder"/>.
    /// </summary>
    public async Task<CodingTaskCheck> CheckTaskAsync(CodingTask task, string workFolder, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(workFolder);
        var report = new StringBuilder();
        var results = new bool[2];
        for (int pass = 0; pass < 2; pass++)
        {
            string name = pass == 0 ? "start" : "solution";
            string folder = Path.Combine(workFolder, name);
            CodingWorkspace.Prepare(task, folder);
            if (pass == 1 && task.Solution is { } solution)
            {
                CodingWorkspace.Overlay(solution, folder);
            }

            var checks = Checks(folder);
            var setup = await SetupAsync(task, checks, cancellationToken).ConfigureAwait(false);
            if (!setup.Passed)
            {
                report.Append($"[{name}] setup failed: {setup.Output}\n");
            }

            var verification = setup.Passed ? await VerifyAsync(task, checks, cancellationToken).ConfigureAwait(false) : null;
            bool passed = verification?.Passed ?? false;
            results[pass] = pass == 0 ? setup.Passed && !passed : passed;
            report.Append($"[{name}] {(passed ? "verification passed" : "verification failed")}\n{verification?.Output ?? ""}");
        }

        return new CodingTaskCheck(results[0], task.Solution is not null && results[1], report.ToString());
    }
}
