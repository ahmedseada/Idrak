// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Reinforcement.Plugins.Classic;

// What every classic environment keeps of its episode: whether one is under way and its steps, and the checks a step
// makes before it runs.
internal struct EpisodeClock
{
    public bool Running;
    public int Steps;

    public void Start()
    {
        Running = true;
        Steps = 0;
    }

    // Counts a step of a running episode (refusing one when none is), after the token's check.
    public void Tick(string kind, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!Running)
        {
            throw new InvalidOperationException($"The {kind} environment has no episode under way: reset it first (also after an episode ended).");
        }

        Steps++;
    }

    // Gymnasium's TimeLimit: the step that reaches the limit is truncated (whether or not it also terminated).
    public bool Truncates(int maxSteps) => maxSteps > 0 && Steps >= maxSteps;

    public void End(bool terminated, bool truncated)
    {
        if (terminated || truncated)
        {
            Running = false;
        }
    }
}
