// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Generation.Abstractions;

/// <summary>One sampled token with the statistics recorded by <see cref="TokenSampler"/>.</summary>
/// <param name="Id">The chosen token id.</param>
/// <param name="Probability">Its probability after temperature and top-k.</param>
/// <param name="Entropy">Entropy of the sampling distribution, in bits.</param>
/// <param name="Alternatives">The five most likely tokens (id, probability), most likely first; id -1 when fewer exist.</param>
public readonly record struct SampledToken(int Id, float Probability, float Entropy, (int Id, float Probability)[] Alternatives);

/// <summary>
/// Chooses the next token of each sequence from the model's logits. <see cref="TokenSampler"/> is the built-in one
/// (temperature, top-k, top-p, min-p and penalties, on the device); implement this to plug in another strategy
/// (constrained or grammar-guided decoding, a custom sampling rule) through <c>Generation.TextGenerator.CreateSampler</c>.
/// </summary>
public interface ITokenSampler : IDisposable
{
    /// <summary>Sequences sampled per step.</summary>
    int Rows { get; }

    /// <summary>Vocabulary size.</summary>
    int Vocabulary { get; }

    /// <summary>The most recently sampled ids, [rows], on the logits' device; the next decoding step reads them from here.</summary>
    Tensor Ids { get; }

    /// <summary>
    /// True when <see cref="Sample"/> only queues work on the device (no reads back to the host), so a decoding step that
    /// samples can be recorded once as a CUDA graph and replayed. A sampler that decides on the host returns false, and
    /// generation then runs every step as it is.
    /// </summary>
    bool Recordable { get; }

    /// <summary>
    /// Samples one token per row from the last position of <paramref name="logits"/> ([rows, vocabulary] or
    /// [rows, steps, vocabulary]) into <see cref="Ids"/>, and records the step's statistics.
    /// </summary>
    void Sample(Tensor logits);

    /// <summary>Sets the recent tokens of every row (the prompt), which penalties look at. Called before sampling.</summary>
    void SetHistory(IReadOnlyList<int> tokens);

    /// <summary>Restarts step counting (and any random stream) from step 0.</summary>
    void Reset();

    /// <summary>The statistics of steps [<paramref name="fromStep"/>, <paramref name="toStep"/>) as [step][row] tokens.</summary>
    SampledToken[][] Read(int fromStep, int toStep);
}

/// <summary>What a generation asks a sampler for: where it runs, its shape, and the settings to sample with.</summary>
/// <param name="Device">The logits' device.</param>
/// <param name="Rows">Sequences sampled per step.</param>
/// <param name="Vocabulary">Vocabulary size.</param>
/// <param name="MaxSteps">Steps whose statistics are kept.</param>
/// <param name="HistoryCapacity">Recent tokens per row the penalties may look at.</param>
/// <param name="Options">Temperature, top-k, top-p, min-p, penalties and seed.</param>
public sealed record SamplerRequest(Device Device, int Rows, int Vocabulary, int MaxSteps, int HistoryCapacity, GenerationOptions Options);
