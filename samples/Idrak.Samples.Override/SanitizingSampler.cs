// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Generation.Abstractions;

namespace Idrak.Samples.Override;

/// <summary>
/// The app's own token sampler (plan 10, "The override loop"): the library's sampler, except that logits which are not
/// finite (NaN, infinite: a model that overflows now and then) are replaced before sampling, so a bad step yields a
/// valid token instead of an arbitrary one. It fixes that one case and leaves everything else to the library default
/// (<see cref="TokenSampler.Create"/>), which it wraps; the app's tests check it with the testing kit.
/// </summary>
public sealed class SanitizingSampler : ITokenSampler
{
    /// <summary>The value a NaN or -∞ logit becomes (never chosen while another token is finite).</summary>
    public const float Lowest = -1e30f;

    /// <summary>The value a +∞ logit becomes (chosen as a finite very large value would be).</summary>
    public const float Highest = 1e30f;

    private readonly ITokenSampler _library;

    private SanitizingSampler(SamplerRequest request) => _library = TokenSampler.Create(request);

    /// <summary>The factory a generation calls (<c>TextGenerator.CreateSampler = SanitizingSampler.Create</c>).</summary>
    public static ITokenSampler Create(SamplerRequest request) => new SanitizingSampler(request);

    /// <summary>Non-finite logits replaced so far.</summary>
    public int Replaced { get; private set; }

    /// <inheritdoc />
    public int Rows => _library.Rows;

    /// <inheritdoc />
    public int Vocabulary => _library.Vocabulary;

    /// <inheritdoc />
    public Tensor Ids => _library.Ids;

    /// <summary>False: the logits are read on the host to find the bad values, so a decoding step is not recorded as a graph.</summary>
    public bool Recordable => false;

    /// <inheritdoc />
    public void Sample(Tensor logits)
    {
        ArgumentNullException.ThrowIfNull(logits);
        var values = logits.ToArray();
        int replaced = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
            {
                values[i] = float.IsPositiveInfinity(values[i]) ? Highest : Lowest;
                replaced++;
            }
        }

        if (replaced == 0)
        {
            _library.Sample(logits);                                          // the common case: the library's path as it is
            return;
        }

        Replaced += replaced;
        using var clean = Tensor.From(values, logits.Shape, logits.Device);
        _library.Sample(clean);
    }

    /// <inheritdoc />
    public void SetHistory(IReadOnlyList<int> tokens) => _library.SetHistory(tokens);

    /// <inheritdoc />
    public void Reset() => _library.Reset();

    /// <inheritdoc />
    public SampledToken[][] Read(int fromStep, int toStep) => _library.Read(fromStep, toStep);

    /// <inheritdoc />
    public void Dispose() => _library.Dispose();
}
