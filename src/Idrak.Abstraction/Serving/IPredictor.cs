// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Serving;

/// <summary>A prediction service: one input in, one answer out (also batched and asynchronous).</summary>
/// <typeparam name="TIn">The input type (for example a record describing a house).</typeparam>
/// <typeparam name="TOut">The answer type (for example a price, or a <c>ClassPrediction</c>).</typeparam>
public interface IPredictor<TIn, TOut>
{
    /// <summary>Predicts one input.</summary>
    ValueTask<TOut> PredictAsync(TIn input, CancellationToken cancellationToken = default);

    /// <summary>Predicts several inputs as one batch.</summary>
    ValueTask<IReadOnlyList<TOut>> PredictAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken = default);
}
