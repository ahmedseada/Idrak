// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Inference;

/// <summary>
/// The predictor kind of the inference engine: copies are <see cref="Predictor{TIn, TOut}"/>s, which are thread-safe,
/// so requests share them; with <see cref="PredictorBuilder{TIn, TOut}.Batching"/>, single requests are micro-batched.
/// </summary>
internal sealed class PredictorModel<TIn, TOut> : EngineModel<Predictor<TIn, TOut>>, IPredictor<TIn, TOut>
{
    /// <summary>The kind's name in the engine's status.</summary>
    public const string KindName = "predictor";

    private readonly PredictorBuilder<TIn, TOut> _builder;
    private readonly Func<Module> _load;
    private readonly bool _ownsModel;
    private IEngineBatcher<TIn, TOut>? _batcher;

    public PredictorModel(PredictorBuilder<TIn, TOut> builder, Func<Module> load, bool ownsModel, bool reloadable)
        : base(KindName, new EngineHosting
        {
            Instances = builder.Settings.Instances ?? 1,
            SharedCopies = true,
            Reloadable = reloadable,
            KeepAlive = builder.Settings.KeepAlive,
            QueueLimit = builder.Settings.QueueLimit,
            Timeout = builder.Settings.Timeout,
        })
    {
        _builder = builder;
        _load = load;
        _ownsModel = ownsModel;
    }

    protected override void OnAttached()
    {
        if (_builder.Settings.Batching is var (maxBatch, maxWait))
        {
            _batcher = Host.Batcher<TIn, TOut>(maxBatch, maxWait, PredictBatchAsync);
        }
    }

    public override Predictor<TIn, TOut> LoadCopy() => _builder.Create(_load(), _ownsModel);

    public override void UnloadCopy(Predictor<TIn, TOut> copy) => copy.Dispose();

    public override ModelDescription Describe(Predictor<TIn, TOut> copy) => new(Name, Kind, copy.Model.ParameterCount, copy.Device, null);

    public async ValueTask<TOut> PredictAsync(TIn input, CancellationToken cancellationToken = default) =>
        _batcher is { } batcher
            ? await batcher.SubmitAsync(input, cancellationToken).ConfigureAwait(false)
            : (await PredictBatchAsync([input], cancellationToken).ConfigureAwait(false))[0];

    public async ValueTask<IReadOnlyList<TOut>> PredictAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken = default) =>
        await PredictBatchAsync(inputs, cancellationToken).ConfigureAwait(false);

    private async Task<IReadOnlyList<TOut>> PredictBatchAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken)
    {
        using var lease = await Host.AcquireAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var predictor = lease.Copy;
            var clock = Stopwatch.StartNew();
            var results = await Task.Run(() => predictor.Predict(inputs), lease.Token).ConfigureAwait(false);
            lease.Completed(inputs.Count, inputs.Count, clock.Elapsed);
            return results;
        }
        catch (Exception ex)
        {
            throw lease.Fail(ex);
        }
    }
}
