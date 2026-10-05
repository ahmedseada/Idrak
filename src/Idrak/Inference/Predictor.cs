// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Layers;

namespace Idrak.Inference;

/// <summary>A prediction service: one input in, one answer out (also batched and asynchronous).</summary>
/// <typeparam name="TIn">The input type (for example a record describing a house).</typeparam>
/// <typeparam name="TOut">The answer type (for example a price, or a <see cref="ClassPrediction"/>).</typeparam>
public interface IPredictor<TIn, TOut>
{
    /// <summary>Predicts one input.</summary>
    ValueTask<TOut> PredictAsync(TIn input, CancellationToken cancellationToken = default);

    /// <summary>Predicts several inputs as one batch.</summary>
    ValueTask<IReadOnlyList<TOut>> PredictAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken = default);
}

/// <summary>The score of one class.</summary>
public sealed record ClassScore(string Class, float Score);

/// <summary>
/// A classification answer: the best class, its index and score (a probability when <see cref="PredictorBuilder{TIn, TOut}.Softmax"/>
/// is part of the predictor, otherwise the raw output), and every class with its score, best first.
/// </summary>
public sealed record ClassPrediction(string Class, int Index, float Probability, IReadOnlyList<ClassScore> Scores);

/// <summary>Entry points for <see cref="PredictorBuilder{TIn, TOut}"/>.</summary>
public static class Predictor
{
    /// <summary>
    /// Starts a predictor for <paramref name="model"/>. Without further steps it maps rows of numbers to rows of outputs
    /// exactly like <see cref="Module.Predict(float[,])"/>; each step adds one of the calls you would otherwise write
    /// around it (scaling, typed input and output, softmax, class names).
    /// </summary>
    public static PredictorBuilder<float[], float[]> For(Module model) => new(new PredictorSettings(model, ownsModel: false), r => r, r => r);

    /// <summary>
    /// Opens a predictor saved with <see cref="Predictor{TIn, TOut}.Save"/>: the model is rebuilt from its stored architecture
    /// on <paramref name="device"/> and every stored step (scalers, input shape, softmax, classes) is restored. Add the
    /// typed <c>Input</c>/<c>Output</c> mappings again (functions cannot be stored in a file).
    /// </summary>
    public static PredictorBuilder<float[], float[]> Load(string path, Device? device = null)
    {
        using var package = ModelPackage.Open(path);
        if (!package.Contains(PackageEntryKind.Architecture, ModelPackage.DefaultModelName))
        {
            throw new InvalidOperationException($"{path} has no architecture (the model was not made with the network builder); use Predictor.Load(path, model) with a model you build.");
        }

        var model = package.BuildModel(device: device);
        return Restore(package, new PredictorSettings(model, ownsModel: true));
    }

    /// <summary>Opens a saved predictor, loading its weights into <paramref name="model"/> (built by you with the same layers).</summary>
    public static PredictorBuilder<float[], float[]> Load(string path, Module model)
    {
        using var package = ModelPackage.Open(path);
        package.LoadWeights(model);
        return Restore(package, new PredictorSettings(model, ownsModel: false));
    }

    internal static PredictorBuilder<float[], float[]> Restore(ModelPackageReader package, PredictorSettings settings)
    {
        var builder = new PredictorBuilder<float[], float[]>(settings, r => r, r => r);
        if (package.Contains(PackageEntryKind.StandardScaler, "features") || package.Contains(PackageEntryKind.MinMaxScaler, "features"))
        {
            builder.ScaleInputs(package.Scaler("features"));
        }

        if (package.Contains(PackageEntryKind.StandardScaler, "targets") || package.Contains(PackageEntryKind.MinMaxScaler, "targets"))
        {
            builder.UnscaleOutputs(package.Scaler("targets"));
        }

        if (package.Contains(PackageEntryKind.Json, "predictor"))
        {
            var json = package.Json("predictor");
            if (json["inputShape"] is JsonArray shape)
            {
                builder.InputShape([.. shape.Select(v => (int)v!)]);
            }

            if ((bool?)json["softmax"] == true)
            {
                builder.Softmax();
            }

            if ((int?)json["batchSize"] is { } batch)
            {
                builder.BatchSize(batch);
            }

            if (json["classes"] is JsonArray classes)
            {
                settings.StoredClasses = [.. classes.Select(c => (string)c!)];
            }
        }

        return builder;
    }
}

/// <summary>The settings shared by a predictor builder as its input and output types change.</summary>
internal sealed class PredictorSettings(Module? model, bool ownsModel)
{
    /// <summary>The model; null for the templates the inference engine uses to create one predictor per model copy.</summary>
    public Module? Model { get; } = model;
    public bool OwnsModel { get; } = ownsModel;
    public IScaler? FeatureScaler { get; set; }
    public IScaler? TargetScaler { get; set; }
    public Device? Device { get; set; }
    public int[]? InputShape { get; set; }
    public int? BatchSize { get; set; }
    public bool Softmax { get; set; }
    public IReadOnlyList<string>? Classes { get; set; }
    public IReadOnlyList<string>? StoredClasses { get; set; }
    public object? WarmUp { get; set; }

    // Engine-only settings (see InferenceEngine).
    public (int MaxBatch, TimeSpan MaxWait)? Batching { get; set; }
    public int? Instances { get; set; }
    public TimeSpan? KeepAlive { get; set; }
    public bool KeepAliveSet { get; set; }
    public int? QueueLimit { get; set; }
    public TimeSpan? Timeout { get; set; }

    public bool HasEngineSettings => Batching is not null || Instances is not null || KeepAliveSet || QueueLimit is not null || Timeout is not null;
}

/// <summary>
/// Builds a <see cref="Predictor{TIn, TOut}"/>. Every step is optional and does nothing unless added; the order of work
/// at prediction time is fixed: input mapping → input scaling → model → output unscaling → softmax → output mapping.
/// </summary>
public sealed class PredictorBuilder<TIn, TOut>
{
    internal PredictorBuilder(PredictorSettings settings, Func<TIn, float[]> input, Func<float[], TOut> output)
    {
        Settings = settings;
        InputMap = input;
        OutputMap = output;
    }

    internal PredictorSettings Settings { get; }

    internal Func<TIn, float[]> InputMap { get; }

    internal Func<float[], TOut> OutputMap { get; }

    /// <summary>Applies <c>scaler.Transform</c> to every input row before the model (use the scaler fitted on the training features).</summary>
    public PredictorBuilder<TIn, TOut> ScaleInputs(IScaler scaler)
    {
        Settings.FeatureScaler = scaler;
        return this;
    }

    /// <summary>Applies <c>scaler.InverseTransform</c> to every output row after the model (use the scaler fitted on the training targets).</summary>
    public PredictorBuilder<TIn, TOut> UnscaleOutputs(IScaler scaler)
    {
        Settings.TargetScaler = scaler;
        return this;
    }

    /// <summary>Creates the input tensors on <paramref name="device"/>; the model's parameters must be there too.</summary>
    public PredictorBuilder<TIn, TOut> OnDevice(Device device)
    {
        Settings.Device = device;
        return this;
    }

    /// <summary>The shape of one input sample, e.g. <c>(28, 1)</c> for a GRU over 28 steps or <c>(1, 16, 16)</c> for images. Without it rows are [N, features].</summary>
    public PredictorBuilder<TIn, TOut> InputShape(params int[] shape)
    {
        Settings.InputShape = [.. shape];
        return this;
    }

    /// <summary>
    /// Splits large lists into batches of <paramref name="rows"/> (like <see cref="Training.Trainer.Predict"/>'s <c>batchSize</c>).
    /// Without it, batches of <see cref="Predictor{TIn, TOut}.DefaultBatchSize"/> rows.
    /// </summary>
    public PredictorBuilder<TIn, TOut> BatchSize(int rows)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        Settings.BatchSize = rows;
        return this;
    }

    /// <summary>Applies softmax to every output row (turns classifier logits into probabilities).</summary>
    public PredictorBuilder<TIn, TOut> Softmax()
    {
        Settings.Softmax = true;
        return this;
    }

    /// <summary>Maps your input type to one row of numbers for the model.</summary>
    public PredictorBuilder<T, TOut> Input<T>(Func<T, float[]> map) => Copy<T, TOut>(map, OutputMap);

    /// <summary>Maps one row of outputs (after unscaling and softmax, if added) to your answer type.</summary>
    public PredictorBuilder<TIn, T> Output<T>(Func<float[], T> map) => Copy<TIn, T>(InputMap, map);

    /// <summary>Answers with the class whose output is largest, named by <paramref name="names"/> (in output order), with every class's score.</summary>
    public PredictorBuilder<TIn, ClassPrediction> Classes(IReadOnlyList<string> names)
    {
        ArgumentNullException.ThrowIfNull(names);
        Settings.Classes = [.. names];
        string[] classes = [.. names];
        return Copy<TIn, ClassPrediction>(InputMap, row =>
        {
            if (row.Length != classes.Length)
            {
                throw new InvalidOperationException($"The model produced {row.Length} outputs for {classes.Length} class names.");
            }

            int best = 0;
            for (int i = 1; i < row.Length; i++)
            {
                if (row[i] > row[best])
                {
                    best = i;
                }
            }

            var scores = row.Select((s, i) => new ClassScore(classes[i], s)).OrderByDescending(s => s.Score).ToList();
            return new ClassPrediction(classes[best], best, row[best], scores);
        });
    }

    /// <summary>The class names stored in a loaded predictor package, for <see cref="Classes"/> (null if none were stored).</summary>
    public IReadOnlyList<string>? StoredClasses => Settings.StoredClasses;

    /// <summary>Runs one prediction with <paramref name="sample"/> when the predictor is built (or loaded by the engine), so the first real call is not slow.</summary>
    public PredictorBuilder<TIn, TOut> WarmUp(TIn sample)
    {
        Settings.WarmUp = sample;
        return this;
    }

    // ------------------------------------------------------------------ engine-only settings

    /// <summary>Engine only: combines requests arriving within <paramref name="maxWait"/> into one batch of up to <paramref name="maxBatch"/> rows.</summary>
    public PredictorBuilder<TIn, TOut> Batching(int maxBatch, TimeSpan maxWait)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBatch);
        Settings.Batching = (maxBatch, maxWait);
        return this;
    }

    /// <summary>Engine only: loads <paramref name="count"/> copies of the model (needs a model source the engine can load again).</summary>
    public PredictorBuilder<TIn, TOut> Instances(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        Settings.Instances = count;
        return this;
    }

    /// <summary>Engine only: unloads the model after it has been idle this long (null keeps it forever, <see cref="TimeSpan.Zero"/> unloads after each request).</summary>
    public PredictorBuilder<TIn, TOut> KeepAlive(TimeSpan? idle)
    {
        Settings.KeepAlive = idle;
        Settings.KeepAliveSet = true;
        return this;
    }

    /// <summary>Engine only: rejects new requests while <paramref name="waiting"/> are already queued.</summary>
    public PredictorBuilder<TIn, TOut> QueueLimit(int waiting)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(waiting);
        Settings.QueueLimit = waiting;
        return this;
    }

    /// <summary>Engine only: fails a request that has not completed after <paramref name="limit"/> (queue wait included).</summary>
    public PredictorBuilder<TIn, TOut> Timeout(TimeSpan limit)
    {
        Settings.Timeout = limit;
        return this;
    }

    /// <summary>Creates the predictor: switches the model to evaluation mode once (so concurrent calls are safe) and runs the warm-up, if set.</summary>
    public Predictor<TIn, TOut> Build()
    {
        if (Settings.HasEngineSettings)
        {
            throw new InvalidOperationException("Batching, Instances, KeepAlive, QueueLimit and Timeout apply only when the predictor is hosted by an InferenceEngine.");
        }

        return Create(Settings.Model ?? throw new InvalidOperationException("This builder belongs to an inference engine; it has no model of its own."),
            Settings.OwnsModel);
    }

    /// <summary>A builder over a template (no model), for the inference engine.</summary>
    internal static PredictorBuilder<float[], float[]> Template() => new(new PredictorSettings(null, ownsModel: false), r => r, r => r);

    internal Predictor<TIn, TOut> Create(Module model, bool ownsModel)
    {
        var predictor = new Predictor<TIn, TOut>(model, ownsModel, Settings, InputMap, OutputMap);
        if (Settings.WarmUp is TIn sample)
        {
            predictor.Predict(sample);
        }

        return predictor;
    }

    private PredictorBuilder<TIn2, TOut2> Copy<TIn2, TOut2>(Func<TIn2, float[]> input, Func<float[], TOut2> output)
    {
        if (Settings.WarmUp is not null && typeof(TIn2) != typeof(TIn))
        {
            Settings.WarmUp = null;                 // the sample was of the old input type; set WarmUp after Input<T>
        }

        return new PredictorBuilder<TIn2, TOut2>(Settings, input, output);
    }
}

/// <summary>
/// Maps inputs to answers with a model: input mapping, scaling, <see cref="Module.Predict(Tensor)"/>, unscaling,
/// softmax and output mapping, as configured by <see cref="PredictorBuilder{TIn, TOut}"/>. Thread-safe: the model is in
/// evaluation mode and every call works on its own tensors.
/// </summary>
public sealed class Predictor<TIn, TOut> : IPredictor<TIn, TOut>, IDisposable
{
    private readonly PredictorSettings _settings;
    private readonly Func<TIn, float[]> _input;
    private readonly Func<float[], TOut> _output;
    private readonly bool _ownsModel;

    internal Predictor(Module model, bool ownsModel, PredictorSettings settings, Func<TIn, float[]> input, Func<float[], TOut> output)
    {
        Model = model;
        _ownsModel = ownsModel;
        _settings = settings;
        _input = input;
        _output = output;
        Device = settings.Device ?? model.WeightsDevice ?? Device.Default;
        if (model.Parameters().FirstOrDefault() is { } p && p.Device != Device)
        {
            throw new InvalidOperationException($"The model is on {p.Device} but the predictor was set to use {Device}; move the model with model.To(device) first.");
        }

        model.Eval();
    }

    /// <summary>The model.</summary>
    public Module Model { get; }

    /// <summary>Where input tensors are created.</summary>
    public Device Device { get; }

    /// <summary>The class names, when the predictor answers with <see cref="ClassPrediction"/>.</summary>
    public IReadOnlyList<string>? Classes => _settings.Classes;

    /// <summary>Predicts one input.</summary>
    public TOut Predict(TIn input) => Predict([input])[0];

    /// <summary>
    /// Rows per batch when <see cref="PredictorBuilder{TIn, TOut}.BatchSize"/> is not set. A whole data set as one batch
    /// can exceed what a device runs in one launch (10,000 MNIST images through a convolution are 7.8 million im2col rows)
    /// or holds in memory.
    /// </summary>
    public const int DefaultBatchSize = 512;

    /// <summary>Predicts several inputs, in batches of <see cref="PredictorBuilder{TIn, TOut}.BatchSize"/> (<see cref="DefaultBatchSize"/> if not set).</summary>
    public IReadOnlyList<TOut> Predict(IReadOnlyList<TIn> inputs)
    {
        var results = new List<TOut>(inputs.Count);
        int batch = _settings.BatchSize ?? DefaultBatchSize;
        for (int start = 0; start < inputs.Count; start += batch)
        {
            int count = Math.Min(batch, inputs.Count - start);
            var rows = new float[count][];
            for (int i = 0; i < count; i++)
            {
                rows[i] = _input(inputs[start + i]);
            }

            foreach (var row in PredictRows(rows))
            {
                results.Add(_output(row));
            }
        }

        return results;
    }

    /// <summary>
    /// Predicts every row of <paramref name="data"/> (raw, unscaled features, as loaded), in batches of
    /// <see cref="PredictorBuilder{TIn, TOut}.BatchSize"/> (<see cref="DefaultBatchSize"/> if not set); targets are ignored.
    /// </summary>
    public IReadOnlyList<TOut> Predict(Dataset data)
    {
        var results = new List<TOut>(data.Count);
        int batch = _settings.BatchSize ?? DefaultBatchSize;
        for (int start = 0; start < data.Count; start += batch)
        {
            int count = Math.Min(batch, data.Count - start);
            var rows = new float[count][];
            for (int i = 0; i < count; i++)
            {
                rows[i] = data.GetFeatures(start + i).ToArray();
            }

            foreach (var row in PredictRows(rows))
            {
                results.Add(_output(row));
            }
        }

        return results;
    }

    /// <inheritdoc />
    public ValueTask<TOut> PredictAsync(TIn input, CancellationToken cancellationToken = default) => ValueTask.FromResult(Predict(input));

    /// <inheritdoc />
    public ValueTask<IReadOnlyList<TOut>> PredictAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(Predict(inputs));

    /// <summary>
    /// Saves the model's weights, its architecture (when it was made with the network builder), the scalers (as
    /// "features" and "targets"), and the input shape, batch size, softmax and class names, as one package.
    /// </summary>
    public void Save(string path)
    {
        var writer = ModelPackage.Create(path).Weights(Model);
        if (Network.ArchitectureOf(Model) is { } architecture)
        {
            writer.Architecture(ModelPackage.DefaultModelName, architecture);
        }
        else if (Model is GraphModule graph)
        {
            writer.Architecture(ModelPackage.DefaultModelName, graph.ToJson());
        }
        else if (DecoderSpec.Of(Model) is { } decoder)
        {
            writer.Architecture(ModelPackage.DefaultModelName, decoder.ToJson());
        }

        if (_settings.FeatureScaler is { } features)
        {
            writer.Scaler("features", features);
        }

        if (_settings.TargetScaler is { } targets)
        {
            writer.Scaler("targets", targets);
        }

        var json = new JsonObject { ["softmax"] = _settings.Softmax };
        if (_settings.InputShape is { } shape)
        {
            json["inputShape"] = new JsonArray([.. shape.Select(v => (JsonNode)v)]);
        }

        if (_settings.BatchSize is { } batch)
        {
            json["batchSize"] = batch;
        }

        if (_settings.Classes is { } classes)
        {
            json["classes"] = new JsonArray([.. classes.Select(c => (JsonNode)c)]);
        }

        writer.Json("predictor", (JsonNode)json).Save();
    }

    /// <summary>Disposes the model when the predictor created it (<see cref="Predictor.Load(string, Device?)"/>).</summary>
    public void Dispose()
    {
        if (_ownsModel)
        {
            Model.Dispose();
        }
    }

    /// <summary>Scales, predicts and unscales a batch of raw rows; returns one output row per input row.</summary>
    internal float[][] PredictRows(float[][] rows)
    {
        if (rows.Length == 0)
        {
            return [];
        }

        int width = rows[0].Length;
        var flat = new float[rows.Length * width];
        for (int i = 0; i < rows.Length; i++)
        {
            if (rows[i].Length != width)
            {
                throw new ArgumentException($"Input row {i} has {rows[i].Length} values; row 0 has {width}.");
            }

            rows[i].CopyTo(flat, i * width);
        }

        int[] sample = _settings.InputShape ?? [width];
        if (sample.Aggregate(1, (a, b) => a * b) != width)
        {
            throw new ArgumentException($"An input row has {width} values, but the input shape {Tensor.FormatShape(sample)} needs {sample.Aggregate(1, (a, b) => a * b)}.");
        }

        _settings.FeatureScaler?.Transform(flat, _settings.InputShape is null ? width : sample[^1]);
        float[] values;
        int outputs;
        using (var x = Tensor.From(flat, [rows.Length, .. sample], Device))
        using (var y = Model.Predict(x))
        {
            if (_settings.Softmax && _settings.TargetScaler is null)
            {
                using var p = y.Softmax();
                values = p.ToArray();
            }
            else
            {
                values = y.ToArray();
            }

            outputs = y.Size / rows.Length;
        }

        if (_settings.TargetScaler is { } targets)
        {
            targets.InverseTransform(values, outputs);
            if (_settings.Softmax)
            {
                SoftmaxRows(values, outputs);
            }
        }

        var result = new float[rows.Length][];
        for (int i = 0; i < rows.Length; i++)
        {
            result[i] = values.AsSpan(i * outputs, outputs).ToArray();
        }

        return result;
    }

    private static void SoftmaxRows(float[] values, int columns)
    {
        for (int start = 0; start < values.Length; start += columns)
        {
            var row = values.AsSpan(start, columns);
            float max = float.NegativeInfinity;
            foreach (float v in row) max = MathF.Max(max, v);
            float sum = 0;
            for (int i = 0; i < row.Length; i++) sum += row[i] = MathF.Exp(row[i] - max);
            for (int i = 0; i < row.Length; i++) row[i] /= sum;
        }
    }
}
