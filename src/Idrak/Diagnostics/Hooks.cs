// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

namespace Idrak.Diagnostics;

/// <summary>Writes human-readable progress to the console (or any <see cref="TextWriter"/>).</summary>
/// <param name="levels">What to print. <see cref="TelemetryLevel.Training"/> gives one line per epoch.</param>
/// <param name="epochInterval">Print every n-th epoch (the first and last are always printed; "*" marks a new best).</param>
/// <param name="batchInterval">Print every n-th batch when <see cref="TelemetryLevel.Batches"/> is on.</param>
/// <param name="output">Destination; defaults to <see cref="Console.Out"/>.</param>
public sealed class ConsoleLogger(TelemetryLevel levels = TelemetryLevel.Training, int epochInterval = 1, int batchInterval = 1, TextWriter? output = null)
    : ITelemetryHook
{
    private readonly TextWriter _out = output ?? Console.Out;

    /// <inheritdoc />
    public TelemetryLevel Levels { get; } = levels;

    /// <inheritdoc />
    public void OnTrainingStarted(in TrainingStarted e)
    {
        _out.WriteLine(e.Model);
        _out.WriteLine(
            $"Training on {e.Device} ({e.Device.Name}) | {e.TrainingSamples:N0} samples" +
            (e.ValidationSamples is { } v ? $", {v:N0} validation" : "") +
            $" | batch {e.BatchSize}, {e.BatchesPerEpoch} steps/epoch | {e.Optimizer} lr={e.LearningRate:G4} | {e.ParameterCount:N0} parameters | {e.CpuThreads} CPU threads");
    }

    /// <inheritdoc />
    public void OnBatchCompleted(in BatchCompleted e)
    {
        if (e.Batch % Math.Max(1, batchInterval) != 0 && e.Batch != e.BatchesPerEpoch)
        {
            return;
        }

        _out.WriteLine(
            $"  epoch {e.Epoch} batch {e.Batch}/{e.BatchesPerEpoch}  loss {e.Loss:F6}" +
            (e.GradientNorm is { } g ? $"  |grad| {g:F4}" : "") +
            $"  data {e.DataTime.TotalMilliseconds:F2} ms  compute {e.ComputeTime.TotalMilliseconds:F2} ms");
    }

    /// <inheritdoc />
    public void OnEpochCompleted(in EpochCompleted e)
    {
        if (e.Epoch != 1 && e.Epoch != e.Epochs && e.Epoch % Math.Max(1, epochInterval) != 0)
        {
            return;
        }

        var line = new StringBuilder();
        line.Append(CultureInfo.InvariantCulture, $"Epoch {e.Epoch.ToString().PadLeft(e.Epochs.ToString().Length)}/{e.Epochs}  loss {e.Loss:F6}");
        foreach (var (name, value) in e.Metrics)
        {
            line.Append(CultureInfo.InvariantCulture, $"  {name} {value:F4}");
        }

        if (e.ValidationLoss is { } vl)
        {
            line.Append(CultureInfo.InvariantCulture, $"  val_loss {vl:F6}");
            foreach (var (name, value) in e.ValidationMetrics ?? new Dictionary<string, double>())
            {
                line.Append(CultureInfo.InvariantCulture, $"  val_{name} {value:F4}");
            }
        }

        line.Append(CultureInfo.InvariantCulture, $"  {e.Duration.TotalMilliseconds:F1} ms  {e.SamplesPerSecond:N0} samples/s");
        if (e.IsBest)
        {
            line.Append("  *");
        }

        _out.WriteLine(line);
    }

    /// <inheritdoc />
    public void OnTrainingCompleted(in TrainingCompleted e) =>
        _out.WriteLine(
            $"Finished {e.EpochsRun} epochs in {e.Duration.TotalSeconds:F2} s" +
            (e.StoppedEarly ? " (early stop)" : e.Cancelled ? " (cancelled)" : "") +
            $" | best epoch {e.BestEpoch} loss {e.BestLoss:F6}");

    /// <inheritdoc />
    public void OnLayerForward(in LayerForward e) =>
        _out.WriteLine(
            $"  {new string(' ', e.Depth * 2)}{e.Layer}: {Tensor.FormatShape(e.InputShape)} -> {Tensor.FormatShape(e.OutputShape)}  {e.Duration.TotalMicroseconds:F1} µs");

    /// <inheritdoc />
    public void OnOperation(in OperationCompleted e) =>
        _out.WriteLine($"    {(e.Backward ? "∇" : " ")}{e.Operation} {Tensor.FormatShape(e.Shape)} on {e.Device}  {e.Duration.TotalMicroseconds:F1} µs");

    /// <inheritdoc />
    public void OnInference(in InferenceCompleted e) =>
        _out.WriteLine($"Inference {e.Model}: {e.Samples:N0} samples on {e.Device} in {e.Latency.TotalMilliseconds:F3} ms ({e.SamplesPerSecond:N0} samples/s)");

    /// <inheritdoc />
    public void OnToolCall(in ToolCallCompleted e) =>
        _out.WriteLine($"Tool {e.Tool}({e.Arguments}): {(e.Succeeded ? "ok" : "failed: " + e.Error)} in {e.Duration.TotalMilliseconds:F1} ms");

    /// <inheritdoc />
    public void OnEngine(in EngineEvent e) => _out.WriteLine(e.Kind switch
    {
        EngineEventKind.ModelLoaded => $"Engine: loaded {e.Model} in {e.Duration.TotalMilliseconds:F1} ms",
        EngineEventKind.ModelUnloaded => $"Engine: unloaded {e.Model}",
        EngineEventKind.RequestCompleted => $"Engine: {e.Model} request in {e.Duration.TotalMilliseconds:F2} ms (queued {e.QueueWait.TotalMilliseconds:F2} ms, batch {e.BatchSize})",
        _ => $"Engine: {e.Model} request rejected: {e.Reason}",
    });
}

/// <summary>Keeps every epoch (and optionally batch) event in memory, for charts, reports or CSV export.</summary>
public sealed class MetricsRecorder(TelemetryLevel levels = TelemetryLevel.Training) : ITelemetryHook
{
    private readonly List<EpochCompleted> _epochs = [];
    private readonly List<BatchCompleted> _batches = [];
    private readonly Lock _lock = new();

    /// <inheritdoc />
    public TelemetryLevel Levels { get; } = levels;

    /// <summary>Recorded epochs, in order.</summary>
    public IReadOnlyList<EpochCompleted> Epochs
    {
        get
        {
            lock (_lock)
            {
                return [.. _epochs];
            }
        }
    }

    /// <summary>Recorded batches, in order (needs <see cref="TelemetryLevel.Batches"/>).</summary>
    public IReadOnlyList<BatchCompleted> Batches
    {
        get
        {
            lock (_lock)
            {
                return [.. _batches];
            }
        }
    }

    /// <inheritdoc />
    public void OnEpochCompleted(in EpochCompleted e)
    {
        lock (_lock)
        {
            _epochs.Add(e);
        }
    }

    /// <inheritdoc />
    public void OnBatchCompleted(in BatchCompleted e)
    {
        lock (_lock)
        {
            _batches.Add(e);
        }
    }

    /// <summary>Writes one row per epoch: epoch, loss, each metric, validation loss and metrics, timing, memory.</summary>
    public void SaveCsv(string path)
    {
        var epochs = Epochs;
        var metricNames = epochs.SelectMany(e => e.Metrics.Keys).Distinct().ToList();
        var validationNames = epochs.SelectMany(e => e.ValidationMetrics?.Keys ?? []).Distinct().ToList();
        var inv = CultureInfo.InvariantCulture;
        using var writer = new StreamWriter(path);
        writer.WriteLine(string.Join(',',
            new[] { "epoch", "loss" }
                .Concat(metricNames)
                .Append("val_loss")
                .Concat(validationNames.Select(n => "val_" + n))
                .Concat(["learning_rate", "duration_ms", "samples_per_second", "memory_in_use_bytes"])));
        foreach (var e in epochs)
        {
            var cells = new List<string> { e.Epoch.ToString(inv), e.Loss.ToString("R", inv) };
            cells.AddRange(metricNames.Select(n => e.Metrics.TryGetValue(n, out var v) ? v.ToString("R", inv) : ""));
            cells.Add(e.ValidationLoss?.ToString("R", inv) ?? "");
            cells.AddRange(validationNames.Select(n => e.ValidationMetrics?.TryGetValue(n, out var v) == true ? v.ToString("R", inv) : ""));
            cells.Add(e.LearningRate.ToString("R", inv));
            cells.Add(e.Duration.TotalMilliseconds.ToString("F3", inv));
            cells.Add(e.SamplesPerSecond.ToString("F1", inv));
            cells.Add(e.Memory.InUse.ToString(inv));
            writer.WriteLine(string.Join(',', cells));
        }
    }
}

/// <summary>One telemetry event with the time it was published. <see cref="Event"/> is one of the event structs, boxed.</summary>
public sealed record TelemetryRecord(DateTimeOffset Timestamp, object Event);

/// <summary>
/// Bridges telemetry into a <see cref="Channel{T}"/> so a background consumer (file, database,
/// dashboard, network) can process events without slowing training. The producer never blocks:
/// when the channel is full the oldest event is dropped.
/// </summary>
/// <example>
/// <code>
/// var channel = new ChannelTelemetry(TelemetryLevel.Training | TelemetryLevel.Batches);
/// using var sub = Telemetry.Subscribe(channel);
/// _ = Task.Run(async () =>
/// {
///     await foreach (var record in channel.Reader.ReadAllAsync())
///         if (record.Event is EpochCompleted epoch) await SendToDashboard(epoch);
/// });
/// </code>
/// </example>
public sealed class ChannelTelemetry : ITelemetryHook
{
    private readonly Channel<TelemetryRecord> _channel;

    /// <summary>Creates the bridge.</summary>
    /// <param name="levels">What to capture.</param>
    /// <param name="capacity">Maximum buffered events before the oldest are dropped.</param>
    public ChannelTelemetry(TelemetryLevel levels, int capacity = 100_000)
    {
        Levels = levels;
        _channel = Channel.CreateBounded<TelemetryRecord>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false,
        });
    }

    /// <inheritdoc />
    public TelemetryLevel Levels { get; }

    /// <summary>Where consumers read events from.</summary>
    public ChannelReader<TelemetryRecord> Reader => _channel.Reader;

    /// <summary>Signals that no more events will be written, so <c>ReadAllAsync</c> loops end.</summary>
    public void Complete() => _channel.Writer.TryComplete();

    private void Write(object e) => _channel.Writer.TryWrite(new TelemetryRecord(DateTimeOffset.UtcNow, e));

    /// <inheritdoc />
    public void OnTrainingStarted(in TrainingStarted e) => Write(e);

    /// <inheritdoc />
    public void OnBatchCompleted(in BatchCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnEpochCompleted(in EpochCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnTrainingCompleted(in TrainingCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnLayerForward(in LayerForward e) => Write(e);

    /// <inheritdoc />
    public void OnOperation(in OperationCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnInference(in InferenceCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnToolCall(in ToolCallCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnEngine(in EngineEvent e) => Write(e);
}

/// <summary>
/// Appends every event as one JSON object per line (JSON Lines) to a file, written on a background
/// task through a <see cref="ChannelTelemetry"/>. Dispose it to flush and close the file.
/// </summary>
public sealed class JsonLinesLogger : ITelemetryHook, IDisposable, IAsyncDisposable
{
    private readonly ChannelTelemetry _channel;
    private readonly Task _writer;

    /// <summary>Starts logging to <paramref name="path"/> (overwritten if it exists).</summary>
    public JsonLinesLogger(string path, TelemetryLevel levels = TelemetryLevel.Training | TelemetryLevel.Batches | TelemetryLevel.Inference)
    {
        _channel = new ChannelTelemetry(levels);
        var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16, useAsync: true);
        _writer = Task.Run(() => WriteAsync(stream));
    }

    /// <inheritdoc />
    public TelemetryLevel Levels => _channel.Levels;

    /// <inheritdoc />
    public void OnTrainingStarted(in TrainingStarted e) => _channel.OnTrainingStarted(in e);

    /// <inheritdoc />
    public void OnBatchCompleted(in BatchCompleted e) => _channel.OnBatchCompleted(in e);

    /// <inheritdoc />
    public void OnEpochCompleted(in EpochCompleted e) => _channel.OnEpochCompleted(in e);

    /// <inheritdoc />
    public void OnTrainingCompleted(in TrainingCompleted e) => _channel.OnTrainingCompleted(in e);

    /// <inheritdoc />
    public void OnLayerForward(in LayerForward e) => _channel.OnLayerForward(in e);

    /// <inheritdoc />
    public void OnOperation(in OperationCompleted e) => _channel.OnOperation(in e);

    /// <inheritdoc />
    public void OnInference(in InferenceCompleted e) => _channel.OnInference(in e);

    /// <inheritdoc />
    public void OnToolCall(in ToolCallCompleted e) => _channel.OnToolCall(in e);

    /// <inheritdoc />
    public void OnEngine(in EngineEvent e) => _channel.OnEngine(in e);

    private async Task WriteAsync(FileStream stream)
    {
        await using (stream)
        {
            var buffer = new System.Buffers.ArrayBufferWriter<byte>(4096);
            await foreach (var record in _channel.Reader.ReadAllAsync())
            {
                buffer.ResetWrittenCount();
                using (var json = new Utf8JsonWriter(buffer))
                {
                    TelemetryJson.Write(json, record);
                }

                "\n"u8.CopyTo(buffer.GetSpan(1));
                buffer.Advance(1);
                await stream.WriteAsync(buffer.WrittenMemory);
                if (_channel.Reader.Count == 0)
                {
                    await stream.FlushAsync();
                }
            }
        }
    }

    /// <inheritdoc />
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        _channel.Complete();
        await _writer.ConfigureAwait(false);
    }
}

/// <summary>Reflection-free JSON serialization of telemetry records (trimming and AOT safe).</summary>
public static class TelemetryJson
{
    /// <summary>Writes <paramref name="record"/> as a single JSON object.</summary>
    public static void Write(Utf8JsonWriter w, TelemetryRecord record)
    {
        w.WriteStartObject();
        w.WriteString("time", record.Timestamp);
        switch (record.Event)
        {
            case TrainingStarted e:
                w.WriteString("event", "training_started");
                w.WriteString("device", e.Device.ToString());
                w.WriteString("optimizer", e.Optimizer);
                w.WriteNumber("epochs", e.Epochs);
                w.WriteNumber("training_samples", e.TrainingSamples);
                if (e.ValidationSamples is { } vs) w.WriteNumber("validation_samples", vs);
                w.WriteNumber("batch_size", e.BatchSize);
                w.WriteNumber("batches_per_epoch", e.BatchesPerEpoch);
                w.WriteNumber("parameters", e.ParameterCount);
                w.WriteNumber("learning_rate", e.LearningRate);
                w.WriteNumber("cpu_threads", e.CpuThreads);
                w.WriteString("model", e.Model);
                break;
            case BatchCompleted e:
                w.WriteString("event", "batch");
                w.WriteNumber("epoch", e.Epoch);
                w.WriteNumber("batch", e.Batch);
                w.WriteNumber("step", e.Step);
                w.WriteNumber("batch_size", e.BatchSize);
                Number(w, "loss", e.Loss);
                w.WriteNumber("learning_rate", e.LearningRate);
                if (e.GradientNorm is { } g) Number(w, "gradient_norm", g);
                w.WriteNumber("data_ms", e.DataTime.TotalMilliseconds);
                w.WriteNumber("compute_ms", e.ComputeTime.TotalMilliseconds);
                break;
            case EpochCompleted e:
                w.WriteString("event", "epoch");
                w.WriteNumber("epoch", e.Epoch);
                Number(w, "loss", e.Loss);
                Dictionary(w, "metrics", e.Metrics);
                if (e.ValidationLoss is { } vl) Number(w, "val_loss", vl);
                if (e.ValidationMetrics is { } vm) Dictionary(w, "val_metrics", vm);
                w.WriteNumber("learning_rate", e.LearningRate);
                w.WriteNumber("duration_ms", e.Duration.TotalMilliseconds);
                Number(w, "samples_per_second", e.SamplesPerSecond);
                w.WriteNumber("memory_in_use", e.Memory.InUse);
                w.WriteNumber("memory_cached", e.Memory.Cached);
                w.WriteBoolean("best", e.IsBest);
                break;
            case TrainingCompleted e:
                w.WriteString("event", "training_completed");
                w.WriteNumber("epochs_run", e.EpochsRun);
                w.WriteNumber("duration_ms", e.Duration.TotalMilliseconds);
                Number(w, "final_loss", e.FinalLoss);
                w.WriteNumber("best_epoch", e.BestEpoch);
                Number(w, "best_loss", e.BestLoss);
                w.WriteBoolean("stopped_early", e.StoppedEarly);
                w.WriteBoolean("cancelled", e.Cancelled);
                break;
            case LayerForward e:
                w.WriteString("event", "layer");
                w.WriteString("layer", e.Layer);
                w.WriteString("type", e.LayerType);
                w.WriteNumber("depth", e.Depth);
                Shape(w, "input_shape", e.InputShape);
                Shape(w, "output_shape", e.OutputShape);
                w.WriteNumber("duration_us", e.Duration.TotalMicroseconds);
                w.WriteString("device", e.Device.ToString());
                w.WriteBoolean("training", e.Training);
                break;
            case OperationCompleted e:
                w.WriteString("event", "operation");
                w.WriteString("operation", e.Operation);
                w.WriteBoolean("backward", e.Backward);
                Shape(w, "shape", e.Shape);
                w.WriteString("device", e.Device.ToString());
                w.WriteNumber("duration_us", e.Duration.TotalMicroseconds);
                break;
            case InferenceCompleted e:
                w.WriteString("event", "inference");
                w.WriteString("model", e.Model);
                w.WriteNumber("samples", e.Samples);
                Shape(w, "input_shape", e.InputShape);
                Shape(w, "output_shape", e.OutputShape);
                w.WriteString("device", e.Device.ToString());
                w.WriteNumber("latency_ms", e.Latency.TotalMilliseconds);
                Number(w, "samples_per_second", e.SamplesPerSecond);
                break;
            case ToolCallCompleted e:
                w.WriteString("event", "tool_call");
                w.WriteString("tool", e.Tool);
                w.WriteString("arguments", e.Arguments);
                w.WriteNumber("duration_ms", e.Duration.TotalMilliseconds);
                w.WriteBoolean("succeeded", e.Succeeded);
                if (e.Error is { } error) w.WriteString("error", error);
                break;
            case EngineEvent e:
                w.WriteString("event", "engine");
                w.WriteString("kind", e.Kind.ToString());
                w.WriteString("model", e.Model);
                w.WriteNumber("duration_ms", e.Duration.TotalMilliseconds);
                w.WriteNumber("queue_wait_ms", e.QueueWait.TotalMilliseconds);
                w.WriteNumber("batch_size", e.BatchSize);
                if (e.Reason is { } reason) w.WriteString("reason", reason);
                break;
        }

        w.WriteEndObject();
    }

    // JSON has no NaN/Infinity; write them as null.
    private static void Number(Utf8JsonWriter w, string name, double value)
    {
        if (double.IsFinite(value))
        {
            w.WriteNumber(name, value);
        }
        else
        {
            w.WriteNull(name);
        }
    }

    private static void Dictionary(Utf8JsonWriter w, string name, IReadOnlyDictionary<string, double> values)
    {
        w.WriteStartObject(name);
        foreach (var (key, value) in values)
        {
            Number(w, key, value);
        }

        w.WriteEndObject();
    }

    private static void Shape(Utf8JsonWriter w, string name, int[] shape)
    {
        w.WriteStartArray(name);
        foreach (int d in shape)
        {
            w.WriteNumberValue(d);
        }

        w.WriteEndArray();
    }
}
