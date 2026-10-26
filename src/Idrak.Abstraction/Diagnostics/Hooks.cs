// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

namespace Idrak.Abstraction.Diagnostics;

/// <summary>Writes human-readable progress to the console (or any <see cref="TextWriter"/>).</summary>
/// <param name="levels">
/// What to print. <see cref="TelemetryLevel.Training"/> gives one line per epoch. Device failures
/// (<see cref="TelemetryLevel.Devices"/>) and an app's overrides failing or compared (<see cref="TelemetryLevel.Overrides"/>)
/// are always printed, unless <paramref name="levels"/> is <see cref="TelemetryLevel.None"/>.
/// </param>
/// <param name="epochInterval">Print every n-th epoch (the first and last are always printed; "*" marks a new best).</param>
/// <param name="batchInterval">Print every n-th batch when <see cref="TelemetryLevel.Batches"/> is on.</param>
/// <param name="output">Destination; defaults to <see cref="Console.Out"/>.</param>
public sealed class ConsoleLogger(TelemetryLevel levels = TelemetryLevel.Training, int epochInterval = 1, int batchInterval = 1, TextWriter? output = null)
    : ITelemetryHook
{
    private readonly TextWriter _out = output ?? Console.Out;

    /// <inheritdoc />
    public TelemetryLevel Levels { get; } = levels == TelemetryLevel.None ? levels : levels | TelemetryLevel.Devices | TelemetryLevel.Overrides | TelemetryLevel.Suggestions;

    /// <inheritdoc />
    public void OnTrainingStarted(in TrainingStarted e)
    {
        _out.WriteLine(e.Model);
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Training on {e.Device} ({e.Device.Name}) | {e.TrainingSamples:N0} samples{(e.ValidationSamples is { } v ? string.Create(CultureInfo.InvariantCulture, $", {v:N0} validation") : "")}" +
            $" | batch {e.BatchSize}, {e.BatchesPerEpoch} steps/epoch | {e.Optimizer} lr={e.LearningRate:G4} | {e.ParameterCount:N0} parameters | {e.CpuThreads} CPU threads"));
    }

    /// <inheritdoc />
    public void OnBatchCompleted(in BatchCompleted e)
    {
        if (e.Batch % Math.Max(1, batchInterval) != 0 && e.Batch != e.BatchesPerEpoch)
        {
            return;
        }

        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"  epoch {e.Epoch} batch {e.Batch}/{e.BatchesPerEpoch}  loss {e.Loss:F6}{(e.GradientNorm is { } g ? string.Create(CultureInfo.InvariantCulture, $"  |grad| {g:F4}") : "")}" +
            $"  data {e.DataTime.TotalMilliseconds:F2} ms  compute {e.ComputeTime.TotalMilliseconds:F2} ms"));
    }

    /// <inheritdoc />
    public void OnEpochCompleted(in EpochCompleted e)
    {
        if (e.Epoch != 1 && e.Epoch != e.Epochs && e.Epoch % Math.Max(1, epochInterval) != 0)
        {
            return;
        }

        var line = new StringBuilder();
        line.Append(CultureInfo.InvariantCulture, $"Epoch {e.Epoch.ToString(CultureInfo.InvariantCulture).PadLeft(e.Epochs.ToString(CultureInfo.InvariantCulture).Length)}/{e.Epochs}  loss {e.Loss:F6}");
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
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Finished {e.EpochsRun} epochs in {e.Duration.TotalSeconds:F2} s{(e.StoppedEarly ? " (early stop)" : e.Cancelled ? " (cancelled)" : "")}" +
            $" | best epoch {e.BestEpoch} loss {e.BestLoss:F6}"));

    /// <inheritdoc />
    public void OnLayerForward(in LayerForward e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"  {new string(' ', e.Depth * 2)}{e.Layer}: {Tensor.FormatShape(e.InputShape)} -> {Tensor.FormatShape(e.OutputShape)}  {e.Duration.TotalMicroseconds:F1} µs"));

    /// <inheritdoc />
    public void OnOperation(in OperationCompleted e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    {(e.Backward ? "∇" : " ")}{e.Operation} {Tensor.FormatShape(e.Shape)} on {e.Device}  {e.Duration.TotalMicroseconds:F1} µs"));

    /// <inheritdoc />
    public void OnInference(in InferenceCompleted e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Inference {e.Model}: {e.Samples:N0} samples on {e.Device} in {e.Latency.TotalMilliseconds:F3} ms ({e.SamplesPerSecond:N0} samples/s)"));

    /// <inheritdoc />
    public void OnToolCall(in ToolCallCompleted e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Tool {e.Tool}({e.Arguments}): {(e.Succeeded ? "ok" : "failed: " + e.Error)} in {e.Duration.TotalMilliseconds:F1} ms"));

    /// <inheritdoc />
    public void OnInferenceStage(in InferenceStage e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Read {e.Stage} on {e.Device}: {e.Duration.TotalMilliseconds:F2} ms ({e.Count:N0}{(e.Bytes > 0 ? $", {e.Bytes / 1048576.0:F1} MB" : "")})"));

    /// <inheritdoc />
    public void OnEngine(in EngineEvent e) => _out.WriteLine(e.Kind switch
    {
        EngineEventKind.ModelLoaded => string.Create(CultureInfo.InvariantCulture, $"Engine: loaded {e.Model} in {e.Duration.TotalMilliseconds:F1} ms"),
        EngineEventKind.ModelUnloaded => $"Engine: unloaded {e.Model}",
        EngineEventKind.RequestCompleted => string.Create(CultureInfo.InvariantCulture, $"Engine: {e.Model} request in {e.Duration.TotalMilliseconds:F2} ms (queued {e.QueueWait.TotalMilliseconds:F2} ms, batch {e.BatchSize})"),
        _ => $"Engine: {e.Model} request rejected: {e.Reason}",
    });

    /// <inheritdoc />
    public void OnDeviceFailed(in DeviceFailed e) =>
        _out.WriteLine($"Device {e.Device}{(e.Operation is { } op ? " " + op : "")} failed: {e.Message}"
            + (e.RetriedOnHost ? " (retried on the CPU)" : "") + (e.Hint is { } hint ? Environment.NewLine + "  " + hint : ""));

    /// <inheritdoc />
    public void OnHostFallback(in HostFallback e) =>
        _out.WriteLine(e.KernelsUnavailable is { } why
            ? $"Device {e.Device} {e.Operation} runs on the host: the device's kernels are unavailable ({why}), so each call copies its operands to the CPU and back"
            : $"Device {e.Device} {e.Operation} runs on the host: the device has no kernel of its own for it here, so each call copies its operands to the CPU and back");

    /// <inheritdoc />
    public void OnTuning(in TuningCompleted e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Device {e.Device} tuned {e.Operation}: {e.Candidates} candidates in {e.Duration.TotalMilliseconds:F0} ms (once per shape class{(e.Nested ? ", inside another tuning" : "")})"));

    /// <inheritdoc />
    public void OnDeviceListed(in DeviceListed e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Device {e.Device} listed {(e.Cold ? "cold" : "warm")}: started in {e.Start.TotalMilliseconds:F1} ms, described in {e.Describe.TotalMilliseconds:F1} ms{(e.Error is { } error ? $" (could not start: {error})" : "")}"));

    /// <inheritdoc />
    public void OnGraphRecordingFailed(in GraphRecordingFailed e) =>
        _out.WriteLine($"Device {e.Device} could not record {e.Work} as a graph: {e.Reason}; it runs as ordinary steps");

    /// <inheritdoc />
    public void OnOverrideFailed(in OverrideFailed e) =>
        _out.WriteLine($"Override {e.Registry}/{e.Slot} ({e.Implementation}, {e.Origin}) threw {e.Exception.GetType().Name}: {e.Exception.Message}; "
            + (e.FellBack ? "the library default answered" : "the error reached the caller") + (e.Hint is { } hint ? Environment.NewLine + "  " + hint : ""));

    /// <inheritdoc />
    public void OnOverrideCompared(in OverrideCompared e) =>
        _out.WriteLine($"Shadow {e.Registry}/{e.Slot} ({e.Implementation}): {(e.Agreed ? "agrees" : "differs: " + e.Difference)}; "
                       + string.Create(CultureInfo.InvariantCulture, $"library {e.LibraryTime.TotalMilliseconds:F3} ms, {e.LibraryBytes:N0} B; override {e.OverrideTime.TotalMilliseconds:F3} ms, {e.OverrideBytes:N0} B"));

    /// <inheritdoc />
    public void OnSuggestion(in SettingSuggestion e) => _out.WriteLine($"Suggested for {e.Device}: {e.Setting} = {e.Value} ({e.Reason}) [{e.Source}]");

    /// <inheritdoc />
    public void OnTransfer(in TransferCompleted e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Transfer {e.Transport} {e.From}->{e.To} (tag {e.Tag}): {e.Bytes:N0} B in {e.Wall.TotalMilliseconds:F3} ms") + Simulated(e.Simulated));

    /// <inheritdoc />
    public void OnCollective(in CollectiveCompleted e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Collective {e.Collective} {e.Operation} rank {e.Rank}/{e.WorldSize}: sent {e.BytesSent:N0} B, received {e.BytesReceived:N0} B in {e.Wall.TotalMilliseconds:F3} ms, blocked {e.Blocked.TotalMilliseconds:F3} ms, slowest rank {e.SlowestRank}")
            + Simulated(e.Simulated) + (e.Staging is { } staging ? $", staged on the {staging}" : ""));

    /// <inheritdoc />
    public void OnGradientSync(in GradientsSynced e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Gradients {e.Sync} ({e.Compression}) rank {e.Rank}/{e.WorldSize}{(e.Step >= 0 ? $" step {e.Step}" : "")}: {e.Buckets} bucket(s) of {e.BucketBytes:N0} B, {e.Overlapped} during the backward pass, {e.Bytes:N0} B; "
            + $"backward {e.Backward.TotalMilliseconds:F3} ms, network {e.Network.TotalMilliseconds:F3} ms, exposed {e.Exposed.TotalMilliseconds:F3} ms") + (e.Lost is { } lost ? $"; overlap lost: {lost}" : ""));

    /// <inheritdoc />
    public void OnPipelineStep(in PipelineStepped e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Pipeline {e.Schedule} rank {e.Rank} stage {e.Stage}/{e.Stages}{(e.Replica > 0 ? $" replica {e.Replica}" : "")}{(e.Step >= 0 ? $" step {e.Step}" : "")}: {e.MicroBatches} micro-batch(es) in {e.Wall.TotalMilliseconds:F3} ms, "
            + $"blocked {e.Blocked.TotalMilliseconds:F3} ms, bubble {e.Bubble:P1} (schedule's {e.IdealBubble:P1}), sent {e.BytesSent:N0} B, received {e.BytesReceived:N0} B")
            + (e.Staging is { } staging ? $", activations through the {staging}" : ""));

    /// <inheritdoc />
    public void OnTransportChosen(in TransportChosen e) => _out.WriteLine($"Ranks {e.RankA} and {e.RankB} use {e.Transport} ({e.Reason})");

    /// <inheritdoc />
    public void OnChoicesShared(in MeasuredChoicesShared e) =>
        _out.WriteLine(e.Taken ? string.Create(CultureInfo.InvariantCulture, $"Rank {e.Rank} ({e.Device}) runs rank 0's measured choices ({e.Choices} known): {e.Detail}")
            : $"Rank {e.Rank} ({e.Device}) keeps its own measured choices: {e.Detail}");

    /// <inheritdoc />
    public void OnRankState(in RankStateChanged e) =>
        _out.WriteLine($"Rank {e.Rank} {e.State.ToString().ToLowerInvariant()}" + (e.Step >= 0 ? string.Create(CultureInfo.InvariantCulture, $" at step {e.Step}") : "") + (e.Detail is { } d ? ": " + d : "")
            + (e.Seed is { } seed ? string.Create(CultureInfo.InvariantCulture, $" (replay with --seed {seed} --step {e.Step})") : ""));

    /// <inheritdoc />
    public void OnFault(in FaultInjected e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Fault {e.Kind} at step {e.Step} on rank(s) {string.Join(",", e.Ranks)} (factor {e.Factor:G4}, {(e.Duration > TimeSpan.Zero ? $"{e.Duration.TotalMilliseconds:F1} ms" : "for the rest of the run")}); replay with --seed {e.Seed} --step {e.Step}"));

    /// <inheritdoc />
    public void OnCalibrated(in ProfileCalibrated e) =>
        _out.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"Calibrated {e.Link}: {e.BandwidthBefore / 1e6:F1} -> {e.BandwidthAfter / 1e6:F1} MB/s, latency {e.LatencyBefore.TotalMicroseconds:F1} -> {e.LatencyAfter.TotalMicroseconds:F1} us ({e.Samples} transfers, residual {e.Residual * 1e3:F3} ms)"));

    private static string Simulated(TimeSpan simulated) =>
        simulated > TimeSpan.Zero ? string.Create(CultureInfo.InvariantCulture, $", simulated {simulated.TotalMilliseconds:F3} ms") : "";
}

/// <summary>Keeps every epoch (and optionally batch) event in memory, for charts, reports or CSV export.</summary>
public sealed class MetricsRecorder(TelemetryLevel levels = TelemetryLevel.Training) : ITelemetryHook
{
    private readonly List<EpochCompleted> _epochs = [];
    private readonly List<BatchCompleted> _batches = [];
    private readonly List<CollectiveCompleted> _collectives = [];
    private readonly List<GradientsSynced> _syncs = [];
    private readonly List<PipelineStepped> _pipelineSteps = [];
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

    /// <summary>Recorded collective operations, in order (needs <see cref="TelemetryLevel.Network"/>): each step's network time, bytes and wait.</summary>
    public IReadOnlyList<CollectiveCompleted> Collectives
    {
        get
        {
            lock (_lock)
            {
                return [.. _collectives];
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

    /// <summary>Recorded gradient synchronisations, one per rank and step (needs <see cref="TelemetryLevel.Network"/>): each step's buckets, overlap and exposed network time.</summary>
    public IReadOnlyList<GradientsSynced> Synchronisations
    {
        get
        {
            lock (_lock)
            {
                return [.. _syncs];
            }
        }
    }

    /// <summary>Recorded pipeline steps, one per rank and step (needs <see cref="TelemetryLevel.Network"/>): each stage's time, blocked time and bubble.</summary>
    public IReadOnlyList<PipelineStepped> PipelineSteps
    {
        get
        {
            lock (_lock)
            {
                return [.. _pipelineSteps];
            }
        }
    }

    /// <inheritdoc />
    public void OnCollective(in CollectiveCompleted e)
    {
        lock (_lock)
        {
            _collectives.Add(e);
        }
    }

    /// <inheritdoc />
    public void OnGradientSync(in GradientsSynced e)
    {
        lock (_lock)
        {
            _syncs.Add(e);
        }
    }

    /// <inheritdoc />
    public void OnPipelineStep(in PipelineStepped e)
    {
        lock (_lock)
        {
            _pipelineSteps.Add(e);
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
    public void OnInferenceStage(in InferenceStage e) => Write(e);

    /// <inheritdoc />
    public void OnEngine(in EngineEvent e) => Write(e);

    /// <inheritdoc />
    public void OnDeviceFailed(in DeviceFailed e) => Write(e);

    /// <inheritdoc />
    public void OnHostFallback(in HostFallback e) => Write(e);

    /// <inheritdoc />
    public void OnTuning(in TuningCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnDeviceListed(in DeviceListed e) => Write(e);

    /// <inheritdoc />
    public void OnGraphRecordingFailed(in GraphRecordingFailed e) => Write(e);

    /// <inheritdoc />
    public void OnOverrideFailed(in OverrideFailed e) => Write(e);

    /// <inheritdoc />
    public void OnOverrideCompared(in OverrideCompared e) => Write(e);

    /// <inheritdoc />
    public void OnSuggestion(in SettingSuggestion e) => Write(e);

    /// <inheritdoc />
    public void OnTransfer(in TransferCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnCollective(in CollectiveCompleted e) => Write(e);

    /// <inheritdoc />
    public void OnGradientSync(in GradientsSynced e) => Write(e);

    /// <inheritdoc />
    public void OnPipelineStep(in PipelineStepped e) => Write(e);

    /// <inheritdoc />
    public void OnTransportChosen(in TransportChosen e) => Write(e);

    /// <inheritdoc />
    public void OnChoicesShared(in MeasuredChoicesShared e) => Write(e);

    /// <inheritdoc />
    public void OnRankState(in RankStateChanged e) => Write(e);

    /// <inheritdoc />
    public void OnFault(in FaultInjected e) => Write(e);

    /// <inheritdoc />
    public void OnCalibrated(in ProfileCalibrated e) => Write(e);
}

/// <summary>
/// Appends every event as one JSON object per line (JSON Lines) to a file, written on a background
/// task through a <see cref="ChannelTelemetry"/>. Dispose it to flush and close the file.
/// </summary>
public sealed class JsonLinesLogger : ITelemetryHook, IDisposable, IAsyncDisposable
{
    private readonly ChannelTelemetry _channel;
    private readonly Task _writer;

    /// <summary>
    /// Starts logging to <paramref name="path"/> (overwritten if it exists). Device failures
    /// (<see cref="TelemetryLevel.Devices"/>) and an app's overrides failing or compared
    /// (<see cref="TelemetryLevel.Overrides"/>) are always logged, unless <paramref name="levels"/> is <see cref="TelemetryLevel.None"/>.
    /// </summary>
    public JsonLinesLogger(string path, TelemetryLevel levels = TelemetryLevel.Training | TelemetryLevel.Batches | TelemetryLevel.Inference)
    {
        _channel = new ChannelTelemetry(levels == TelemetryLevel.None ? levels : levels | TelemetryLevel.Devices | TelemetryLevel.Overrides | TelemetryLevel.Suggestions);
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
    public void OnInferenceStage(in InferenceStage e) => _channel.OnInferenceStage(in e);

    /// <inheritdoc />
    public void OnEngine(in EngineEvent e) => _channel.OnEngine(in e);

    /// <inheritdoc />
    public void OnDeviceFailed(in DeviceFailed e) => _channel.OnDeviceFailed(in e);

    /// <inheritdoc />
    public void OnHostFallback(in HostFallback e) => _channel.OnHostFallback(in e);

    /// <inheritdoc />
    public void OnTuning(in TuningCompleted e) => _channel.OnTuning(in e);

    /// <inheritdoc />
    public void OnDeviceListed(in DeviceListed e) => _channel.OnDeviceListed(in e);

    /// <inheritdoc />
    public void OnGraphRecordingFailed(in GraphRecordingFailed e) => _channel.OnGraphRecordingFailed(in e);

    /// <inheritdoc />
    public void OnOverrideFailed(in OverrideFailed e) => _channel.OnOverrideFailed(in e);

    /// <inheritdoc />
    public void OnOverrideCompared(in OverrideCompared e) => _channel.OnOverrideCompared(in e);

    /// <inheritdoc />
    public void OnSuggestion(in SettingSuggestion e) => _channel.OnSuggestion(in e);

    /// <inheritdoc />
    public void OnTransfer(in TransferCompleted e) => _channel.OnTransfer(in e);

    /// <inheritdoc />
    public void OnCollective(in CollectiveCompleted e) => _channel.OnCollective(in e);

    /// <inheritdoc />
    public void OnGradientSync(in GradientsSynced e) => _channel.OnGradientSync(in e);

    /// <inheritdoc />
    public void OnPipelineStep(in PipelineStepped e) => _channel.OnPipelineStep(in e);

    /// <inheritdoc />
    public void OnTransportChosen(in TransportChosen e) => _channel.OnTransportChosen(in e);

    /// <inheritdoc />
    public void OnChoicesShared(in MeasuredChoicesShared e) => _channel.OnChoicesShared(in e);

    /// <inheritdoc />
    public void OnRankState(in RankStateChanged e) => _channel.OnRankState(in e);

    /// <inheritdoc />
    public void OnFault(in FaultInjected e) => _channel.OnFault(in e);

    /// <inheritdoc />
    public void OnCalibrated(in ProfileCalibrated e) => _channel.OnCalibrated(in e);

    private async Task WriteAsync(FileStream stream)
    {
        await using (stream)
        {
            var buffer = new System.Buffers.ArrayBufferWriter<byte>(4096);
            await using var json = new Utf8JsonWriter(buffer);                    // one writer for every record, reset per line
            await foreach (var record in _channel.Reader.ReadAllAsync())
            {
                buffer.ResetWrittenCount();
                json.Reset(buffer);
                TelemetryJson.Write(json, record);
                json.Flush();
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
            case InferenceStage e:
                w.WriteString("event", "inference_stage");
                w.WriteString("stage", e.Stage.ToString());
                w.WriteString("device", e.Device);
                w.WriteNumber("duration_ms", e.Duration.TotalMilliseconds);
                w.WriteNumber("count", e.Count);
                w.WriteNumber("bytes", e.Bytes);
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
            case DeviceFailed e:
                w.WriteString("event", "device_failed");
                w.WriteString("device", e.Device);
                if (e.Operation is { } operation) w.WriteString("operation", operation);
                w.WriteString("message", e.Message);
                w.WriteBoolean("retried_on_host", e.RetriedOnHost);
                if (e.Hint is { } hint) w.WriteString("hint", hint);
                break;
            case HostFallback e:
                w.WriteString("event", "host_fallback");
                w.WriteString("device", e.Device);
                w.WriteString("operation", e.Operation);
                if (e.KernelsUnavailable is { } why) w.WriteString("kernels_unavailable", why);
                break;
            case TuningCompleted e:
                w.WriteString("event", "tuning");
                w.WriteString("device", e.Device);
                w.WriteString("operation", e.Operation);
                w.WriteNumber("candidates", e.Candidates);
                w.WriteNumber("duration_ms", e.Duration.TotalMilliseconds);
                w.WriteBoolean("nested", e.Nested);
                break;
            case DeviceListed e:
                w.WriteString("event", "device_listed");
                w.WriteString("device", e.Device);
                w.WriteNumber("start_ms", e.Start.TotalMilliseconds);
                w.WriteNumber("describe_ms", e.Describe.TotalMilliseconds);
                w.WriteBoolean("cold", e.Cold);
                if (e.Error is { } listedError) w.WriteString("error", listedError);
                break;
            case GraphRecordingFailed e:
                w.WriteString("event", "graph_recording_failed");
                w.WriteString("device", e.Device);
                w.WriteString("work", e.Work);
                w.WriteString("reason", e.Reason);
                break;
            case OverrideFailed e:
                w.WriteString("event", "override_failed");
                w.WriteString("registry", e.Registry);
                w.WriteString("slot", e.Slot);
                w.WriteString("implementation", e.Implementation);
                w.WriteString("origin", e.Origin);
                w.WriteString("error", e.Exception.GetType().FullName);
                w.WriteString("message", e.Exception.Message);
                w.WriteBoolean("fell_back", e.FellBack);
                if (e.Hint is { } overrideHint) w.WriteString("hint", overrideHint);
                break;
            case OverrideCompared e:
                w.WriteString("event", "override_compared");
                w.WriteString("registry", e.Registry);
                w.WriteString("slot", e.Slot);
                w.WriteString("implementation", e.Implementation);
                w.WriteString("origin", e.Origin);
                w.WriteBoolean("agreed", e.Agreed);
                if (e.Difference is { } difference) w.WriteString("difference", difference);
                if (e.Exception is { } thrown) w.WriteString("error", thrown.GetType().FullName);
                w.WriteNumber("library_ms", e.LibraryTime.TotalMilliseconds);
                w.WriteNumber("override_ms", e.OverrideTime.TotalMilliseconds);
                w.WriteNumber("library_bytes", e.LibraryBytes);
                w.WriteNumber("override_bytes", e.OverrideBytes);
                break;
            case SettingSuggestion e:
                w.WriteString("event", "setting_suggestion");
                w.WriteString("setting", e.Setting);
                w.WriteString("value", e.Value);
                w.WriteString("reason", e.Reason);
                w.WriteString("device", e.Device);
                w.WriteString("source", e.Source);
                break;
            case TransferCompleted e:
                w.WriteString("event", "transfer");
                w.WriteString("transport", e.Transport);
                w.WriteNumber("from", e.From);
                w.WriteNumber("to", e.To);
                w.WriteNumber("tag", e.Tag);
                w.WriteNumber("bytes", e.Bytes);
                w.WriteNumber("wall_ms", e.Wall.TotalMilliseconds);
                w.WriteNumber("simulated_ms", e.Simulated.TotalMilliseconds);
                break;
            case CollectiveCompleted e:
                w.WriteString("event", "collective");
                w.WriteString("collective", e.Collective);
                w.WriteString("operation", e.Operation);
                w.WriteNumber("rank", e.Rank);
                w.WriteNumber("world_size", e.WorldSize);
                w.WriteNumber("bytes_sent", e.BytesSent);
                w.WriteNumber("bytes_received", e.BytesReceived);
                w.WriteNumber("wall_ms", e.Wall.TotalMilliseconds);
                w.WriteNumber("simulated_ms", e.Simulated.TotalMilliseconds);
                w.WriteNumber("blocked_ms", e.Blocked.TotalMilliseconds);
                w.WriteNumber("slowest_rank", e.SlowestRank);
                if (e.Staging is { } staging)
                {
                    w.WriteString("staging", staging);
                }

                break;
            case PipelineStepped e:
                w.WriteString("event", "pipeline_step");
                w.WriteString("schedule", e.Schedule);
                w.WriteNumber("rank", e.Rank);
                w.WriteNumber("stage", e.Stage);
                w.WriteNumber("stages", e.Stages);
                w.WriteNumber("replica", e.Replica);
                w.WriteNumber("micro_batches", e.MicroBatches);
                w.WriteNumber("step", e.Step);
                w.WriteNumber("wall_ms", e.Wall.TotalMilliseconds);
                w.WriteNumber("blocked_ms", e.Blocked.TotalMilliseconds);
                Number(w, "bubble", e.Bubble);
                Number(w, "ideal_bubble", e.IdealBubble);
                w.WriteNumber("bytes_sent", e.BytesSent);
                w.WriteNumber("bytes_received", e.BytesReceived);
                if (e.Staging is { } pipelineStaging)
                {
                    w.WriteString("staging", pipelineStaging);
                }

                break;
            case GradientsSynced e:
                w.WriteString("event", "gradient_sync");
                w.WriteString("sync", e.Sync);
                w.WriteString("compression", e.Compression);
                w.WriteNumber("rank", e.Rank);
                w.WriteNumber("world_size", e.WorldSize);
                w.WriteNumber("step", e.Step);
                w.WriteNumber("buckets", e.Buckets);
                w.WriteNumber("overlapped", e.Overlapped);
                w.WriteNumber("bucket_bytes", e.BucketBytes);
                w.WriteNumber("bytes", e.Bytes);
                w.WriteNumber("backward_ms", e.Backward.TotalMilliseconds);
                w.WriteNumber("network_ms", e.Network.TotalMilliseconds);
                w.WriteNumber("exposed_ms", e.Exposed.TotalMilliseconds);
                if (e.Lost is { } lost)
                {
                    w.WriteString("lost", lost);
                }

                break;
            case TransportChosen e:
                w.WriteString("event", "transport_chosen");
                w.WriteNumber("rank_a", e.RankA);
                w.WriteNumber("rank_b", e.RankB);
                w.WriteString("transport", e.Transport);
                w.WriteString("reason", e.Reason);
                break;
            case MeasuredChoicesShared e:
                w.WriteString("event", "choices_shared");
                w.WriteNumber("rank", e.Rank);
                w.WriteString("device", e.Device);
                w.WriteBoolean("taken", e.Taken);
                w.WriteNumber("choices", e.Choices);
                w.WriteString("detail", e.Detail);
                break;
            case RankStateChanged e:
                w.WriteString("event", "rank_state");
                w.WriteNumber("rank", e.Rank);
                w.WriteString("state", e.State.ToString().ToLowerInvariant());
                w.WriteNumber("step", e.Step);
                if (e.Detail is { } detail) w.WriteString("detail", detail);
                if (e.Seed is { } rankSeed) w.WriteNumber("seed", rankSeed);
                break;
            case FaultInjected e:
                w.WriteString("event", "fault");
                w.WriteNumber("seed", e.Seed);
                w.WriteNumber("step", e.Step);
                w.WriteString("kind", e.Kind.ToString());
                w.WriteStartArray("ranks");
                foreach (int rank in e.Ranks)
                {
                    w.WriteNumberValue(rank);
                }

                w.WriteEndArray();
                w.WriteNumber("duration_ms", e.Duration.TotalMilliseconds);
                Number(w, "factor", e.Factor);
                break;
            case ProfileCalibrated e:
                w.WriteString("event", "profile_calibrated");
                w.WriteString("link", e.Link);
                Number(w, "bandwidth_before", e.BandwidthBefore);
                Number(w, "bandwidth_after", e.BandwidthAfter);
                w.WriteNumber("latency_before_us", e.LatencyBefore.TotalMicroseconds);
                w.WriteNumber("latency_after_us", e.LatencyAfter.TotalMicroseconds);
                Number(w, "residual_s", e.Residual);
                w.WriteNumber("samples", e.Samples);
                break;
        }

        w.WriteEndObject();
    }

    // JSON has no NaN or infinity: those are written as null (FiniteJsonNumbers).
    private static void Number(Utf8JsonWriter w, string name, double value) => FiniteJsonNumbers.WriteNumber(w, name, value);

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
