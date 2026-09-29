using System.Diagnostics;
using System.Runtime.CompilerServices;
using Idrak.Layers;

namespace Idrak.Diagnostics;

/// <summary>Which kinds of telemetry a hook wants. Sources that nobody asked for are never produced.</summary>
[Flags]
public enum TelemetryLevel
{
    /// <summary>Nothing.</summary>
    None = 0,

    /// <summary>Training start and end, and one summary per epoch (loss, validation loss, metrics, throughput, memory).</summary>
    Training = 1 << 0,

    /// <summary>One event per optimizer step with the batch loss. Reading the loss synchronizes the GPU once per batch.</summary>
    Batches = 1 << 1,

    /// <summary>Gradient L2 norm on every batch event (costs extra kernels and a synchronization per batch).</summary>
    Gradients = 1 << 2,

    /// <summary>Every module's forward pass, with shapes and timing.</summary>
    Layers = 1 << 3,

    /// <summary>Every tensor operation, forward and backward, with shapes and timing. Very verbose.</summary>
    Operations = 1 << 4,

    /// <summary>Every <see cref="Module.Predict(Tensor)"/> call: batch size, latency and throughput.</summary>
    Inference = 1 << 5,

    /// <summary>Every tool call made during a chat (<see cref="Generation.ToolRegistry"/>): name, duration, outcome.</summary>
    Tools = 1 << 6,

    /// <summary>Inference-engine events (<see cref="Inference.InferenceEngine"/>): model loaded or unloaded, request completed or rejected.</summary>
    Engine = 1 << 7,

    /// <summary>Everything.</summary>
    All = Training | Batches | Gradients | Layers | Operations | Inference | Tools | Engine,
}

/// <summary>
/// Receives telemetry. Implement only the methods you need; the others default to no-ops.
/// Hooks are called synchronously on the thread doing the work, so keep them fast, or use
/// <see cref="ChannelTelemetry"/> to hand events to a background consumer.
/// </summary>
public interface ITelemetryHook
{
    /// <summary>The events this hook wants. Read once, when the hook is subscribed.</summary>
    TelemetryLevel Levels { get; }

    /// <summary>A <see cref="Training.Trainer"/> started fitting.</summary>
    void OnTrainingStarted(in TrainingStarted e)
    {
    }

    /// <summary>An optimizer step finished (<see cref="TelemetryLevel.Batches"/>).</summary>
    void OnBatchCompleted(in BatchCompleted e)
    {
    }

    /// <summary>An epoch (including validation) finished.</summary>
    void OnEpochCompleted(in EpochCompleted e)
    {
    }

    /// <summary>Fitting finished, normally or by early stopping.</summary>
    void OnTrainingCompleted(in TrainingCompleted e)
    {
    }

    /// <summary>A module's forward pass finished (<see cref="TelemetryLevel.Layers"/>).</summary>
    void OnLayerForward(in LayerForward e)
    {
    }

    /// <summary>A tensor operation or its gradient finished (<see cref="TelemetryLevel.Operations"/>).</summary>
    void OnOperation(in OperationCompleted e)
    {
    }

    /// <summary>An inference call finished (<see cref="TelemetryLevel.Inference"/>).</summary>
    void OnInference(in InferenceCompleted e)
    {
    }

    /// <summary>A tool call finished (<see cref="TelemetryLevel.Tools"/>).</summary>
    void OnToolCall(in ToolCallCompleted e)
    {
    }

    /// <summary>The inference engine loaded or unloaded a model, or completed or rejected a request (<see cref="TelemetryLevel.Engine"/>).</summary>
    void OnEngine(in EngineEvent e)
    {
    }
}

/// <summary>
/// The telemetry hub. Library code publishes here; hooks subscribe here.
/// With no subscribers every publishing site costs one static field read and a bitwise AND.
/// </summary>
/// <example>
/// <code>
/// using var console = Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training));
/// using var file = Telemetry.Subscribe(new JsonLinesLogger("training.jsonl", TelemetryLevel.Training | TelemetryLevel.Batches));
/// </code>
/// </example>
public static class Telemetry
{
    private static readonly Lock SubscriptionLock = new();
    private static ITelemetryHook[] s_hooks = [];
    private static TelemetryLevel s_levels;

    [ThreadStatic]
    private static int t_layerDepth;

    /// <summary>The union of the levels of all subscribed hooks.</summary>
    public static TelemetryLevel ActiveLevels => s_levels;

    /// <summary>
    /// GPU work runs asynchronously, so by default layer and operation timings measure launch time.
    /// Set this to synchronize the device before each measurement for true execution times
    /// (slower; for profiling only).
    /// </summary>
    public static bool SynchronizeForTiming { get; set; }

    /// <summary>Starts a <see cref="TelemetryBuilder"/>: add hooks, then <see cref="TelemetryBuilder.Start"/> subscribes them all.</summary>
    public static TelemetryBuilder Configure() => new();

    /// <summary>Starts sending events to <paramref name="hook"/>. Dispose the result to unsubscribe.</summary>
    public static IDisposable Subscribe(ITelemetryHook hook)
    {
        ArgumentNullException.ThrowIfNull(hook);
        lock (SubscriptionLock)
        {
            s_hooks = [.. s_hooks, hook];
            Recompute();
        }

        return new Subscription(hook);
    }

    /// <summary>Whether any subscriber wants <paramref name="level"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsEnabled(TelemetryLevel level) => (s_levels & level) != 0;

    private static void Unsubscribe(ITelemetryHook hook)
    {
        lock (SubscriptionLock)
        {
            s_hooks = [.. s_hooks.Where(h => !ReferenceEquals(h, hook))];
            Recompute();
        }
    }

    private static void Recompute()
    {
        var levels = TelemetryLevel.None;
        foreach (var h in s_hooks)
        {
            levels |= h.Levels;
        }

        s_levels = levels;
    }

    /// <summary>Returns a start timestamp when <paramref name="level"/> is enabled, otherwise 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static long Start(TelemetryLevel level) => IsEnabled(level) ? Stopwatch.GetTimestamp() : 0;

    internal static TimeSpan Elapsed(long start, Device? device)
    {
        if (SynchronizeForTiming)
        {
            device?.Synchronize();
        }

        return Stopwatch.GetElapsedTime(start);
    }

    internal static void TrainingStarted(in TrainingStarted e)
    {
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Training) != 0)
            {
                h.OnTrainingStarted(in e);
            }
        }
    }

    internal static void BatchCompleted(in BatchCompleted e)
    {
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Batches) != 0)
            {
                h.OnBatchCompleted(in e);
            }
        }
    }

    internal static void EpochCompleted(in EpochCompleted e)
    {
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Training) != 0)
            {
                h.OnEpochCompleted(in e);
            }
        }
    }

    internal static void TrainingCompleted(in TrainingCompleted e)
    {
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Training) != 0)
            {
                h.OnTrainingCompleted(in e);
            }
        }
    }

    internal static void ToolCall(in ToolCallCompleted e)
    {
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Tools) != 0)
            {
                h.OnToolCall(in e);
            }
        }
    }

    internal static void Engine(in EngineEvent e)
    {
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Engine) != 0)
            {
                h.OnEngine(in e);
            }
        }
    }

    internal static void Inference(in InferenceCompleted e)
    {
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Inference) != 0)
            {
                h.OnInference(in e);
            }
        }
    }

    internal static int EnterLayer() => t_layerDepth++;

    internal static void LeaveLayer(int depth) => t_layerDepth = depth;

    internal static void LayerForward(Module module, Tensor input, Tensor output, long start, int depth)
    {
        t_layerDepth = depth;
        var e = new LayerForward(module.DisplayName, module.GetType().Name, depth, input.Shape.ToArray(), output.Shape.ToArray(),
            Elapsed(start, output.Device), output.Device, module.IsTraining);
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Layers) != 0)
            {
                h.OnLayerForward(in e);
            }
        }
    }

    internal static void Operation(string name, Tensor output, long start, bool backward)
    {
        var e = new OperationCompleted(name, backward, output.Shape.ToArray(), output.Size, output.Device, Elapsed(start, output.Device));
        foreach (var h in s_hooks)
        {
            if ((h.Levels & TelemetryLevel.Operations) != 0)
            {
                h.OnOperation(in e);
            }
        }
    }

    private sealed class Subscription(ITelemetryHook hook) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                Unsubscribe(hook);
            }
        }
    }
}
