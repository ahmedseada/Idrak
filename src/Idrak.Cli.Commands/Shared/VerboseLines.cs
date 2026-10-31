// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Cli.Shared;

/// <summary>
/// <c>--verbose</c> (or <c>IDRAK_VERBOSE=1</c>), common to every command: what the library does, printed as it happens,
/// besides the lines a command prints with it: each prompt pass (rows, positions, pieces, time), decoding summed every
/// few seconds and before the next prompt pass, the vision encoder and sound features, each kernel choice a device
/// measured, each kernel set a device loaded (from the kernel cache or made first, with its time and size), the first
/// host fallback of each operation, each device failure (a kernel refused, a launch or runtime call
/// that failed, kernels a profile cannot time: each different one once), and each graph recording that failed (with its
/// reason), and the Model Context Protocol served and called (each session, request with its method, tool, status, bytes
/// and time, tool call, refusal, and cancellation sent). On the error output, over the progress line (which redraws under it); the main output and <c>--json</c> stay
/// as they are.
/// Stages are read at <c>TelemetryLevel.Stages</c>, which waits for nothing, so the run's timing is the run's.
/// <c>--profile</c> is the report at the end; both can be on. The test runner takes both too.
/// </summary>
internal sealed class VerboseLines : ITelemetryHook, IDisposable
{
    /// <summary>The variable that turns the lines on for every command, as <c>--verbose</c> does: 1, true, on or yes.</summary>
    internal const string Variable = "IDRAK_VERBOSE";

    private static readonly long FlushTicks = Stopwatch.Frequency * 5;
    private readonly Lock _lock = new();
    private readonly TextWriter _error;
    private readonly bool _console;
    private readonly IDisposable _subscription;
    private readonly HashSet<(string Device, string? Operation, string Message)> _failures = [];
    private long _tokens, _chunks, _flushed = Stopwatch.GetTimestamp();
    private TimeSpan _decoding;

    internal VerboseLines(TextWriter error)
    {
        _error = error;
        _console = ReferenceEquals(error, Console.Error) && !Console.IsErrorRedirected;
        _subscription = Telemetry.Subscribe(this);
    }

    /// <summary>The lines for the command <paramref name="context"/> runs when it is verbose (<see cref="CommandContext.Verbose"/>); null otherwise.</summary>
    internal static VerboseLines? Start(CommandContext context) => context.Verbose ? new VerboseLines(context.ErrorOutput) : null;

    /// <summary>Whether <see cref="Variable"/> turns verbose output on.</summary>
    /// <exception cref="UsageException">It holds something else than on or off.</exception>
    internal static bool Requested()
    {
        string? value = Environment.GetEnvironmentVariable(Variable)?.Trim();
        return value?.ToLowerInvariant() switch
        {
            null or "" or "0" or "false" or "off" or "no" => false,
            "1" or "true" or "on" or "yes" => true,
            _ => throw new UsageException($"{Variable}={value}: give 1 (every command verbose, as --verbose) or 0."),
        };
    }

    public TelemetryLevel Levels => TelemetryLevel.Stages | TelemetryLevel.Devices | TelemetryLevel.Mcp;

    public void OnMcp(in McpEvent e)
    {
        lock (_lock)
        {
            Line($"mcp: {e.Describe()}");
        }
    }

    public void OnInferenceStage(in InferenceStage e)
    {
        lock (_lock)
        {
            switch (e.Stage)
            {
                case InferenceStageKind.Prefill:
                    FlushLocked();
                    long each = e.Count / Math.Max(1, e.Rows);
                    string rows = e.Rows > 1 ? $"{e.Rows} rows × " : "";
                    string pieces = e.Pieces > 1 ? $" in {e.Pieces} pieces of {e.PieceSize:N0}" : "";
                    Line($"prompt pass on {e.Device}: {rows}{each:N0} positions{pieces}, {Time(e.Duration)}");
                    break;
                case InferenceStageKind.DecodeChunk:
                    (_tokens, _chunks, _decoding) = (_tokens + e.Count, _chunks + 1, _decoding + e.Duration);
                    if (Stopwatch.GetTimestamp() - _flushed >= FlushTicks)
                    {
                        FlushLocked();
                    }

                    break;
                case InferenceStageKind.Encoder:
                    Line($"vision encoder on {e.Device}: {e.Count:N0} images, {Time(e.Duration)}");
                    break;
                case InferenceStageKind.AudioFeatures:
                    Line($"sound features on {e.Device}: {e.Count:N0} frames, {Time(e.Duration)}");
                    break;
            }
        }
    }

    public void OnTuning(in TuningCompleted e)
    {
        lock (_lock)
        {
            Line($"measured {e.Operation} on {e.Device}: {e.Candidates} choices, {Time(e.Duration)}{(e.Nested ? " (within another measuring)" : "")}");
        }
    }

    public void OnKernelsLoaded(in KernelsLoaded e)
    {
        lock (_lock)
        {
            string how = e.FromCache ? "from the kernel cache (warm)" : "made and loaded (cold)";
            Line($"kernels on {e.Device}: {e.Kernels} {e.Backend} kernels {how}, {Time(e.Duration)}, {e.Bytes:N0} bytes{(e.Detail is { } detail ? $" ({detail})" : "")}");
        }
    }

    public void OnHostFallback(in HostFallback e)
    {
        lock (_lock)
        {
            if (e.KernelsUnavailable is { } why)
            {
                Line($"{e.Operation} ran on the host: {e.Device}'s kernels are unavailable ({why}; its later calls are counted in the report)");
            }
            else
            {
                Line($"{e.Operation} ran on the host: {e.Device} has no kernel for it (its later calls are counted in the report)");
            }
        }
    }

    public void OnDeviceFailed(in DeviceFailed e)
    {
        lock (_lock)
        {
            // Each different failure once: an operation retried on the host on every call would repeat it.
            if (_failures.Add((e.Device, e.Operation, e.Message)))
            {
                string operation = e.Operation is { } op ? $" ({op})" : "";
                Line($"{e.Device} failed{operation}: {e.Message}{(e.RetriedOnHost ? "; retried on the CPU" : "")}");
            }
        }
    }

    public void OnGraphRecordingFailed(in GraphRecordingFailed e)
    {
        lock (_lock)
        {
            Line($"{e.Device} could not record {e.Work} as a graph: {e.Reason}; it runs as ordinary steps");
        }
    }

    /// <summary>Stops listening and prints the decoding not printed yet.</summary>
    public void Dispose()
    {
        _subscription.Dispose();
        lock (_lock)
        {
            FlushLocked();
        }
    }

    private void FlushLocked()
    {
        if (_chunks > 0)
        {
            double seconds = Math.Max(_decoding.TotalSeconds, 1e-9);
            Line($"decoding: {_tokens:N0} tokens in {_chunks:N0} chunks, {Time(_decoding)} ({_tokens / seconds:N0} tokens/s)");
            (_tokens, _chunks, _decoding) = (0, 0, TimeSpan.Zero);
        }

        _flushed = Stopwatch.GetTimestamp();
    }

    private void Line(FormattableString text)
    {
        string line = "verbose: " + text.ToString(CultureInfo.InvariantCulture);
        if (_console)
        {
            int width = 0;
            try
            {
                width = Console.BufferWidth - 1;
            }
            catch (IOException)
            {
            }

            _error.WriteLine("\r" + line.PadRight(width));                  // over the progress line, which redraws under it
        }
        else
        {
            _error.WriteLine(line);
        }
    }

    private static string Time(TimeSpan time) => time.TotalSeconds < 1
        ? string.Create(CultureInfo.InvariantCulture, $"{time.TotalMilliseconds:0} ms")
        : string.Create(CultureInfo.InvariantCulture, $"{time.TotalSeconds:0.00} s");
}
