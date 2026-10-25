// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Cli.Shared;

/// <summary>
/// <c>--verbose</c> (or <c>IDRAK_VERBOSE=1</c>), common to every command: what the library does, printed as it happens,
/// besides the lines a command prints with it: each prompt pass (rows, positions, pieces, time), decoding summed every
/// few seconds and before the next prompt pass, the vision encoder and sound features, each kernel choice a device
/// measured, and the first host fallback of each operation. On the error output, over the progress line (which redraws
/// under them); the main output and <c>--json</c> stay as they are. Stages are read at <c>TelemetryLevel.Stages</c>,
/// which waits for nothing, so the run's timing is the run's. <c>--profile</c> is the report at the end; both can be on.
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

    public TelemetryLevel Levels => TelemetryLevel.Stages | TelemetryLevel.Devices;

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

    public void OnHostFallback(in HostFallback e)
    {
        lock (_lock)
        {
            Line($"{e.Operation} ran on the host: {e.Device} has no kernel for it (its later calls are counted in the report)");
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
