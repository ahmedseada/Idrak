// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Mcp.Abstractions;

namespace Idrak.Mcp.Diagnostics;

// "mcp": what MCP served and called during a run (the telemetry's McpEvent at TelemetryLevel.Mcp, which waits for
// nothing): requests by transport, method and tool (count, errors, p50, p99 and total time, bytes in and out), tool calls
// (finished, failed, cancelled, p50, p99), sessions, refusals by status and reason, cancellations a client sent. Its
// memory does not grow with the calls: each row keeps a log-scale histogram of fixed size for its percentiles.
internal sealed class McpProbe : IProfileProbe
{
    public string Name => "mcp";

    public string Summary => "the Model Context Protocol served and called: requests by method and tool (count, errors, p50, p99, bytes), tool calls, sessions, refusals";

    public bool Intrusive => false;

    public bool Supports(Device device) => true;

    public Action Begin(ProfileRun run)
    {
        var hook = new Sums();
        var subscription = Telemetry.Subscribe(hook);
        return () =>
        {
            subscription.Dispose();
            hook.Record(run);
        };
    }

    // Times in microseconds, 16 buckets an octave (each about 4.4 % wide): a percentile is read within that.
    internal sealed class Histogram
    {
        private const int PerOctave = 16, Buckets = 48 * PerOctave;
        private readonly long[] _counts = new long[Buckets];

        public long Count { get; private set; }

        public void Add(TimeSpan time)
        {
            double us = Math.Max(0, time.TotalMicroseconds);
            _counts[Math.Clamp((int)(Math.Log2(us + 1) * PerOctave), 0, Buckets - 1)]++;
            Count++;
        }

        /// <summary>The <paramref name="quantile"/> (0..1) in ms, from its bucket's middle.</summary>
        public double Quantile(double quantile)
        {
            if (Count == 0)
            {
                return 0;
            }

            long rank = Math.Max(1, (long)Math.Ceiling(quantile * Count)), seen = 0;
            for (int i = 0; i < Buckets; i++)
            {
                seen += _counts[i];
                if (seen >= rank)
                {
                    return (Math.Pow(2, (i + 0.5) / PerOctave) - 1) / 1000;
                }
            }

            return 0;
        }
    }

    private sealed class Row
    {
        public long Calls, Errors, Cancelled, BytesIn, BytesOut;
        public double Ms;
        public readonly Histogram Times = new();
    }

    private sealed class Sums : ITelemetryHook
    {
        private readonly Lock _gate = new();
        private readonly Dictionary<(string Transport, string Method, string Tool), Row> _requests = [];
        private readonly Dictionary<string, Row> _tools = new(StringComparer.Ordinal);
        private readonly Dictionary<(int Status, string Reason), long> _refused = [];
        private readonly Histogram _all = new();
        private long _sessions, _closed, _cancelsSent, _sessionBytesIn, _sessionBytesOut;

        public TelemetryLevel Levels => TelemetryLevel.Mcp;

        public void OnMcp(in McpEvent e)
        {
            lock (_gate)
            {
                switch (e.Kind)
                {
                    case McpEventKind.RequestAnswered:
                    {
                        var key = (e.Transport, e.Method ?? "", e.Tool ?? "");
                        if (!_requests.TryGetValue(key, out var row))
                        {
                            _requests[key] = row = new Row();
                        }

                        row.Calls++;
                        row.Errors += e.Status >= 400 || e.Reason is not null ? 1 : 0;
                        row.BytesIn += e.BytesIn;
                        row.BytesOut += e.BytesOut;
                        row.Ms += e.Duration.TotalMilliseconds;
                        row.Times.Add(e.Duration);
                        _all.Add(e.Duration);
                        break;
                    }

                    case McpEventKind.ToolCallFinished or McpEventKind.ToolCallFailed or McpEventKind.ToolCallCancelled:
                    {
                        string tool = e.Tool ?? "";
                        if (!_tools.TryGetValue(tool, out var row))
                        {
                            _tools[tool] = row = new Row();
                        }

                        row.Calls++;
                        row.Errors += e.Kind == McpEventKind.ToolCallFailed ? 1 : 0;
                        row.Cancelled += e.Kind == McpEventKind.ToolCallCancelled ? 1 : 0;
                        row.Ms += e.Duration.TotalMilliseconds;
                        row.Times.Add(e.Duration);
                        break;
                    }

                    case McpEventKind.Refused:
                        var refusal = (e.Status, e.Reason ?? "");
                        _refused[refusal] = _refused.GetValueOrDefault(refusal) + 1;
                        break;
                    case McpEventKind.SessionOpened:
                        _sessions++;
                        break;
                    case McpEventKind.SessionClosed:
                        _closed++;
                        if (e.Transport != McpServerTransports.Http)
                        {
                            (_sessionBytesIn, _sessionBytesOut) = (_sessionBytesIn + e.BytesIn, _sessionBytesOut + e.BytesOut);
                        }

                        break;
                    case McpEventKind.CancelSent:
                        _cancelsSent++;
                        break;
                }
            }
        }

        public void Record(ProfileRun run)
        {
            lock (_gate)
            {
                if (_requests.Count == 0 && _tools.Count == 0 && _refused.Count == 0 && _sessions == 0 && _cancelsSent == 0)
                {
                    return;                                              // the run used no MCP: its report stays as it was
                }

                run.Record("mcp requests", _all.Count, "calls");
                run.Record("mcp request p50", _all.Quantile(0.5), "ms");
                run.Record("mcp request p99", _all.Quantile(0.99), "ms");
                run.Record("mcp request time", _requests.Values.Sum(r => r.Ms), "ms");
                run.Record("mcp bytes in", (_requests.Values.Sum(r => r.BytesIn) + _sessionBytesIn) / 1024.0, "KB");
                run.Record("mcp bytes out", (_requests.Values.Sum(r => r.BytesOut) + _sessionBytesOut) / 1024.0, "KB");
                run.Record("mcp tool calls", _tools.Values.Sum(r => r.Calls), "calls");
                run.Record("mcp sessions", _sessions);
                run.Record("mcp sessions closed", _closed);
                run.Record("mcp refused", _refused.Values.Sum(), "calls");
                if (_cancelsSent > 0)
                {
                    run.Record("mcp cancels sent", _cancelsSent, "calls");
                }

                if (_requests.Count > 0)
                {
                    run.Add(new ProfileTable("mcp requests", ["transport", "method", "tool", "calls", "errors", "p50 ms", "p99 ms", "total ms", "B in", "B out"],
                        [.. _requests.OrderByDescending(r => r.Value.Ms).Select(r => (IReadOnlyList<string>)[r.Key.Transport, r.Key.Method, r.Key.Tool, Text(r.Value.Calls),
                            Text(r.Value.Errors), Text(r.Value.Times.Quantile(0.5)), Text(r.Value.Times.Quantile(0.99)), Text(r.Value.Ms), Text(r.Value.BytesIn), Text(r.Value.BytesOut)])]));
                }

                if (_tools.Count > 0)
                {
                    run.Add(new ProfileTable("mcp tool calls", ["tool", "calls", "failed", "cancelled", "p50 ms", "p99 ms", "total ms"],
                        [.. _tools.OrderByDescending(t => t.Value.Ms).Select(t => (IReadOnlyList<string>)[t.Key, Text(t.Value.Calls), Text(t.Value.Errors), Text(t.Value.Cancelled),
                            Text(t.Value.Times.Quantile(0.5)), Text(t.Value.Times.Quantile(0.99)), Text(t.Value.Ms)])]));
                }

                if (_refused.Count > 0)
                {
                    run.Add(new ProfileTable("mcp refused", ["status", "reason", "count"],
                        [.. _refused.OrderByDescending(r => r.Value).Select(r => (IReadOnlyList<string>)[Text(r.Key.Status), r.Key.Reason, Text(r.Value)])]));
                }
            }
        }

        private static string Text(double value) => value.ToString(value == Math.Floor(value) ? "0" : "0.###", CultureInfo.InvariantCulture);
    }
}
