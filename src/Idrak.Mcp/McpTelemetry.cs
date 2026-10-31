// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Idrak.Mcp;

// What Idrak.Mcp reports through the library's telemetry (TelemetryLevel.Mcp, McpEvent). Every site reads
// Telemetry.IsEnabled first (one static read and a bitwise AND); nothing below runs, parses or times when nobody listens.
internal static class McpTelemetry
{
    /// <summary>Whether a hook listens at <see cref="TelemetryLevel.Mcp"/>.</summary>
    public static bool Enabled
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Telemetry.IsEnabled(TelemetryLevel.Mcp);
    }

    /// <summary>A timestamp when a hook listens, else 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Start() => Telemetry.Start(TelemetryLevel.Mcp);

    /// <summary>The time since <paramref name="start"/> (0 when it was not taken).</summary>
    public static TimeSpan Since(long start) => start == 0 ? TimeSpan.Zero : Stopwatch.GetElapsedTime(start);

    public static void Emit(McpEventKind kind, string transport, string? session = null, string? method = null, string? tool = null, int status = 0, long bytesIn = 0,
        long bytesOut = 0, long start = 0, string? address = null, string? client = null, string? reason = null) =>
        Telemetry.Mcp(new McpEvent(kind, transport, session, method, tool, status, bytesIn, bytesOut, Since(start), address, client, reason));

    // The tool a tools/call request names.
    public static string? ToolOf(JsonRpcMessage message) =>
        message is JsonRpcRequest { Method: RequestMethods.ToolsCall, Params: JsonObject p } && p["name"] is JsonValue v && v.TryGetValue(out string? name) ? name : null;

    // "name version" of the client: initialize's params.clientInfo, or the 2026-07-28 per-request _meta's.
    public static string? ClientOf(JsonRpcMessage message)
    {
        if (message is not JsonRpcRequest { Params: JsonObject p })
        {
            return null;
        }

        var info = p["clientInfo"] as JsonObject ?? (p["_meta"] as JsonObject)?["io.modelcontextprotocol/clientInfo"] as JsonObject;
        return info is null ? null : $"{(string?)info["name"]} {(string?)info["version"]}".Trim();
    }

    public static string MethodOf(JsonRpcMessage message) => message switch
    {
        JsonRpcRequest r => r.Method,
        JsonRpcNotification n => n.Method,
        JsonRpcError => "error",
        _ => "response",
    };

    // ------------------------------------------------------------------ server tool calls (every transport)

    // Started, finished, cancelled or failed, for every tool of the server (a registry's, a model's, the app's own).
    private static readonly McpRequestFilter<CallToolRequestParams, CallToolResult> ToolFilter = next => (request, cancellationToken) =>
        Enabled ? ObservedAsync(next, request, cancellationToken) : next(request, cancellationToken);

    /// <summary>Adds the tool-call telemetry to <paramref name="server"/>'s filters, once.</summary>
    public static void Attach(McpServerOptions server)
    {
        var filters = server.Filters.Request.CallToolFilters;
        lock (filters)
        {
            if (!filters.Contains(ToolFilter))
            {
                filters.Insert(0, ToolFilter);                       // outermost: its time is the whole call
            }
        }
    }

    private static async ValueTask<CallToolResult> ObservedAsync(McpRequestHandler<CallToolRequestParams, CallToolResult> next, RequestContext<CallToolRequestParams> request,
        CancellationToken cancellationToken)
    {
        string tool = request.Params?.Name ?? "";
        string? session = request.Server.SessionId;
        Emit(McpEventKind.ToolCallStarted, "", session, RequestMethods.ToolsCall, tool);
        long start = Stopwatch.GetTimestamp();
        try
        {
            var result = await next(request, cancellationToken).ConfigureAwait(false);
            if (cancellationToken.IsCancellationRequested)
            {
                Emit(McpEventKind.ToolCallCancelled, "", session, RequestMethods.ToolsCall, tool, start: start);
            }
            else if (result.IsError == true)
            {
                string text = McpTools.TextOf(result.Content);
                Emit(McpEventKind.ToolCallFailed, "", session, RequestMethods.ToolsCall, tool, start: start, reason: text.Length > 200 ? text[..200] + "..." : text);
            }
            else
            {
                Emit(McpEventKind.ToolCallFinished, "", session, RequestMethods.ToolsCall, tool, start: start);
            }

            return result;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            Emit(McpEventKind.ToolCallCancelled, "", session, RequestMethods.ToolsCall, tool, start: start);
            throw;
        }
        catch (Exception e)
        {
            Emit(McpEventKind.ToolCallFailed, "", session, RequestMethods.ToolsCall, tool, start: start, reason: e.Message);
            throw;
        }
    }
}
