// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;
using Idrak.Mcp.Abstractions;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Idrak.Mcp.Transports;

// "stdio": one session over the process's standard input and output (or the request's streams), the SDK's stream
// transport behind the contract. It ends when the client closes its end, or when it is stopped. The session's messages
// pass through an observer (ObservedTransport) that reports them to the telemetry when a hook listens.
internal sealed class StdioServerTransportKind : IMcpServerTransport
{
    public string Name => McpServerTransports.Stdio;

    public string Summary => "the process's standard input and output: one client, the one that started the process";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public ValueTask<IMcpListener> StartAsync(McpTransportRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        HttpServerTransportKind.CheckKeys(this, request);
        cancellationToken.ThrowIfCancellationRequested();
        McpTelemetry.Attach(request.Server);
        var input = new CountingStream(request.Input ?? Console.OpenStandardInput());
        var output = new CountingStream(request.Output ?? Console.OpenStandardOutput());
        string name = request.Server.ServerInfo?.Name ?? "idrak";
        var transport = new ObservedTransport(new StreamServerTransport(input, output, name), input, output);
        return ValueTask.FromResult<IMcpListener>(new Listener(McpServer.Create(transport, request.Server), transport));
    }

    private sealed class Listener : IMcpListener
    {
        private readonly McpServer _server;
        private readonly ObservedTransport _transport;
        private readonly CancellationTokenSource _stop = new();
        private int _disposed;

        public Listener(McpServer server, ObservedTransport transport)
        {
            (_server, _transport) = (server, transport);
            Completion = RunAsync();
        }

        public string Address => McpServerTransports.Stdio;

        public string? Token => null;

        public Task Completion { get; }

        private async Task RunAsync()
        {
            string reason = "the client left";
            try
            {
                await _server.RunAsync(_stop.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (_stop.IsCancellationRequested)
            {
                reason = "the server stopped";
            }
            finally
            {
                _transport.Closed(_stop.IsCancellationRequested ? "the server stopped" : reason, _server.ClientInfo);
            }
        }

        public async Task StopAsync()
        {
            if (Volatile.Read(ref _disposed) == 1)
            {
                await Completion.ConfigureAwait(false);
                return;
            }

            await _stop.CancelAsync().ConfigureAwait(false);
            try
            {
                await Completion.ConfigureAwait(false);
            }
            finally
            {
                if (Interlocked.Exchange(ref _disposed, 1) == 0)
                {
                    await _server.DisposeAsync().ConfigureAwait(false);
                    _stop.Dispose();
                }
            }
        }

        public async ValueTask DisposeAsync() => await StopAsync().ConfigureAwait(false);
    }
}

// The SDK's transport with its messages watched: a request read is timed to its answer, a session opens at the client's
// first request and closes with the bytes its streams carried. With no hook at TelemetryLevel.Mcp, each message costs
// the level's read and an empty-dictionary check; nothing is parsed, timed or kept.
internal sealed class ObservedTransport : ITransport
{
    private readonly ITransport _inner;
    private readonly CountingStream _input, _output;
    private readonly ConcurrentDictionary<RequestId, (long Start, string Method, string? Tool)> _pending = new();
    private readonly long _created = Stopwatch.GetTimestamp();
    private int _opened;

    public ObservedTransport(ITransport inner, CountingStream input, CountingStream output)
    {
        (_inner, _input, _output) = (inner, input, output);
        MessageReader = new Reader(this, inner.MessageReader);
    }

    public string? SessionId => _inner.SessionId;

    public ChannelReader<JsonRpcMessage> MessageReader { get; }

    public Task SendMessageAsync(JsonRpcMessage message, CancellationToken cancellationToken = default)
    {
        if (!_pending.IsEmpty && message is JsonRpcMessageWithId answer and (JsonRpcResponse or JsonRpcError) && _pending.TryRemove(answer.Id, out var asked))
        {
            Answered(asked, (message as JsonRpcError)?.Error.Message);
        }

        return _inner.SendMessageAsync(message, cancellationToken);
    }

    public ValueTask DisposeAsync() => _inner.DisposeAsync();

    private void Answered((long Start, string Method, string? Tool) asked, string? reason) =>
        McpTelemetry.Emit(McpEventKind.RequestAnswered, McpServerTransports.Stdio, null, asked.Method, asked.Tool, start: asked.Start, reason: reason);

    private void Read(JsonRpcMessage message)
    {
        if (!McpTelemetry.Enabled)
        {
            return;
        }

        string method = McpTelemetry.MethodOf(message);
        string? tool = McpTelemetry.ToolOf(message);
        if (message is JsonRpcRequest request)
        {
            if (Interlocked.Exchange(ref _opened, 1) == 0)
            {
                McpTelemetry.Emit(McpEventKind.SessionOpened, McpServerTransports.Stdio, address: McpServerTransports.Stdio, client: McpTelemetry.ClientOf(request));
            }

            _pending[request.Id] = (Stopwatch.GetTimestamp(), method, tool);
        }
        else if (message is JsonRpcNotification { Method: NotificationMethods.CancelledNotification, Params: { } p }
                 && p["requestId"] is { } id && _pending.TryRemove(RequestIdOf(id), out var asked))
        {
            Answered(asked, "cancelled by the client");           // a cancelled request is answered with nothing
        }

        McpTelemetry.Emit(McpEventKind.RequestReceived, McpServerTransports.Stdio, null, method, tool);
    }

    private static RequestId RequestIdOf(System.Text.Json.Nodes.JsonNode id) =>
        id is System.Text.Json.Nodes.JsonValue v && v.TryGetValue(out long number) ? new RequestId(number) : new RequestId(id.ToString());

    /// <summary>The session ended: its lifetime, its client and the bytes it read and wrote.</summary>
    public void Closed(string reason, Implementation? client)
    {
        _pending.Clear();
        if (McpTelemetry.Enabled && Volatile.Read(ref _opened) == 1)
        {
            McpTelemetry.Emit(McpEventKind.SessionClosed, McpServerTransports.Stdio, bytesIn: _input.Count, bytesOut: _output.Count, start: _created,
                address: McpServerTransports.Stdio, client: client is null ? null : $"{client.Name} {client.Version}", reason: reason);
        }
    }

    private sealed class Reader(ObservedTransport owner, ChannelReader<JsonRpcMessage> inner) : ChannelReader<JsonRpcMessage>
    {
        public override Task Completion => inner.Completion;

        public override bool TryRead(out JsonRpcMessage item)
        {
            if (!inner.TryRead(out item!))
            {
                return false;
            }

            owner.Read(item);
            return true;
        }

        public override ValueTask<bool> WaitToReadAsync(CancellationToken cancellationToken = default) => inner.WaitToReadAsync(cancellationToken);
    }
}

// Counts the bytes through a stream (a stdio session's totals); everything else is the inner stream's.
internal sealed class CountingStream(Stream inner) : Stream
{
    private long _count;

    public long Count => Volatile.Read(ref _count);

    public override bool CanRead => inner.CanRead;

    public override bool CanSeek => false;

    public override bool CanWrite => inner.CanWrite;

    public override long Length => throw new NotSupportedException();

    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

    public override int Read(byte[] buffer, int offset, int count) => Add(inner.Read(buffer, offset, count));

    public override int Read(Span<byte> buffer) => Add(inner.Read(buffer));

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
        Add(await inner.ReadAsync(buffer, cancellationToken).ConfigureAwait(false));

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    public override void Write(byte[] buffer, int offset, int count)
    {
        inner.Write(buffer, offset, count);
        Add(count);
    }

    public override void Write(ReadOnlySpan<byte> buffer)
    {
        inner.Write(buffer);
        Add(buffer.Length);
    }

    public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
    {
        Add(buffer.Length);
        return inner.WriteAsync(buffer, cancellationToken);
    }

    public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        WriteAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    public override void Flush() => inner.Flush();

    public override Task FlushAsync(CancellationToken cancellationToken) => inner.FlushAsync(cancellationToken);

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    public override void SetLength(long value) => throw new NotSupportedException();

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            inner.Dispose();
        }

        base.Dispose(disposing);
    }

    public override ValueTask DisposeAsync() => inner.DisposeAsync();

    private int Add(int count)
    {
        Interlocked.Add(ref _count, count);
        return count;
    }
}
