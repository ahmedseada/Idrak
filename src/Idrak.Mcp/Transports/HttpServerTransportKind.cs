// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Text;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using Idrak.Mcp.Abstractions;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Idrak.Mcp.Transports;

// "http": MCP streamable HTTP (POST a JSON-RPC message, answered as server-sent events; GET the session's own event
// stream; DELETE ends the session) on a socket of its own, so it needs no web framework and behaves alike on every
// platform (the system's HTTP listeners differ: one needs a URL reservation for 127.0.0.1 and cannot take port 0). The
// protocol's session logic is the SDK's StreamableHttpServerTransport, one per session; this file is the HTTP/1.1 around it.
internal sealed class HttpServerTransportKind : IMcpServerTransport
{
    public string Name => McpServerTransports.Http;

    public string Summary => "streamable HTTP on a loopback port (the system picks one unless asked), a bearer token required: any number of clients, each its own session";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    // Any option key the transport does not take is refused, naming the ones it does.
    internal static void CheckKeys(IMcpServerTransport transport, McpTransportRequest request)
    {
        foreach (string key in request.Options?.Keys ?? [])
        {
            if (!transport.Keys.Contains(key))
            {
                throw new ArgumentException($"The MCP transport '{transport.Name}' takes no option '{key}' (it takes: {(transport.Keys.Count == 0 ? "none" : string.Join(", ", transport.Keys))}).", nameof(request));
            }
        }
    }

    public ValueTask<IMcpListener> StartAsync(McpTransportRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Server);
        CheckKeys(this, request);
        cancellationToken.ThrowIfCancellationRequested();
        McpTelemetry.Attach(request.Server);
        var endPoint = ParseAddress(request.Address, request.AllowRemote);
        if (request.Path is not { Length: > 0 } path || path[0] != '/' || path.AsSpan().ContainsAny(" ?#\r\n"))
        {
            throw new ArgumentException($"The MCP path '{request.Path}' must start with '/' and hold no space, '?' or '#'.", nameof(request));
        }

        if (request.MaxRequestBytes < 1024)
        {
            throw new ArgumentException($"The largest MCP request ({request.MaxRequestBytes} bytes) must be at least 1024 bytes.", nameof(request));
        }

        string? token = request.RequireToken ? request.Token is { Length: > 0 } given ? given : NewToken(32) : null;
        if (token is not null && token.AsSpan().ContainsAny(" \t\r\n"))
        {
            throw new ArgumentException("The MCP bearer token may not hold spaces or line breaks.", nameof(request));
        }

        var socket = new Socket(endPoint.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
        try
        {
            if (endPoint.AddressFamily == AddressFamily.InterNetworkV6 && endPoint.Address.Equals(IPAddress.IPv6Any))
            {
                socket.DualMode = true;
            }

            socket.Bind(endPoint);
            socket.Listen(512);
        }
        catch
        {
            socket.Dispose();
            throw;
        }

        return ValueTask.FromResult<IMcpListener>(new HttpMcpListener(socket, request, path, token));
    }

    // host:port, [v6]:port, :port or port; localhost is the loopback address. Off this machine only when allowed.
    internal static IPEndPoint ParseAddress(string? address, bool allowRemote)
    {
        string text = string.IsNullOrWhiteSpace(address) ? "127.0.0.1:0" : address.Trim();
        if (text.StartsWith("http://", StringComparison.OrdinalIgnoreCase))
        {
            text = text["http://".Length..].TrimEnd('/');
        }

        string host;
        string port;
        int colon = text.LastIndexOf(':');
        if (text.StartsWith('[') && text.IndexOf(']') is var close and > 0)
        {
            host = text[1..close];
            port = close + 1 < text.Length && text[close + 1] == ':' ? text[(close + 2)..] : "0";
        }
        else if (colon >= 0 && text.IndexOf(':') == colon)
        {
            host = colon == 0 ? "127.0.0.1" : text[..colon];
            port = text[(colon + 1)..];
        }
        else if (int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out _))
        {
            host = "127.0.0.1";
            port = text;
        }
        else
        {
            host = text;
            port = "0";
        }

        if (!int.TryParse(port, NumberStyles.None, CultureInfo.InvariantCulture, out int number) || number > IPEndPoint.MaxPort)
        {
            throw new ArgumentException($"The MCP address '{address}' has no valid port (host:port, port 0: the system picks one).", nameof(address));
        }

        IPAddress ip;
        if (host.Equals("localhost", StringComparison.OrdinalIgnoreCase))
        {
            ip = IPAddress.Loopback;
        }
        else if (!IPAddress.TryParse(host, out ip!))
        {
            ip = (Dns.GetHostAddresses(host) is { Length: > 0 } found ? found[0] : null)
                 ?? throw new ArgumentException($"The MCP address '{address}' names a host that does not resolve.", nameof(address));
        }

        if (!IPAddress.IsLoopback(ip) && !allowRemote)
        {
            throw new ArgumentException($"The MCP address '{address}' is reachable from other machines; MCP listens on this machine only (127.0.0.1) unless "
                                        + "remote clients are allowed explicitly (McpHostOptions.AllowRemote, --mcp-allow-remote).", nameof(address));
        }

        return new IPEndPoint(ip, number);
    }

    // Random bytes as URL-safe base64 (a token, a session id).
    internal static string NewToken(int bytes)
    {
        Span<byte> random = stackalloc byte[bytes];
        RandomNumberGenerator.Fill(random);
        return Base64Url.EncodeToString(random);
    }
}

// The listener: accepts connections, serves each on its own task, keeps the sessions by id and closes idle ones.
internal sealed class HttpMcpListener : IMcpListener
{
    internal const int MaxHead = 64 * 1024;          // the largest request line and headers (431 beyond)
    private const int ReadBuffer = 16 * 1024;

    private static readonly byte[] Continue = "HTTP/1.1 100 Continue\r\n\r\n"u8.ToArray();

    private static readonly JsonTypeInfo<JsonRpcMessage> MessageType = (JsonTypeInfo<JsonRpcMessage>)McpJsonUtilities.DefaultOptions.GetTypeInfo(typeof(JsonRpcMessage));

    private static readonly JsonTypeInfo<CancelledNotificationParams> CancelType =
        (JsonTypeInfo<CancelledNotificationParams>)McpJsonUtilities.DefaultOptions.GetTypeInfo(typeof(CancelledNotificationParams));

    private readonly Socket _socket;
    private readonly McpTransportRequest _request;
    private readonly byte[] _path;
    private readonly byte[]? _token;
    private readonly CancellationTokenSource _stop = new();
    private readonly ConcurrentDictionary<string, Session> _sessions = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, Session>.AlternateLookup<ReadOnlySpan<char>> _sessionsBySpan;
    private readonly ConcurrentDictionary<NetworkStream, Task> _connections = new();
    private readonly TaskCompletionSource _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly Timer? _sweep;
    private readonly Task _accepting;
    private string? _lastVersion;                    // the last Mcp-Protocol-Version read, kept to read it without a new string
    private int _stopping;
    private int _open;                               // sessions open (and being opened), against MaxSessions

    // The sessionless requests in flight by id (2026-07-28 and later: no session to send a notifications/cancelled to),
    // so a cancellation finds its call; an id two clients use at once is ambiguous, and cancelled by its connection only.
    private readonly Dictionary<RequestId, (CancellationTokenSource? Abort, int Count)> _sessionless = [];

    public HttpMcpListener(Socket socket, McpTransportRequest request, string path, string? token)
    {
        _socket = socket;
        _request = request;
        _path = Encoding.UTF8.GetBytes(path);
        Token = token;
        _token = token is null ? null : Encoding.UTF8.GetBytes(token);
        _sessionsBySpan = _sessions.GetAlternateLookup<ReadOnlySpan<char>>();
        var local = (IPEndPoint)socket.LocalEndPoint!;
        string host = local.Address.Equals(IPAddress.Any) ? "127.0.0.1" : local.Address.Equals(IPAddress.IPv6Any) ? "[::1]"
            : local.AddressFamily == AddressFamily.InterNetworkV6 ? $"[{local.Address}]" : local.Address.ToString();
        Address = $"http://{host}:{local.Port.ToString(CultureInfo.InvariantCulture)}{path}";
        if (request.SessionIdleTimeout > TimeSpan.Zero)
        {
            var period = TimeSpan.FromTicks(Math.Clamp(request.SessionIdleTimeout.Ticks / 4, TimeSpan.TicksPerSecond, TimeSpan.TicksPerMinute));
            _sweep = new Timer(static s => ((HttpMcpListener)s!).CloseIdle(), this, period, period);
        }

        _accepting = AcceptAsync();
    }

    public string Address { get; }

    public string? Token { get; }

    public Task Completion => _completion.Task;

    /// <summary>The sessions open now (tests).</summary>
    internal int SessionCount => _sessions.Count;

    private async Task AcceptAsync()
    {
        try
        {
            while (!_stop.IsCancellationRequested)
            {
                var client = await _socket.AcceptAsync(_stop.Token).ConfigureAwait(false);
                client.NoDelay = true;
                var stream = new NetworkStream(client, ownsSocket: true);
                _connections[stream] = Task.CompletedTask;                       // listed first: it may end before Run returns
                _connections.TryUpdate(stream, Task.Run(() => ServeConnectionAsync(stream)), Task.CompletedTask);
            }
        }
        catch (Exception e) when (e is OperationCanceledException or ObjectDisposedException or SocketException && _stop.IsCancellationRequested)
        {
            // stopped
        }
        catch (Exception e)
        {
            _ = StopCoreAsync(e);
        }
    }

    // ------------------------------------------------------------------ one connection: requests one after another

    private async Task ServeConnectionAsync(NetworkStream stream)
    {
        byte[] buffer = ArrayPool<byte>.Shared.Rent(ReadBuffer);
        int filled = 0;
        Task<int>? pending = null;          // a read started while a request was answered (it tells a client that left)
        try
        {
            while (!_stop.IsCancellationRequested)
            {
                int headEnd;
                while ((headEnd = buffer.AsSpan(0, filled).IndexOf("\r\n\r\n"u8)) < 0)
                {
                    if (filled == buffer.Length)
                    {
                        if (buffer.Length >= MaxHead)
                        {
                            await RefuseAsync(stream, default, 431, "Request Header Fields Too Large"u8).ConfigureAwait(false);
                            return;
                        }

                        buffer = Grow(buffer, filled, buffer.Length * 2);
                    }

                    int read = pending is null ? await stream.ReadAsync(buffer.AsMemory(filled), _stop.Token).ConfigureAwait(false) : await pending.ConfigureAwait(false);
                    pending = null;
                    if (read == 0)
                    {
                        return;
                    }

                    filled += read;
                }

                if (!Head.TryParse(buffer.AsSpan(0, headEnd + 4), out var head))
                {
                    await RefuseAsync(stream, default, 400, "Bad Request: the request line or a header cannot be read"u8).ConfigureAwait(false);
                    return;
                }

                // The answer leaves what follows the request (a pipelined next one) at the front of the buffer, and may
                // leave a read open into the buffer after it (started while a call ran, to see a client that leaves).
                var (left, close, open) = await AnswerAsync(stream, head, buffer, headEnd + 4, filled).ConfigureAwait(false);
                pending = open;
                if (close)
                {
                    return;
                }

                filled = left;
            }
        }
        catch (Exception e) when (e is OperationCanceledException or IOException or SocketException or ObjectDisposedException)
        {
            // the client left, or the listener stopped
        }
        finally
        {
            if (pending is not null)
            {
                try
                {
                    await stream.DisposeAsync().ConfigureAwait(false);   // ends the pending read before its buffer goes back
                    await pending.ConfigureAwait(false);
                }
                catch (Exception)
                {
                    // the read ends with the socket
                }
            }

            await stream.DisposeAsync().ConfigureAwait(false);
            ArrayPool<byte>.Shared.Return(buffer);
            _connections.TryRemove(stream, out _);
        }
    }

    // Answers the request whose head is buffer[..consumed]; its body starts there. Returns how much of the buffer is
    // left once the request's bytes are dropped (moved to the front), whether to close, and a read kept open on the
    // connection (into the buffer after what is left) when there is one.
    private async Task<(int Filled, bool Close, Task<int>? Pending)> AnswerAsync(NetworkStream stream, Head head, byte[] buffer, int consumed, int filled)
    {
        long start = McpTelemetry.Start();
        var request = buffer.AsMemory(0, consumed);
        bool close = head.Close;
        if (!head.PathIs(request.Span, _path))
        {
            await RefuseAsync(stream, head, 404, "Not Found"u8).ConfigureAwait(false);
            return (0, true, null);
        }

        if (!OriginAllowed(head, request.Span))
        {
            await RefuseAsync(stream, head, 403, "Forbidden: the request comes from a page off this machine"u8).ConfigureAwait(false);
            return (0, true, null);
        }

        if (_token is not null && !Authorized(head, request.Span))
        {
            await RefuseAsync(stream, head, 401, "Unauthorized: send Authorization: Bearer TOKEN"u8, extra: "WWW-Authenticate: Bearer\r\n"u8).ConfigureAwait(false);
            return (0, true, null);
        }

        // The protocol revision is read now, while the head is in the buffer (reading the body moves what follows it).
        string? version = head.ProtocolVersion.Length > 0 ? ProtocolVersion(request.Span[head.ProtocolVersion.Range]) : null;
        bool sessionless = head.ProtocolVersion.Length > 0 && Sessionless(request.Span[head.ProtocolVersion.Range]);
        if (sessionless)
        {
            // 2026-07-28 and later: every request stands alone (no Mcp-Session-Id, no initialize, no GET stream, no DELETE).
            if (head.Method != HttpMethod.Post)
            {
                await RefuseAsync(stream, head, 405, "Method Not Allowed: the 2026-07-28 and later protocol revisions have no sessions: POST each request"u8,
                    extra: "Allow: POST\r\n"u8).ConfigureAwait(false);
                return (0, true, null);
            }

            return await PostAsync(stream, head, buffer, consumed, filled, null, version, sessionless: true, start).ConfigureAwait(false);
        }

        Session? session = null;
        if (head.SessionId.Length > 0 && !TryFindSession(request.Span[head.SessionId.Range], out session))
        {
            await RefuseAsync(stream, head, 404, """{"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"},"id":null}"""u8, json: true).ConfigureAwait(false);
            return (0, true, null);
        }

        switch (head.Method)
        {
            case HttpMethod.Post:
                return await PostAsync(stream, head, buffer, consumed, filled, session, version, sessionless: false, start).ConfigureAwait(false);
            case HttpMethod.Get:
                if (session is null)
                {
                    await RefuseAsync(stream, head, 400, "Bad Request: GET needs an Mcp-Session-Id"u8).ConfigureAwait(false);
                    return (0, true, null);
                }

                await GetAsync(stream, session, start).ConfigureAwait(false);
                return (0, true, null);
            case HttpMethod.Delete:
                if (session is null)
                {
                    await RefuseAsync(stream, head, 400, "Bad Request: DELETE needs an Mcp-Session-Id"u8).ConfigureAwait(false);
                    return (0, true, null);
                }

                await CloseSessionAsync(session, "deleted by its client").ConfigureAwait(false);
                if (head.ContentLength != 0 || head.Chunked)
                {
                    close = true;                                            // a body nobody reads
                }

                int sent = await WriteSimpleAsync(stream, 200, default, null, close).ConfigureAwait(false);
                if (start != 0)
                {
                    McpTelemetry.Emit(McpEventKind.RequestAnswered, McpServerTransports.Http, session.Id, "DELETE", status: 200, bytesOut: sent, start: start);
                }

                return (Compact(buffer, consumed, filled), close, null);
            default:
                await RefuseAsync(stream, head, 405, "Method Not Allowed"u8).ConfigureAwait(false);
                return (0, true, null);
        }
    }

    // A revision without sessions: 2026-07-28 (the one that removed them) or later; revisions are dates, compared as text.
    private static bool Sessionless(ReadOnlySpan<byte> version) => version.SequenceCompareTo("2026-07-28"u8) >= 0;

    // A refusal: reported (status and reason) when a hook listens, then answered and the connection closed.
    private Task<int> RefuseAsync(NetworkStream stream, Head head, int status, ReadOnlySpan<byte> body, string? sessionId = null, bool json = false,
        ReadOnlySpan<byte> extra = default)
    {
        if (McpTelemetry.Enabled)
        {
            string reason = Encoding.UTF8.GetString(body);
            if (json && reason.IndexOf("\"message\":\"", StringComparison.Ordinal) is var at and >= 0)
            {
                reason = reason[(at + 11)..reason.IndexOf('"', at + 11)];
            }

            McpTelemetry.Emit(McpEventKind.Refused, McpServerTransports.Http, sessionId, MethodName(head.Method), status: status, address: Remote(stream), reason: reason);
        }

        return WriteSimpleAsync(stream, status, body, sessionId, close: true, json, extra);
    }

    private static string MethodName(HttpMethod method) => method switch
    {
        HttpMethod.Post => "POST",
        HttpMethod.Get => "GET",
        HttpMethod.Delete => "DELETE",
        _ => "other",
    };

    private static string? Remote(NetworkStream stream)
    {
        try
        {
            return stream.Socket.RemoteEndPoint?.ToString();
        }
        catch (ObjectDisposedException)
        {
            return null;
        }
    }

    private async Task<(int Filled, bool Close, Task<int>? Pending)> PostAsync(NetworkStream stream, Head head, byte[] buffer, int consumed, int filled, Session? session,
        string? version, bool sessionless, long start)
    {
        if (head.ExpectContinue)
        {
            await stream.WriteAsync(Continue, _stop.Token).ConfigureAwait(false);
        }

        // The body, into a pooled buffer of its own (what follows it stays in the connection's buffer).
        var (body, length, used, status) = await ReadBodyAsync(stream, head, buffer, consumed, filled).ConfigureAwait(false);
        filled = used.Filled;
        JsonRpcMessage? message;
        try
        {
            if (status != 0)
            {
                await RefuseAsync(stream, head, status, status == 413 ? "Request Entity Too Large"u8 : "Bad Request: the body cannot be read"u8, session?.Id).ConfigureAwait(false);
                return (0, true, null);
            }

            try
            {
                message = JsonSerializer.Deserialize(body.AsSpan(0, length), MessageType);
            }
            catch (JsonException)
            {
                message = null;
            }
        }
        finally
        {
            if (body.Length > 0)
            {
                ArrayPool<byte>.Shared.Return(body);
            }
        }

        filled = Compact(buffer, used.Position, filled);
        if (message is null)
        {
            Refused(stream, head, 400, "Parse error: the body is not a JSON-RPC message", session?.Id);
            await WriteSimpleAsync(stream, 400, """{"jsonrpc":"2.0","error":{"code":-32700,"message":"Parse error"},"id":null}"""u8, null, close: head.Close, json: true)
                .ConfigureAwait(false);
            return (filled, head.Close, null);
        }

        string? method = null, tool = null;
        if (start != 0)
        {
            (method, tool) = (McpTelemetry.MethodOf(message), McpTelemetry.ToolOf(message));
            McpTelemetry.Emit(McpEventKind.RequestReceived, McpServerTransports.Http, session?.Id, method, tool, bytesIn: length);
        }

        if (version is not null)
        {
            message.Context ??= new JsonRpcMessageContext();
            message.Context.ProtocolVersion = version;
        }

        RequestId sessionlessId = default;
        if (sessionless)
        {
            // A notification needs no server: a cancellation finds its call here; any other is accepted and has nothing to do.
            if (message is JsonRpcNotification notification)
            {
                if (notification.Method == NotificationMethods.CancelledNotification)
                {
                    CancelSessionless(notification);
                }

                int accepted = await WriteSimpleAsync(stream, 202, default, null, head.Close).ConfigureAwait(false);
                if (start != 0)
                {
                    McpTelemetry.Emit(McpEventKind.RequestAnswered, McpServerTransports.Http, null, method, tool, 202, length, accepted, start);
                }

                return (filled, head.Close, null);
            }

            if (message is not JsonRpcRequest call)
            {
                Refused(stream, head, 400, "a sessionless message is a request or a notification", null);
                await WriteSimpleAsync(stream, 400, """{"jsonrpc":"2.0","error":{"code":-32600,"message":"Invalid Request: without a session, only requests and notifications are taken"},"id":null}"""u8,
                    null, close: head.Close, json: true).ConfigureAwait(false);
                return (filled, head.Close, null);
            }

            sessionlessId = call.Id;
            session = Session.Stateless(_request.Server);
        }
        else if (session is null)
        {
            if (message is not JsonRpcRequest { Method: RequestMethods.Initialize })
            {
                Refused(stream, head, 400, "a session starts with an initialize request", null);
                await WriteSimpleAsync(stream, 400, """{"jsonrpc":"2.0","error":{"code":-32000,"message":"Bad Request: a session starts with an initialize request; send the Mcp-Session-Id it answered with"},"id":null}"""u8,
                    null, close: head.Close, json: true).ConfigureAwait(false);
                return (filled, head.Close, null);
            }

            session = StartSession(start == 0 ? null : McpTelemetry.ClientOf(message), start == 0 ? null : Remote(stream));
            if (session is null)
            {
                int retry = RetryAfterSeconds();
                string text = string.Create(CultureInfo.InvariantCulture,
                    $$"""{"jsonrpc":"2.0","error":{"code":-32000,"message":"Service Unavailable: the server holds the most sessions it keeps open ({{_request.MaxSessions}}); retry after {{retry}} s"},"id":null}""");
                Refused(stream, head, 503, $"the most sessions open ({_request.MaxSessions.ToString(CultureInfo.InvariantCulture)})", null);
                await WriteSimpleAsync(stream, 503, Encoding.UTF8.GetBytes(text), null, close: head.Close, json: true,
                    Encoding.ASCII.GetBytes(string.Create(CultureInfo.InvariantCulture, $"Retry-After: {retry}\r\n"))).ConfigureAwait(false);
                return (filled, head.Close, null);
            }
        }

        // Keep a read open on the connection while the call runs: it ends at once when the client leaves (0 bytes), which
        // cancels the call; bytes it reads are the client's next request, kept in the buffer.
        Task<int>? probe = null;
        int offset = filled;
        using var abort = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token, session.Stopping);
        if (!head.Close && offset < buffer.Length)
        {
            probe = stream.ReadAsync(buffer.AsMemory(offset), _stop.Token).AsTask();
            _ = probe.ContinueWith(static (read, state) =>
            {
                if (!read.IsCompletedSuccessfully || read.Result == 0)
                {
                    try
                    {
                        ((CancellationTokenSource)state!).Cancel();
                    }
                    catch (ObjectDisposedException)
                    {
                        // the call ended first
                    }
                }
            }, abort, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
        }

        if (sessionless)
        {
            TrackSessionless(sessionlessId, abort, add: true);
        }

        Interlocked.Increment(ref session.Active);
        var response = new ResponseStream(stream, sessionless ? null : session.Id, head.Close);
        bool closeAfter = head.Close;
        string? failure = null;
        try
        {
            bool wrote = sessionless
                ? await session.Transport.HandlePostRequestAsync(message, response, response.OnFirstMessage, abort.Token).ConfigureAwait(false)
                : await session.Transport.HandlePostRequestAsync(message, response, abort.Token).ConfigureAwait(false);
            if (wrote || response.Started)
            {
                await response.CompleteAsync(_stop.Token).ConfigureAwait(false);
            }
            else
            {
                response.Status = 202;
                response.Sent += await WriteSimpleAsync(stream, 202, default, sessionless ? null : session.Id, head.Close).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) when (abort.IsCancellationRequested)
        {
            closeAfter = true;
            failure = "cancelled: the client left or the server stopped";
            if (!_stop.IsCancellationRequested && message is JsonRpcRequest call)
            {
                await CancelCallAsync(session, call.Id).ConfigureAwait(false);       // the client left: its call stops
            }
        }
        catch when (probe is not null)
        {
            // The connection is lost: end the read into its buffer before the buffer can go back to the pool.
            await stream.DisposeAsync().ConfigureAwait(false);
            try
            {
                await probe.ConfigureAwait(false);
            }
            catch (Exception)
            {
                // ended with the socket
            }

            throw;
        }
        finally
        {
            response.Release();
            session.Touch();
            Interlocked.Decrement(ref session.Active);
            if (sessionless)
            {
                TrackSessionless(sessionlessId, abort, add: false);
                await session.DisposeAsync().ConfigureAwait(false);         // a sessionless request's server lives as long as the request
            }

            if (start != 0)
            {
                McpTelemetry.Emit(McpEventKind.RequestAnswered, McpServerTransports.Http, sessionless ? null : session.Id, method, tool, response.Status, length, response.Sent,
                    start, reason: failure);
            }
        }

        return (offset, closeAfter, probe);
    }

    // A refusal answered on a connection kept open (a body that does not parse, a request without its session): reported.
    private static void Refused(NetworkStream stream, Head head, int status, string reason, string? sessionId)
    {
        if (McpTelemetry.Enabled)
        {
            McpTelemetry.Emit(McpEventKind.Refused, McpServerTransports.Http, sessionId, MethodName(head.Method), status: status, address: Remote(stream), reason: reason);
        }
    }

    private void TrackSessionless(RequestId id, CancellationTokenSource abort, bool add)
    {
        lock (_sessionless)
        {
            var (source, count) = _sessionless.GetValueOrDefault(id);
            if (add)
            {
                _sessionless[id] = (count == 0 ? abort : null, count + 1);
            }
            else if (count <= 1)
            {
                _sessionless.Remove(id);
            }
            else
            {
                _sessionless[id] = (ReferenceEquals(source, abort) ? null : source, count - 1);
            }
        }
    }

    // notifications/cancelled without a session: the call of that id stops, when exactly one is in flight.
    private void CancelSessionless(JsonRpcNotification notification)
    {
        CancelledNotificationParams? cancel;
        try
        {
            cancel = notification.Params?.Deserialize(CancelType);
        }
        catch (JsonException)
        {
            return;
        }

        CancellationTokenSource? abort = null;
        if (cancel is not null)
        {
            lock (_sessionless)
            {
                abort = _sessionless.TryGetValue(cancel.RequestId, out var entry) && entry.Count == 1 ? entry.Abort : null;
            }
        }

        try
        {
            abort?.Cancel();
        }
        catch (ObjectDisposedException)
        {
            // the call ended first
        }
    }

    // A call whose client left: the server hears a cancellation for it, as if the client had sent one.
    private async Task CancelCallAsync(Session session, RequestId id)
    {
        try
        {
            var cancel = new JsonRpcNotification
            {
                Method = NotificationMethods.CancelledNotification,
                Params = JsonSerializer.SerializeToNode(new CancelledNotificationParams { RequestId = id, Reason = "the client disconnected" }, CancelType),
            };
            await session.Transport.HandlePostRequestAsync(cancel, Stream.Null, _stop.Token).ConfigureAwait(false);
        }
        catch (Exception e) when (e is OperationCanceledException or InvalidOperationException or ObjectDisposedException)
        {
            // the session is closing anyway
        }
    }

    private async Task GetAsync(NetworkStream stream, Session session, long start)
    {
        if (Interlocked.Exchange(ref session.Streaming, 1) == 1)
        {
            await RefuseAsync(stream, default, 409, "Conflict: the session's event stream is open already"u8, session.Id).ConfigureAwait(false);
            return;
        }

        using var abort = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token, session.Stopping);
        byte[] probeBuffer = ArrayPool<byte>.Shared.Rent(256);
        var probe = stream.ReadAsync(probeBuffer.AsMemory(), abort.Token).AsTask();
        _ = probe.ContinueWith(static (_, state) =>
        {
            try
            {
                ((CancellationTokenSource)state!).Cancel();       // the client closed the stream (or sent bytes it may not)
            }
            catch (ObjectDisposedException)
            {
                // ended first
            }
        }, abort, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);

        var response = new ResponseStream(stream, session.Id, close: true) { FlushEachWrite = true };
        Interlocked.Increment(ref session.Active);
        try
        {
            await response.FlushAsync(abort.Token).ConfigureAwait(false);        // the headers now: the client waits for them
            await session.Transport.HandleGetRequestAsync(response, abort.Token).ConfigureAwait(false);
            await response.CompleteAsync(_stop.Token).ConfigureAwait(false);
        }
        catch (Exception e) when (e is OperationCanceledException or IOException or ObjectDisposedException)
        {
            // the client left, the session closed or the listener stopped
        }
        finally
        {
            response.Release();
            Interlocked.Decrement(ref session.Active);
            session.Touch();
            Volatile.Write(ref session.Streaming, 0);
            await abort.CancelAsync().ConfigureAwait(false);
            await stream.DisposeAsync().ConfigureAwait(false);                  // ends the probe before its buffer goes back
            try
            {
                await probe.ConfigureAwait(false);
            }
            catch (Exception)
            {
                // ended with the socket
            }

            ArrayPool<byte>.Shared.Return(probeBuffer);
            if (start != 0)
            {
                McpTelemetry.Emit(McpEventKind.RequestAnswered, McpServerTransports.Http, session.Id, "GET", status: 200, bytesOut: response.Sent, start: start,
                    reason: "the session's event stream");
            }
        }
    }

    // The request's body: Content-Length or chunked, into a pooled buffer (empty when there is none). Status 413 when it
    // is larger than allowed, 400 when it cannot be read. Used: how much of the connection's buffer the request took.
    private async Task<(byte[] Body, int Length, (int Position, int Filled) Used, int Status)> ReadBodyAsync(NetworkStream stream, Head head, byte[] buffer, int consumed, int filled)
    {
        long max = _request.MaxRequestBytes;
        if (!head.Chunked)
        {
            long length = head.ContentLength < 0 ? 0 : head.ContentLength;
            if (length > max || length > Array.MaxLength)
            {
                return ([], 0, (consumed, filled), 413);
            }

            if (length == 0)
            {
                return ([], 0, (consumed, filled), 0);
            }

            byte[] body = ArrayPool<byte>.Shared.Rent((int)length);
            int have = Math.Min((int)length, filled - consumed);
            buffer.AsSpan(consumed, have).CopyTo(body);
            try
            {
                while (have < length)
                {
                    int read = await stream.ReadAsync(body.AsMemory(have, (int)length - have), _stop.Token).ConfigureAwait(false);
                    if (read == 0)
                    {
                        throw new IOException("the client left in the middle of its request");
                    }

                    have += read;
                }
            }
            catch
            {
                ArrayPool<byte>.Shared.Return(body);
                throw;
            }

            return (body, (int)length, (consumed + Math.Min((int)length, filled - consumed), filled), 0);
        }

        // Chunked: size lines, data, and an empty chunk. Everything the client sent goes through the connection's buffer
        // (which is large enough for any size line and the trailer).
        byte[] data = ArrayPool<byte>.Shared.Rent(4096);
        int written = 0;
        int position = consumed;
        try
        {
            while (true)
            {
                int line;
                while ((line = buffer.AsSpan(position, filled - position).IndexOf("\r\n"u8)) < 0)
                {
                    if (filled - position > 64)
                    {
                        return (Release(data), 0, (position, filled), 400);
                    }

                    (position, filled) = await FillAsync(stream, buffer, position, filled).ConfigureAwait(false);
                }

                var sizeText = buffer.AsSpan(position, line);
                int semicolon = sizeText.IndexOf((byte)';');
                if (semicolon >= 0)
                {
                    sizeText = sizeText[..semicolon];
                }

                if (!Utf8Parser.TryParse(sizeText.Trim((byte)' '), out long size, out int parsed, 'X') || parsed != sizeText.Trim((byte)' ').Length)
                {
                    return (Release(data), 0, (position, filled), 400);
                }

                position += line + 2;
                if (size == 0)
                {
                    // Trailer lines up to the empty one.
                    while (true)
                    {
                        while ((line = buffer.AsSpan(position, filled - position).IndexOf("\r\n"u8)) < 0)
                        {
                            (position, filled) = await FillAsync(stream, buffer, position, filled).ConfigureAwait(false);
                        }

                        position += line + 2;
                        if (line == 0)
                        {
                            return (data, written, (position, filled), 0);
                        }
                    }
                }

                if (written + size > max)
                {
                    return (Release(data), 0, (position, filled), 413);
                }

                if (data.Length < written + size)
                {
                    data = Grow(data, written, (int)Math.Max(data.Length * 2L, written + size));
                }

                long left = size;
                while (left > 0)
                {
                    if (position == filled)
                    {
                        (position, filled) = await FillAsync(stream, buffer, position, filled).ConfigureAwait(false);
                    }

                    int take = (int)Math.Min(left, filled - position);
                    buffer.AsSpan(position, take).CopyTo(data.AsSpan(written));
                    written += take;
                    position += take;
                    left -= take;
                }

                while (filled - position < 2)
                {
                    (position, filled) = await FillAsync(stream, buffer, position, filled).ConfigureAwait(false);
                }

                if (!buffer.AsSpan(position, 2).SequenceEqual("\r\n"u8))
                {
                    return (Release(data), 0, (position, filled), 400);
                }

                position += 2;
            }
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(data);
            throw;
        }

        static byte[] Release(byte[] rented)
        {
            ArrayPool<byte>.Shared.Return(rented);
            return [];
        }
    }

    // Reads more into the connection's buffer; the request head before `consumed` is no longer needed once its body is
    // read, so the data from `position` moves to just after it when the buffer is full.
    private async Task<(int Position, int Filled)> FillAsync(NetworkStream stream, byte[] buffer, int position, int filled)
    {
        if (filled == buffer.Length)
        {
            // The head is parsed already; the unread rest moves to the front (the caller's offsets are relative to it).
            buffer.AsSpan(position, filled - position).CopyTo(buffer);
            filled -= position;
            position = 0;
        }

        int read = await stream.ReadAsync(buffer.AsMemory(filled), _stop.Token).ConfigureAwait(false);
        return read == 0 ? throw new IOException("the client left in the middle of its request") : (position, filled + read);
    }

    // ------------------------------------------------------------------ sessions

    // A new session, unless MaxSessions are open (then null, or the idle one unused the longest closed for it when allowed).
    private Session? StartSession(string? client, string? address)
    {
        int max = _request.MaxSessions;
        if (Interlocked.Increment(ref _open) > max && max > 0 && !(_request.ReplaceIdleSessions && ReplaceIdle()))
        {
            Interlocked.Decrement(ref _open);
            return null;
        }

        string id = HttpServerTransportKind.NewToken(16);
        var transport = new StreamableHttpServerTransport(null) { SessionId = id };
        var server = McpServer.Create(transport, _request.Server);
        var session = new Session(id, transport, server) { Client = client, Address = address };
        _sessions[id] = session;
        if (McpTelemetry.Enabled)
        {
            McpTelemetry.Emit(McpEventKind.SessionOpened, McpServerTransports.Http, id, address: address, client: client);
        }

        if (_stop.IsCancellationRequested)
        {
            _ = CloseSessionAsync(session, "the server stopped");
        }

        return session;
    }

    // Closes the idle session unused the longest for a new one; false when every session is busy. Only a session this call
    // removed counts, so two new sessions at the limit never take the place of the same old one.
    private bool ReplaceIdle()
    {
        for (int attempt = 0; attempt < 4; attempt++)
        {
            Session? oldest = null;
            foreach (var (_, session) in _sessions)
            {
                if (Volatile.Read(ref session.Active) == 0 && (oldest is null || Volatile.Read(ref session.LastUsed) < Volatile.Read(ref oldest.LastUsed)))
                {
                    oldest = session;
                }
            }

            if (oldest is null)
            {
                return false;
            }

            if (Remove(oldest, string.Create(CultureInfo.InvariantCulture, $"replaced by a new session: the most open is {_request.MaxSessions}")))
            {
                _ = oldest.DisposeAsync().AsTask();
                return true;
            }
        }

        return false;
    }

    // Seconds until the first idle session would close for idleness (1 when none will soon, or idle sessions never close).
    private int RetryAfterSeconds()
    {
        long limit = (long)_request.SessionIdleTimeout.TotalMilliseconds, now = Environment.TickCount64, soonest = long.MaxValue;
        if (limit > 0)
        {
            foreach (var (_, session) in _sessions)
            {
                long left = Volatile.Read(ref session.Active) == 0 ? limit - (now - Volatile.Read(ref session.LastUsed)) : limit;
                soonest = Math.Min(soonest, left);
            }
        }

        return soonest == long.MaxValue ? 1 : (int)Math.Clamp((soonest + 999) / 1000, 1, int.MaxValue);
    }

    // Removes and closes a session (once, whoever asks first): reported with its lifetime and why.
    private async Task CloseSessionAsync(Session session, string reason)
    {
        if (Remove(session, reason))
        {
            await session.DisposeAsync().ConfigureAwait(false);
        }
    }

    // Takes a session out of the table and its place out of the count (true for the one caller that did), reported.
    private bool Remove(Session session, string reason)
    {
        if (!_sessions.TryRemove(new KeyValuePair<string, Session>(session.Id, session)))
        {
            return false;
        }

        Interlocked.Decrement(ref _open);
        if (McpTelemetry.Enabled)
        {
            McpTelemetry.Emit(McpEventKind.SessionClosed, McpServerTransports.Http, session.Id, start: session.Created, address: session.Address, client: session.Client, reason: reason);
        }

        return true;
    }

    private bool TryFindSession(ReadOnlySpan<byte> id, out Session? session)
    {
        session = null;
        if (id.Length > 128)
        {
            return false;
        }

        Span<char> chars = stackalloc char[128];
        int count = Encoding.Latin1.GetChars(id, chars);
        return _sessionsBySpan.TryGetValue(chars[..count], out session);
    }

    private void CloseIdle()
    {
        long now = Environment.TickCount64, limit = (long)_request.SessionIdleTimeout.TotalMilliseconds;
        foreach (var (id, session) in _sessions)
        {
            if (Volatile.Read(ref session.Active) == 0 && now - Volatile.Read(ref session.LastUsed) > limit)
            {
                _ = CloseSessionAsync(session, string.Create(CultureInfo.InvariantCulture, $"idle for more than {_request.SessionIdleTimeout}"));
            }
        }
    }

    private string ProtocolVersion(ReadOnlySpan<byte> value)
    {
        string? last = _lastVersion;
        if (last is not null && last.Length == value.Length && Ascii.Equals(value, last))
        {
            return last;
        }

        return _lastVersion = Encoding.Latin1.GetString(value);
    }

    // A browser page's request carries its Origin: one off this machine is refused (DNS rebinding) unless remote clients are allowed.
    private bool OriginAllowed(Head head, ReadOnlySpan<byte> request)
    {
        if (head.Origin.Length == 0 || _request.AllowRemote)
        {
            return true;
        }

        var origin = request[head.Origin.Range];
        if (origin.SequenceEqual("null"u8))
        {
            return false;
        }

        return Uri.TryCreate(Encoding.Latin1.GetString(origin), UriKind.Absolute, out var uri)
               && (uri.IsLoopback || uri.Host.Equals("localhost", StringComparison.OrdinalIgnoreCase));
    }

    private bool Authorized(Head head, ReadOnlySpan<byte> request)
    {
        var value = request[head.Authorization.Range];
        if (value.Length < 7 || !Ascii.EqualsIgnoreCase(value[..7], "Bearer "u8))
        {
            return false;
        }

        return CryptographicOperations.FixedTimeEquals(value[7..].Trim((byte)' '), _token);
    }

    // ------------------------------------------------------------------ writing

    // A whole answer with a Content-Length: the status, the session id when there is one, and a short body.
    // The bytes sent.
    private Task<int> WriteSimpleAsync(NetworkStream stream, int status, ReadOnlySpan<byte> body, string? sessionId, bool close, bool json = false, ReadOnlySpan<byte> extra = default)
    {
        byte[] message = ArrayPool<byte>.Shared.Rent(512 + extra.Length + body.Length);
        int length = Head.WriteStatus(message, status, body.Length, json ? "application/json"u8 : "text/plain; charset=utf-8"u8, sessionId, close, chunked: false, extra);
        body.CopyTo(message.AsSpan(length));
        return SendAsync(stream, message, length + body.Length);
    }

    private async Task<int> SendAsync(NetworkStream stream, byte[] message, int length)
    {
        try
        {
            await stream.WriteAsync(message.AsMemory(0, length), _stop.Token).ConfigureAwait(false);
            return length;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(message);
        }
    }

    // Drops the request's bytes (the first `used`): what follows moves to the front; how much that is.
    private static int Compact(byte[] buffer, int used, int filled)
    {
        buffer.AsSpan(used, filled - used).CopyTo(buffer);
        return filled - used;
    }

    private static byte[] Grow(byte[] old, int used, int size)
    {
        byte[] bigger = ArrayPool<byte>.Shared.Rent(size);
        old.AsSpan(0, used).CopyTo(bigger);
        ArrayPool<byte>.Shared.Return(old);
        return bigger;
    }

    // ------------------------------------------------------------------ stopping

    public Task StopAsync() => StopCoreAsync(null);

    private async Task StopCoreAsync(Exception? error)
    {
        if (Interlocked.Exchange(ref _stopping, 1) == 1)
        {
            await _completion.Task.ConfigureAwait(false);
            return;
        }

        try
        {
            await _stop.CancelAsync().ConfigureAwait(false);
            if (_sweep is not null)
            {
                await _sweep.DisposeAsync().ConfigureAwait(false);
            }

            _socket.Dispose();
            await _accepting.ConfigureAwait(false);
            foreach (var (_, session) in _sessions)
            {
                await CloseSessionAsync(session, "the server stopped").ConfigureAwait(false);
            }

            foreach (var (stream, _) in _connections)
            {
                stream.Dispose();
            }

            try
            {
                await Task.WhenAll(_connections.Values).ConfigureAwait(false);
            }
            catch (Exception)
            {
                // a connection that failed is closed with the rest
            }
        }
        finally
        {
            if (error is null)
            {
                _completion.TrySetResult();
            }
            else
            {
                _completion.TrySetException(error);
            }
        }
    }

    public async ValueTask DisposeAsync() => await StopAsync().ConfigureAwait(false);

    // One client's session: the SDK's transport and server, its run, and how it is used. A sessionless request (2026-07-28
    // and later) gets one of its own, stateless, for as long as it is answered.
    private sealed class Session(string id, StreamableHttpServerTransport transport, McpServer server) : IAsyncDisposable
    {
        private readonly CancellationTokenSource _stop = new();
        private readonly Task _run = RunAsync(server, CancellationToken.None);
        private int _disposed;

        public int Active;               // requests answered now
        public int Streaming;            // 1 while the GET event stream is open
        public long LastUsed = Environment.TickCount64;

        public string Id => id;

        public long Created { get; } = Stopwatch.GetTimestamp();

        public string? Client { get; init; }

        public string? Address { get; init; }

        public StreamableHttpServerTransport Transport => transport;

        public CancellationToken Stopping => _stop.Token;

        public static Session Stateless(McpServerOptions options)
        {
            var transport = new StreamableHttpServerTransport(null) { Stateless = true };
            return new Session("", transport, McpServer.Create(transport, options));
        }

        public void Touch() => Volatile.Write(ref LastUsed, Environment.TickCount64);

        private static async Task RunAsync(McpServer server, CancellationToken token)
        {
            await Task.Yield();
            try
            {
                await server.RunAsync(token).ConfigureAwait(false);
            }
            catch (Exception)
            {
                // the session closed
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 1)
            {
                return;
            }

            await _stop.CancelAsync().ConfigureAwait(false);
            try
            {
                await transport.DisposeAsync().ConfigureAwait(false);
                await server.DisposeAsync().ConfigureAwait(false);
                await _run.ConfigureAwait(false);
            }
            catch (Exception)
            {
                // closing anyway
            }
        }
    }
}

internal enum HttpMethod : byte
{
    Other,
    Get,
    Post,
    Delete,
}

// Where a value lies in the request's head.
internal readonly record struct Slice(int Start, int Length)
{
    public Range Range => new(Start, Start + Length);
}

// A request's line and the headers MCP needs, as offsets into the bytes read (nothing is copied or made a string).
internal readonly record struct Head(HttpMethod Method, Slice Target, bool Close, long ContentLength, bool Chunked, bool ExpectContinue,
    Slice Authorization, Slice SessionId, Slice ProtocolVersion, Slice Origin)
{
    public bool PathIs(ReadOnlySpan<byte> request, ReadOnlySpan<byte> path)
    {
        var target = request[Target.Range];
        int query = target.IndexOfAny((byte)'?', (byte)'#');
        if (query >= 0)
        {
            target = target[..query];
        }

        return target.SequenceEqual(path) || (target.Length == path.Length + 1 && target[^1] == '/' && target[..^1].SequenceEqual(path));
    }

    public static bool TryParse(ReadOnlySpan<byte> head, out Head result)
    {
        result = default;
        int lineEnd = head.IndexOf("\r\n"u8);
        if (lineEnd <= 0)
        {
            return false;
        }

        var line = head[..lineEnd];
        int space = line.IndexOf((byte)' ');
        int last = line.LastIndexOf((byte)' ');
        if (space <= 0 || last <= space)
        {
            return false;
        }

        var method = line[..space];
        var version = line[(last + 1)..];
        var kind = method.SequenceEqual("POST"u8) ? HttpMethod.Post : method.SequenceEqual("GET"u8) ? HttpMethod.Get
            : method.SequenceEqual("DELETE"u8) ? HttpMethod.Delete : HttpMethod.Other;
        bool http10 = version.SequenceEqual("HTTP/1.0"u8);
        if (!http10 && !version.SequenceEqual("HTTP/1.1"u8))
        {
            return false;
        }

        bool close = http10, chunked = false, expect = false;
        long length = -1;
        Slice authorization = default, session = default, protocol = default, origin = default;
        int position = lineEnd + 2;
        while (position < head.Length)
        {
            int end = head[position..].IndexOf("\r\n"u8);
            if (end <= 0)
            {
                break;                                   // the empty line that ends the head
            }

            var header = head.Slice(position, end);
            int colon = header.IndexOf((byte)':');
            if (colon <= 0)
            {
                return false;
            }

            var name = header[..colon];
            var raw = header[(colon + 1)..];
            int lead = raw.Length - raw.TrimStart((byte)' ').TrimStart((byte)'\t').Length;
            var value = raw.Trim((byte)' ').Trim((byte)'\t');
            var slice = new Slice(position + colon + 1 + lead, value.Length);
            if (Ascii.EqualsIgnoreCase(name, "Content-Length"u8))
            {
                if (!Utf8Parser.TryParse(value, out length, out int used) || used != value.Length || length < 0)
                {
                    return false;
                }
            }
            else if (Ascii.EqualsIgnoreCase(name, "Transfer-Encoding"u8))
            {
                chunked = Contains(value, "chunked"u8);
            }
            else if (Ascii.EqualsIgnoreCase(name, "Connection"u8))
            {
                close = Contains(value, "close"u8) || (http10 && !Contains(value, "keep-alive"u8));
            }
            else if (Ascii.EqualsIgnoreCase(name, "Authorization"u8))
            {
                authorization = slice;
            }
            else if (Ascii.EqualsIgnoreCase(name, "Mcp-Session-Id"u8))
            {
                session = slice;
            }
            else if (Ascii.EqualsIgnoreCase(name, "Mcp-Protocol-Version"u8))
            {
                protocol = slice;
            }
            else if (Ascii.EqualsIgnoreCase(name, "Origin"u8))
            {
                origin = slice;
            }
            else if (Ascii.EqualsIgnoreCase(name, "Expect"u8))
            {
                expect = Contains(value, "100-continue"u8);
            }

            position += end + 2;
        }

        result = new Head(kind, new Slice(space + 1, last - space - 1), close, length, chunked, expect, authorization, session, protocol, origin);
        return true;
    }

    private static bool Contains(ReadOnlySpan<byte> value, ReadOnlySpan<byte> word)
    {
        for (int i = 0; i + word.Length <= value.Length; i++)
        {
            if (Ascii.EqualsIgnoreCase(value.Slice(i, word.Length), word))
            {
                return true;
            }
        }

        return false;
    }

    // Writes the status line and headers into `into`; the length written.
    public static int WriteStatus(Span<byte> into, int status, long contentLength, ReadOnlySpan<byte> contentType, string? sessionId, bool close, bool chunked,
        ReadOnlySpan<byte> extra)
    {
        int n = 0;
        Append(into, ref n, "HTTP/1.1 "u8);
        status.TryFormat(into[n..], out int w, default, CultureInfo.InvariantCulture);
        n += w;
        Append(into, ref n, status switch
        {
            200 => " OK\r\n"u8,
            202 => " Accepted\r\n"u8,
            400 => " Bad Request\r\n"u8,
            401 => " Unauthorized\r\n"u8,
            403 => " Forbidden\r\n"u8,
            404 => " Not Found\r\n"u8,
            405 => " Method Not Allowed\r\n"u8,
            409 => " Conflict\r\n"u8,
            413 => " Content Too Large\r\n"u8,
            431 => " Request Header Fields Too Large\r\n"u8,
            503 => " Service Unavailable\r\n"u8,
            _ => " Error\r\n"u8,
        });
        if (status == 405 && extra.IndexOf("Allow:"u8) < 0)
        {
            Append(into, ref n, "Allow: GET, POST, DELETE\r\n"u8);
        }

        if (contentType.Length > 0 && (chunked || contentLength > 0))
        {
            Append(into, ref n, "Content-Type: "u8);
            Append(into, ref n, contentType);
            Append(into, ref n, "\r\n"u8);
        }

        if (chunked)
        {
            Append(into, ref n, "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n"u8);
        }
        else
        {
            Append(into, ref n, "Content-Length: "u8);
            contentLength.TryFormat(into[n..], out w, default, CultureInfo.InvariantCulture);
            n += w;
            Append(into, ref n, "\r\n"u8);
        }

        if (sessionId is not null)
        {
            Append(into, ref n, "Mcp-Session-Id: "u8);
            n += Encoding.Latin1.GetBytes(sessionId, into[n..]);
            Append(into, ref n, "\r\n"u8);
        }

        if (close)
        {
            Append(into, ref n, "Connection: close\r\n"u8);
        }

        Append(into, ref n, extra);
        Append(into, ref n, "\r\n"u8);
        return n;
    }

    private static void Append(Span<byte> into, ref int n, ReadOnlySpan<byte> bytes)
    {
        bytes.CopyTo(into[n..]);
        n += bytes.Length;
    }
}

// The answer to one POST or GET: the SDK writes server-sent events into it. It keeps them in a pooled buffer and sends
// on a flush, the headers with the first send; an answer never flushed before it ends goes out whole, in one send, with
// a Content-Length (the usual answer to a quick call), else as chunks.
internal sealed class ResponseStream(NetworkStream stream, string? sessionId, bool close) : Stream
{
    private const int Reserve = 512;                     // room before the data for the headers and a chunk's size line

    private byte[]? _buffer = ArrayPool<byte>.Shared.Rent(4096);
    private int _length;                                 // data bytes held, after Reserve
    private bool _completed;

    /// <summary>The headers went out.</summary>
    public bool Started { get; private set; }

    /// <summary>Sends each write at once (the GET event stream: events are rare, and wait for nothing).</summary>
    public bool FlushEachWrite { get; init; }

    /// <summary>The status the headers carry (200), until they went out.</summary>
    public int Status { get; set; } = 200;

    /// <summary>The bytes sent on the connection for this answer, headers included.</summary>
    public long Sent { get; set; }

    /// <summary>
    /// The SDK's call before the first message is written (sessionless requests): a JSON-RPC error that comes first sets
    /// the status as the 2026-07-28 revision maps it (method not found 404; an unsupported or missing protocol version or
    /// header 400).
    /// </summary>
    public ValueTask OnFirstMessage(JsonRpcMessage? first)
    {
        if (!Started && first is JsonRpcError { Error: { } error })
        {
            Status = error.Code == (int)McpErrorCode.MethodNotFound ? 404 : error.Code is >= -32022 and <= -32020 ? 400 : Status;
        }

        return default;
    }

    public override bool CanRead => false;

    public override bool CanSeek => false;

    public override bool CanWrite => true;

    public override long Length => throw new NotSupportedException();

    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

    public override void Write(ReadOnlySpan<byte> buffer)
    {
        var held = _buffer ?? throw new ObjectDisposedException(nameof(ResponseStream));
        int need = Reserve + _length + buffer.Length + 7;       // the chunk's end and the last chunk
        if (held.Length < need)
        {
            byte[] bigger = ArrayPool<byte>.Shared.Rent(Math.Max(need, held.Length * 2));
            held.AsSpan(Reserve, _length).CopyTo(bigger.AsSpan(Reserve));
            ArrayPool<byte>.Shared.Return(held);
            _buffer = held = bigger;
        }

        buffer.CopyTo(held.AsSpan(Reserve + _length));
        _length += buffer.Length;
    }

    public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

    public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
    {
        Write(buffer.Span);
        if (FlushEachWrite)
        {
            await SendAsync(final: false, cancellationToken).ConfigureAwait(false);
        }
    }

    public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        WriteAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    public override Task FlushAsync(CancellationToken cancellationToken) => _completed ? Task.CompletedTask : SendAsync(final: false, cancellationToken);

    public override void Flush() => FlushAsync(CancellationToken.None).GetAwaiter().GetResult();

    /// <summary>Sends what is held and ends the answer.</summary>
    public async Task CompleteAsync(CancellationToken cancellationToken)
    {
        if (!_completed)
        {
            await SendAsync(final: true, cancellationToken).ConfigureAwait(false);
            _completed = true;
        }
    }

    private async Task SendAsync(bool final, CancellationToken cancellationToken)
    {
        var held = _buffer ?? throw new ObjectDisposedException(nameof(ResponseStream));
        if (!final && _length == 0 && Started)
        {
            return;
        }

        int start = Reserve, end = Reserve + _length;
        Span<byte> head = stackalloc byte[Reserve];
        int headLength = 0;
        if (!Started && final)
        {
            headLength = Head.WriteStatus(head, Status, _length, "text/event-stream"u8, sessionId, close, chunked: false, default);
        }
        else
        {
            if (!Started)
            {
                headLength = Head.WriteStatus(head, Status, 0, "text/event-stream"u8, sessionId, close, chunked: true, default);
            }

            if (_length > 0)
            {
                _length.TryFormat(head[headLength..], out int w, "X", CultureInfo.InvariantCulture);
                headLength += w;
                "\r\n"u8.CopyTo(head[headLength..]);
                headLength += 2;
                "\r\n"u8.CopyTo(held.AsSpan(end));
                end += 2;
            }

            if (final)
            {
                "0\r\n\r\n"u8.CopyTo(held.AsSpan(end));
                end += 5;
            }
        }

        start -= headLength;
        head[..headLength].CopyTo(held.AsSpan(start));
        Started = true;
        _length = 0;
        if (end > start)
        {
            Sent += end - start;
            await stream.WriteAsync(held.AsMemory(start, end - start), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Returns the buffer to the pool (after the last write).</summary>
    public void Release()
    {
        if (_buffer is { } held)
        {
            _buffer = null;
            ArrayPool<byte>.Shared.Return(held);
        }
    }

    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    public override void SetLength(long value) => throw new NotSupportedException();
}
