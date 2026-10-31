// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Mcp.Abstractions;
using ModelContextProtocol.Server;

namespace Idrak.Mcp.Transports;

// "stdio": one session over the process's standard input and output (or the request's streams), the SDK's stream
// transport behind the contract. It ends when the client closes its end, or when it is stopped.
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
        var input = request.Input ?? Console.OpenStandardInput();
        var output = request.Output ?? Console.OpenStandardOutput();
        string name = request.Server.ServerInfo?.Name ?? "idrak";
        return ValueTask.FromResult<IMcpListener>(new Listener(McpServer.Create(new StreamServerTransport(input, output, name), request.Server)));
    }

    private sealed class Listener : IMcpListener
    {
        private readonly McpServer _server;
        private readonly CancellationTokenSource _stop = new();
        private int _disposed;

        public Listener(McpServer server)
        {
            _server = server;
            Completion = RunAsync();
        }

        public string Address => McpServerTransports.Stdio;

        public string? Token => null;

        public Task Completion { get; }

        private async Task RunAsync()
        {
            try
            {
                await _server.RunAsync(_stop.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (_stop.IsCancellationRequested)
            {
                // stopped
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
