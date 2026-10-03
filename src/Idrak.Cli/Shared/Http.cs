// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Net.Http.Headers;
using Idrak.Datasets;

namespace Idrak.Cli.Shared;

/// <summary>
/// The network for every command, so the common options mean the same everywhere: <c>--offline</c> (a download is an
/// error naming the missing file; what is cached is used without asking the network), <c>--timeout</c> (every request
/// gives up when the command's time runs out), <c>--cache</c> (downloads land in the cache folder the command uses), and
/// the progress line (<see cref="ProgressLine"/>, on a terminal only; none with <c>--plain</c>, <c>--quiet</c> or
/// <c>--json</c>).
/// </summary>
internal static class Http
{
    /// <summary>
    /// A downloader into <paramref name="cacheFolder"/> (default: the command's cache folder's <c>downloads</c>) that
    /// honours <c>--offline</c> and <c>--timeout</c>, logs to the verbose output and draws a progress line per file.
    /// </summary>
    public static Downloader Downloader(CommandContext context, string? cacheFolder = null, bool refresh = false)
    {
        var http = new HttpClient(new Guard(context, offline: true) { InnerHandler = new SocketsHttpHandler { AutomaticDecompression = DecompressionMethods.None } })
        {
            Timeout = Timeout.InfiniteTimeSpan,
        };
        http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("Idrak", "1.0"));
        StopAtTimeout(context, http);
        return new Downloader(http, cacheFolder ?? ModelCache.Downloads(context.CacheFolder))
        {
            Refresh = refresh,
            Attempts = context.Offline ? 1 : 5,
            Log = line => context.Detail("  " + line),
            Progress = new DownloadProgressLines(context),
        };
    }

    /// <summary>
    /// A client for calls to a running server (ping, api, server ...): each call gives up after <c>--timeout</c>, or
    /// <paramref name="fallback"/> when it is not given (null: no limit, as for streamed answers).
    /// </summary>
    public static HttpClient Client(CommandContext context, TimeSpan? fallback = null)
    {
        var http = new HttpClient(new Guard(context, offline: false) { InnerHandler = new SocketsHttpHandler() })
        {
            Timeout = context.Timeout ?? fallback ?? Timeout.InfiniteTimeSpan,
        };
        StopAtTimeout(context, http);
        return http;
    }

    // A body being read when --timeout runs out is cut off too (closing the client ends its connections).
    private static void StopAtTimeout(CommandContext context, HttpClient http)
    {
        if (context.Timeout is not null)
        {
            context.TimeoutToken.Register(http.Dispose);
        }
    }

    /// <summary>The message of a request <c>--offline</c> refused: the file that is missing and how to get it.</summary>
    public static string OfflineMessage(Uri? url)
    {
        string file = url is null ? "a file" : Uri.UnescapeDataString(url.Segments.LastOrDefault()?.Trim('/') ?? url.Host);
        return $"--offline: {file} is not in the cache and would be downloaded from {url}; run the command without --offline "
               + "(or 'idrak pull MODEL' while online) to fetch it.";
    }

    // Refuses requests with --offline (downloads only) and ties each request to the command's --timeout.
    private sealed class Guard(CommandContext context, bool offline) : DelegatingHandler
    {
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (offline && context.Offline)
            {
                throw new OfflineException(OfflineMessage(request.RequestUri));
            }

            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, context.TimeoutToken);
            return await base.SendAsync(request, linked.Token).ConfigureAwait(false);
        }
    }
}

/// <summary>
/// A request <c>--offline</c> refused. It reads as a network failure, so the library falls back to a copy it downloaded
/// before where it has one (a model's revision check); a file that must be downloaded fails with this message, which
/// names the missing file and the fix (the downloader makes one attempt when offline).
/// </summary>
internal sealed class OfflineException(string message) : HttpRequestException(message);

/// <summary>A <see cref="ProgressLine"/> per downloaded file, from the downloader's progress reports.</summary>
internal sealed class DownloadProgressLines(CommandContext context) : IProgress<DownloadProgress>
{
    private readonly Lock _lock = new();
    private ProgressLine? _line;
    private string? _file;

    public void Report(DownloadProgress value)
    {
        lock (_lock)
        {
            string file = value.File.Length > 0 ? value.File : Path.GetFileName(new Uri(value.Url).AbsolutePath);
            if (file != _file)
            {
                _line?.Finish();
                _line = new ProgressLine(context, file, value.Total, ProgressUnit.Bytes);
                _file = file;
            }

            _line!.Total = value.Total;
            _line.Report(value.Received);
            if (value.Completed)
            {
                _line.Finish();
                _line = null;
                _file = null;
            }
        }
    }
}
