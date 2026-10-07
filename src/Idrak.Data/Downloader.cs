// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Data;

/// <summary>Progress of a download.</summary>
/// <param name="Url">What is downloaded.</param>
/// <param name="Received">Bytes so far.</param>
/// <param name="Total">Size, when the server tells it.</param>
/// <param name="File">The file's name.</param>
/// <param name="Elapsed">Time since this download (or its resumption) started.</param>
/// <param name="Completed">True on the last report of a finished download.</param>
public sealed record DownloadProgress(string Url, long Received, long? Total, string File = "", TimeSpan Elapsed = default, bool Completed = false)
{
    /// <summary>Bytes per second so far.</summary>
    public double BytesPerSecond => Elapsed.TotalSeconds > 0 ? Received / Elapsed.TotalSeconds : 0;
}

/// <summary>
/// Downloads files once into a cache folder and returns the local path: interrupted downloads resume (HTTP ranges),
/// failures are retried, a file appears only when complete. The cache is <c>IDRAK_CACHE</c> or
/// <c>~/.cache/idrak</c>, under <c>downloads/</c>, laid out by source: huggingface/datasets/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/…,
/// github/&lt;owner&gt;/&lt;repo&gt;/…, kaggle/…, zenodo/…, and urls/&lt;host&gt;/&lt;path&gt; for plain URLs.
/// </summary>
public sealed class Downloader : IDownloader
{
    private static readonly Lazy<Downloader> SharedInstance = new(() => new Downloader());
    private readonly HttpClient _http;

    /// <summary>A downloader using <paramref name="http"/> (default: its own client) and <paramref name="cacheFolder"/>.</summary>
    public Downloader(HttpClient? http = null, string? cacheFolder = null)
    {
        _http = http ?? new HttpClient(new HttpClientHandler { AutomaticDecompression = DecompressionMethods.None }) { Timeout = Timeout.InfiniteTimeSpan };
        if (http is null)
        {
            _http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("Idrak", "1.0"));
        }

        CacheFolder = cacheFolder ?? Path.Combine(DefaultCacheRoot, "downloads");
    }

    /// <summary>The downloader <see cref="DatasetRows"/> sources use unless given another.</summary>
    public static Downloader Shared => SharedInstance.Value;

    /// <summary><c>IDRAK_CACHE</c>, or <c>~/.cache/idrak</c>.</summary>
    public static string DefaultCacheRoot => Environment.GetEnvironmentVariable("IDRAK_CACHE")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");

    /// <summary>Where files are kept.</summary>
    public string CacheFolder { get; }

    /// <summary>Called as downloads progress (several times a second) and once more when each completes.</summary>
    public IProgress<DownloadProgress>? Progress { get; init; }

    /// <summary>
    /// Called with a line for each step: what a source resolved (files, splits, fallbacks), files found in the cache,
    /// downloads started, resumed and retried.
    /// </summary>
    public Action<string>? Log { get; init; }

    /// <summary>Re-download files even when cached.</summary>
    public bool Refresh { get; init; }

    /// <summary>Attempts per download.</summary>
    public int Attempts { get; init; } = 5;

    /// <summary>The wait before the first retry; it doubles with each attempt (at most 30 s).</summary>
    public TimeSpan RetryDelay { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>The client, for sources that call web APIs.</summary>
    public HttpClient Http => _http;

    /// <summary>Downloads (or finds in the cache) <paramref name="url"/>; see <see cref="DownloadAsync"/>.</summary>
    public string Download(string url, IReadOnlyDictionary<string, string>? headers = null, string? cachePath = null) =>
        DownloadAsync(url, headers, cachePath).GetAwaiter().GetResult();

    /// <summary>
    /// The local path of <paramref name="url"/>, downloading it first unless cached. <paramref name="cachePath"/> is where
    /// the file goes in the cache, '/'-separated (e.g. "huggingface/datasets/openai/gsm8k/1a2b3c4d5e6f/main/test.parquet";
    /// the extension decides how it is read); default: urls/&lt;host&gt;/&lt;path of the URL&gt;.
    /// </summary>
    public async Task<string> DownloadAsync(string url, IReadOnlyDictionary<string, string>? headers = null, string? cachePath = null,
        CancellationToken cancellationToken = default)
    {
        string target = PathFor(cachePath ?? UrlPath(url));
        string folder = Path.GetDirectoryName(target)!;
        if (File.Exists(target) && !Refresh)
        {
            Log?.Invoke($"cached {Path.GetFileName(target)} ({Size(new FileInfo(target).Length)})");
            return target;
        }

        Directory.CreateDirectory(folder);
        string partial = target + ".part";
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                await DownloadOnceAsync(url, headers, partial, cancellationToken).ConfigureAwait(false);
                File.Move(partial, target, overwrite: true);
                return target;
            }
            catch (Exception ex) when (attempt < Attempts && ex is HttpRequestException { StatusCode: null or >= HttpStatusCode.InternalServerError or HttpStatusCode.TooManyRequests }
                                           or IOException && ex is not FileNotFoundException)
            {
                Log?.Invoke($"{Path.GetFileName(target)}: {ex.Message.Split('\n')[0]}; retrying in {Backoff(attempt).TotalSeconds:0.#} s (attempt {attempt + 1} of {Attempts})");
                await Task.Delay(Backoff(attempt), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>A GET with <paramref name="headers"/>, for API calls; throws with the server's message on failure.</summary>
    public async Task<string> GetStringAsync(string url, IReadOnlyDictionary<string, string>? headers = null, CancellationToken cancellationToken = default)
    {
        for (int attempt = 1; ; attempt++)
        {
            using var request = Request(url, headers);
            using var response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            if (response.IsSuccessStatusCode)
            {
                return body;
            }

            if (attempt >= Attempts || response.StatusCode is not (HttpStatusCode.TooManyRequests or >= HttpStatusCode.InternalServerError))
            {
                throw Failure(url, response.StatusCode, body);
            }

            await Task.Delay(Backoff(attempt), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>GETs <paramref name="url"/> and every following page named by a <c>Link: &lt;…&gt;; rel="next"</c> header.</summary>
    public async Task<List<string>> GetPagesAsync(string url, IReadOnlyDictionary<string, string>? headers = null, CancellationToken cancellationToken = default)
    {
        var pages = new List<string>();
        string? next = url;
        while (next is not null && pages.Count < 10_000)
        {
            using var request = Request(next, headers);
            using var response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                throw Failure(next, response.StatusCode, body);
            }

            pages.Add(body);
            next = response.Headers.TryGetValues("Link", out var links) ? NextLink(string.Join(",", links)) : null;
        }

        return pages;
    }

    private static string? NextLink(string header)
    {
        foreach (var part in header.Split(','))
        {
            int open = part.IndexOf('<'), close = part.IndexOf('>');
            if (open >= 0 && close > open && part.Contains("rel=\"next\"", StringComparison.Ordinal))
            {
                return part[(open + 1)..close].Trim();
            }
        }

        return null;
    }

    private async Task DownloadOnceAsync(string url, IReadOnlyDictionary<string, string>? headers, string partial, CancellationToken cancellationToken)
    {
        long existing = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        using var request = Request(url, headers);
        if (existing > 0)
        {
            request.Headers.Range = new RangeHeaderValue(existing, null);
        }

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable && existing > 0)
        {
            return;                                                 // the partial file is already complete
        }

        if (!response.IsSuccessStatusCode)
        {
            throw Failure(url, response.StatusCode, await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
        }

        bool resumed = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        long? total = response.Content.Headers.ContentLength is { } length ? length + (resumed ? existing : 0) : null;
        string name = Path.GetFileName(partial)[..^5];
        Log?.Invoke(resumed ? $"resuming {name} at {Size(existing)} of {(total is { } t0 ? Size(t0) : "?")}"
                            : $"downloading {name}{(total is { } t1 ? $" ({Size(t1)})" : "")}");
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var file = new FileStream(partial, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);
        var buffer = new byte[1 << 20];
        long received = resumed ? existing : 0;
        long lastReport = 0;
        int read;
        while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await file.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            received += read;
            if (clock.ElapsedMilliseconds - lastReport >= 100)
            {
                lastReport = clock.ElapsedMilliseconds;
                Progress?.Report(new DownloadProgress(url, received, total, name, clock.Elapsed));
            }
        }

        if (total is { } expected && received != expected)
        {
            throw new IOException($"{url}: received {received} of {expected} bytes.");
        }

        Progress?.Report(new DownloadProgress(url, received, total, name, clock.Elapsed, Completed: true));
    }

    /// <summary>A byte count for people: 512 B, 3.4 MB, 1.20 GB.</summary>
    public static string Size(long bytes) => bytes switch
    {
        < 1024 => $"{bytes} B",
        < 1 << 20 => $"{bytes / 1024.0:0.#} KB",
        < 1 << 30 => $"{bytes / (double)(1 << 20):0.#} MB",
        _ => $"{bytes / (double)(1 << 30):0.00} GB",
    };

    private TimeSpan Backoff(int attempt) => TimeSpan.FromTicks(Math.Min(TimeSpan.FromSeconds(30).Ticks, RetryDelay.Ticks << Math.Min(attempt - 1, 20)));

    private static HttpRequestMessage Request(string url, IReadOnlyDictionary<string, string>? headers)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, url);
        foreach (var (name, value) in headers ?? new Dictionary<string, string>())
        {
            request.Headers.TryAddWithoutValidation(name, value);
        }

        return request;
    }

    private static HttpRequestException Failure(string url, HttpStatusCode status, string body)
    {
        string hint = status switch
        {
            HttpStatusCode.Unauthorized => " (sign-in needed: set the token)",
            HttpStatusCode.Forbidden => " (no access: a gated or private resource, or the token lacks permission)",
            HttpStatusCode.NotFound => " (not found, or private without a token)",
            _ => "",
        };
        string detail = body.Length > 300 ? body[..300] + "…" : body;
        return new HttpRequestException($"{url}: {(int)status} {status}{hint}. {detail}".TrimEnd(), null, status);
    }

    /// <summary>The local path a cache path (as given to <see cref="DownloadAsync"/>) maps to, whether or not it is downloaded yet.</summary>
    public string PathFor(string cachePath) => Path.Combine([CacheFolder, .. Segments(cachePath)]);

    // urls/<host>/<path>; a query string becomes a short hash in the file name, so different queries do not collide.
    private static string UrlPath(string url)
    {
        var uri = new Uri(url);
        string path = Uri.UnescapeDataString(uri.AbsolutePath).Trim('/');
        if (path.Length == 0)
        {
            path = "index";
        }

        if (uri.Query.Length > 1)
        {
            string hash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(uri.Query)))[..8];
            string extension = Path.GetExtension(path);
            path = $"{path[..^extension.Length]}-{hash}{extension}";
        }

        return $"urls/{uri.Host}{(uri.IsDefaultPort ? "" : "_" + uri.Port)}/{path}";
    }

    // The folders and file name of a cache path, each made safe for the file system; ".." and empty parts are dropped.
    private static string[] Segments(string cachePath)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var parts = cachePath.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries)
            .Where(p => p is not ("." or ".."))
            .Select(p => new string([.. p.Select(c => invalid.Contains(c) || c is ':' or '?' or '*' or '"' or '<' or '>' or '|' ? '_' : c)]).TrimEnd('.', ' '))
            .Where(p => p.Length > 0)
            .ToArray();
        return parts.Length > 0 ? parts : ["download"];
    }
}
