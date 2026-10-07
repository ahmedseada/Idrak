// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>
/// Fetches remote files into a local cache and calls web APIs, for dataset and model sources. Idrak.Datasets implements
/// it (<c>Downloader</c>: resumed and retried downloads into <c>IDRAK_CACHE</c>); an application can pass its own (a
/// mirror, an authenticated proxy, an offline cache) wherever a source takes one.
/// </summary>
public interface IDownloader
{
    /// <summary>The folder files are kept in.</summary>
    string CacheFolder { get; }

    /// <summary>Called with a line for each step (what a source resolved, files found in the cache, downloads started and retried), or null.</summary>
    Action<string>? Log { get; }

    /// <summary>
    /// The local path of <paramref name="url"/>, downloading it first unless cached. <paramref name="cachePath"/> is where
    /// the file goes in the cache, '/'-separated (e.g. "huggingface/datasets/openai/gsm8k/1a2b3c4d5e6f/main/test.parquet";
    /// the extension decides how it is read); null for a path derived from the URL.
    /// </summary>
    Task<string> DownloadAsync(string url, IReadOnlyDictionary<string, string>? headers = null, string? cachePath = null,
        CancellationToken cancellationToken = default);

    /// <summary>Downloads (or finds in the cache) <paramref name="url"/>; see <see cref="DownloadAsync"/>.</summary>
    string Download(string url, IReadOnlyDictionary<string, string>? headers = null, string? cachePath = null) =>
        DownloadAsync(url, headers, cachePath).GetAwaiter().GetResult();

    /// <summary>A GET with <paramref name="headers"/>, for API calls; throws with the server's message on failure.</summary>
    Task<string> GetStringAsync(string url, IReadOnlyDictionary<string, string>? headers = null, CancellationToken cancellationToken = default);

    /// <summary>GETs <paramref name="url"/> and every following page named by a <c>Link: &lt;…&gt;; rel="next"</c> header.</summary>
    Task<List<string>> GetPagesAsync(string url, IReadOnlyDictionary<string, string>? headers = null, CancellationToken cancellationToken = default);

    /// <summary>The local path a cache path (as given to <see cref="DownloadAsync"/>) maps to, whether or not it is downloaded yet.</summary>
    string PathFor(string cachePath);
}
