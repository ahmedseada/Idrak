// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Tar;
using System.IO.Compression;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Datasets;

// Datasets: file formats (Parquet checked against pyarrow), archives, operations, the download cache.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DatasetsGroup =
    [
    ];

    // Answers fixed URLs, as the real services would.
    internal sealed class FakeRouter : HttpMessageHandler
    {
        private readonly Dictionary<string, Func<HttpRequestMessage, HttpResponseMessage>> _routes = [];

        public void Json(string url, string json, string? next = null, HttpStatusCode status = HttpStatusCode.OK) => _routes[url] = _ =>
        {
            var response = new HttpResponseMessage(status) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
            if (next is not null)
            {
                response.Headers.TryAddWithoutValidation("Link", $"<{next}>; rel=\"next\"");
            }

            return response;
        };

        public void Bytes(string url, byte[] body, string? requireToken = null, string? requireAuthorization = null) => _routes[url] = request =>
        {
            string? given = request.Headers.Authorization?.ToString();
            string? wanted = requireAuthorization ?? (requireToken is null ? null : $"Bearer {requireToken}");
            if (wanted is not null && given != wanted)
            {
                return new HttpResponseMessage(HttpStatusCode.Unauthorized) { Content = new StringContent("unauthorized") };
            }

            var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) };
            response.Content.Headers.ContentLength = body.Length;
            return response;
        };

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(_routes.TryGetValue(request.RequestUri!.ToString(), out var route)
                ? route(request)
                : new HttpResponseMessage(HttpStatusCode.NotFound) { Content = new StringContent($"no route for {request.RequestUri}") });
    }

    internal sealed class FakeFileServer : HttpMessageHandler
    {
        public Dictionary<string, byte[]> Files { get; } = [];

        public int Requests { get; private set; }

        public int RangeRequests { get; private set; }

        public string? LastAuthorization { get; private set; }

        public int? FailAfterBytes { get; set; }

        public int Errors { get; set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests++;
            LastAuthorization = request.Headers.Authorization?.ToString();
            if (Errors > 0)
            {
                Errors--;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable) { Content = new StringContent("busy") });
            }

            string url = request.RequestUri!.ToString();
            if (!Files.TryGetValue(url, out var body))
            {
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound) { Content = new StringContent("{\"error\":\"not found\"}") });
            }

            long from = request.Headers.Range?.Ranges.First().From ?? 0;
            RangeRequests += from > 0 ? 1 : 0;
            var slice = body.AsSpan((int)from).ToArray();
            Stream content = new MemoryStream(slice);
            if (FailAfterBytes is { } limit && from == 0)
            {
                content = new BreakingStream(slice, limit);
            }

            var response = new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK) { Content = new StreamContent(content) };
            response.Content.Headers.ContentLength = slice.Length;
            return Task.FromResult(response);
        }
    }

    // Delivers some bytes, then fails as a dropped connection does.
    internal sealed class BreakingStream(byte[] data, int limit) : MemoryStream(data)
    {
        public override int Read(byte[] buffer, int offset, int count)
        {
            if (Position >= limit)
            {
                throw new IOException("connection reset");
            }

            return base.Read(buffer, offset, (int)Math.Min(count, limit - Position));
        }

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (Position >= limit)
            {
                throw new IOException("connection reset");
            }

            return base.ReadAsync(buffer[..(int)Math.Min(buffer.Length, limit - Position)], cancellationToken);
        }
    }
}

// A consumer's normalizer for the tests: lower case, one space between words.
internal sealed class CaseAndSpacing : ITextNormalizer
{
    public string Normalize(string text) => string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToLowerInvariant();
}
