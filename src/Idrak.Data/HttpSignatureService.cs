// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;

namespace Idrak.Data;

/// <summary>
/// The library's signature service for "http" and "https" (registered in <see cref="SignatureServices"/>, one under each
/// scheme): RFC 3161's and RFC 6960's HTTP bindings (a timestamp or a status request POSTed with its media type) and
/// RFC 5280's distribution points (a revocation list fetched), through the base library's HTTP client, one for the
/// process. The answer is read up to the request's limit and within its timeout. Asked only for what the app named or
/// turned on.
/// </summary>
internal sealed class HttpSignatureService : ISignatureService
{
    private static readonly Lazy<HttpClient> Client = new(() => new HttpClient(new SocketsHttpHandler
    {
        PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        AutomaticDecompression = DecompressionMethods.None,
    })
    {
        Timeout = Timeout.InfiniteTimeSpan,                                     // each request's own timeout bounds it
    });

    public HttpSignatureService(string scheme) => Name = scheme;

    public string Name { get; }

    public string Summary => $"timestamps (RFC 3161), revocation lists and OCSP status (RFC 6960) over {Name.ToUpperInvariant()}, through the base library's HTTP client";

    public byte[] Send(SignatureServiceRequest request, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        string what = request.Kind switch
        {
            SignatureServiceKind.Timestamp => "timestamp",
            SignatureServiceKind.Ocsp => "OCSP",
            _ => "revocation list",
        };
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        timeout.CancelAfter(request.Timeout);
        using var message = new HttpRequestMessage(request.Kind == SignatureServiceKind.RevocationList ? HttpMethod.Get : HttpMethod.Post, request.Address);
        if (request.Kind != SignatureServiceKind.RevocationList)
        {
            message.Content = new ReadOnlyMemoryContent(request.Body);
            message.Content.Headers.ContentType = new MediaTypeHeaderValue(request.Kind == SignatureServiceKind.Timestamp ? "application/timestamp-query" : "application/ocsp-request");
        }

        try
        {
            using var response = Client.Value.Send(message, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
            if (!response.IsSuccessStatusCode)
            {
                throw new IOException(string.Create(CultureInfo.InvariantCulture, $"{request.Address} answered {(int)response.StatusCode} {response.ReasonPhrase} to a {what} request."));
            }

            if (response.Content.Headers.ContentLength is { } length && length > request.MaxResponseBytes)
            {
                throw new IOException(string.Create(CultureInfo.InvariantCulture,
                    $"{request.Address} answers a {what} request with {length:N0} bytes, more than the request allows ({request.MaxResponseBytes:N0}, SignatureServiceRequest.MaxResponseBytes)."));
            }

            using var stream = response.Content.ReadAsStream(timeout.Token);
            using var answer = new MemoryStream();
            byte[] chunk = ArrayPool<byte>.Shared.Rent(64 * 1024);
            try
            {
                int read;
                while ((read = stream.Read(chunk, 0, chunk.Length)) > 0)
                {
                    timeout.Token.ThrowIfCancellationRequested();
                    if (answer.Length + read > request.MaxResponseBytes)
                    {
                        throw new IOException(string.Create(CultureInfo.InvariantCulture,
                            $"{request.Address} answers a {what} request with more than the {request.MaxResponseBytes:N0} bytes the request allows (SignatureServiceRequest.MaxResponseBytes)."));
                    }

                    answer.Write(chunk, 0, read);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(chunk);
            }

            return answer.ToArray();
        }
        catch (OperationCanceledException e) when (!cancellation.IsCancellationRequested)
        {
            throw new IOException(string.Create(CultureInfo.InvariantCulture, $"{request.Address} did not answer a {what} request within {request.Timeout.TotalSeconds:0.#} seconds (the request's Timeout)."), e);
        }
        catch (HttpRequestException e)
        {
            throw new IOException($"{request.Address} could not be reached for a {what} request: {e.Message}", e);
        }
    }
}
