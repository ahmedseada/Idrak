// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>What a signature asks of a service on a network.</summary>
public enum SignatureServiceKind : byte
{
    /// <summary>A timestamp (RFC 3161): the body is a <c>TimeStampReq</c>, the answer a <c>TimeStampResp</c>.</summary>
    Timestamp,

    /// <summary>A certificate revocation list (RFC 5280) from its distribution point: no body, the answer is the list.</summary>
    RevocationList,

    /// <summary>A certificate's status (RFC 6960): the body is an <c>OCSPRequest</c>, the answer an <c>OCSPResponse</c>.</summary>
    Ocsp,
}

/// <summary>
/// A request to a signature service: where it goes, what it asks, the request's DER bytes, and how long the answer may take
/// and how large it may be.
/// </summary>
/// <param name="Address">The service's address (a timestamp authority, a distribution point, a responder).</param>
/// <param name="Kind">What is asked.</param>
public sealed record SignatureServiceRequest(Uri Address, SignatureServiceKind Kind)
{
    private readonly long _maxResponseBytes;
    private readonly TimeSpan _timeout = TimeSpan.FromSeconds(30);

    /// <summary>The request's DER bytes (empty for a revocation list).</summary>
    public ReadOnlyMemory<byte> Body { get; init; }

    /// <summary>How long the service may take to answer (30 seconds).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive.</exception>
    public TimeSpan Timeout
    {
        get => _timeout;
        init => _timeout = value > TimeSpan.Zero ? value : throw new ArgumentOutOfRangeException(nameof(Timeout), value, "A service's timeout is more than zero.");
    }

    /// <summary>The most bytes the answer may hold: a quarter of the memory free unless set (as <see cref="DocumentOptions.MaxStreamBytes"/>).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive.</exception>
    public long MaxResponseBytes
    {
        get => _maxResponseBytes > 0 ? _maxResponseBytes : Math.Min(Array.MaxLength, DocumentSettings.FreeMemory() / 4);
        init => _maxResponseBytes = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxResponseBytes), value, "An answer's limit is a number of bytes, more than 0.");
    }
}

/// <summary>
/// Reaches the services a signature needs on a network: a timestamp authority, a revocation list's distribution point, a
/// status responder. Registered in <see cref="SignatureServices"/> under the address scheme it serves ("http", "https"), so
/// an app reaches them its own way (a proxy, credentials, an offline cache) by registering its own under that scheme. A
/// service is asked only for what the app named (a timestamp authority) or turned on (fetching revocation data).
/// </summary>
public interface ISignatureService
{
    /// <summary>Sends <paramref name="request"/> and gives back the answer's bytes (the service's DER answer).</summary>
    /// <exception cref="IOException">The service could not be reached, refused the request or answered past the request's limits (the message says which).</exception>
    byte[] Send(SignatureServiceRequest request, CancellationToken cancellation = default);
}

/// <summary>
/// The signature services by the address scheme they serve (ignoring case). Built in (Idrak.Data): "http" and "https",
/// over the base library's HTTP client.
/// </summary>
public static class SignatureServices
{
}
