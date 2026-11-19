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
public interface ISignatureService : IKind
{
    /// <summary>Sends <paramref name="request"/> and gives back the answer's bytes (the service's DER answer).</summary>
    /// <exception cref="IOException">The service could not be reached, refused the request or answered past the request's limits (the message says which).</exception>
    byte[] Send(SignatureServiceRequest request, CancellationToken cancellation = default);
}

/// <summary>
/// The signature services by the address scheme they serve (ignoring case). Built in (Idrak.Data): "http" and "https", over
/// the base library's HTTP client. A package or an app registers its own with <see cref="Register"/>; one registered under a
/// library name takes its place, and the library's comes back with <see cref="Unregister"/>.
/// </summary>
public static class SignatureServices
{
    private static readonly KindTable<ISignatureService> Table = new(nameof(SignatureServices), "Signature service",
        "add one with SignatureServices.Register (its Name is the address scheme it serves), or reference a package that registers it",
        comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a service answers for an authority on a network: another one asked in its place would answer for itself (another timestamp, another response), so nothing can be compared or fallen back to");

    static SignatureServices() => LibraryDefaults.Ensure(typeof(SignatureServices));

    /// <summary>Registers <paramref name="service"/> under its name, the scheme it serves (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISignatureService service) => Table.Register(service, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's service <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names (schemes), in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The service registered as <paramref name="name"/>, or null.</summary>
    public static ISignatureService? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The service registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones and what to do.</exception>
    public static ISignatureService Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The service that reaches <paramref name="address"/> (by its scheme).</summary>
    /// <exception cref="NotRegisteredException">No service serves the address's scheme.</exception>
    public static ISignatureService For(Uri address)
    {
        ArgumentNullException.ThrowIfNull(address);
        return Table.Get(address.Scheme);
    }

    /// <summary>Sends <paramref name="request"/> through the service of its address's scheme.</summary>
    /// <exception cref="NotRegisteredException">No service serves the address's scheme.</exception>
    /// <exception cref="IOException">The service failed (see <see cref="ISignatureService.Send"/>).</exception>
    public static byte[] Send(SignatureServiceRequest request, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return For(request.Address).Send(request, cancellation);
    }

    /// <summary>The library's service <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ISignatureService? Default(string name) => Table.Default(name);

    /// <summary>Who registered the service <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered service as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();
}
