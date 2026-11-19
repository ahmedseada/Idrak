// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Numerics;
using System.Security.Cryptography;

namespace Idrak.Data.Pdf;

/// <summary>What a revocation source says of one certificate at a time.</summary>
internal enum CmsStatus : byte
{
    /// <summary>Nothing usable: no source names the certificate, or none was valid at the time.</summary>
    Unknown,

    /// <summary>Not revoked at the time.</summary>
    Good,

    /// <summary>Revoked at or before the time.</summary>
    Revoked,
}

/// <summary>
/// A certificate's status from one source, and what the source was: when revoked, the date and reason; whether the source
/// was stale at the time (its next update, with the check's grace, before it); the reasons a list that finds the certificate
/// good covers (every reason unless the list covers only some, RFC 5280 5.2.5).
/// </summary>
internal readonly record struct CmsStatusAnswer(CmsStatus Status, string Detail, DateTimeOffset? RevokedAt = null, string? Reason = null, bool Stale = false,
    int Reasons = CmsCrl.AllReasons);

/// <summary>
/// The signatures of revocation lists and OCSP responses verified within one check: each source's signature (and a delegated
/// responder's link) once a check, however many certificates ask about it; never kept across checks, whose trust is their own.
/// A source is a <see cref="CmsCrl"/>, a <see cref="CmsOcsp"/> or a <see cref="CmsCertificate"/> (verified by its issuer's key).
/// </summary>
internal sealed class CmsVerifications
{
    [ThreadStatic]
    private static int t_made;

    private readonly Dictionary<(object Source, CmsCertificate Signer), string?> _done = [];

    /// <summary>How many signatures of sources this thread has verified (a test reads it around one check).</summary>
    public static int Made
    {
        get => t_made;
        set => t_made = value;
    }

    /// <summary>Why <paramref name="signer"/>'s key does not verify <paramref name="source"/> (null when it does), verified the first time only.</summary>
    public string? Verify(object source, CmsCertificate signer)
    {
        if (!_done.TryGetValue((source, signer), out string? found))
        {
            found = Counted(source, signer);
            _done[(source, signer)] = found;
        }

        return found;
    }

    /// <summary>Verifies through <paramref name="verified"/> when a check gives one, else once, here.</summary>
    public static string? Once(CmsVerifications? verified, object source, CmsCertificate signer) =>
        verified is null ? Counted(source, signer) : verified.Verify(source, signer);

    private static string? Counted(object source, CmsCertificate signer)
    {
        t_made++;
        return source switch
        {
            CmsCrl crl => crl.VerifyWith(signer),
            CmsOcsp ocsp => ocsp.VerifyWith(signer),
            CmsCertificate certificate => CmsPath.VerifiedBy(certificate, signer),
            _ => throw new ArgumentException($"{source.GetType().Name} is not a source a check verifies", nameof(source)),
        };
    }
}

/// <summary>
/// A distribution point (RFC 5280, 4.2.1.13; a list's issuing distribution point's name, 5.2.5): its full name, or its name
/// relative to its lists' issuer (an RDN, as encoded), the reasons it covers (null: every one) and its lists' issuer
/// (cRLIssuer, an indirect list's).
/// </summary>
internal sealed class CmsDistributionPoint(IReadOnlyList<CmsGeneralName>? fullName, ReadOnlyMemory<byte> relative, int? reasons, IReadOnlyList<CmsGeneralName>? crlIssuer)
{
    private static readonly Asn1Tag ReasonsTag = new(TagClass.ContextSpecific, 1);

    public IReadOnlyList<CmsGeneralName>? FullName { get; } = fullName;

    public ReadOnlyMemory<byte> Relative { get; } = relative;

    public int? Reasons { get; } = reasons;

    public IReadOnlyList<CmsGeneralName>? CrlIssuer { get; } = crlIssuer;

    /// <summary>The addresses (URIs) its full name gives.</summary>
    public IEnumerable<string> Addresses => (FullName ?? []).Where(n => n.Form == CmsNameForm.Uri).Select(n => n.Text);

    /// <summary>Every point of a CRLDistributionPoints (or FreshestCRL) value.</summary>
    public static List<CmsDistributionPoint> ReadAll(ReadOnlyMemory<byte> value)
    {
        var points = new List<CmsDistributionPoint>();
        var sequence = CmsAsn.Reader(value).ReadSequence();
        while (sequence.HasData)
        {
            var point = sequence.ReadSequence();
            var (full, relative) = CmsAsn.Next(point, CmsAsn.Context0) ? ReadName(point.ReadSequence(CmsAsn.Context0)) : (null, default);
            int? reasons = CmsAsn.Next(point, ReasonsTag) ? ReasonsOf(point, ReasonsTag) : null;
            var issuer = CmsAsn.Next(point, CmsAsn.Context2) ? CmsGeneralName.ReadAll(point.ReadSequence(CmsAsn.Context2)) : null;
            point.ThrowIfNotEmpty();
            points.Add(new CmsDistributionPoint(full, relative, reasons, issuer));
        }

        return points;
    }

    /// <summary>A DistributionPointName (inside its [0]): a full name [0], or a name relative to the lists' issuer [1].</summary>
    public static (IReadOnlyList<CmsGeneralName>? Full, ReadOnlyMemory<byte> Relative) ReadName(AsnReader name)
    {
        if (CmsAsn.Next(name, CmsAsn.Context0))
        {
            return (CmsGeneralName.ReadAll(name.ReadSequence(CmsAsn.Context0)), default);
        }

        if (CmsAsn.Next(name, CmsAsn.Context1))
        {
            return (null, name.ReadEncodedValue());
        }

        throw new InvalidDataException("a distribution point's name that is neither a full name nor a relative one");
    }

    /// <summary>ReasonFlags (RFC 5280 4.2.1.13) as a mask of bits 1 (keyCompromise) to 8 (aACompromise).</summary>
    public static int ReasonsOf(AsnReader reader, Asn1Tag tag)
    {
        byte[] bits = reader.ReadBitString(out _, tag);
        int mask = 0;
        for (int k = 1; k <= 8; k++)
        {
            if (k / 8 < bits.Length && (bits[k / 8] & (0x80 >> (k % 8))) != 0)
            {
                mask |= 1 << k;
            }
        }

        return mask;
    }

    /// <summary>
    /// The names the point stands for: its full name's, or its relative name on <paramref name="baseName"/> (the lists' issuer)
    /// as a directory name; a point with neither stands for its CRL issuers' names. Directory names are read for comparison.
    /// </summary>
    public List<(CmsGeneralName Name, CmsNameAttribute[][]? Dn)> Names(ReadOnlyMemory<byte> baseName)
    {
        var names = new List<(CmsGeneralName, CmsNameAttribute[][]?)>();
        if (FullName is not null || !Relative.IsEmpty)
        {
            foreach (var name in FullName ?? [])
            {
                names.Add((name, name.Form == CmsNameForm.Directory ? CmsDn.Read(name.Value) : null));
            }

            if (!Relative.IsEmpty && CmsDn.Read(baseName) is { } named)
            {
                try
                {
                    var set = CmsAsn.Reader(Relative).ReadSetOf(skipSortOrderValidation: true, new Asn1Tag(TagClass.ContextSpecific, 1, isConstructed: true));
                    CmsNameAttribute[][] full = [.. named, CmsDn.Attributes(set)];
                    names.Add((new CmsGeneralName(CmsNameForm.Directory, default), full));
                }
                catch (Exception e) when (CmsAsn.IsDamage(e))
                {
                    // A relative name that cannot be read names nothing.
                }
            }

            return names;
        }

        foreach (var name in CrlIssuer ?? [])
        {
            names.Add((name, name.Form == CmsNameForm.Directory ? CmsDn.Read(name.Value) : null));
        }

        return names;
    }

    /// <summary>Whether two lists of names share one (the same form and value; directory names compared as names).</summary>
    public static bool Share(List<(CmsGeneralName Name, CmsNameAttribute[][]? Dn)> a, List<(CmsGeneralName Name, CmsNameAttribute[][]? Dn)> b)
    {
        foreach (var (x, xDn) in a)
        {
            foreach (var (y, yDn) in b)
            {
                if (x.Form == y.Form && (x.Form == CmsNameForm.Directory ? xDn is not null && yDn is not null && CmsDn.Same(xDn, yDn) : x.Value.Span.SequenceEqual(y.Value.Span)))
                {
                    return true;
                }
            }
        }

        return false;
    }
}

/// <summary>
/// A list's issuing distribution point (RFC 5280, 5.2.5): the distribution point it is for, the certificates it covers (end
/// entities only, CAs only), the reasons it covers (null: every one), whether it is indirect, and whether it holds attribute
/// certificates only.
/// </summary>
internal sealed class CmsCrlScope
{
    public CmsDistributionPoint? Point { get; private init; }

    public bool OnlyUser { get; private init; }

    public bool OnlyCa { get; private init; }

    public int? Reasons { get; private init; }

    public bool Indirect { get; private init; }

    public bool OnlyAttribute { get; private init; }

    public static CmsCrlScope Read(ReadOnlyMemory<byte> value)
    {
        var point = CmsAsn.Reader(value).ReadSequence();
        CmsDistributionPoint? name = null;
        if (CmsAsn.Next(point, CmsAsn.Context0))
        {
            var (full, relative) = CmsDistributionPoint.ReadName(point.ReadSequence(CmsAsn.Context0));
            name = new CmsDistributionPoint(full, relative, null, null);
        }

        bool Flag(int number)
        {
            var tag = new Asn1Tag(TagClass.ContextSpecific, number);
            return CmsAsn.Next(point, tag) && point.ReadBoolean(tag);
        }

        bool user = Flag(1), ca = Flag(2);
        var reasonsTag = new Asn1Tag(TagClass.ContextSpecific, 3);
        int? reasons = CmsAsn.Next(point, reasonsTag) ? CmsDistributionPoint.ReasonsOf(point, reasonsTag) : null;
        bool indirect = Flag(4), attribute = Flag(5);
        point.ThrowIfNotEmpty();
        return new CmsCrlScope { Point = name, OnlyUser = user, OnlyCa = ca, Reasons = reasons, Indirect = indirect, OnlyAttribute = attribute };
    }
}

/// <summary>
/// A certificate revocation list (RFC 5280, 5) read on its DER: its issuer, times and number, the serial numbers it revokes
/// (with their dates, reasons and, in an indirect list, their certificates' issuers), its scope (the issuing distribution
/// point: which certificates and reasons it covers, whether it is indirect), whether it is a delta list and of which base,
/// and the signature over its signed part. A list with a critical extension not read here, an entry with one, an entry
/// naming another issuer in a list that is not indirect, or naming a serial number of an issuer twice says so, and is not
/// used.
/// </summary>
internal sealed class CmsCrl
{
    /// <summary>Every revocation reason (ReasonFlags' bits 1, keyCompromise, to 8, aACompromise).</summary>
    public const int AllReasons = 0x1FE;

    private readonly List<ReadOnlyMemory<byte>> _issuers = [];                // the entries' certificates' issuers, the list's own first
    private readonly Dictionary<(int Issuer, BigInteger Serial), (DateTimeOffset Date, int? Reason)> _revoked = [];

    private CmsCrl(ReadOnlyMemory<byte> raw) => Raw = raw;

    public ReadOnlyMemory<byte> Raw { get; }

    public ReadOnlyMemory<byte> Tbs { get; private set; }

    public CmsAlgorithm SignatureAlgorithm { get; private set; }

    public byte[] Signature { get; private set; } = [];

    public ReadOnlyMemory<byte> Issuer { get; private set; }

    public DateTimeOffset ThisUpdate { get; private set; }

    public DateTimeOffset? NextUpdate { get; private set; }

    /// <summary>The CRL number, and a delta list's base list's number (deltaCRLIndicator), or null.</summary>
    public BigInteger? Number { get; private set; }

    public BigInteger? DeltaBase { get; private set; }

    public byte[]? AuthorityKeyId { get; private set; }

    /// <summary>The issuing distribution point (null: the list covers every certificate of its issuer, for every reason), and its value as encoded.</summary>
    public CmsCrlScope? Scope { get; private set; }

    public ReadOnlyMemory<byte> ScopeEncoding { get; private set; }

    /// <summary>The addresses of its delta lists (freshest CRL).</summary>
    public IReadOnlyList<string> FreshestAddresses { get; private set; } = [];

    /// <summary>Why the list is not used (an unread critical extension, an entry with one, a serial named twice), or null.</summary>
    public string? NotUsable { get; private set; }

    public int Count => _revoked.Count;

    public bool Indirect => Scope?.Indirect == true;

    public bool IsDelta => DeltaBase is not null;

    /// <summary>The reasons it covers (its issuing distribution point's onlySomeReasons, else every one).</summary>
    public int Reasons => Scope?.Reasons ?? AllReasons;

    /// <summary>The list <paramref name="der"/> holds.</summary>
    /// <exception cref="InvalidDataException">It is not a CertificateList's DER.</exception>
    public static CmsCrl Parse(ReadOnlyMemory<byte> der)
    {
        var crl = new CmsCrl(der);
        try
        {
            var list = CmsAsn.Reader(der).ReadSequence();
            crl.Tbs = list.ReadEncodedValue();
            crl.SignatureAlgorithm = CmsAlgorithm.Read(list);
            crl.Signature = list.ReadBitString(out _);
            var tbs = CmsAsn.Reader(crl.Tbs).ReadSequence();
            if (CmsAsn.Next(tbs, Asn1Tag.Integer))
            {
                tbs.ReadInteger();                                             // the version
            }

            CmsAlgorithm.Read(tbs);
            crl.Issuer = tbs.ReadEncodedValue();
            crl._issuers.Add(crl.Issuer);
            crl.ThisUpdate = CmsAsn.ReadTime(tbs);
            if (tbs.HasData && (tbs.PeekTag().HasSameClassAndValue(Asn1Tag.UtcTime) || tbs.PeekTag().HasSameClassAndValue(Asn1Tag.GeneralizedTime)))
            {
                crl.NextUpdate = CmsAsn.ReadTime(tbs);
            }

            bool named = false;
            if (CmsAsn.Next(tbs, Asn1Tag.Sequence))
            {
                var entries = tbs.ReadSequence();
                int issuer = 0;                                                // an entry's issuer stays until another entry names one
                while (entries.HasData)
                {
                    var entry = entries.ReadSequence();
                    var serial = entry.ReadInteger();
                    var date = CmsAsn.ReadTime(entry);
                    int? reason = null;
                    if (entry.HasData)
                    {
                        foreach (var (oid, critical, value) in Extensions(entry.ReadSequence()))
                        {
                            if (oid == CmsOids.CrlReason)
                            {
                                reason = CmsAsn.ReadEnumerated(CmsAsn.Reader(value));
                            }
                            else if (oid == CmsOids.CertificateIssuer)
                            {
                                named = true;
                                issuer = crl.EntryIssuer(value);
                            }
                            else if (critical && oid != CmsOids.InvalidityDate)
                            {
                                crl.NotUsable ??= $"an entry with the critical extension {oid}, which is not read";
                            }
                        }
                    }

                    if (!crl._revoked.TryAdd((issuer, serial), (date, reason)))
                    {
                        crl.NotUsable ??= $"a list naming the serial number {serial} twice";
                    }
                }
            }

            if (CmsAsn.Next(tbs, CmsAsn.Context0))
            {
                foreach (var (oid, critical, value) in Extensions(tbs.ReadSequence(CmsAsn.Context0).ReadSequence()))
                {
                    crl.Extension(oid, critical, value);
                }
            }

            if (named && !crl.Indirect)
            {
                crl.NotUsable ??= "an entry naming its certificate's issuer (certificateIssuer) in a list that is not indirect";
            }

            if (crl.Scope?.OnlyAttribute == true)
            {
                crl.NotUsable ??= "a list of attribute certificates only";
            }

            if (crl.IsDelta && crl.Number is null)
            {
                crl.NotUsable ??= "a delta list without its CRL number";
            }
        }
        catch (Exception e) when (CmsAsn.IsDamage(e) && e is not InvalidDataException)
        {
            throw CmsAsn.Damaged("a revocation list", e);
        }

        return crl;
    }

    // A list's extension: its number, a delta's base, its scope, its key identifier, its deltas' addresses; any other
    // critical one makes the list unusable, as does one of these that cannot be read.
    private void Extension(string oid, bool critical, ReadOnlyMemory<byte> value)
    {
        try
        {
            switch (oid)
            {
                case CmsOids.CrlNumber:
                    Number = CmsAsn.Reader(value).ReadInteger();
                    break;
                case CmsOids.DeltaCrlIndicator:
                    DeltaBase = CmsAsn.Reader(value).ReadInteger();
                    break;
                case CmsOids.IssuingDistributionPoint:
                    Scope = CmsCrlScope.Read(value);
                    ScopeEncoding = value;
                    break;
                case CmsOids.AuthorityKeyId:
                    var authority = CmsAsn.Reader(value).ReadSequence();
                    var keyId = new Asn1Tag(TagClass.ContextSpecific, 0);
                    AuthorityKeyId = CmsAsn.Next(authority, keyId) ? authority.ReadOctetString(keyId) : null;
                    break;
                case CmsOids.FreshestCrl:
                    FreshestAddresses = [.. CmsDistributionPoint.ReadAll(value).SelectMany(p => p.Addresses)];
                    break;
                default:
                    if (critical)
                    {
                        NotUsable ??= $"the critical extension {oid}, which is not read";
                    }

                    break;
            }
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            NotUsable ??= $"the extension {oid}, which cannot be read ({e.Message})";
        }
    }

    // The index of the issuer an entry's certificateIssuer names (its directory name; -1, matching no certificate, without one).
    private int EntryIssuer(ReadOnlyMemory<byte> value)
    {
        foreach (var name in CmsGeneralName.ReadAll(CmsAsn.Reader(value).ReadSequence()))
        {
            if (name.Form == CmsNameForm.Directory)
            {
                int found = _issuers.FindIndex(i => i.Span.SequenceEqual(name.Value.Span));
                if (found >= 0)
                {
                    return found;
                }

                _issuers.Add(name.Value);
                return _issuers.Count - 1;
            }
        }

        return -1;
    }

    /// <summary>The entry naming <paramref name="certificate"/> (its issuer and serial number), or null.</summary>
    public (DateTimeOffset Date, int? Reason)? Entry(CmsCertificate certificate)
    {
        for (int i = 0; i < _issuers.Count; i++)
        {
            if (_issuers[i].Span.SequenceEqual(certificate.Issuer.Span) && _revoked.TryGetValue((i, certificate.Serial), out var entry))
            {
                return entry;
            }
        }

        return null;
    }

    /// <summary>Whether the list is stale at <paramref name="at"/>: its next update, with <paramref name="grace"/>, before it.</summary>
    public bool StaleAt(DateTimeOffset at, TimeSpan grace) => NextUpdate is { } next && at > next && at - next > grace;

    /// <summary>Why <paramref name="signer"/>'s key does not verify the list's signature, or null.</summary>
    public string? VerifyWith(CmsCertificate signer)
    {
        var scheme = CmsKeys.Scheme(SignatureAlgorithm, null, out string? problem);
        return scheme is { } s ? CmsKeys.Verify(signer, s, CmsDigest.Of(s.Digest, Tbs.Span), Signature) : problem;
    }

    /// <summary>
    /// Why the list does not cover <paramref name="certificate"/> by its issuing distribution point (RFC 5280 6.3.3 (b) (2)):
    /// end entities or CAs only, or another distribution point than the certificate's (a certificate naming none is its
    /// issuer's); null when it covers it.
    /// </summary>
    public string? OutOfScope(CmsCertificate certificate)
    {
        if (Scope is not { } scope)
        {
            return null;
        }

        if (scope.OnlyUser && certificate.IsCa == true)
        {
            return "covers end-entity certificates only";
        }

        if (scope.OnlyCa && certificate.IsCa != true)
        {
            return "covers CA certificates only";
        }

        if (scope.Point is not { } point)
        {
            return null;
        }

        var covered = point.Names(Issuer);
        if (certificate.CrlPoints.Count == 0)
        {
            var issuer = new CmsGeneralName(CmsNameForm.Directory, certificate.Issuer);
            var own = new List<(CmsGeneralName Name, CmsNameAttribute[][]? Dn)> { (issuer, CmsDn.Read(certificate.Issuer)) };
            return CmsDistributionPoint.Share(covered, own) ? null : "is for another distribution point than the certificate's issuer";
        }

        foreach (var named in certificate.CrlPoints)
        {
            var issuer = named.CrlIssuer?.FirstOrDefault(n => n.Form == CmsNameForm.Directory) is { Value.IsEmpty: false } crlIssuer ? crlIssuer.Value : certificate.Issuer;
            if (CmsDistributionPoint.Share(covered, named.Names(issuer)))
            {
                return null;
            }
        }

        return "is for another distribution point than the certificate names";
    }

    /// <summary>
    /// What this list says of <paramref name="certificate"/> at <paramref name="at"/>, signed by <paramref name="signer"/> (the
    /// certificate's issuer for a direct list; the list's issuer for an indirect one the certificate's distribution point
    /// names), with the newest of <paramref name="deltas"/> that applies to it (RFC 5280 5.2.4: its issuer's and scope's, for
    /// a base this list's number reaches, newer than it); <see cref="CmsStatus.Unknown"/> with why when it says nothing usable
    /// (another issuer's list, out of its scope, stale beyond <paramref name="grace"/>, not verifying, not usable). Signatures
    /// are verified through <paramref name="verified"/> when a check gives it.
    /// </summary>
    public CmsStatusAnswer StatusOf(CmsCertificate certificate, CmsCertificate signer, DateTimeOffset at, TimeSpan grace = default, CmsVerifications? verified = null,
        IReadOnlyList<CmsCrl>? deltas = null)
    {
        bool direct = Issuer.Span.SequenceEqual(certificate.Issuer.Span);
        if (IsDelta || (!direct && !(Indirect && certificate.NamesCrlIssuer(Issuer))) || !Issuer.Span.SequenceEqual(signer.Subject.Span))
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, "another issuer's list");
        }

        string list = direct ? $"{signer.Named}'s list" : $"{signer.Named}'s indirect list";
        if (NotUsable is not null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{list} is {NotUsable}");
        }

        if (OutOfScope(certificate) is { } scope)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{list} {scope}");
        }

        if (StaleAt(at, grace))
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{list} of {CmsText.Time(ThisUpdate)} is stale at {CmsText.Time(at)} (next update {CmsText.Time(NextUpdate!.Value)}{CmsRevocation.Grace(grace)})", Stale: true);
        }

        if (!signer.Allows(CmsCertificate.CrlSign))
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{signer.Named}'s key may not sign revocation lists (key usage)");
        }

        if (CmsVerifications.Once(verified, this, signer) is { } failed)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{list} does not verify: {failed}");
        }

        var delta = Delta(deltas, signer, at, grace, verified);
        if (delta?.Entry(certificate) is { } changed)
        {
            string newer = $"{signer.Named}'s {(direct ? "" : "indirect ")}delta list of {CmsText.Time(delta.ThisUpdate)}";
            return changed.Reason == 8
                ? new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} taken off {list} of {CmsText.Time(ThisUpdate)} (removeFromCRL) by {newer}", Reasons: Reasons)
                : Listed(certificate, changed, newer, at);
        }

        if (Entry(certificate) is { } revoked && revoked.Reason != 8)          // removeFromCRL (8) takes a hold back
        {
            return Listed(certificate, revoked, $"{list} of {CmsText.Time(ThisUpdate)}", at);
        }

        string also = delta is null ? "" : $" or its delta list of {CmsText.Time(delta.ThisUpdate)}";
        return new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} not on {list} of {CmsText.Time(ThisUpdate)}{also}", Reasons: Reasons);
    }

    // A certificate the list (or its delta) names: revoked at the time, or only after it.
    private CmsStatusAnswer Listed(CmsCertificate certificate, (DateTimeOffset Date, int? Reason) entry, string by, DateTimeOffset at)
    {
        string reason = CmsRevocation.ReasonName(entry.Reason);
        return entry.Date <= at
            ? new CmsStatusAnswer(CmsStatus.Revoked, $"{certificate.Named} revoked on {CmsText.Time(entry.Date)} ({reason}) by {by}", entry.Date, reason)
            : new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} revoked only on {CmsText.Time(entry.Date)} ({reason}), after {CmsText.Time(at)}", entry.Date, reason, Reasons: Reasons);
    }

    // The newest delta list that applies to this one: the same issuer, scope and key identifier, its base number at most this
    // list's, its own above it, usable, fresh, and signed by the same key.
    private CmsCrl? Delta(IReadOnlyList<CmsCrl>? deltas, CmsCertificate signer, DateTimeOffset at, TimeSpan grace, CmsVerifications? verified)
    {
        if (deltas is null || Number is not { } number)
        {
            return null;
        }

        CmsCrl? newest = null;
        foreach (var delta in deltas)
        {
            if (delta.DeltaBase is not { } basis || delta.Number is not { } deltaNumber || basis > number || deltaNumber <= number || delta.NotUsable is not null
                || (newest is not null && newest.Number >= deltaNumber) || !delta.Issuer.Span.SequenceEqual(Issuer.Span)
                || !delta.ScopeEncoding.Span.SequenceEqual(ScopeEncoding.Span) || !SameKeyId(delta.AuthorityKeyId, AuthorityKeyId) || delta.StaleAt(at, grace))
            {
                continue;
            }

            if (CmsVerifications.Once(verified, delta, signer) is null)
            {
                newest = delta;
            }
        }

        return newest;
    }

    private static bool SameKeyId(byte[]? a, byte[]? b) => a is null ? b is null : b is not null && a.AsSpan().SequenceEqual(b);

    internal static List<(string Oid, bool Critical, ReadOnlyMemory<byte> Value)> Extensions(AsnReader sequence)
    {
        var list = new List<(string, bool, ReadOnlyMemory<byte>)>();
        while (sequence.HasData)
        {
            var extension = sequence.ReadSequence();
            string oid = extension.ReadObjectIdentifier();
            bool critical = CmsAsn.Next(extension, Asn1Tag.Boolean) && extension.ReadBoolean();
            list.Add((oid, critical, extension.ReadOctetString()));
        }

        return list;
    }
}

/// <summary>
/// An OCSP response (RFC 6960) read on its DER: a successful basic response's signed data (who answers, when, each
/// certificate's status), its signature and the certificates it carries.
/// </summary>
internal sealed class CmsOcsp
{
    private CmsOcsp(ReadOnlyMemory<byte> raw) => Raw = raw;

    public ReadOnlyMemory<byte> Raw { get; }

    public ReadOnlyMemory<byte> Tbs { get; private set; }

    public CmsAlgorithm SignatureAlgorithm { get; private set; }

    public byte[] Signature { get; private set; } = [];

    /// <summary>The responder by name (its encoding) or by its key's SHA-1.</summary>
    public ReadOnlyMemory<byte> ResponderName { get; private set; }

    public byte[]? ResponderKeyHash { get; private set; }

    public DateTimeOffset ProducedAt { get; private set; }

    public IReadOnlyList<CmsCertificate> Certificates { get; private set; } = [];

    public IReadOnlyList<Answer> Responses { get; private set; } = [];

    /// <summary>The response's status when it is not successful (named), else null.</summary>
    public string? Failure { get; private set; }

    /// <summary>One certificate's answer: who it names (the issuer's name and key hashes, the serial), its status and times.</summary>
    public sealed record Answer(HashAlgorithmName? Digest, byte[] IssuerNameHash, byte[] IssuerKeyHash, BigInteger Serial, CmsStatus Status, DateTimeOffset? RevokedAt,
        int? Reason, DateTimeOffset ThisUpdate, DateTimeOffset? NextUpdate);

    /// <summary>The response <paramref name="der"/> holds.</summary>
    /// <exception cref="InvalidDataException">It is not an OCSPResponse's DER, or not a basic one.</exception>
    public static CmsOcsp Parse(ReadOnlyMemory<byte> der)
    {
        var ocsp = new CmsOcsp(der);
        try
        {
            var response = CmsAsn.Reader(der).ReadSequence();
            int status = CmsAsn.ReadEnumerated(response);
            if (status != 0)
            {
                ocsp.Failure = status switch
                {
                    1 => "malformedRequest",
                    2 => "internalError",
                    3 => "tryLater",
                    5 => "sigRequired",
                    6 => "unauthorized",
                    _ => $"status {status}",
                };
                return ocsp;
            }

            var bytes = response.ReadSequence(CmsAsn.Context0).ReadSequence();
            string type = bytes.ReadObjectIdentifier();
            if (type != CmsOids.OcspBasic)
            {
                throw new InvalidDataException($"an OCSP response of the type {type}, not a basic one");
            }

            var basic = CmsAsn.Reader(bytes.ReadOctetString()).ReadSequence();
            ocsp.Tbs = basic.ReadEncodedValue();
            ocsp.SignatureAlgorithm = CmsAlgorithm.Read(basic);
            ocsp.Signature = basic.ReadBitString(out _);
            var certificates = new List<CmsCertificate>();
            if (CmsAsn.Next(basic, CmsAsn.Context0))
            {
                var list = basic.ReadSequence(CmsAsn.Context0).ReadSequence();
                while (list.HasData)
                {
                    if (CmsCertificate.TryParse(list.ReadEncodedValue().Span) is { } certificate)
                    {
                        certificates.Add(certificate);
                    }
                }
            }

            ocsp.Certificates = certificates;
            var data = CmsAsn.Reader(ocsp.Tbs).ReadSequence();
            if (CmsAsn.Next(data, CmsAsn.Context0))
            {
                data.ReadSequence(CmsAsn.Context0).ReadInteger();              // the version
            }

            if (CmsAsn.Next(data, CmsAsn.Context1))
            {
                ocsp.ResponderName = data.ReadSequence(CmsAsn.Context1).ReadEncodedValue();
            }
            else
            {
                ocsp.ResponderKeyHash = data.ReadSequence(CmsAsn.Context2).ReadOctetString();
            }

            ocsp.ProducedAt = data.ReadGeneralizedTime();
            var answers = new List<Answer>();
            var responses = data.ReadSequence();
            while (responses.HasData)
            {
                var single = responses.ReadSequence();
                var id = single.ReadSequence();
                var digest = CmsDigest.FromOid(CmsAlgorithm.Read(id).Oid);
                byte[] nameHash = id.ReadOctetString(), keyHash = id.ReadOctetString();
                var serial = id.ReadInteger();
                var tag = single.PeekTag();
                CmsStatus answer;
                DateTimeOffset? revokedAt = null;
                int? reason = null;
                if (tag.TagValue == 0 && tag.TagClass == TagClass.ContextSpecific)
                {
                    single.ReadNull(new Asn1Tag(TagClass.ContextSpecific, 0));
                    answer = CmsStatus.Good;
                }
                else if (tag.TagValue == 1 && tag.TagClass == TagClass.ContextSpecific)
                {
                    var revoked = single.ReadSequence(new Asn1Tag(TagClass.ContextSpecific, 1, isConstructed: true));
                    revokedAt = revoked.ReadGeneralizedTime();
                    if (CmsAsn.Next(revoked, CmsAsn.Context0))
                    {
                        reason = CmsAsn.ReadEnumerated(revoked.ReadSequence(CmsAsn.Context0));
                    }

                    answer = CmsStatus.Revoked;
                }
                else
                {
                    single.ReadEncodedValue();
                    answer = CmsStatus.Unknown;
                }

                var thisUpdate = single.ReadGeneralizedTime();
                DateTimeOffset? nextUpdate = CmsAsn.Next(single, CmsAsn.Context0) ? single.ReadSequence(CmsAsn.Context0).ReadGeneralizedTime() : null;
                answers.Add(new Answer(digest, nameHash, keyHash, serial, answer, revokedAt, reason, thisUpdate, nextUpdate));
            }

            ocsp.Responses = answers;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e) && e is not InvalidDataException)
        {
            throw CmsAsn.Damaged("an OCSP response", e);
        }

        return ocsp;
    }

    /// <summary>
    /// What this response says of <paramref name="certificate"/> at <paramref name="at"/>: when it names the certificate (by
    /// its issuer's name and key and its serial), is signed by <paramref name="issuer"/> itself or by a responder
    /// <paramref name="issuer"/> issued for OCSP signing, and was not stale at the time (its next update, with
    /// <paramref name="grace"/>, before it). Signatures are verified through <paramref name="verified"/> when a check gives it.
    /// </summary>
    public CmsStatusAnswer StatusOf(CmsCertificate certificate, CmsCertificate issuer, DateTimeOffset at, TimeSpan grace = default, CmsVerifications? verified = null)
    {
        if (Failure is not null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"an OCSP response that is not successful ({Failure})");
        }

        Answer? found = null;
        foreach (var single in Responses)
        {
            if (single.Serial == certificate.Serial && single.Digest is { } digest
                && CmsDigest.Of(digest, issuer.Subject.Span).AsSpan().SequenceEqual(single.IssuerNameHash)
                && CmsDigest.Of(digest, issuer.PublicKey).AsSpan().SequenceEqual(single.IssuerKeyHash))
            {
                found = single;
                break;
            }
        }

        if (found is null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, "a response for another certificate");
        }

        if (Responder(issuer, verified) is not { } responder)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"an OCSP response for {certificate.Named} signed neither by {issuer.Named} nor by a responder it authorized");
        }

        string? failed = CmsVerifications.Once(verified, this, responder);
        if (failed is not null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"the OCSP response for {certificate.Named} does not verify: {failed}");
        }

        if (found.NextUpdate is { } next && at > next && at - next > grace)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown,
                $"the OCSP response for {certificate.Named} of {CmsText.Time(found.ThisUpdate)} is stale at {CmsText.Time(at)} (next update {CmsText.Time(next)}{CmsRevocation.Grace(grace)})", Stale: true);
        }

        string by = ReferenceEquals(responder, issuer) ? issuer.Named : $"{responder.Named} for {issuer.Named}";
        switch (found.Status)
        {
            case CmsStatus.Good:
                return new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} good in {by}'s OCSP response of {CmsText.Time(found.ThisUpdate)}");
            case CmsStatus.Revoked when found.RevokedAt is { } when:
                string reason = CmsRevocation.ReasonName(found.Reason);
                return when <= at
                    ? new CmsStatusAnswer(CmsStatus.Revoked, $"{certificate.Named} revoked on {CmsText.Time(when)} ({reason}) in {by}'s OCSP response", when, reason)
                    : new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} revoked only on {CmsText.Time(when)} ({reason}), after {CmsText.Time(at)}", when, reason);
            default:
                return new CmsStatusAnswer(CmsStatus.Unknown, $"{by}'s OCSP response does not know {certificate.Named}");
        }
    }

    /// <summary>Why <paramref name="responder"/>'s key does not verify the response's signature, or null.</summary>
    public string? VerifyWith(CmsCertificate responder)
    {
        var scheme = CmsKeys.Scheme(SignatureAlgorithm, null, out string? problem);
        return scheme is { } s ? CmsKeys.Verify(responder, s, CmsDigest.Of(s.Digest, Tbs.Span), Signature) : problem;
    }

    // The certificate that signed the response: the issuer itself, or one the issuer issued for OCSP signing (RFC 6960, 4.2.2.2).
    private CmsCertificate? Responder(CmsCertificate issuer, CmsVerifications? verified)
    {
        if (Names(issuer))
        {
            return issuer;
        }

        foreach (var candidate in Certificates)
        {
            if (Names(candidate) && candidate.NamesIssuer(issuer) && candidate.HasPurpose(CmsOids.OcspSigning) && candidate.Allows(CmsCertificate.DigitalSignature)
                && CmsVerifications.Once(verified, candidate, issuer) is null)
            {
                return candidate;
            }
        }

        return null;
    }

    private bool Names(CmsCertificate certificate) =>
        ResponderKeyHash is { } hash ? SHA1.HashData(certificate.PublicKey).AsSpan().SequenceEqual(hash) : ResponderName.Span.SequenceEqual(certificate.Subject.Span);
}

/// <summary>Revocation reasons by name, and the OCSP request a responder is asked with (RFC 6960, 4.1).</summary>
internal static class CmsRevocation
{
    /// <summary>The grace a stale source was checked with, for its message ("" without one).</summary>
    public static string Grace(TimeSpan grace) => grace > TimeSpan.Zero ? $", taken {CmsText.Span(grace)} past it" : "";

    public static string ReasonName(int? reason) => reason switch
    {
        null => "no reason given",
        0 => "unspecified",
        1 => "keyCompromise",
        2 => "cACompromise",
        3 => "affiliationChanged",
        4 => "superseded",
        5 => "cessationOfOperation",
        6 => "certificateHold",
        8 => "removeFromCRL",
        9 => "privilegeWithdrawn",
        10 => "aACompromise",
        _ => $"reason {reason}",
    };

    /// <summary>An OCSPRequest for <paramref name="certificate"/> issued by <paramref name="issuer"/> (its CertID by SHA-1, as responders expect), with a nonce.</summary>
    public static byte[] Request(CmsCertificate certificate, CmsCertificate issuer)
    {
        var writer = new AsnWriter(AsnEncodingRules.DER);
        using (writer.PushSequence())                                          // OCSPRequest
        {
            using (writer.PushSequence())                                      // TBSRequest
            {
                using (writer.PushSequence())                                  // requestList
                {
                    using (writer.PushSequence())                              // Request
                    {
                        using (writer.PushSequence())                          // CertID
                        {
                            CmsAsn.WriteAlgorithm(writer, CmsOids.Sha1, nullParameters: true);
                            writer.WriteOctetString(SHA1.HashData(issuer.Subject.Span));
                            writer.WriteOctetString(SHA1.HashData(issuer.PublicKey));
                            writer.WriteInteger(certificate.Serial);
                        }
                    }
                }

                using (writer.PushSequence(CmsAsn.Context2))                   // requestExtensions
                {
                    using (writer.PushSequence())
                    {
                        using (writer.PushSequence())
                        {
                            writer.WriteObjectIdentifier(CmsOids.OcspNonce);
                            var nonce = new AsnWriter(AsnEncodingRules.DER);
                            nonce.WriteOctetString(RandomNumberGenerator.GetBytes(16));
                            writer.WriteOctetString(nonce.Encode());
                        }
                    }
                }
            }
        }

        return writer.Encode();
    }
}
