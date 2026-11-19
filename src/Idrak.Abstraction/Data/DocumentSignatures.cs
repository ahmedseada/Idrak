// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>The form a signature takes.</summary>
public enum DocumentSignatureKind : byte
{
    /// <summary>A detached CMS signature of the signed bytes (no baseline level stated).</summary>
    Cms,

    /// <summary>PAdES baseline B-B: a CMS signature with the signer's certificate bound to it.</summary>
    BaselineB,

    /// <summary>PAdES baseline B-T: B-B with a trusted timestamp.</summary>
    BaselineT,

    /// <summary>PAdES baseline B-LT: B-T with the validation data (certificates, revocation) in the file.</summary>
    BaselineLt,

    /// <summary>PAdES baseline B-LTA: B-LT with a document timestamp over it.</summary>
    BaselineLta,

    /// <summary>The legacy form: an RSA (PKCS #1) signature with the certificates beside it.</summary>
    LegacyPkcs1,

    /// <summary>The legacy form: a PKCS #7 signature of a SHA-1 digest of the signed bytes.</summary>
    LegacyPkcs7Sha1,

    /// <summary>Another form, named in <see cref="DocumentSignature.KindName"/>.</summary>
    Other,
}

/// <summary>What changed in a document after a signature was made.</summary>
public enum DocumentSignatureChanges : byte
{
    /// <summary>Nothing: the signature covers the whole file.</summary>
    None,

    /// <summary>Changes the signature's permissions allow (forms filled, annotations added, later signatures).</summary>
    Allowed,

    /// <summary>Changes the signature's permissions do not allow.</summary>
    NotAllowed,
}

/// <summary>One step of a signature check, passed or not, and why.</summary>
/// <param name="Name">The step: <see cref="SignatureCheck.Digest"/>, <see cref="SignatureCheck.Signature"/>, <see cref="SignatureCheck.Chain"/>, <see cref="SignatureCheck.Timestamp"/>, <see cref="SignatureCheck.Revocation"/> or <see cref="SignatureCheck.Changes"/>.</param>
/// <param name="Passed">Whether it passed.</param>
/// <param name="Detail">What was found, for a person.</param>
public readonly record struct SignatureCheckStep(string Name, bool Passed, string Detail);

/// <summary>A signature checked: each step, and whether all passed. A signature never makes a document trusted on its own.</summary>
/// <param name="Steps">The steps, in the order they ran.</param>
public sealed record SignatureCheck(IReadOnlyList<SignatureCheckStep> Steps)
{
    /// <summary>The digest of the signed bytes is the one signed.</summary>
    public const string Digest = "digest";

    /// <summary>The signature over the signed attributes verifies with the signer's key.</summary>
    public const string Signature = "signature";

    /// <summary>The signer's certificate chains to a certificate the app trusts.</summary>
    public const string Chain = "chain";

    /// <summary>The timestamp's token verifies and covers the signature.</summary>
    public const string Timestamp = "timestamp";

    /// <summary>No certificate of the chain was revoked (from the document's own validation data, or online when allowed).</summary>
    public const string Revocation = "revocation";

    /// <summary>What changed after signing is allowed.</summary>
    public const string Changes = "changes";

    /// <summary>Whether every step passed (false when there is none).</summary>
    public bool Valid => Steps.Count > 0 && Steps.All(s => s.Passed);

    /// <summary>
    /// The certificate policies the signer's path is valid for (RFC 5280 6.1's user-constrained policy set, as object
    /// identifiers in ordinal order; <see cref="SignatureCheckOptions.AnyPolicy"/> when the path holds every policy), or
    /// empty: no path was built, or its certificates leave no policy (which passes unless an explicit policy is required).
    /// </summary>
    public IReadOnlyList<string> Policies { get; init; } = [];
}

/// <summary>
/// What a signature check trusts and may do. Only the certificates the app gives are trusted; the machine's store only when
/// the app asks for it, and nothing is fetched from the network unless the app turns it on
/// (<see cref="OnlineVariable"/>).
/// </summary>
public sealed record SignatureCheckOptions
{
    /// <summary>The variable that lets a check fetch revocation data online (off unless set to on): "IDRAK_DOCUMENT_SIGNATURE_ONLINE".</summary>
    public const string OnlineVariable = "IDRAK_DOCUMENT_SIGNATURE_ONLINE";

    /// <summary>The variable that sets <see cref="MinimumRsaKeyBits"/> (0 to 16384): "IDRAK_DOCUMENT_SIGNATURE_RSA_BITS".</summary>
    public const string MinimumRsaKeyBitsVariable = "IDRAK_DOCUMENT_SIGNATURE_RSA_BITS";

    /// <summary>The variable that sets <see cref="MaxChainLength"/> (1 to 64): "IDRAK_DOCUMENT_SIGNATURE_CHAIN".</summary>
    public const string MaxChainLengthVariable = "IDRAK_DOCUMENT_SIGNATURE_CHAIN";

    /// <summary>
    /// The variable that sets <see cref="InitialPolicies"/> (policy object identifiers separated by commas or spaces; "any" or
    /// unset: any policy): "IDRAK_DOCUMENT_SIGNATURE_POLICIES".
    /// </summary>
    public const string InitialPoliciesVariable = "IDRAK_DOCUMENT_SIGNATURE_POLICIES";

    /// <summary>The variable that sets <see cref="RequireExplicitPolicy"/> (on or off): "IDRAK_DOCUMENT_SIGNATURE_EXPLICIT_POLICY".</summary>
    public const string RequireExplicitPolicyVariable = "IDRAK_DOCUMENT_SIGNATURE_EXPLICIT_POLICY";

    /// <summary>The variable that sets <see cref="InhibitPolicyMapping"/> (on or off): "IDRAK_DOCUMENT_SIGNATURE_INHIBIT_MAPPING".</summary>
    public const string InhibitPolicyMappingVariable = "IDRAK_DOCUMENT_SIGNATURE_INHIBIT_MAPPING";

    /// <summary>The variable that sets <see cref="InhibitAnyPolicy"/> (on or off): "IDRAK_DOCUMENT_SIGNATURE_INHIBIT_ANY_POLICY".</summary>
    public const string InhibitAnyPolicyVariable = "IDRAK_DOCUMENT_SIGNATURE_INHIBIT_ANY_POLICY";

    /// <summary>
    /// The variable that sets <see cref="FreshnessGrace"/> (seconds, or a number with s, m, h or d; 0 to 3650 days):
    /// "IDRAK_DOCUMENT_SIGNATURE_GRACE".
    /// </summary>
    public const string FreshnessGraceVariable = "IDRAK_DOCUMENT_SIGNATURE_GRACE";

    /// <summary>The variable that sets <see cref="MaxNameConstraintChecks"/> (1 to 1073741824): "IDRAK_DOCUMENT_SIGNATURE_NAME_CHECKS".</summary>
    public const string MaxNameConstraintChecksVariable = "IDRAK_DOCUMENT_SIGNATURE_NAME_CHECKS";

    /// <summary>The policy identifier anyPolicy (RFC 5280 4.2.1.4): "2.5.29.32.0".</summary>
    public const string AnyPolicy = "2.5.29.32.0";

    private static readonly TimeSpan MostGrace = TimeSpan.FromDays(3650);

    private readonly int _minimumRsaKeyBits = DocumentSettings.Int(MinimumRsaKeyBitsVariable, 0, 16384) ?? 2048;
    private readonly int _maxChainLength = DocumentSettings.Int(MaxChainLengthVariable, 1, 64) ?? 12;
    private readonly IReadOnlyList<string> _initialPolicies = PoliciesOf(InitialPoliciesVariable);
    private readonly TimeSpan _freshnessGrace = GraceOf(FreshnessGraceVariable) ?? TimeSpan.Zero;
    private readonly int _maxNameConstraintChecks = DocumentSettings.Int(MaxNameConstraintChecksVariable, 1, 1 << 30) ?? 1 << 20;

    /// <summary>The certificates trusted as roots of a signer's chain.</summary>
    public IReadOnlyList<X509Certificate2> Trusted { get; init; } = [];

    /// <summary>Whether the machine's store of trusted roots is trusted too (false).</summary>
    public bool UseMachineStore { get; init; }

    /// <summary>
    /// Whether revocation data the document lacks may be fetched online (off unless <see cref="OnlineVariable"/> turns it on),
    /// from the addresses the certificates name, through <see cref="SignatureServices"/>.
    /// </summary>
    public bool Online { get; init; } = DocumentSettings.Switch(OnlineVariable) ?? false;

    /// <summary>The time the chain is checked at, or null for the signature's trusted time (its timestamp's), else now.</summary>
    public DateTimeOffset? At { get; init; }

    /// <summary>
    /// Whether a certificate with no revocation data (in the document, or online when <see cref="Online"/>) fails the
    /// revocation step (true), or passes it as "not checked" (false: what is found revoked always fails).
    /// </summary>
    public bool RequireRevocation { get; init; }

    /// <summary>How long a service asked online may take to answer (30 seconds).</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// The fewest bits an RSA key of a signer's path may have, the signer's own and the trusted one's included (2048, as
    /// pyHanko's default policy; <see cref="MinimumRsaKeyBitsVariable"/>): a path with a shorter key fails the chain step,
    /// naming the key and its size, while its signatures still verify. 0 takes a key of any size (an archive of old
    /// documents).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Below 0 or above 16384.</exception>
    public int MinimumRsaKeyBits
    {
        get => _minimumRsaKeyBits;
        init => _minimumRsaKeyBits = value is >= 0 and <= 16384 ? value
            : throw new ArgumentOutOfRangeException(nameof(MinimumRsaKeyBits), value, "The fewest bits an RSA key may have: 0 (any size) to 16384.");
    }

    /// <summary>
    /// The most certificates a signer's path may hold, the signer's and the trusted one's included (12;
    /// <see cref="MaxChainLengthVariable"/>): a longer path fails the chain step, naming the limit. The path search also tries
    /// at most 64 issuers for each certificate a path may hold, so the certificates a document carries cannot keep a check
    /// searching.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Below 1 or above 64.</exception>
    public int MaxChainLength
    {
        get => _maxChainLength;
        init => _maxChainLength = value is >= 1 and <= 64 ? value
            : throw new ArgumentOutOfRangeException(nameof(MaxChainLength), value, "The most certificates a path may hold: 1 to 64.");
    }

    /// <summary>
    /// The certificate policies a signer's path must be valid for, as object identifiers (RFC 5280 6.1.1's
    /// user-initial-policy-set; <see cref="InitialPoliciesVariable"/>): empty, the default, takes any policy (anyPolicy, as
    /// does a list holding <see cref="AnyPolicy"/>). A path for none of them fails the chain step only when an explicit
    /// policy is required (<see cref="RequireExplicitPolicy"/>, or a CA's policy constraints); the policies a path is valid
    /// for are <see cref="SignatureCheck.Policies"/>. A timestamp authority's path is checked for any policy.
    /// </summary>
    /// <exception cref="ArgumentException">An item that is not a dotted object identifier.</exception>
    public IReadOnlyList<string> InitialPolicies
    {
        get => _initialPolicies;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            foreach (string policy in value)
            {
                if (!IsOid(policy))
                {
                    throw new ArgumentException($"'{policy}' is not a policy's object identifier (such as 2.999.1.2; {AnyPolicy} for any policy).", nameof(InitialPolicies));
                }
            }

            _initialPolicies = [.. value];
        }
    }

    /// <summary>
    /// Whether a signer's path must be valid for a policy of <see cref="InitialPolicies"/> (RFC 5280's initial-explicit-policy;
    /// false; <see cref="RequireExplicitPolicyVariable"/>): with it, a path whose certificates leave none fails the chain step.
    /// </summary>
    public bool RequireExplicitPolicy { get; init; } = DocumentSettings.Switch(RequireExplicitPolicyVariable) ?? false;

    /// <summary>
    /// Whether the policy mappings of a signer's path's CAs are refused (RFC 5280's initial-policy-mapping-inhibit; false;
    /// <see cref="InhibitPolicyMappingVariable"/>): with it, a policy a CA maps leaves the path.
    /// </summary>
    public bool InhibitPolicyMapping { get; init; } = DocumentSettings.Switch(InhibitPolicyMappingVariable) ?? false;

    /// <summary>
    /// Whether anyPolicy in a signer's path's certificates is not taken as every policy (RFC 5280's initial-any-policy-inhibit;
    /// false; <see cref="InhibitAnyPolicyVariable"/>).
    /// </summary>
    public bool InhibitAnyPolicy { get; init; } = DocumentSettings.Switch(InhibitAnyPolicyVariable) ?? false;

    /// <summary>
    /// How long past its next update a revocation list or an OCSP response is still taken as fresh (zero: none;
    /// <see cref="FreshnessGraceVariable"/>). Revocation data whose next update, with this grace, is before the time a check
    /// is made at is stale: fetched again when <see cref="Online"/>, else the revocation step says "not checked (stale)", which
    /// fails with <see cref="RequireRevocation"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Negative, or more than 3650 days.</exception>
    public TimeSpan FreshnessGrace
    {
        get => _freshnessGrace;
        init => _freshnessGrace = value >= TimeSpan.Zero && value <= MostGrace ? value
            : throw new ArgumentOutOfRangeException(nameof(FreshnessGrace), value, "A revocation list's grace past its next update: zero to 3650 days.");
    }

    /// <summary>
    /// The most name comparisons a path's name constraints may ask for (1,048,576, OpenSSL's bound;
    /// <see cref="MaxNameConstraintChecksVariable"/>): each constrained certificate's names (its subject's attributes and its
    /// alternative names) times the constraints that apply to it, over the path. A path asking for more fails the chain step,
    /// naming the limit, before any is compared, so the certificates a document carries cannot keep a check comparing.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Below 1 or above 1073741824.</exception>
    public int MaxNameConstraintChecks
    {
        get => _maxNameConstraintChecks;
        init => _maxNameConstraintChecks = value is >= 1 and <= 1 << 30 ? value
            : throw new ArgumentOutOfRangeException(nameof(MaxNameConstraintChecks), value, "The most name comparisons a path's constraints may ask for: 1 to 1073741824.");
    }

    // A dotted object identifier: two or more arcs of digits, the first 0, 1 or 2.
    private static bool IsOid(string? text)
    {
        if (string.IsNullOrEmpty(text) || text[0] is not ('0' or '1' or '2'))
        {
            return false;
        }

        int arcs = 0;
        foreach (var arc in text.AsSpan().Split('.'))
        {
            var digits = text.AsSpan()[arc];
            if (digits.IsEmpty)
            {
                return false;
            }

            foreach (char digit in digits)
            {
                if (!char.IsAsciiDigit(digit))
                {
                    return false;
                }
            }

            arcs++;
        }

        return arcs >= 2;
    }

    // The initial policies a variable names: object identifiers separated by commas or spaces; "any" (or unset) is any.
    private static IReadOnlyList<string> PoliciesOf(string variable)
    {
        if (DocumentSettings.Text(variable) is not { } text || text.Equals("any", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        var policies = new List<string>();
        foreach (string item in text.Split([',', ' ', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            policies.Add(IsOid(item) ? item : throw new ArgumentException($"{variable}={text}: policy object identifiers separated by commas (such as 2.999.1.2), or any.", variable));
        }

        return policies;
    }

    // A grace a variable names: seconds, or a number followed by s, m, h or d; 0 to 3650 days.
    private static TimeSpan? GraceOf(string variable)
    {
        if (DocumentSettings.Text(variable) is not { } text)
        {
            return null;
        }

        var number = text.AsSpan();
        double unit = 1;
        if (number.Length > 0 && char.ToLowerInvariant(number[^1]) is 's' or 'm' or 'h' or 'd')
        {
            unit = char.ToLowerInvariant(number[^1]) switch { 'm' => 60, 'h' => 3600, 'd' => 86400, _ => 1 };
            number = number[..^1].TrimEnd();
        }

        return double.TryParse(number, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double value)
               && value >= 0 && value * unit <= MostGrace.TotalSeconds
            ? TimeSpan.FromSeconds(value * unit)
            : throw new ArgumentException($"{variable}={text}: a grace of 0 to 3650 days (seconds, or a number with s, m, h or d, such as 3600, 30m, 12h or 7d).", variable);
    }
}

/// <summary>
/// A signature in a document: who signed, when, what it covers and in which form, and a check of it against the trust the
/// app gives (<see cref="Check"/>).
/// </summary>
public sealed class DocumentSignature
{
    private readonly Func<SignatureCheckOptions, SignatureCheck> _check;

    /// <summary>The signature of the field <paramref name="field"/>, checked by <paramref name="check"/>.</summary>
    /// <param name="field">The signature field's full name.</param>
    /// <param name="kind">Its form.</param>
    /// <param name="check">Checks it (thread-safe).</param>
    public DocumentSignature(string field, DocumentSignatureKind kind, Func<SignatureCheckOptions, SignatureCheck> check)
    {
        ArgumentNullException.ThrowIfNull(field);
        ArgumentNullException.ThrowIfNull(check);
        (Field, Kind, _check) = (field, kind, check);
    }

    /// <summary>The signature field's full name.</summary>
    public string Field { get; }

    /// <summary>Its form.</summary>
    public DocumentSignatureKind Kind { get; }

    /// <summary>The form's name as the document gives it, or null.</summary>
    public string? KindName { get; init; }

    /// <summary>The signer's certificate, or null when the signature holds none.</summary>
    public X509Certificate2? Signer { get; init; }

    /// <summary>When the signer says it signed (not checked), or null.</summary>
    public DateTimeOffset? SignedAt { get; init; }

    /// <summary>When a timestamp says it was signed (checked by <see cref="Check"/>), or null when it has none.</summary>
    public DateTimeOffset? TimestampedAt { get; init; }

    /// <summary>Why it was signed, or null.</summary>
    public string? Reason { get; init; }

    /// <summary>Where it was signed, or null.</summary>
    public string? Location { get; init; }

    /// <summary>The bytes it covers: offset and length pairs into the file.</summary>
    public IReadOnlyList<long> ByteRange { get; init; } = [];

    /// <summary>Whether it covers the whole file (true), else what changed after it was made (<see cref="Changes"/>).</summary>
    public bool CoversWholeFile => Changes == DocumentSignatureChanges.None;

    /// <summary>What changed after it was made.</summary>
    public DocumentSignatureChanges Changes { get; init; }

    /// <summary>Checks it: the digest, the signature, the chain against <paramref name="options"/>' trust, the timestamp, revocation and the changes made since.</summary>
    public SignatureCheck Check(SignatureCheckOptions? options = null) => _check(options ?? new SignatureCheckOptions());
}

/// <summary>The PAdES baseline level a document is signed at.</summary>
public enum DocumentSignatureLevel : byte
{
    /// <summary>B-B: the signature with the signer's certificate bound to it.</summary>
    BaselineB,

    /// <summary>B-T: with a timestamp from the authority the app names (<see cref="SigningOptions.TimestampAuthority"/>).</summary>
    BaselineT,

    /// <summary>B-LT: B-T with the chain's revocation data added to the document.</summary>
    BaselineLt,
}

/// <summary>
/// How a document is signed: the certificate with its private key, the chain to embed, the digest and the level, and the
/// field's appearance. Nothing is fetched unless the app names a timestamp authority.
/// </summary>
/// <param name="Certificate">The signer's certificate, with its private key.</param>
public sealed record SigningOptions(X509Certificate2 Certificate)
{
    /// <summary>The certificates between the signer's and a root, to embed.</summary>
    public IReadOnlyList<X509Certificate2> Chain { get; init; } = [];

    /// <summary>The digest (SHA-256).</summary>
    public HashAlgorithmName Digest { get; init; } = HashAlgorithmName.SHA256;

    /// <summary>How an RSA key signs (<see cref="RSASignaturePadding.Pkcs1"/>; <see cref="RSASignaturePadding.Pss"/> for RSA-PSS); not used by other keys.</summary>
    public RSASignaturePadding RsaPadding { get; init; } = RSASignaturePadding.Pkcs1;

    /// <summary>The level (<see cref="DocumentSignatureLevel.BaselineB"/>).</summary>
    public DocumentSignatureLevel Level { get; init; }

    /// <summary>The timestamp authority's address, for <see cref="DocumentSignatureLevel.BaselineT"/> and above; null for none.</summary>
    public Uri? TimestampAuthority { get; init; }

    /// <summary>The time the signer claims to sign at (the signature dictionary's), or null for now.</summary>
    public DateTimeOffset? SigningTime { get; init; }

    /// <summary>
    /// Revocation data for <see cref="DocumentSignatureLevel.BaselineLt"/>, as the issuers sign it (DER): revocation lists and
    /// status responses for the chain's certificates (the signer's, the chain's and the timestamp authority's), added to the
    /// document with the certificates.
    /// </summary>
    public IReadOnlyList<ReadOnlyMemory<byte>> Revocation { get; init; } = [];

    /// <summary>
    /// Whether revocation data <see cref="Revocation"/> lacks for <see cref="DocumentSignatureLevel.BaselineLt"/> may be
    /// fetched from the addresses the certificates name, through <see cref="SignatureServices"/> (off unless
    /// <see cref="SignatureCheckOptions.OnlineVariable"/> turns it on).
    /// </summary>
    public bool Online { get; init; } = DocumentSettings.Switch(SignatureCheckOptions.OnlineVariable) ?? false;

    /// <summary>How long a service asked (the timestamp authority, a revocation address) may take to answer (30 seconds).</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>Why it is signed, or null.</summary>
    public string? Reason { get; init; }

    /// <summary>Where it is signed, or null.</summary>
    public string? Location { get; init; }

    /// <summary>The page a new signature field goes on (its index), when the field does not exist yet.</summary>
    public int Page { get; init; }

    /// <summary>The box of a new signature field; null for an invisible signature.</summary>
    public DocumentBox? Box { get; init; }

    /// <summary>The text the field shows, or null.</summary>
    public string? AppearanceText { get; init; }

    /// <summary>An image the field shows, or null.</summary>
    public ImageData? AppearanceImage { get; init; }

    // The certificate is named, never printed with its key.
    private bool PrintMembers(StringBuilder builder)
    {
        builder.Append("Certificate = ").Append(Certificate.Subject).Append(", Chain = ").Append(Chain.Count).Append(" certificates, Digest = ").Append(Digest.Name)
            .Append(", Level = ").Append(Level).Append(", TimestampAuthority = ").Append(TimestampAuthority?.ToString() ?? "none");
        return true;
    }
}

/// <summary>What a password-protected document lets a user who opens it with the user password do.</summary>
[Flags]
public enum DocumentPermissions
{
    /// <summary>Nothing but reading.</summary>
    None = 0,

    /// <summary>Print.</summary>
    Print = 1,

    /// <summary>Change the content.</summary>
    Modify = 2,

    /// <summary>Copy text and images out.</summary>
    Copy = 4,

    /// <summary>Add and change annotations, and fill forms.</summary>
    Annotate = 8,

    /// <summary>Fill forms (even without <see cref="Annotate"/>).</summary>
    FillForms = 16,

    /// <summary>Copy out for accessibility.</summary>
    Accessibility = 32,

    /// <summary>Insert, remove, rotate and reorder pages.</summary>
    Assemble = 64,

    /// <summary>Print at full quality.</summary>
    PrintHighQuality = 128,

    /// <summary>Everything.</summary>
    All = Print | Modify | Copy | Annotate | FillForms | Accessibility | Assemble | PrintHighQuality,
}

/// <summary>
/// How a written document is encrypted: with the format's strongest cipher (AES-256 where the format has it), a user
/// password (to open it) and an owner password (to change what is permitted), and the permissions a user gets.
/// </summary>
public sealed record DocumentEncryption
{
    /// <summary>The password to open it (null or empty: it opens without one, the permissions still apply).</summary>
    public string? UserPassword { get; init; }

    /// <summary>The password that lifts the permissions (null: one the app cannot know is made, so they cannot be lifted).</summary>
    public string? OwnerPassword { get; init; }

    /// <summary>What a user may do (<see cref="DocumentPermissions.All"/>).</summary>
    public DocumentPermissions Permissions { get; init; } = DocumentPermissions.All;

    // The passwords are never printed.
    private bool PrintMembers(StringBuilder builder)
    {
        builder.Append("UserPassword = ").Append(UserPassword is null ? "none" : "(set)").Append(", OwnerPassword = ").Append(OwnerPassword is null ? "none" : "(set)")
            .Append(", Permissions = ").Append(Permissions);
        return true;
    }
}

/// <summary>
/// How a document that was read is encrypted (<see cref="DocumentInfo.Security"/>), as its security handler states it:
/// only what stays the same when the file is saved again with its own encryption (the handler, its revision, the key's
/// length, the ciphers, the permissions, whether the metadata is encrypted), never a key, a salt, an initialization vector
/// or the file's identifier, which the standard makes random; and whether its owner password opened it.
/// </summary>
public sealed record DocumentSecurity
{
    /// <summary>The security handler, as the file names it ("Standard" for PDF's password handler).</summary>
    public string Handler { get; init; } = "";

    /// <summary>The handler's revision (PDF's standard handler: 2 to 6).</summary>
    public int Revision { get; init; }

    /// <summary>The file key's length in bits (40 to 128 for RC4 and AES-128, 256 for AES-256).</summary>
    public int KeyBits { get; init; }

    /// <summary>How strings are encrypted: "RC4", "AES-128", "AES-256", or "none".</summary>
    public string Strings { get; init; } = "";

    /// <summary>How streams are encrypted: "RC4", "AES-128", "AES-256", or "none".</summary>
    public string Streams { get; init; } = "";

    /// <summary>How attached files are encrypted: "RC4", "AES-128", "AES-256", or "none".</summary>
    public string EmbeddedFiles { get; init; } = "";

    /// <summary>What a user who opens the file with the user password may do, as the file states it (a reader enforces none of it).</summary>
    public DocumentPermissions Permissions { get; init; }

    /// <summary>Whether the document's metadata is encrypted with the rest (false: left plain for search tools).</summary>
    public bool MetadataEncrypted { get; init; } = true;

    /// <summary>
    /// Whether the owner password opened the document (<see cref="DocumentOptions.Password"/>), which alone may change or
    /// remove its encryption (<see cref="IDocumentEditor.Encrypt"/>); false when the user password (or the empty one)
    /// opened it. A property of how it was opened, not of the file.
    /// </summary>
    public bool OwnerAccess { get; init; }
}
