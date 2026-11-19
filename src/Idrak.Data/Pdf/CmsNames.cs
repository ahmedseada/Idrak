// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Globalization;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>The forms a GeneralName takes (RFC 5280, 4.2.1.6), by their context tags.</summary>
internal enum CmsNameForm : byte
{
    OtherName,
    Rfc822,
    Dns,
    X400,
    Directory,
    EdiParty,
    Uri,
    IpAddress,
    RegisteredId,
}

/// <summary>
/// A GeneralName read on its DER (RFC 5280, 4.2.1.6): its form and its value: the IA5 text's bytes for rfc822Name, dNSName
/// and uniformResourceIdentifier; the address's bytes for iPAddress (with its mask's, in a constraint); the Name's encoding
/// for directoryName; the whole element for the forms not compared here (otherName, x400Address, ediPartyName,
/// registeredID).
/// </summary>
internal readonly struct CmsGeneralName
{
    private readonly string? _text;

    public CmsGeneralName(CmsNameForm form, ReadOnlyMemory<byte> value)
    {
        (Form, Value) = (form, value);
        _text = form is CmsNameForm.Rfc822 or CmsNameForm.Dns or CmsNameForm.Uri ? Encoding.Latin1.GetString(value.Span) : null;
    }

    public CmsNameForm Form { get; }

    public ReadOnlyMemory<byte> Value { get; }

    /// <summary>The value as text, decoded once (the IA5 forms; a byte above 127 makes it a name that is not valid; empty for the others).</summary>
    public string Text => _text ?? "";

    /// <summary>Whether the form is one a name constraint is evaluated for here (rfc822Name, dNSName, directoryName, URI, iPAddress).</summary>
    public bool Compared => Form is CmsNameForm.Rfc822 or CmsNameForm.Dns or CmsNameForm.Directory or CmsNameForm.Uri or CmsNameForm.IpAddress;

    public string FormName => FormNameOf(Form);

    public static string FormNameOf(CmsNameForm form) => form switch
    {
        CmsNameForm.OtherName => "otherName",
        CmsNameForm.Rfc822 => "rfc822Name",
        CmsNameForm.Dns => "dNSName",
        CmsNameForm.X400 => "x400Address",
        CmsNameForm.Directory => "directoryName",
        CmsNameForm.EdiParty => "ediPartyName",
        CmsNameForm.Uri => "uniformResourceIdentifier",
        CmsNameForm.IpAddress => "iPAddress",
        _ => "registeredID",
    };

    /// <summary>The name for messages: its form and its value.</summary>
    public override string ToString() => Form switch
    {
        CmsNameForm.Rfc822 or CmsNameForm.Dns or CmsNameForm.Uri => $"{FormName} '{Text}'",
        CmsNameForm.Directory => $"{FormName} '{CmsName.Describe(Value)}'",
        CmsNameForm.IpAddress => $"{FormName} {CmsNameRules.Address(Value.Span)}",
        _ => $"{FormName} name",
    };

    /// <summary>The GeneralName at the reader's position.</summary>
    /// <exception cref="InvalidDataException">The element is not a GeneralName.</exception>
    public static CmsGeneralName Read(AsnReader reader)
    {
        var tag = reader.PeekTag();
        if (tag.TagClass != TagClass.ContextSpecific || tag.TagValue is < 0 or > 8)
        {
            throw new InvalidDataException($"a general name tagged [{tag.TagValue}] of the class {tag.TagClass}");
        }

        var form = (CmsNameForm)tag.TagValue;
        switch (form)
        {
            case CmsNameForm.Rfc822 or CmsNameForm.Dns or CmsNameForm.Uri or CmsNameForm.IpAddress:
                return new CmsGeneralName(form, reader.ReadOctetString(new Asn1Tag(TagClass.ContextSpecific, tag.TagValue)));
            case CmsNameForm.Directory:
                var inner = reader.ReadSequence(new Asn1Tag(TagClass.ContextSpecific, 4, isConstructed: true));
                var name = inner.ReadEncodedValue();
                inner.ThrowIfNotEmpty();
                return new CmsGeneralName(form, name);
            default:
                return new CmsGeneralName(form, reader.ReadEncodedValue());
        }
    }

    /// <summary>Every GeneralName a GeneralNames sequence holds.</summary>
    public static List<CmsGeneralName> ReadAll(AsnReader names)
    {
        var list = new List<CmsGeneralName>();
        while (names.HasData)
        {
            list.Add(Read(names));
        }

        return list;
    }
}

/// <summary>An attribute of a distinguished name: its type, its value as text (the string types; null for others), and its value's encoding.</summary>
internal readonly struct CmsNameAttribute(string type, string? text, ReadOnlyMemory<byte> value)
{
    public string Type { get; } = type;

    public string? Text { get; } = text;

    public ReadOnlyMemory<byte> Value { get; } = value;

    /// <summary>
    /// Whether <paramref name="other"/> is the same attribute: the same type, and strings equal with insignificant spaces and
    /// case set aside (RFC 5280 7.1's comparison on RFC 4518's spaces, case folded by the Unicode simple mapping the base
    /// library's ordinal comparison holds on every platform), other values equal byte for byte.
    /// </summary>
    public bool Matches(CmsNameAttribute other) =>
        string.Equals(Type, other.Type, StringComparison.Ordinal)
        && (Text is { } a && other.Text is { } b ? CmsNameRules.SameText(a, b) : Value.Span.SequenceEqual(other.Value.Span));
}

/// <summary>Distinguished names read for comparison (RFC 5280, 7.1): their relative names in order, each its attributes.</summary>
internal static class CmsDn
{
    /// <summary>The name's relative distinguished names, each its attributes; null when it cannot be read.</summary>
    public static CmsNameAttribute[][]? Read(ReadOnlyMemory<byte> name)
    {
        try
        {
            var rdns = new List<CmsNameAttribute[]>();
            var sequence = CmsAsn.Reader(name).ReadSequence();
            while (sequence.HasData)
            {
                var attributes = Attributes(sequence.ReadSetOf(skipSortOrderValidation: true));
                if (attributes.Length == 0)
                {
                    return null;
                }

                rdns.Add(attributes);
            }

            return [.. rdns];
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return null;
        }
    }

    /// <summary>A relative name's attributes, read from its SET OF AttributeTypeAndValue.</summary>
    /// <exception cref="AsnContentException">An attribute that cannot be read.</exception>
    public static CmsNameAttribute[] Attributes(AsnReader set)
    {
        var attributes = new List<CmsNameAttribute>(1);
        while (set.HasData)
        {
            var attribute = set.ReadSequence();
            string type = attribute.ReadObjectIdentifier();
            var value = attribute.PeekEncodedValue();
            attributes.Add(new CmsNameAttribute(type, Text(attribute), value));
        }

        return [.. attributes];
    }

    /// <summary>Whether <paramref name="name"/> is in the subtree <paramref name="constraint"/> names: its relative names begin with the constraint's.</summary>
    public static bool Within(CmsNameAttribute[][] name, CmsNameAttribute[][] constraint)
    {
        if (name.Length < constraint.Length)
        {
            return false;
        }

        for (int i = 0; i < constraint.Length; i++)
        {
            if (!SameRdn(name[i], constraint[i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Whether two names are the same name (every relative name the same).</summary>
    public static bool Same(CmsNameAttribute[][] a, CmsNameAttribute[][] b) => a.Length == b.Length && Within(a, b);

    // A relative name's attributes are a set: each of one has its match in the other.
    private static bool SameRdn(CmsNameAttribute[] a, CmsNameAttribute[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        foreach (var wanted in b)
        {
            bool found = false;
            foreach (var attribute in a)
            {
                if (attribute.Matches(wanted))
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return false;
            }
        }

        return true;
    }

    // A string value's text, or null (the value read as it is) for any other type or a string that does not decode.
    private static string? Text(AsnReader attribute)
    {
        var tag = attribute.PeekTag();
        if (tag.TagClass == TagClass.Universal && (UniversalTagNumber)tag.TagValue is UniversalTagNumber.UTF8String or UniversalTagNumber.PrintableString
                or UniversalTagNumber.IA5String or UniversalTagNumber.BMPString or UniversalTagNumber.NumericString or UniversalTagNumber.VisibleString
                or UniversalTagNumber.T61String or UniversalTagNumber.UniversalString)
        {
            try
            {
                return attribute.ReadCharacterString((UniversalTagNumber)tag.TagValue);
            }
            catch (Exception e) when (CmsAsn.IsDamage(e))
            {
                // A string that does not decode is compared by its bytes.
            }
        }

        attribute.ReadEncodedValue();
        return null;
    }
}

/// <summary>A certificate's name constraints (RFC 5280, 4.2.1.10): its permitted and excluded subtrees (null: absent).</summary>
internal sealed class CmsNameConstraints
{
    private CmsNameConstraints(IReadOnlyList<CmsSubtree>? permitted, IReadOnlyList<CmsSubtree>? excluded) => (Permitted, Excluded) = (permitted, excluded);

    public IReadOnlyList<CmsSubtree>? Permitted { get; }

    public IReadOnlyList<CmsSubtree>? Excluded { get; }

    /// <summary>The constraints in all (permitted and excluded).</summary>
    public int Count => (Permitted?.Count ?? 0) + (Excluded?.Count ?? 0);

    /// <summary>
    /// The constraints the extension's value holds, and why they cannot be used (null when they can): an empty list, a
    /// subtree with a minimum or a maximum, a base that is not a valid name of its form (a wildcard or a leading period in a
    /// dNSName, an address without its mask, a malformed mailbox).
    /// </summary>
    public static CmsNameConstraints Read(ReadOnlyMemory<byte> value, out string? problem)
    {
        problem = null;
        var sequence = CmsAsn.Reader(value).ReadSequence();
        IReadOnlyList<CmsSubtree>? permitted = null, excluded = null;
        if (CmsAsn.Next(sequence, CmsAsn.Context0))
        {
            permitted = Subtrees(sequence.ReadSequence(CmsAsn.Context0), "permitted", ref problem);
        }

        if (CmsAsn.Next(sequence, CmsAsn.Context1))
        {
            excluded = Subtrees(sequence.ReadSequence(CmsAsn.Context1), "excluded", ref problem);
        }

        if (sequence.HasData)
        {
            problem ??= "name constraints with more than their permitted and excluded subtrees";
        }

        if (permitted is null && excluded is null)
        {
            problem ??= "name constraints naming no subtree (RFC 5280 4.2.1.10: permitted, excluded or both)";
        }

        return new CmsNameConstraints(permitted, excluded);
    }

    private static List<CmsSubtree> Subtrees(AsnReader reader, string which, ref string? problem)
    {
        var subtrees = new List<CmsSubtree>();
        var minimum = new Asn1Tag(TagClass.ContextSpecific, 0);
        var maximum = new Asn1Tag(TagClass.ContextSpecific, 1);
        while (reader.HasData)
        {
            var subtree = reader.ReadSequence();
            var name = CmsGeneralName.Read(subtree);
            if (CmsAsn.Next(subtree, minimum) && !subtree.ReadInteger(minimum).IsZero)
            {
                problem ??= $"a {which} name constraint with a minimum other than 0 (RFC 5280 4.2.1.10)";
            }

            if (CmsAsn.Next(subtree, maximum))
            {
                subtree.ReadInteger(maximum);
                problem ??= $"a {which} name constraint with a maximum (RFC 5280 4.2.1.10: none)";
            }

            subtree.ThrowIfNotEmpty();
            if (CmsNameRules.ConstraintProblem(name) is { } why)
            {
                problem ??= $"the {which} name constraint {name}, which {why}";
            }

            subtrees.Add(new CmsSubtree(name, name.Form == CmsNameForm.Directory ? CmsDn.Read(name.Value) : null));
        }

        if (subtrees.Count == 0)
        {
            problem ??= $"an empty list of {which} subtrees in its name constraints (RFC 5280 4.2.1.10: one or more)";
        }

        return subtrees;
    }
}

/// <summary>A name constraint's subtree: its base, and its distinguished name read when it is a directoryName.</summary>
internal readonly record struct CmsSubtree(CmsGeneralName Base, CmsNameAttribute[][]? Dn);

/// <summary>
/// The rules names are checked and compared by (RFC 5280, 4.2.1.6, 4.2.1.10 and 7): DNS names in their preferred syntax,
/// compared case-insensitively in ASCII (RFC 4343); mailboxes by their local part exactly and their host case-insensitively;
/// a URI by its host; addresses under a mask.
/// </summary>
internal static class CmsNameRules
{
    /// <summary>
    /// Whether <paramref name="name"/> is a DNS name in its preferred syntax: labels of letters, digits, hyphens and
    /// underscores joined by periods, each 1 to 63 long, 253 at most; a constraint may be empty (every name), a name may
    /// start with the wildcard label "*" when <paramref name="wildcard"/>.
    /// </summary>
    public static bool Dns(ReadOnlySpan<char> name, bool constraint, bool wildcard)
    {
        if (name.IsEmpty)
        {
            return constraint;
        }

        if (name.Length > 253)
        {
            return false;
        }

        if (wildcard && name.StartsWith("*.", StringComparison.Ordinal))
        {
            name = name[2..];
        }

        foreach (var range in name.Split('.'))
        {
            var label = name[range];
            if (label.Length is 0 or > 63)
            {
                return false;
            }

            foreach (char c in label)
            {
                if (!char.IsAsciiLetterOrDigit(c) && c is not ('-' or '_'))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>Why <paramref name="text"/> is not a valid mailbox (a constraint: a mailbox, a host, or a domain with a leading period), or null.</summary>
    public static string? EmailProblem(string text, bool constraint)
    {
        if (!Ascii(text))
        {
            return "holds a character outside ASCII";
        }

        int at = text.IndexOf('@');
        if (at < 0)
        {
            if (!constraint)
            {
                return "is not a mailbox (local-part@host)";
            }

            var host = text.AsSpan(text.StartsWith('.') ? 1 : 0);
            return !host.IsEmpty && Dns(host, constraint: false, wildcard: false) ? null : "is neither a mailbox, a host nor a domain";
        }

        return at > 0 && text.IndexOf('@', at + 1) < 0 && Dns(text.AsSpan(at + 1), constraint: false, wildcard: false) ? null : "is not a mailbox (local-part@host)";
    }

    /// <summary>Why a constraint's base is not a valid name of its form, or null (the forms not compared here are taken as they are).</summary>
    public static string? ConstraintProblem(CmsGeneralName name)
    {
        switch (name.Form)
        {
            case CmsNameForm.Rfc822:
                return EmailProblem(name.Text, constraint: true);
            case CmsNameForm.Dns:
                string dns = name.Text;
                return Ascii(dns) && Dns(dns, constraint: true, wildcard: false) ? null : "is not a DNS name in its preferred syntax (no wildcard, no leading period)";
            case CmsNameForm.Uri:
                string uri = name.Text;
                var host = uri.AsSpan(uri.StartsWith('.') ? 1 : 0);
                return Ascii(uri) && !host.IsEmpty && Dns(host, constraint: false, wildcard: false) ? null : "is neither a host nor a domain with a leading period";
            case CmsNameForm.IpAddress:
                return IpConstraint(name.Value.Span) ? null : "is not an address and its mask (8 or 32 bytes, the mask's ones first)";
            case CmsNameForm.Directory:
                return CmsDn.Read(name.Value) is null ? "cannot be read" : null;
            default:
                return null;
        }
    }

    /// <summary>Why a name of a certificate is not a valid name of its form (so a constraint cannot be checked on it), or null.</summary>
    public static string? NameProblem(CmsGeneralName name, CmsNameAttribute[][]? dn)
    {
        switch (name.Form)
        {
            case CmsNameForm.Rfc822:
                return EmailProblem(name.Text, constraint: false);
            case CmsNameForm.Dns:
                string dns = name.Text;
                return Ascii(dns) && !string.IsNullOrEmpty(dns) && Dns(dns, constraint: false, wildcard: true) ? null : "is not a DNS name in its preferred syntax";
            case CmsNameForm.Uri:
                return Ascii(name.Text) && UriHost(name.Text) is not null ? null : "names no host by its domain name (RFC 5280 4.2.1.10: a URI's constraint applies to its host)";
            case CmsNameForm.IpAddress:
                return name.Value.Length is 4 or 16 ? null : "is not an IPv4 or IPv6 address (4 or 16 bytes)";
            case CmsNameForm.Directory:
                return dn is null ? "cannot be read" : null;
            default:
                return null;
        }
    }

    /// <summary>The host of a URI with an authority (scheme://[userinfo@]host[:port]...), or null when it has none, or names an IP address.</summary>
    public static string? UriHost(string uri)
    {
        int scheme = uri.IndexOf(':');
        if (scheme <= 0 || !uri.AsSpan(scheme + 1).StartsWith("//", StringComparison.Ordinal))
        {
            return null;
        }

        var authority = uri.AsSpan(scheme + 3);
        int end = authority.IndexOfAny('/', '?', '#');
        if (end >= 0)
        {
            authority = authority[..end];
        }

        int user = authority.LastIndexOf('@');
        if (user >= 0)
        {
            authority = authority[(user + 1)..];
        }

        if (authority.StartsWith('['))
        {
            return null;                                                       // an IPv6 literal
        }

        int port = authority.LastIndexOf(':');
        if (port >= 0)
        {
            authority = authority[..port];
        }

        if (authority.IsEmpty || !Dns(authority, constraint: false, wildcard: false))
        {
            return null;
        }

        foreach (char c in authority)
        {
            if (!char.IsAsciiDigit(c) && c != '.')
            {
                return authority.ToString();
            }
        }

        return null;                                                           // an IPv4 literal
    }

    /// <summary>Whether a DNS name is in a dNSName constraint's subtree: the name itself, or with labels added on its left (every name for an empty one).</summary>
    public static bool DnsWithin(ReadOnlySpan<char> name, ReadOnlySpan<char> constraint) =>
        constraint.IsEmpty || name.Equals(constraint, StringComparison.OrdinalIgnoreCase)
        || (name.Length > constraint.Length && name[^(constraint.Length + 1)] == '.' && name.EndsWith(constraint, StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Whether a wildcard name (*.base) may name a host in a dNSName constraint's subtree: its base is in the subtree (so all
    /// its names are), or the constraint is one label below its base (the one name the wildcard stands for). A permitted
    /// subtree holds a wildcard name only when its base is in it; an excluded one excludes it when it may name a host there.
    /// </summary>
    public static bool WildcardMayMatch(ReadOnlySpan<char> wildcardBase, ReadOnlySpan<char> constraint) =>
        DnsWithin(wildcardBase, constraint)
        || (constraint.Length > wildcardBase.Length + 1 && constraint[^(wildcardBase.Length + 1)] == '.' && constraint.EndsWith(wildcardBase, StringComparison.OrdinalIgnoreCase)
            && constraint[..^(wildcardBase.Length + 1)].IndexOf('.') < 0);

    /// <summary>Whether a host is in a host constraint's subtree: ".domain" its hosts below it (not itself), "host" itself.</summary>
    public static bool HostWithin(ReadOnlySpan<char> host, ReadOnlySpan<char> constraint) =>
        constraint.StartsWith('.')
            ? host.Length > constraint.Length && host.EndsWith(constraint, StringComparison.OrdinalIgnoreCase)
            : host.Equals(constraint, StringComparison.OrdinalIgnoreCase);

    /// <summary>Whether a mailbox is in an rfc822Name constraint's subtree: a mailbox itself (its local part exactly), a host's mailboxes, or a domain's hosts'.</summary>
    public static bool EmailWithin(string mailbox, string constraint)
    {
        int at = mailbox.IndexOf('@');
        var local = mailbox.AsSpan(0, at);
        var host = mailbox.AsSpan(at + 1);
        int bound = constraint.IndexOf('@');
        return bound >= 0
            ? local.SequenceEqual(constraint.AsSpan(0, bound)) && host.Equals(constraint.AsSpan(bound + 1), StringComparison.OrdinalIgnoreCase)
            : HostWithin(host, constraint);
    }

    /// <summary>Whether a constraint's value is an address and a mask (8 bytes for IPv4, 32 for IPv6) whose ones come before its zeros.</summary>
    public static bool IpConstraint(ReadOnlySpan<byte> value)
    {
        if (value.Length is not (8 or 32))
        {
            return false;
        }

        bool zero = false;
        foreach (byte b in value[(value.Length / 2)..])
        {
            for (int bit = 7; bit >= 0; bit--)
            {
                bool one = ((b >> bit) & 1) != 0;
                if (one && zero)
                {
                    return false;
                }

                zero |= !one;
            }
        }

        return true;
    }

    /// <summary>Whether an address (4 or 16 bytes) is under a constraint's address and mask of its own family.</summary>
    public static bool IpWithin(ReadOnlySpan<byte> address, ReadOnlySpan<byte> constraint)
    {
        if (address.Length * 2 != constraint.Length)
        {
            return false;
        }

        for (int i = 0; i < address.Length; i++)
        {
            int mask = constraint[address.Length + i];
            if ((address[i] & mask) != (constraint[i] & mask))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>An address for messages (192.0.2.1; 2001:db8:0:0:0:0:0:1), a constraint's with its prefix length (192.0.2.0/24).</summary>
    public static string Address(ReadOnlySpan<byte> value)
    {
        int length = value.Length is 8 or 32 ? value.Length / 2 : value.Length;
        var text = new StringBuilder();
        if (length == 4)
        {
            for (int i = 0; i < 4; i++)
            {
                text.Append(i == 0 ? "" : ".").Append(value[i].ToString(CultureInfo.InvariantCulture));
            }
        }
        else if (length == 16)
        {
            for (int i = 0; i < 16; i += 2)
            {
                text.Append(i == 0 ? "" : ":").Append(((value[i] << 8) | value[i + 1]).ToString("x", CultureInfo.InvariantCulture));
            }
        }
        else
        {
            return $"of {value.Length} bytes ({Convert.ToHexString(value)})";
        }

        if (value.Length == length * 2)
        {
            int prefix = 0;
            foreach (byte b in value[length..])
            {
                prefix += System.Numerics.BitOperations.PopCount(b);
            }

            text.Append('/').Append(prefix.ToString(CultureInfo.InvariantCulture));
        }

        return text.ToString();
    }

    /// <summary>Whether two directory strings are the same with insignificant spaces and case set aside (see <see cref="CmsNameAttribute.Matches"/>).</summary>
    public static bool SameText(string a, string b) =>
        string.Equals(a, b, StringComparison.Ordinal) || string.Equals(Spaced(a), Spaced(b), StringComparison.OrdinalIgnoreCase);

    private static bool Ascii(string text)
    {
        foreach (char c in text)
        {
            if (c > 127)
            {
                return false;
            }
        }

        return true;
    }

    // RFC 4518's insignificant spaces: none at either end, a run of white space inside as one space.
    private static string Spaced(string text)
    {
        var spaced = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                space = spaced.Length > 0;
                continue;
            }

            if (space)
            {
                spaced.Append(' ');
                space = false;
            }

            spaced.Append(c);
        }

        return spaced.ToString();
    }
}

/// <summary>
/// The name constraints of a certification path evaluated (RFC 5280, 4.2.1.10, 6.1.3 (b) and (c), 6.1.4 (g)): every CA's
/// constraints, and the trusted certificate's (as most validators and x509-limbo read them), over the subject and every
/// alternative name of each certificate below it (a self-issued one only when it is the first), whether the extension is
/// marked critical or not: a name of a form a CA permits must be in one of its permitted subtrees of that form, and none may
/// be in an excluded one. rfc822Name, dNSName (a wildcard name refused when it may name an excluded host), directoryName,
/// uniformResourceIdentifier (its host) and iPAddress (IPv4 and IPv6, under a mask) are evaluated; a constraint of another
/// form fails the path, named, when a certificate below holds a name of that form. The comparisons a path asks for are
/// counted before any is made, against <see cref="SignatureCheckOptions.MaxNameConstraintChecks"/>.
/// </summary>
internal static class CmsNameCheck
{
    /// <summary>Why the path's names break its name constraints, or null when they hold (or there are none).</summary>
    public static string? Check(List<CmsCertificate> path, int maxChecks)
    {
        bool any = false;
        foreach (var certificate in path)
        {
            if (certificate.NameConstraints is not null)
            {
                if (certificate.IsCa != true)
                {
                    return $"{certificate.Named}'s certificate has name constraints but is not a CA (RFC 5280 4.2.1.10: only a CA's certificate holds them)";
                }

                any = true;
            }
        }

        if (!any)
        {
            return null;
        }

        // Each constrained certificate's names, read once, and the comparisons they ask for, before any is made.
        var names = new List<(CmsGeneralName Name, CmsNameAttribute[][]? Dn)>?[path.Count];
        var counts = new int[path.Count];
        long work = 0;
        CmsCertificate? first = null;
        for (int k = path.Count - 1; k >= 1; k--)
        {
            if (path[k].NameConstraints is not { } constraints)
            {
                continue;
            }

            first ??= path[k];
            for (int j = k - 1; j >= 0; j--)
            {
                if (j > 0 && path[j].SelfIssued)
                {
                    continue;
                }

                if (names[j] is null)
                {
                    (names[j], counts[j], string? unreadable) = NamesOf(path[j]);
                    if (unreadable is not null)
                    {
                        return $"{path[j].Named}'s subject alternative names cannot be read ({unreadable}), so {path[k].Named}'s name constraints cannot be checked on them";
                    }
                }

                work += (long)constraints.Count * counts[j];
            }
        }

        if (work > maxChecks)
        {
            return string.Create(CultureInfo.InvariantCulture,
                $"the path's name constraints ({first!.Named}'s first) ask for {work} name comparisons, more than the {maxChecks} a check makes (SignatureCheckOptions.MaxNameConstraintChecks, {SignatureCheckOptions.MaxNameConstraintChecksVariable})");
        }

        for (int k = path.Count - 1; k >= 1; k--)
        {
            if (path[k].NameConstraints is not { } constraints)
            {
                continue;
            }

            for (int j = k - 1; j >= 0; j--)
            {
                if (j > 0 && path[j].SelfIssued)
                {
                    continue;
                }

                foreach (var (name, dn) in names[j]!)
                {
                    if (Evaluate(path[k], constraints, path[j], name, dn) is { } broken)
                    {
                        return broken;
                    }
                }
            }
        }

        return null;
    }

    // A certificate's names: its subject (when not empty), the e-mail addresses in its subject when it has no alternative
    // names (RFC 5280 4.2.1.10's legacy rule), and its alternative names; how many names it has (its subject's attributes
    // and its alternative names); or why its alternative names cannot be read.
    private static (List<(CmsGeneralName, CmsNameAttribute[][]?)> Names, int Count, string? Problem) NamesOf(CmsCertificate certificate)
    {
        var names = new List<(CmsGeneralName, CmsNameAttribute[][]?)>();
        int count = 0;
        var subject = CmsDn.Read(certificate.Subject);
        if (subject is null)
        {
            names.Add((new CmsGeneralName(CmsNameForm.Directory, certificate.Subject), null));
            count++;
        }
        else if (subject.Length > 0)
        {
            names.Add((new CmsGeneralName(CmsNameForm.Directory, certificate.Subject), subject));
            foreach (var rdn in subject)
            {
                count += rdn.Length;
                if (certificate.AltNames is null)
                {
                    foreach (var attribute in rdn)
                    {
                        if (attribute.Type == CmsOids.EmailAddress && attribute.Text is { } mailbox)
                        {
                            names.Add((new CmsGeneralName(CmsNameForm.Rfc822, Encoding.Latin1.GetBytes(mailbox)), null));
                        }
                    }
                }
            }
        }

        if (certificate.AltNamesProblem is { } problem)
        {
            return (names, count, problem);
        }

        foreach (var name in certificate.AltNames ?? [])
        {
            names.Add((name, name.Form == CmsNameForm.Directory ? CmsDn.Read(name.Value) : null));
            count++;
        }

        return (names, count, null);
    }

    // One name of a certificate against one CA's constraints: null when they hold.
    private static string? Evaluate(CmsCertificate ca, CmsNameConstraints constraints, CmsCertificate certificate, CmsGeneralName name, CmsNameAttribute[][]? dn)
    {
        bool permittedForm = false, excludedForm = false;
        foreach (var subtree in constraints.Permitted ?? [])
        {
            permittedForm |= subtree.Base.Form == name.Form;
        }

        foreach (var subtree in constraints.Excluded ?? [])
        {
            excludedForm |= subtree.Base.Form == name.Form;
        }

        if (!permittedForm && !excludedForm)
        {
            return null;                                                       // the form is not constrained
        }

        if (!name.Compared)
        {
            return $"{ca.Named}'s name constraints limit {name.FormName} names, which this check does not evaluate, and {certificate.Named}'s certificate has one";
        }

        if (CmsNameRules.NameProblem(name, dn) is { } invalid)
        {
            return $"{certificate.Named}'s {name} {invalid}, so {ca.Named}'s name constraints cannot be checked on it";
        }

        if (permittedForm)
        {
            bool within = false;
            foreach (var subtree in constraints.Permitted!)
            {
                if (subtree.Base.Form == name.Form && Within(name, dn, subtree, excluded: false))
                {
                    within = true;
                    break;
                }
            }

            if (!within)
            {
                return $"{certificate.Named}'s {name} is outside the names {ca.Named}'s name constraints permit";
            }
        }

        foreach (var subtree in constraints.Excluded ?? [])
        {
            if (subtree.Base.Form == name.Form && Within(name, dn, subtree, excluded: true))
            {
                return $"{certificate.Named}'s {name} is in {subtree.Base}, which {ca.Named}'s name constraints exclude";
            }
        }

        return null;
    }

    // Whether a valid name is in a subtree of its form (an excluded subtree takes a wildcard name that may name a host in it).
    private static bool Within(CmsGeneralName name, CmsNameAttribute[][]? dn, CmsSubtree subtree, bool excluded)
    {
        switch (name.Form)
        {
            case CmsNameForm.Rfc822:
                return CmsNameRules.EmailWithin(name.Text, subtree.Base.Text);
            case CmsNameForm.Dns:
                string dns = name.Text, constraint = subtree.Base.Text;
                if (dns.StartsWith("*.", StringComparison.Ordinal))
                {
                    return excluded ? CmsNameRules.WildcardMayMatch(dns.AsSpan(2), constraint) : CmsNameRules.DnsWithin(dns.AsSpan(2), constraint);
                }

                return CmsNameRules.DnsWithin(dns, constraint);
            case CmsNameForm.Uri:
                return CmsNameRules.UriHost(name.Text) is { } host && CmsNameRules.HostWithin(host, subtree.Base.Text);
            case CmsNameForm.IpAddress:
                return CmsNameRules.IpWithin(name.Value.Span, subtree.Base.Value.Span);
            case CmsNameForm.Directory:
                return dn is not null && subtree.Dn is { } within && CmsDn.Within(dn, within);
            default:
                return false;
        }
    }
}
