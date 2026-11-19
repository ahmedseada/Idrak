// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using Idrak.Abstraction.Text;
using PaddingMode = System.Security.Cryptography.PaddingMode;

namespace Idrak.Data.Pdf;

/// <summary>
/// The standard security handler (ISO 32000-2, 7.6.4), reading side: revisions 2-4 (RC4 of 40 to 128 bits, and AES-128
/// through crypt filters: <c>/CF</c>, <c>/StmF</c>, <c>/StrF</c>, <c>/EFF</c>, <c>/Identity</c>, <c>/EncryptMetadata</c>)
/// and revisions 5-6 (AES-256; revision 6's hash with SHA-256, -384 and -512, passwords prepared by SASLprep). Opening
/// tries the empty user password first, then <see cref="DocumentOptions.Password"/> as the user and as the owner
/// password; the file key is found once, kept while the document is open and cleared when it is disposed. Strings are
/// decrypted when their object is read at its offset (those inside object streams come decrypted with the stream; the
/// encryption dictionary's, a cross-reference stream's and a signature's contents are not encrypted), streams before
/// their filters (not a cross-reference stream, nor metadata under <c>/EncryptMetadata false</c>, nor what a crypt filter
/// of the stream's own or <c>/Identity</c> leaves plain). Safe from several threads.
/// </summary>
internal sealed partial class PdfSecurity : IDisposable
{
    // A password longer than this many characters is cut before it is prepared (revisions 5 and 6 keep 127 bytes of it).
    private const int MostPasswordChars = 1024;

    private readonly string _name;
    private readonly int _encryptNumber;
    private readonly int _keyLength;
    private readonly byte[] _o, _u, _oe, _ue, _id;
    private readonly int _permissions;
    private readonly bool _encryptMetadata;
    private readonly (PdfName Name, PdfCryptMethod Method, string Cfm)[] _filters;
    private readonly PdfCryptMethod _strings, _streams, _embedded;

    // AES ciphers to reuse, each used by one thread at a time: keyed with the file key (AES-256), or keyed per object.
    private readonly ConcurrentBag<Aes> _fileKeyed = new(), _objectKeyed = new();
    private byte[] _key = [];
    private volatile bool _disposed;

    private PdfSecurity(PdfFile file, PdfDictionary dictionary, int encryptNumber)
    {
        _name = file.Name;
        _encryptNumber = encryptNumber;
        long v = Integer(file, dictionary, "V"u8) ?? 0;
        Revision = (int)Math.Clamp(Integer(file, dictionary, "R"u8) ?? throw Damaged("has no /R (its revision)"), -1, 99);
        if (Revision is < 2 or > 6)
        {
            throw new NotSupportedException(
                $"{_name}: the file is encrypted by revision {Revision} of the standard security handler, which the standard does not define (revisions 2 to 6 are read).");
        }

        if (v is not (1 or 2 or 4 or 5))
        {
            throw new NotSupportedException(
                $"{_name}: the file is encrypted by the standard security handler with the algorithm /V {v}, which is not read (1, 2, 4 and 5 are; 3 was never published).");
        }

        _o = Bytes(file, dictionary, "O"u8) ?? throw Damaged("has no /O (the owner key)");
        _u = Bytes(file, dictionary, "U"u8) ?? throw Damaged("has no /U (the user key)");
        int keys = Revision >= 5 ? 48 : 32;
        if (_o.Length < keys || _u.Length < keys)
        {
            throw Damaged($"has an /O of {_o.Length} bytes and a /U of {_u.Length}; revision {Revision} writes {keys} each");
        }

        _oe = Revision >= 5 ? Bytes(file, dictionary, "OE"u8) ?? [] : [];
        _ue = Revision >= 5 ? Bytes(file, dictionary, "UE"u8) ?? [] : [];
        if (Revision >= 5 && (_oe.Length < 32 || _ue.Length < 32))
        {
            throw Damaged($"has an /OE of {_oe.Length} bytes and a /UE of {_ue.Length}; revision {Revision} writes 32 each");
        }

        _permissions = unchecked((int)(Integer(file, dictionary, "P"u8) ?? throw Damaged("has no /P (the permissions)")));
        _encryptMetadata = file.Resolve(dictionary["EncryptMetadata"u8]) is not { Kind: PdfKind.Boolean } metadata || metadata.AsBoolean;
        _id = file.Resolve(file.Trailer["ID"u8]).AsArray is { Count: > 0 } ids && file.Resolve(ids[0]).AsString is { } first ? first.Bytes : [];

        // How strings and streams are encrypted: crypt filters from /V 4, RC4 before.
        long? filterBits = null;
        if (v >= 4)
        {
            var list = new List<(PdfName, PdfCryptMethod, string)>();
            var lengths = new List<long?>();
            if (file.Resolve(dictionary["CF"u8]).AsDictionary is { } filters)
            {
                for (int i = 0; i < filters.Count; i++)
                {
                    var filter = file.Resolve(filters.ValueAt(i)).AsDictionary;
                    string cfm = filter is null ? "None" : file.Resolve(filter["CFM"u8]).AsName?.ToString() ?? "None";
                    list.Add((filters.KeyAt(i), Method(cfm), cfm));
                    lengths.Add(filter is not null && file.Resolve(filter["Length"u8]).TryInteger(out long length) ? length : null);
                }
            }

            _filters = [.. list];
            PdfName? stmF = file.Resolve(dictionary["StmF"u8]).AsName, strF = file.Resolve(dictionary["StrF"u8]).AsName, eff = file.Resolve(dictionary["EFF"u8]).AsName;
            _streams = Named(stmF);
            _strings = Named(strF);
            _embedded = eff is null ? _streams : Named(eff);
            foreach (var (key, name, method) in new[] { ("StmF", stmF, _streams), ("StrF", strF, _strings), ("EFF", eff ?? stmF, _embedded) })
            {
                if (method == PdfCryptMethod.Unknown)
                {
                    throw new NotSupportedException(
                        $"{_name}: the file is encrypted with the crypt filter /{name} (its /{key}), whose method /{_filters[IndexOf(name!)].Cfm} the standard handler does not define (None, V2, AESV2 and AESV3 are read).");
                }
            }

            // The key's length: the stream filter's /Length (in bytes, or in bits as some writers put it), else the string filter's.
            foreach (var used in new[] { stmF, strF })
            {
                if (filterBits is null && used is not null && IndexOf(used) is int at and >= 0 && lengths[at] is long length)
                {
                    filterBits = length < 40 ? length * 8 : length;
                }
            }
        }
        else
        {
            _filters = [];
            _streams = _strings = _embedded = PdfCryptMethod.Rc4;
        }

        if (Revision >= 5)
        {
            _keyLength = 32;
        }
        else if (Revision == 2)
        {
            _keyLength = 5;
        }
        else
        {
            long bits = filterBits ?? Integer(file, dictionary, "Length"u8) ?? (v >= 4 ? 128 : 40);
            if (bits is < 40 or > 128 || bits % 8 != 0)
            {
                throw Damaged($"states a key of {bits} bits (/Length); revision {Revision} keys are 40 to 128 bits, in whole bytes");
            }

            _keyLength = (int)(bits / 8);
        }
    }

    /// <summary>The standard handler's revision (2 to 6).</summary>
    public int Revision { get; }

    /// <summary>Whether the document was opened by its owner password (else by its user password).</summary>
    public bool Owner { get; private set; }

    /// <summary>How strings are decrypted (<c>/StrF</c>).</summary>
    public PdfCryptMethod Strings => _strings;

    /// <summary>How streams are decrypted (<c>/StmF</c>).</summary>
    public PdfCryptMethod Streams => _streams;

    /// <summary>Whether the file key is cleared (the document disposed): the tests check that nothing keeps it.</summary>
    public bool Cleared => _key.AsSpan().IndexOfAnyExcept((byte)0) < 0;

    /// <summary>
    /// The encryption as <see cref="DocumentInfo.Security"/> gives it: the handler, its revision, the key's length, the
    /// strings', streams' and attached files' ciphers, the permissions and whether metadata is encrypted; nothing random
    /// (the key, <c>/O</c>, <c>/U</c>, <c>/OE</c>, <c>/UE</c>, <c>/Perms</c>, the file's identifier).
    /// </summary>
    public DocumentSecurity Describe() => new()
    {
        Handler = "Standard",
        Revision = Revision,
        KeyBits = _keyLength * 8,
        Strings = CipherName(_strings),
        Streams = CipherName(_streams),
        EmbeddedFiles = CipherName(_embedded),
        Permissions = Permissions,
        MetadataEncrypted = _encryptMetadata,
    };

    /// <summary>
    /// Unlocks <paramref name="file"/> when its trailer names an encryption dictionary: the standard handler's file key
    /// found from the empty password or <paramref name="password"/>, and every string and stream read from then on
    /// decrypted. Nothing is done for a file that is not encrypted.
    /// </summary>
    /// <exception cref="NotSupportedException">Another security handler (a public-key one), or a variant of the standard one that is not read (each named).</exception>
    /// <exception cref="InvalidDataException">
    /// Neither the empty password nor <paramref name="password"/> opens the file (the message names
    /// <see cref="DocumentOptions.Password"/>), or its encryption dictionary is damaged.
    /// </exception>
    public static void Attach(PdfFile file, string? password)
    {
        var encrypt = file.Trailer["Encrypt"u8];
        if (encrypt.IsNull)
        {
            return;
        }

        PdfDictionary? dictionary;
        try
        {
            dictionary = file.Dictionary(encrypt);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            throw new InvalidDataException($"{file.Name}: the file is encrypted, but its encryption dictionary cannot be read ({e.Message}).", e);
        }

        if (dictionary is null)
        {
            throw new InvalidDataException($"{file.Name}: the file is encrypted, but its /Encrypt ({encrypt}) is not an encryption dictionary.");
        }

        string handler = file.Resolve(dictionary["Filter"u8]).AsName?.ToString() ?? "(none named)";
        if (handler != "Standard")
        {
            throw new NotSupportedException(
                $"{file.Name}: the file is encrypted with the security handler {handler}, which is not read (only the standard, password-based handler is).");
        }

        var security = new PdfSecurity(file, dictionary, encrypt.IsReference ? encrypt.AsReference.Number : -1);
        try
        {
            security.Authenticate(password);
            file.Unlock(security);
            security = null;
        }
        finally
        {
            security?.Dispose();
        }
    }

    /// <summary>
    /// <paramref name="value"/>, object <paramref name="number"/> as read at its offset, with every string in it decrypted
    /// (the same value when it holds none, or is one the standard leaves plain).
    /// </summary>
    /// <exception cref="InvalidDataException">A string cannot be decrypted (an AES key the file's entries make wrong).</exception>
    public PdfValue DecryptStrings(PdfValue value, int number, int generation)
    {
        if (_strings == PdfCryptMethod.Identity || number == _encryptNumber || value.Kind is not (PdfKind.String or PdfKind.Array or PdfKind.Dictionary or PdfKind.Stream))
        {
            return value;
        }

        if (value.AsStream is { } stream && stream.Dictionary["Type"u8].IsName("XRef"u8))
        {
            return value;   // a cross-reference stream's dictionary is not encrypted (7.5.8.2)
        }

        Span<byte> buffer = stackalloc byte[16];
        var key = new StringKey(buffer, number, generation);
        try
        {
            return TryStrings(value, ref key, out var decrypted) ? decrypted : value;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(buffer);
        }
    }

    /// <summary>
    /// The stored bytes of <paramref name="stream"/> (<paramref name="stored"/>, disposed here) decrypted into a pooled
    /// buffer, or <paramref name="stored"/> itself when the stream is not encrypted. A crypt filter of the stream's own,
    /// first in <paramref name="filters"/>, says how, and is taken off the chain. The result is never longer than the
    /// stored bytes, so it stays within <see cref="DocumentOptions.MaxStreamBytes"/> as they do.
    /// </summary>
    /// <exception cref="InvalidDataException">The data cannot be decrypted (an AES key the file's entries make wrong).</exception>
    /// <exception cref="NotSupportedException">The stream's own crypt filter has a method the standard handler does not define.</exception>
    public PdfWindow Decrypt(PdfStream stream, PdfWindow stored, ref PdfFilter[] filters)
    {
        byte[]? rented = null;
        try
        {
            var method = StreamMethod(stream, ref filters);
            if (method == PdfCryptMethod.Identity)
            {
                return stored;
            }

            var data = stored.Span;
            rented = ArrayPool<byte>.Shared.Rent(Math.Max(16, data.Length));
            Span<byte> objectKey = stackalloc byte[16];
            int length;
            try
            {
                int keyLength = ObjectKey(method, stream.Number, stream.Generation, objectKey);
                length = Decrypt(method, objectKey[..keyLength], data, rented);
            }
            finally
            {
                CryptographicOperations.ZeroMemory(objectKey);
            }

            var plain = PdfWindow.Own(rented, length);
            rented = null;
            stored.Dispose();
            return plain;
        }
        catch
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }

            stored.Dispose();
            throw;
        }
    }

    /// <summary>Clears the file key and drops the ciphers keyed with it.</summary>
    public void Dispose()
    {
        _disposed = true;
        CryptographicOperations.ZeroMemory(_key);
        Drain();
    }

    // The file key from the empty password, then from `password` (as the user's, then the owner's); an error naming the
    // option when none opens the file.
    private void Authenticate(string? password)
    {
        byte[] key = new byte[_keyLength];
        using var aes = Revision >= 5 ? Aes.Create() : null;
        bool found = Check([], owner: false, key, aes);
        if (!found && string.IsNullOrEmpty(password))
        {
            found = Check([], owner: true, key, aes);   // an empty owner password
        }
        else if (!found)
        {
            found = CheckGiven(password!, key, aes);
        }

        if (!found)
        {
            CryptographicOperations.ZeroMemory(key);
            string how = $"the standard security handler, revision {Revision}, {Cipher}";
            throw new InvalidDataException(string.IsNullOrEmpty(password)
                ? $"{_name}: the file is encrypted ({how}) and opens only with a password: give its user or owner password in DocumentOptions.Password."
                : $"{_name}: the password given in DocumentOptions.Password is neither the user nor the owner password of the file (encrypted by {how}).");
        }

        _key = key;
    }

    // The password given, as this revision takes it (PDFDocEncoding; SASLprep and UTF-8 from revision 5), then as some
    // writers took it (UTF-8 unprepared) when that differs: each as the user's, then the owner's password.
    private bool CheckGiven(string password, Span<byte> key, Aes? aes)
    {
        var text = password.AsSpan(0, Math.Min(password.Length, MostPasswordChars));
        Span<byte> first = stackalloc byte[PdfCrypt.PasswordBytes];
        Span<byte> second = stackalloc byte[PdfCrypt.PasswordBytes];
        try
        {
            int most = Revision >= 5 ? PdfCrypt.PasswordBytes : PdfCrypt.LegacyPasswordBytes;
            int a = -1;
            if (Revision >= 5)
            {
                char[] prepared = ArrayPool<char>.Shared.Rent(text.Length * UnicodeNormalization.MaxDecompositionLength(compatibility: true) + 1);
                try
                {
                    int n = PdfCrypt.SaslPrep(text, prepared);
                    a = n < 0 ? -1 : PdfCrypt.Utf8Password(prepared.AsSpan(0, n), first, most);
                }
                finally
                {
                    ArrayPool<char>.Shared.Return(prepared, clearArray: true);
                }
            }
            else
            {
                a = PdfCrypt.LegacyPassword(text, first);
            }

            int b = PdfCrypt.Utf8Password(text, second, most);
            if (a >= 0 && (Check(first[..a], owner: false, key, aes) || Check(first[..a], owner: true, key, aes)))
            {
                return true;
            }

            return (a < 0 || !first[..a].SequenceEqual(second[..b])) && (Check(second[..b], owner: false, key, aes) || Check(second[..b], owner: true, key, aes));
        }
        finally
        {
            CryptographicOperations.ZeroMemory(first);
            CryptographicOperations.ZeroMemory(second);
        }
    }

    // Whether `password` is the user's (or the owner's) password; the file key in `key` when it is.
    private bool Check(ReadOnlySpan<byte> password, bool owner, Span<byte> key, Aes? aes)
    {
        bool found;
        if (Revision >= 5)
        {
            found = owner ? CheckOwner(password, key, aes!) : CheckUser(password, key, aes!);
        }
        else
        {
            Span<byte> padded = stackalloc byte[PdfCrypt.LegacyPasswordBytes];
            PdfCrypt.Pad(password, padded);
            found = owner ? CheckLegacyOwner(padded, key) : CheckLegacyUser(padded, key);
            CryptographicOperations.ZeroMemory(padded);
        }

        Owner = found && owner;
        return found;
    }

    // Algorithm 2, then 4 (revision 2) or 5 (3 and 4): the file key from a padded user password, and whether it makes /U.
    private bool CheckLegacyUser(ReadOnlySpan<byte> padded, Span<byte> key)
    {
        int n = _keyLength;
        Span<byte> hash = stackalloc byte[16];
        Span<byte> step = stackalloc byte[16];
        Span<byte> check = stackalloc byte[32];
        Span<byte> permissions = stackalloc byte[4];
        using var md5 = IncrementalHash.CreateHash(HashAlgorithmName.MD5);   // one context for every MD5 of the check
        try
        {
            BinaryPrimitives.WriteInt32LittleEndian(permissions, _permissions);
            md5.AppendData(padded);
            md5.AppendData(_o.AsSpan(0, 32));
            md5.AppendData(permissions);
            md5.AppendData(_id);
            if (Revision >= 4 && !_encryptMetadata)
            {
                md5.AppendData([0xFF, 0xFF, 0xFF, 0xFF]);
            }

            md5.GetHashAndReset(hash);
            if (Revision >= 3)
            {
                for (int i = 0; i < 50; i++)
                {
                    md5.AppendData(hash[..n]);
                    md5.GetHashAndReset(hash);
                }
            }

            hash[..n].CopyTo(key);
            if (Revision == 2)
            {
                PdfCrypt.Rc4(key[..n], PdfCrypt.Padding, check);
                return CryptographicOperations.FixedTimeEquals(check, _u.AsSpan(0, 32));
            }

            md5.AppendData(PdfCrypt.Padding);
            md5.AppendData(_id);
            md5.GetHashAndReset(check[..16]);

            for (int i = 0; i <= 19; i++)
            {
                for (int k = 0; k < n; k++)
                {
                    step[k] = (byte)(key[k] ^ i);
                }

                PdfCrypt.Rc4(step[..n], check[..16], check[..16]);
            }

            return CryptographicOperations.FixedTimeEquals(check[..16], _u.AsSpan(0, 16));
        }
        finally
        {
            CryptographicOperations.ZeroMemory(hash);
            CryptographicOperations.ZeroMemory(step);
            CryptographicOperations.ZeroMemory(check);
        }
    }

    // Algorithm 7: the padded user password /O holds under a padded owner password (Algorithm 3's key), checked as the user's.
    private bool CheckLegacyOwner(ReadOnlySpan<byte> padded, Span<byte> key)
    {
        int n = _keyLength;
        Span<byte> hash = stackalloc byte[16];
        Span<byte> step = stackalloc byte[16];
        Span<byte> user = stackalloc byte[32];
        try
        {
            using (var md5 = IncrementalHash.CreateHash(HashAlgorithmName.MD5))
            {
                md5.AppendData(padded);
                md5.GetHashAndReset(hash);
                for (int i = 0; Revision >= 3 && i < 50; i++)
                {
                    md5.AppendData(hash);
                    md5.GetHashAndReset(hash);
                }
            }

            _o.AsSpan(0, 32).CopyTo(user);
            if (Revision == 2)
            {
                PdfCrypt.Rc4(hash[..n], user, user);
            }
            else
            {
                for (int i = 19; i >= 0; i--)
                {
                    for (int k = 0; k < n; k++)
                    {
                        step[k] = (byte)(hash[k] ^ i);
                    }

                    PdfCrypt.Rc4(step[..n], user, user);
                }
            }

            return CheckLegacyUser(user, key);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(hash);
            CryptographicOperations.ZeroMemory(step);
            CryptographicOperations.ZeroMemory(user);
        }
    }

    // Algorithm 2.A, the user password: the hash of it and /U's validation salt is /U's first 32 bytes; its hash with the
    // key salt decrypts /UE into the file key.
    private bool CheckUser(ReadOnlySpan<byte> password, Span<byte> key, Aes aes) =>
        CheckAes(password, _u.AsSpan(32, 8), _u.AsSpan(40, 8), [], _u.AsSpan(0, 32), _ue, key, aes);

    // Algorithm 2.A, the owner password: the same with /O, its salts and /OE, the 48 bytes of /U hashed in.
    private bool CheckOwner(ReadOnlySpan<byte> password, Span<byte> key, Aes aes) =>
        CheckAes(password, _o.AsSpan(32, 8), _o.AsSpan(40, 8), _u.AsSpan(0, 48), _o.AsSpan(0, 32), _oe, key, aes);

    private bool CheckAes(ReadOnlySpan<byte> password, ReadOnlySpan<byte> validation, ReadOnlySpan<byte> keySalt, ReadOnlySpan<byte> userKey,
        ReadOnlySpan<byte> expected, ReadOnlySpan<byte> encryptedKey, Span<byte> key, Aes aes)
    {
        Span<byte> hash = stackalloc byte[32];
        Span<byte> iv = stackalloc byte[16];
        iv.Clear();   // /UE and /OE are one AES-256-CBC pass with a zero IV
        try
        {
            PdfCrypt.Hash(Revision, password, validation, userKey, hash, aes);
            if (!CryptographicOperations.FixedTimeEquals(hash, expected))
            {
                return false;
            }

            PdfCrypt.Hash(Revision, password, keySalt, userKey, hash, aes);
            aes.SetKey(hash);
            aes.DecryptCbc(encryptedKey[..32], iv, key, PaddingMode.None);
            return true;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(hash);
        }
    }

    // How a stream is decrypted: not a cross-reference stream; its own crypt filter (taken off the chain) when it has one
    // first; metadata left plain when the file says so; embedded files by /EFF; any other by /StmF.
    private PdfCryptMethod StreamMethod(PdfStream stream, ref PdfFilter[] filters)
    {
        var type = stream.Dictionary["Type"u8];
        if (type.IsName("XRef"u8) || stream.Number == _encryptNumber)
        {
            return PdfCryptMethod.Identity;
        }

        if (filters.Length > 0 && filters[0].Name.Is("Crypt"u8))
        {
            var name = filters[0].Parameters?["Name"u8].AsName;
            filters = filters[1..];
            var method = Named(name);
            if (method == PdfCryptMethod.Unknown)
            {
                throw new NotSupportedException(
                    $"the stream is encrypted with a crypt filter of its own, /{name}, whose method /{_filters[IndexOf(name!)].Cfm} the standard handler does not define (None, V2, AESV2 and AESV3 are read)");
            }

            return method;
        }

        if (!_encryptMetadata && type.IsName("Metadata"u8))
        {
            return PdfCryptMethod.Identity;
        }

        return type.IsName("EmbeddedFile"u8) ? _embedded : _streams;
    }

    // The crypt filter `name` names (7.6.6): Identity when none or "Identity"; one /CF does not hold is taken as Identity,
    // as readers take it (its data then fails as damaged where it is encrypted).
    private PdfCryptMethod Named(PdfName? name)
    {
        if (name is null || name.Is("Identity"u8))
        {
            return PdfCryptMethod.Identity;
        }

        int at = IndexOf(name);
        return at < 0 ? PdfCryptMethod.Identity : _filters[at].Method;
    }

    // Where /CF lists the crypt filter `name`, or -1.
    private int IndexOf(PdfName name)
    {
        for (int i = 0; i < _filters.Length; i++)
        {
            if (_filters[i].Name.Bytes.SequenceEqual(name.Bytes))
            {
                return i;
            }
        }

        return -1;
    }

    private static PdfCryptMethod Method(string cfm) => cfm switch
    {
        "None" => PdfCryptMethod.Identity,
        "V2" => PdfCryptMethod.Rc4,
        "AESV2" => PdfCryptMethod.Aes128,
        "AESV3" => PdfCryptMethod.Aes256,
        _ => PdfCryptMethod.Unknown,
    };

    // The key `method` decrypts object number/generation with, into `objectKey` (16 bytes): its length (0 for AES-256,
    // which uses the file key).
    private int ObjectKey(PdfCryptMethod method, int number, int generation, Span<byte> objectKey)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (method is PdfCryptMethod.Identity or PdfCryptMethod.Aes256)
        {
            return 0;
        }

        int length = PdfCrypt.ObjectKey(_key, number, generation, method == PdfCryptMethod.Aes128, objectKey);
        if (method == PdfCryptMethod.Aes128 && length != 16)
        {
            throw new InvalidDataException($"an AES-128 crypt filter with a key of {_keyLength} bytes (the encryption dictionary's /Length); AES-128 needs 16");
        }

        return length;
    }

    // `data` decrypted by `method` with `objectKey` (or the file key) into `output` (as long as `data` at least): the
    // plain length. AES data is its IV and whole blocks; a block cut short is left out, and the padding removed when it
    // is one.
    private int Decrypt(PdfCryptMethod method, ReadOnlySpan<byte> objectKey, ReadOnlySpan<byte> data, Span<byte> output)
    {
        switch (method)
        {
            case PdfCryptMethod.Rc4:
                PdfCrypt.Rc4(objectKey, data, output);
                return data.Length;
            case PdfCryptMethod.Aes128:
            case PdfCryptMethod.Aes256:
                if (method == PdfCryptMethod.Aes256 && _key.Length != 32)
                {
                    throw new InvalidDataException($"an AES-256 crypt filter in a file of revision {Revision}, whose key is {_key.Length} bytes; AES-256 needs revision 5 or 6");
                }

                if (data.Length < 32)
                {
                    return 0;   // an IV alone (an empty string or stream), or less
                }

                bool fileKeyed = method == PdfCryptMethod.Aes256;
                var aes = Rent(fileKeyed);
                try
                {
                    if (!fileKeyed)
                    {
                        aes.SetKey(objectKey);
                    }

                    int written = aes.DecryptCbc(data.Slice(16, (data.Length - 16) & ~15), data[..16], output, PaddingMode.None);
                    int pad = output[written - 1];
                    if (pad is >= 1 and <= 16 && output.Slice(written - pad, pad).IndexOfAnyExcept((byte)pad) < 0)
                    {
                        written -= pad;
                    }

                    return written;
                }
                catch (CryptographicException e)
                {
                    throw new InvalidDataException($"AES data that cannot be decrypted ({e.Message})", e);
                }
                finally
                {
                    Return(aes, fileKeyed);
                }

            default:
                data.CopyTo(output);
                return data.Length;
        }
    }

    // Every string in `value` decrypted, the containers that hold one copied (the parsed objects stay unchanged); false
    // when it holds none.
    private bool TryStrings(PdfValue value, ref StringKey key, out PdfValue decrypted)
    {
        decrypted = value;
        switch (value.Kind)
        {
            case PdfKind.String:
                decrypted = PdfValue.Of(new PdfString(DecryptString(value.AsString!.Bytes, ref key)));
                return true;
            case PdfKind.Array:
                var items = value.AsArray!.Items;
                PdfValue[]? copy = null;
                for (int i = 0; i < items.Length; i++)
                {
                    if (TryStrings(items[i], ref key, out var item))
                    {
                        copy ??= (PdfValue[])items.Clone();
                        copy[i] = item;
                    }
                }

                if (copy is null)
                {
                    return false;
                }

                decrypted = PdfValue.Of(new PdfArray(copy));
                return true;
            case PdfKind.Dictionary:
                if (!TryDictionary(value.AsDictionary!, ref key, out var dictionary))
                {
                    return false;
                }

                decrypted = PdfValue.Of(dictionary);
                return true;
            case PdfKind.Stream:
                var stream = value.AsStream!;
                if (!TryDictionary(stream.Dictionary, ref key, out var streamDictionary))
                {
                    return false;
                }

                decrypted = PdfValue.Of(new PdfStream(streamDictionary, stream.Number, stream.Generation, stream.Offset, stream.Length, stream.LengthRecovered));
                return true;
            default:
                return false;
        }
    }

    private bool TryDictionary(PdfDictionary dictionary, ref StringKey key, out PdfDictionary decrypted)
    {
        decrypted = dictionary;

        // A signature's /Contents (the signature itself) is not encrypted.
        bool signature = dictionary["Type"u8].IsName("Sig"u8) || dictionary["Type"u8].IsName("DocTimeStamp"u8)
                         || (dictionary["Contents"u8].Kind == PdfKind.String && dictionary["ByteRange"u8].Kind == PdfKind.Array);
        PdfValue[]? values = null;
        for (int i = 0; i < dictionary.Count; i++)
        {
            if (signature && dictionary.KeyAt(i).Is("Contents"u8))
            {
                continue;
            }

            if (TryStrings(dictionary.ValueAt(i), ref key, out var item))
            {
                if (values is null)
                {
                    values = new PdfValue[dictionary.Count];
                    for (int k = 0; k < values.Length; k++)
                    {
                        values[k] = dictionary.ValueAt(k);
                    }
                }

                values[i] = item;
            }
        }

        if (values is null)
        {
            return false;
        }

        var keys = new PdfName[dictionary.Count];
        for (int k = 0; k < keys.Length; k++)
        {
            keys[k] = dictionary.KeyAt(k);
        }

        decrypted = new PdfDictionary(keys, values);
        return true;
    }

    // One string's bytes decrypted with its object's key (made at the first string of the object).
    private byte[] DecryptString(byte[] stored, ref StringKey key)
    {
        if (stored.Length == 0)
        {
            return stored;
        }

        if (key.Length < 0)
        {
            key.Length = ObjectKey(_strings, key.Number, key.Generation, key.Buffer);
        }

        if (_strings == PdfCryptMethod.Rc4)
        {
            byte[] plain = new byte[stored.Length];
            Decrypt(_strings, key.Buffer[..key.Length], stored, plain);
            return plain;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(stored.Length);
        try
        {
            int length = Decrypt(_strings, key.Buffer[..key.Length], stored, buffer);
            return buffer.AsSpan(0, length).ToArray();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private Aes Rent(bool fileKeyed)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((fileKeyed ? _fileKeyed : _objectKeyed).TryTake(out var aes))
        {
            return aes;
        }

        aes = Aes.Create();
        if (fileKeyed)
        {
            aes.SetKey(_key);
        }

        return aes;
    }

    private void Return(Aes aes, bool fileKeyed)
    {
        (fileKeyed ? _fileKeyed : _objectKeyed).Add(aes);
        if (_disposed)
        {
            Drain();   // disposed meanwhile: no cipher keeps a key
        }
    }

    private void Drain()
    {
        while (_fileKeyed.TryTake(out var aes))
        {
            aes.Dispose();
        }

        while (_objectKeyed.TryTake(out var aes))
        {
            aes.Dispose();
        }
    }

    private string Cipher => (_streams != PdfCryptMethod.Identity ? _streams : _strings) switch
    {
        PdfCryptMethod.Rc4 => $"RC4 of {_keyLength * 8} bits",
        PdfCryptMethod.Aes128 => "AES-128",
        PdfCryptMethod.Aes256 => "AES-256",
        _ => "no cipher",
    };

    // A crypt method as DocumentSecurity names it.
    private static string CipherName(PdfCryptMethod method) => method switch
    {
        PdfCryptMethod.Identity => "none",
        PdfCryptMethod.Rc4 => "RC4",
        PdfCryptMethod.Aes128 => "AES-128",
        PdfCryptMethod.Aes256 => "AES-256",
        _ => "unknown",
    };

    private InvalidDataException Damaged(string what) => new($"{_name}: the file is encrypted, but its encryption dictionary {what}: it is damaged.");

    private static long? Integer(PdfFile file, PdfDictionary dictionary, ReadOnlySpan<byte> key) =>
        file.Resolve(dictionary[key]).TryInteger(out long value) ? value : null;

    // A string entry's bytes (the encryption dictionary's strings are not encrypted).
    private static byte[]? Bytes(PdfFile file, PdfDictionary dictionary, ReadOnlySpan<byte> key) => file.Resolve(dictionary[key]).AsString?.Bytes;

    // The object a walk over strings decrypts for, and its key once made.
    private ref struct StringKey(Span<byte> buffer, int number, int generation)
    {
        public readonly Span<byte> Buffer = buffer;
        public readonly int Number = number;
        public readonly int Generation = generation;
        public int Length = -1;
    }
}
