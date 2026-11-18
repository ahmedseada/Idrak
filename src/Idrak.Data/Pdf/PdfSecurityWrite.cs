// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Security.Cryptography;
using Idrak.Abstraction.Text;
using PaddingMode = System.Security.Cryptography.PaddingMode;

namespace Idrak.Data.Pdf;

/// <summary>Which of a security handler's crypt filters encrypts a stream being written.</summary>
internal enum PdfCryptUse : byte
{
    /// <summary>Any stream: <c>/StmF</c>.</summary>
    Stream,

    /// <summary>An embedded file's data: <c>/EFF</c> (<c>/StmF</c> when it names none).</summary>
    EmbeddedFile,

    /// <summary>A metadata stream: <c>/StmF</c>, or none under <c>/EncryptMetadata false</c>.</summary>
    Metadata,
}

/// <summary>
/// The standard security handler, writing side: a handler made for a new file (<see cref="Create"/>: revision 6, AES-256,
/// ISO 32000-2 7.6.4.4 Algorithms 8, 9 and 10 for <c>/U</c> and <c>/UE</c>, <c>/O</c> and <c>/OE</c>, <c>/Perms</c>), and
/// strings and streams encrypted by any handler, a new one or one a document was opened with (RC4 and AES-128 with each
/// object's own key, AES-256 with the file key; AES with a fresh IV and its padding), so a file keeps its own encryption
/// when it is updated or written again. Every buffer that held a password or an intermediate key is cleared before it is
/// left (rule 56); the file key is cleared when the handler is disposed.
/// </summary>
internal sealed partial class PdfSecurity
{
    // The permission bits (ISO 32000-2, Table 22): 7, 8 and 13 to 32 set; the ones a user may do added (3-6, 9-12).
    private const uint ReservedPermissionBits = 0xFFFFF0C0;

    private static readonly (DocumentPermissions Permission, int Bit)[] PermissionBitsOf =
    [
        (DocumentPermissions.Print, 3), (DocumentPermissions.Modify, 4), (DocumentPermissions.Copy, 5), (DocumentPermissions.Annotate, 6),
        (DocumentPermissions.FillForms, 9), (DocumentPermissions.Accessibility, 10), (DocumentPermissions.Assemble, 11), (DocumentPermissions.PrintHighQuality, 12),
    ];

    // A handler made for a new file: revision 6, AES-256 for strings, streams and embedded files, metadata encrypted.
    private PdfSecurity(byte[] key, byte[] o, byte[] u, byte[] oe, byte[] ue, int permissions)
    {
        _name = "the document written";
        _encryptNumber = -1;
        Revision = 6;
        _keyLength = 32;
        (_o, _u, _oe, _ue, _id, _permissions, _encryptMetadata) = (o, u, oe, ue, [], permissions, true);
        _filters = [(PdfName.From("StdCF"u8), PdfCryptMethod.Aes256, "AESV3")];
        _strings = _streams = _embedded = PdfCryptMethod.Aes256;
        _key = key;
        Owner = true;
    }

    /// <summary>The permissions as <c>/P</c> holds them (a signed 32-bit number).</summary>
    public int PermissionBits => _permissions;

    /// <summary>What a user who opens the file with the user password may do, as <c>/P</c> says.</summary>
    public DocumentPermissions Permissions => ToPermissions(_permissions);

    /// <summary>The most bytes <paramref name="length"/> bytes take encrypted (AES: an IV and up to a block of padding).</summary>
    public static int MostEncryptedLength(int length) => length + 32;

    /// <summary>
    /// A handler for a new file: a fresh 32-byte file key, <c>/U</c> and <c>/UE</c> from the user password (none: the
    /// empty one), <c>/O</c> and <c>/OE</c> from the owner password (none: 32 random bytes nobody knows, so the permissions
    /// cannot be lifted), <c>/P</c> and <c>/Perms</c> from the permissions. Passwords are prepared by SASLprep and taken as
    /// UTF-8 up to 127 bytes, as readers take them (the text as typed when it cannot be prepared).
    /// </summary>
    public static PdfSecurity Create(DocumentEncryption encryption)
    {
        ArgumentNullException.ThrowIfNull(encryption);
        byte[] key = new byte[32];
        RandomNumberGenerator.Fill(key);
        Span<byte> user = stackalloc byte[PdfCrypt.PasswordBytes];
        Span<byte> owner = stackalloc byte[PdfCrypt.PasswordBytes];
        Span<byte> salts = stackalloc byte[32];
        Span<byte> hash = stackalloc byte[32];
        Span<byte> iv = stackalloc byte[16];
        iv.Clear();                                                             // /UE and /OE: one AES-256-CBC pass with a zero IV
        try
        {
            int userLength = Prepared(encryption.UserPassword, user);
            int ownerLength;
            if (encryption.OwnerPassword is null)
            {
                RandomNumberGenerator.Fill(owner[..32]);
                ownerLength = 32;
            }
            else
            {
                ownerLength = Prepared(encryption.OwnerPassword, owner);
            }

            RandomNumberGenerator.Fill(salts);
            byte[] u = new byte[48], ue = new byte[32], o = new byte[48], oe = new byte[32];
            using var aes = Aes.Create();

            // Algorithm 8: /U the hash of the password and the validation salt, then both salts; /UE the file key under the
            // hash of the password and the key salt.
            PdfCrypt.Hash(6, user[..userLength], salts[..8], [], hash, aes);
            hash.CopyTo(u);
            salts[..16].CopyTo(u.AsSpan(32));
            PdfCrypt.Hash(6, user[..userLength], salts[8..16], [], hash, aes);
            aes.SetKey(hash);
            aes.EncryptCbc(key, iv, ue, PaddingMode.None);

            // Algorithm 9: the same for the owner password, the 48 bytes of /U hashed in.
            PdfCrypt.Hash(6, owner[..ownerLength], salts[16..24], u, hash, aes);
            hash.CopyTo(o);
            salts[16..32].CopyTo(o.AsSpan(32));
            PdfCrypt.Hash(6, owner[..ownerLength], salts[24..32], u, hash, aes);
            aes.SetKey(hash);
            aes.EncryptCbc(key, iv, oe, PaddingMode.None);
            return new PdfSecurity(key, o, u, oe, ue, ToBits(encryption.Permissions));
        }
        catch
        {
            CryptographicOperations.ZeroMemory(key);
            throw;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(user);
            CryptographicOperations.ZeroMemory(owner);
            CryptographicOperations.ZeroMemory(hash);
        }
    }

    /// <summary>
    /// Writes the encryption dictionary of a handler <see cref="Create"/> made: the standard filter, revision 6, one crypt
    /// filter (<c>/AESV3</c>) for strings and streams, <c>/O</c>, <c>/U</c>, <c>/OE</c>, <c>/UE</c>, <c>/P</c> and
    /// <c>/Perms</c> (written as they are: the dictionary itself is never encrypted).
    /// </summary>
    public void WriteDictionary(PdfSyntax s)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        s.Raw("<</Filter/Standard/V 5/R 6/Length 256/CF<</StdCF<</AuthEvent/DocOpen/CFM/AESV3/Length 32>>>>/StmF/StdCF/StrF/StdCF/O"u8).HexString(_o)
            .Raw("/U"u8).HexString(_u).Raw("/OE"u8).HexString(_oe).Raw("/UE"u8).HexString(_ue).Raw("/P "u8).Integer(_permissions)
            .Raw("/Perms"u8).HexString(Perms()).Raw("/EncryptMetadata true>>"u8);
    }

    /// <summary>
    /// Whether <paramref name="perms"/> (an encryption dictionary's <c>/Perms</c>) decrypts under the file key to this
    /// handler's <c>/P</c>, its metadata flag and "adb" (Algorithm 13).
    /// </summary>
    public bool CheckPerms(ReadOnlySpan<byte> perms)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (perms.Length < 16 || _key.Length != 32)
        {
            return false;
        }

        Span<byte> plain = stackalloc byte[16];
        try
        {
            using var aes = Aes.Create();
            aes.SetKey(_key);
            aes.DecryptEcb(perms[..16], plain, PaddingMode.None);
            return plain[9..12].SequenceEqual("adb"u8) && BinaryPrimitives.ReadInt32LittleEndian(plain) == _permissions
                   && plain[8] == (_encryptMetadata ? (byte)'T' : (byte)'F');
        }
        finally
        {
            CryptographicOperations.ZeroMemory(plain);
        }
    }

    /// <summary>The crypt method a stream of the kind <paramref name="use"/> names is encrypted with.</summary>
    public PdfCryptMethod StreamMethod(PdfCryptUse use) => use switch
    {
        PdfCryptUse.EmbeddedFile => _embedded,
        PdfCryptUse.Metadata => _encryptMetadata ? _streams : PdfCryptMethod.Identity,
        _ => _streams,
    };

    /// <summary>
    /// <paramref name="plain"/> encrypted as a string of object <paramref name="number"/> and <paramref name="generation"/>
    /// (<c>/StrF</c>) into <paramref name="output"/> (<see cref="MostEncryptedLength"/> bytes at least); the length written.
    /// </summary>
    public int EncryptString(ReadOnlySpan<byte> plain, int number, int generation, Span<byte> output) => Encrypt(_strings, plain, number, generation, output);

    /// <summary>
    /// <paramref name="data"/> encrypted as a stream of object <paramref name="number"/> and <paramref name="generation"/>
    /// by the crypt filter <paramref name="use"/> names, into <paramref name="output"/> (<see cref="MostEncryptedLength"/>
    /// bytes at least); the length written.
    /// </summary>
    public int EncryptStream(ReadOnlySpan<byte> data, int number, int generation, PdfCryptUse use, Span<byte> output) =>
        Encrypt(StreamMethod(use), data, number, generation, output);

    /// <summary>The permissions <paramref name="bits"/> (a <c>/P</c>) give a user.</summary>
    public static DocumentPermissions ToPermissions(int bits)
    {
        var permissions = DocumentPermissions.None;
        foreach (var (permission, bit) in PermissionBitsOf)
        {
            if ((bits & (1 << (bit - 1))) != 0)
            {
                permissions |= permission;
            }
        }

        return permissions;
    }

    /// <summary>The <c>/P</c> that gives a user <paramref name="permissions"/>.</summary>
    public static int ToBits(DocumentPermissions permissions)
    {
        uint bits = ReservedPermissionBits;
        foreach (var (permission, bit) in PermissionBitsOf)
        {
            if ((permissions & permission) != 0)
            {
                bits |= 1u << (bit - 1);
            }
        }

        return unchecked((int)bits);
    }

    // /Perms (Algorithm 10): /P as 64 bits little-endian (the high half all ones), 'T' or 'F' for the metadata, "adb" and
    // four random bytes, encrypted by AES-256 in ECB mode with the file key.
    private byte[] Perms()
    {
        Span<byte> plain = stackalloc byte[16];
        try
        {
            BinaryPrimitives.WriteInt32LittleEndian(plain, _permissions);
            BinaryPrimitives.WriteUInt32LittleEndian(plain[4..], 0xFFFFFFFF);
            plain[8] = _encryptMetadata ? (byte)'T' : (byte)'F';
            "adb"u8.CopyTo(plain[9..]);
            RandomNumberGenerator.Fill(plain[12..]);
            byte[] perms = new byte[16];
            using var aes = Aes.Create();
            aes.SetKey(_key);
            aes.EncryptEcb(plain, perms, PaddingMode.None);
            return perms;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(plain);
        }
    }

    // `plain` encrypted by `method` for object number/generation into `output`: RC4 with the object's key (the same
    // length), AES-128 with the object's key or AES-256 with the file key (a fresh IV, then the data padded to whole
    // blocks); copied as it is under Identity.
    private int Encrypt(PdfCryptMethod method, ReadOnlySpan<byte> plain, int number, int generation, Span<byte> output)
    {
        Span<byte> objectKey = stackalloc byte[16];
        try
        {
            switch (method)
            {
                case PdfCryptMethod.Identity:
                    plain.CopyTo(output);
                    return plain.Length;
                case PdfCryptMethod.Rc4:
                    int length = ObjectKey(method, number, generation, objectKey);
                    PdfCrypt.Rc4(objectKey[..length], plain, output);
                    return plain.Length;
                case PdfCryptMethod.Aes128:
                case PdfCryptMethod.Aes256:
                    if (method == PdfCryptMethod.Aes256 && _key.Length != 32)
                    {
                        throw new InvalidOperationException($"An AES-256 crypt filter with a file key of {_key.Length} bytes (revision {Revision}): AES-256 needs revision 5 or 6.");
                    }

                    bool fileKeyed = method == PdfCryptMethod.Aes256;
                    int keyLength = ObjectKey(method, number, generation, objectKey);
                    RandomNumberGenerator.Fill(output[..16]);
                    var aes = Rent(fileKeyed);
                    try
                    {
                        if (!fileKeyed)
                        {
                            aes.SetKey(objectKey[..keyLength]);
                        }

                        return 16 + aes.EncryptCbc(plain, output[..16], output[16..], PaddingMode.PKCS7);
                    }
                    finally
                    {
                        Return(aes, fileKeyed);
                    }

                default:
                    throw new NotSupportedException($"The file's crypt filter has a method the standard handler does not define: {method}.");
            }
        }
        finally
        {
            CryptographicOperations.ZeroMemory(objectKey);
        }
    }

    // A password as revisions 5 and 6 take it, into `bytes`: SASLprep, then UTF-8 cut at 127 bytes (the text as typed
    // when it cannot be prepared); its length (0 for none).
    private static int Prepared(string? password, Span<byte> bytes)
    {
        if (string.IsNullOrEmpty(password))
        {
            return 0;
        }

        var text = password.AsSpan(0, Math.Min(password.Length, MostPasswordChars));
        char[] prepared = ArrayPool<char>.Shared.Rent(text.Length * UnicodeNormalization.MaxDecompositionLength(compatibility: true) + 1);
        try
        {
            int n = PdfCrypt.SaslPrep(text, prepared);
            return n < 0
                ? PdfCrypt.Utf8Password(text, bytes, PdfCrypt.PasswordBytes)
                : PdfCrypt.Utf8Password(prepared.AsSpan(0, n), bytes, PdfCrypt.PasswordBytes);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(prepared, clearArray: true);
        }
    }
}
