// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Security.Cryptography;
using System.Text;
using Idrak.Abstraction.Text;
using PaddingMode = System.Security.Cryptography.PaddingMode;

namespace Idrak.Data.Pdf;

/// <summary>How a crypt filter decrypts (ISO 32000-2, 7.6.6): not at all, RC4 or AES-128 with an object's own key, or AES-256 with the file key.</summary>
internal enum PdfCryptMethod : byte
{
    /// <summary>Not encrypted: the <c>Identity</c> filter, or <c>/CFM /None</c>.</summary>
    Identity,

    /// <summary>RC4 with the object's key: <c>/CFM /V2</c>, and every string and stream of revisions 2 and 3.</summary>
    Rc4,

    /// <summary>AES-128 in CBC mode with the object's key: <c>/CFM /AESV2</c>.</summary>
    Aes128,

    /// <summary>AES-256 in CBC mode with the file key: <c>/CFM /AESV3</c>.</summary>
    Aes256,

    /// <summary>A method the standard handler does not define (named when it is used).</summary>
    Unknown,
}

/// <summary>
/// The pieces of the standard security handler (ISO 32000-2, 7.6.2-7.6.4): RC4, written here (the base library has none);
/// AES, MD5 and SHA-2 through <see cref="System.Security.Cryptography"/>; an object's own key; the hash of revisions 5 and
/// 6 (Algorithm 2.B); and passwords prepared as each revision asks (PDFDocEncoding up to 32 bytes for revisions 2-4,
/// SASLprep and UTF-8 up to 127 bytes for 5 and 6). Every buffer that holds a password or a key is cleared before it is
/// left (rule 56).
/// </summary>
internal static class PdfCrypt
{
    /// <summary>The most bytes of a password revisions 2-4 use.</summary>
    public const int LegacyPasswordBytes = 32;

    /// <summary>The most bytes of a password revisions 5 and 6 use.</summary>
    public const int PasswordBytes = 127;

    // PDFDocEncoding byte by byte (the character each byte stands for), to find a password's bytes.
    private static readonly char[] DocEncoding = BuildDocEncoding();

    /// <summary>The 32 bytes a password of revisions 2-4 is padded with (Algorithm 2, step a).</summary>
    public static ReadOnlySpan<byte> Padding =>
    [
        0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
        0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
    ];

    /// <summary>
    /// RC4: <paramref name="input"/> through the key stream of <paramref name="key"/> (1 to 256 bytes) into
    /// <paramref name="output"/> (as long as the input; it may be the input itself).
    /// </summary>
    public static void Rc4(ReadOnlySpan<byte> key, ReadOnlySpan<byte> input, Span<byte> output)
    {
        Span<byte> s = stackalloc byte[256];
        for (int i = 0; i < 256; i++)
        {
            s[i] = (byte)i;
        }

        int j = 0;
        for (int i = 0, k = 0; i < 256; i++)
        {
            j = (j + s[i] + key[k]) & 0xFF;
            (s[i], s[j]) = (s[j], s[i]);
            if (++k == key.Length)
            {
                k = 0;
            }
        }

        int x = 0, y = 0;
        for (int n = 0; n < input.Length; n++)
        {
            x = (x + 1) & 0xFF;
            y = (y + s[x]) & 0xFF;
            (s[x], s[y]) = (s[y], s[x]);
            output[n] = (byte)(input[n] ^ s[(s[x] + s[y]) & 0xFF]);
        }

        CryptographicOperations.ZeroMemory(s);
    }

    /// <summary><paramref name="password"/> (its first 32 bytes) padded to 32 bytes with <see cref="Padding"/> into <paramref name="padded"/>.</summary>
    public static void Pad(ReadOnlySpan<byte> password, Span<byte> padded)
    {
        int length = Math.Min(password.Length, LegacyPasswordBytes);
        password[..length].CopyTo(padded);
        Padding[..(LegacyPasswordBytes - length)].CopyTo(padded[length..]);
    }

    /// <summary>
    /// An object's own key (Algorithm 1): MD5 of the file key, the object's number (three bytes) and generation (two
    /// bytes), little-endian, and for AES the bytes "sAlT"; its first n + 5 bytes, at most 16. Returns its length.
    /// </summary>
    public static int ObjectKey(ReadOnlySpan<byte> fileKey, int number, int generation, bool aes, Span<byte> objectKey)
    {
        Span<byte> input = stackalloc byte[64];
        int n = Math.Min(fileKey.Length, 32);
        fileKey[..n].CopyTo(input);
        input[n] = (byte)number;
        input[n + 1] = (byte)(number >> 8);
        input[n + 2] = (byte)(number >> 16);
        input[n + 3] = (byte)generation;
        input[n + 4] = (byte)(generation >> 8);
        int length = n + 5;
        if (aes)
        {
            "sAlT"u8.CopyTo(input[length..]);
            length += 4;
        }

        Span<byte> hash = stackalloc byte[16];
        MD5.HashData(input[..length], hash);
        int keyLength = Math.Min(n + 5, 16);
        hash[..keyLength].CopyTo(objectKey);
        CryptographicOperations.ZeroMemory(input);
        CryptographicOperations.ZeroMemory(hash);
        return keyLength;
    }

    /// <summary>
    /// The hash of revisions 5 and 6 (Algorithm 2.B; revision 5: SHA-256 alone) of a password, an eight-byte salt and
    /// the user key (48 bytes when the owner password is checked, none otherwise), into the 32 bytes of
    /// <paramref name="result"/>. <paramref name="aes"/> is the cipher the rounds of revision 6 encrypt with (its key is
    /// set here).
    /// </summary>
    public static void Hash(int revision, ReadOnlySpan<byte> password, ReadOnlySpan<byte> salt, ReadOnlySpan<byte> userKey, Span<byte> result, Aes aes)
    {
        password = password[..Math.Min(password.Length, PasswordBytes)];
        Span<byte> k = stackalloc byte[64];
        int unit = password.Length + 64 + userKey.Length;   // a round's K1 and E: 64 units each, K at most 64 bytes
        byte[] buffer = ArrayPool<byte>.Shared.Rent(revision >= 6 ? unit * 64 * 2 : unit + salt.Length);
        try
        {
            var first = buffer.AsSpan(0, password.Length + salt.Length + userKey.Length);
            password.CopyTo(first);
            salt.CopyTo(first[password.Length..]);
            userKey.CopyTo(first[(password.Length + salt.Length)..]);
            SHA256.HashData(first, k);
            int kLength = 32;
            if (revision >= 6)
            {
                Span<byte> iv = stackalloc byte[16];
                for (int round = 1; ; round++)
                {
                    // K1: the password, K and the user key, 64 times; E: K1 through AES-128-CBC with K's halves as key and IV.
                    int length = password.Length + kLength + userKey.Length;
                    var k1 = buffer.AsSpan(0, length * 64);
                    password.CopyTo(k1);
                    k[..kLength].CopyTo(k1[password.Length..]);
                    userKey.CopyTo(k1[(password.Length + kLength)..]);
                    for (int copy = 1; copy < 64; copy++)
                    {
                        k1[..length].CopyTo(k1[(copy * length)..]);
                    }

                    var e = buffer.AsSpan(length * 64, length * 64);
                    k[16..32].CopyTo(iv);
                    aes.SetKey(k[..16]);
                    aes.EncryptCbc(k1, iv, e, PaddingMode.None);

                    // The first 16 bytes of E as a number, modulo 3 (256 is 1 modulo 3: the sum of the bytes), choose the next hash.
                    int sum = 0;
                    foreach (byte b in e[..16])
                    {
                        sum += b;
                    }

                    kLength = (sum % 3) switch
                    {
                        0 => SHA256.HashData(e, k),
                        1 => SHA384.HashData(e, k),
                        _ => SHA512.HashData(e, k),
                    };

                    if (round >= 64 && e[^1] <= round - 32)
                    {
                        break;
                    }
                }

                CryptographicOperations.ZeroMemory(iv);
            }

            k[..32].CopyTo(result);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(k);
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);
        }
    }

    /// <summary>
    /// The bytes of <paramref name="password"/> as revisions 2-4 take it, PDFDocEncoding, into <paramref name="bytes"/>
    /// (32 bytes at least): its length, at most 32; -1 when a character is not in PDFDocEncoding.
    /// </summary>
    public static int LegacyPassword(ReadOnlySpan<char> password, Span<byte> bytes)
    {
        int length = Math.Min(password.Length, LegacyPasswordBytes);
        for (int i = 0; i < length; i++)
        {
            char c = password[i];
            int b = c < 0x7F && c >= 0x20 ? c : Array.IndexOf(DocEncoding, c);
            if (b < 0 || c == '\uFFFD')
            {
                bytes[..i].Clear();
                return -1;
            }

            bytes[i] = (byte)b;
        }

        return length;
    }

    /// <summary>
    /// The UTF-8 bytes of <paramref name="password"/> into <paramref name="bytes"/> (at least <paramref name="most"/>),
    /// cut after <paramref name="most"/> bytes (where they fall, as the standard says); the length.
    /// </summary>
    public static int Utf8Password(ReadOnlySpan<char> password, Span<byte> bytes, int most)
    {
        byte[] whole = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(password.Length));
        try
        {
            int length = Encoding.UTF8.GetBytes(password, whole);
            length = Math.Min(length, most);
            whole.AsSpan(0, length).CopyTo(bytes);
            return length;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(whole, clearArray: true);
        }
    }

    /// <summary>
    /// <paramref name="password"/> prepared by SASLprep (RFC 4013) as revisions 5 and 6 ask, into
    /// <paramref name="prepared"/>: the characters mapped to nothing dropped (the soft hyphen, joiners, variation
    /// selectors), the other spaces turned into U+0020, then NFKC through the library's own normalization (the same on
    /// every platform). Its length; -1 when it cannot be prepared (a lone surrogate) or does not fit. The prohibited
    /// characters are not refused: a reader only tries the password.
    /// </summary>
    public static int SaslPrep(ReadOnlySpan<char> password, Span<char> prepared)
    {
        char[] mapped = ArrayPool<char>.Shared.Rent(Math.Max(1, password.Length));
        try
        {
            int n = 0;
            foreach (char c in password)
            {
                if (MappedToNothing(c))
                {
                    continue;
                }

                mapped[n++] = IsNonAsciiSpace(c) ? ' ' : c;
            }

            try
            {
                return UnicodeNormalization.TryNormalize(mapped.AsSpan(0, n), prepared, out int written, NormalizationForm.FormKC) ? written : -1;
            }
            catch (ArgumentException)
            {
                return -1;   // invalid UTF-16: the password is tried as it is
            }
        }
        finally
        {
            ArrayPool<char>.Shared.Return(mapped, clearArray: true);
        }
    }

    // RFC 3454, table B.1: commonly mapped to nothing.
    private static bool MappedToNothing(char c) =>
        c is '\u00AD' or '\u034F' or '\u1806' or '\u180B' or '\u180C' or '\u180D' or '\u200B' or '\u200C' or '\u200D' or '\u2060' or '\uFEFF'
        || (c >= '\uFE00' && c <= '\uFE0F');

    // RFC 3454, table C.1.2: the spaces that are not ASCII's.
    private static bool IsNonAsciiSpace(char c) =>
        c is '\u00A0' or '\u1680' or '\u202F' or '\u205F' or '\u3000' || (c >= '\u2000' && c <= '\u200B');

    private static char[] BuildDocEncoding()
    {
        var table = new char[256];
        for (int b = 0; b < 256; b++)
        {
            table[b] = PdfText.Decode([(byte)b])[0];
        }

        return table;
    }
}
