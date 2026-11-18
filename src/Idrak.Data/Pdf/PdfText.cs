// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// A PDF's text strings (ISO 32000-2, 7.9.2.2): UTF-16BE after its byte order mark, UTF-8 after its byte order mark
/// (PDF 2.0), else PDFDocEncoding (Annex D); and its dates (7.9.4, <c>D:YYYYMMDDHHmmSSOHH'mm</c>, any tail left out).
/// </summary>
internal static class PdfText
{
    // PDFDocEncoding where it differs from Latin-1: 0x18-0x1F (spacing accents), 0x80-0xA0 (punctuation, ligatures,
    // letters, the euro); 0x7F, 0x9F and 0xAD are undefined (U+FFFD).
    private static readonly char[] DocEncoding = BuildDocEncoding();

    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>A text string's characters.</summary>
    public static string Decode(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF)
        {
            return WithoutLanguageMarks(Encoding.BigEndianUnicode.GetString(bytes[2..][..((bytes.Length - 2) & ~1)]));
        }

        if (bytes.Length >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE)
        {
            // Little-endian with its mark: not in the standard, written by some programs, read as they meant it.
            return WithoutLanguageMarks(Encoding.Unicode.GetString(bytes[2..][..((bytes.Length - 2) & ~1)]));
        }

        if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
        {
            return WithoutLanguageMarks(Encoding.UTF8.GetString(bytes[3..]));
        }

        char[] chars = ArrayPool<char>.Shared.Rent(bytes.Length);
        try
        {
            for (int i = 0; i < bytes.Length; i++)
            {
                chars[i] = DocEncoding[bytes[i]];
            }

            return new string(chars, 0, bytes.Length);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(chars);
        }
    }

    /// <summary>Bytes as UTF-8 when they are, else one character a byte (Latin-1).</summary>
    public static string Utf8OrLatin1(ReadOnlySpan<byte> bytes)
    {
        try
        {
            return StrictUtf8.GetString(bytes);
        }
        catch (DecoderFallbackException)
        {
            return Latin1(bytes);
        }
    }

    /// <summary>One character a byte.</summary>
    public static string Latin1(ReadOnlySpan<byte> bytes) => Encoding.Latin1.GetString(bytes);

    /// <summary>
    /// A date string: <c>D:</c> (optional), the year, then the month, day, hour, minute and second as far as given, then
    /// the offset (<c>Z</c>, or <c>+</c> / <c>-</c> with hours and, after an apostrophe or not, minutes; the closing
    /// apostrophe optional). A date without an offset is taken at offset zero. Null when it is not a date.
    /// </summary>
    public static DateTimeOffset? Date(ReadOnlySpan<byte> bytes)
    {
        var s = bytes.Trim(" \t\r\n\0"u8);
        if (s.StartsWith("D:"u8))
        {
            s = s[2..];
        }

        int i = 0;
        if (!Digits(s, ref i, 4, out int year) || year < 1)
        {
            return null;
        }

        int month = 1, day = 1, hour = 0, minute = 0, second = 0;
        _ = Digits(s, ref i, 2, out month, 1) && Digits(s, ref i, 2, out day, 1) && Digits(s, ref i, 2, out hour) && Digits(s, ref i, 2, out minute)
            && Digits(s, ref i, 2, out second);
        if (month is < 1 or > 12 || day < 1 || day > DateTime.DaysInMonth(year, month) || hour > 23 || minute > 59 || second > 59)
        {
            return null;
        }

        var offset = TimeSpan.Zero;
        if (i < s.Length && s[i] is (byte)'+' or (byte)'-')
        {
            int sign = s[i++] == (byte)'-' ? -1 : 1;
            if (!Digits(s, ref i, 2, out int hours) || hours > 23)
            {
                return null;
            }

            if (i < s.Length && s[i] == (byte)'\'')
            {
                i++;
            }

            if (!Digits(s, ref i, 2, out int minutes) || minutes > 59)
            {
                minutes = 0;
            }

            offset = new TimeSpan(sign * hours, sign * minutes, 0);
        }

        if (offset.Duration() > TimeSpan.FromHours(14))
        {
            return null;   // past the offsets clocks use: not a date
        }

        try
        {
            return new DateTimeOffset(year, month, day, hour, minute, second, offset);
        }
        catch (ArgumentException)
        {
            return null;   // a time whose UTC falls before year 1 or after 9999
        }
    }

    // Reads `count` digits at s[i]; false (and `value` = `otherwise`) when they are not there.
    private static bool Digits(ReadOnlySpan<byte> s, ref int i, int count, out int value, int otherwise = 0)
    {
        value = otherwise;
        if (i + count > s.Length)
        {
            return false;
        }

        int v = 0;
        for (int k = 0; k < count; k++)
        {
            byte c = s[i + k];
            if (!char.IsAsciiDigit((char)c))
            {
                return false;
            }

            v = v * 10 + (c - '0');
        }

        value = v;
        i += count;
        return true;
    }

    // A language mark (U+001B, a language code, U+001B) says which language the text is in; it is not text.
    private static string WithoutLanguageMarks(string text)
    {
        int start = text.IndexOf('\u001B', StringComparison.Ordinal);
        if (start < 0)
        {
            return text;
        }

        var builder = new StringBuilder(text.Length);
        int at = 0;
        while (start >= 0)
        {
            int end = text.IndexOf('\u001B', start + 1);
            if (end < 0)
            {
                break;
            }

            builder.Append(text, at, start - at);
            at = end + 1;
            start = text.IndexOf('\u001B', at);
        }

        return builder.Append(text, at, text.Length - at).ToString();
    }

    private static char[] BuildDocEncoding()
    {
        var table = new char[256];
        for (int i = 0; i < 256; i++)
        {
            table[i] = (char)i;
        }

        "˘ˇˆ˙˝˛˚˜".CopyTo(table.AsSpan(0x18));
        ("•†‡…—–ƒ⁄‹›−‰„“”‘"
         + "’‚™ﬁﬂŁŒŠŸŽıłœšž�€").CopyTo(table.AsSpan(0x80));
        table[0x7F] = '�';
        table[0xAD] = '�';
        return table;
    }
}
