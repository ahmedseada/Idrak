// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Data.Pdf;

/// <summary>
/// Thrown when an object runs past the bytes a parser was given while the file holds more: the caller reads a larger
/// window and parses again. Never reaches a caller of the document.
/// </summary>
internal sealed class PdfTruncatedException() : Exception("The object runs past the bytes read.");

/// <summary>
/// The errors that are a limit of the options passed, or a cycle in the file: an <see cref="InvalidDataException"/> marked
/// so, never repaired around and always reported (the message names the limit and its option, or the cycle).
/// </summary>
internal static class PdfLimit
{
    private const string Mark = "Idrak.Data.Pdf.Limit";

    /// <summary>A limit's (or a cycle's) error saying <paramref name="message"/>.</summary>
    public static InvalidDataException Error(string message)
    {
        var error = new InvalidDataException(message);
        error.Data[Mark] = true;
        return error;
    }

    /// <summary>Whether <paramref name="e"/> is a limit's or a cycle's error.</summary>
    public static bool Is(Exception e) => e is InvalidDataException && e.Data.Contains(Mark);
}

/// <summary>
/// The PDF lexer and parser (ISO 32000-2, 7.2 and 7.3) over a span of the file's bytes: whitespace and comments, numbers
/// (invariant), literal strings (escapes, octal, balanced parentheses, line ends), hex strings, names (<c>#xx</c>),
/// arrays, dictionaries and indirect references, nested no deeper than the document's limit. Nothing is turned into text:
/// keys and keywords are compared as bytes.
/// </summary>
internal ref struct PdfParser
{
    // The bytes that end a token: whitespace and the delimiters.
    private static readonly SearchValues<byte> Ends = SearchValues.Create("\0\t\n\f\r ()<>[]{}/%"u8);
    private static readonly SearchValues<byte> Hex = SearchValues.Create("0123456789abcdefABCDEF"u8);
    private static readonly SearchValues<byte> LiteralSpecial = SearchValues.Create("()\\\r"u8);

    private readonly ReadOnlySpan<byte> _data;
    private readonly int _maxDepth;
    private readonly bool _complete;
    private int _position;

    /// <summary>A parser over <paramref name="data"/>.</summary>
    /// <param name="data">The bytes.</param>
    /// <param name="maxDepth">The deepest nesting of arrays and dictionaries allowed.</param>
    /// <param name="complete">Whether the bytes end where the file (or the object's container) ends; when not, running past them throws <see cref="PdfTruncatedException"/>.</param>
    public PdfParser(ReadOnlySpan<byte> data, int maxDepth, bool complete)
    {
        _data = data;
        _maxDepth = maxDepth;
        _complete = complete;
    }

    public int Position
    {
        readonly get => _position;
        set => _position = value;
    }

    public static bool IsWhitespace(byte b) => b is 0 or 9 or 10 or 12 or 13 or 32;

    public static bool IsDelimiter(byte b) =>
        b is (byte)'(' or (byte)')' or (byte)'<' or (byte)'>' or (byte)'[' or (byte)']' or (byte)'{' or (byte)'}' or (byte)'/' or (byte)'%';

    /// <summary>Skips whitespace and comments; at the end of the bytes, asks for more when the file goes on.</summary>
    public void SkipWhitespace()
    {
        SkipWhitespaceToEnd();
        if (_position >= _data.Length)
        {
            NeedMore();
        }
    }

    /// <summary>Skips whitespace and comments, without asking for more bytes at the end.</summary>
    public void SkipWhitespaceToEnd()
    {
        while (_position < _data.Length)
        {
            byte b = _data[_position];
            if (IsWhitespace(b))
            {
                _position++;
            }
            else if (b == (byte)'%')
            {
                int end = _data[_position..].IndexOfAny((byte)'\r', (byte)'\n');
                _position = end < 0 ? _data.Length : _position + end;
            }
            else
            {
                return;
            }
        }
    }

    /// <summary>Whether the next token is the keyword <paramref name="keyword"/> (consumed when it is).</summary>
    public bool TryKeyword(ReadOnlySpan<byte> keyword)
    {
        int start = _position;
        SkipWhitespace();
        var rest = _data[_position..];
        if (rest.StartsWith(keyword) && (rest.Length == keyword.Length || IsWhitespace(rest[keyword.Length]) || IsDelimiter(rest[keyword.Length])))
        {
            _position += keyword.Length;
            return true;
        }

        if (rest.Length < keyword.Length && keyword.StartsWith(rest))
        {
            NeedMore();
        }

        _position = start;
        return false;
    }

    /// <summary>Reads a whole number that is not negative (an object or generation number, an offset), or fails without moving.</summary>
    public bool TryUnsigned(out long value)
    {
        int start = _position;
        value = 0;
        SkipWhitespace();
        int digits = 0;
        while (_position + digits < _data.Length && char.IsAsciiDigit((char)_data[_position + digits]))
        {
            if (value > (long.MaxValue - 9) / 10)
            {
                _position = start;
                return false;
            }

            value = value * 10 + (_data[_position + digits] - '0');
            digits++;
        }

        int end = _position + digits;
        if (end >= _data.Length && digits > 0)
        {
            NeedMore();
        }

        if (digits == 0 || (end < _data.Length && !IsWhitespace(_data[end]) && !IsDelimiter(_data[end])))
        {
            _position = start;
            value = 0;
            return false;
        }

        _position = end;
        return true;
    }

    /// <summary>Reads an object header <c>n g obj</c>, or fails without moving.</summary>
    public bool TryObjectHeader(out int number, out int generation)
    {
        int start = _position;
        number = generation = 0;
        if (TryUnsigned(out long n) && TryUnsigned(out long g) && n <= int.MaxValue && g <= int.MaxValue && TryKeyword("obj"u8))
        {
            (number, generation) = ((int)n, (int)g);
            return true;
        }

        _position = start;
        return false;
    }

    /// <summary>
    /// After the keyword <c>stream</c>: skips its end of line (CR LF or LF, as the standard says, or a lone CR, as some
    /// writers put) and gives where the stream's data starts.
    /// </summary>
    public int StreamStart()
    {
        if (_position >= _data.Length)
        {
            NeedMore();
            return _position;
        }

        if (_data[_position] == (byte)'\r')
        {
            _position++;
            if (_position >= _data.Length)
            {
                NeedMore();
            }
            else if (_data[_position] == (byte)'\n')
            {
                _position++;
            }
        }
        else if (_data[_position] == (byte)'\n')
        {
            _position++;
        }

        return _position;
    }

    /// <summary>Reads one object (an <c>n g R</c> reference is one object).</summary>
    /// <exception cref="InvalidDataException">The bytes are not an object, or they nest deeper than the limit (a <see cref="PdfLimit"/> error).</exception>
    public PdfValue ReadValue() => ReadValue(0);

    private PdfValue ReadValue(int depth)
    {
        SkipWhitespace();
        if (_position >= _data.Length)
        {
            throw Damaged("the data ends where an object was expected");
        }

        byte b = _data[_position];
        switch (b)
        {
            case (byte)'/':
                _position++;
                return PdfValue.Of(ReadName());
            case (byte)'(':
                _position++;
                return PdfValue.Of(new PdfString(ReadLiteral()));
            case (byte)'<':
                if (_position + 1 >= _data.Length)
                {
                    NeedMore();
                    throw Damaged("a '<' at the end of the data");
                }

                if (_data[_position + 1] == (byte)'<')
                {
                    _position += 2;
                    return ReadDictionaryBody(depth + 1);
                }

                _position++;
                return PdfValue.Of(new PdfString(ReadHex()));
            case (byte)'[':
                _position++;
                return ReadArrayBody(depth + 1);
            case (byte)'+' or (byte)'-' or (byte)'.' or >= (byte)'0' and <= (byte)'9':
                return ReadNumberOrReference();
            default:
                if (IsDelimiter(b))
                {
                    throw Damaged($"'{(char)b}' where an object was expected");
                }

                var word = Token();
                if (word.SequenceEqual("null"u8))
                {
                    return PdfValue.Null;
                }

                if (word.SequenceEqual("true"u8))
                {
                    return PdfValue.Boolean(true);
                }

                if (word.SequenceEqual("false"u8))
                {
                    return PdfValue.Boolean(false);
                }

                throw Damaged($"the keyword '{PdfText.Latin1(word[..Math.Min(word.Length, 32)])}' where an object was expected");
        }
    }

    // A dictionary after its "<<": names and values until ">>".
    private PdfValue ReadDictionaryBody(int depth)
    {
        Deeper(depth);
        var keys = new List<PdfName>(8);
        var values = new List<PdfValue>(8);
        while (true)
        {
            SkipWhitespace();
            if (_position >= _data.Length)
            {
                throw Damaged("a dictionary without its '>>'");
            }

            byte b = _data[_position];
            if (b == (byte)'>')
            {
                if (_position + 1 >= _data.Length)
                {
                    NeedMore();
                }
                else if (_data[_position + 1] == (byte)'>')
                {
                    _position += 2;
                    break;
                }

                throw Damaged("a single '>' inside a dictionary");
            }

            if (b != (byte)'/')
            {
                throw Damaged($"'{(char)b}' where a dictionary's key was expected");
            }

            _position++;
            var key = ReadName();
            var value = ReadValue(depth);
            int existing = -1;
            for (int i = 0; i < keys.Count; i++)
            {
                if (keys[i].Bytes.SequenceEqual(key.Bytes))
                {
                    existing = i;
                    break;
                }
            }

            if (existing >= 0)
            {
                values[existing] = value;
            }
            else
            {
                keys.Add(key);
                values.Add(value);
            }
        }

        return keys.Count == 0 ? PdfValue.Of(PdfDictionary.Empty) : PdfValue.Of(new PdfDictionary([.. keys], [.. values]));
    }

    // An array after its "[": objects until "]".
    private PdfValue ReadArrayBody(int depth)
    {
        Deeper(depth);
        var items = new List<PdfValue>();
        while (true)
        {
            SkipWhitespace();
            if (_position >= _data.Length)
            {
                throw Damaged("an array without its ']'");
            }

            if (_data[_position] == (byte)']')
            {
                _position++;
                break;
            }

            items.Add(ReadValue(depth));
        }

        return items.Count == 0 ? PdfValue.Of(PdfArray.Empty) : PdfValue.Of(new PdfArray([.. items]));
    }

    private readonly void Deeper(int depth)
    {
        if (depth > _maxDepth)
        {
            throw PdfLimit.Error($"objects nested deeper than {_maxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth)");
        }

        if (!RuntimeHelpers.TryEnsureSufficientExecutionStack())
        {
            throw PdfLimit.Error(
                $"objects nested {depth} levels deep, deeper than this thread's stack holds; lower the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth, {_maxDepth} now)");
        }
    }

    // A number; a whole number followed by another and "R" is a reference.
    private PdfValue ReadNumberOrReference()
    {
        var token = Token();
        if (!TryNumber(token, out long integer, out double real, out bool isInteger))
        {
            throw Damaged($"'{PdfText.Latin1(token[..Math.Min(token.Length, 32)])}' is not a number");
        }

        if (!isInteger)
        {
            return PdfValue.Real(real);
        }

        if (integer < 0 || integer > int.MaxValue || !char.IsAsciiDigit((char)token[0]))
        {
            return PdfValue.Integer(integer);
        }

        // The look-ahead: whitespace, digits, whitespace, "R" and the token's end.
        int p = _position;
        while (p < _data.Length && IsWhitespace(_data[p]))
        {
            p++;
        }

        int digits = 0;
        while (p + digits < _data.Length && char.IsAsciiDigit((char)_data[p + digits]))
        {
            digits++;
        }

        int q = p + digits;
        if (q >= _data.Length)
        {
            NeedMore();
            return PdfValue.Integer(integer);
        }

        if (digits == 0 || digits > 10 || !IsWhitespace(_data[q]))
        {
            return PdfValue.Integer(integer);
        }

        while (q < _data.Length && IsWhitespace(_data[q]))
        {
            q++;
        }

        if (q >= _data.Length)
        {
            NeedMore();
            return PdfValue.Integer(integer);
        }

        if (_data[q] != (byte)'R')
        {
            return PdfValue.Integer(integer);
        }

        if (q + 1 >= _data.Length)
        {
            NeedMore();
        }
        else if (!IsWhitespace(_data[q + 1]) && !IsDelimiter(_data[q + 1]))
        {
            return PdfValue.Integer(integer);
        }

        if (!long.TryParse(_data.Slice(p, digits), NumberStyles.None, CultureInfo.InvariantCulture, out long generation) || generation > int.MaxValue)
        {
            return PdfValue.Integer(integer);
        }

        _position = q + 1;
        return PdfValue.Reference((int)integer, (int)generation);
    }

    /// <summary>Parses a number token: an integer, or a real (a sign, digits and one point; extra signs and points as most readers take them).</summary>
    public static bool TryNumber(ReadOnlySpan<byte> token, out long integer, out double real, out bool isInteger)
    {
        integer = 0;
        real = 0;
        isInteger = false;
        if (token.IsEmpty)
        {
            return false;
        }

        if (token.IndexOf((byte)'.') < 0 && long.TryParse(token, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out integer))
        {
            isInteger = true;
            real = integer;
            return true;
        }

        if (double.TryParse(token, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out real) && double.IsFinite(real))
        {
            return true;
        }

        // Damage writers make: "--5", "-.5." or "1.2.3": the signs, then the digits up to the second point.
        bool negative = false;
        int i = 0;
        while (i < token.Length && token[i] is (byte)'+' or (byte)'-')
        {
            negative ^= token[i] == (byte)'-';
            i++;
        }

        double value = 0, scale = 0;
        bool any = false;
        for (; i < token.Length; i++)
        {
            byte c = token[i];
            if (c == (byte)'.')
            {
                if (scale != 0)
                {
                    break;
                }

                scale = 1;
            }
            else if (char.IsAsciiDigit((char)c))
            {
                any = true;
                if (scale == 0)
                {
                    value = value * 10 + (c - '0');
                }
                else
                {
                    scale /= 10;
                    value += (c - '0') * scale;
                }
            }
            else
            {
                return false;
            }
        }

        real = negative ? -value : value;
        return any && double.IsFinite(real);
    }

    /// <summary>The regular characters from here to the next whitespace or delimiter.</summary>
    public ReadOnlySpan<byte> Token()
    {
        var rest = _data[_position..];
        int end = rest.IndexOfAny(Ends);
        if (end < 0)
        {
            NeedMore();
            end = rest.Length;
        }

        _position += end;
        return rest[..end];
    }

    // A name after its "/": regular characters, "#xx" decoded.
    private PdfName ReadName()
    {
        var raw = Token();
        if (raw.IndexOf((byte)'#') < 0)
        {
            return PdfName.From(raw);
        }

        byte[] decoded = ArrayPool<byte>.Shared.Rent(raw.Length);
        try
        {
            int n = 0;
            for (int i = 0; i < raw.Length; i++)
            {
                if (raw[i] == (byte)'#' && i + 2 < raw.Length && Hex.Contains(raw[i + 1]) && Hex.Contains(raw[i + 2]))
                {
                    decoded[n++] = (byte)(HexValue(raw[i + 1]) << 4 | HexValue(raw[i + 2]));
                    i += 2;
                }
                else
                {
                    decoded[n++] = raw[i];
                }
            }

            return PdfName.From(decoded.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(decoded);
        }
    }

    // A literal string after its "(": balanced parentheses, escapes, octal, line continuations; an end of line inside it
    // (CR, LF or CR LF) is one LF. The runs between special bytes are copied whole; the result is never longer than the
    // bytes it is read from.
    private byte[] ReadLiteral()
    {
        var rest = _data[_position..];
        byte[] buffer = ArrayPool<byte>.Shared.Rent(Math.Min(rest.Length, 256));
        int n = 0, nesting = 1, i = 0;
        try
        {
            while (true)
            {
                int run = rest[i..].IndexOfAny(LiteralSpecial);
                int copy = run < 0 ? rest.Length - i : run;
                Append(ref buffer, n, rest.Slice(i, copy));
                n += copy;
                i += copy;
                if (run < 0)
                {
                    NeedMore();
                    break;   // a string the file never closes ends with it
                }

                byte b = rest[i++];
                if (b == (byte)')' && --nesting == 0)
                {
                    break;
                }

                if (b is (byte)'(' or (byte)')')
                {
                    nesting += b == (byte)'(' ? 1 : 0;
                    Append(ref buffer, n++, [b]);
                }
                else if (b == (byte)'\r')
                {
                    if (i >= rest.Length)
                    {
                        NeedMore();
                    }
                    else if (rest[i] == (byte)'\n')
                    {
                        i++;
                    }

                    Append(ref buffer, n++, "\n"u8);
                }
                else
                {
                    if (i >= rest.Length)
                    {
                        NeedMore();
                        break;
                    }

                    n = Escape(rest, ref i, ref buffer, n);
                }
            }

            _position += i;
            return buffer.AsSpan(0, n).ToArray();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // The escape after a backslash at rest[i]; returns the new length.
    private readonly int Escape(ReadOnlySpan<byte> rest, ref int i, ref byte[] buffer, int n)
    {
        byte e = rest[i++];
        switch (e)
        {
            case (byte)'n':
                Append(ref buffer, n, "\n"u8);
                return n + 1;
            case (byte)'r':
                Append(ref buffer, n, "\r"u8);
                return n + 1;
            case (byte)'t':
                Append(ref buffer, n, "\t"u8);
                return n + 1;
            case (byte)'b':
                Append(ref buffer, n, "\b"u8);
                return n + 1;
            case (byte)'f':
                Append(ref buffer, n, "\f"u8);
                return n + 1;
            case (byte)'\r':
                // A line continuation: the backslash and the end of line are dropped.
                if (i >= rest.Length)
                {
                    NeedMore();
                }
                else if (rest[i] == (byte)'\n')
                {
                    i++;
                }

                return n;
            case (byte)'\n':
                return n;
            case >= (byte)'0' and <= (byte)'7':
                int value = e - '0';
                for (int k = 0; k < 2 && i < rest.Length && rest[i] is >= (byte)'0' and <= (byte)'7'; k++)
                {
                    value = value * 8 + (rest[i++] - '0');
                }

                if (i >= rest.Length)
                {
                    NeedMore();
                }

                Append(ref buffer, n, [(byte)value]);   // a high-order overflow is ignored, as the standard says
                return n + 1;
            default:
                // "\(", "\)", "\\" and any other character: the character itself.
                Append(ref buffer, n, [e]);
                return n + 1;
        }
    }

    // Writes bytes at buffer[n..], growing the rented buffer when they do not fit.
    private static void Append(ref byte[] buffer, int n, ReadOnlySpan<byte> bytes)
    {
        if (n + bytes.Length > buffer.Length)
        {
            byte[] larger = ArrayPool<byte>.Shared.Rent(Math.Max(buffer.Length * 2, n + bytes.Length));
            buffer.AsSpan(0, n).CopyTo(larger);
            ArrayPool<byte>.Shared.Return(buffer);
            buffer = larger;
        }

        bytes.CopyTo(buffer.AsSpan(n));
    }

    // A hex string after its "<": pairs of digits, whitespace skipped, a last odd digit followed by 0.
    private byte[] ReadHex()
    {
        var rest = _data[_position..];
        int end = rest.IndexOf((byte)'>');
        if (end < 0)
        {
            NeedMore();
            throw Damaged("a hex string without its '>'");
        }

        var digits = rest[..end];
        var output = new byte[(digits.Length + 1) / 2];
        int n = 0, half = -1;
        foreach (byte c in digits)
        {
            if (IsWhitespace(c))
            {
                continue;
            }

            if (!Hex.Contains(c))
            {
                throw Damaged($"'{(char)c}' in a hex string");
            }

            if (half < 0)
            {
                half = HexValue(c);
            }
            else
            {
                output[n++] = (byte)(half << 4 | HexValue(c));
                half = -1;
            }
        }

        if (half >= 0)
        {
            output[n++] = (byte)(half << 4);
        }

        _position += end + 1;
        return n == output.Length ? output : output[..n];
    }

    public static int HexValue(byte c) => c switch
    {
        >= (byte)'0' and <= (byte)'9' => c - '0',
        >= (byte)'a' and <= (byte)'f' => c - 'a' + 10,
        _ => c - 'A' + 10,
    };

    // At the end of the bytes given: when the file goes on, the caller reads more and parses again.
    private readonly void NeedMore()
    {
        if (!_complete)
        {
            throw new PdfTruncatedException();
        }
    }

    private readonly InvalidDataException Damaged(string what) => new($"{what}, at byte {_position} of the object");
}
