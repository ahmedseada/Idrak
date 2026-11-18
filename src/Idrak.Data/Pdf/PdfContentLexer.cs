// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Data.Pdf;

/// <summary>The kinds of token in a content stream or a CMap.</summary>
internal enum PdfToken : byte
{
    End,
    Number,
    Name,
    LiteralString,
    HexString,
    ArrayStart,
    ArrayEnd,
    Dictionary,
    Keyword,
}

/// <summary>
/// The lexer of content streams and CMaps (ISO 32000-2, 7.8.2 and 9.10.3): whitespace and comments skipped, each token
/// given as where it lies in the bytes (a name after its slash, a string between its delimiters, still escaped; a
/// dictionary whole, balanced), numbers parsed from the bytes, operators and other keywords as they are. Nothing is
/// allocated; damaged bytes give the tokens they can (a stray delimiter is skipped), never an error.
/// </summary>
internal ref struct PdfContentLexer
{
    private static readonly SearchValues<byte> Ends = SearchValues.Create("\0\t\n\f\r ()<>[]{}/%"u8);
    private static readonly SearchValues<byte> LiteralSpecial = SearchValues.Create("()\\"u8);
    private static readonly SearchValues<byte> DictionarySpecial = SearchValues.Create("<>()%"u8);

    private readonly ReadOnlySpan<byte> _data;
    private int _position;

    public PdfContentLexer(ReadOnlySpan<byte> data) => _data = data;

    /// <summary>Where the next token starts looking.</summary>
    public int Position
    {
        readonly get => _position;
        set => _position = value;
    }

    public readonly ReadOnlySpan<byte> Data => _data;

    /// <summary>The next token: its kind, where it lies (<paramref name="start"/>, <paramref name="length"/>) and, for a number, its value.</summary>
    public PdfToken Next(out int start, out int length, out double number)
    {
        number = 0;
        while (true)
        {
            SkipWhitespace();
            start = _position;
            length = 0;
            if (_position >= _data.Length)
            {
                return PdfToken.End;
            }

            byte b = _data[_position];
            switch (b)
            {
                case (byte)'/':
                    start = ++_position;
                    _position = TokenEnd(_position);
                    length = _position - start;
                    return PdfToken.Name;
                case (byte)'(':
                    start = ++_position;
                    _position = LiteralEnd(_data, _position, out length);
                    return PdfToken.LiteralString;
                case (byte)'<':
                    if (_position + 1 < _data.Length && _data[_position + 1] == (byte)'<')
                    {
                        _position = DictionaryEnd(_data, _position);
                        length = _position - start;
                        return PdfToken.Dictionary;
                    }

                    start = ++_position;
                    int close = _data[_position..].IndexOf((byte)'>');
                    length = close < 0 ? _data.Length - _position : close;
                    _position = close < 0 ? _data.Length : _position + close + 1;
                    return PdfToken.HexString;
                case (byte)'[':
                    _position++;
                    return PdfToken.ArrayStart;
                case (byte)']':
                    _position++;
                    return PdfToken.ArrayEnd;
                case (byte)'{' or (byte)'}':
                    _position++;
                    length = 1;
                    return PdfToken.Keyword;
                case (byte)')' or (byte)'>':
                    _position++;   // a stray delimiter: skipped
                    continue;
                default:
                    _position = TokenEnd(_position);
                    length = _position - start;
                    if (length == 0)
                    {
                        _position++;
                        continue;
                    }

                    if (b is (byte)'+' or (byte)'-' or (byte)'.' or >= (byte)'0' and <= (byte)'9'
                        && PdfParser.TryNumber(_data.Slice(start, length), out _, out number, out _))
                    {
                        return PdfToken.Number;
                    }

                    return PdfToken.Keyword;
            }
        }
    }

    /// <summary>Skips whitespace and comments.</summary>
    public void SkipWhitespace()
    {
        while (_position < _data.Length)
        {
            byte b = _data[_position];
            if (PdfParser.IsWhitespace(b))
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

    private readonly int TokenEnd(int from)
    {
        int end = _data[from..].IndexOfAny(Ends);
        return end < 0 ? _data.Length : from + end;
    }

    // Past a literal string's closing parenthesis (nesting and escapes followed), from just after its opening one.
    private static int LiteralEnd(ReadOnlySpan<byte> data, int from, out int length)
    {
        int nesting = 1, i = from;
        while (true)
        {
            int next = data[i..].IndexOfAny(LiteralSpecial);
            if (next < 0)
            {
                length = data.Length - from;
                return data.Length;   // never closed: ends with the data
            }

            i += next;
            byte b = data[i++];
            if (b == (byte)'\\')
            {
                i = Math.Min(i + 1, data.Length);
            }
            else if (b == (byte)'(')
            {
                nesting++;
            }
            else if (--nesting == 0)
            {
                length = i - 1 - from;
                return i;
            }
        }
    }

    // Past a dictionary's closing ">>" (nested dictionaries, strings and comments followed), from its "<<".
    private static int DictionaryEnd(ReadOnlySpan<byte> data, int from)
    {
        int depth = 0, i = from;
        while (i < data.Length)
        {
            int next = data[i..].IndexOfAny(DictionarySpecial);
            if (next < 0)
            {
                return data.Length;
            }

            i += next;
            switch (data[i])
            {
                case (byte)'(':
                    i = LiteralEnd(data, i + 1, out _);
                    break;
                case (byte)'%':
                    int end = data[i..].IndexOfAny((byte)'\r', (byte)'\n');
                    i = end < 0 ? data.Length : i + end;
                    break;
                case (byte)'<':
                    if (i + 1 < data.Length && data[i + 1] == (byte)'<')
                    {
                        depth++;
                        i += 2;
                    }
                    else
                    {
                        int close = data[i..].IndexOf((byte)'>');
                        i = close < 0 ? data.Length : i + close + 1;
                    }

                    break;
                default:   // '>' or ')'
                    if (data[i] == (byte)'>' && i + 1 < data.Length && data[i + 1] == (byte)'>')
                    {
                        i += 2;
                        if (--depth == 0)
                        {
                            return i;
                        }
                    }
                    else
                    {
                        i++;
                    }

                    break;
            }
        }

        return data.Length;
    }

    /// <summary>
    /// A literal string's bytes (as <see cref="Next"/> gives them, between its parentheses) with its escapes decoded into
    /// <paramref name="destination"/> (as long as the raw bytes at least: decoding never lengthens); the count written.
    /// An end of line inside it (CR, LF or CR LF) is one LF.
    /// </summary>
    public static int DecodeLiteral(ReadOnlySpan<byte> raw, Span<byte> destination)
    {
        int n = 0;
        for (int i = 0; i < raw.Length; i++)
        {
            byte b = raw[i];
            if (b == (byte)'\r')
            {
                destination[n++] = (byte)'\n';
                if (i + 1 < raw.Length && raw[i + 1] == (byte)'\n')
                {
                    i++;
                }

                continue;
            }

            if (b != (byte)'\\' || i + 1 >= raw.Length)
            {
                destination[n++] = b;
                continue;
            }

            byte e = raw[++i];
            switch (e)
            {
                case (byte)'n':
                    destination[n++] = (byte)'\n';
                    break;
                case (byte)'r':
                    destination[n++] = (byte)'\r';
                    break;
                case (byte)'t':
                    destination[n++] = (byte)'\t';
                    break;
                case (byte)'b':
                    destination[n++] = (byte)'\b';
                    break;
                case (byte)'f':
                    destination[n++] = (byte)'\f';
                    break;
                case (byte)'\r':
                    if (i + 1 < raw.Length && raw[i + 1] == (byte)'\n')
                    {
                        i++;   // a line continuation: the backslash and the end of line are dropped
                    }

                    break;
                case (byte)'\n':
                    break;
                case >= (byte)'0' and <= (byte)'7':
                    int value = e - '0';
                    for (int k = 0; k < 2 && i + 1 < raw.Length && raw[i + 1] is >= (byte)'0' and <= (byte)'7'; k++)
                    {
                        value = value * 8 + (raw[++i] - '0');
                    }

                    destination[n++] = (byte)value;
                    break;
                default:
                    destination[n++] = e;
                    break;
            }
        }

        return n;
    }

    /// <summary>A hex string's digits decoded into <paramref name="destination"/> (half as long as the digits, rounded up); whitespace and other bytes skipped, a last odd digit followed by 0.</summary>
    public static int DecodeHex(ReadOnlySpan<byte> raw, Span<byte> destination)
    {
        int n = 0, half = -1;
        foreach (byte c in raw)
        {
            int d = c is >= (byte)'0' and <= (byte)'9' ? c - '0' : c is >= (byte)'a' and <= (byte)'f' ? c - 'a' + 10 : c is >= (byte)'A' and <= (byte)'F' ? c - 'A' + 10 : -1;
            if (d < 0)
            {
                continue;
            }

            if (half < 0)
            {
                half = d;
            }
            else
            {
                destination[n++] = (byte)(half << 4 | d);
                half = -1;
            }
        }

        if (half >= 0)
        {
            destination[n++] = (byte)(half << 4);
        }

        return n;
    }

    /// <summary>A name's bytes with <c>#xx</c> decoded into <paramref name="destination"/> (as long as the raw bytes); the count written.</summary>
    public static int DecodeName(ReadOnlySpan<byte> raw, Span<byte> destination)
    {
        int n = 0;
        for (int i = 0; i < raw.Length; i++)
        {
            if (raw[i] == (byte)'#' && i + 2 < raw.Length && IsHex(raw[i + 1]) && IsHex(raw[i + 2]))
            {
                destination[n++] = (byte)(PdfParser.HexValue(raw[i + 1]) << 4 | PdfParser.HexValue(raw[i + 2]));
                i += 2;
            }
            else
            {
                destination[n++] = raw[i];
            }
        }

        return n;
    }

    private static bool IsHex(byte c) => c is >= (byte)'0' and <= (byte)'9' or >= (byte)'a' and <= (byte)'f' or >= (byte)'A' and <= (byte)'F';
}
