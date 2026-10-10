// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

/// <summary>
/// The library's JSON repair (<see cref="JsonRepairs.Lenient"/>), as Python's json_repair repairs a model's JSON
/// before parsing it: text or a code fence around it, strings and brackets left open by an answer cut at its token limit,
/// trailing or missing commas, single-quoted and curly-quoted strings, unquoted keys, Python's True, False and None, and
/// quotes inside a value (a quote ends a string only when what follows it can follow a string). JSON that parses as it is
/// comes back unchanged and not marked repaired.
/// </summary>
internal sealed class LenientJsonRepair : IJsonRepair
{
    public string Name => JsonRepairs.Lenient;

    public string Summary => "the first object or array of the text, repaired as json_repair does (open strings and brackets, commas, quotes, bare keys, Python literals, code fences)";

    public JsonRepairResult? Repair(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string body = Unfenced(text);
        int start = body.AsSpan().IndexOfAny('{', '[');
        if (start < 0)
        {
            return null;
        }

        try
        {
            return new JsonRepairResult(JsonNode.Parse(body.AsSpan(start).ToString()), Repaired: false);
        }
        catch (JsonException)
        {
        }

        return new JsonRepairResult(new Reader(body, start).Value(), Repaired: true);
    }

    // The text inside a ``` fence (its language tag dropped), or the text itself.
    private static string Unfenced(string text)
    {
        int open = text.IndexOf("```", StringComparison.Ordinal);
        if (open < 0)
        {
            return text;
        }

        int line = text.IndexOf('\n', open);
        if (line < 0)
        {
            return text[(open + 3)..];
        }

        int close = text.IndexOf("```", line, StringComparison.Ordinal);
        return close < 0 ? text[(line + 1)..] : text[(line + 1)..close];
    }

    // A forgiving reader: every value it starts it finishes, whatever the text lacks.
    private sealed class Reader(string text, int position)
    {
        private int _at = position;

        private bool End => _at >= text.Length;

        private char Current => text[_at];

        public JsonNode? Value()
        {
            SkipSpace();
            if (End)
            {
                return null;
            }

            return Current switch
            {
                '{' => Object(),
                '[' => Array(),
                '"' or '\'' or '“' or '”' => JsonValue.Create(String()),
                _ => Bare(inObject: false),
            };
        }

        private JsonObject Object()
        {
            _at++;                                                                // {
            var obj = new JsonObject();
            while (true)
            {
                SkipSpace();
                if (End)
                {
                    return obj;
                }

                if (Current == '}')
                {
                    _at++;
                    return obj;
                }

                if (Current is ',' or ']')
                {
                    _at++;                                                        // a stray separator
                    continue;
                }

                string key = Current is '"' or '\'' or '“' or '”' ? String() : BareKey();
                SkipSpace();
                if (!End && Current is ':' or '=')
                {
                    _at++;
                    SkipSpace();
                }

                // A key the answer was cut after, or one with no value, keeps an empty value.
                JsonNode? value = End || Current is ',' or '}' ? JsonValue.Create("") : Value();
                obj[key] = value;                                                 // a repeated key keeps the last value
                SkipSpace();
                if (!End && Current == ',')
                {
                    _at++;
                }
            }
        }

        private JsonArray Array()
        {
            _at++;                                                                // [
            var array = new JsonArray();
            while (true)
            {
                SkipSpace();
                if (End)
                {
                    return array;
                }

                if (Current == ']')
                {
                    _at++;
                    return array;
                }

                if (Current is ',' or '}')
                {
                    _at++;
                    continue;
                }

                int before = _at;
                array.Add(Value());
                if (_at == before)
                {
                    _at++;                                                        // never stall on a character nothing reads
                }

                SkipSpace();
                if (!End && Current == ',')
                {
                    _at++;
                }
            }
        }

        // A quoted string: its quote (", ' or a curly quote) closes it only when what follows can follow a string; an
        // unescaped line break stays in it; the end of the text closes it.
        private string String()
        {
            char open = Current;
            char close = open is '“' or '”' ? '”' : open;
            _at++;
            var builder = new StringBuilder();
            while (!End)
            {
                char c = Current;
                if (c == '\\' && _at + 1 < text.Length)
                {
                    _at += 2;
                    char e = text[_at - 1];
                    switch (e)
                    {
                        case 'n': builder.Append('\n'); break;
                        case 't': builder.Append('\t'); break;
                        case 'r': builder.Append('\r'); break;
                        case 'b': builder.Append('\b'); break;
                        case 'f': builder.Append('\f'); break;
                        case 'u' when _at + 4 <= text.Length
                                      && int.TryParse(text.AsSpan(_at, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int code):
                            builder.Append((char)code);
                            _at += 4;
                            break;
                        default: builder.Append(e); break;
                    }

                    continue;
                }

                if (c == close || (close == '"' && c == '”'))
                {
                    if (Closes(_at + 1))
                    {
                        _at++;
                        return builder.ToString();
                    }
                }

                builder.Append(c);
                _at++;
            }

            return builder.ToString();
        }

        // Whether a quote before `at` ends its string: what follows it (after spaces) is the end, ',', ':', '}' or ']',
        // or a line break followed by a key or value of the next entry.
        private bool Closes(int at)
        {
            bool lineBreak = false;
            while (at < text.Length && char.IsWhiteSpace(text[at]))
            {
                lineBreak |= text[at] == '\n';
                at++;
            }

            return at >= text.Length || text[at] is ',' or ':' or '}' or ']' || (lineBreak && text[at] is '"' or '\'');
        }

        // An unquoted key: up to ':' (or a separator), trimmed.
        private string BareKey()
        {
            int start = _at;
            while (!End && Current is not (':' or '=' or ',' or '}' or '\n'))
            {
                _at++;
            }

            return text[start.._at].Trim().Trim('"', '\'');
        }

        // A number, a literal (true, false, null and Python's True, False, None) or unquoted text up to a separator.
        private JsonNode? Bare(bool inObject)
        {
            int start = _at;
            while (!End && Current is not (',' or '}' or ']' or '\n') && !(inObject && Current == ':'))
            {
                _at++;
            }

            string word = text[start.._at].Trim();
            switch (word)
            {
                case "true" or "True": return JsonValue.Create(true);
                case "false" or "False": return JsonValue.Create(false);
                case "null" or "None": return null;
            }

            if (word.Length > 0 && (char.IsAsciiDigit(word[0]) || word[0] is '-' or '+' or '.')
                && double.TryParse(word, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) && double.IsFinite(number))
            {
                return long.TryParse(word, NumberStyles.Integer, CultureInfo.InvariantCulture, out long whole) ? JsonValue.Create(whole) : JsonValue.Create(number);
            }

            return JsonValue.Create(word);
        }

        private void SkipSpace()
        {
            while (!End && char.IsWhiteSpace(Current))
            {
                _at++;
            }
        }
    }
}
