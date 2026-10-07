// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using System.Text.Json;
using System.Text;
using Idrak.Data.Abstractions;
using Idrak.Data;
using Idrak;

// Datasets: the CSV and JSON Lines readers, which work on blocks, give what a character-at-a-time (line-at-a-time) reader gives.
internal static partial class Tests
{
    private static void DatasetReadersMatchReference(Device device)
    {
        _ = device;
        var random = new Random(7);
        var raw = new ReadOptions { InferTypes = false };

        // CSV: random text of quotes, delimiters and line breaks (\r\n, \n, \r), read whole and in random small pieces so
        // records, quotes and \r\n pairs fall across block edges; a few texts are longer than a block.
        string[] alphabet = ["a", "b", " ", ",", ",", "\"", "\"", "\"\"", "\r", "\n", "\r\n", "\t", "é", "日本"];
        for (int t = 0; t < 400; t++)
        {
            int parts = t % 50 == 0 ? 60_000 : random.Next(0, 80);
            var text = new StringBuilder(t % 7 == 0 ? "﻿" : "");
            for (int i = 0; i < parts; i++)
            {
                text.Append(alphabet[random.Next(alphabet.Length)]);
            }

            var bytes = Encoding.UTF8.GetBytes(text.ToString());
            foreach (char delimiter in new[] { ',', '\t' })
            {
                string expected = Rows(ReferenceDelimited(bytes, delimiter));
                var format = delimiter == ',' ? DataFormat.Csv : DataFormat.Tsv;
                string whole = Rows(DataFiles.ReadStream(() => new MemoryStream(bytes), "t.csv", format, raw));
                string pieces = Rows(DataFiles.ReadStream(() => new TrickleStream(bytes, random.Next()), "t.csv", format, raw));
                Check(whole == expected && pieces == expected, $"csv text {t} ({delimiter switch { ',' => "comma", _ => "tab" }}) reads as before");
            }
        }

        string Csv(string text) => Rows(DataFiles.ReadStream(() => new MemoryStream(Encoding.UTF8.GetBytes(text)), "t.csv", DataFormat.Csv, raw));
        Check(Csv("a,b\r\n\"x,\"\"y\"\"\r\nz\",\r\n\r\n1,2") == "{\"a\":\"x,\\u0022y\\u0022\\r\\nz\",\"b\":\"\"}\n{\"a\":\"1\",\"b\":\"2\"}\n", "csv quotes, CRLF, blank line, no last line break");
        Check(Csv("a,b\r1,2\r3") == "{\"a\":\"1\",\"b\":\"2\"}\n{\"a\":\"3\",\"b\":\"\"}\n", "csv lone \\r ends records");
        Check(Csv("a,b\n\"q\"x\"y,\"\"\n") == "{\"a\":\"qx\\u0022y\",\"b\":\"\"}\n", "csv text after a closing quote, empty quoted field");
        string longField = new('x', 100_000);
        Check(Csv($"a,b\n\"{longField}\",{longField}\n") == $"{{\"a\":\"{longField}\",\"b\":\"{longField}\"}}\n", "csv fields longer than a block");

        // JSON Lines: objects, other values, blank lines (Unicode spaces too), \r\n, \n and \r, invalid UTF-8, byte order marks.
        string[] values = ["{\"a\":1}", "{\"b\":\"é 日本\",\"c\":[1,2,{\"d\":null}]}", "[1,2]", "3", "\"s\"", " { \"e\" : true } ", "{\"long\":\"" + new string('y', 70_000) + "\"}"];
        string[] blanks = ["", " ", "\t", " ", " 　 ", "\v\f"];
        string[] breaks = ["\n", "\r\n", "\r"];
        for (int t = 0; t < 300; t++)
        {
            var stream = new MemoryStream();
            if (t % 5 == 0)
            {
                stream.Write([0xEF, 0xBB, 0xBF]);
            }

            int lines = random.Next(0, 12), odd = random.Next(Math.Max(lines, 1));
            for (int i = 0; i < lines; i++)
            {
                string line = random.Next(4) == 0 ? blanks[random.Next(blanks.Length)] : values[random.Next(t % 3 == 0 ? values.Length : values.Length - 1)];
                stream.Write(Encoding.UTF8.GetBytes(line));
                if (t % 11 == 0 && i == odd)
                {
                    stream.Write([0xFF]);                           // invalid UTF-8: after a value, so it is a parse error
                }
                else if (t % 13 == 0 && i == odd)
                {
                    stream.Write("{\"bad\":\""u8);
                    stream.Write([0xC3, 0x28, 0x80]);               // invalid UTF-8 inside a string: replaced as text, read
                    stream.Write("\"}"u8);
                }

                if (i < lines - 1 || random.Next(2) == 0)
                {
                    stream.Write(Encoding.UTF8.GetBytes(breaks[random.Next(breaks.Length)]));
                }
            }

            var bytes = stream.ToArray();
            string expected = Outcome(() => Rows(ReferenceJsonLines(bytes, "t.jsonl")));
            string whole = Outcome(() => Rows(DataFiles.ReadStream(() => new MemoryStream(bytes), "t.jsonl", DataFormat.JsonLines, raw)));
            string pieces = Outcome(() => Rows(DataFiles.ReadStream(() => new TrickleStream(bytes, random.Next()), "t.jsonl", DataFormat.JsonLines, raw)));
            Check(whole == expected && pieces == expected, $"json lines text {t} reads as before: {whole[..Math.Min(whole.Length, 200)]} / {expected[..Math.Min(expected.Length, 200)]}");
        }

        // UTF-16 with a byte order mark is read as text, as StreamReader detects it.
        var utf16 = Encoding.Unicode.GetPreamble().Concat(Encoding.Unicode.GetBytes("{\"a\":\"é\"}\r\n\r\n[1]\n")).ToArray();
        Check(Rows(DataFiles.ReadStream(() => new TrickleStream(utf16, 1), "t.jsonl", DataFormat.JsonLines, raw)) == Rows(ReferenceJsonLines(utf16, "t.jsonl")), "json lines in UTF-16");
        string error = Outcome(() => Rows(DataFiles.ReadStream(() => new MemoryStream("{\"a\":1}\r\n\r\n{\"a\":\n"u8.ToArray()), "t.jsonl", DataFormat.JsonLines, raw)));
        Check(error.StartsWith("InvalidDataException: t.jsonl, line 3:", StringComparison.Ordinal), $"json lines errors name the line: {error}");
    }

    private static string Rows(IEnumerable<JsonObject> rows)
    {
        var text = new StringBuilder();
        foreach (var row in rows)
        {
            text.Append(row.ToJsonString()).Append('\n');
        }

        return text.ToString();
    }

    private static string Outcome(Func<string> read)
    {
        try
        {
            return read();
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    // The CSV reader as it was: a character at a time, without type inference.
    private static IEnumerable<JsonObject> ReferenceDelimited(byte[] data, char delimiter)
    {
        using var reader = new StreamReader(new MemoryStream(data), Encoding.UTF8, true, 1 << 16);
        string[]? header = null;
        var fields = new List<string>();
        var field = new StringBuilder();
        bool quoted = false, any = false;
        var records = new List<List<string>>();
        int c;
        while ((c = reader.Read()) >= 0)
        {
            char ch = (char)c;
            any = true;
            if (quoted)
            {
                if (ch == '"')
                {
                    if (reader.Peek() == '"')
                    {
                        field.Append('"');
                        reader.Read();
                    }
                    else
                    {
                        quoted = false;
                    }
                }
                else
                {
                    field.Append(ch);
                }
            }
            else if (ch == '"' && field.Length == 0)
            {
                quoted = true;
            }
            else if (ch == delimiter)
            {
                fields.Add(field.ToString());
                field.Clear();
            }
            else if (ch is '\n' or '\r')
            {
                if (ch == '\r' && reader.Peek() == '\n')
                {
                    reader.Read();
                }

                fields.Add(field.ToString());
                field.Clear();
                records.Add(fields);
                fields = [];
                any = false;
            }
            else
            {
                field.Append(ch);
            }
        }

        if (any)
        {
            fields.Add(field.ToString());
            records.Add(fields);
        }

        foreach (var record in records)
        {
            if (header is null)
            {
                header = [.. record.Select((h, i) => h.Length > 0 ? h : $"column{i + 1}")];
                continue;
            }

            if (record.Count == 1 && record[0].Length == 0)
            {
                continue;
            }

            var row = new JsonObject();
            for (int i = 0; i < header.Length; i++)
            {
                row[header[i]] = i < record.Count ? record[i] : "";
            }

            yield return row;
        }
    }

    // The JSON Lines reader as it was: lines decoded to text, then parsed.
    private static IEnumerable<JsonObject> ReferenceJsonLines(byte[] data, string path)
    {
        using var reader = new StreamReader(new MemoryStream(data), Encoding.UTF8, true, 1 << 16);
        int number = 0;
        while (reader.ReadLine() is { } line)
        {
            number++;
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            JsonNode? node;
            try
            {
                node = JsonNode.Parse(line);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"{path}, line {number}: {ex.Message}", ex);
            }

            yield return node as JsonObject ?? new JsonObject { ["value"] = node };
        }
    }

    // Hands out a few bytes per read, as a network or decompressing stream may.
    private sealed class TrickleStream(byte[] data, int seed) : MemoryStream(data)
    {
        private readonly Random _random = new(seed);

        public override int Read(byte[] buffer, int offset, int count) => base.Read(buffer, offset, Math.Min(count, _random.Next(1, 40)));

        public override int Read(Span<byte> buffer) => base.Read(buffer[..Math.Min(buffer.Length, _random.Next(1, 40))]);
    }
}
