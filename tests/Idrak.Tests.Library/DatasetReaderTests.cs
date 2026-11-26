// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Data.Abstractions;

// Datasets: the CSV and JSON Lines readers, which work on blocks, give what a character-at-a-time (line-at-a-time) reader gives.
internal abstract partial class Tests
{

    internal static string Rows(IEnumerable<JsonObject> rows)
    {
        var text = new StringBuilder();
        foreach (var row in rows)
        {
            text.Append(row.ToJsonString()).Append('\n');
        }

        return text.ToString();
    }

    internal static string Outcome(Func<string> read)
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

    // Hands out a few bytes per read, as a network or decompressing stream may.
    internal sealed class TrickleStream(byte[] data, int seed) : MemoryStream(data)
    {
        private readonly Random _random = new(seed);

        public override int Read(byte[] buffer, int offset, int count) => base.Read(buffer, offset, Math.Min(count, _random.Next(1, 40)));

        public override int Read(Span<byte> buffer) => base.Read(buffer[..Math.Min(buffer.Length, _random.Next(1, 40))]);
    }
}
