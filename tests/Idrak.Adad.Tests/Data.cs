// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Idrak.Adad.Types;

namespace Idrak.Adad.Tests;

// The tests' data, made exactly as the reference harness makes it (test-output\adad\piece1-q4k\reference\Program.cs), so
// the hashes it recorded from the source compare with Adad's.
internal static class Data
{
    /// <summary>The test shapes (m, n, k): real products of the two models and edges (rows and columns past a tile, a short k).</summary>
    public static readonly (int M, int N, int K)[] Shapes =
    [
        (128, 1024, 2048), (512, 2048, 2048), (512, 6144, 2048), (512, 2048, 6144), (128, 3072, 3072), (512, 1024, 3072),
        (128, 3072, 8192), (200, 300, 768), (37, 70, 256),
    ];

    /// <summary>The seeded rows of the encoders' test (48 × 2048 from seed 1, then 8 × 6144 from seed 2, with its edges).</summary>
    public static float[] EncoderRows(int rows, int k, int seed)
    {
        var random = new Random(seed);
        var values = new float[rows * k];
        for (int i = 0; i < values.Length; i++)
        {
            float v = (float)(random.NextDouble() + random.NextDouble() + random.NextDouble() + random.NextDouble() - 2.0) * 0.03f;
            values[i] = i % 97 == 0 ? v * 9f : v;
        }

        values.AsSpan(0, 256).Clear();
        values.AsSpan(256, 256).Fill(0.0125f);
        (values[512], values[520]) = (0.5f, -0.5f);
        (values[680], values[690], values[700]) = (-0.5f, 0.5f, -0.5f);
        values.AsSpan(736, 32).Fill(1e-40f);
        values[770] = float.NaN;
        values[1030] = float.PositiveInfinity;
        return values;
    }

    /// <summary>Random valid blocks: random bytes, d and dmin random small finite halves.</summary>
    public static byte[] RandomBlocks(SuperBlock s, int rows, int k, int seed)
    {
        var random = new Random(seed);
        var bytes = new byte[(long)rows * (k / s.Values) * s.Bytes];
        random.NextBytes(bytes);
        for (long at = 0; at < bytes.Length; at += s.Bytes)
        {
            BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan((int)at + s.ScaleOffset), (Half)(random.NextSingle() * 0.002f));
            if (s.MinOffset >= 0)
            {
                BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan((int)at + s.MinOffset), (Half)(random.NextSingle() * 0.002f));
            }
        }

        return bytes;
    }

    /// <summary>Random rows of x: about normal (the sum of four uniforms).</summary>
    public static float[] RandomRows(int rows, int k, int seed)
    {
        var random = new Random(seed);
        var x = new float[(long)rows * k];
        for (long i = 0; i < x.Length; i++)
        {
            x[i] = (float)(random.NextDouble() + random.NextDouble() + random.NextDouble() + random.NextDouble() - 2.0);
        }

        return x;
    }

    public static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    public static string Sha(ReadOnlySpan<float> values) => Sha(MemoryMarshal.AsBytes(values));

    public static string ShaText(string text) => Sha(System.Text.Encoding.ASCII.GetBytes(text.Replace("\r\n", "\n", StringComparison.Ordinal)));
}
