// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using Idrak.Adad.Devices.Cuda;
using Idrak.Adad.Types;

namespace Idrak.Adad.Tests;

internal static class TypeTests
{
    // The seeded blocks: each row of the encoders' rows encoded alone, rows in order.
    private static IEnumerable<(byte[] Bytes, int Rows, int K)> SeededBlocks()
    {
        foreach (var (rows, k, seed) in new[] { (48, 2048, 1), (8, 6144, 2) })
        {
            var values = Data.EncoderRows(rows, k, seed);
            int rowBytes = k / 256 * 144;
            var bytes = new byte[rows * rowBytes];
            for (int r = 0; r < rows; r++)
            {
                SuperBlocks.EncodeQ4K(values.AsSpan(r * k, k), bytes.AsSpan(r * rowBytes, rowBytes));
            }

            yield return (bytes, rows, k);
        }
    }

    public static void EncoderKeepsTheSourceBytes(Checks t)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var (bytes, _, _) in SeededBlocks())
        {
            hash.AppendData(bytes);
        }

        string actual = Convert.ToHexStringLower(hash.GetHashAndReset());
        t.Check(actual == Recorded.EncodeSeeded, $"q4_k: the encoder's bytes hash to {actual}, the source's {Recorded.EncodeSeeded}");
    }

    public static void DecoderMatchesTheSource(Checks t)
    {
        var s = SuperBlocks.Q4K;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var (bytes, rows, k) in SeededBlocks())
        {
            var floats = new float[rows * k];
            s.Decode(bytes, floats);
            hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(floats.AsSpan()));
        }

        string seeded = Convert.ToHexStringLower(hash.GetHashAndReset());
        t.Check(seeded == Recorded.DecodeSeeded, $"q4_k: the seeded blocks decode to {seeded}, the source's {Recorded.DecodeSeeded}");
        var random = Data.RandomBlocks(s, 64, 2048, 5);
        var values = new float[64 * 2048];
        s.Decode(random, values);
        string randomSha = Data.Sha(values);
        t.Check(randomSha == Recorded.DecodeRandom, $"q4_k: the random valid blocks decode to {randomSha}, the source's {Recorded.DecodeRandom}");

        // The decoder's per-value form (SubScale, CodeAt: what the GPU lanes' CPU models read) gives the same floats.
        int differs = 0;
        for (int b = 0; b < random.Length / s.Bytes && differs == 0; b++)
        {
            var block = random.AsSpan(b * s.Bytes, s.Bytes);
            for (int v = 0; v < s.Values; v++)
            {
                var (scale, min) = s.SubScale(block, v / s.SubValues);
                float value = scale * s.CodeAt(block, v) - min;
                if (BitConverter.SingleToInt32Bits(value) != BitConverter.SingleToInt32Bits(values[b * s.Values + v]))
                {
                    differs++;
                    t.Check(false, $"block {b} value {v}: SubScale and CodeAt give {value:R}, Decode {values[b * s.Values + v]:R}");
                }
            }
        }
    }

    public static void DescriptionId(Checks t)
    {
        var s = SuperBlocks.Q4K;
        uint hash = 2166136261;
        foreach (char c in s.Code)
        {
            hash = (hash ^ c) * 16777619;
        }

        t.Check(s.Hash == hash && s.Id == (int)(hash & 0x7FFFFFFF), $"q4_k: hash {s.Hash:x8} id {s.Id}, FNV-1a of its code {hash:x8}");
        t.Check(SuperKernels.SuperCode(s) == Recorded.SuperCode, $"q4_k's kernels' name part {SuperKernels.SuperCode(s)}, the source's {Recorded.SuperCode}");
        var copy = s with { };
        t.Check(copy.Id == s.Id, "an equal description has the same id");
        var other = s with { CodeBias = 1 };
        t.Check(other.Id != s.Id, "another description has another id");
        t.Check(SuperMmaKernels.Fits(s) && SuperMmaKernels.FitsInt8(s), "A and B read q4_k's description");
        t.Check(!SuperMmaKernels.FitsInt8(other), "B does not read a description whose codes carry a bias");
    }
}
