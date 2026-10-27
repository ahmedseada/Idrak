// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

// A tied int8 or int4 head packed from blocks of its embedding table's rows (no float32 table, no transposed copy): the
// same bits as packing the transposed table whole.
internal abstract partial class Tests
{
    internal static void PackedColumnsMatchWhole(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                             // the quantizers run on the host
        }

        var random = new Random(47);
        var before = ComputeResources.CpuVectors;
        try
        {
            // Small shapes (one block, partial groups and words) with edge values, and one whose columns take several blocks
            // (the last one partial).
            foreach (var (rows, columns) in new[] { (32, 4), (40, 9), (64, 37), (97, 130), (5, 3), (4100, 1030) })
            {
                var values = new float[rows * columns];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(random.NextDouble() * 2 - 1) * (i % 7 == 0 ? 30f : 1f);
                }

                for (int r = 0; r < rows; r++)
                {
                    values[r * columns] = 0f;
                    if (columns > 1)
                    {
                        values[r * columns + 1] = r % 2 == 0 ? 0.5f : -0.5f;
                    }

                    if (columns > 2)
                    {
                        values[r * columns + 2] = (float)random.NextDouble() * 1e-31f;
                    }
                }

                if (columns > 8)
                {
                    values[3 * columns + 5] = 1e-40f;
                    values[1 * columns + 6] = 3e38f;
                    values[2 * columns + 7] = float.PositiveInfinity;
                    values[(rows - 1) * columns + 8] = float.NaN;
                }

                // Column c of the matrix, its rows contiguous: what a tied head reads from its table's row c.
                int blocks = 0;
                void Read(int first, int count, Span<float> destination)
                {
                    blocks++;
                    for (int c = 0; c < count; c++)
                    {
                        for (int r = 0; r < rows; r++)
                        {
                            destination[c * rows + r] = values[r * columns + first + c];
                        }
                    }
                }

                foreach (var paths in new[] { CpuVectorPaths.Scalar, CpuVectorPaths.Automatic })
                {
                    ComputeResources.CpuVectors = paths;
                    using var whole4 = Int4Weight.Quantize(values, rows, columns, device);
                    blocks = 0;
                    using var streamed4 = Int4Weight.QuantizeColumns(rows, columns, device, Read);
                    SameBits($"int4 [{rows}, {columns}] ({paths}, {blocks} block(s))", whole4, streamed4);
                    using var whole8 = Int8Weight.Quantize(values, rows, columns, device);
                    using var streamed8 = Int8Weight.QuantizeColumns(rows, columns, device, Read);
                    SameBits($"int8 [{rows}, {columns}] ({paths})", whole8, streamed8);
                }

                // bfloat16 rounded from blocks of rows (a float32 checkpoint's table) equals rounding the whole matrix.
                int rowBlocks = 0;
                using var rounded = BFloat16Weight.FromRows(rows, columns, device, (first, count, destination) =>
                {
                    rowBlocks++;
                    values.AsSpan(first * columns, count * columns).CopyTo(destination);
                    return true;
                });
                using var wholeHalves = BFloat16Weight.FromValues(values, rows, columns, device);
                Check(rounded is not null, $"bf16 [{rows}, {columns}]: rounded from rows");
                if (rounded is not null)
                {
                    SameBits($"bf16 [{rows}, {columns}] ({rowBlocks} block(s) of rows)", wholeHalves, rounded);
                }

                var transposedValues = new float[values.Length];
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < columns; c++)
                    {
                        transposedValues[c * rows + r] = values[r * columns + c];
                    }
                }

                using var roundedTransposed = BFloat16Weight.FromRows(rows, columns, device, (first, count, destination) =>
                {
                    values.AsSpan(first * columns, count * columns).CopyTo(destination);
                    return true;
                }, transposed: true);
                using var wholeTransposed = BFloat16Weight.FromValues(transposedValues, columns, rows, device);
                if (roundedTransposed is not null)
                {
                    SameBits($"bf16 [{rows}, {columns}] transposed from blocks of rows", wholeTransposed, roundedTransposed);
                }

                if (rows == 4100)
                {
                    Check(rowBlocks > 1, $"[{rows}, {columns}] is rounded in more than one block of rows ({rowBlocks})");
                    Check(blocks > 1, $"[{rows}, {columns}] is read in more than one block ({blocks})");
                }
            }
        }
        finally
        {
            ComputeResources.CpuVectors = before;
        }
    }

    internal static void TiedPackedHeadFromRows(Device device)
    {
        // A safetensors file's rows read in blocks equal its whole read, for each stored type.
        string files = TempFolder("rows");
        try
        {
            var random = new Random(5);
            var table = Enumerable.Range(0, 37 * 11).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            foreach (var type in new[] { SafeTensorType.F32, SafeTensorType.F16, SafeTensorType.BF16 })
            {
                string path = Path.Combine(files, $"{type}.safetensors");
                SafeTensorsWriter.Write(path, [("t", new[] { 37, 11 }, table)], type);
                using var opened = SafeTensorsReader.Open(path);
                ITensorStore store = opened;
                var whole = store.Read("t");
                var rows = new float[whole.Length];
                for (int first = 0; first < 37; first += 10)
                {
                    int count = Math.Min(10, 37 - first);
                    Check(store.ReadRows("t", first, count, rows.AsSpan(first * 11, count * 11)), $"{type}: rows are read in blocks");
                }

                Check(rows.AsSpan().SequenceEqual(whole), $"{type}: the rows read in blocks are the whole read's values");
            }
        }
        finally
        {
            Directory.Delete(files, recursive: true);
        }

        // A tied model loaded int8 and int4 from float32 and bfloat16 checkpoints: its head is the transposed table packed
        // whole, bit for bit (the head is built from blocks of the table's rows).
        var spec = TinyLlamaSpec() with { TieEmbeddings = true };
        foreach (var type in new[] { SafeTensorType.F32, SafeTensorType.BF16 })
        {
            string folder = TempFolder($"tied-{type}");
            try
            {
                PretrainedModel.Create(folder, spec, "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 3, Type = type }).Dispose();
                float[] table;
                using (var opened = SafeTensorsReader.Open(Path.Combine(folder, "model.safetensors")))
                {
                    table = ((ITensorStore)opened).Read("model.embed_tokens.weight");
                }

                var transposed = new float[table.Length];
                for (int v = 0; v < spec.Vocabulary; v++)
                {
                    for (int d = 0; d < spec.Dim; d++)
                    {
                        transposed[d * spec.Vocabulary + v] = table[v * spec.Dim + d];
                    }
                }

                using var expectedTable = BFloat16Weight.FromValues(table, spec.Vocabulary, spec.Dim, device);
                using (var halves = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, BFloat16 = true }))
                {
                    // A tied bf16 head (the only table) from the checkpoint: the transposed table rounded, bit for bit.
                    using var expectedHead = BFloat16Weight.FromValues(transposed, spec.Dim, spec.Vocabulary, device);
                    var halfHead = halves.Network.Descendants().OfType<Linear>().Last().BFloat16;
                    Check(halfHead is not null, $"{type} bf16: the tied head is bfloat16");
                    if (halfHead is not null)
                    {
                        SameBits($"{type} bf16 tied head", expectedHead, halfHead);
                    }
                }


                foreach (bool int4 in new[] { false, true })
                {
                    using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Int4 = int4, Int8 = !int4 });
                    var embedding = model.Network.Descendants().OfType<Embedding>().Single().BFloat16;
                    Check(embedding is not null, $"{type} {(int4 ? "int4" : "int8")}: the table beside a packed head is bfloat16");
                    if (embedding is not null)
                    {
                        SameBits($"{type} {(int4 ? "int4" : "int8")} embedding table", expectedTable, embedding);
                    }

                    var head = model.Network.Descendants().OfType<Linear>().Last().KernelWeight;
                    using IKernelPackedWeight expected = int4 ? Int4Weight.Quantize(transposed, spec.Dim, spec.Vocabulary, device)
                        : Int8Weight.Quantize(transposed, spec.Dim, spec.Vocabulary, device);
                    Check(head is not null && head.GetType() == expected.GetType(), $"{type} {(int4 ? "int4" : "int8")}: the tied head is packed ({head?.GetType().Name})");
                    if (head is not null)
                    {
                        SameBits($"{type} {(int4 ? "int4" : "int8")} tied head", expected, head);
                    }
                }
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    // The packed values and scales of two weights are the same bits.
    private static void SameBits(string what, IKernelPackedWeight expected, IKernelPackedWeight actual)
    {
        var (a, b) = (expected.PackedValues.ToArray(), actual.PackedValues.ToArray());
        var (sa, sb) = (expected.ScaleValues?.ToArray() ?? [], actual.ScaleValues?.ToArray() ?? []);
        int word = a.Length != b.Length ? -2 : Enumerable.Range(0, a.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]), -1);
        int scale = sa.Length != sb.Length ? -2 : Enumerable.Range(0, sa.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(sa[i]) != BitConverter.SingleToInt32Bits(sb[i]), -1);
        Check(word == -1 && scale == -1, $"{what}: packed from columns equals packed whole (first differing word {word}, scale {scale}; -2: lengths {a.Length}/{b.Length}, {sa.Length}/{sb.Length})");
    }
}
