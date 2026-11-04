// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Idrak;
using Idrak.Abstraction;
using Idrak.Models;
using Idrak.Models.Abstractions;

// The low-bit GGUF types and the safetensors float8 / int8 dtypes (plan 36, C14), against their references: the fixtures
// were written by gguf-py 0.19.0's numpy dequantizers (ggml's reference) and by ml_dtypes 0.6.0 and numpy float32 (see
// the commit that added them): data/gguf/quants-lowbit.gguf.gz, data/safetensors/f8-i8.safetensors.gz and
// data/safetensors/tiny-llama-f8-i8.safetensors.gz.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] QuantTypesGroup =
    [
        ("quant types: GGUF IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, TQ1_0, TQ2_0 and MXFP4 are library types of GgufTypes and a GGUF file of each reads to gguf-py's values bit for bit; every IQ grid entry is gguf-py's", d => { CpuOnly(d); GgufLowBitTypes(); }),
        ("quant types: safetensors F8_E4M3 and F8_E5M2 (every code), and F8 and I8 weights with a scale for the tensor, per row, per block (a partial block, a stacked [experts, rows, columns] weight), per group of columns and with a zero point read to numpy's float32 values bit for bit, whole, by rows and transposed; the scales are hidden and named, an I8 tensor without one stays a buffer, and F8 is never written", d => { CpuOnly(d); SafeTensorsQuantized(); }),
        ("quant types: a quantized checkpoint loads: tiny Llama with F8_E4M3 attention (32 x 32 block scales), I8 MLP (per-channel scales) and activation scales beside them gives each weight the reference's values (SHA-256) and the logits of the same weights stored as float32", d => { CpuOnly(d); QuantizedCheckpointLoads(); }),
    ];

    // A gzipped fixture as a temporary file (the caller deletes its folder).
    private static string Gunzip(string relative, string folder, string name)
    {
        string path = Path.Combine(folder, name);
        using var source = new GZipStream(File.OpenRead(TestData(relative)), CompressionMode.Decompress);
        using var target = File.Create(path);
        source.CopyTo(target);
        return path;
    }

    // The values' bits are the expected values' bits (NaNs and signed zeros included).
    private static void AssertSameBits(float[] actual, float[] expected, string what)
    {
        Check(actual.Length == expected.Length, $"{what}: {actual.Length} values, expected {expected.Length}");
        for (int i = 0; i < actual.Length; i++)
        {
            if (BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(expected[i]))
            {
                int differ = actual.Where((v, j) => BitConverter.SingleToUInt32Bits(v) != BitConverter.SingleToUInt32Bits(expected[j])).Count();
                Check(false, $"{what}: value {i} is {actual[i]:R} (0x{BitConverter.SingleToUInt32Bits(actual[i]):X8}), the reference's {expected[i]:R} "
                             + $"(0x{BitConverter.SingleToUInt32Bits(expected[i]):X8}); {differ} of {actual.Length} differ");
            }
        }
    }


    private static void GgufLowBitTypes()
    {
        string folder = TempFolder();
        try
        {
            using var file = GgufFile.Open(Gunzip("gguf/quants-lowbit.gguf.gz", folder, "quants-lowbit.gguf"));
            var types = new (string Name, int Id, int Values, int Bytes)[]
            {
                ("IQ1_S", 19, 256, 50), ("IQ1_M", 29, 256, 56), ("IQ2_XXS", 16, 256, 66), ("IQ2_XS", 17, 256, 74), ("IQ2_S", 22, 256, 82),
                ("IQ3_XXS", 18, 256, 98), ("IQ3_S", 21, 256, 110), ("TQ1_0", 34, 256, 54), ("TQ2_0", 35, 256, 66), ("MXFP4", 39, 32, 17),
            };
            foreach (var (name, id, values, bytes) in types)
            {
                var type = GgufTypes.Find(id);
                Check(type is { } t && t.Name == name && t.BlockValues == values && t.BlockBytes == bytes && GgufTypes.Default(id) is not null
                      && GgufFile.TypeName(id) == name && file.Tensors[$"q.{name}"].Type == id,
                    $"{name} (#{id}) is a library type of GgufTypes with {values}-value blocks of {bytes} bytes: {type?.Name} {type?.BlockValues} {type?.BlockBytes}");
                AssertSameBits(file.Read($"q.{name}"), file.Read($"expected.{name}"), $"{name} read from a GGUF file against gguf-py");
            }

            foreach (var (name, grid, width) in new (string, sbyte[], int)[]
                     {
                         ("IQ1_S", GgufGrids.IQ1S, 8), ("IQ2_XXS", GgufGrids.IQ2XXS, 8), ("IQ2_XS", GgufGrids.IQ2XS, 8), ("IQ2_S", GgufGrids.IQ2S, 8),
                         ("IQ3_XXS", GgufGrids.IQ3XXS, 4), ("IQ3_S", GgufGrids.IQ3S, 4),
                     })
            {
                var info = file.Tensors[$"grid.{name}"];
                Check(info.Shape.SequenceEqual([grid.Length / width, width]), $"{name}'s grid: [{string.Join(", ", info.Shape)}] in the fixture, {grid.Length / width} entries of {width} here");
                AssertSameBits([.. grid.Select(v => (float)v)], file.Read($"grid.{name}"), $"{name}'s grid against gguf-py's init_grid");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SafeTensorsQuantized()
    {
        string folder = TempFolder();
        try
        {
            using var reader = SafeTensorsReader.Open(Gunzip("safetensors/f8-i8.safetensors.gz", folder, "f8-i8.safetensors"));
            string[] weights = ["e4m3.all", "e5m2.all", "block.weight", "group.weight", "channel.weight", "tensor.weight", "zero.weight", "experts.weight", "plain.weight", "norm.weight"];
            foreach (string name in weights)
            {
                var expected = reader.Read("expected." + name);
                var info = reader.Tensors[name];
                AssertSameBits(reader.Read(name), expected, $"{name} ({info.Type} [{string.Join(", ", info.Shape)}]) against the reference");

                // Rows from the middle (a scale block's edge within them), and the transpose of a matrix.
                int rows = info.Shape[0], rowValues = (int)(info.Count / rows), first = rows / 3, count = Math.Max(1, rows - first - 1);
                var part = new float[count * rowValues];
                Check(reader.ReadRows(name, first, count, part), $"{name}: rows read");
                AssertSameBits(part, expected.AsSpan(first * rowValues, part.Length).ToArray(), $"{name}: rows {first} to {first + count}");
                if (info.Shape.Length == 2)
                {
                    AssertSameBits(reader.ReadTransposed(name), Idrak.Abstraction.Devices.HostParallel.Transpose(expected, info.Shape[0], info.Shape[1]), $"{name}: transposed");
                }
            }

            Check(reader.Tensors["channel.weight"].Type == SafeTensorType.I8 && reader.Tensors["block.weight"].BytesPerValue == 1 && reader.Tensors["norm.weight"].BytesPerValue == 4,
                "an I8 weight with a scale is a tensor of type I8; F8 and I8 take one byte a value");
            var scales = reader.QuantizationScales;
            Check(scales["block.weight_scale_inv"] == "block.weight" && scales["group.weight_scale"] == "group.weight" && scales["zero.weight_zero_point"] == "zero.weight"
                  && scales["experts.weight_scale_inv"] == "experts.weight" && scales["channel.input_scale"] == "activations"
                  && !reader.Tensors.Keys.Any(n => n.EndsWith("_scale", StringComparison.Ordinal) || n.EndsWith("_scale_inv", StringComparison.Ordinal) || n.EndsWith("_zero_point", StringComparison.Ordinal)),
                $"the scales are named with what they scale, and are not tensors: {string.Join(", ", scales.Select(p => $"{p.Key} → {p.Value}"))}");
            Check(reader.LeftOut.TryGetValue("buffer.ids", out string? type) && type == "I8" && !reader.Tensors.ContainsKey("buffer.ids"), "an I8 tensor without a scale is a buffer, left out");
            ThrowsNaming<NotSupportedException>(() => reader.Read("block.weight_scale_inv"), "reading a weight's scale", "quantization scale of 'block.weight'");
            ThrowsNaming<NotSupportedException>(() => reader.Read("channel.input_scale"), "reading an activation scale", "activation scale");
            Check(reader.SameValues("tensor.weight", "tensor.weight") && !reader.SameValues("block.weight", "expected.norm.weight"), "SameValues compares scaled weights");
            ThrowsNaming<NotSupportedException>(() => SafeTensorsWriter.Write(Path.Combine(folder, "out.safetensors"), new List<(string, int[], float[])> { ("x", new[] { 1 }, new[] { 1f }) }, SafeTensorType.F8_E4M3),
                "writing F8", "read only");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void QuantizedCheckpointLoads()
    {
        string root = TempFolder();
        try
        {
            // The quantized checkpoint beside tiny Llama's config and tokenizer, and the same weights as float32.
            string quantized = Directory.CreateDirectory(Path.Combine(root, "quantized")).FullName, plain = Directory.CreateDirectory(Path.Combine(root, "float32")).FullName;
            Gunzip("safetensors/tiny-llama-f8-i8.safetensors.gz", quantized, "model.safetensors");
            foreach (string file in new[] { "config.json", "tokenizer.json", "tokenizer_config.json" })
            {
                File.Copy(TestData($"gguf/tiny-llama-hf/{file}"), Path.Combine(quantized, file));
                File.Copy(TestData($"gguf/tiny-llama-hf/{file}"), Path.Combine(plain, file));
            }

            using (var reader = SafeTensorsReader.Open(quantized))
            {
                int checkedWeights = 0;
                foreach (var (key, sum) in reader.Metadata.Where(p => p.Key.StartsWith("idrak.expected_sha256.", StringComparison.Ordinal)))
                {
                    string name = key["idrak.expected_sha256.".Length..];
                    var values = reader.Read(name);
                    string ours = Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
                    Check(ours == sum, $"{name} ({reader.Tensors[name].Type}): its values' SHA-256 {ours}, the reference's {sum}");
                    checkedWeights++;
                }

                Check(checkedWeights == 14 && reader.Tensors.Values.Count(t => t.Type == SafeTensorType.F8_E4M3) == 8 && reader.Tensors.Values.Count(t => t.Type == SafeTensorType.I8) == 6,
                    $"8 F8 attention projections and 6 I8 MLP projections checked: {checkedWeights}");
                SafeTensorsWriter.Write(Path.Combine(plain, "model.safetensors"), reader.Tensors.Keys.Select(n => (n, reader.Tensors[n].Shape, reader.Read(n))).ToList(), SafeTensorType.F32);
            }

            using var fromQuantized = PretrainedModel.Load(quantized, new PretrainedOptions { Device = Device.Cpu });
            using var fromPlain = PretrainedModel.Load(plain, new PretrainedOptions { Device = Device.Cpu });
            using var input = Tensor.From([5f, 17f, 101f, 3f, 64f, 200f, 9f, 42f], [1, 8], Device.Cpu);
            AssertSameBits(fromQuantized.Network.Predict(input).ToArray(), fromPlain.Network.Predict(input).ToArray(), "logits of the quantized checkpoint against its float32 copy");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
