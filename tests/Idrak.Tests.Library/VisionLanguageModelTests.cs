// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

// NumPy's .npy arrays, which the reference fixtures made with transformers are saved as.
internal abstract partial class Tests
{
    // A NumPy .npy array of little-endian float32 or int64 values, in C order.
    internal static float[] ReadNpyFloat32(string path)
    {
        var (bytes, start) = ReadNpy(path, "<f4");
        var values = new float[(bytes.Length - start) / 4];
        Buffer.BlockCopy(bytes, start, values, 0, values.Length * 4);
        return values;
    }

    internal static long[] ReadNpyInt64(string path)
    {
        var (bytes, start) = ReadNpy(path, "<i8");
        var values = new long[(bytes.Length - start) / 8];
        Buffer.BlockCopy(bytes, start, values, 0, values.Length * 8);
        return values;
    }

    internal static (byte[] Bytes, int Start) ReadNpy(string path, string type)
    {
        var bytes = File.ReadAllBytes(path);
        int major = bytes[6];
        int length = major == 1 ? BitConverter.ToUInt16(bytes, 8) : BitConverter.ToInt32(bytes, 8);
        int start = (major == 1 ? 10 : 12) + length;
        string header = Encoding.ASCII.GetString(bytes, major == 1 ? 10 : 12, length);
        Check(header.Contains($"'descr': '{type}'", StringComparison.Ordinal) && header.Contains("'fortran_order': False", StringComparison.Ordinal), $"{path}: {header}");
        return (bytes, start);
    }
}
