// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>Parallel loops over host arrays for model loading and weight conversion (large arrays, on all cores).</summary>
internal static class HostParallel
{
    /// <summary>Runs <paramref name="body"/>(first, last) over [0, <paramref name="count"/>) in chunks of at least <paramref name="grain"/>.</summary>
    public static void For(int count, int grain, Action<int, int> body)
    {
        int chunks = Math.Max(1, Math.Min(ComputeResources.MaxCpuThreads * 4, count / Math.Max(1, grain)));
        if (chunks == 1)
        {
            body(0, count);
            return;
        }

        Parallel.For(0, chunks, ComputeResources.ParallelOptions, c => body((int)((long)count * c / chunks), (int)((long)count * (c + 1) / chunks)));
    }

    /// <summary>The transpose of a row-major [rows, columns] matrix, in L1-sized tiles (Cpu.CpuTuning.TransposeTile) on all cores.</summary>
    public static float[] Transpose(float[] values, int rows, int columns)
    {
        int Tile = Cpu.CpuTuning.TransposeSide;
        var result = new float[values.Length];
        int tileRows = (rows + Tile - 1) / Tile;
        For(tileRows, 1, (first, last) =>
        {
            for (int tr = first; tr < last; tr++)
            {
                int r0 = tr * Tile, r1 = Math.Min(rows, r0 + Tile);
                for (int c0 = 0; c0 < columns; c0 += Tile)
                {
                    int c1 = Math.Min(columns, c0 + Tile);
                    for (int r = r0; r < r1; r++)
                    {
                        for (int c = c0; c < c1; c++)
                        {
                            result[c * rows + r] = values[r * columns + c];
                        }
                    }
                }
            }
        });
        return result;
    }
}
