// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Weights kept in their file's format (plan 37, Q0) on Vulkan: the types it has an expansion kernel for (StoredFormats)
// widen on the device into the float32 [k, n] weights, then the float32 product runs over them, the same call the weights
// expanded at load make, so every output is theirs bit for bit. The expansion is one matrix for one product, freed after
// it (the model stays stored); a type without a kernel takes the host fallback, which a load never asks for (it expands
// such weights at load: WeightStorage.Multiplies).
internal sealed partial class VulkanBackend
{
    // The element types this device widens with kernels of its own (BackendCapabilities.StoredFormats).
    internal static readonly HashSet<string> StoredTypes = new(StringComparer.OrdinalIgnoreCase) { "q8_0", "q4_0", "int8", "float8_e4m3", "float8_e5m2" };

    // A 256-entry table of each one-byte code type's values (its CPU conversion's), made once and kept with the backend.
    private readonly Dictionary<string, Storage> _codeTables = new(StringComparer.Ordinal);

    public override void StoredMatMulKernel(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout)
    {
        long values = (long)n * k;
        if (!StoredTypes.Contains(weight.ElementType.Name) || values > int.MaxValue || BlockBytes((int)values) > MaxStorageBytes)
        {
            base.StoredMatMulKernel(x, weight, scales, zeros, y, m, n, k, layout);
            return;
        }

        var w = Allocate((int)values, zeroed: false);
        try
        {
            ExpandStoredKernel(weight, scales, zeros, w, n, k, layout);
            MatMul(x, w, y, m, n, k, false, false, 0f);
        }
        finally
        {
            w.Release();                                                    // reused in queue order (Return)
        }
    }

    public override void ExpandStoredKernel(Storage weight, Storage? scales, Storage? zeros, Storage w, int n, int k, StoredScales layout)
    {
        var type = weight.ElementType;
        bool fits = Fit(weight, w) && (scales is null || Fit(scales)) && (zeros is null || Fit(zeros));
        string name = type.Name.ToLowerInvariant();
        Span<byte> b = stackalloc byte[24];
        switch (name)
        {
            case "q8_0" or "q4_0" when fits && k % 32 == 0 && scales is null:
                Grid(name == "q8_0" ? "stored_q8_0" : "stored_q4_0", (long)n * k, [weight, w], new Push(b).I(n).I(k).Bytes);
                return;
            case "int8" or "float8_e4m3" or "float8_e5m2" when fits:
                var table = CodeTable(type);
                int form = scales is null ? 0 : zeros is null ? 1 : 2;
                Grid("stored_codes", (long)n * k, [weight, table, scales ?? table, zeros ?? table, w],
                    new Push(b).I(n).I(k).I(layout.BlockRows).I(layout.BlockColumns).I(layout.ScaleColumns).I(form).Bytes);
                return;
            default:
                base.ExpandStoredKernel(weight, scales, zeros, w, n, k, layout);
                return;
        }
    }

    // The values of every one-byte code of `type`, as its CPU conversion widens them, on the device.
    private Storage CodeTable(ElementType type)
    {
        lock (_codeTables)
        {
            if (!_codeTables.TryGetValue(type.Name, out var table))
            {
                var codes = new byte[256];
                for (int c = 0; c < codes.Length; c++)
                {
                    codes[c] = (byte)c;
                }

                var values = new float[256];
                type.Decode(codes, values);
                table = Allocate(256, zeroed: false);
                Upload(values, table);
                _codeTables[type.Name] = table;
            }

            return table;
        }
    }
}
