// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

// The packed-weight cache of a load (PretrainedOptions.PackedCache): a cold load packs and keeps, a warm one reads the same
// bits back; an entry is made again when its file changes, kept beside other formats, pruned when stale; off keeps nothing.
internal abstract partial class Tests
{
    internal static void PackedWeightCacheLoads(Device device)
    {
        if (device.Type != DeviceType.Cpu || !FirstRun(nameof(PackedWeightCacheLoads)))
        {
            return;                                                             // the cache is the host's; one device shows it
        }

        Check(ComputeResources.SwitchedOff("0") && ComputeResources.SwitchedOff(" OFF ") && ComputeResources.SwitchedOff("no") && ComputeResources.SwitchedOff("False")
              && !ComputeResources.SwitchedOff(null) && !ComputeResources.SwitchedOff("1") && !ComputeResources.SwitchedOff("on") && !ComputeResources.SwitchedOff(""),
            "IDRAK_PACKED_CACHE: 0, off, false and no turn it off; anything else keeps it on");
        string folder = TempFolder("packed-model"), cache = TempFolder("packed-cache");
        bool before = ComputeResources.PackedWeightCache;
        try
        {
            var spec = TinyLlamaSpec() with { TieEmbeddings = true };
            PretrainedModel.Create(folder, spec, "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 9 }).Dispose();
            using var ids = Tensor.From([1f, 5f, 9f, 3f], [1, 4], device);

            // The packed bits of every layer and the logits of a load, with what its cache did.
            (List<(float[] Packed, float[] Scales)> Bits, float[] Logits, PackedWeightCacheUse? Use) Load(bool int4, bool? keep = true, string? where = null)
            {
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Int4 = int4, Int8 = !int4, PackedCache = keep, PackedCacheFolder = where ?? cache });
                var bits = model.Network.Descendants().OfType<Linear>().Select(l => l.KernelWeight!).Select(w => (w.PackedValues.ToArray(), w.ScaleValues?.ToArray() ?? [])).ToList();
                using (Autograd.NoGrad())
                {
                    return (bits, model.Network.Forward(ids).ToArray(), model.PackedCache);
                }
            }

            static bool Same(List<(float[] Packed, float[] Scales)> a, List<(float[] Packed, float[] Scales)> b) =>
                a.Count == b.Count && a.Zip(b).All(p => p.First.Packed.AsSpan().SequenceEqual(p.Second.Packed) && p.First.Scales.AsSpan().SequenceEqual(p.Second.Scales));

            static bool SameLogits(float[] a, float[] b) => a.Select(BitConverter.SingleToInt32Bits).SequenceEqual(b.Select(BitConverter.SingleToInt32Bits));

            int Entries(string format) => Directory.Exists(cache) ? Directory.EnumerateFiles(cache, "*.ipw", SearchOption.AllDirectories).Count(f => Path.GetFileName(Path.GetDirectoryName(f)) == format) : 0;

            foreach (bool int4 in new[] { true, false })
            {
                string format = int4 ? "int4" : "int8";
                var (packedBits, packedLogits, _) = Load(int4, keep: false);
                var cold = Load(int4);
                int layers = cold.Bits.Count;
                Check(cold.Use is { Read: 0 } coldUse && coldUse.Written == layers && Entries(format) == layers,
                    $"{format} cold: every packed layer made and kept ({cold.Use}; {layers} layers, {Entries(format)} entries)");
                var warm = Load(int4);
                Check(warm.Use is { Written: 0 } warmUse && warmUse.Read == layers, $"{format} warm: every packed layer read back ({warm.Use})");
                Check(Same(packedBits, cold.Bits) && Same(packedBits, warm.Bits), $"{format}: the packed bits of cold and warm loads are the uncached load's");
                Check(SameLogits(packedLogits, cold.Logits) && SameLogits(packedLogits, warm.Logits), $"{format}: the logits of cold and warm loads are the uncached load's, bit for bit");
            }

            Check(Load(int4: true).Use is { Read: > 0, Written: 0 }, "int4 entries stay beside int8's: an int8 load does not prune them");

            // The checkpoint's file changes (its time of change): every entry is made again, written over in place.
            int count = Entries("int4");
            File.SetLastWriteTimeUtc(Path.Combine(folder, "model.safetensors"), DateTime.UtcNow.AddMinutes(1));
            var changed = Load(int4: true);
            Check(changed.Use is { Read: 0 } changedUse && changedUse.Written == count && Entries("int4") == count, $"a changed file makes every entry again ({changed.Use}, {Entries("int4")} entries)");

            // A broken entry is a miss, made again; a stray entry is pruned; a recent temporary file stays.
            string int4Folder = Directory.EnumerateDirectories(cache, "int4", SearchOption.AllDirectories).Single();
            string broken = Directory.EnumerateFiles(int4Folder, "*.ipw").First();
            File.WriteAllBytes(broken, [1, 2, 3]);
            string stray = Path.Combine(int4Folder, "0000.ipw"), temporary = Path.Combine(int4Folder, "0001.ipw.1.1.tmp");
            File.WriteAllBytes(stray, [0]);
            File.WriteAllBytes(temporary, [0]);
            var repaired = Load(int4: true);
            Check(repaired.Use is { Written: 1 } repairedUse && repairedUse.Read == count - 1 && SameLogits(repaired.Logits, changed.Logits),
                $"a broken entry is packed again, the rest read ({repaired.Use})");
            Check(!File.Exists(stray) && File.Exists(temporary), "a stray entry is pruned; a temporary file under an hour old stays");

            // Off: by the load's option, or by the library's setting when the load sets none; no file is made.
            string unused = Path.Combine(cache, "unused");
            Check(Load(int4: true, keep: false, where: unused).Use is null && !Directory.Exists(unused), "PackedCache = false keeps nothing");
            ComputeResources.PackedWeightCache = false;
            Check(Load(int4: true, keep: null, where: unused).Use is null && !Directory.Exists(unused), "ComputeResources.PackedWeightCache = false keeps nothing");
            ComputeResources.PackedWeightCache = true;
            Check(Load(int4: true, keep: null).Use is { Read: > 0 }, "the library's setting on: the load reads its entries");
            using (var halves = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, BFloat16 = true, PackedCacheFolder = unused }))
            {
                Check(halves.PackedCache is null && !Directory.Exists(unused), "bfloat16 weights have no layout: nothing kept");
            }

            // A model whose folder is gone: its entries are removed by the next process's first load from that cache.
            string gone = TempFolder("packed-gone");
            PretrainedModel.Create(gone, spec, "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 10 }).Dispose();
            PackedWeightCacheUse? goneUse;
            using (var goneModel = PretrainedModel.Load(gone, new PretrainedOptions { Device = device, Int4 = true, PackedCacheFolder = cache }))
            {
                goneUse = goneModel.PackedCache;
            }

            Directory.Delete(gone, recursive: true);
            Check(goneUse is { Written: > 0 } && Directory.Exists(goneUse.Folder), $"the gone model's entries were kept ({goneUse})");
            Idrak.Models.PackedWeightCache.SweepGone(cache);
            Check(goneUse is not null && !Directory.Exists(goneUse.Folder) && Load(int4: true).Use is { Read: > 0 }, "a gone model's entries are swept; a present model's stay");
        }
        finally
        {
            ComputeResources.PackedWeightCache = before;
            Directory.Delete(folder, recursive: true);
            Directory.Delete(cache, recursive: true);
        }
    }
}
