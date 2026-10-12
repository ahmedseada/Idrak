// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The library's advisor for a loaded <see cref="PretrainedModel"/> ("pretrained model" in <see cref="SettingAdvisors"/>),
/// asked when <see cref="PretrainedModel.Load"/> ends: bfloat16 weights for float32 ones read from a bfloat16 checkpoint
/// (the same values in half the memory), and a smaller weights' format when the device has nothing left beyond its headroom.
/// A figure the device does not report decides nothing.
/// </summary>
internal sealed class PretrainedModelAdvisor : ISettingAdvisor
{
    /// <summary>The advisor's name: "pretrained model".</summary>
    public const string AdvisorName = "pretrained model";

    public string Name => AdvisorName;

    public string Summary => "a loaded model: bfloat16 weights for a bfloat16 checkpoint read as float32; a smaller weights' format when the device has nothing left beyond its headroom";

    public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation)
    {
        ArgumentNullException.ThrowIfNull(situation);
        if (situation.Subject is not PretrainedModel model)
        {
            yield break;
        }

        var projections = model.Network.Descendants().OfType<Linear>().Where(l => l.Name != "head").ToList();
        long weights = projections.Sum(l => (long)l.InFeatures * l.OutFeatures);
        if (weights == 0)
        {
            yield break;
        }

        string device = situation.Device.ToString();
        long held = projections.SelectMany(l => l.Parameters().Concat(l.Buffers())).Select(t => t.Storage).Distinct().Sum(s => 4L * s.Length);
        string format = projections.GroupBy(l => l.PackedWeight?.Name ?? "float32").MaxBy(g => g.Count())!.Key;
        if (format == "float32" && model.CheckpointFormat == WeightFormat.BFloat16)
        {
            yield return new("PretrainedOptions.BFloat16", "true",
                $"the checkpoint stores bfloat16: the same values in half the projections' memory, about {Mb(held - 2 * weights)} MB less", device, AdvisorName);
        }

        var backend = situation.Device.Backend;
        if (backend.AvailableMemory() is not { } available || backend.Hardware.MemoryBytes is not { } total)
        {
            yield break;
        }

        long room = available - (long)(Math.Clamp(situation.Headroom, 0, 1) * total);
        if (room > 0)
        {
            yield break;
        }

        // Nothing left beyond the headroom after loading: the next smaller format, with what it saves.
        var (option, smaller) = format switch
        {
            "int8" => ("PretrainedOptions.Int4", (long)(0.625 * weights)),
            "int4" => ((string?)null, 0L),
            _ => ("PretrainedOptions.Int8", weights),
        };
        if (option is not null && held > smaller)
        {
            yield return new(option, "true",
                $"no room left beyond the headroom after loading ({Mb(-room)} MB short): about {Mb(held - smaller)} MB less than {format}", device, AdvisorName);
        }
    }

    private static string Mb(long bytes) => (bytes / 1048576.0).ToString("F0", CultureInfo.InvariantCulture);
}
