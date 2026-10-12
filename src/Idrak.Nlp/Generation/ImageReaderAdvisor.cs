// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Diagnostics;
using Idrak.Layers;

namespace Idrak.Generation;

/// <summary>
/// The library's advisor for an <see cref="ImageReader"/> ("image reader" in <see cref="SettingAdvisors"/>): the encoder's
/// and the prompt's bfloat16 products where the device runs them on matrix units, and from the first read's figures the
/// pages that fit together, a bfloat16 cache when it fits more, and the weights' format the device has room for. A
/// figure the device does not report decides nothing.
/// </summary>
internal sealed class ImageReaderAdvisor : ISettingAdvisor
{
    /// <summary>The advisor's name: "image reader".</summary>
    public const string AdvisorName = "image reader";

    public string Name => AdvisorName;

    public string Summary => "an image reader: bfloat16 encoder and prompt where the device has matrix units; from the first read, the pages that fit together, a bfloat16 cache, the weights' format";

    public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation)
    {
        ArgumentNullException.ThrowIfNull(situation);
        if (situation.Subject is not ImageReader reader)
        {
            yield break;
        }

        var backend = situation.Device.Backend;
        string device = situation.Device.ToString();
        var options = reader.Options;
        if (backend.TensorCoresUnavailable() is null)
        {
            if (options.EncoderPrecision is null)
            {
                yield return Suggestion(nameof(ImageReaderOptions.EncoderPrecision), nameof(MatMulPrecision.BFloat16),
                    "the device runs bfloat16 products on matrix units: the encoder several times faster, its features within bfloat16's rounding", device);
            }

            if (options.PrefillPrecision is null)
            {
                yield return Suggestion(nameof(ImageReaderOptions.PrefillPrecision), nameof(MatMulPrecision.BFloat16),
                    "the device runs bfloat16 products on matrix units: the prompt's pass faster, the answer's steps unchanged", device);
            }
        }

        if (situation.Measured is not { Items: > 0, PeakAbove: > 0 } measured || backend.Hardware.MemoryBytes is not { } total)
        {
            yield break;
        }

        // What a page takes (its cache, activations, the encoder's work), and the room beyond the headroom kept free.
        long perPage = Math.Max(1, measured.PeakAbove / measured.Items);
        long room = measured.Available - (long)(Math.Clamp(situation.Headroom, 0, 1) * total);
        int fit = room >= 0 ? 1 + (int)Math.Min(1024, room / perPage) : 0;
        if (fit >= 2 && options.MaxBatch != fit)
        {
            yield return Suggestion(nameof(ImageReaderOptions.MaxBatch), Text(fit),
                $"{fit} pages fit together (ImageReader.ReadBatch): about {Mb(perPage)} MB a page, {Mb(room)} MB free beyond the headroom", device);
        }

        var model = reader.Model;
        if (room >= 0 && options.CacheFormat == KeyValueFormat.Float32 && measured.Positions > 0)
        {
            long cache = ModelMemory.KeyValueCacheBytes(model.Spec, measured.Positions, KeyValueLayouts.For(KeyValueFormat.Float32));
            int fitHalf = 1 + (int)Math.Min(1024, room / Math.Max(1, perPage - cache / 2));
            if (fitHalf > fit)
            {
                yield return Suggestion(nameof(ImageReaderOptions.CacheFormat), nameof(KeyValueFormat.BFloat16),
                    $"half the cache's {Mb(cache)} MB a page: {fitHalf} pages fit together instead of {fit} (the answers can change)", device);
            }
        }

        // The weights' format: bfloat16 (the checkpoint's own values for most) where the device has room for it, int8 where
        // it has too little beside a page.
        var projections = model.Network.Descendants().OfType<Linear>().Where(l => l.Name != "head").ToList();
        long weights = projections.Sum(l => (long)l.InFeatures * l.OutFeatures);
        if (weights == 0)
        {
            yield break;
        }

        long held = projections.SelectMany(l => l.Parameters().Concat(l.Buffers())).Select(t => t.Storage).Distinct().Sum(s => 4L * s.Length);
        string? format = projections.GroupBy(l => l.PackedWeight?.Name).MaxBy(g => g.Count())?.Key;
        if (format is "int8" or "int4")
        {
            long more = 2 * weights - held;
            if (room - more >= perPage)
            {
                yield return new("PretrainedOptions.BFloat16", "true",
                    $"bfloat16 weights fit: about {Mb(more)} MB more than {format}, a page's {Mb(perPage)} MB still free beyond the headroom", device, AdvisorName);
            }
        }
        else if (room < 0)
        {
            yield return new("PretrainedOptions.Int8", "true",
                $"the headroom is short by {Mb(-room)} MB beside a page: int8 weights hold about {Mb(held - weights)} MB less than {format ?? "float32"}", device, AdvisorName);
        }
    }

    private static SettingSuggestion Suggestion(string option, string value, string reason, string device) =>
        new(nameof(ImageReaderOptions) + "." + option, value, reason, device, AdvisorName);

    private static string Text(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static string Mb(long bytes) => (bytes / 1048576.0).ToString("F0", CultureInfo.InvariantCulture);
}
