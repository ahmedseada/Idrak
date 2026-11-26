// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Plugins.Gemma;

// Gemma 3's pan and scan through the Gemma 3 plug-in (plan 11, "Pan and scan"): the crop rule as transformers'
// pan_and_scan has it, and its settings as vision options.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] PanAndScanGroup =
    [
        ("pan and scan: the crop rule, the vision options (unknown keys name Gemma 3's, values checked)", PanAndScanSettings),
    ];

    internal static void PanAndScanSettings(Device device)
    {
        _ = device;
        // The rule (transformers' pan_and_scan), Gemma3Processor's defaults: 256-pixel crops, 4 at most, ratio 1.2.
        var on = Gemma3PanAndScan.Default with { Enabled = true };
        Check(Gemma3PanAndScan.Default.Crops(1000, 700).Count == 0, "off by default");
        var scan = on.Crops(1000, 700);                    // a 700 x 1000 page: ratio 1.43 rounds to 1, at least 2 crops
        Check(scan is [(0, 0, 500, 700), (500, 0, 500, 700)], $"a 700 x 1000 page: {string.Join("; ", scan)}");
        Check(on.Crops(3508, 2480).Count == 2 && on.Crops(896, 896).Count == 0 && on.Crops(1100, 1000).Count == 0, "an A4 scan, a square, ratio 1.1");
        Check(on.Crops(1000, 4000) is { Count: 4 } wide && wide.All(c => c.Width == 1000 && c.Height == 1000), "ratio 4: four crops across");
        Check(on.Crops(1000, 9000).Count == 4, "ratio 9: no more than 4");
        Check(on.Crops(300, 500).Count == 0, "crops under 256 pixels: none");
        Check(on.Crops(1001, 600) is [(0, 0, 501, 600), (501, 0, 500, 600)], "the last crop cut at the edge");
        Check((on with { MaxCrops = 1 }).Crops(1000, 3000) is [(0, 0, 1000, 3000)], "max 1: the whole image as its one crop, as transformers does");

        // Options: transformers' names (and pan_and_scan for the switch); anything else is refused, naming Gemma 3's keys.
        var parsed = Gemma3PanAndScan.Default.With(VisionOptions.Parse(["pan_and_scan=true", "pan_and_scan_max_num_crops=2", "pan_and_scan_min_ratio_to_activate=1.5"]));
        Check(parsed is { Enabled: true, MaxCrops: 2, MinCropSize: 256, MinRatio: 1.5 }, $"parsed: {parsed}");
        foreach (var (options, words) in new[]
        {
            (VisionOptions.Parse(["crops=4"]), "do_pan_and_scan, pan_and_scan, pan_and_scan_min_crop_size"),
            (VisionOptions.Parse(["do_pan_and_scan=maybe"]), "true or false"),
            (VisionOptions.Parse(["pan_and_scan_max_num_crops=0"]), "at least 1"),
            (VisionOptions.Parse(["pan_and_scan_min_crop_size=1.5"]), "whole number"),
            (VisionOptions.Parse(["do_pan_and_scan=true", "pan_and_scan=false"]), "disagree"),
        })
        {
            try
            {
                Gemma3PanAndScan.Default.With(options);
                Check(false, $"{options} is refused");
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Contains(words, StringComparison.Ordinal), $"{options}: {e.Message}");
            }
        }
    }
}
