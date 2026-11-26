// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Models.Abstractions;

// Plan 37, W0 and W1: what crosses the bus per token (OffloadTraffic), and the plans of a decoder split between the device
// and the CPU (WeightPlacements).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] OffloadTelemetryGroup =
    [
        ("offload telemetry: the counts add up, a snapshot's difference is what crossed since, the line and the JSON give per-token figures", OffloadTrafficCounts),
    ];

    private static readonly (string Name, Action<Device> Run)[] WeightPlacementGroup =
    [
        ("weight placement: device, split, stream and auto plan from the request (the layers the free memory holds beside the fixed weights, the working memory and each layer's cache; the user's count; stream refused where the device cannot and the model does not fit; auto keeps a model that fits on the device); device layers parse", PlacementPlans),
        ("weight placement: layers follow the free memory at step boundaries (more on the device when it frees, fewer under pressure, a user's count kept)", LayersFollowMemory),
    ];

    private static void OffloadTrafficCounts(Device device)
    {
        _ = device;
        var before = OffloadTraffic.Read();
        OffloadTraffic.CountStaged(100);
        OffloadTraffic.CountPrefetched(50);
        OffloadTraffic.CountMapped(25);
        OffloadTraffic.CountLayer("", staged: true, mapped: false);
        OffloadTraffic.CountLayer("Linear up", staged: false, mapped: true);
        OffloadTraffic.CountBoundary(16);
        var counted = OffloadTraffic.Read().Since(before);
        Check(counted is { StagedBytes: 100, PrefetchedBytes: 50, MappedBytes: 25, BoundaryBytes: 16, BoundaryCrossings: 1, LayersStaged: 1, LayersMapped: 1, BusBytes: 191 }
              && counted.MappedLayers.Count == 1 && counted.MappedLayers["Linear up"] == 1 && !counted.IsEmpty, $"the counts: {counted}");
        var json = counted.ToJson(2);
        string line = counted.Describe(2);
        Check((long)json["bytes"]! == 191 && (long)json["bytes_per_token"]! == 95 && (double)json["layers_staged_per_token"]! == 0.5 && (long)json["mapped_layers"]!["Linear up"]! == 1
              && line.StartsWith("bus: 191 B, 95 B a token", StringComparison.Ordinal) && line.Contains("Linear up ×1", StringComparison.Ordinal), $"per token: {line} {json}");
        Check(OffloadTraffic.Read().Since(OffloadTraffic.Read()).IsEmpty, "nothing since a snapshot of now");
    }

    private static void PlacementPlans(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(PlacementPlans)))
        {
            return;
        }

        var gpu = Device.Cpu;                                            // the plans read the request's figures, not the device's
        WeightPlacementRequest Request(long? free, int? asked = null, bool stream = false) =>
            new(gpu, 10, LayerBytes: 100, FixedBytes: 50, LayerCacheBytes: 10, WorkingBytes: 40, free, asked, stream);
        Check(Request(50 + 40 + 3 * 110 + 109).LayersThatFit == 3 && Request(10).LayersThatFit == 0 && Request(null).LayersThatFit == 10 && Request(5000).Fits,
            "the layers that fit: room after the fixed weights and the working memory, a layer and its cache each");
        var device_ = new WeightPlacements.DevicePlacement().Place(Request(200));
        var split = new WeightPlacements.SplitPlacement().Place(Request(50 + 40 + 4 * 110));
        var asked = new WeightPlacements.SplitPlacement().Place(Request(50 + 40 + 4 * 110, asked: 7));
        Check(device_ is { DeviceLayers: 10, Stream: false, Split: false } && split is { DeviceLayers: 4, Layers: 10, Split: true } && asked is { DeviceLayers: 7, Split: true }
              && split.ToString().StartsWith("split: 4 of 10 layers on cpu, 6 computed by the CPU", StringComparison.Ordinal),
            $"device all; split as the memory holds ({split}); split as asked ({asked})");
        var streamed = new WeightPlacements.StreamPlacement().Place(Request(300, stream: true));
        Check(streamed is { Stream: true, DeviceLayers: 10 } && new WeightPlacements.StreamPlacement().Place(Request(5000)) is { Stream: false, DeviceLayers: 10 }, $"stream: {streamed}");
        bool refused = false;
        try
        {
            new WeightPlacements.StreamPlacement().Place(Request(300));
        }
        catch (NotSupportedException)
        {
            refused = true;
        }

        Check(refused, "stream where the device cannot and the model does not fit: refused");
        Check(new WeightPlacements.AutoPlacement().Place(Request(5000)) is { DeviceLayers: 10, Split: false, Stream: false }
              && new WeightPlacements.AutoPlacement().Place(Request(300)) is { DeviceLayers: 1, Split: true }
              && new WeightPlacements.AutoPlacement().Place(Request(5000, asked: 3)) is { DeviceLayers: 3, Split: true },
            "auto: the device when it fits, a split where the device cannot stream, the user's count when given");
        bool bad = false;
        try
        {
            WeightPlacements.ParseDeviceLayers("many", "--device-layers");
        }
        catch (ArgumentException e)
        {
            bad = e.Message.Contains("--device-layers", StringComparison.Ordinal);
        }

        Check(WeightPlacements.ParseDeviceLayers("12", "x") == 12 && WeightPlacements.ParseDeviceLayers("auto", "x") is null && WeightPlacements.ParseDeviceLayers(null, "x") is null && bad,
            "device layers: a count, auto or nothing; anything else names where it came from");

        // Each placement states its own name; a person's or a variable's text is matched against them where it comes in, and
        // with none the default (auto, or IDRAK_WEIGHT_PLACEMENT's) is reached directly.
        Check(WeightPlacements.All.Select(p => p.Name).SequenceEqual([WeightPlacements.AllOnDevice, WeightPlacements.Split, WeightPlacements.Stream, WeightPlacements.Auto])
              && WeightPlacements.Named("SPLIT", "x") is WeightPlacements.SplitPlacement && WeightPlacements.Named("stream ", "x") is WeightPlacements.StreamPlacement
              && (Environment.GetEnvironmentVariable(WeightPlacements.PlacementVariable) is { Length: > 0 } || WeightPlacements.Default is WeightPlacements.AutoPlacement),
            "the placements by their own names, auto by default");
        string unnamed = "";
        try
        {
            WeightPlacements.Named("half", WeightPlacements.PlacementVariable);
        }
        catch (ArgumentException e)
        {
            unnamed = e.Message;
        }

        Check(unnamed.Contains(WeightPlacements.PlacementVariable, StringComparison.Ordinal) && unnamed.Contains("device, split, stream, auto", StringComparison.Ordinal),
            $"a name no placement states refused naming where it came from and theirs: {unnamed}");
    }

    private static void LayersFollowMemory(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(LayersFollowMemory)))
        {
            return;
        }

        // Follow: as many more as the room past the working memory holds; under pressure as many fewer as give it back.
        var r = new WeightPlacementRequest(Device.Cpu, 10, LayerBytes: 100, FixedBytes: 50, LayerCacheBytes: 10, WorkingBytes: 40, AvailableBytes: 40 + 2 * 110 + 5, DeviceLayers: null, CanStream: false);
        Check(WeightPlacements.Follow(r, 3) == 5 && WeightPlacements.Follow(r with { AvailableBytes = 0 }, 3) == 2 && WeightPlacements.Follow(r with { AvailableBytes = 100 }, 3) == 3
              && WeightPlacements.Follow(r with { AvailableBytes = 0 }, 0) == 0 && WeightPlacements.Follow(r with { AvailableBytes = long.MaxValue / 4 }, 3) == 10
              && WeightPlacements.Follow(r with { DeviceLayers = 4 }, 3) == 3 && WeightPlacements.Follow(r with { AvailableBytes = null }, 3) == 3
              && ((IWeightPlacement)new WeightPlacements.DevicePlacement()).Adjust(r, 3) == 3 && new WeightPlacements.SplitPlacement().Adjust(r, 3) == 5 && new WeightPlacements.AutoPlacement().Adjust(r, 3) == 5,
            "grow by what the room holds, shrink by what the pressure asks, keep a user's count; device keeps its layers");
    }
}
