// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Devices;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 37, W0 and W1: what crosses the bus per token (OffloadTraffic, printed by idrak run), and a decoder split between the
// device and the CPU (WeightPlacements), whose results are the all-device run's.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] OffloadTelemetryGroup =
    [
        ("offload telemetry: the counts add up, a snapshot's difference is what crossed since, the line and the JSON give per-token figures; a model on one device moves nothing across the bus", OffloadTrafficCounts),
        ("offload telemetry: a split model's activations are counted at each crossing (both ways, a step's positions × width × 4 bytes), no weight crosses; idrak run -v -j prints the placement and a \"bus\" object", SplitTrafficCounted),
    ];

    private static readonly (string Name, Action<Device> Run)[] WeightPlacementGroup =
    [
        ("weight placement: device, split, stream and auto plan from the request (the layers the free memory holds beside the fixed weights, the working memory and each layer's cache; the user's count; stream refused where the device cannot and the model does not fit; auto keeps a model that fits on the device); the names are registered, an unknown one is not; device layers parse", PlacementPlans),
        ("weight placement: a decoder split between the device and the CPU gives the all-device and the all-CPU logits (a pass, a prompt and cached steps, a truncated context), its later layers' weights and caches on the CPU; a free memory that holds part of it splits it at load (auto) instead of failing; on the CPU nothing is split", SplitMatchesOneDevice),
    ];

    private static string PlacementModel => PretrainData("tiny-llama");

    private static void OffloadTrafficCounts(Device device)
    {
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

        // A model on one device: nothing crosses while it decodes.
        using var model = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device });
        var start = OffloadTraffic.Read();
        Decode(model, truncate: false);
        Check(OffloadTraffic.Read().Since(start).IsEmpty, $"{device}: a model on one device moves nothing across the bus ({OffloadTraffic.Read().Since(start).Describe(4)})");
    }

    private static void SplitTrafficCounted(Device device)
    {
        int dim = TinyLlamaSpec().Dim;
        if (device.Type != DeviceType.Cpu)
        {
            using var split = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Split, DeviceLayers = 1 });
            var start = OffloadTraffic.Read();
            Decode(split, truncate: false);                              // a plain pass and a prompt of 4 positions, then 3 steps of one
            var crossed = OffloadTraffic.Read().Since(start);
            long expected = 2L * (4 + 4 + 3) * dim * sizeof(float);      // to the CPU and back, at every position
            Check(crossed is { StagedBytes: 0, PrefetchedBytes: 0, MappedBytes: 0, BoundaryCrossings: 10 } && crossed.BoundaryBytes == expected,
                $"{device}: activations only, {expected} bytes in 10 crossings ({crossed.Describe(5)})");
        }

        var (code, text, error) = RunIdrakOn(device, null, "run", CliModel, "Hello", "--max-tokens", "3", "--temperature", "0", "-j", "-v", "--device-layers", "1");
        Check(code == 0, $"idrak run on {device}: {error}");
        var bus = JsonOf(text, "run -j -v")["bus"] as JsonObject;
        Check(bus is not null && (long)bus["staged_bytes"]! == 0 && (long)bus["mapped_bytes"]! == 0, $"a \"bus\" object, no weight staged or read in place: {text}");
        if (device.Type == DeviceType.Cpu)
        {
            Check((long)bus!["bytes"]! == 0 && !error.Contains("placement", StringComparison.Ordinal), $"on the CPU nothing is split and nothing crosses: {bus} {error}");
        }
        else
        {
            Check((long)bus!["boundary_bytes"]! > 0 && (long)bus["boundary_crossings"]! % 2 == 0 && error.Contains("placement auto: 1 of 2 layers on", StringComparison.Ordinal),
                $"{device}: the split said and its activations counted: {bus} {error}");
        }
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
        var device_ = WeightPlacements.Get("device").Place(Request(200));
        var split = WeightPlacements.Get("SPLIT").Place(Request(50 + 40 + 4 * 110));
        var asked = WeightPlacements.Get("split").Place(Request(50 + 40 + 4 * 110, asked: 7));
        Check(device_ is { DeviceLayers: 10, Stream: false, Split: false } && split is { DeviceLayers: 4, Layers: 10, Split: true } && asked is { DeviceLayers: 7, Split: true }
              && split.ToString().StartsWith("split: 4 of 10 layers on cpu, 6 computed by the CPU", StringComparison.Ordinal),
            $"device all; split as the memory holds ({split}); split as asked ({asked})");
        var streamed = WeightPlacements.Get("stream").Place(Request(300, stream: true));
        Check(streamed is { Stream: true, DeviceLayers: 10 } && WeightPlacements.Get("stream").Place(Request(5000)) is { Stream: false, DeviceLayers: 10 }, $"stream: {streamed}");
        bool refused = false;
        try
        {
            WeightPlacements.Get("stream").Place(Request(300));
        }
        catch (NotSupportedException)
        {
            refused = true;
        }

        Check(refused, "stream where the device cannot and the model does not fit: refused");
        Check(WeightPlacements.Get("auto").Place(Request(5000)) is { DeviceLayers: 10, Split: false, Stream: false }
              && WeightPlacements.Get("auto").Place(Request(300)) is { DeviceLayers: 1, Split: true }
              && WeightPlacements.Get("auto").Place(Request(5000, asked: 3)) is { DeviceLayers: 3, Split: true },
            "auto: the device when it fits, a split where the device cannot stream, the user's count when given");
        bool unknown = false;
        try
        {
            WeightPlacements.Get("nowhere");
        }
        catch (NotSupportedException e)
        {
            unknown = e.Message.Contains("not registered", StringComparison.Ordinal);
        }

        Check(unknown && WeightPlacements.Names.Order(StringComparer.Ordinal).SequenceEqual(["auto", "device", "split", "stream"]), "the names; an unknown one is not registered");
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
    }

    private static void SplitMatchesOneDevice(Device device)
    {
        using var whole = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.AllOnDevice });
        if (device.Type == DeviceType.Cpu)
        {
            using var cpu = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Split, DeviceLayers = 1 });
            Check(cpu.Placement is null && cpu.Network.Descendants().OfType<DecoderBlock>().All(b => b.PlacedOn is null)
                  && Decode(cpu, truncate: true).Zip(Decode(whole, truncate: true)).All(p => p.First.SequenceEqual(p.Second)),
                "on the CPU nothing is split: the same logits");
            return;
        }

        using var host = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = Device.Cpu });
        using var split = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Split, DeviceLayers = 1 });
        var blocks = split.Network.Descendants().OfType<DecoderBlock>().ToList();
        Check(split.Placement is { Split: true, DeviceLayers: 1, Layers: 2 } && blocks[0].PlacedOn is null && blocks[1].PlacedOn == Device.Cpu
              && blocks[0].Parameters().All(p => p.Device == device) && blocks[1].Parameters().All(p => p.Device == Device.Cpu)
              && split.Notes.Any(n => n.StartsWith("placement split: 1 of 2 layers on", StringComparison.Ordinal)),
            $"layer 0 on {device}, layer 1 on the CPU ({split.Placement})");

        var expected = Decode(whole, truncate: true);
        var onCpu = Decode(host, truncate: true);
        var both = Decode(split, truncate: true);
        for (int i = 0; i < expected.Count; i++)
        {
            AssertClose(expected[i], both[i], 2e-3f, $"pass {i}: split against all on {device}");
            AssertClose(onCpu[i], both[i], 2e-3f, $"pass {i}: split against all on the CPU");
        }

        // The caches: layer 1's on the CPU (the context's counterpart there), layer 0's on the device.
        using (var context = new DecodingContext(device, 1, 32))
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            split.Network.ForwardCached(Tensor.From([1f, 5f, 9f], [1, 3], device), context);
            var mirror = context.On(Device.Cpu);
            Check(mirror.Device == Device.Cpu && mirror.Length == 3 && context.Length == 3 && mirror.CacheBytes > 0 && context.CacheBytes > 0 && !context.Recordable,
                $"the CPU layer's cache in system memory, in step ({mirror.Length} positions, {mirror.CacheBytes} bytes); the step is not recordable");
        }

        // Auto with a free memory that holds the fixed weights, the working memory and one layer: split at load.
        var spec = whole.Spec;
        int contextLength = Math.Min(AutoWeights.DefaultContext, spec.MaxPositions);
        var request = AutoWeights.PlacementRequest(spec, "float32", contextLength, device, null, null);
        long budget = request.FixedBytes + request.WorkingBytes + request.LayerBytes + request.LayerCacheBytes + request.LayerBytes / 2;
        using var auto = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Auto, AutoWeightsBudget = budget });
        Check(auto.Placement is { Split: true, DeviceLayers: 1 } && auto.Network.Descendants().OfType<DecoderBlock>().Last().PlacedOn == Device.Cpu,
            $"a budget of one layer: auto splits ({auto.Placement})");
        var autoLogits = Decode(auto, truncate: false);
        AssertClose(expected[0], autoLogits[0], 2e-3f, "auto's split: the all-device logits");
    }

    // Logits of a plain pass, a prompt of 4 positions and 3 cached steps; with truncate, the context cut back to the prompt
    // and the first step fed again (its logits must be the same).
    private static List<float[]> Decode(PretrainedModel model, bool truncate)
    {
        var device = model.Device;
        var results = new List<float[]>();
        using var context = new DecodingContext(device, 1, 32);
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            results.Add(model.Network.Forward(Tensor.From([1f, 5f, 9f, 3f], [1, 4], device)).ToArray());
            results.Add(model.Network.ForwardCached(Tensor.From([1f, 5f, 9f, 3f], [1, 4], device), context).ToArray());
            foreach (float id in new[] { 7f, 2f, 11f })
            {
                results.Add(model.Network.ForwardCached(Tensor.From([id], [1, 1], device), context).ToArray());
            }

            if (truncate)
            {
                context.Truncate(4);
                results.Add(model.Network.ForwardCached(Tensor.From([7f], [1, 1], device), context).ToArray());
                Check(results[^1].SequenceEqual(results[2]), $"{device}: the step fed again after a truncation gives its logits");
            }
        }

        return results;
    }
}
