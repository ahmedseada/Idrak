// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// What a weight placement is asked (plan 37, W1): a decoder of <paramref name="Layers"/> layers to place on
/// <paramref name="Device"/>, each layer's weights taking <paramref name="LayerBytes"/>, the parts that stay on the device
/// whatever the split (token table, final norm, head) <paramref name="FixedBytes"/>, each device layer's key/value cache for
/// the run <paramref name="LayerCacheBytes"/>, the run's other working memory <paramref name="WorkingBytes"/>, and what the
/// device reports free now (<paramref name="AvailableBytes"/>: <see cref="Backend.AvailableMemory"/>, within the user's
/// limit, or null when it reports none). Every figure is measured or counted from the model, never a constant of a card.
/// </summary>
/// <param name="Device">The device the model runs on.</param>
/// <param name="Layers">The decoder's layers.</param>
/// <param name="LayerBytes">One layer's weights, in the format they are loaded in.</param>
/// <param name="FixedBytes">The weights that stay on the device in every placement.</param>
/// <param name="LayerCacheBytes">One device layer's key/value cache for the run's context.</param>
/// <param name="WorkingBytes">The run's other working memory on the device (a step's activations and scores, a headroom).</param>
/// <param name="AvailableBytes">The device's free memory now (or the user's budget), or null when it reports none.</param>
/// <param name="DeviceLayers">The layers the user put on the device (<c>IDRAK_DEVICE_LAYERS</c>, <c>--device-layers</c>), or null to size them.</param>
/// <param name="CanStream">Whether the device can keep weights in system memory and stage them (<see cref="IMemoryOffload"/>).</param>
public sealed record WeightPlacementRequest(Device Device, int Layers, long LayerBytes, long FixedBytes, long LayerCacheBytes, long WorkingBytes, long? AvailableBytes,
    int? DeviceLayers, bool CanStream)
{
    /// <summary>
    /// The most layers the device holds beside the fixed weights and the working memory, each with its cache: every one when
    /// it reports no free memory.
    /// </summary>
    public int LayersThatFit
    {
        get
        {
            if (AvailableBytes is not { } free)
            {
                return Layers;
            }

            long room = free - FixedBytes - WorkingBytes;
            long each = Math.Max(1, LayerBytes + LayerCacheBytes);
            return room <= 0 ? 0 : (int)Math.Min(Layers, room / each);
        }
    }

    /// <summary>Whether the whole model fits the device.</summary>
    public bool Fits => LayersThatFit >= Layers;
}

/// <summary>
/// Where a decoder's weights go (<see cref="IWeightPlacement.Place"/>): its first <paramref name="DeviceLayers"/> layers on
/// the device and the rest computed by the CPU from system memory, or (<paramref name="Stream"/>) every layer on the device
/// with what does not fit kept in system memory and staged per pass (<see cref="ComputeResources.OffloadToHostMemory"/>).
/// </summary>
/// <param name="Placement">The placement that chose it.</param>
/// <param name="DeviceLayers">The layers on the device (the rest on the CPU).</param>
/// <param name="Layers">The decoder's layers.</param>
/// <param name="Stream">Whether weights that do not fit are staged from system memory (offload) instead of computed by the CPU.</param>
/// <param name="Reason">Why, for people: what was measured.</param>
public sealed record WeightPlan(string Placement, int DeviceLayers, int Layers, bool Stream, string Reason)
{
    /// <summary>Whether some layers are computed by the CPU.</summary>
    public bool Split => !Stream && DeviceLayers < Layers;

    /// <summary>"split: 20 of 32 layers on cuda:0, 12 on the CPU (…)".</summary>
    public override string ToString() => $"{Placement}: {Reason}";
}

/// <summary>A weight placement (<see cref="WeightPlacements"/>): how a decoder's layers are shared between the device and the CPU.</summary>
public interface IWeightPlacement : IKind
{
    /// <summary>The plan for <paramref name="request"/>.</summary>
    /// <exception cref="NotSupportedException">The placement cannot run on the request's device (stream without offload support).</exception>
    WeightPlan Place(WeightPlacementRequest request);
}

/// <summary>
/// The weight placements, by name (ignoring case): how a decoder whose weights may not fit its device is shared between
/// the device and the CPU (plan 37, W1). The library registers <see cref="AllOnDevice"/> (every layer on the device; a model
/// that does not fit fails as before), <see cref="Split"/> (as Ollama and llama.cpp do: the first layers on the device as
/// far as its measured free memory allows, the rest computed by the CPU from system memory, their key/value caches there
/// too, so only the activations cross the bus, at each boundary), <see cref="Stream"/> (every layer on the device, what
/// does not fit kept in system memory and staged per pass: <see cref="ComputeResources.OffloadToHostMemory"/>) and
/// <see cref="Auto"/> (the default: the device when the model fits, else split or stream, whichever the bus and the CPU
/// measured at load make faster). A load takes the placement from its options, else <see cref="EnvironmentVariable"/>
/// (IDRAK_WEIGHT_PLACEMENT), else <see cref="Auto"/>; a name nobody registered is refused with "not registered".
/// </summary>
public static class WeightPlacements
{
    /// <summary>Every layer on the device.</summary>
    public const string AllOnDevice = "device";

    /// <summary>The first layers on the device, the rest computed by the CPU from system memory.</summary>
    public const string Split = "split";

    /// <summary>Every layer on the device; what does not fit staged from system memory per pass (offload).</summary>
    public const string Stream = "stream";

    /// <summary>The device when the model fits, else split or stream by what the bus and the CPU measure.</summary>
    public const string Auto = "auto";

    /// <summary>The variable naming the placement when an option does not: "IDRAK_WEIGHT_PLACEMENT".</summary>
    public const string EnvironmentVariable = "IDRAK_WEIGHT_PLACEMENT";

    /// <summary>The variable giving the layers on the device (a count, or "auto" to size them): "IDRAK_DEVICE_LAYERS".</summary>
    public const string DeviceLayersVariable = "IDRAK_DEVICE_LAYERS";

    private static readonly SlotTable<string, IWeightPlacement> Table = new(nameof(WeightPlacements), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a placement decides where a model's weights live: falling back to another would place them otherwise than asked");

    static WeightPlacements() => Overrides.AsLibraryDefaults(() =>
    {
        Register(new DevicePlacement());
        Register(new SplitPlacement());
        Register(new StreamPlacement());
        Register(new AutoPlacement());
    });

    /// <summary>
    /// The placement named by <see cref="EnvironmentVariable"/>; when it is not set, <see cref="Stream"/> while
    /// <see cref="ComputeResources.OffloadToHostMemory"/> is on (IDRAK_OFFLOAD=1 alone keeps its staging, which training
    /// needs), else <see cref="Auto"/>.
    /// </summary>
    public static string DefaultName => Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim()
        : ComputeResources.OffloadToHostMemory ? Stream : Auto;

    /// <summary>The layers <see cref="DeviceLayersVariable"/> puts on the device, or null when it is not set or "auto".</summary>
    /// <exception cref="ArgumentException">Neither a count of 0 or more nor "auto".</exception>
    public static int? DefaultDeviceLayers => ParseDeviceLayers(Environment.GetEnvironmentVariable(DeviceLayersVariable), DeviceLayersVariable);

    /// <summary>A device-layers setting: a count of 0 or more, or null for "auto" (or nothing given); <paramref name="where"/> names it in the error.</summary>
    /// <exception cref="ArgumentException">Neither a count of 0 or more nor "auto".</exception>
    public static int? ParseDeviceLayers(string? value, string where)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Trim().Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return int.TryParse(value.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out int layers)
            ? layers
            : throw new ArgumentException($"{where} '{value}': the layers on the device, a count of 0 or more, or auto.");
    }

    /// <summary>Registers <paramref name="placement"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IWeightPlacement placement)
    {
        ArgumentNullException.ThrowIfNull(placement);
        ArgumentException.ThrowIfNullOrWhiteSpace(placement.Name);
        Table.Register(placement.Name, placement, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's placement <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The placement registered as <paramref name="name"/>, or null.</summary>
    public static IWeightPlacement? Find(string name) => Table.Find(name);

    /// <summary>The placement registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static IWeightPlacement Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(WeightPlacements), name, Table.Keys,
            $"Weight placement '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with WeightPlacements.Register.");

    /// <summary>The library's placement <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IWeightPlacement? Default(string name) => Table.Default(name);

    /// <summary>Who registered the placement <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered placement as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(p => $"{p.Name}: {p.Summary}"));

    // The bus and the CPU as measured once per device in this process (rule 75): bytes a second.
    private static readonly ConcurrentDictionary<string, double> BusRates = new(StringComparer.Ordinal);
    private static double s_cpuRate;

    /// <summary>
    /// The bytes a second <paramref name="device"/> takes from system memory, measured once per device and process: a
    /// block of <paramref name="probeBytes"/> uploaded once to warm the path, then timed (the fastest of three).
    /// </summary>
    public static double MeasureBus(Device device, long probeBytes)
    {
        ArgumentNullException.ThrowIfNull(device);
        return BusRates.GetOrAdd(device.ToString(), _ =>
        {
            int count = Probe(probeBytes);
            var values = new float[count];
            var backend = device.Backend;
            var storage = backend.Allocate(count, zeroed: false);
            try
            {
                double best = double.MaxValue;
                for (int i = 0; i < 4; i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    backend.Upload(values, storage);
                    backend.Synchronize();
                    double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
                    if (i > 0)
                    {
                        best = Math.Min(best, seconds);                         // the first warms the path (rule 76)
                    }
                }

                return count * (double)sizeof(float) / Math.Max(best, 1e-9);
            }
            finally
            {
                backend.Return(storage);
            }
        });
    }

    /// <summary>
    /// The bytes of weights a second the CPU reads while it computes a decoding step's product (one row against a weight of
    /// <paramref name="probeBytes"/>), measured once per process: the fastest of three warm runs.
    /// </summary>
    public static double MeasureCpu(long probeBytes)
    {
        double known = Volatile.Read(ref s_cpuRate);
        if (known > 0)
        {
            return known;
        }

        int count = Probe(probeBytes);
        int k = 1024, n = Math.Max(1, count / k);
        var cpu = Device.Cpu;
        using var weight = Tensor.Zeros([k, n], cpu);
        using var row = Tensor.Zeros([1, k], cpu);
        double best = double.MaxValue;
        using (Autograd.NoGrad())
        {
            for (int i = 0; i < 4; i++)
            {
                long start = Stopwatch.GetTimestamp();
                using (row.MatMul(weight))
                {
                }

                double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
                if (i > 0)
                {
                    best = Math.Min(best, seconds);                             // the first warms the path (rule 76)
                }
            }
        }

        double rate = (double)k * n * sizeof(float) / Math.Max(best, 1e-9);
        Volatile.Write(ref s_cpuRate, rate);
        return rate;
    }

    // A probe's floats: a layer's bytes, within what a quick measurement can move (a megabyte to 64 MB).
    private static int Probe(long bytes) => (int)(Math.Clamp(bytes, 1L << 20, 64L << 20) / sizeof(float));

    // Gigabytes with two decimals, for the reasons.
    internal static string Gigabytes(long bytes) => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e9:0.##} GB");

    private static string Rate(double bytesPerSecond) => string.Create(CultureInfo.InvariantCulture, $"{bytesPerSecond / 1e9:0.#} GB/s");

    // The device layers a split takes: the user's count (within the model), else what fits.
    private static WeightPlan SplitPlan(WeightPlacementRequest r, string placement, string? why = null)
    {
        int layers = Math.Clamp(r.DeviceLayers ?? r.LayersThatFit, 0, r.Layers);
        string sized = r.DeviceLayers is not null ? "as asked" : r.AvailableBytes is { } free
            ? $"as {Gigabytes(free)} free holds, {Gigabytes(r.LayerBytes)} a layer" : "the device reports no free memory";
        return new WeightPlan(placement, layers, r.Layers, false,
            $"{layers} of {r.Layers} layers on {r.Device}, {r.Layers - layers} computed by the CPU from system memory ({sized}{(why is null ? "" : "; " + why)})");
    }

    private sealed class DevicePlacement : IWeightPlacement
    {
        public string Name => AllOnDevice;

        public string Summary => "every layer on the device (a model that does not fit fails as before)";

        public WeightPlan Place(WeightPlacementRequest request) =>
            new(Name, request.Layers, request.Layers, false, $"every layer on {request.Device}");
    }

    private sealed class SplitPlacement : IWeightPlacement
    {
        public string Name => Split;

        public string Summary => "the first layers on the device as its free memory allows (or IDRAK_DEVICE_LAYERS), the rest computed by the CPU from system memory";

        public WeightPlan Place(WeightPlacementRequest request) => SplitPlan(request, Name);
    }

    private sealed class StreamPlacement : IWeightPlacement
    {
        public string Name => Stream;

        public string Summary => "every layer on the device; what does not fit kept in system memory and staged each pass (offload)";

        public WeightPlan Place(WeightPlacementRequest request) => request.CanStream
            ? new(Name, request.Layers, request.Layers, true,
                $"every layer on {request.Device}, {Math.Max(0, request.Layers - request.LayersThatFit)} of them kept in system memory and staged each pass")
            : request.Fits
                ? new(Name, request.Layers, request.Layers, false, $"every layer on {request.Device} (it fits; the device cannot stream weights)")
                : throw new NotSupportedException($"{request.Device} cannot keep weights in system memory (no offload support) and the model does not fit it: use the '{Split}' placement there.");
    }

    private sealed class AutoPlacement : IWeightPlacement
    {
        public string Name => Auto;

        public string Summary => "the device when the model fits; else split or stream, whichever the bus and the CPU measured at load make faster";

        public WeightPlan Place(WeightPlacementRequest request)
        {
            if (request.DeviceLayers is not null)
            {
                return SplitPlan(request, Name);
            }

            if (request.Fits)
            {
                return new(Name, request.Layers, request.Layers, false, $"every layer on {request.Device} (it fits)");
            }

            if (!request.CanStream)
            {
                return SplitPlan(request, Name, "the device cannot stream weights");
            }

            // The layers that do not fit, a token's reading of their weights: over the bus when streamed, from system memory
            // by the CPU when split. Whichever is faster here wins.
            long probe = Math.Max(1, request.LayerBytes);
            double bus = MeasureBus(request.Device, probe), cpu = MeasureCpu(probe);
            string measured = $"bus {Rate(bus)}, CPU {Rate(cpu)} measured";
            return cpu >= bus
                ? SplitPlan(request, Name, measured)
                : new(Name, request.Layers, request.Layers, true,
                    $"every layer on {request.Device}, {request.Layers - request.LayersThatFit} kept in system memory and staged each pass ({measured})");
        }
    }
}
