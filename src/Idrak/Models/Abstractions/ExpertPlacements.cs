// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// What an expert placement is asked (plan 37, R1): a mixture-of-experts model whose <paramref name="ExpertLayers"/> layers
/// each hold <paramref name="Experts"/> experts of <paramref name="ExpertBytes"/> (in the format they are loaded in), to
/// run on <paramref name="Device"/>. <paramref name="OtherBytes"/> is everything else the device holds for the run: the
/// other weights, the key/value caches and the working memory. <paramref name="AvailableBytes"/> is what the device reports
/// free now (<see cref="Backend.AvailableMemory"/>), or null when it reports none. <paramref name="CacheBytes"/> is the
/// user's limit on the experts kept on the device, or null to size them from the free memory. Every figure is measured
/// or counted from the model, never a constant of a card.
/// </summary>
/// <param name="Device">The device the model runs on.</param>
/// <param name="ExpertLayers">The mixture-of-experts layers.</param>
/// <param name="Experts">The experts of each such layer.</param>
/// <param name="ExpertsPerToken">The experts each token is routed to.</param>
/// <param name="ExpertBytes">One expert's weights, in the format they are loaded in.</param>
/// <param name="OtherBytes">What else the device holds for the run (the other weights, the caches, the working memory).</param>
/// <param name="AvailableBytes">The device's free memory now (or the user's budget), or null when it reports none.</param>
/// <param name="CacheBytes">The user's limit on the experts' bytes on the device (<see cref="ExpertPlacements.CacheVariable"/>), or null.</param>
/// <param name="ColdOnCpu">Whether a cold expert is computed by the CPU from system memory instead of copied in (<see cref="ExpertPlacements.ColdVariable"/>).</param>
public sealed record ExpertPlacementRequest(Device Device, int ExpertLayers, int Experts, int ExpertsPerToken, long ExpertBytes, long OtherBytes, long? AvailableBytes,
    long? CacheBytes, bool ColdOnCpu = false)
{
    /// <summary>Every expert's bytes.</summary>
    public long ExpertsBytes => (long)ExpertLayers * Experts * ExpertBytes;

    /// <summary>Whether every expert fits on the device beside the rest: always when it reports no free memory.</summary>
    public bool Fits => AvailableBytes is not { } free || OtherBytes + ExpertsBytes <= free;

    /// <summary>
    /// The experts of each layer the device keeps: what the user's limit holds, else what the free memory beside the rest
    /// holds; at least one, at most every one.
    /// </summary>
    public int DeviceExperts
    {
        get
        {
            long room = CacheBytes ?? (AvailableBytes is { } free ? free - OtherBytes : ExpertsBytes);
            long each = Math.Max(1, (long)Math.Max(1, ExpertLayers) * ExpertBytes);
            return (int)Math.Clamp(room / each, 1, Math.Max(1, Experts));
        }
    }
}

/// <summary>
/// Where a mixture-of-experts model's experts live (<see cref="IExpertPlacement.Place"/>): every one on the device, or
/// (<paramref name="Offload"/>) the cold ones in system memory and the <paramref name="DeviceExperts"/> most used of each
/// layer in a cache on the device, a cold one copied in when the router picks it or (<paramref name="ColdOnCpu"/>)
/// computed by the CPU.
/// </summary>
/// <param name="Placement">The placement that chose it.</param>
/// <param name="Offload">Whether cold experts live in system memory.</param>
/// <param name="DeviceExperts">The experts of each layer kept on the device.</param>
/// <param name="ColdOnCpu">Whether a cold expert is computed by the CPU instead of copied in.</param>
/// <param name="Reason">Why, for people: what was measured.</param>
public sealed record ExpertPlan(string Placement, bool Offload, int DeviceExperts, bool ColdOnCpu, string Reason)
{
    /// <summary>"offload: 8 of 64 experts a layer on cuda:0 (…)".</summary>
    public override string ToString() => $"{Placement}: {Reason}";
}

/// <summary>An expert placement (<see cref="ExpertPlacements"/>): where a mixture-of-experts model's experts live.</summary>
public interface IExpertPlacement : IKind
{
    /// <summary>The plan for <paramref name="request"/>.</summary>
    ExpertPlan Place(ExpertPlacementRequest request);
}

/// <summary>
/// The expert placements, by name (ignoring case): where a mixture-of-experts model's experts live (plan 37, R1). The
/// library registers three:
/// - <see cref="Resident"/>: every expert on the device, as before.
/// - <see cref="Offload"/>: the cold experts in system memory and the most used of each layer in a cache on the device.
///   The cache is sized by the measured free memory or <see cref="CacheVariable"/>, and what stays in it is learned from
///   the routing's counts at run time. A cold expert is copied in when the router picks it, or computed by the CPU
///   (<see cref="ColdVariable"/>), as llama.cpp's <c>--cpu-moe</c> does.
/// - <see cref="Auto"/> (the default): resident when the model fits the measured free device memory, else offload.
/// A load takes the placement from its options, else <see cref="EnvironmentVariable"/> (IDRAK_MOE_EXPERTS), else
/// <see cref="Auto"/>; a name nobody registered is refused with "not registered".
/// </summary>
public static class ExpertPlacements
{
    /// <summary>Every expert on the device.</summary>
    public const string Resident = "resident";

    /// <summary>Cold experts in system memory, the most used in a device cache.</summary>
    public const string Offload = "offload";

    /// <summary>Resident when the model fits the device's free memory, else offload.</summary>
    public const string Auto = "auto";

    /// <summary>A cold expert copied into the device cache when the router picks it.</summary>
    public const string ColdCopy = "copy";

    /// <summary>A cold expert computed by the CPU from system memory.</summary>
    public const string ColdCpu = "cpu";

    /// <summary>The variable naming the placement when an option does not: "IDRAK_MOE_EXPERTS".</summary>
    public const string EnvironmentVariable = "IDRAK_MOE_EXPERTS";

    /// <summary>The variable giving the megabytes of experts the device keeps (a limit over the measured free memory): "IDRAK_MOE_DEVICE_EXPERTS_MB".</summary>
    public const string CacheVariable = "IDRAK_MOE_DEVICE_EXPERTS_MB";

    /// <summary>The variable choosing what a cold expert does, <see cref="ColdCopy"/> (the default) or <see cref="ColdCpu"/>: "IDRAK_MOE_COLD".</summary>
    public const string ColdVariable = "IDRAK_MOE_COLD";

    private static readonly SlotTable<string, IExpertPlacement> Table = new(nameof(ExpertPlacements), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a placement decides where a model's experts live: falling back to another would place them otherwise than asked");

    static ExpertPlacements() => Overrides.AsLibraryDefaults(() =>
    {
        Register(new ResidentPlacement());
        Register(new OffloadPlacement());
        Register(new AutoPlacement());
    });

    /// <summary>The placement named by <see cref="EnvironmentVariable"/>, else <see cref="Auto"/>.</summary>
    public static string DefaultName => Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim() : Auto;

    /// <summary>The bytes <see cref="CacheVariable"/> gives the device's experts, or null when it is not set.</summary>
    /// <exception cref="ArgumentException">Not a count of megabytes of 0 or more.</exception>
    public static long? DefaultCacheBytes => ParseCacheMegabytes(Environment.GetEnvironmentVariable(CacheVariable), CacheVariable);

    /// <summary>Whether <see cref="ColdVariable"/> asks cold experts to run on the CPU (off unless it says <see cref="ColdCpu"/>).</summary>
    /// <exception cref="ArgumentException">Neither copy nor cpu.</exception>
    public static bool DefaultColdOnCpu => ParseCold(Environment.GetEnvironmentVariable(ColdVariable), ColdVariable);

    /// <summary>A device experts setting in megabytes: a count of 0 or more, or null when nothing is given; <paramref name="where"/> names it in the error.</summary>
    /// <exception cref="ArgumentException">Not a count of 0 or more.</exception>
    public static long? ParseCacheMegabytes(string? value, string where)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return long.TryParse(value.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out long megabytes) && megabytes <= long.MaxValue >> 20
            ? megabytes << 20
            : throw new ArgumentException($"{where} '{value}': the megabytes of experts kept on the device, a count of 0 or more.");
    }

    /// <summary>A cold-experts setting: true for <see cref="ColdCpu"/>, false for <see cref="ColdCopy"/> or nothing given; <paramref name="where"/> names it in the error.</summary>
    /// <exception cref="ArgumentException">Neither copy nor cpu.</exception>
    public static bool ParseCold(string? value, string where) => value?.Trim().ToLowerInvariant() switch
    {
        null or "" or ColdCopy => false,
        ColdCpu => true,
        _ => throw new ArgumentException($"{where} '{value}': what a cold expert does, {ColdCopy} (copied to the device when picked) or {ColdCpu} (computed by the CPU)."),
    };

    /// <summary>Registers <paramref name="placement"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IExpertPlacement placement)
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
    public static IExpertPlacement? Find(string name) => Table.Find(name);

    /// <summary>The placement registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static IExpertPlacement Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(ExpertPlacements), name, Table.Keys,
            $"Expert placement '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with ExpertPlacements.Register.");

    /// <summary>The library's placement <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IExpertPlacement? Default(string name) => Table.Default(name);

    /// <summary>Who registered the placement <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered placement as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(p => $"{p.Name}: {p.Summary}"));

    /// <summary>The offload plan for <paramref name="request"/>, named <paramref name="placement"/> (for a placement of one's own that offloads).</summary>
    public static ExpertPlan OffloadPlan(ExpertPlacementRequest request, string placement)
    {
        ArgumentNullException.ThrowIfNull(request);
        int kept = request.DeviceExperts;
        string sized = request.CacheBytes is { } limit ? $"{WeightPlacements.Gigabytes(limit)} asked ({CacheVariable})"
            : request.AvailableBytes is { } free ? $"as {WeightPlacements.Gigabytes(free)} free holds beside {WeightPlacements.Gigabytes(request.OtherBytes)} of the rest"
            : "the device reports no free memory";
        return new ExpertPlan(placement, true, kept, request.ColdOnCpu,
            $"{kept} of {request.Experts} experts a layer on {request.Device}, the rest in system memory, {(request.ColdOnCpu ? "computed by the CPU" : "copied in when picked")} "
            + $"({sized}; {WeightPlacements.Gigabytes(request.ExpertBytes)} an expert)");
    }

    private sealed class ResidentPlacement : IExpertPlacement
    {
        public string Name => Resident;

        public string Summary => "every expert on the device (a model that does not fit fails as before, or follows the weight placement)";

        public ExpertPlan Place(ExpertPlacementRequest request) =>
            new(Name, false, request.Experts, false, $"every expert on {request.Device}");
    }

    private sealed class OffloadPlacement : IExpertPlacement
    {
        public string Name => Offload;

        public string Summary => "the most used experts of each layer on the device (as its free memory or IDRAK_MOE_DEVICE_EXPERTS_MB allows), the rest in system memory, copied in when picked (or computed by the CPU: IDRAK_MOE_COLD=cpu)";

        public ExpertPlan Place(ExpertPlacementRequest request) => OffloadPlan(request, Name);
    }

    private sealed class AutoPlacement : IExpertPlacement
    {
        public string Name => Auto;

        public string Summary => "every expert on the device when the model fits its measured free memory, else offload";

        public ExpertPlan Place(ExpertPlacementRequest request) => request.Fits && request.CacheBytes is null
            ? new(Name, false, request.Experts, false, $"every expert on {request.Device} (it fits)")
            : OffloadPlan(request, Name);
    }
}
