// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Text;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// A link profile as a calibration measured it (plan 23, phase C8: <c>idrak cluster calibrate</c>): the bandwidth and
/// latency fitted by least squares over measured transfers, the residual of the fit, and the rate of a network adapter a
/// machine's ranks share when a bench of concurrent pairs showed one.
/// </summary>
/// <param name="Link">The profile's name as the topology names it ("ethernet-1g", "direct-link-10g").</param>
/// <param name="BandwidthBytesPerSecond">The fitted bandwidth, bytes a second.</param>
/// <param name="Latency">The fitted latency.</param>
/// <param name="Jitter">The spread of the measured times around the fit (their standard deviation).</param>
/// <param name="Residual">The fit's root-mean-square residual, seconds.</param>
/// <param name="Samples">The measured transfers the fit used.</param>
/// <param name="AdapterBytesPerSecond">The shared adapter's rate, or 0 when every pair has the link to itself.</param>
/// <param name="Calibrated">When it was written.</param>
/// <param name="Source">What it was fitted from (a run's log or a bench's JSON).</param>
public sealed record CalibratedLink(string Link, double BandwidthBytesPerSecond, TimeSpan Latency, TimeSpan Jitter, double Residual, int Samples,
    double AdapterBytesPerSecond, DateTimeOffset Calibrated, string Source)
{
    /// <summary><paramref name="profile"/> with this calibration's figures (its loss and failure rates kept).</summary>
    public ILinkProfile Apply(ILinkProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        string summary = string.Create(CultureInfo.InvariantCulture, $"calibrated {Calibrated:yyyy-MM-dd} from {Samples} transfers (residual {Residual * 1e6:F1} us)");
        return new LinkProfile(profile.Name, BandwidthBytesPerSecond > 0 ? BandwidthBytesPerSecond : profile.BandwidthBytesPerSecond, Latency)
        {
            Summary = summary,
            Jitter = Jitter,
            LossRate = profile.LossRate,
            FailureRate = profile.FailureRate,
            AdapterBytesPerSecond = AdapterBytesPerSecond,
        };
    }
}

/// <summary>
/// A size a cluster measured on this machine and keeps (plan 23, phase C8): the chunk a transport's collectives send
/// ("chunk", per transport) and a gradient synchronisation's bucket ("bucket", per run shape), with why it was chosen.
/// </summary>
/// <param name="Kind">"chunk" or "bucket".</param>
/// <param name="Key">What it was measured for (a transport's name; a synchronisation, collective, world size and gradient size).</param>
/// <param name="Value">The bytes chosen.</param>
/// <param name="Measured">When.</param>
/// <param name="Detail">The figures that chose it.</param>
public sealed record ClusterTuning(string Kind, string Key, long Value, DateTimeOffset Measured, string Detail);

/// <summary>
/// Where a cluster keeps what it measured on this machine (plan 23, phase C8), beside the backends' tuning caches: the
/// calibrated link profiles, one text file per link under <c>{cache}/cluster/</c> (header <c>idrak-cluster-profiles 1</c>,
/// then <c>link bandwidth latency jitter residual samples adapter date source</c>), and the measured chunk and bucket sizes
/// in <c>{cache}/cluster/tuning.txt</c> (header <c>idrak-cluster-tuning 1</c>). The cache folder is
/// <see cref="ComputeResources.CacheFolder"/> (<c>IDRAK_CACHE</c>, or <c>~/.cache/idrak</c>); <c>IDRAK_TUNING_CACHE=0</c> keeps nothing; <c>IDRAK_CLUSTER_PROFILES_FILE</c> puts every
/// calibrated profile in one file of its own and <c>IDRAK_CLUSTER_TUNING_FILE</c> moves the sizes' file (an empty value
/// keeps none). A file whose header does not match is ignored and written anew. <see cref="SerializeProfiles"/>,
/// <see cref="ParseProfiles"/>, <see cref="SerializeTuning"/> and <see cref="ParseTuning"/> are pure, so the formats are
/// tested without a link. <see cref="LinkProfiles.Resolve"/> reads the calibrations.
/// </summary>
public static class ClusterCache
{
    /// <summary>The first line of a calibrated profiles' file.</summary>
    public const string ProfilesHeader = "idrak-cluster-profiles 1";

    /// <summary>The first line of the measured sizes' file.</summary>
    public const string TuningHeader = "idrak-cluster-tuning 1";

    private static readonly Lock Gate = new();
    private static readonly ConcurrentDictionary<string, (DateTime Written, long Length, IReadOnlyDictionary<string, CalibratedLink> Links)> Read = new(StringComparer.Ordinal);

    /// <summary>The cluster's folder of the tuning cache, or null when the cache is off (<c>IDRAK_TUNING_CACHE=0</c>).</summary>
    public static string? Folder
    {
        get
        {
            if (Environment.GetEnvironmentVariable("IDRAK_TUNING_CACHE") is "0" or "false" or "off")
            {
                return null;
            }

            return Path.Combine(ComputeResources.CacheFolder, "cluster");
        }
    }

    /// <summary>
    /// The one file every calibrated profile is kept in, when <paramref name="file"/> or <c>IDRAK_CLUSTER_PROFILES_FILE</c>
    /// names one; null: a file per link under <see cref="Folder"/>. An empty name keeps none (<see cref="KeepsProfiles"/>).
    /// </summary>
    public static string? ProfilesFile(string? file = null) => file ?? Environment.GetEnvironmentVariable("IDRAK_CLUSTER_PROFILES_FILE");

    /// <summary>Whether calibrated profiles are kept and read at all.</summary>
    public static bool KeepsProfiles(string? file = null) => ProfilesFile(file) is { } named ? named.Length > 0 : Folder is not null;

    /// <summary>The file the calibration of <paramref name="link"/> is kept in, or null when none is kept.</summary>
    public static string? ProfilePath(string link, string? file = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(link);
        if (ProfilesFile(file) is { } named)
        {
            return named.Length > 0 ? named : null;
        }

        return Folder is { } folder ? Path.Combine(folder, "link-" + Safe(link) + ".txt") : null;
    }

    /// <summary>The file the measured chunk and bucket sizes are kept in (<paramref name="file"/>, <c>IDRAK_CLUSTER_TUNING_FILE</c>, else the folder's), or null when none is kept.</summary>
    public static string? TuningPath(string? file = null)
    {
        if ((file ?? Environment.GetEnvironmentVariable("IDRAK_CLUSTER_TUNING_FILE")) is { } named)
        {
            return named.Length > 0 ? named : null;
        }

        return Folder is { } folder ? Path.Combine(folder, "tuning.txt") : null;
    }

    /// <summary>The calibration kept for <paramref name="link"/>, or null.</summary>
    public static CalibratedLink? Calibrated(string link, string? file = null)
    {
        if (ProfilePath(link, file) is not { } path)
        {
            return null;
        }

        return Links(path).TryGetValue(link, out var calibrated) ? calibrated : null;
    }

    /// <summary>Every calibration kept, by link name.</summary>
    public static IReadOnlyList<CalibratedLink> AllCalibrated(string? file = null)
    {
        if (ProfilesFile(file) is { } named)
        {
            return named.Length > 0 ? [.. Links(named).Values.OrderBy(l => l.Link, StringComparer.Ordinal)] : [];
        }

        if (Folder is not { } folder || !Directory.Exists(folder))
        {
            return [];
        }

        return [.. Directory.GetFiles(folder, "link-*.txt").Order(StringComparer.Ordinal).SelectMany(p => Links(p).Values).OrderBy(l => l.Link, StringComparer.Ordinal)];
    }

    /// <summary>The files calibrations are kept in now.</summary>
    public static IReadOnlyList<string> ProfileFiles(string? file = null) =>
        ProfilesFile(file) is { } named ? named.Length > 0 && File.Exists(named) ? [named] : []
        : Folder is { } folder && Directory.Exists(folder) ? [.. Directory.GetFiles(folder, "link-*.txt").Order(StringComparer.Ordinal)] : [];

    /// <summary>Keeps <paramref name="calibrated"/> (over an earlier calibration of its link); returns the file, or null when none is kept.</summary>
    public static string? Keep(CalibratedLink calibrated, string? file = null)
    {
        ArgumentNullException.ThrowIfNull(calibrated);
        if (ProfilePath(calibrated.Link, file) is not { } path)
        {
            return null;
        }

        lock (Gate)
        {
            var links = File.Exists(path) ? ParseProfiles(ReadText(path)).ToDictionary(l => l.Link, StringComparer.Ordinal) : new Dictionary<string, CalibratedLink>(StringComparer.Ordinal);
            links[calibrated.Link] = calibrated;
            WriteText(path, SerializeProfiles(links.Values.OrderBy(l => l.Link, StringComparer.Ordinal)));
            Read.TryRemove(path, out _);
        }

        return path;
    }

    /// <summary>The size kept for <paramref name="kind"/> and <paramref name="key"/>, or null.</summary>
    public static ClusterTuning? FindTuning(string kind, string key, string? file = null) =>
        TuningPath(file) is { } path && File.Exists(path)
            ? ParseTuning(ReadText(path)).LastOrDefault(t => t.Kind == kind && t.Key == key)
            : null;

    /// <summary>Every size kept.</summary>
    public static IReadOnlyList<ClusterTuning> AllTuning(string? file = null) =>
        TuningPath(file) is { } path && File.Exists(path) ? ParseTuning(ReadText(path)) : [];

    /// <summary>Keeps <paramref name="entry"/> (over an earlier one of its kind and key); returns the file, or null when none is kept.</summary>
    public static string? Keep(ClusterTuning entry, string? file = null)
    {
        ArgumentNullException.ThrowIfNull(entry);
        if (TuningPath(file) is not { } path)
        {
            return null;
        }

        lock (Gate)
        {
            var kept = File.Exists(path) ? ParseTuning(ReadText(path)).Where(t => t.Kind != entry.Kind || t.Key != entry.Key).ToList() : [];
            kept.Add(entry);
            WriteText(path, SerializeTuning(kept));
        }

        return path;
    }

    /// <summary>The calibrated profiles' file text: the header, then one line per link.</summary>
    public static string SerializeProfiles(IEnumerable<CalibratedLink> links)
    {
        ArgumentNullException.ThrowIfNull(links);
        var text = new StringBuilder(ProfilesHeader).Append('\n');
        foreach (var l in links)
        {
            text.Append(string.Create(CultureInfo.InvariantCulture,
                $"{l.Link} {l.BandwidthBytesPerSecond:R} {l.Latency.TotalSeconds:R} {l.Jitter.TotalSeconds:R} {l.Residual:R} {l.Samples} {l.AdapterBytesPerSecond:R} {l.Calibrated:yyyy-MM-ddTHH:mm:ssK} {OneLine(l.Source)}\n"));
        }

        return text.ToString();
    }

    /// <summary>The calibrations in a profiles' file's text; none when its header is not <see cref="ProfilesHeader"/>, and a malformed line is skipped.</summary>
    public static IReadOnlyList<CalibratedLink> ParseProfiles(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var lines = text.Split('\n');
        if (lines.Length == 0 || lines[0].TrimEnd('\r') != ProfilesHeader)
        {
            return [];
        }

        var links = new List<CalibratedLink>();
        var inv = CultureInfo.InvariantCulture;
        foreach (string raw in lines.Skip(1))
        {
            var f = raw.TrimEnd('\r').Split(' ', 9);
            if (f.Length < 8 || f[0].Length == 0
                || !double.TryParse(f[1], NumberStyles.Float, inv, out double bandwidth) || !double.IsFinite(bandwidth) || bandwidth < 0
                || !double.TryParse(f[2], NumberStyles.Float, inv, out double latency) || !double.IsFinite(latency) || latency < 0
                || !double.TryParse(f[3], NumberStyles.Float, inv, out double jitter) || !double.IsFinite(jitter) || jitter < 0
                || !double.TryParse(f[4], NumberStyles.Float, inv, out double residual)
                || !int.TryParse(f[5], NumberStyles.None, inv, out int samples)
                || !double.TryParse(f[6], NumberStyles.Float, inv, out double adapter) || !double.IsFinite(adapter) || adapter < 0
                || !DateTimeOffset.TryParse(f[7], inv, DateTimeStyles.None, out var date))
            {
                continue;
            }

            links.Add(new CalibratedLink(f[0], bandwidth, TimeSpan.FromSeconds(latency), TimeSpan.FromSeconds(jitter), residual, samples, adapter, date, f.Length > 8 ? f[8] : ""));
        }

        return links;
    }

    /// <summary>The measured sizes' file text: the header, then one line per size.</summary>
    public static string SerializeTuning(IEnumerable<ClusterTuning> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);
        var text = new StringBuilder(TuningHeader).Append('\n');
        foreach (var e in entries)
        {
            text.Append(string.Create(CultureInfo.InvariantCulture, $"{e.Kind} {e.Key} {e.Value} {e.Measured:yyyy-MM-ddTHH:mm:ssK} {OneLine(e.Detail)}\n"));
        }

        return text.ToString();
    }

    /// <summary>The sizes in a sizes' file's text; none when its header is not <see cref="TuningHeader"/>, and a malformed line is skipped.</summary>
    public static IReadOnlyList<ClusterTuning> ParseTuning(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var lines = text.Split('\n');
        if (lines.Length == 0 || lines[0].TrimEnd('\r') != TuningHeader)
        {
            return [];
        }

        var entries = new List<ClusterTuning>();
        foreach (string raw in lines.Skip(1))
        {
            var f = raw.TrimEnd('\r').Split(' ', 5);
            if (f.Length < 4 || f[0].Length == 0 || f[1].Length == 0
                || !long.TryParse(f[2], NumberStyles.None, CultureInfo.InvariantCulture, out long value)
                || !DateTimeOffset.TryParse(f[3], CultureInfo.InvariantCulture, DateTimeStyles.None, out var date))
            {
                continue;
            }

            entries.Add(new ClusterTuning(f[0], f[1], value, date, f.Length > 4 ? f[4] : ""));
        }

        return entries;
    }

    // The calibrations in a file, read once per version of it (a calibration written by this process is seen at once).
    private static IReadOnlyDictionary<string, CalibratedLink> Links(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists)
        {
            return new Dictionary<string, CalibratedLink>();
        }

        if (Read.TryGetValue(path, out var known) && known.Written == info.LastWriteTimeUtc && known.Length == info.Length)
        {
            return known.Links;
        }

        IReadOnlyDictionary<string, CalibratedLink> links;
        try
        {
            links = ParseProfiles(ReadText(path)).GroupBy(l => l.Link, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Last(), StringComparer.Ordinal);
        }
        catch (IOException)
        {
            return known.Links ?? new Dictionary<string, CalibratedLink>();     // being written by another process: the last version read
        }

        Read[path] = (info.LastWriteTimeUtc, info.Length, links);
        return links;
    }

    private static string ReadText(string path)
    {
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                return File.ReadAllText(path);
            }
            catch (IOException) when (attempt < 5)
            {
                Thread.Sleep(10 << attempt);                                  // another process is replacing it
            }
        }
    }

    // Written beside the file, then moved over it, so a reader never sees half a file.
    private static void WriteText(string path, string text)
    {
        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        string temporary = path + "." + Environment.ProcessId.ToString(CultureInfo.InvariantCulture) + "." + Environment.CurrentManagedThreadId.ToString(CultureInfo.InvariantCulture) + ".tmp";
        File.WriteAllText(temporary, text);
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                File.Move(temporary, path, overwrite: true);
                return;
            }
            catch (IOException) when (attempt < 5)
            {
                Thread.Sleep(10 << attempt);
            }
        }
    }

    private static string Safe(string link) => new([.. link.Select(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '.' or '_' ? char.ToLowerInvariant(c) : '_')]);

    private static string OneLine(string text) => text.ReplaceLineEndings(" ");
}
