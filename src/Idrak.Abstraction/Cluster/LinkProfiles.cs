// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// How a link between two ranks behaves, for the simulator's virtual clock and for planning (plan 23): bandwidth,
/// latency, jitter, loss and failure. A profile is data: an app's profile, or one corrected by a calibration against a
/// measured run, takes the place of the built-in one of that name.
/// </summary>
public interface ILinkProfile
{
    /// <summary>The profile's name ("ethernet-1g").</summary>
    string Name { get; }

    /// <summary>Bytes a second one direction carries; 0 when the link's rate is not known (declare it or measure it).</summary>
    double BandwidthBytesPerSecond { get; }

    /// <summary>The time a message of no bytes takes.</summary>
    TimeSpan Latency { get; }

    /// <summary>The width of a seeded uniform draw added to each transfer.</summary>
    TimeSpan Jitter { get; }

    /// <summary>The share of messages lost (each one is sent again after <see cref="Latency"/>), from 0 to 1.</summary>
    double LossRate { get; }

    /// <summary>The chance per step that the link fails, drawn by a failure model, from 0 to 1.</summary>
    double FailureRate { get; }

    /// <summary>The time <paramref name="bytes"/> take: bytes / bandwidth + latency + a jitter drawn from <paramref name="seeded"/>.</summary>
    /// <exception cref="InvalidOperationException">The bandwidth is not known.</exception>
    TimeSpan Predict(long bytes, Random seeded);

    /// <summary>A copy with the fields given changed (a calibration's correction, a rate from a name's suffix).</summary>
    ILinkProfile With(double? bandwidthBytesPerSecond = null, TimeSpan? latency = null, TimeSpan? jitter = null, double? lossRate = null,
        double? failureRate = null, string? name = null);
}

/// <summary>A link profile held as plain values: the library's built-ins and calibrated profiles are these.</summary>
public sealed record LinkProfile : ILinkProfile
{
    /// <summary>A profile of <paramref name="name"/> carrying <paramref name="bandwidthBytesPerSecond"/> after <paramref name="latency"/>.</summary>
    public LinkProfile(string name, double bandwidthBytesPerSecond, TimeSpan latency)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        BandwidthBytesPerSecond = bandwidthBytesPerSecond;
        Latency = latency;
        Validate(this);
    }

    /// <inheritdoc />
    public string Name { get; init; }

    /// <summary>One line on what the link is, for listings.</summary>
    public string Summary { get; init; } = "";

    /// <inheritdoc />
    public double BandwidthBytesPerSecond { get; init; }

    /// <inheritdoc />
    public TimeSpan Latency { get; init; }

    /// <inheritdoc />
    public TimeSpan Jitter { get; init; }

    /// <inheritdoc />
    public double LossRate { get; init; }

    /// <inheritdoc />
    public double FailureRate { get; init; }

    /// <summary>
    /// The rate of one network adapter a machine's ranks share on this link, or 0 when each pair of ranks has the link to
    /// itself (bytes a second; a calibration from a bench of concurrent pairs finds it, plan 23 C8). A topology gives each
    /// pair the adapter's rate over the ranks of the larger node it joins, never more than <see cref="BandwidthBytesPerSecond"/>.
    /// </summary>
    public double AdapterBytesPerSecond { get; init; }

    /// <inheritdoc />
    public TimeSpan Predict(long bytes, Random seeded)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        ArgumentNullException.ThrowIfNull(seeded);
        if (!(BandwidthBytesPerSecond > 0))
        {
            throw new InvalidOperationException($"Link '{Name}' has no known bandwidth: give the rate in its name (\"{Name}-10g\") or measure it with idrak cluster bench.");
        }

        double seconds = Latency.TotalSeconds + bytes / BandwidthBytesPerSecond;
        if (Jitter > TimeSpan.Zero)
        {
            seconds += Jitter.TotalSeconds * seeded.NextDouble();
        }

        return TimeSpan.FromSeconds(seconds);
    }

    /// <inheritdoc />
    public ILinkProfile With(double? bandwidthBytesPerSecond = null, TimeSpan? latency = null, TimeSpan? jitter = null, double? lossRate = null,
        double? failureRate = null, string? name = null)
    {
        var copy = this with
        {
            Name = name ?? Name,
            BandwidthBytesPerSecond = bandwidthBytesPerSecond ?? BandwidthBytesPerSecond,
            Latency = latency ?? Latency,
            Jitter = jitter ?? Jitter,
            LossRate = lossRate ?? LossRate,
            FailureRate = failureRate ?? FailureRate,
        };
        Validate(copy);
        return copy;
    }

    /// <summary>Throws when <paramref name="profile"/>'s values are out of range.</summary>
    /// <exception cref="ArgumentException">A negative or non-finite figure, or a rate outside 0 to 1.</exception>
    public static void Validate(ILinkProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrWhiteSpace(profile.Name);
        if (!double.IsFinite(profile.BandwidthBytesPerSecond) || profile.BandwidthBytesPerSecond < 0)
        {
            throw new ArgumentException($"Link '{profile.Name}': bandwidth {profile.BandwidthBytesPerSecond} is not a number of bytes a second (0 or more).");
        }

        if (profile.Latency < TimeSpan.Zero || profile.Jitter < TimeSpan.Zero)
        {
            throw new ArgumentException($"Link '{profile.Name}': latency and jitter cannot be negative.");
        }

        if (profile.LossRate is < 0 or > 1 or double.NaN || profile.FailureRate is < 0 or > 1 or double.NaN)
        {
            throw new ArgumentException($"Link '{profile.Name}': loss and failure rates are shares from 0 to 1.");
        }

        if (profile is LinkProfile { AdapterBytesPerSecond: var adapter } && (!double.IsFinite(adapter) || adapter < 0))
        {
            throw new ArgumentException($"Link '{profile.Name}': the shared adapter's rate {adapter} is not a number of bytes a second (0 or more).");
        }
    }
}

/// <summary>
/// The link profiles, by name (plan 23). Idrak.Cluster registers the built-ins; a name may end in a rate that sets the
/// bandwidth of the profile before it (<c>direct-link-10g</c>: "direct-link" at 10 Gb/s; <c>wifi-300m</c>: 300 Mb/s),
/// in bits a second as links are sold, powers of 1,000. A calibration (<c>idrak cluster calibrate</c>, kept by
/// <see cref="ClusterCache"/>) takes the library's profile's place in <see cref="Resolve"/>; an app's registration stays
/// above both.
/// </summary>
public static class LinkProfiles
{
    private static readonly SlotTable<string, ILinkProfile> Table = new(nameof(LinkProfiles), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a profile is data with no call to fall back from: an app's or a calibrated profile takes the built-in's place, which stays behind it");

    static LinkProfiles() => LibraryDefaults.Ensure(typeof(LinkProfiles));

    /// <summary>The registered profiles, in registration order.</summary>
    public static IReadOnlyList<ILinkProfile> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="profile"/> under its name (over the library's of that name, which stays behind it).</summary>
    /// <exception cref="ArgumentException">A figure out of range (<see cref="LinkProfile.Validate"/>).</exception>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ILinkProfile profile)
    {
        LinkProfile.Validate(profile);
        Table.Register(profile.Name, profile, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's profile of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The profile registered under exactly that name, or null.</summary>
    public static ILinkProfile? Find(string name) => Table.Find(name);

    /// <summary>
    /// The profile <paramref name="name"/> names: the one registered under it, or one registered under the part before a
    /// rate suffix, with that rate (<c>direct-link-10g</c>).
    /// </summary>
    /// <exception cref="NotSupportedException">Neither is registered.</exception>
    public static ILinkProfile Resolve(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (Table.Find(name) is { } exact)
        {
            // A calibrated profile of that name takes the library's place; an app's registration stays above both.
            return !ReferenceEquals(Table.Default(name), exact) ? exact
                : ClusterCache.Calibrated(name) is { } calibrated ? calibrated.Apply(exact) : exact;
        }

        int dash = name.LastIndexOf('-');
        if (dash > 0 && TryParseRate(name.AsSpan(dash + 1), out double bytesPerSecond) && Table.Find(name.AsSpan(0, dash)) is { } named)
        {
            var rated = named.With(bandwidthBytesPerSecond: bytesPerSecond, name: name);
            return ClusterCache.Calibrated(name) is { } calibrated ? calibrated.Apply(rated) : rated;
        }

        throw new NotRegisteredException(nameof(LinkProfiles), name, Table.Keys,
            $"Link profile '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); a registered name may end in a rate (\"-10g\", \"-300m\"), or add one with LinkProfiles.Register.");
    }

    /// <summary>The library's profile of that name, or null.</summary>
    public static ILinkProfile? Default(string name) => Table.Default(name);

    /// <summary>Who registered the profile of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// Reads a link rate in bits a second: a number then <c>g</c> (gigabits) or <c>m</c> (megabits), powers of 1,000
    /// ("10g", "2.5g", "300m"), as bytes a second.
    /// </summary>
    public static bool TryParseRate(ReadOnlySpan<char> text, out double bytesPerSecond)
    {
        bytesPerSecond = 0;
        text = text.Trim();
        if (text.Length < 2)
        {
            return false;
        }

        double scale = char.ToLowerInvariant(text[^1]) switch
        {
            'g' => 1e9,
            'm' => 1e6,
            _ => 0,
        };
        if (scale == 0 || !double.TryParse(text[..^1], NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double count) || !(count > 0))
        {
            return false;
        }

        bytesPerSecond = count * scale / 8;
        return true;
    }
}
