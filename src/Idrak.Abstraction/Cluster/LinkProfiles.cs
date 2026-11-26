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
/// The link profiles' names (plan 23): a name may end in a rate that sets the bandwidth of the profile before it
/// (<c>direct-link-10g</c>: "direct-link" at 10 Gb/s; <c>wifi-300m</c>: 300 Mb/s), in bits a second as links are sold,
/// powers of 1,000. Resolving a name to its profile is not on the new design yet.
/// </summary>
public static class LinkProfiles
{
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
