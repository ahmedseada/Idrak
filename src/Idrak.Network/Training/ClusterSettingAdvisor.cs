// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Networking.Training;

/// <summary>
/// The library's "cluster" advisor (plan 23, decision 4; <see cref="SettingAdvisors"/>): from what a data-parallel run
/// measured over its first warm steps (<see cref="ClusterStepMeasure"/>), how to send less when the network takes more of
/// a step than <see cref="ClusterOptions.SuggestShare"/>. Each suggestion names its figures and how to set it; none is
/// applied. Nothing is suggested at or under the share.
/// <list type="bullet">
/// <item>The hierarchical collective, when ring is set over more than one node and a node holds several ranks.</item>
/// <item>A bfloat16 wire, when the buckets travel uncompressed and a sample slice of the gradients reduced over both
/// wires lost at most <see cref="ClusterOptions.SuggestBFloat16Tolerance"/> of its largest value (unset:
/// <see cref="DefaultBFloat16Tolerance"/>, bfloat16's rounding step).</item>
/// <item>When the network's share is still above it (with bf16's half the bytes where suggested): gradient accumulation
/// and local SGD by the factor that brings it down (the step's compute that many times per synchronisation), and top-k
/// with the share of values that does (all-reduce runs that send every value).</item>
/// <item>The bucket size, when it is set to one bucket for every gradient and the backward pass ran beside nothing: the
/// buckets that let the network go during the backward pass.</item>
/// </list>
/// </summary>
internal sealed class ClusterSettingAdvisor : ISettingAdvisor
{
    /// <summary>The advisor's name: "cluster".</summary>
    public const string AdvisorName = "cluster";

    /// <summary>
    /// The default bf16 tolerance, bfloat16's own rounding step, 2^-7 (the owner's choice, decision 4): Idrak's bf16 ring
    /// rounds both the contributions it sends and the reduced sum it gathers, so a slice loses up to about two half-steps
    /// of its largest value (measured 1.47e-3, 2.17e-3 and 4.58e-3 on the tests' slices). The PyTorch DDP fixture's hook,
    /// which sums in float32, lost 2.71e-3 in the same measure (each tensor a slice; 1.33e-3 of the whole model's largest
    /// gradient), which would refuse bf16 on some of those slices.
    /// </summary>
    public const double DefaultBFloat16Tolerance = 1.0 / 128;

    public string Name => AdvisorName;

    public string Summary => "a data-parallel run whose network takes more of a step than IDRAK_CLUSTER_SUGGEST_SHARE: a bf16 wire within its measured error, "
                             + "gradient accumulation, local SGD, top-k, the bucket size, the hierarchical collective";

    public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation)
    {
        ArgumentNullException.ThrowIfNull(situation);
        if (situation.Subject is not ClusterStepMeasure m || m.Steps == 0 || m.Step <= TimeSpan.Zero)
        {
            yield break;
        }

        var o = m.Options;
        double threshold = o.SuggestShare, share = m.Share;
        if (!(share > threshold))
        {
            yield break;
        }

        string device = situation.Device.ToString();
        double step = m.Step.TotalSeconds, exposed = m.Exposed.TotalSeconds, compute = m.Compute.TotalSeconds;
        string figures = Text($"the network takes {Pct(share)} of a step ({Ms(exposed)} of {Ms(step)} ms{(m.Simulated ? " on the simulated links" : "")}, ")
                         + Text($"{m.Steps} warm steps, {m.WorldSize} ranks), above IDRAK_CLUSTER_SUGGEST_SHARE's {Pct(threshold)}");

        if (m.Nodes > 1 && m.WorldSize > m.Nodes && string.Equals(m.Collective, "ring", StringComparison.OrdinalIgnoreCase))
        {
            yield return new("ClusterOptions.Collective", "hierarchical",
                figures + Text($"; ring sends every rank's share across the links between the {m.Nodes} nodes, hierarchical sums inside each node first ")
                + "and sends only the nodes' sums across (cluster sim --collective hierarchical, IDRAK_CLUSTER_COLLECTIVE=hierarchical)", device, AdvisorName);
        }

        bool half = false;
        if (m.Compression == "none" && m.BFloat16Error is { } error)
        {
            double tolerance = o.SuggestBFloat16Tolerance ?? DefaultBFloat16Tolerance;
            if (error <= tolerance)
            {
                half = true;
                yield return new("ClusterOptions.Wire", "bf16",
                    figures + Text($"; a bf16 wire sends half the bytes, and a slice of {m.SliceValues} gradient values reduced over both wires lost {error:G3} of its largest value, ")
                    + Text($"within the tolerance {tolerance:G3} (IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE) (cluster sim --wire bf16, IDRAK_CLUSTER_WIRE=bf16)"), device, AdvisorName);
            }
        }

        // What is left once bf16 halves the bytes (where suggested): a factor of the compute per synchronisation brings the
        // share to the threshold.
        double left = half ? exposed / 2 : exposed;
        if (compute > 0 && left / (compute + left) > threshold)
        {
            double needed = left * (1 - threshold) / (threshold * compute);
            int factor = needed >= int.MaxValue ? int.MaxValue : Math.Max(2, (int)Math.Ceiling(needed));
            double after = left / ((factor * compute) + left);
            string still = half ? Text($", {Pct(left / (compute + left))} with bf16") : "";
            yield return new("GradientAccumulation", Text($"{factor}"),
                figures + still + Text($"; {factor} times the rows per synchronisation brings it to {Pct(after)} ")
                + Text($"(pretrain and tune --accumulate, times {factor}; a Trainer's batch {factor} times as large)"), device, AdvisorName);
            if (m.Sync == "all-reduce" && o.LocalSgdEvery == 0)
            {
                yield return new("ClusterOptions.LocalSgdEvery", Text($"{factor}"),
                    figures + still + Text($"; averaging the weights every {factor} steps instead of summing the gradients every step brings it to {Pct(after)} ")
                    + Text($"(cluster sim --local-sgd {factor}, IDRAK_CLUSTER_LOCAL_SGD={factor}; the ranks' weights drift apart between averages)"), device, AdvisorName);
            }

            if (m.Sync == "all-reduce" && m.Compression == "none" && o.TopK == 0 && exposed > 0)
            {
                // Top-k all-gathers each rank's (index, value) pairs: about share x ranks the bytes a ring all-reduce moves.
                double topk = threshold * compute / ((1 - threshold) * exposed * m.WorldSize);
                string value = topk.ToString("G2", CultureInfo.InvariantCulture);
                if (topk * m.WorldSize < 1 && double.Parse(value, CultureInfo.InvariantCulture) > 0)
                {
                    yield return new("ClusterOptions.TopK", value,
                        figures + Text($"; top-k sends each rank's largest {value} of the values as (index, value) pairs, about {topk * m.WorldSize:G2} of the bytes, ")
                        + Text($"bringing it to {Pct(threshold)}; error feedback keeps the rest for the next step (cluster sim --topk {value}, IDRAK_CLUSTER_TOPK={value})"), device, AdvisorName);
                }
            }
        }

        if (o.BucketBytes is not null && o.Overlap && m.Buckets == 1 && m.Backward > TimeSpan.Zero && m.Network > TimeSpan.Zero)
        {
            double network = m.Network.TotalSeconds, backward = m.Backward.TotalSeconds;
            int buckets = (int)Math.Clamp(Math.Ceiling(network / Math.Max(network - backward, network / BucketSizing.MostBuckets)), 2, BucketSizing.MostBuckets);
            long kib = Math.Max(1, ((m.GradientBytes / buckets) + 1023) / 1024);
            string value = Text($"{kib}KiB");
            yield return new("ClusterOptions.BucketBytes", value,
                figures + Text($"; one bucket of {m.BucketBytes / 1024.0:F0} KiB holds every gradient, so none of the network's {Ms(network)} ms goes during the backward pass's {Ms(backward)} ms: ")
                + Text($"{buckets} buckets send all but the last while it runs (cluster sim --bucket-bytes {value}, IDRAK_CLUSTER_BUCKET_BYTES={value}; unset, it is measured on the first step)"),
                device, AdvisorName);
        }
    }

    private static string Text(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

    private static string Pct(double share) => (share * 100).ToString("F0", CultureInfo.InvariantCulture) + "%";

    private static string Ms(double seconds) => (seconds * 1e3).ToString("F2", CultureInfo.InvariantCulture);
}
