// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using Idrak.Networking.Compression;

namespace Idrak.Networking.Training;

/// <summary>
/// The measurement behind the "cluster" setting advisor (plan 23, decision 4; <see cref="ClusterSettingAdvisor"/>). While
/// it is on (<see cref="ClusterOptions.Suggest"/>, unset: while someone listens at <see cref="TelemetryLevel.Suggestions"/>)
/// the reduction times its first warm steps (the first, cold step aside, rule 76; <see cref="ClusterOptions.SuggestSteps"/>
/// of them): each step's time from one step's end to the next, with the wall time of its reduction replaced by the network
/// time the step waited for (the virtual clock's on a simulated transport, so a simulated step is the cluster's), the
/// network's time and the backward pass's. As the last of them ends, it reduces its figures so every rank gets the means
/// over the ranks; rank 0 asks the "cluster" advisor (<see cref="ClusterSettingAdvisor"/>, reached directly: the advisors'
/// registry is gone) and the measurement ends for the run. Nothing a run computes changes: the figures are reduced after
/// the step's own reduction. Off, the cost is one field read a step. A group that changes (a fault) ends the measurement on
/// every member.
/// <para>
/// Every rank must decide alike (the figures' collective is taken by all or none): an in-process run resolves the setting
/// once as it starts (<c>InProcessCluster.Run</c>: set, or whether someone listens then) and every rank takes it, with its
/// warm steps, from its context; collectives without that context decide once per run whichever rank comes
/// first (keyed by the run's options instance); ranks in processes of their own agree in one collective as they start.
/// Each suggestion is published once a run (the published set is the run's too). The figures' collective is network
/// traffic, on a simulated transport's clock like any other: the event log's header keeps the setting and a replay from
/// the log (<c>ClusterReplay.Run</c> with it, <c>idrak cluster replay --from</c>) measures alike.
/// </para>
/// </summary>
internal partial class GradientReduction
{
    // A run's decision and what it published, shared by its ranks in one process (and by its synchronisations: a pretraining
    // run's stages each start one), held as long as the run's options are.
    private static readonly ConditionalWeakTable<ClusterOptions, RunAdvice> Runs = new();

    private AdviceMeter? _meter;
    private RunAdvice? _own;

    private sealed class RunAdvice
    {
        public bool? Wanted;
        public readonly HashSet<(string Setting, string Value)> Published = [];
    }

    private sealed class AdviceMeter
    {
        public int Warm;                                                      // the warm steps measured (the run's setting)
        public int Steps;
        public double Step, Exposed, Network, Backward;                     // seconds, summed over the measured steps
        public bool Simulated;
        public (TimeSpan Exposed, TimeSpan Network, TimeSpan Backward, TimeSpan ReduceWall, bool OnClock)? Pending;   // the step's, until it ends
    }

    // On while someone listens for suggestions, unless the settings say otherwise.
    private bool AdviceWanted => World > 1 && (Options.Suggest ?? Telemetry.IsEnabled(TelemetryLevel.Suggestions));

    // This step is one the advisor measures.
    private bool AdviceStep => _meter is { } meter && StepsInGroup >= 2 && StepsInGroup <= 1L + meter.Warm;

    // The run's setting where the run resolved it (an in-process run does as it starts, and a replay takes its log's): the
    // decision and its warm steps; else the synchronisation's own options.
    private ClusterOptions? RunSetting => Collectives.Context?.Options is { Suggest: not null } run ? run : null;

    private RunAdvice Run => Collectives.Context?.Options is { } run ? Runs.GetValue(run, static _ => new RunAdvice()) : _own ??= new RunAdvice();

    // The run's decision for ranks of one process: the first rank's, every other's the same.
    private bool DecideInProcess()
    {
        if (RunSetting is { } setting)
        {
            return World > 1 && setting.Suggest == true;
        }

        var run = Run;
        lock (run)
        {
            run.Wanted ??= AdviceWanted;
            return run.Wanted.Value;
        }
    }

    private void StartAdvice(bool on)
    {
        var setting = RunSetting ?? Options;
        _meter = on && World > 1 ? new AdviceMeter { Warm = setting.SuggestSteps } : null;
    }

    private void StopAdvice() => _meter = null;

    // Ranks in processes of their own: on when any rank's is (one float, once, over a real transport).
    private bool AgreeAdvice()
    {
        Span<float> wanted = [AdviceWanted ? 1f : 0f];
        Values.AllReduce(wanted, ReduceOp.Max);
        return wanted[0] > 0f;
    }

    protected override void StepTimed(TimeSpan wall)
    {
        if (_meter is not { Pending: { } step } meter)
        {
            return;
        }

        meter.Pending = null;
        meter.Steps++;
        meter.Step += Math.Max(0, (wall - step.ReduceWall + step.Exposed).TotalSeconds);
        meter.Exposed += step.Exposed.TotalSeconds;
        meter.Network += step.Network.TotalSeconds;
        meter.Backward += step.Backward.TotalSeconds;
        meter.Simulated |= step.OnClock;
        if (StepsInGroup >= 1L + meter.Warm)
        {
            try
            {
                Advise(meter);
            }
            finally
            {
                _meter = null;                                 // measured once a run
            }
        }
    }

    // The figures' means over the ranks, then rank 0 asks the advisor.
    private void Advise(AdviceMeter meter)
    {
        Span<float> figures = [(float)meter.Step, (float)meter.Exposed, (float)meter.Network, (float)meter.Backward];
        Values.AllReduce(figures);
        double count = (double)World * Math.Max(1, meter.Steps);
        double step = figures[0] / count, exposed = figures[1] / count, network = figures[2] / count, backward = figures[3] / count;

        if (Collectives.Rank != 0)
        {
            return;                                            // rank 0 publishes the run's suggestions
        }

        string compression = _layout.Sharded ? "none" : _compressor.Name;
        var measure = new ClusterStepMeasure(Options, Name, Collectives.Name, compression, World, Collectives.Context?.Topology.Nodes.Count ?? 1, meter.Steps,
            TimeSpan.FromSeconds(step), TimeSpan.FromSeconds(exposed), TimeSpan.FromSeconds(network), TimeSpan.FromSeconds(backward),
            (long)_layout.Total * sizeof(float), (long)_layout.BucketValues * sizeof(float), _layout.Count, meter.Simulated);
        if (!Telemetry.IsEnabled(TelemetryLevel.Suggestions))
        {
            return;
        }

        var published = Run.Published;
        lock (published)
        {
            foreach (var suggestion in new ClusterSettingAdvisor().Suggest(new AdviceSituation(Collectives.Device, measure, null), measure))
            {
                if (published.Add((suggestion.Setting, suggestion.Value)))
                {
                    Telemetry.Suggestion(in suggestion);
                }
            }
        }
    }
}
