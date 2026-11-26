// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Cluster;

// Plan 27, R7: the ranks agree on a stop: the settings that space the agreement's exchanges and bound a rank's wait.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClusterStopGroup =
    [
        ("cluster stop: the settings (IDRAK_CLUSTER_STOP_EVERY, _STOP_WAIT; a bad value named) and ClusterOptions' list of variables", StopAgreementRules),
    ];

    // In-process ranks are awaited this long at most: far within the collectives' timeout of the options below.
    private static readonly TimeSpan StopBound = TimeSpan.FromSeconds(150);

    private static ClusterOptions StopOptions() => DpOptions() with { Timeout = TimeSpan.FromMinutes(10) };

    private static void StopAgreementRules(Device device)
    {
        if (!FirstRun(nameof(StopAgreementRules)))
        {
            return;
        }

        _ = device;

        // The settings and their variables.
        var defaults = new ClusterOptions();
        Check(defaults.StopEvery == 1 && defaults.StopWait == TimeSpan.Zero, "defaults: every boundary, no give-up");
        var set = ClusterOptions.From(n => n switch
        {
            "IDRAK_CLUSTER_STOP_EVERY" => "3",
            "IDRAK_CLUSTER_STOP_WAIT" => "1.5",
            _ => null,
        }, null);
        Check(set.StopEvery == 3 && set.StopWait == TimeSpan.FromSeconds(1.5), "the variables set them");
        ThrowsNaming<ArgumentException>(() => ClusterOptions.From(n => n == "IDRAK_CLUSTER_STOP_EVERY" ? "0" : null, null), "a stop check every 0 boundaries", "IDRAK_CLUSTER_STOP_EVERY=0");
        ThrowsNaming<ArgumentException>(() => ClusterOptions.From(n => n == "IDRAK_CLUSTER_STOP_WAIT" ? "-1" : null, null), "a negative wait", "IDRAK_CLUSTER_STOP_WAIT=-1");
        Check(new[] { "IDRAK_CLUSTER_STOP_EVERY", "IDRAK_CLUSTER_STOP_WAIT" }.All(ClusterOptions.Variables.Contains), "ClusterOptions lists the variables");
    }
}
