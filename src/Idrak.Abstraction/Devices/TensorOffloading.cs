// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Weight offloading as tensors see it. Idrak's layers move weights between device and system memory; tensors only
/// record which layer produced them and call these hooks, which Idrak sets when it loads (offloading needs its devices).
/// </summary>
internal static class TensorOffloading
{
    /// <summary>The layer (an opaque group) running on this thread, recorded on the tensors it produces.</summary>
    [ThreadStatic]
    public static object? Current;

    /// <summary>A parameter's trainability changed on a device that offloads (frozen weights may move to system memory).</summary>
    public static Action<Tensor, bool>? TrainableChanged;

    /// <summary>Stages offloaded weights back for a backward pass over <c>order</c>; null when nothing is offloaded.</summary>
    public static Func<Device, List<Tensor>, IBackwardStaging?>? ForBackward;
}

/// <summary>Brings offloaded weights back group by group during a backward pass.</summary>
internal interface IBackwardStaging : IDisposable
{
    /// <summary>Before the node at <paramref name="index"/> of the backward order runs.</summary>
    void Before(int index);
}
