// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Weight offloading as tensors and modules see it (see <see cref="IMemoryOffload"/>). The library that ships the
/// offloading devices (Idrak) stages layers' weights between device and system memory; tensors only record which layer
/// produced them, and modules and tensors call these hooks, which that library sets when its devices register. Each hook
/// is null until then, and costs a field read.
/// </summary>
public static class TensorOffloading
{
    [ThreadStatic]
    private static object? t_layer;

    /// <summary>
    /// The layer (an opaque group: a module, or modules one fused pass computes) whose forward runs on this thread with
    /// its weights staged; tensors produced meanwhile record it, so the backward pass can stage the same weights.
    /// </summary>
    public static object? CurrentLayer
    {
        get => t_layer;
        set => t_layer = value;
    }

    /// <summary>Stages a module's offloaded weights before its forward; the result (if any) is disposed after it.</summary>
    public static Func<Module, Tensor, IDisposable?>? EnterForward { get; set; }

    /// <summary>A parameter's trainability changed on a device that offloads (frozen weights may move to system memory).</summary>
    public static Action<Tensor, bool>? TrainableChanged { get; set; }

    /// <summary>
    /// Stages offloaded weights back during a backward pass on a device with offloaded storages, given the layer each
    /// node of the backward order was recorded under (<see cref="CurrentLayer"/>, null for none); null when there is
    /// nothing to stage.
    /// </summary>
    public static Func<Device, IReadOnlyList<object?>, IBackwardStaging?>? ForBackward { get; set; }

    /// <summary>Whether offloading is on for a device's offload support: asked for, or something is offloaded already.</summary>
    public static bool Active(IMemoryOffload offload) => ComputeResources.OffloadToHostMemory || offload.OffloadedCount > 0;

    /// <summary>Raises a storage's offload priority (frozen weights, optimizer state go to system memory first); never lowers it.</summary>
    public static void MarkCold(Storage storage, OffloadPriority priority)
    {
        if (storage.OffloadPriority < priority && storage.Backend.Offload is { } offload)
        {
            offload.SetPriority(storage, priority);
        }
    }

    /// <summary>Between optimizer steps: cold data makes room for the next step, or offloaded tensors come back.</summary>
    public static void StepBoundary(Device device)
    {
        if (device.Backend.Offload is { } offload && Active(offload))
        {
            offload.Rebalance();
        }
    }
}

/// <summary>Brings offloaded weights back group by group during a backward pass (<see cref="TensorOffloading.ForBackward"/>).</summary>
public interface IBackwardStaging : IDisposable
{
    /// <summary>Before the node at <paramref name="index"/> of the backward order runs.</summary>
    void Before(int index);
}
