// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Weight offloading as tensors and modules see it (see <see cref="IMemoryOffload"/>, which devices implement). The library
/// that ships the layers (Idrak) stages their weights between device and system memory; tensors only record which layer
/// produced them, and modules and tensors call the hooks (<see cref="ITensorOffloadingHooks"/>) that library registers
/// when the device registry fills (<see cref="Register"/>). <see cref="Hooks"/> is null until then (always, without
/// Idrak), and costs a field read.
/// </summary>
public static class TensorOffloading
{
    [ThreadStatic]
    private static object? t_layer;

    private static ITensorOffloadingHooks? s_hooks;

    /// <summary>The registered hooks, or null when none are (nothing stages weights).</summary>
    public static ITensorOffloadingHooks? Hooks => s_hooks;

    /// <summary>Registers the hooks modules and tensors call (one set: a later registration replaces it).</summary>
    public static void Register(ITensorOffloadingHooks hooks)
    {
        ArgumentNullException.ThrowIfNull(hooks);
        Volatile.Write(ref s_hooks, hooks);
    }

    /// <summary>
    /// The layer (an opaque group: a module, or modules one fused pass computes) whose forward runs on this thread with
    /// its weights staged; tensors produced meanwhile record it, so the backward pass can stage the same weights.
    /// </summary>
    public static object? CurrentLayer
    {
        get => t_layer;
        set => t_layer = value;
    }

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

    /// <summary>
    /// Before an inference pass of <paramref name="model"/> (a generation's prompt, an image encoder's images): with
    /// offloading active, the model's frozen weights are marked cold (they are, before any layer has run), the device
    /// rebalances as between steps, then keeps its headroom free (<see cref="IMemoryOffload.KeepFree"/>), so a model that
    /// did not fit computes with its activations and staged weights on the device.
    /// </summary>
    public static void BeforeInference(Device device, Module model)
    {
        if (device.Backend.Offload is { } offload && Active(offload))
        {
            Hooks?.MarkFrozen(model);
            offload.Rebalance();
            offload.KeepFree();
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

/// <summary>
/// What modules and tensors call to offload weights (<see cref="TensorOffloading.Register"/>): the library that ships the
/// layers implements it.
/// </summary>
public interface ITensorOffloadingHooks
{
    /// <summary>Stages a module's offloaded weights before its forward; the result (if any) is disposed after it.</summary>
    IDisposable? EnterForward(Module module, Tensor input);

    /// <summary>A parameter's trainability changed on a device that offloads (frozen weights may move to system memory).</summary>
    void TrainableChanged(Tensor parameter, bool trainable);

    /// <summary>
    /// Stages offloaded weights back during a backward pass on a device with offloaded storages, given the layer each
    /// node of the backward order was recorded under (<see cref="TensorOffloading.CurrentLayer"/>, null for none); null
    /// when there is nothing to stage.
    /// </summary>
    IBackwardStaging? ForBackward(Device device, IReadOnlyList<object?> layers);

    /// <summary>Marks a model's frozen weights cold (the weights its layers only read while they compute).</summary>
    void MarkFrozen(Module model);
}

/// <summary>Brings offloaded weights back group by group during a backward pass (<see cref="ITensorOffloadingHooks.ForBackward"/>).</summary>
public interface IBackwardStaging : IDisposable
{
    /// <summary>Before the node at <paramref name="index"/> of the backward order runs.</summary>
    void Before(int index);
}
