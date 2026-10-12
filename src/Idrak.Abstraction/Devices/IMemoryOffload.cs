// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>How readily a storage moves to system memory when the device fills up (higher moves first).</summary>
public enum OffloadPriority : byte
{
    /// <summary>Used all the time (activations, trained weights): stays on the device while anything colder can move.</summary>
    Hot = 0,

    /// <summary>Weights no step changes (frozen layers, packed int8 / int4 / bfloat16 weights): read once per pass.</summary>
    Frozen = 1,

    /// <summary>Optimizer state (moments): read and written once per optimizer step.</summary>
    OptimizerState = 2,
}

/// <summary>
/// Where a frozen lookup table lives on a device that can keep storages in system memory (<see cref="IMemoryOffload"/>):
/// a table a lookup reads a few rows of (a token embedding beside a head of another format), which the device reads in
/// place over PCIe for a few kilobytes a token. Set for every model with <see cref="ComputeResources.LookupTables"/>
/// (IDRAK_LOOKUP_TABLES), or for one build with its options.
/// </summary>
public enum LookupTablePlacement : byte
{
    /// <summary>On the device until an allocation finds the device full; then in system memory for good, and the allocation retried.</summary>
    Automatic = 0,

    /// <summary>In system memory from the start: the device's memory for everything else, the table's pinned in the system's.</summary>
    HostMemory = 1,

    /// <summary>On the device always: a full device fails the allocation as before.</summary>
    DeviceMemory = 2,
}

/// <summary>
/// Keeping tensors in system memory for a device whose own memory is full, implemented by each GPU backend that can
/// (CUDA today; other GPU backends add their own). The library only talks to this interface: a device without it
/// (the CPU, whose tensors are in system memory already) offloads nothing, and every caller simply skips the step.
/// </summary>
/// <remarks>
/// An offloaded storage keeps working everywhere: the device reads system memory directly, only slower. The rest is
/// about speed: <see cref="Prefetch"/> and <see cref="Stage"/> give a layer device copies of its offloaded weights while
/// it computes, <see cref="MoveToDevice"/> and <see cref="Rebalance"/> bring storages back when memory frees up, and the
/// priorities decide which storages leave first.
/// </remarks>
public interface IMemoryOffload
{
    /// <summary>Storages that live in system memory now (a quick test for "is there anything to stage or bring back").</summary>
    int OffloadedCount { get; }

    /// <summary>Whether <paramref name="storage"/> lives in system memory (a staged copy on the device counts as offloaded).</summary>
    bool IsOffloaded(Storage storage);

    /// <summary>Records how readily <paramref name="storage"/> may move to system memory when the device fills up.</summary>
    void SetPriority(Storage storage, OffloadPriority priority);

    /// <summary>
    /// Moves <paramref name="storage"/> to system memory now (its values are kept); <paramref name="keep"/> keeps it there
    /// (no <see cref="Rebalance"/>). False when it cannot move now (a graph may use its address, or it is evicted).
    /// </summary>
    bool MoveToHost(Storage storage, bool keep);

    /// <summary>Moves an offloaded storage back into device memory if it fits; false otherwise.</summary>
    bool MoveToDevice(Storage storage);

    /// <summary>
    /// Called between steps. With <paramref name="makeRoom"/>, when the last step put its own data in system memory
    /// because the device was full, moves as much cold data (optimizer state, then frozen weights) out instead. Otherwise
    /// brings offloaded storages back while the device keeps its headroom (<see cref="ComputeResources.OffloadReturnHeadroom"/>) free,
    /// hottest first (storages moved with keep = true stay). Returns how many moved.
    /// </summary>
    int Rebalance(bool makeRoom = true);

    /// <summary>
    /// Before an inference pass while part of a model lives in system memory (the device is as full as it gets): moves
    /// cold storages (frozen weights, largest first) out until the device has its headroom
    /// (<see cref="ComputeResources.OffloadReturnHeadroom"/>) free, so the pass's activations and the staged copies of
    /// offloaded weights fit on the device instead of landing in system memory, where the device would compute over PCIe.
    /// Returns how many moved; by default none.
    /// </summary>
    int KeepFree() => 0;

    /// <summary>Starts copying the offloaded ones among <paramref name="storages"/> to the device in the background.</summary>
    void Prefetch(IReadOnlyList<Storage> storages);

    /// <summary>
    /// Gives the offloaded ones among <paramref name="storages"/> device copies (prefetched ones are waited for, the
    /// others copied now) until the returned token is disposed; null when there is nothing to stage. Kernels then read
    /// the device copies; the storages must not be written meanwhile (weights during a forward or backward pass).
    /// </summary>
    IDisposable? Stage(IReadOnlyList<Storage> storages);

    /// <summary>
    /// Marks <paramref name="storage"/> as a lookup table (<see cref="LookupTablePlacement.Automatic"/>): when an allocation
    /// finds the device full, it goes to system memory for good (<see cref="PlaceOnHost"/>) before the allocation fails.
    /// By default nothing is marked.
    /// </summary>
    void MarkLookupTable(Storage storage)
    {
    }

    /// <summary>
    /// Moves <paramref name="storage"/> to system memory for good (its values kept), read in place by the kernels: it never
    /// comes back on its own, is never staged, and does not count as offloaded (<see cref="OffloadedCount"/>), so nothing
    /// else changes. False when it cannot move now (a graph may hold its address); by default false.
    /// </summary>
    bool PlaceOnHost(Storage storage) => false;
}
