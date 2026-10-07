// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Backends;
using Idrak.Layers;

namespace Idrak;

/// <summary>
/// Where the library meets <see cref="IMemoryOffload"/> (implemented by GPU backends that can keep tensors in system
/// memory): layers stage their offloaded weights on the device while they compute, forward and backward, with the next
/// layer's copied in the background; frozen weights and optimizer state are marked cold; offloaded tensors move between
/// steps. A device without offload support (the CPU) skips all of it, and with nothing offloaded each hook is one check.
/// </summary>
internal static class Offloading
{
    // Tensors live in Idrak.Abstraction and reach the offloading of Idrak's layers through these hooks, set when this
    // assembly's devices are registered (LibraryDevices): only they offload, so the hooks are in place before anything is.
    internal static void ConnectTensors()
    {
        TensorOffloading.TrainableChanged = TrainableChanged;
        TensorOffloading.ForBackward = ForBackward;
        TensorOffloading.EnterForward = EnterForward;
    }

    // Module.Forward's hook: a scope only when the layer was entered (boxed then; nothing is allocated otherwise).
    private static IDisposable? EnterForward(Module module, Tensor input) => Enter(module, input) is { Entered: true } scope ? scope : null;

    // The layer whose forward runs on this thread (tagged on the autograd nodes it records, for the backward pass), and
    // the layer staged before it (to learn which layer follows which).
    // A layer (Module) or layers computed in one fused pass (LayerGroup). Kept in TensorOffloading.CurrentLayer, where the
    // tensors (in Idrak.Abstraction) read it.

    [ThreadStatic]
    private static object? t_previous;

    // The layer that ran after each layer last time: its weights are copied while this one computes.
    private static readonly ConditionalWeakTable<object, object> NextLayer = new();

    // Layers one fused pass computes together (q/k/v, gate/up): staged as one unit; one object per set, kept with its
    // first layer, so the order learned and the backward tags see the same unit every pass.
    private sealed class LayerGroup(Module[] layers)
    {
        public readonly Module[] Layers = layers;
    }

    private static readonly ConditionalWeakTable<Module, LayerGroup> Groups = new();

    private static LayerGroup GroupOf(IReadOnlyList<Module> layers)
    {
        if (Groups.TryGetValue(layers[0], out var group) && group.Layers.Length == layers.Count)
        {
            bool same = true;
            for (int i = 1; i < layers.Count && same; i++)
            {
                same = ReferenceEquals(group.Layers[i], layers[i]);
            }

            if (same)
            {
                return group;
            }
        }

        group = new LayerGroup([.. layers]);
        Groups.AddOrUpdate(layers[0], group);
        return group;
    }

    /// <summary>Off: layers read offloaded weights over PCIe as they compute instead of staging them (for --bench-offload).</summary>
    internal static bool StageWeights = true;

    /// <summary>The layer (or fused layers) whose forward runs on this thread while weights are offloaded (null otherwise).</summary>
    public static object? Current => TensorOffloading.CurrentLayer;

    private static bool Active(IMemoryOffload offload) => TensorOffloading.Active(offload);

    /// <summary>
    /// Called by <see cref="Module.Forward"/>: for a layer without sub-layers, marks its frozen weights cold, stages its
    /// offloaded ones on the device and starts copying the next layer's. Dispose the result after the forward.
    /// </summary>
    public static LayerScope Enter(Module module, Tensor input)
    {
        if (input.Device.Backend.Offload is not { } offload || !Active(offload) || !StageWeights || module.Children().Any())
        {
            return default;
        }

        return Enter(module, ReadOnlyStorages(module, mark: true), offload);
    }

    /// <summary>
    /// <see cref="Enter(Module, Tensor)"/> for layers one fused pass computes together (q/k/v, gate/up, a projection with its
    /// residual and normalization): their offloaded weights are staged as one unit, forward and backward.
    /// </summary>
    public static LayerScope EnterMany(IReadOnlyList<Module> layers, Tensor input)
    {
        if (input.Device.Backend.Offload is not { } offload || !Active(offload) || !StageWeights || layers.Count == 0)
        {
            return default;
        }

        if (layers.Count == 1)
        {
            return Enter(layers[0], input);
        }

        var group = GroupOf(layers);
        return Enter(group, ReadOnlyStorages(group, mark: true), offload);
    }

    private static LayerScope Enter(object layer, List<Storage> storages, IMemoryOffload offload)
    {
        if (storages.Count == 0)
        {
            return default;                                                // no weights (activations, dropout): not a step in the order
        }

        var previous = TensorOffloading.CurrentLayer;
        TensorOffloading.CurrentLayer = layer;
        if (t_previous is { } before && !ReferenceEquals(before, layer))
        {
            NextLayer.AddOrUpdate(before, layer);
        }

        t_previous = layer;
        if (offload.OffloadedCount == 0)
        {
            return new LayerScope(null, previous, entered: true);
        }

        var token = offload.Stage(storages);
        if (NextLayer.TryGetValue(layer, out var next))
        {
            offload.Prefetch(ReadOnlyStorages(next, mark: false));
        }

        return new LayerScope(token, previous, entered: true);
    }

    /// <summary>Ends a layer's forward: unstages its weights and restores the enclosing layer.</summary>
    public readonly struct LayerScope(IDisposable? token, object? previous, bool entered) : IDisposable
    {
        /// <summary>Whether the layer was entered (its weights staged or its order recorded); false for the empty scope.</summary>
        public bool Entered => entered;

        public void Dispose()
        {
            token?.Dispose();
            if (entered)
            {
                TensorOffloading.CurrentLayer = previous;
            }
        }
    }

    // The tensors a layer only reads while it computes: its parameters (changed by optimizer steps, between passes) and a
    // linear layer's int8 / int4 / bfloat16 weights. Embedding tables are left out: a lookup reads only the rows it needs,
    // which system memory serves without copying the whole table. With `mark`, frozen ones are marked cold.
    private static List<Storage> ReadOnlyStorages(object layer, bool mark)
    {
        var storages = new List<Storage>();
        if (layer is LayerGroup group)
        {
            foreach (var member in group.Layers)
            {
                storages.AddRange(ReadOnlyStorages(member, mark));
            }

            return storages;
        }

        var module = (Module)layer;
        if (module is Embedding)
        {
            return storages;
        }

        foreach (var p in module.Parameters())
        {
            storages.Add(p.Storage);
            if (mark && !p.RequiresGrad)
            {
                MarkCold(p.Storage, OffloadPriority.Frozen);
            }
        }

        if (module is Linear)
        {
            foreach (var b in module.Buffers())
            {
                storages.Add(b.Storage);
                if (mark)
                {
                    MarkCold(b.Storage, OffloadPriority.Frozen);
                }
            }
        }

        return storages;
    }

    /// <summary>Marks <paramref name="storage"/> as moving to system memory before hotter data (never lowers a priority).</summary>
    public static void MarkCold(Storage storage, OffloadPriority priority) => TensorOffloading.MarkCold(storage, priority);

    /// <summary>A parameter's <see cref="Tensor.RequiresGrad"/> changed: frozen weights are cold, trained ones hot.</summary>
    public static void TrainableChanged(Tensor parameter, bool trainable)
    {
        var storage = parameter.Storage;
        if (storage.Backend.Offload is not { } offload)
        {
            return;
        }

        if (!trainable && storage.OffloadPriority == OffloadPriority.Hot)
        {
            offload.SetPriority(storage, OffloadPriority.Frozen);
        }
        else if (trainable && storage.OffloadPriority == OffloadPriority.Frozen)
        {
            offload.SetPriority(storage, OffloadPriority.Hot);
        }
    }

    /// <summary>Between optimizer steps: makes room by moving cold data out, or brings offloaded tensors back (see <see cref="IMemoryOffload.Rebalance"/>).</summary>
    public static void StepBoundary(Device device) => TensorOffloading.StepBoundary(device);

    /// <summary>
    /// Stages each layer's offloaded weights during the backward pass, in the order its nodes run (<paramref name="layers"/>:
    /// the layer whose forward recorded each node), and copies the next layer's in the background. Null when nothing is
    /// offloaded on <paramref name="device"/>.
    /// </summary>
    public static BackwardStaging? ForBackward(Device device, IReadOnlyList<object?> layers)
    {
        if (device.Backend.Offload is not { OffloadedCount: > 0 } offload || !StageWeights)
        {
            return null;
        }

        var groups = new List<(int Index, object Layer)>();
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is { } layer && (groups.Count == 0 || !ReferenceEquals(groups[^1].Layer, layer)))
            {
                groups.Add((i, layer));
            }
        }

        return groups.Count == 0 ? null : new BackwardStaging(offload, groups);
    }

    internal sealed class BackwardStaging(IMemoryOffload offload, List<(int Index, object Layer)> groups) : IBackwardStaging
    {
        private int _group = -1;
        private IDisposable? _token;

        /// <summary>Before the node at <paramref name="index"/> of the order runs.</summary>
        public void Before(int index)
        {
            if (_group + 1 >= groups.Count || groups[_group + 1].Index > index)
            {
                return;
            }

            while (_group + 1 < groups.Count && groups[_group + 1].Index <= index)
            {
                _group++;                                                  // a skipped node may have started a group
            }

            _token?.Dispose();
            _token = offload.Stage(ReadOnlyStorages(groups[_group].Layer, mark: false));
            for (int g = _group + 1; g < groups.Count; g++)
            {
                if (!ReferenceEquals(groups[g].Layer, groups[_group].Layer))
                {
                    offload.Prefetch(ReadOnlyStorages(groups[g].Layer, mark: false));
                    break;
                }
            }
        }

        public void Dispose()
        {
            _token?.Dispose();
            _token = null;
        }
    }
}
