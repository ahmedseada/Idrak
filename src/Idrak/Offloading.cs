// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

namespace Idrak;

/// <summary>
/// Where the library meets <see cref="IMemoryOffload"/> (implemented by GPU backends that can keep tensors in system
/// memory): layers stage their offloaded weights on the device while they compute, forward and backward, with the next
/// layer's copied in the background; frozen weights and optimizer state are marked cold; offloaded tensors move between
/// steps. A device without offload support (the CPU) skips all of it, and with nothing offloaded each hook is one check.
/// </summary>
internal static class Offloading
{
    // Tensors live in Idrak.Abstraction and reach the offloading of Idrak's layers through these hooks, registered when the
    // device registry fills (LibraryRegistrations, which runs before Idrak.Gpu's): only devices offload, so the hooks are in
    // place before anything is.
    internal static void ConnectTensors() => TensorOffloading.Register(new TensorHooks());

    // The hooks tensors and modules call (ITensorOffloadingHooks), Idrak's staging behind each.
    private sealed class TensorHooks : ITensorOffloadingHooks
    {
        public IDisposable? EnterForward(Module module, Tensor input) => Offloading.EnterForward(module, input);

        public void TrainableChanged(Tensor parameter, bool trainable) => Offloading.TrainableChanged(parameter, trainable);

        public IBackwardStaging? ForBackward(Device device, IReadOnlyList<object?> layers) => Offloading.ForBackward(device, layers);

        public void MarkFrozen(Module model) => Offloading.MarkFrozen(model);
    }

    // Module.Forward's hook: a scope only when the layer was entered, from this thread's pool (no allocation per layer once
    // warm); nothing otherwise.
    private static IDisposable? EnterForward(Module module, Tensor input) => Enter(module, input) is { Entered: true } scope ? ForwardScope.Of(scope) : null;

    // A layer's scope as Module.Forward disposes it: pooled per thread, each given back when its forward ends.
    private sealed class ForwardScope : IDisposable
    {
        [ThreadStatic]
        private static Stack<ForwardScope>? t_pool;

        private LayerScope _scope;
        private bool _open;

        public static ForwardScope Of(LayerScope scope)
        {
            var pool = t_pool ??= new Stack<ForwardScope>();
            var taken = pool.Count > 0 ? pool.Pop() : new ForwardScope();
            (taken._scope, taken._open) = (scope, true);
            return taken;
        }

        public void Dispose()
        {
            if (!_open)
            {
                return;
            }

            var scope = _scope;
            (_scope, _open) = (default, false);
            scope.Dispose();
            (t_pool ??= new Stack<ForwardScope>()).Push(this);
        }
    }

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
        if (input.Device.Backend.Offload is not { } offload || !Active(offload))
        {
            return default;
        }

        var weights = WeightsOf(module, mark: true);
        if (!weights.Leaf)
        {
            return default;
        }

        if (!StageWeights)
        {
            CountMapped(module, offload);
            return default;
        }

        return Enter(module, weights.Storages, offload);
    }

    // Staging off (--bench-offload): the layer's offloaded weights are read over the bus as it computes (W0's telemetry).
    private static void CountMapped(Module module, IMemoryOffload offload)
    {
        if (offload.OffloadedCount == 0)
        {
            return;
        }

        long bytes = 0;
        foreach (var storage in WeightsOf(module, mark: false).Storages)
        {
            if (offload.IsOffloaded(storage))
            {
                bytes += (long)storage.Length * sizeof(float);
            }
        }

        if (bytes > 0)
        {
            OffloadTraffic.CountMapped(bytes);
            OffloadTraffic.CountLayer(LayerName(module), staged: false, mapped: true);
        }
    }

    // A layer's name in the telemetry: its type and its name ("Linear gate"), the members' for layers one pass computes.
    private static string LayerName(object layer) => layer switch
    {
        LayerGroup group => string.Join(" + ", group.Layers.Select(m => LayerName(m))),
        Module m => string.IsNullOrEmpty(m.Name) ? m.GetType().Name : $"{m.GetType().Name} {m.Name}",
        _ => layer.GetType().Name,
    };

    /// <summary>
    /// <see cref="Enter(Module, Tensor)"/> for layers one fused pass computes together (q/k/v, gate/up, a projection with its
    /// residual and normalization): their offloaded weights are staged as one unit, forward and backward.
    /// </summary>
    public static LayerScope EnterMany(IReadOnlyList<Module> layers, Tensor input)
    {
        if (input.Device.Backend.Offload is not { } offload || !Active(offload) || layers.Count == 0)
        {
            return default;
        }

        if (!StageWeights)
        {
            foreach (var layer in layers)
            {
                CountMapped(layer, offload);
            }

            return default;
        }

        if (layers.Count == 1)
        {
            return Enter(layers[0], input);
        }

        var group = GroupOf(layers);
        return Enter(group, WeightsOf(group, mark: true).Storages, offload);
    }

    private static LayerScope Enter(object layer, Storage[] storages, IMemoryOffload offload)
    {
        if (storages.Length == 0)
        {
            return default;                                                // no weights (activations, dropout): not a step in the order
        }

        var previous = TensorOffloading.CurrentLayer;
        TensorOffloading.CurrentLayer = layer;
        if (t_previous is { } before && !ReferenceEquals(before, layer)
            && (!NextLayer.TryGetValue(before, out var known) || !ReferenceEquals(known, layer)))
        {
            NextLayer.AddOrUpdate(before, layer);                          // only when the order changed: nothing a pass once warm
        }

        t_previous = layer;
        if (offload.OffloadedCount == 0)
        {
            return new LayerScope(null, previous, entered: true);
        }

        // W0's telemetry (OffloadTraffic): whether this pass copied the layer's weights in or read them over the bus.
        long mappedBefore = OffloadTraffic.MappedBytes;
        var token = offload.Stage(storages);
        bool mapped = OffloadTraffic.MappedBytes != mappedBefore;
        if (token is not null || mapped)
        {
            OffloadTraffic.CountLayer(mapped ? LayerName(layer) : "", token is not null, mapped);
        }

        if (NextLayer.TryGetValue(layer, out var next))
        {
            offload.Prefetch(WeightsOf(next, mark: false).Storages);
        }

        return new LayerScope(token, previous, entered: true);
    }

    /// <summary>
    /// Enters every layer of a fused pass through <see cref="ForwardHooks"/> (a sharded run gathers each one's weights,
    /// plan 23 phase C6); the result ends them all, last first. Null when none had anything to do.
    /// </summary>
    public static IDisposable? Gather(IReadOnlyList<Module> layers)
    {
        List<IDisposable>? scopes = null;
        foreach (var layer in layers)
        {
            if (ForwardHooks.Enter(layer) is { } scope)
            {
                (scopes ??= []).Add(scope);
            }
        }

        return scopes is null ? null : new Scopes(scopes);
    }

    private sealed class Scopes(List<IDisposable> scopes) : IDisposable
    {
        public void Dispose()
        {
            for (int i = scopes.Count - 1; i >= 0; i--)
            {
                scopes[i].Dispose();
            }

            scopes.Clear();
        }
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
    // linear layer's int8 / int4 / bfloat16 weights, as the layer says (IOffloadableLayer; else its parameters). Embedding
    // tables are left out: a lookup reads only the rows it needs, which system memory serves without copying the whole
    // table. With `mark`, frozen ones are marked cold.
    private static List<(Tensor Tensor, Storage Storage)> ReadOnlyTensors(object layer, bool mark)
    {
        var tensors = new List<(Tensor, Storage)>();
        if (layer is LayerGroup group)
        {
            foreach (var member in group.Layers)
            {
                tensors.AddRange(ReadOnlyTensors(member, mark));
            }

            return tensors;
        }

        var module = (Module)layer;
        var offloadable = module as IOffloadableLayer;
        foreach (var p in offloadable?.StagedParameters() ?? module.Parameters())
        {
            tensors.Add((p, p.Storage));
            if (mark && !p.RequiresGrad)
            {
                MarkCold(p.Storage, OffloadPriority.Frozen);
            }
        }

        foreach (var b in offloadable?.StagedBuffers() ?? [])
        {
            tensors.Add((b, b.Storage));
            if (mark)
            {
                MarkCold(b.Storage, OffloadPriority.Frozen);
            }
        }

        return tensors;
    }

    // A layer's read-only tensors and their storages, found once (rule 75: no list a layer and token once warm) and checked
    // on each use: every tensor alive with the same storage, else found again (a weight replaced: quantized, merged, moved).
    // Leaf: the layer has no sub-layers (only those stage their own weights); a group's members are leaves.
    private sealed class LayerWeights(Tensor[] tensors, Storage[] storages, bool leaf, bool marked)
    {
        public readonly Tensor[] Tensors = tensors;
        public readonly Storage[] Storages = storages;
        public readonly bool Leaf = leaf;

        // Whether its frozen weights were marked cold when found (a layer first met as the next one, to prefetch, is not).
        public readonly bool Marked = marked;

        public bool Current()
        {
            for (int i = 0; i < Tensors.Length; i++)
            {
                if (Tensors[i].IsDisposed || !ReferenceEquals(Tensors[i].Storage, Storages[i]))
                {
                    return false;
                }
            }

            return true;
        }
    }

    private static readonly ConditionalWeakTable<object, LayerWeights> Weights = new();

    // The layer's read-only storages (cached, see LayerWeights); with `mark`, frozen ones are marked cold when found.
    private static LayerWeights WeightsOf(object layer, bool mark)
    {
        if (Weights.TryGetValue(layer, out var known) && (known.Marked || !mark) && known.Current())
        {
            return known;
        }

        bool leaf = layer is LayerGroup || !((Module)layer).Children().Any();
        var found = leaf ? ReadOnlyTensors(layer, mark) : new List<(Tensor Tensor, Storage Storage)>();
        known = new LayerWeights([.. found.Select(t => t.Tensor)], [.. found.Select(t => t.Storage)], leaf, mark || !leaf);
        Weights.AddOrUpdate(layer, known);
        return known;
    }

    // Marks the frozen weights of every layer of `root` cold, as staging a layer does when it first computes (before any
    // pass, so the first one can already make room).
    private static void MarkFrozen(Module root)
    {
        foreach (var module in root.Descendants().Prepend(root))
        {
            if (module is IOffloadableLayer || !module.Children().Any())
            {
                ReadOnlyTensors(module, mark: true);
            }
        }
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
            _token = offload.Stage(WeightsOf(groups[_group].Layer, mark: false).Storages);
            for (int g = _group + 1; g < groups.Count; g++)
            {
                if (!ReferenceEquals(groups[g].Layer, groups[_group].Layer))
                {
                    offload.Prefetch(WeightsOf(groups[g].Layer, mark: false).Storages);
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
