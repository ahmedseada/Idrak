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
    // The layer whose forward runs on this thread (tagged on the autograd nodes it records, for the backward pass), and
    // the layer staged before it (to learn which layer follows which).
    [ThreadStatic]
    private static Module? t_current;

    [ThreadStatic]
    private static Module? t_previous;

    // The layer that ran after each layer last time: its weights are copied while this one computes.
    private static readonly ConditionalWeakTable<Module, Module> NextLayer = new();

    /// <summary>Off: layers read offloaded weights over PCIe as they compute instead of staging them (for --bench-offload).</summary>
    internal static bool StageWeights = true;

    /// <summary>The layer whose forward runs on this thread while weights are offloaded (null otherwise).</summary>
    public static Module? Current => t_current;

    private static bool Active(IMemoryOffload offload) => ComputeResources.OffloadToHostMemory || offload.OffloadedCount > 0;

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

        var storages = ReadOnlyStorages(module, mark: true);
        if (storages.Count == 0)
        {
            return default;                                                // no weights (activations, dropout): not a step in the order
        }

        var previous = t_current;
        t_current = module;
        if (t_previous is { } before && !ReferenceEquals(before, module))
        {
            NextLayer.AddOrUpdate(before, module);
        }

        t_previous = module;
        if (offload.OffloadedCount == 0)
        {
            return new LayerScope(null, previous, entered: true);
        }

        var token = offload.Stage(storages);
        if (NextLayer.TryGetValue(module, out var next))
        {
            offload.Prefetch(ReadOnlyStorages(next, mark: false));
        }

        return new LayerScope(token, previous, entered: true);
    }

    /// <summary>Ends a layer's forward: unstages its weights and restores the enclosing layer.</summary>
    public readonly struct LayerScope(IDisposable? token, Module? previous, bool entered) : IDisposable
    {
        public void Dispose()
        {
            token?.Dispose();
            if (entered)
            {
                t_current = previous;
            }
        }
    }

    // The tensors a layer only reads while it computes: its parameters (changed by optimizer steps, between passes) and a
    // linear layer's int8 / int4 / bfloat16 weights. Embedding tables are left out: a lookup reads only the rows it needs,
    // which system memory serves without copying the whole table. With `mark`, frozen ones are marked cold.
    private static List<Storage> ReadOnlyStorages(Module module, bool mark)
    {
        var storages = new List<Storage>();
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
    public static void MarkCold(Storage storage, OffloadPriority priority)
    {
        if (storage.OffloadPriority < priority && storage.Backend.Offload is { } offload)
        {
            offload.SetPriority(storage, priority);
        }
    }

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
    public static void StepBoundary(Device device)
    {
        if (device.Backend.Offload is { } offload && Active(offload))
        {
            offload.Rebalance();
        }
    }

    /// <summary>
    /// Stages each layer's offloaded weights during the backward pass, in the order its nodes run (nodes are tagged with
    /// the layer whose forward recorded them), and copies the next layer's in the background. Null when nothing is
    /// offloaded on <paramref name="device"/>.
    /// </summary>
    public static BackwardStaging? ForBackward(Device device, List<Tensor> order)
    {
        if (device.Backend.Offload is not { OffloadedCount: > 0 } offload || !StageWeights)
        {
            return null;
        }

        var groups = new List<(int Index, Module Layer)>();
        for (int i = 0; i < order.Count; i++)
        {
            if (order[i].StageGroup is { } layer && (groups.Count == 0 || !ReferenceEquals(groups[^1].Layer, layer)))
            {
                groups.Add((i, layer));
            }
        }

        return groups.Count == 0 ? null : new BackwardStaging(offload, groups);
    }

    internal sealed class BackwardStaging(IMemoryOffload offload, List<(int Index, Module Layer)> groups) : IDisposable
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
