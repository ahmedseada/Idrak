// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Layers.Abstractions;
using Idrak.Models.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// Where a built decoder's layers are and what they are (plan 37, W1 and W2): the first <see cref="DeviceLayers"/> on the
/// device, the rest on the host (the CPU) when the model is split; each layer built on its first use when the load is lazy
/// (<see cref="LazyLayer"/>); and, when the placement follows the memory, layers moved between the device and the host at
/// step boundaries (<see cref="DecodingContext.StepEnded"/>) as the measured free memory changes, through the placement's
/// <see cref="IWeightPlacement.Adjust"/>. Moving a layer moves its weights and its key/value cache; the next norms computed
/// ahead with a block (<see cref="DecoderBlock.NextNorm"/>) and the crossings are linked again after each change.
/// </summary>
internal sealed class DecoderLayers
{
    private readonly Module[] _slots;
    private readonly Module?[] _norms;
    private readonly Module _finalNorm;
    private readonly Func<int, Device, DecoderBlock>? _build;
    private readonly ConditionalWeakTable<DecodingContext, object> _watched = new();
    private DecodingContext? _lastWatched;

    /// <summary>The layers' manager over <paramref name="slots"/> (a block, or a lazy layer standing for one, per layer).</summary>
    /// <param name="slots">The layers in order: <see cref="DecoderBlock"/>s, or <see cref="LazyLayer"/>s when <paramref name="build"/> is given.</param>
    /// <param name="norms">Each layer's attention norm when the blocks are built lazily (made at load: the block before reads it), else null.</param>
    /// <param name="finalNorm">The norm after the last layer.</param>
    /// <param name="device">The model's device.</param>
    /// <param name="host">Where the layers past <paramref name="deviceLayers"/> compute.</param>
    /// <param name="deviceLayers">The layers on the device now.</param>
    /// <param name="build">Builds layer i on a device (a lazy load), or null.</param>
    public DecoderLayers(Module[] slots, Module?[] norms, Module finalNorm, Device device, Device host, int deviceLayers, Func<int, Device, DecoderBlock>? build)
    {
        (_slots, _norms, _finalNorm, Device, Host, _build) = (slots, norms, finalNorm, device, host, build);
        DeviceLayers = device == host ? slots.Length : Math.Clamp(deviceLayers, 0, slots.Length);
        foreach (var slot in slots)
        {
            if (slot is DecoderBlock block)
            {
                block.Layers = this;
            }
            else if (slot is LazyLayer lazy)
            {
                lazy.Layers = this;
            }
        }
    }

    /// <summary>The model's device.</summary>
    public Device Device { get; }

    /// <summary>Where the layers past <see cref="DeviceLayers"/> compute.</summary>
    public Device Host { get; }

    /// <summary>The layers on the device now (the first ones).</summary>
    public int DeviceLayers { get; private set; }

    /// <summary>The decoder's layers.</summary>
    public int Count => _slots.Length;

    /// <summary>The layers built so far (all, unless the load is lazy).</summary>
    public int Built => _slots.Count(s => s is DecoderBlock);

    /// <summary>The model holding the layers (a lazy layer is replaced in it by the block it builds), set once it is made.</summary>
    public Sequential? Model { get; set; }

    /// <summary>The placement the layers follow the memory with, its request at load (the working memory), or null: they stay.</summary>
    public (IWeightPlacement Placement, WeightPlacementRequest Request)? Follows { get; set; }

    /// <summary>The device layer <paramref name="i"/> computes on now.</summary>
    public Device Target(int i) => i >= DeviceLayers ? Host : Device;

    /// <summary>Layer <paramref name="i"/>'s block, built now on its device when the load is lazy and it was not yet.</summary>
    public DecoderBlock Block(int i)
    {
        lock (_slots)
        {
            if (_slots[i] is DecoderBlock built)
            {
                return built;
            }

            var lazy = (LazyLayer)_slots[i];
            var block = _build!(i, Target(i));
            block.Name = lazy.Name;
            block.Layers = this;
            _slots[i] = block;
            Model?.Replace(lazy, block);
            Relink();
            return block;
        }
    }

    /// <summary>Builds every layer not built yet (a lazy load read whole).</summary>
    public void BuildAll()
    {
        for (int i = 0; i < _slots.Length; i++)
        {
            Block(i);
        }
    }

    /// <summary>
    /// Links the built blocks to where they compute: a host layer's input moves there (<see cref="DecoderBlock.PlacedOn"/>),
    /// the last layer's output comes back to the device for the final norm and head when it is the host's
    /// (<see cref="DecoderBlock.OutputTo"/>), and a block computes the next norm ahead only when that norm is on its device.
    /// </summary>
    public void Relink()
    {
        int count = _slots.Length;
        for (int i = 0; i < count; i++)
        {
            if (_slots[i] is not DecoderBlock block)
            {
                continue;
            }

            var target = Target(i);
            block.PlacedOn = target == Device ? null : target;
            block.OutputTo = i == count - 1 && target != Device ? Device : null;
            var next = i + 1 < count ? NormOf(i + 1) : _finalNorm;
            var nextTarget = i + 1 < count ? Target(i + 1) : Device;
            block.NextNorm = next is not null && nextTarget == target ? next as RMSNorm : null;
        }
    }

    // Layer i's attention norm: its block's, or the one made at load for a lazy layer.
    private Module? NormOf(int i) => _slots[i] is DecoderBlock block ? block.AttentionNorm : _norms[i];

    /// <summary>A cached pass of <paramref name="context"/> reached a layer: the layers follow the memory at its step boundaries.</summary>
    public void Watch(DecodingContext context)
    {
        if (Follows is null || ReferenceEquals(_lastWatched, context))
        {
            return;
        }

        _lastWatched = context;
        if (!_watched.TryGetValue(context, out _))
        {
            _watched.Add(context, this);
            context.StepEnded += StepEnded;
        }
    }

    // A step boundary: the placement says how many layers the device holds now; the ones to move go with their caches.
    private void StepEnded(DecodingContext context)
    {
        if (Follows is not { } follows || Device == Host)
        {
            return;
        }

        var request = Measured(follows.Request, context);
        int target = follows.Placement.Adjust(request, DeviceLayers);
        if (target != DeviceLayers)
        {
            MoveTo(Math.Clamp(target, 0, Count), context);
        }
    }

    // The request now: the device's free memory measured (within the user's limit), one built layer's weights and cache as held.
    private WeightPlacementRequest Measured(WeightPlacementRequest atLoad, DecodingContext context)
    {
        var backend = Device.Backend;
        long? free = backend.AvailableMemory();
        var usage = backend.GetMemoryUsage();
        if (usage.Limit is { } limit)
        {
            free = Math.Min(free ?? long.MaxValue, Math.Max(0, limit - usage.InUse));
        }

        long layerBytes = atLoad.LayerBytes, cacheBytes = atLoad.LayerCacheBytes;
        var block = _measured;
        if (block is null)
        {
            // A built block's weights, measured once (rule 75): the layers of a decoder take alike.
            foreach (var slot in _slots)
            {
                if (slot is DecoderBlock built)
                {
                    (_measured, _measuredBytes, block) = (built, Bytes(built), built);
                    break;
                }
            }
        }

        if (block is not null)
        {
            layerBytes = _measuredBytes;
            cacheBytes = context.FindCache(block.Attention)?.Bytes ?? cacheBytes;     // grows with the context
        }

        return atLoad with { AvailableBytes = free, LayerBytes = layerBytes, LayerCacheBytes = cacheBytes };
    }

    private DecoderBlock? _measured;
    private long _measuredBytes;

    // The bytes a block's weights take (its parameters and packed buffers).
    private static long Bytes(Module block) =>
        block.Parameters().Concat(block.Buffers()).Sum(t => (long)t.Storage.Length * sizeof(float));

    /// <summary>
    /// Moves layers so that the first <paramref name="deviceLayers"/> are the device's (between steps of
    /// <paramref name="context"/>, or before any): each moved block's weights go to its new device and its key/value cache
    /// to <paramref name="context"/>'s counterpart there; a layer not built yet is built where it then belongs. Counted as
    /// bus traffic (<see cref="OffloadTraffic.CountMoved"/>).
    /// </summary>
    public void MoveTo(int deviceLayers, DecodingContext? context)
    {
        deviceLayers = Math.Clamp(deviceLayers, 0, Count);
        lock (_slots)
        {
            int from = Math.Min(DeviceLayers, deviceLayers), to = Math.Max(DeviceLayers, deviceLayers);
            var destination = deviceLayers > DeviceLayers ? Device : Host;
            for (int i = from; i < to; i++)
            {
                if (_slots[i] is not DecoderBlock block)
                {
                    continue;                                            // built where it belongs when first used
                }

                long bytes = Bytes(block);
                block.To(destination);
                if (context is not null)
                {
                    bytes += context.MoveCache(block.Attention, destination);
                }

                OffloadTraffic.CountMoved(bytes);
            }

            DeviceLayers = deviceLayers;
            Relink();
        }
    }
}

/// <summary>
/// A decoder layer not built yet (a lazy load, plan 37 W2): its weights are read and its block built on its first forward
/// (<see cref="DecoderLayers.Block"/>), which then takes its place in the model; until then it holds no memory.
/// </summary>
internal sealed class LazyLayer(int index) : Module, ICachedModule
{
    /// <summary>The layer's index in the decoder.</summary>
    public int Index { get; } = index;

    /// <summary>The layers it belongs to.</summary>
    public DecoderLayers Layers { get; set; } = null!;

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Layers.Block(Index).Forward(input);

    /// <inheritdoc />
    public Tensor ForwardCached(Tensor input, DecodingContext context) => Layers.Block(Index).ForwardCached(input, context);

    /// <inheritdoc />
    public override string ToString() => $"LazyLayer({Name}: read on first use)";
}
