// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Pinned system-memory buffers ("slots") that a GPU backend copies to and from in the background, queued after the
/// work already issued, so the host can compute on one slot while others are in flight (used by
/// <c>Idrak.Optimizers.HostOptimizer</c>). A device without it (the CPU) returns null from
/// <see cref="Backend.CreateHostStaging"/> and callers copy synchronously.
/// </summary>
public interface IHostStaging : IDisposable
{
    /// <summary>Floats each slot holds.</summary>
    int SlotFloats { get; }

    /// <summary>The slot's memory (wait for its last copy with <see cref="Wait"/> before reading or writing it).</summary>
    Span<float> Slot(int slot);

    /// <summary>Queues a copy of <paramref name="floats"/> values of <paramref name="source"/> from <paramref name="offset"/> into the slot.</summary>
    void Download(Storage source, int offset, int slot, int floats);

    /// <summary>
    /// Whether <see cref="Download(Storage, int, int, int, int)"/> places a copy anywhere in its slot, so several pieces
    /// (a gradient bucket's, plan 23 phase C6) share one slot; false (the default): only at the slot's start.
    /// </summary>
    bool PlacesCopies => false;

    /// <summary>
    /// Queues a copy of <paramref name="floats"/> values of <paramref name="source"/> from <paramref name="offset"/> into
    /// the slot at <paramref name="slotOffset"/>. By default only at the slot's start (<see cref="PlacesCopies"/> false).
    /// </summary>
    /// <exception cref="NotSupportedException">A <paramref name="slotOffset"/> other than 0 where the staging does not place copies.</exception>
    void Download(Storage source, int offset, int slot, int slotOffset, int floats)
    {
        if (slotOffset != 0)
        {
            throw new NotSupportedException($"{GetType().Name} copies to the start of a slot only (PlacesCopies is false).");
        }

        Download(source, offset, slot, floats);
    }

    /// <summary>Queues a copy of the slot's first <paramref name="floats"/> values into <paramref name="destination"/> at <paramref name="offset"/>.</summary>
    void Upload(int slot, Storage destination, int offset, int floats);

    /// <summary>Waits until the slot's last queued copy has finished (no wait when there is none).</summary>
    void Wait(int slot);
}
