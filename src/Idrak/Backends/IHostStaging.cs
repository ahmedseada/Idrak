namespace Idrak.Backends;

/// <summary>
/// Pinned system-memory buffers ("slots") that a GPU backend copies to and from in the background, queued after the
/// work already issued, so the host can compute on one slot while others are in flight (used by
/// <see cref="Optimizers.HostOptimizer"/>). A device without it (the CPU) returns null from
/// <see cref="Backend.CreateHostStaging"/> and callers copy synchronously.
/// </summary>
internal interface IHostStaging : IDisposable
{
    /// <summary>Floats each slot holds.</summary>
    int SlotFloats { get; }

    /// <summary>The slot's memory (wait for its last copy with <see cref="Wait"/> before reading or writing it).</summary>
    Span<float> Slot(int slot);

    /// <summary>Queues a copy of <paramref name="floats"/> values of <paramref name="source"/> from <paramref name="offset"/> into the slot.</summary>
    void Download(Storage source, int offset, int slot, int floats);

    /// <summary>Queues a copy of the slot's first <paramref name="floats"/> values into <paramref name="destination"/> at <paramref name="offset"/>.</summary>
    void Upload(int slot, Storage destination, int offset, int floats);

    /// <summary>Waits until the slot's last queued copy has finished (no wait when there is none).</summary>
    void Wait(int slot);
}
