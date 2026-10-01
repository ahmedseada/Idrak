namespace Idrak.Backends;

/// <summary>How readily a storage moves to system memory when the device fills up (higher moves first).</summary>
internal enum OffloadPriority : byte
{
    /// <summary>Used all the time (activations, trained weights): stays on the device while anything colder can move.</summary>
    Hot = 0,

    /// <summary>Weights no step changes (frozen layers, packed int8 / int4 / bfloat16 weights): read once per pass.</summary>
    Frozen = 1,

    /// <summary>Optimizer state (moments): read and written once per optimizer step.</summary>
    OptimizerState = 2,
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
internal interface IMemoryOffload
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

    /// <summary>Starts copying the offloaded ones among <paramref name="storages"/> to the device in the background.</summary>
    void Prefetch(IReadOnlyList<Storage> storages);

    /// <summary>
    /// Gives the offloaded ones among <paramref name="storages"/> device copies (prefetched ones are waited for, the
    /// others copied now) until the returned token is disposed; null when there is nothing to stage. Kernels then read
    /// the device copies; the storages must not be written meanwhile (weights during a forward or backward pass).
    /// </summary>
    IDisposable? Stage(IReadOnlyList<Storage> storages);
}
