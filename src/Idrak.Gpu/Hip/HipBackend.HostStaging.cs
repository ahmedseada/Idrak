// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// IHostStaging on HIP, as CUDA's (plan 35, HIP parity): page-locked buffers (hipHostMalloc) copied with hipMemcpyAsync on
// the work stream, so each copy runs after the work queued before it (a gradient's download after the backward pass that
// wrote it, a weight's upload before the next forward pass reads it) while the host carries on. One event per slot marks
// its last copy. A runtime without pinned memory or events gives no staging (null), and the callers copy as they do
// without one. UploadRange writes straight to the offset.
internal sealed unsafe partial class HipBackend
{
    /// <summary>The flag of hipHostMalloc's portable memory (pinned for every device of the process).</summary>
    private const uint HostMallocPortable = 1;

    public override IHostStaging? CreateHostStaging(int slots, int slotFloats) =>
        hipHostMalloc != null && hipHostFree != null && hipEventCreate != null && hipEventRecord != null && hipEventSynchronize != null && slots > 0
            ? new HostStaging(this, slots, slotFloats)
            : null;

    // Straight to the offset (a gradient piece of a parameter cut across buckets, plan 23 C8b, as CUDA's).
    public override void UploadRange(ReadOnlySpan<float> source, Storage destination, int offset)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if ((uint)offset > (uint)destination.Length || source.Length > destination.Length - offset)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"Floats {offset} to {offset + source.Length} of a storage of {destination.Length}.");
        }

        if (source.IsEmpty)
        {
            return;
        }

        using var operation = EnterOperation();                            // addresses read until the copy is queued (the move guard)
        MakeCurrent();
        fixed (float* p = source)
        {
            Check(hipMemcpyAsync((void*)(P(destination) + (ulong)offset * sizeof(float)), p, (nuint)source.Length * sizeof(float), MemcpyHostToDevice, _stream),
                nameof(hipMemcpyAsync));
            WaitStream();                                                               // the span is not ours after return
        }
    }

    private sealed class HostStaging : ISlotStaging
    {
        private readonly HipBackend _backend;
        private readonly IntPtr[] _buffers;
        private readonly IntPtr[] _events;
        private readonly bool[] _pending;

        public HostStaging(HipBackend backend, int slots, int slotFloats)
        {
            _backend = backend;
            SlotFloats = Math.Max(1, slotFloats);
            _buffers = new IntPtr[slots];
            _events = new IntPtr[slots];
            _pending = new bool[slots];
            backend.MakeCurrent();
            try
            {
                for (int i = 0; i < slots; i++)
                {
                    void* buffer;
                    Check(hipHostMalloc(&buffer, (nuint)SlotFloats * sizeof(float), HostMallocPortable), nameof(hipHostMalloc));
                    _buffers[i] = (IntPtr)buffer;
                    IntPtr created;
                    Check(hipEventCreate(&created), nameof(hipEventCreate));
                    _events[i] = created;
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public int SlotFloats { get; }

        public Span<float> Slot(int slot) => new((void*)_buffers[slot], SlotFloats);

        public void Download(Storage source, int offset, int slot, int floats) => Download(source, offset, slot, 0, floats);

        // Any place in the slot: a gradient bucket's pieces queued into one slot (plan 23, phase C6).
        public void Download(Storage source, int offset, int slot, int slotOffset, int floats)
        {
            if ((uint)slotOffset > (uint)SlotFloats || floats > SlotFloats - slotOffset)
            {
                throw new ArgumentOutOfRangeException(nameof(slotOffset), slotOffset, $"{floats} floats at {slotOffset} do not fit a slot of {SlotFloats}");
            }

            using var operation = _backend.EnterOperation();                   // addresses read until the copy is queued (the move guard)
            _backend.MakeCurrent();
            Check(hipMemcpyAsync((float*)_buffers[slot] + slotOffset, (void*)(P(source) + (ulong)offset * sizeof(float)), (nuint)floats * sizeof(float),
                MemcpyDeviceToHost, _backend._stream), nameof(hipMemcpyAsync));
            Record(slot);
        }

        public void Upload(int slot, Storage destination, int offset, int floats)
        {
            using var operation = _backend.EnterOperation();                   // addresses read until the copy is queued (the move guard)
            _backend.MakeCurrent();
            Check(hipMemcpyAsync((void*)(P(destination) + (ulong)offset * sizeof(float)), (void*)_buffers[slot], (nuint)floats * sizeof(float), MemcpyHostToDevice,
                _backend._stream), nameof(hipMemcpyAsync));
            Record(slot);
        }

        private void Record(int slot)
        {
            Check(hipEventRecord(_events[slot], _backend._stream), nameof(hipEventRecord));
            _pending[slot] = true;
        }

        public void Wait(int slot)
        {
            if (_pending[slot])
            {
                _backend.MakeCurrent();                                         // a waiting thread of its own (the overlap worker) has no device set yet
                Check(hipEventSynchronize(_events[slot]), nameof(hipEventSynchronize));
                _pending[slot] = false;
            }
        }

        public void Dispose()
        {
            _backend.MakeCurrent();
            for (int i = 0; i < _buffers.Length; i++)
            {
                if (_events[i] != IntPtr.Zero)
                {
                    Wait(i);
                    if (hipEventDestroy != null)
                    {
                        _ = hipEventDestroy(_events[i]);
                    }

                    _events[i] = IntPtr.Zero;
                }

                if (_buffers[i] != IntPtr.Zero)
                {
                    _ = hipHostFree((void*)_buffers[i]);
                    _buffers[i] = IntPtr.Zero;
                }
            }
        }
    }
}
