using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// IHostStaging on CUDA: page-locked buffers (cuMemHostAlloc) copied with cuMemcpy*Async on the work stream, so each copy
// runs after the work queued before it (a gradient's download after the backward pass that wrote it, a weight's upload
// before the next forward pass reads it) while the host carries on. One event per slot marks its last copy.
internal sealed unsafe partial class CudaBackend
{
    public override IHostStaging? CreateHostStaging(int slots, int slotFloats) => new HostStaging(this, slots, slotFloats);

    private sealed class HostStaging : IHostStaging
    {
        private readonly CudaBackend _backend;
        private readonly IntPtr[] _buffers;
        private readonly IntPtr[] _events;
        private readonly bool[] _pending;

        public HostStaging(CudaBackend backend, int slots, int slotFloats)
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
                    Check(cuMemHostAlloc(out _buffers[i], (nuint)SlotFloats * sizeof(float), HostAllocPortable), nameof(cuMemHostAlloc));
                    _events[i] = CreateEvent();
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

        public void Download(Storage source, int offset, int slot, int floats)
        {
            _backend.MakeCurrent();
            using var use = _backend.UseStream();
            Check(cuMemcpyDtoHAsync((void*)_buffers[slot], P(source) + (ulong)offset * sizeof(float), (nuint)floats * sizeof(float), _backend._stream),
                nameof(cuMemcpyDtoHAsync));
            Record(slot);
        }

        public void Upload(int slot, Storage destination, int offset, int floats)
        {
            _backend.MakeCurrent();
            using var use = _backend.UseStream();
            Check(cuMemcpyHtoDAsync(P(destination) + (ulong)offset * sizeof(float), (void*)_buffers[slot], (nuint)floats * sizeof(float), _backend._stream),
                nameof(cuMemcpyHtoDAsync));
            Record(slot);
        }

        private void Record(int slot)
        {
            Check(cuEventRecord(_events[slot], _backend._stream), nameof(cuEventRecord));
            _pending[slot] = true;
        }

        public void Wait(int slot)
        {
            if (_pending[slot])
            {
                Check(cuEventSynchronize(_events[slot]), nameof(cuEventSynchronize));
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
                    cuEventDestroy(_events[i]);
                    _events[i] = IntPtr.Zero;
                }

                if (_buffers[i] != IntPtr.Zero)
                {
                    cuMemFreeHost(_buffers[i]);
                    _buffers[i] = IntPtr.Zero;
                }
            }
        }
    }
}
