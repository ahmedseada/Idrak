using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// Host-to-device uploads through a pinned staging ring. A synchronous cuMemcpyHtoD from pageable memory runs on the
// legacy stream, which waits for everything queued on the (blocking) work stream first: a 1 KB Tensor.From behind a
// queued decoding step or product costs the whole queue. Small uploads instead copy the values into pinned memory
// (the caller's span is free again on return) and queue cuMemcpyHtoDAsync on the work stream, so they stay ordered
// with every kernel before and after them exactly as before, without the host waiting. Each copy records an event;
// its part of the ring is reused only after the event completed.
internal sealed unsafe partial class CudaBackend
{
    /// <summary>Pinned staging memory for asynchronous uploads (allocated on first use).</summary>
    internal const int StagingBytes = 16 << 20;

    /// <summary>Uploads up to this size go through the staging ring; larger ones keep the synchronous copy.</summary>
    internal const int AsyncUploadBytes = 4 << 20;

    /// <summary>Every upload takes the synchronous copy (for --bench-gemv, to compare the two paths).</summary>
    internal static bool SynchronousUploads;

    /// <summary>Uploads to the device that took the staging ring (for tests and diagnostics).</summary>
    internal long AsyncUploads;

    private readonly Lock _staging = new();
    private IntPtr _stagingHost;
    private int _stagingHead;                                                      // where the newest copy ends
    private bool _stagingFailed;

    // Copies still in flight, oldest first (their ring regions are busy until the event completes), and spare events.
    private readonly Queue<(int Offset, int Bytes, IntPtr Done)> _staged = new();
    private readonly Stack<IntPtr> _spareEvents = new();

    // Queues the copy through the staging ring; false when the upload must take the synchronous path (too large, a
    // graph is being recorded on this thread, or pinned memory is unavailable). Called with the stream gate held.
    private bool TryUploadAsync(ReadOnlySpan<float> source, Storage destination)
    {
        long bytes = (long)source.Length * sizeof(float);
        if (SynchronousUploads || bytes == 0 || bytes > AsyncUploadBytes || Environment.CurrentManagedThreadId == Volatile.Read(ref _captureThread))
        {
            return false;
        }

        lock (_staging)
        {
            if (_stagingHost == IntPtr.Zero)
            {
                if (_stagingFailed || cuMemHostAlloc(out _stagingHost, StagingBytes, HostAllocPortable) != 0)
                {
                    _stagingHost = IntPtr.Zero;
                    _stagingFailed = true;
                    return false;
                }
            }

            int size = (int)bytes, offset = ReserveStaging(size);
            byte* staging = (byte*)_stagingHost + offset;
            fixed (float* p = source)
            {
                Buffer.MemoryCopy(p, staging, size, size);
            }

            Check(cuMemcpyHtoDAsync(P(destination), staging, (nuint)size, _stream), nameof(cuMemcpyHtoDAsync));
            IntPtr done = _spareEvents.Count > 0 ? _spareEvents.Pop() : CreateEvent();
            Check(cuEventRecord(done, _stream), nameof(cuEventRecord));
            _staged.Enqueue((offset, size, done));
            _stagingHead = offset + size;
            AsyncUploads++;
            return true;
        }
    }

    private static IntPtr CreateEvent()
    {
        Check(cuEventCreate(out IntPtr e, EventDisableTiming), nameof(cuEventCreate));
        return e;
    }

    // A free region of `size` bytes in the ring (under the staging lock): retires finished copies, and waits for the
    // oldest pending one while the region would overlap it. In-flight regions are allocated in ring order, so they
    // occupy [oldest, head) — or, once wrapped, [oldest, end) and [0, head).
    private int ReserveStaging(int size)
    {
        while (true)
        {
            while (_staged.Count > 0 && cuEventQuery(_staged.Peek().Done) == 0)
            {
                _spareEvents.Push(_staged.Dequeue().Done);
            }

            if (_staged.Count == 0)
            {
                _stagingHead = 0;
                return 0;
            }

            int tail = _staged.Peek().Offset;
            bool wrapped = _stagingHead <= tail;                          // the newest copy lies before the oldest one
            if (!wrapped && _stagingHead + size <= StagingBytes)
            {
                return _stagingHead;
            }

            if (!wrapped && size <= tail)
            {
                return 0;
            }

            if (wrapped && _stagingHead + size <= tail)
            {
                return _stagingHead;
            }

            var (_, _, oldest) = _staged.Dequeue();
            Check(cuEventSynchronize(oldest), nameof(cuEventSynchronize));
            _spareEvents.Push(oldest);
        }
    }
}
