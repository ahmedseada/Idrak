using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// Fused inference kernels, incremental-decoding kernels and CUDA Graph capture/replay.
internal sealed unsafe partial class CudaBackend
{
    public override void ScaleMaskSoftmax(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale) =>
        LaunchRows(K("scale_mask_softmax_f32"), rows, P(x), mask is null ? P(x) : P(mask), P(y),
            U(cols), U(Math.Max(maskRows, 1)), F(scale), U(mask is null ? 0 : 1), U(rows));

    public override void LayerNormFused(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps) =>
        LaunchRows(K("layernorm_fused_f32"), rows, P(x), P(gamma), P(beta), P(y), U(cols), F(eps), U(rows));

    public override void LayerNormTrain(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps) =>
        LaunchRows(K("layernorm_train_f32"), rows, P(x), P(gamma), P(beta), P(y), P(stats), U(cols), F(eps), U(rows));

    public override void LayerNormBackward(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        if (dx is not null)
        {
            LaunchRows(K("layernorm_bwd_f32"), rows, P(x), P(gamma), P(dy), P(stats), P(dx), U(cols), U(rows));
        }

        if (dgamma is not null || dbeta is not null)
        {
            const int Chunk = 64;
            Launch(K("layernorm_bwd_params_f32"), (uint)((cols + 255) / 256), (uint)((rows + Chunk - 1) / Chunk), 1, 256, 1,
                P(x), P(dy), P(stats), dgamma is null ? 0UL : P(dgamma), dbeta is null ? 0UL : P(dbeta), U(rows), U(cols), U(Chunk));
        }
    }

    public override void BiasGelu(Storage x, Storage bias, Storage y, int n, int cols) =>
        Launch1D(K("bias_gelu_f32"), n, P(x), P(bias), P(y), U(cols), U(n));

    public override void DecoderMask(Storage position, Storage mask, int rows, int capacity)
    {
        int n = rows * capacity;
        Launch1D(K("decoder_mask_f32"), n, P(position), P(mask), U(capacity), U(n));
    }

    public override void KeyValueWrite(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int n = heads * steps * dim;
        Launch1D(K("kv_write_f32"), n, P(source), P(cache), P(position), U(steps * dim), U(capacity * dim), U(dim), U(n));
    }

    public override void SampleRows(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed)
    {
        float invT = 1f / MathF.Max(temperature, 1e-3f);
        if (topK <= 0 || topK > PtxKernels.CandidateSlots || vocabulary <= 4 * PtxKernels.CandidateSlice)
        {
            LaunchRows(K("sample_rows_f32"), rows, PtxKernels.SamplerThreads, P(logits), P(ids), P(stats), P(step),
                U(vocabulary), U(rowStride), U(rowOffset), F(invT), U(topK), F(topP), F(minP), seed, U(rows), U(rows));
            return;
        }

        // Top-k over a large vocabulary: many blocks each keep their slice's candidates, then one block per row samples
        // among them (every token top-k keeps is a candidate; see PtxKernels.TopKCandidates).
        int blocks = (vocabulary + PtxKernels.CandidateSlice - 1) / PtxKernels.CandidateSlice, slots = blocks * PtxKernels.CandidateSlots;
        var candidates = Allocate(rows * slots, zeroed: false);
        var indices = Allocate(rows * slots, zeroed: false);
        var flags = Allocate(rows, zeroed: true);
        try
        {
            LaunchRows(K("topk_candidates_f32"), rows * blocks, P(logits), P(candidates), P(indices), P(flags),
                U(vocabulary), U(rowStride), U(rowOffset), F(invT), U(topK), U(blocks), U(rows * blocks));
            LaunchRows(K("sample_candidates_f32"), rows, P(logits), P(ids), P(stats), P(step), P(candidates), P(indices), P(flags),
                U(vocabulary), U(rowStride), U(rowOffset), F(invT), U(topK), F(topP), F(minP), seed, U(rows), U(slots), U(rows));
        }
        finally
        {
            candidates.Release();
            indices.Release();
            flags.Release();
        }
    }

    public override void PenalizeRows(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency) =>
        LaunchRows(K("penalize_rows_f32"), rows, P(logits), P(work), P(history), P(length),
            U(vocabulary), U(rowStride), U(rowOffset), U(capacity), U(Math.Min(lastN, capacity)), F(repeat), F(presence), F(frequency), U(rows), U(rows));

    public override void HistoryPush(Storage ids, Storage history, Storage length, int rows, int capacity) =>
        Launch1D(K("history_push_f32"), rows, P(ids), P(history), P(length), U(capacity), U(rows), U(rows));

    public override bool SupportsGraphs => true;

    public override void BeginCapture()
    {
        MakeCurrent();
        _streamGate.EnterWriteLock();                              // other threads' stream work waits for the recording
        try
        {
            lock (_pool)
            {
                if (_captureFree is not null)
                {
                    throw new InvalidOperationException("A graph is already being recorded on this device.");
                }

                _captureFree = [];
                _captureThread = Environment.CurrentManagedThreadId;
            }

            int result = cuStreamBeginCapture(_stream, StreamCaptureModeRelaxed);
            if (result != 0)
            {
                lock (_pool)
                {
                    _captureFree = null;
                    _captureThread = 0;
                }

                Check(result, nameof(cuStreamBeginCapture));
            }
        }
        catch
        {
            _streamGate.ExitWriteLock();
            throw;
        }
    }

    public override (IntPtr Executable, IntPtr Graph, List<Storage> Owned) EndCapture()
    {
        MakeCurrent();
        try
        {
            var owned = TakeCaptureBlocks();
            Check(cuStreamEndCapture(_stream, out IntPtr graph), nameof(cuStreamEndCapture));
            int result = cuGraphInstantiateWithFlags(out IntPtr executable, graph, 0);
            if (result != 0)
            {
                cuGraphDestroy(graph);
                owned.ForEach(o => o.Release());
                Check(result, nameof(cuGraphInstantiateWithFlags));
            }

            GraphCreated();
            return (executable, graph, owned);
        }
        finally
        {
            EndRecording();
        }
    }

    public override List<Storage> AbortCapture()
    {
        MakeCurrent();
        try
        {
            var owned = TakeCaptureBlocks();
            if (cuStreamEndCapture(_stream, out IntPtr graph) == 0 && graph != IntPtr.Zero)
            {
                cuGraphDestroy(graph);
            }

            return owned;
        }
        finally
        {
            EndRecording();
        }
    }

    private void EndRecording()
    {
        if (_streamGate.IsWriteLockHeld)
        {
            _streamGate.ExitWriteLock();
        }
    }

    /// <summary>Ends capture-mode allocation and wraps the blocks freed during capture as storages the graph owns.</summary>
    private List<Storage> TakeCaptureBlocks()
    {
        lock (_pool)
        {
            var owned = new List<Storage>();
            foreach (var (length, bucket) in _captureFree ?? [])
            {
                foreach (ulong pointer in bucket)
                {
                    // Re-count them as in use; releasing the storage later returns them to the shared pool.
                    if (_hostBlocks.ContainsKey(pointer))
                    {
                        _memory.Offloaded(BlockBytes(length));
                    }
                    else
                    {
                        _memory.Reused(BlockBytes(length));
                    }

                    owned.Add(new CudaStorage(this, pointer, length));
                }
            }

            _captureFree = null;
            _captureThread = 0;
            return owned;
        }
    }

    public override void ReplayGraph(IntPtr executable)
    {
        MakeCurrent();
        using var use = UseStream();
        Check(cuGraphLaunch(executable, _stream), nameof(cuGraphLaunch));
    }

    public override void DestroyGraph(IntPtr executable, IntPtr graph)
    {
        MakeCurrent();
        cuGraphExecDestroy(executable);
        cuGraphDestroy(graph);
        GraphDestroyed();
    }
}
