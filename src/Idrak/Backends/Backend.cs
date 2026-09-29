namespace Idrak.Backends;

internal enum UnaryOp
{
    Sigmoid,
    Tanh,
    Relu,
    Square,
    Abs,
    Exp,
    Log,
    Gelu,
}

internal enum BinaryOp
{
    Add,
    Sub,
    Mul,
}

/// <summary>Shape parameters of a 2-D convolution or pooling window over NCHW data.</summary>
internal readonly record struct ConvGeometry(
    int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW)
{
    public int OH => (H + 2 * PH - KH) / SH + 1;

    public int OW => (W + 2 * PW - KW) / SW + 1;

    /// <summary>Columns of the unfolded matrix: C * KH * KW.</summary>
    public int PatchSize => C * KH * KW;

    /// <summary>Rows of the unfolded matrix: N * OH * OW.</summary>
    public int Positions => N * OH * OW;
}

/// <summary>
/// A reference-counted block of device memory holding <see cref="Length"/> floats.
/// When the last reference is released the block goes back to its backend's pool.
/// </summary>
internal abstract class Storage(Backend backend, int length)
{
    private int _refs = 1;

    public Backend Backend { get; } = backend;

    public int Length { get; } = length;

    public void AddRef() => Interlocked.Increment(ref _refs);

    public void Release()
    {
        if (Interlocked.Decrement(ref _refs) == 0)
        {
            if (!Evicted)
            {
                Backend.Return(this);
            }

            Released?.Invoke();
            Released = null;
            Packed = null;
        }
    }

    /// <summary>Whether the memory was given back by <see cref="Backend.Evict"/> (the values are recomputed on <see cref="Backend.Restore"/>).</summary>
    public bool Evicted { get; internal set; }

    /// <summary>Recomputes the values into this storage after <see cref="Backend.Restore"/> gave it memory again.</summary>
    public Action<Storage>? Recompute { get; internal set; }

    /// <summary>The tensor whose values are recomputed (its own backward step does not read them), or null.</summary>
    public object? RecomputedFor { get; internal set; }

    /// <summary>Called once when the storage is released for good (for what its recomputation keeps, such as a packed copy).</summary>
    public Action? Released { get; internal set; }

    /// <summary>
    /// A bfloat16 copy of the values (<see cref="Backend.PackBFloat16"/> words) kept while the storage is evicted, which
    /// kernels that read bfloat16 inputs use without unpacking it; null when there is none.
    /// </summary>
    public Storage? Packed { get; internal set; }

    /// <summary>Whether a tensor still holds the storage (it has not been released for good).</summary>
    public bool Alive => Volatile.Read(ref _refs) > 0;
}

/// <summary>
/// The math primitives a device must provide. Every operation works on contiguous
/// row-major float32 buffers. Methods named "...Backward", <see cref="Axpy"/>,
/// <see cref="MulAdd"/>, <see cref="SumRows"/> and <see cref="AddBroadcastScalar"/>
/// accumulate into their output (+=) so gradients from several paths add up.
/// </summary>
internal abstract class Backend
{
    public abstract Storage Allocate(int length, bool zeroed);

    public abstract void Return(Storage storage);

    /// <summary>
    /// Gives the storage's memory back to the pool while the storage (shared by every view of it) stays: kernels given it
    /// fail until <see cref="Restore"/> gives it memory again and <paramref name="recompute"/> refills it (null: the values
    /// are gone for good).
    /// </summary>
    public void Evict(Storage storage, Action<Storage>? recompute)
    {
        if (storage.Evicted)
        {
            return;
        }

        storage.Recompute = recompute;
        Return(storage);
        Detach(storage);
        storage.Evicted = true;
    }

    /// <summary>Gives an evicted storage memory again and recomputes its values; false when it was not evicted or cannot be recomputed.</summary>
    public bool Restore(Storage storage)
    {
        if (!storage.Evicted || storage.Recompute is null)
        {
            return false;
        }

        Attach(storage, Allocate(storage.Length, zeroed: false));
        storage.Evicted = false;
        storage.Recompute!(storage);
        return true;
    }

    // Points the storage at no memory (after its memory went back to the pool), or at the memory of a fresh allocation.
    private protected abstract void Detach(Storage storage);

    private protected abstract void Attach(Storage storage, Storage fresh);

    public abstract MemoryUsage GetMemoryUsage();

    /// <summary>Frees pooled blocks that no tensor is using.</summary>
    public abstract void ReleaseCachedMemory();

    public abstract void Upload(ReadOnlySpan<float> source, Storage destination);

    public abstract void Download(Storage source, Span<float> destination);

    /// <summary>Copies destination.Length floats starting at element <paramref name="offset"/>.</summary>
    public abstract void DownloadRange(Storage source, int offset, Span<float> destination);

    public abstract void Fill(Storage y, int n, float value);

    public abstract void Copy(Storage x, Storage y, int n);

    /// <summary>y = op(x).</summary>
    public abstract void Unary(UnaryOp op, Storage x, Storage y, int n);

    /// <summary>dx += dy * op'(x), where y = op(x) is passed in for ops whose derivative is cheaper from y.</summary>
    public abstract void UnaryBackward(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n);

    /// <summary>c = a op b, element-wise.</summary>
    public abstract void Binary(BinaryOp op, Storage a, Storage b, Storage c, int n);

    /// <summary>y = alpha * x + beta.</summary>
    public abstract void Affine(Storage x, Storage y, int n, float alpha, float beta);

    /// <summary>y += alpha * x.</summary>
    public abstract void Axpy(Storage x, Storage y, int n, float alpha);

    /// <summary>c += a * b, element-wise.</summary>
    public abstract void MulAdd(Storage a, Storage b, Storage c, int n);

    /// <summary>c[r, j] = a[r, j] + v[j].</summary>
    public abstract void AddRowVector(Storage a, Storage v, Storage c, int rows, int cols);

    /// <summary>y[j] += sum over r of x[r, j].</summary>
    public abstract void SumRows(Storage x, Storage y, int rows, int cols);

    /// <summary>result[0] = scale * sum(x).</summary>
    public abstract void Sum(Storage x, Storage result, int n, float scale);

    /// <summary>y[offset] += alpha * x[0] (used to accumulate scalar losses without leaving the device).</summary>
    public abstract void AxpyAt(Storage x, Storage y, int offset, float alpha);

    /// <summary>y[i] += scale * s[0].</summary>
    public abstract void AddBroadcastScalar(Storage s, Storage y, int n, float scale);

    /// <summary>
    /// c[m, n] = op(a) * op(b) + beta * c, where op(a) is [m, k] and op(b) is [k, n].
    /// With <paramref name="transA"/> a is stored as [k, m]; with <paramref name="transB"/> b is stored as [n, k].
    /// </summary>
    public void MatMul(Storage a, Storage b, Storage c, int m, int n, int k, bool transA, bool transB, float beta) =>
        BatchedMatMul(a, b, c, 1, m, n, k, transA, transB, beta);

    /// <summary>
    /// y = x · w for many rows (prompts) with packed weights w (<paramref name="kind"/> as in
    /// <see cref="PackedMatMulMany"/>), expanding w as it is read instead of into a float copy. Returns false when the
    /// device has no such kernel or the shape is too small for it (callers then expand w first).
    /// </summary>
    public virtual bool PackedMatMulLarge(int kind, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k) => false;

    /// <summary>
    /// y = (act(gate) · up) · w for few rows with packed weights w (<paramref name="kind"/> as in
    /// <see cref="PackedMatMulMany"/>; activation 0 = SiLU, 1 = GELU tanh): the gated feed-forward's down projection
    /// without a separate activation pass. Returns false when the device has no fused version.
    /// </summary>
    public virtual bool PackedMatMulGated(int kind, int activation, Storage gate, Storage up, Storage packed, Storage? scales, Storage y,
        int m, int n, int k) => false;

    /// <summary>
    /// y = x · w for few rows with packed weights w (<paramref name="kind"/> as in <see cref="PackedMatMulMany"/>), then
    /// sum = residual + y and normalized = its RMS normalization · (gain + offset) per row (a residual addition and the
    /// next normalization) in the same pass. Returns false when the device has no fused version.
    /// </summary>
    public virtual bool PackedMatMulAddRmsNorm(int kind, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k,
        Storage residual, Storage sum, Storage gain, Storage normalized, float eps, float offset) => false;

    /// <summary>
    /// Several few-row products of packed weights sharing one input x [m, k]: y_j = x · w_j (+ bias_j), with w_j int8
    /// (<paramref name="kind"/> 0, as in <see cref="Int8MatMul"/>), 4-bit (1, <see cref="Int4MatMul"/>) or bfloat16 (2,
    /// <see cref="BFloat16MatMul"/>; no scales). Returns false when the device has no single-pass version (callers then
    /// run the products one by one).
    /// </summary>
    public virtual bool PackedMatMulMany(int kind, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products) => false;

    /// <summary>
    /// <see cref="PackedMatMulMany"/> for a feed-forward block's gate and up projections (two products of equal widths),
    /// also writing hidden = act(gate) · up (activation 0 = SiLU, 1 = GELU tanh, 2 = ReLU) in the same pass. Returns
    /// false when the device has no fused version.
    /// </summary>
    public virtual bool PackedMatMulGatedPair(int kind, int activation, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, Storage hidden) => false;

    /// <summary>
    /// Several products sharing one input: y_j = a · w_j (+ bias_j) for a [m, k] and w_j [k, n_j] (the query, key and
    /// value projections, say). The default computes them one by one; devices may do them in one pass.
    /// </summary>
    public virtual void MatMulMany(Storage a, int m, int k, ReadOnlySpan<(Storage Weight, Storage? Bias, Storage Output, int Columns)> products)
    {
        foreach (var (weight, bias, output, columns) in products)
        {
            MatMul(a, weight, output, m, columns, k, false, false, 0f);
            if (bias is not null)
            {
                AddRowVector(output, bias, output, m, columns);
            }
        }
    }

    /// <summary>
    /// c = a·b + bias (bias [n] added to every row) in one pass, for a [m, k] and b [k, n] as stored. Returns false when
    /// the device has no such pass for these sizes (callers then multiply and add the bias separately).
    /// </summary>
    public virtual bool MatMulBias(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k) => false;

    /// <summary>
    /// c = beta·c + a·op(b) + u·op(v) in one product (a LoRA adapter's term as one more k step): a [m, k], b [k, n]
    /// (transposed: [n, k]), u [m, rank], v [rank, n] (with <paramref name="transB"/>: [n, rank]), rank ≤ 32, all rows
    /// contiguous. Returns false when the device has no such pass (callers then compute the two products separately).
    /// </summary>
    public virtual bool MatMulLowRank(Storage a, Storage b, Storage c, int m, int n, int k, bool transB, float beta, Storage u, Storage v, int rank) => false;

    /// <summary>
    /// c = beta·c + a · Wᵀ (+ u · vᵀ) with W a bfloat16 weight [n, k] as <see cref="Layers.BFloat16Weight"/> packs it (two
    /// values per word along k), read as stored: the input gradient through a frozen bfloat16 layer (and its adapter's
    /// dt · Aᵀ with u = dt [m, rank], v = A [n, rank]) without expanding the weight to float. u null: no low-rank term.
    /// False when the device has no such kernel (callers then expand the weight).
    /// </summary>
    public virtual bool BFloat16TransposedMatMul(Storage a, Storage packed, Storage c, int m, int n, int k, float beta, Storage? u, Storage? v, int rank) => false;

    /// <summary>
    /// k rounded up for <see cref="Float8QuantizeWeight"/>'s values (0 when the device has no FP8 products): the values
    /// take n · paddedK bytes, n · paddedK / 4 floats of storage.
    /// </summary>
    public virtual int Float8PaddedK(int k) => 0;

    /// <summary>
    /// Quantizes a frozen weight w [k, n] (float32) once for <see cref="Float8MatMul"/>: FP8 (e4m3) values, k-major per
    /// column ([n, paddedK] bytes), and one scale per column [n]. Returns false when unsupported.
    /// </summary>
    public virtual bool Float8QuantizeWeight(Storage w, int k, int n, Storage values, Storage scales) => false;

    /// <summary>
    /// y = beta·y + x · w on FP8 tensor cores for x [m, k] (quantized per row as it is read, one scale each) and a weight
    /// quantized by <see cref="Float8QuantizeWeight"/>. Returns false when unsupported.
    /// </summary>
    public virtual bool Float8MatMul(Storage x, int m, int k, Storage values, Storage scales, int n, Storage y, float beta) => false;

    /// <summary>
    /// Products of one input x [m, k] through 1-3 packed layers (<paramref name="kind"/> as in <see cref="PackedMatMulMany"/>),
    /// each with a low-rank term: y_j = x · w_j + u_j · v_j for u_j [m, rank] and v_j [rank, n_j], rank ≤ 32, in one pass.
    /// Returns false when the device has no such pass.
    /// </summary>
    public virtual bool PackedMatMulLowRank(int kind, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage Output, int Columns, Storage U, Storage V)> products, int rank) => false;

    /// <summary>
    /// c = beta·c + op(a)·op(b) on bfloat16 tensor cores, with row strides: a [m, k] stored with <paramref name="lda"/>
    /// floats per row (transposed: [k, m] rows), b [k, n] with <paramref name="ldb"/> (transposed: [n, k] rows), c [m, n]
    /// with <paramref name="ldc"/>; offsets in elements. <paramref name="bias"/> [n] is added to every row (modes None and
    /// Gelu). <see cref="GemmEpilogue.Gelu"/> writes gelu(product + bias) and, when <paramref name="aux"/> is given, the
    /// pre-activations into it (same layout as c); <see cref="GemmEpilogue.GeluGradient"/> multiplies the product by
    /// gelu'(aux). Returns false when the device has no tensor cores (callers use the composed operations).
    /// </summary>
    public virtual bool GemmStrided(Storage a, long aOffset, int lda, bool transA, Storage b, long bOffset, int ldb, bool transB,
        Storage c, long cOffset, int ldc, int m, int n, int k, float beta, Storage? bias = null, GemmEpilogue epilogue = GemmEpilogue.None,
        Storage? aux = null, long auxOffset = 0) => false;

    /// <summary>
    /// Causal attention (positions c ≤ t) read in place from [batch, steps, *] rows: head h (= kv · group + g) of the
    /// queries at q[qOffset + (b·steps + t)·qRow + h·dim], keys and values of kv head kv at k / v[offset + (b·steps + c)·kRow
    /// + kv·dim]; writes y [batch, steps, heads·dim] and, when given, the log-sum-exp [batch·kvHeads, group·steps] for the
    /// backward pass. Returns false when the device has no such kernels (callers rearrange the heads and use
    /// <see cref="AttentionTiled"/>).
    /// </summary>
    public virtual bool AttentionStrided(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage? logSumExp, int batch, int kvHeads, int group, int steps, int dim, float scale) => false;

    /// <summary>
    /// Gradients of <see cref="AttentionStrided"/>: adds to dq, dk, dv laid out as q, k, v (same row strides, their own
    /// offsets) given y, the log-sum-exp and dOutput (y's layout).
    /// </summary>
    public virtual bool AttentionStridedBackward(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset,
        int batch, int kvHeads, int group, int steps, int dim, float scale) => false;

    /// <summary>y[j] += Σ_r x[offset + r·ld + j] for j &lt; cols (column sums of a strided block; the bias gradient of a slice).</summary>
    public virtual void SumColumns(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        throw new NotSupportedException($"{GetType().Name} has no strided column sums.");
    }

    /// <summary>
    /// Until disposed, 8-bit products on this device reuse the quantized form of an operand they have already quantized
    /// (same memory, layout and size), for callers that multiply one unchanged tensor several times (the query, key and
    /// value projections of one input). Null when the device quantizes nothing.
    /// </summary>
    public virtual IDisposable? ReuseQuantizedOperands() => null;

    /// <summary>output = residual + dropout(x) (the mask of <see cref="Dropout"/> with <paramref name="seed"/>).</summary>
    public virtual void AddDropout(Storage residual, Storage x, Storage output, int n, float p, uint seed)
    {
        Dropout(x, output, n, p, seed);
        Axpy(residual, output, n, 1f);
    }

    /// <summary><see cref="MatMul"/> for <paramref name="batch"/> independent, contiguous matrix triples.</summary>
    public abstract void BatchedMatMul(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta);

    /// <summary>Row-wise softmax (or log-softmax) over the last dimension: y[r, :] = softmax(x[r, :]).</summary>
    public abstract void Softmax(Storage x, Storage y, int rows, int cols, bool log);

    /// <summary>
    /// Softmax: dx += y * (dy - Σ dy·y). Log-softmax: dx += dy - exp(y) * Σ dy. Sums are per row; y is the forward output.
    /// </summary>
    public abstract void SoftmaxBackward(Storage y, Storage dy, Storage dx, int rows, int cols, bool log);

    /// <summary>y[r] = index of the largest element of row r.</summary>
    public abstract void ArgMax(Storage x, Storage y, int rows, int cols);

    /// <summary>
    /// y[r] = 1 when the prediction in row r is right, else 0. For one column: (p ≥ threshold) == (t ≥ 0.5);
    /// otherwise argmax(p) == argmax(t).
    /// </summary>
    public abstract void ClassMatch(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold);

    // Grouped normalization. Data is viewed as [outer, groups, inner]; group g holds the M = outer * inner
    // elements x[o, g, i]. BatchNorm uses groups = channels; LayerNorm uses outer = 1, groups = rows, inner = features.

    /// <summary>Per-group mean, (biased) variance and 1 / sqrt(variance + eps).</summary>
    public abstract void NormStats(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps);

    /// <summary>y = (x - mean[g]) * invStd[g].</summary>
    public abstract void NormApply(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner);

    /// <summary>dx += invStd[g] / M * (M * dxhat - sum1[g] - xhat * sum2[g]).</summary>
    public abstract void NormBackward(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner);

    /// <summary>
    /// y (+)= x * scale[g] + shift[g] with g = (i / inner) % groups; a null scale means 1, a null shift means 0.
    /// </summary>
    public abstract void GroupScaleShift(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate);

    /// <summary>sumA[g] += Σ a; sumAB[g] += Σ a·b over each group (see NormStats for the layout). b/sumAB may be null.</summary>
    public abstract void GroupReduce(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner);

    /// <summary>y = 1 / sqrt(x + eps).</summary>
    public abstract void InvSqrt(Storage x, Storage y, int n, float eps);

    /// <summary>Embedding lookup: y[i, :] = table[indices[i], :] for count indices of width dim.</summary>
    public abstract void Gather(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary);

    /// <summary><see cref="Gather"/> from a bfloat16 table packed as in <see cref="BFloat16MatMul"/> ([vocabulary, dim]).</summary>
    public abstract void GatherBFloat16(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary);

    /// <summary>dtable[indices[i], :] += dy[i, :].</summary>
    public abstract void ScatterAdd(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary);

    /// <summary>Unfolds image patches: cols[(n, oh, ow), (c, kh, kw)] = x[n, c, oh*sh - ph + kh, ow*sw - pw + kw] (0 outside).</summary>
    public abstract void Im2Col(Storage x, Storage cols, in ConvGeometry g);

    /// <summary>The adjoint of <see cref="Im2Col"/>: dx += fold(dcols).</summary>
    public abstract void Col2Im(Storage dcols, Storage dx, in ConvGeometry g);

    /// <summary>Max pooling; argmax receives the flat input index of each maximum (as raw int bits).</summary>
    public abstract void MaxPool(Storage x, Storage y, Storage argmax, in ConvGeometry g);

    /// <summary>dx[argmax[i]] += dy[i].</summary>
    public abstract void MaxPoolBackward(Storage dy, Storage argmax, Storage dx, int count);

    /// <summary>
    /// y (+)= x permuted: output element at coordinates (c0..c[r-1]) of <paramref name="outShape"/> comes from
    /// input offset Σ c_k * inStrides[k] (the input strides already reordered by the permutation). Rank ≤ 6.
    /// </summary>
    public abstract void Permute(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate);

    /// <summary>
    /// Strided block copy: dst[dstOffset + r*dstStride + c] (+)= src[srcOffset + r*srcStride + c] for r &lt; rows, c &lt; cols.
    /// Implements narrow, concatenation and their gradients.
    /// </summary>
    public abstract void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate);

    /// <summary>y[o, i] (+)= scale * Σ_d x[o, d, i] for a [outer, dim, inner] view.</summary>
    public abstract void SumAxis(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate);

    /// <summary>dx[o, d, i] += scale * dy[o, i] (the gradient of <see cref="SumAxis"/>).</summary>
    public abstract void BroadcastAxis(Storage dy, Storage dx, int outer, int dim, int inner, float scale);

    /// <summary>SGD with optional momentum: v = momentum * v + g; p -= lr * v (v is null when momentum is 0).</summary>
    public abstract void SgdStep(Storage p, Storage g, Storage? v, int n, float lr, float momentum);

    /// <summary>Adam: m, v moments updated in place; p -= lr * m / (sqrt(v) + eps). lr is already bias-corrected.</summary>
    public abstract void AdamStep(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps);

    /// <summary>
    /// <see cref="AdamStep"/> with 8-bit moments (see <see cref="Optimizers.AdamW8Bit"/>): m and v hold one byte per element
    /// (n bytes, packed four per float), codes into <paramref name="map"/> (256 signed values for m, then 256 unsigned for
    /// v, both in [-1, 1]) scaled per block of <see cref="Optimizers.AdamW8Bit.BlockSize"/> elements by
    /// <paramref name="absMax"/> (the blocks' m scales, then their v scales). The moments are decoded, updated, and
    /// encoded again to the nearest code with new block scales.
    /// </summary>
    /// <remarks>The gradient is multiplied by <paramref name="gradientScale"/> as it is read (gradient clipping) and the
    /// parameter by <paramref name="decay"/> before the update (decoupled weight decay, 1 - lr·λ).</remarks>
    public abstract void AdamStep8Bit(Storage p, Storage g, Storage m, Storage v, Storage absMax, Storage map, int n, float lr, float beta1, float beta2, float eps,
        float gradientScale, float decay);

    /// <summary>total[0] += Σ x² over <paramref name="n"/> elements (a gradient norm without temporary tensors).</summary>
    public abstract void SumSquares(Storage x, Storage total, int n);

    /// <summary>
    /// AdamW over many tensors in a few passes: the global gradient norm clipped to <paramref name="maxNorm"/> (0: no
    /// clipping), p ← p · <paramref name="decay"/>, then Adam with the bias-corrected <paramref name="lr"/>, as
    /// <see cref="AdamStep"/> computes it; with <paramref name="zeroGradients"/> the gradients are zeroed for the next step.
    /// <paramref name="cache"/> keeps the device tables between calls (dispose it when done). False when the device has no
    /// such pass.
    /// </summary>
    public virtual bool FusedAdamW(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients) => false;

    /// <summary>
    /// factor[0] = min(1, maxNorm / √sumSquares[0]) (1 when the sum is 0): the gradient-clipping factor computed where the
    /// gradients are, so clipping needs no host read of the norm.
    /// </summary>
    public abstract void ClipFactor(Storage sumSquares, Storage factor, float maxNorm);

    /// <summary>Inverted dropout: y = keep(i) ? x / (1 - p) : 0, where keep(i) comes from <see cref="DropoutMask"/>.</summary>
    public abstract void Dropout(Storage x, Storage y, int n, float p, uint seed);

    /// <summary>dx += keep(i) ? dy / (1 - p) : 0, regenerating the same mask from the seed.</summary>
    public abstract void DropoutBackward(Storage dy, Storage dx, int n, float p, uint seed);

    public abstract void Synchronize();

    // ---------------------------------------------------------------- fused inference kernels

    /// <summary>y[r, :] = softmax(scale * x[r, :] + mask[r % maskRows, :]) (mask optional).</summary>
    public abstract void ScaleMaskSoftmax(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale);

    /// <summary>y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta over the last dimension.</summary>
    public abstract void LayerNormFused(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps);

    /// <summary><see cref="LayerNormFused"/> that also stores each row's mean (stats[r]) and 1 / sqrt(var + eps) (stats[rows + r]).</summary>
    public abstract void LayerNormTrain(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps);

    /// <summary>
    /// Gradients of <see cref="LayerNormTrain"/> given dy: adds to dx (when given), dgamma += Σ_r dy ∘ x̂ and dbeta += Σ_r dy.
    /// </summary>
    public abstract void LayerNormBackward(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols);

    /// <summary>y[i] = gelu(x[i] + bias[i % cols]).</summary>
    public abstract void BiasGelu(Storage x, Storage bias, Storage y, int n, int cols);

    /// <summary>
    /// y[m, n] = x[m, k] · w, where w[k, j] = q[k, j] · scales[j] and q holds signed bytes packed four per 32-bit element
    /// along each row (rows padded to ceil(n / 4) elements). Suited to few rows (token-by-token decoding).
    /// </summary>
    public abstract void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k);

    /// <summary>
    /// y[m, n] = x[m, k] · w[k, n] with w stored as bfloat16 pairs packed into 32-bit words along each row (word c of
    /// row r holds columns 2c in its low half and 2c + 1 in its high half; rows have ⌈n / 2⌉ words).
    /// </summary>
    public abstract void BFloat16MatMul(Storage x, Storage packed, Storage y, int m, int n, int k);

    /// <summary>w[k, n] = the float32 values of bfloat16 weights packed as in <see cref="BFloat16MatMul"/>.</summary>
    public abstract void BFloat16Dequantize(Storage packed, Storage w, int k, int n);

    /// <summary>
    /// Rounds x [n] to bfloat16 (to nearest, ties to even), two values per word as <see cref="BFloat16Dequantize"/> reads
    /// them back with k = 1: packed holds (n + 1) / 2 words.
    /// </summary>
    public abstract void PackBFloat16(Storage x, Storage packed, int n);

    /// <summary>w[k, n] = q[k, n] · scales[n] (see <see cref="Int8MatMul"/> for the packing).</summary>
    public abstract void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n);

    /// <summary>
    /// y[m, n] = x[m, k] · w with 4-bit weights: w[r, j] = q[r, j] · scales[r / 32, j], where q holds signed nibbles packed
    /// eight per 32-bit word along each row (nibble c of word w is column 8w + c; rows have ⌈n / 8⌉ words) and scales has
    /// one row of 8·⌈n / 8⌉ values per group of 32 weight rows.
    /// </summary>
    public abstract void Int4MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k);

    /// <summary>w[k, n] = the float values of 4-bit weights packed as in <see cref="Int4MatMul"/>.</summary>
    public abstract void Int4Dequantize(Storage q, Storage scales, Storage w, int k, int n);

    // Int8 KV cache: each cached row (one head, one position) is dim bytes packed four per element, words = ceil(dim / 4)
    // elements, with one scale per row in scales[head, position].

    /// <summary>y = x · inv per row, inv[r] = 1 / sqrt(mean(x[r]²) + eps) (stored for the backward pass).</summary>
    public abstract void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps);

    /// <summary>dx += inv[r] · (dy - y · mean(dy · y)) per row, where y is the normalized forward output.</summary>
    public abstract void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols);

    /// <summary>
    /// Rotary position embedding of x [rows = batch·steps·heads, dim] into y (which must already hold x): pair p of a row at
    /// step t rotates by the angle whose cos/sin are cos/sin[positions[t], p]. Pairs are (2p, 2p+1) when interleaved,
    /// else (p, p + half). sign = -1 rotates backwards (the gradient).
    /// </summary>
    public abstract void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign);

    /// <summary>y = x · inv · (gain[c] + offset) per row with inv = 1 / sqrt(mean(x²) + eps): normalization and gain in one pass (inference).</summary>
    public abstract void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset);

    /// <summary>sum = a + b and y = RMS-normalized sum · (gain[c] + offset) per row, in one pass (inference).</summary>
    public abstract void AddRmsNormAffine(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset);

    /// <summary>
    /// RMS normalization with gain of each row (a head's vector), then the rotary embedding as <see cref="Rope"/> with
    /// sign 1 (rows are batch·steps·heads; dimensions beyond 2·half are only normalized). Inference.
    /// </summary>
    public abstract void RmsNormRope(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved);

    /// <summary>
    /// <see cref="RmsNormRope"/> for two tensors sharing the positions and rotary tables (queries and keys): rows1 rows
    /// of x (heads per step: heads) and rows2 of x2 (heads2). Devices may do both in one pass.
    /// </summary>
    public virtual void RmsNormRopePair(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved)
    {
        RmsNormRope(x, gain, cos, sin, positions, y, rows1, cols, eps, offset, heads, steps, half, interleaved);
        RmsNormRope(x2, gain2, cos, sin, positions, y2, rows2, cols, eps2, offset2, heads2, steps, half, interleaved);
    }

    /// <summary>
    /// The attention layer's queries, keys and values [batch, steps, heads·cols] (its projections) put in the layouts
    /// attention reads, in one pass (inference): query heads RMS-normalized with gain (when <paramref name="gainQ"/> is
    /// set; keys then with <paramref name="gainK"/>) and rotated as <see cref="Rope"/> (half 0: no rotation) into
    /// yq [batch, heads, steps, cols]; keys the same and values unchanged into yk and yv [batch, kvHeads, capacity, stride]
    /// at row position[0] + s (a cache; row s when <paramref name="position"/> is null), as floats or bfloat16 pairs.
    /// Returns false when the device has no such kernel (callers then run the steps one by one).
    /// </summary>
    public virtual bool NormRopeHeads(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride, bool bfloat16) => false;

    /// <summary>
    /// Token cross-entropy for language-model training, per row r of logits [rows, vocabulary] with target class t_r and
    /// weight w_r (0 masks the row): losses[r] = w_r · (logsumexp(x_r) - x_r[t_r]); the logits are overwritten with their
    /// gradient scale · w_r · (softmax(x_r) - onehot(t_r)). Targets and weights are float arrays of rows.
    /// </summary>
    public abstract void SoftmaxCrossEntropyRows(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale);

    /// <summary>y = act(gate) · up element-wise; kind 0 = SiLU, 1 = GELU (tanh approximation), 2 = ReLU.</summary>
    public abstract void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind);

    /// <summary>dgate += dy · up · act'(gate) when flags has bit 0, dup += dy · act(gate) when it has bit 1; bits 2 and 3 write dgate and dup (= instead of +=).</summary>
    public abstract void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags);

    /// <summary>
    /// <see cref="GatedActivation"/> with bfloat16 words (two values each, low half first, as <see cref="PackBFloat16"/>
    /// writes them): gate and up read from <paramref name="packedGate"/> and <paramref name="packedUp"/> when flags has
    /// bit 0 (else from <paramref name="gate"/> and <paramref name="up"/>); y written when it has bit 2, gate and up packed
    /// when it has bit 1, y packed when it has bit 3. Storages a flag does not use may be any storage.
    /// </summary>
    public abstract void GatedActivationPacked(Storage gate, Storage up, Storage packedGate, Storage packedUp, Storage y, Storage packedY, int n, int kind, int flags);

    /// <summary><see cref="GatedActivationBackward"/> reading gate and up as bfloat16 words.</summary>
    public abstract void GatedActivationBackwardPacked(Storage packedGate, Storage packedUp, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags);

    /// <summary>Quantizes source [heads·steps, dim] into the int8 cache at positions position[0] + step.</summary>
    public abstract void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim);

    /// <summary>y[r, t, c] = scales[r, c] · Σ_d q[r, t, d] · keys[r, c, d] for every cached position c.</summary>
    public abstract void AttentionScoresInt8(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim);

    /// <summary>y[r, t, d] = Σ_c weights[r, t, c] · scales[r, c] · values[r, c, d].</summary>
    public abstract void AttentionContextInt8(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim);

    /// <summary>
    /// Attention over a key/value cache filled up to the device position: for head h and row i of q [heads, rowsPerHead,
    /// dim], y[h, i] = Σ_c softmax(scale · q[h, i] · keys[h, c]) · values[h, c] over positions c = 0 … position[0] +
    /// (i % steps) (the causal limit of that row's step). Keys and values are [heads, capacity, dim]; unfilled
    /// positions are never read, so the cost follows the context length rather than the capacity.
    /// </summary>
    public abstract void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale);

    /// <summary>
    /// <see cref="AttentionDecode"/> (tiled: <paramref name="tiled"/>, for many query rows) over an int8 cache: keys and
    /// values [heads, capacity, ⌈dim / 4⌉ words] of packed bytes with one scale per cached row (keyScales, valueScales
    /// [heads, capacity]).
    /// </summary>
    public abstract void AttentionInt8(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled);

    /// <summary>
    /// <see cref="AttentionDecode"/> (tiled: <paramref name="tiled"/>) over a bfloat16 cache: keys and values
    /// [heads, capacity, ⌈dim / 2⌉ words], dimension d in the low (even d) or high (odd d) half of word d / 2.
    /// </summary>
    public abstract void AttentionBFloat16(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled);

    /// <summary><see cref="KeyValueWrite"/> into a bfloat16 cache (values rounded to nearest, ties to even).</summary>
    public abstract void KeyValueWriteBFloat16(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim);

    /// <summary>
    /// Gradient of <see cref="AttentionTiled"/> with causal offset 0 (training): given the output, each row's log-sum-exp
    /// and dOutput, adds to dq [heads, rowsPerHead, dim] and dkeys, dvalues [heads, capacity, dim]. The attention weights
    /// are recomputed, never stored.
    /// </summary>
    public abstract void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale);

    /// <summary>
    /// The same attention as <see cref="AttentionDecode"/> for many query rows at once (a prompt, a training sequence),
    /// tiled so query rows share each key and value read; also writes each row's log-sum-exp of the scaled scores to
    /// <paramref name="logSumExp"/> [heads, rowsPerHead] when given.
    /// </summary>
    /// <summary>
    /// Causal attention over packed sequences (training): as <see cref="AttentionTiled"/> with offset 0 and keys and values
    /// [heads, steps, dim], except that several sequences share each row of <paramref name="steps"/> positions, so row i of
    /// head h sees positions c with starts[b·steps + t] ≤ c ≤ t, where t = i % steps and b = h / <paramref name="headsPerRow"/>
    /// is the packed row; starts and ends hold, per position of each packed row, where its sequence begins and stops
    /// (exclusive), as floats. Writes the log-sum-exp when given. Returns false when the device has no such pass.
    /// </summary>
    public virtual bool AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale) => false;

    /// <summary>
    /// <see cref="AttentionTiled"/> (no log-sum-exp) for rows of different lengths decoded together: row i of head h
    /// sees cached positions c with starts[(h / headsPerRow)·steps + i % steps] ≤ c ≤ position[0] + i % steps (a row with
    /// none, padding, gets zeros). Returns false when the device has no such pass.
    /// </summary>
    public virtual bool AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale) => false;

    /// <summary>Whether <see cref="AttentionSegmented"/> and its gradient run for this head size (with the current <see cref="MixedPrecision"/>).</summary>
    public virtual bool SupportsSegmentedAttention(int dim) => false;

    /// <summary>The gradient of <see cref="AttentionSegmented"/> (as <see cref="AttentionTiledBackward"/>); false when unsupported.</summary>
    public virtual bool AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale) => false;

    public virtual void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale) =>
        AttentionDecode(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale);

    // ---------------------------------------------------------------- incremental decoding (positions live on the device)

    /// <summary>mask[i, j] = j ≤ position + i ? 0 : -1e9 for a [rows, capacity] mask; position is read from device memory.</summary>
    public abstract void DecoderMask(Storage position, Storage mask, int rows, int capacity);

    /// <summary>cache[bh, position + t, :] = source[bh, t, :] for [heads, steps, dim] → [heads, capacity, dim].</summary>
    public abstract void KeyValueWrite(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim);

    /// <summary>
    /// Draws one token per row from softmax(logits / temperature), restricted in turn to the top-k scores (topK &gt; 0),
    /// to the smallest set holding topP of the probability (0 &lt; topP &lt; 1; the cut-off is found by bisection on the
    /// score) and to tokens at least minP times as likely as the best one (minP &gt; 0), using the counter-based random
    /// stream (seed, step, row). Writes the token to ids[row] and 13 statistics to stats[(step * rows + row) * 13]:
    /// id, probability, entropy (bits), then the top-5 (id, probability) pairs. The step number is read from device
    /// memory. Row r's logits start at element r * rowStride + rowOffset.
    /// </summary>
    public abstract void SampleRows(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed);

    /// <summary>
    /// Copies each row's logits (row r at r * rowStride + rowOffset) to work[r, :] and applies repetition penalties for
    /// the last min(length, lastN) tokens of the row's history ring history[r, pos % capacity] (length read from device
    /// memory). Each distinct token in the window is penalized once: repeat (x &gt; 0 ? x / repeat : x * repeat), then
    /// x -= presence + frequency * (occurrences in the window).
    /// </summary>
    public abstract void PenalizeRows(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency);

    /// <summary>history[r, length % capacity] = ids[r] for every row (length read from device memory, not advanced).</summary>
    public abstract void HistoryPush(Storage ids, Storage history, Storage length, int rows, int capacity);

    // ---------------------------------------------------------------- graphs

    /// <summary>Whether this device can record and replay work as a graph.</summary>
    public virtual bool SupportsGraphs => false;

    /// <summary>Starts recording all subsequent work on this device instead of running it.</summary>
    public virtual void BeginCapture() => throw new NotSupportedException();

    /// <summary>Stops recording and returns a replayable graph plus the device blocks it owns.</summary>
    public virtual (IntPtr Executable, IntPtr Graph, List<Storage> Owned) EndCapture() => throw new NotSupportedException();

    /// <summary>Stops recording after a failure, discarding the partial graph.</summary>
    public virtual List<Storage> AbortCapture() => [];

    public virtual void ReplayGraph(IntPtr executable) => throw new NotSupportedException();

    public virtual void DestroyGraph(IntPtr executable, IntPtr graph)
    {
    }
}

/// <summary>Stateless counter-based random numbers shared by every backend (the sampler's random stream).</summary>
internal static class CounterRandom
{
    /// <summary>A uniform float in [0, 1) from (seed, step, row) via the MurmurHash3 finalizer.</summary>
    public static float Uniform(uint seed, uint step, uint row)
    {
        uint h = seed ^ (step * 0x9E3779B9u) ^ (row * 0x85EBCA6Bu);
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return (h >> 8) * (1f / 16777216f);
    }
}

/// <summary>
/// Stateless per-element random mask for dropout, identical on every backend: a MurmurHash3
/// finalizer of (index * golden ratio) ^ seed gives 24 random bits. Because the mask is a pure
/// function of (seed, index) it never has to be stored for the backward pass.
/// </summary>
internal static class DropoutMask
{
    public static bool Keep(uint seed, uint index, float p)
    {
        uint h = (index * 0x9E3779B9u) ^ seed;
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return (h >> 8) * (1f / 16777216f) >= p;
    }
}

/// <summary>Thread-safe byte accounting shared by the backends' caching allocators.</summary>
internal sealed class MemoryAccountant(Func<long?> limit, string deviceName)
{
    private long _inUse;
    private long _cached;
    private long _offloaded;

    public MemoryUsage Usage => new(Interlocked.Read(ref _inUse), Interlocked.Read(ref _cached), limit(), Interlocked.Read(ref _offloaded));

    /// <summary>Counts bytes placed in system memory for this device (negative when released).</summary>
    public void Offloaded(long bytes) => Interlocked.Add(ref _offloaded, bytes);

    /// <summary>
    /// Called before allocating <paramref name="bytes"/> of new memory. Returns true when the cache
    /// should be released first to stay under the limit; throws when even that is not enough.
    /// </summary>
    public bool MustReleaseCacheFor(long bytes)
    {
        if (limit() is not { } max)
        {
            return false;
        }

        long inUse = Interlocked.Read(ref _inUse);
        if (inUse + bytes > max)
        {
            throw new ResourceLimitExceededException(
                $"Allocating {bytes:N0} bytes on {deviceName} would exceed its memory limit of {max:N0} bytes ({inUse:N0} in use). " +
                "Raise the limit in ComputeResources, use smaller batches, or dispose tensors you no longer need.");
        }

        return inUse + Interlocked.Read(ref _cached) + bytes > max;
    }

    public void Allocated(long bytes) => Interlocked.Add(ref _inUse, bytes);

    public void Reused(long bytes)
    {
        Interlocked.Add(ref _cached, -bytes);
        Interlocked.Add(ref _inUse, bytes);
    }

    public void Returned(long bytes)
    {
        Interlocked.Add(ref _inUse, -bytes);
        Interlocked.Add(ref _cached, bytes);
    }

    public void Freed(long bytes) => Interlocked.Add(ref _cached, -bytes);
}

/// <summary>What <see cref="Backend.GemmStrided"/> does with each product before storing it.</summary>
public enum GemmEpilogue
{
    /// <summary>Store it (plus the bias).</summary>
    None = 0,

    /// <summary>Store gelu(product + bias) (tanh approximation), keeping product + bias in the auxiliary tensor when given.</summary>
    Gelu = 1,

    /// <summary>Store product · gelu'(auxiliary): the gradient through a GELU whose inputs the auxiliary tensor holds.</summary>
    GeluGradient = 2,
}
