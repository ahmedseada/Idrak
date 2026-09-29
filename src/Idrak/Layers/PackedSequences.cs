namespace Idrak.Layers;

/// <summary>
/// Several sequences side by side in each row of a [rows, length] token batch (sequence packing: no padding between
/// them, and the batch keeps one shape whatever the sequences' lengths). While <see cref="Use"/> is in effect, decoder
/// layers given a batch of that shape number positions from each sequence's start and let each position attend only to
/// its own sequence, so every sequence gets the results it would get alone. The unused end of a row is one more sequence
/// (padding: give it zero loss weight).
/// </summary>
/// <example>
/// <code>
/// using var packing = PackedSequences.Create([[5, 3], [8]], length: 8, device);   // row 0: 5 + 3 tokens, row 1: 8
/// using (packing.Use())
/// {
///     var hidden = network.Forward(tokens);                                     // tokens [2, 8]
/// }
/// </code>
/// </example>
public sealed class PackedSequences : IDisposable
{
    [ThreadStatic]
    private static PackedSequences? t_current;

    private PackedSequences(int rows, int length, Tensor positions, Tensor starts, Tensor ends)
    {
        Rows = rows;
        Length = length;
        Positions = positions;
        Starts = starts;
        Ends = ends;
    }

    /// <summary>
    /// The layout of <paramref name="lengths"/>.Count rows of <paramref name="length"/> positions, row r holding sequences
    /// of lengths[r] in order (their sum at most <paramref name="length"/>), on <paramref name="device"/>.
    /// </summary>
    public static PackedSequences Create(IReadOnlyList<IReadOnlyList<int>> lengths, int length, Device device)
    {
        ArgumentNullException.ThrowIfNull(lengths);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        var (positions, starts, ends) = Layout(lengths, length);
        int rows = lengths.Count;
        return new PackedSequences(rows, length,
            Tensor.Persistent(positions, [rows * length], device, requiresGrad: false),
            Tensor.Persistent(starts, [rows * length], device, requiresGrad: false),
            Tensor.Persistent(ends, [rows * length], device, requiresGrad: false));
    }

    /// <summary>
    /// Lays out other sequences in the same rows, in place: the tensors keep their memory, so work recorded with this
    /// packing (a CUDA graph) sees the new layout when replayed.
    /// </summary>
    public void Update(IReadOnlyList<IReadOnlyList<int>> lengths)
    {
        ArgumentNullException.ThrowIfNull(lengths);
        if (lengths.Count != Rows)
        {
            throw new ArgumentException($"The packing has {Rows} rows; {lengths.Count} given.");
        }

        var (positions, starts, ends) = Layout(lengths, Length);
        Positions.Load(positions);
        Starts.Load(starts);
        Ends.Load(ends);
    }

    private static (float[] Positions, float[] Starts, float[] Ends) Layout(IReadOnlyList<IReadOnlyList<int>> lengths, int length)
    {
        int rows = lengths.Count;
        var positions = new float[rows * length];
        var starts = new float[rows * length];
        var ends = new float[rows * length];
        for (int r = 0; r < rows; r++)
        {
            int start = 0;
            foreach (int count in lengths[r].Append(length - lengths[r].Sum()))
            {
                if (count < 0 || start + count > length)
                {
                    throw new ArgumentException($"Row {r}: sequences of {string.Join(" + ", lengths[r])} tokens do not fit in {length} positions.");
                }

                for (int t = start; t < start + count; t++)
                {
                    positions[r * length + t] = t - start;
                    starts[r * length + t] = start;
                    ends[r * length + t] = start + count;
                }

                start += count;
            }
        }

        return (positions, starts, ends);
    }

    /// <summary>The packing in effect on this thread, or null.</summary>
    public static PackedSequences? Current => t_current;

    /// <summary>Packed rows per batch.</summary>
    public int Rows { get; }

    /// <summary>Positions per row.</summary>
    public int Length { get; }

    /// <summary>Each position's index within its sequence, [rows · length] (floats, as position tensors are).</summary>
    internal Tensor Positions { get; }

    /// <summary>Where each position's sequence starts in its row, [rows · length].</summary>
    internal Tensor Starts { get; }

    /// <summary>Where each position's sequence stops in its row (exclusive), [rows · length].</summary>
    internal Tensor Ends { get; }

    /// <summary>Whether a [n, t] batch is laid out by this packing.</summary>
    internal bool Matches(int n, int t) => n == Rows && t == Length;

    /// <summary>Applies this packing on this thread until the returned scope is disposed (forward and backward passes).</summary>
    public Scope Use()
    {
        var previous = t_current;
        t_current = this;
        return new Scope(previous);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Positions.Dispose();
        Starts.Dispose();
        Ends.Dispose();
    }

    /// <summary>
    /// Whether <paramref name="network"/> can run packed batches on its device: its attention layers are
    /// <see cref="CausalSelfAttention"/> with a head size the device's packed attention supports (on CUDA, bfloat16
    /// tensor cores with head size 64 or 128), and no layer with positions of its own that ignores the packing.
    /// </summary>
    public static bool Supports(Module network)
    {
        ArgumentNullException.ThrowIfNull(network);
        var modules = network.Descendants().ToList();
        var attention = modules.OfType<CausalSelfAttention>().ToList();
        return attention.Count > 0 && !modules.Any(m => m is MultiHeadAttention or PositionalEncoding)
            && attention.All(a => a.Query.Device.Backend.SupportsSegmentedAttention(a.HeadDim));
    }

    /// <summary>Restores the previous packing when disposed.</summary>
    public readonly struct Scope : IDisposable
    {
        private readonly PackedSequences? _previous;

        internal Scope(PackedSequences? previous) => _previous = previous;

        /// <inheritdoc />
        public void Dispose() => t_current = _previous;
    }
}
